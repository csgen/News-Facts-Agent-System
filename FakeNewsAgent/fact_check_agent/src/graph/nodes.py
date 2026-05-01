"""All LangGraph node functions for the Fact-Check Agent graph.

Each node is a pure function: (state) -> dict of partial state updates.
Nodes that need MemoryAgent receive it as a second argument via closure
(see graph.py build_graph()).
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

import fact_check_agent.src.llm_factory as _llm_factory
from fact_check_agent.src.agents import context_claim_agent
from fact_check_agent.src.agents.reflection_agent import (
    query_source_credibility,
    record_verdict_outcome,
)
from fact_check_agent.src.failure_logger import log_failure
from fact_check_agent.src.models.schemas import (
    FactCheckOutput,
    MemoryQueryResponse,
    SimilarClaim,
)
from fact_check_agent.src.models.state import FactCheckState
from fact_check_agent.src.prompts import (
    IMAGE_CLAIM_ASSESSOR_PROMPT,
    JUDGE_PROMPT,
    SKEPTIC_PROMPT,
    SUPPORTER_PROMPT,
    VERDICT_SYNTHESIS_PROMPT,
)
from fact_check_agent.src.tools.cross_modal_tool import check_cross_modal
from fact_check_agent.src.tools.freshness_tool import check_freshness
from fact_check_agent.src.tools.rag_tool import retrieve_similar_claims
from fact_check_agent.src.tools.reranker import rerank_candidates

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent  # memory_agent — type hint only

logger = logging.getLogger(__name__)


# ── Node: receive_claim ───────────────────────────────────────────────────────


def receive_claim(state: FactCheckState) -> dict:
    """Initialise all mutable state fields to defaults."""
    inp = state["input"]
    return {
        "memory_results": None,
        "entity_context": [],
        "fresh_context": [],
        "stale_context": [],
        "context_claims": [],
        "retrieved_chunks": list(inp.prefetched_chunks),
        "debate_transcript": None,
        "vlm_assessment_block": None,
        "source_credibility": None,
        "cross_modal_flag": False,
        "clip_similarity_score": None,
        "output": None,
    }


_TOPIC_LIST = (
    "technology",
    "geopolitics",
    "financial",
    "health",
    "science",
    "sports",
    "entertainment",
    "climate",
    "crime",
    "art",
)


def _classify_topic(claim_text: str, settings) -> str:
    """Classify a claim into one of the standard topic categories via LLM.

    Used only for the frontend path where ClaimIsolator hasn't run.
    Falls back to 'unknown' if the LLM returns an unexpected value or fails.
    """
    try:
        client = _llm_factory.make_llm_client()
        response = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Classify this claim into exactly one topic. "
                        f"Options: {' | '.join(_TOPIC_LIST)}\n\n"
                        f"Claim: {claim_text}\n\n"
                        "Respond with just the single topic word."
                    ),
                }
            ],
            temperature=0,
            max_tokens=10,
        )
        topic = response.choices[0].message.content.strip().lower()
        return topic if topic in _TOPIC_LIST else "unknown"
    except Exception as exc:
        logger.warning("_classify_topic failed: %s", exc)
        return "unknown"


# ── Node: query_memory ────────────────────────────────────────────────────────


def query_memory(state: FactCheckState, memory: "MemoryAgent", settings=None) -> dict:
    """Query MemoryAgent: similar claims (vector) + optional GraphRAG + reranking."""
    from fact_check_agent.src.config import settings as _settings

    if settings is None:
        settings = _settings

    if settings.offline_mode:
        logger.info("query_memory: offline_mode=True — skipping all DB reads")
        return {
            "memory_results": MemoryQueryResponse(results=[], max_confidence=0.0),
            "entity_context": [],
            "source_credibility": {},
        }

    inp = state["input"]

    # ── Stage 1: vector similarity search ────────────────────────────────
    try:
        vector_results = retrieve_similar_claims(inp.claim_text, memory)
    except Exception as exc:
        logger.warning("query_memory: vector search failed — proceeding without memory context: %s", exc)
        log_failure(
            memory=memory,
            claim_id=inp.claim_id,
            node_name="query_memory",
            failure_type="db_error",
            exception=exc,
        )
        vector_results = []

    try:
        entity_ctx = memory.get_entity_context(inp.claim_id)
    except Exception as exc:
        logger.warning("query_memory: entity context lookup failed: %s", exc)
        log_failure(
            memory=memory,
            claim_id=inp.claim_id,
            node_name="query_memory",
            failure_type="db_error",
            exception=exc,
        )
        entity_ctx = []

    # ── Stage 2: GraphRAG — expand via entity-claim traversal ────────────
    graph_results: list[dict] = []
    if settings.use_graph_rag and vector_results:
        claim_ids = [c["claim_id"] for c in vector_results]
        entity_ids = [e["entity_id"] for e in memory.get_entity_ids_for_claims(claim_ids)]
        if entity_ids:
            graph_results = memory.get_graph_claims_for_entities(entity_ids)
            logger.info(
                "GraphRAG: %d entities → %d graph claims", len(entity_ids), len(graph_results)
            )

    # ── Stage 3: RRF merge ────────────────────────────────────────────────
    reranked = rerank_candidates(
        query=inp.claim_text,
        vector_results=vector_results,
        graph_results=graph_results,
        top_k=settings.reranker_top_k,
    )

    # Batch-fetch verdict verified_at from Neo4j — authoritative source for timestamps.
    # The ChromaDB per-claim lookup in rag_tool may miss recently-superseded verdicts
    # or return stale metadata; a single Neo4j query is both faster and more accurate.
    neo4j_timestamps: dict = {}
    try:
        claim_ids_for_ts = [c["claim_id"] for c in reranked if c.get("claim_id")]
        if claim_ids_for_ts:
            neo4j_timestamps = memory.get_verdict_timestamps_for_claims(claim_ids_for_ts)
    except Exception as _ts_err:
        logger.warning("query_memory: Neo4j timestamp batch lookup failed: %s", _ts_err)

    results = [
        SimilarClaim(
            claim_id=c["claim_id"],
            claim_text=c["claim_text"],
            verdict_label=c.get("verdict_label"),
            verdict_confidence=c.get("verdict_confidence"),
            distance=c.get("distance", 0.0),
            # Prefer Neo4j verified_at (authoritative); fall back to ChromaDB value
            verified_at=neo4j_timestamps.get(c["claim_id"]) or c.get("verified_at"),
        )
        for c in reranked
    ]

    max_confidence = max(
        (r.verdict_confidence for r in results if r.verdict_confidence is not None),
        default=0.0,
    )

    # Resolve effective topic — use ClaimIsolator output if available, else classify
    effective_topic = inp.topic_text.strip()
    if not effective_topic:
        effective_topic = _classify_topic(inp.claim_text, settings)

    source_cred = query_source_credibility(
        claim_text=inp.claim_text,
        source_url=inp.source_url,
        memory=memory,
        topic=effective_topic,
    )

    logger.info(
        "query_memory: %d vector + %d graph → %d reranked, max_confidence=%.2f, "
        "graph_rag=%s, topic=%s, source_cred_samples=%d",
        len(vector_results),
        len(graph_results),
        len(results),
        max_confidence,
        settings.use_graph_rag,
        effective_topic,
        source_cred.get("sample_count", 0),
    )

    return {
        "memory_results": MemoryQueryResponse(results=results, max_confidence=max_confidence),
        "entity_context": entity_ctx,
        "source_credibility": source_cred,
        "effective_topic": effective_topic,
    }


# ── Node: return_cached_verdict ───────────────────────────────────────────────

_CACHE_EXACT_DISTANCE = 0.05   # cosine distance below which a hit is "the same claim"
_CACHE_MIN_CONFIDENCE = 0.70   # minimum stored confidence to trust the cached verdict


def return_cached_verdict(state: FactCheckState, memory: "MemoryAgent") -> dict:
    """Short-circuit node: reconstruct FactCheckOutput from the best fresh cache hit.

    Only reached when cache_hit_check() returns "hit". Fetches the full verdict
    reasoning from memory so the UI can display it without re-running synthesis.
    """
    inp = state["input"]
    fresh = state.get("fresh_context") or []

    # Pick the nearest qualifying hit (lowest distance above confidence threshold)
    qualifying = [
        c for c in fresh
        if c.get("distance", 1.0) < _CACHE_EXACT_DISTANCE
        and (c.get("verdict_confidence") or 0.0) >= _CACHE_MIN_CONFIDENCE
        and c.get("verdict_label")
        and c.get("verdict_label") != "misleading"
    ]
    if not qualifying:
        logger.warning("return_cached_verdict: no qualifying hit in fresh_context — state mismatch")
        return {}

    best = min(qualifying, key=lambda c: c.get("distance", 1.0))

    # Fetch full verdict text from memory (evidence_summary stored as ChromaDB document)
    reasoning = f"[Cached verdict for similar claim: \"{best['claim_text'][:120]}\"]"
    evidence_links: list[str] = []
    verdict_id = f"cached_{best['claim_id']}"
    cross_modal_flag = False

    try:
        verdict_raw = memory.get_verdict_by_claim(best["claim_id"])
        if verdict_raw.get("ids"):
            verdict_id = verdict_raw["ids"][0]
        if verdict_raw.get("documents") and verdict_raw["documents"]:
            reasoning = verdict_raw["documents"][0] or reasoning
        if verdict_raw.get("metadatas") and verdict_raw["metadatas"]:
            meta = verdict_raw["metadatas"][0]
            cross_modal_flag = bool(meta.get("image_mismatch", False))
    except Exception as exc:
        logger.warning("return_cached_verdict: verdict fetch failed (%s) — using stub reasoning", exc)

    raw_conf = best.get("verdict_confidence") or 0.0
    confidence_score = int(round(raw_conf * 100)) if raw_conf <= 1.0 else int(round(raw_conf))
    confidence_score = max(0, min(100, confidence_score))

    output = FactCheckOutput(
        verdict_id=verdict_id,
        claim_id=inp.claim_id,
        verdict=best["verdict_label"],
        confidence_score=confidence_score,
        evidence_links=evidence_links,
        reasoning=reasoning,
        cross_modal_flag=cross_modal_flag,
        last_verified_at=best.get("verified_at"),
    )

    logger.info(
        "return_cached_verdict: cache hit → verdict=%s conf=%d dist=%.4f cached_claim=%s",
        output.verdict, output.confidence_score, best["distance"], best["claim_id"],
    )
    return {"output": output}


# ── Node: freshness_check_all ─────────────────────────────────────────────────


def freshness_check_all(state: FactCheckState, settings) -> dict:
    """Tag every retrieved SimilarClaim as fresh or stale using check_freshness().

    Claims without a verified_at timestamp default to stale (safe assumption).
    Claims without a verdict_label are also marked stale — no verdict to reuse.
    """
    results = []
    if state.get("memory_results") and state["memory_results"].results:
        results = state["memory_results"].results

    if settings.offline_mode or not results:
        return {"fresh_context": [], "stale_context": []}

    fresh: list[dict] = []
    stale: list[dict] = []

    for claim in results:
        chunk = claim.model_dump()

        if not claim.verified_at or not claim.verdict_label or claim.verdict_label == "misleading":
            stale.append(chunk)
            continue

        freshness = check_freshness(
            claim_text=claim.claim_text,
            verdict_label=claim.verdict_label,
            verdict_confidence=claim.verdict_confidence or 0.5,
            last_verified_at=claim.verified_at,
            api_key=settings.openai_api_key,
            model=_llm_factory.llm_model_name(),
        )
        chunk["freshness_reason"] = freshness["reason"]
        chunk["freshness_category"] = freshness["claim_category"]

        if freshness["revalidate"]:
            stale.append(chunk)
        else:
            fresh.append(chunk)

    logger.info("freshness_check_all: %d fresh, %d stale", len(fresh), len(stale))
    return {"fresh_context": fresh, "stale_context": stale}


# ── Node: context_claim_agent ─────────────────────────────────────────────────

_CREDIBILITY_FILTER_THRESHOLD = 0.5  # Tavily results below this are dropped


def context_claim_agent_node(state: FactCheckState, memory: "MemoryAgent", settings) -> dict:
    """Generate questions, gather evidence, enrich with per-source credibility, filter low-cred."""
    from fact_check_agent.src.agents.reflection_agent import source_id_from_url

    claims = context_claim_agent.run(
        claim_text=state["input"].claim_text,
        fresh_context=state.get("fresh_context", []),
        prefetched_chunks=state.get("retrieved_chunks", []),
        tavily_api_key=settings.tavily_api_key,
        input_url=state["input"].source_url or "",
    )

    effective_topic = state.get("effective_topic", "") or state["input"].topic_text
    filtered: list[dict] = []

    for claim in claims:
        if claim["source"] == "memory":
            filtered.append(claim)
            continue

        source_url = claim.get("source_url") or ""
        cred: float | None = None

        if source_url:
            source_id = source_id_from_url(source_url)
            try:
                cred = memory.get_source_topic_credibility(source_id, effective_topic)
                if cred is None:
                    cred = memory.get_base_credibility(source_id)
            except Exception:
                pass

        if cred is None:
            cred = _SOURCE_CREDIBILITY.get(claim["source"], _DEFAULT_CREDIBILITY)

        if cred < _CREDIBILITY_FILTER_THRESHOLD:
            logger.info(
                "context_claim_agent: dropped %s result (source_url=%s, credibility=%.3f < %.2f)",
                claim["source"],
                source_url,
                cred,
                _CREDIBILITY_FILTER_THRESHOLD,
            )
            continue

        claim["credibility"] = cred
        filtered.append(claim)

    logger.info(
        "context_claim_agent: %d/%d claims kept after credibility filter (threshold=%.2f)",
        len(filtered),
        len(claims),
        _CREDIBILITY_FILTER_THRESHOLD,
    )
    return {"context_claims": filtered}


# ── Node: vlm_assessment ─────────────────────────────────────────────────────


def vlm_assessment_node(state: FactCheckState, memory=None, settings=None) -> dict:
    """Run VLM visual assessment — generates vlm_assessment_block for the Judge.

    Calls IMAGE_CLAIM_ASSESSOR_PROMPT (v2.1) via the configured vision-capable
    model (GPT-4o for OpenAI; ollama_llm_model for Ollama with vision support).
    Returns "No image available." when no image_url is present or on any failure.
    """
    inp = state["input"]
    image_url = getattr(inp, "image_url", None)
    if not image_url:
        return {"vlm_assessment_block": "No image available."}

    from fact_check_agent.src.tools.cross_modal_tool import _ensure_base64_uri

    prompt = IMAGE_CLAIM_ASSESSOR_PROMPT.format(claim_text=inp.claim_text)
    client = _llm_factory.make_vlm_client()
    model = _llm_factory.vlm_model_name()

    try:
        if settings.llm_provider == "ollama":
            data_uri = _ensure_base64_uri(image_url)
            if data_uri is None:
                logger.warning("vlm_assessment: could not fetch image %s — skipping", image_url)
                return {"vlm_assessment_block": "No image available."}
            content = [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)

        assessment = float(result.get("assessment", 0.0))
        block = (
            f"Caption: {result.get('caption', 'N/A')}\n"
            f"Visual Evidence: {result.get('visual_evidence', 'N/A')}\n"
            f"Assessment: {assessment:+.2f} "
            f"(+0.25=strong visual support, +0.10=partial, 0.0=irrelevant, "
            f"-0.10=partial refutation, -0.25=strong refutation)\n"
            f"Explanation: {result.get('explanation', 'N/A')}"
        )
        logger.info("vlm_assessment: image=%s assessment=%.2f", image_url[:80], assessment)
        return {"vlm_assessment_block": block}

    except Exception as exc:
        logger.error("vlm_assessment_node failed: %s — skipping VLM signal", exc)
        log_failure(
            memory=memory,
            claim_id=inp.claim_id,
            node_name="vlm_assessment",
            failure_type="llm_api_error",
            exception=exc,
        )
        return {"vlm_assessment_block": "No image available."}


# ── Node: synthesize_verdict ──────────────────────────────────────────────────

# Credibility assigned to claims by source when no stored confidence is available
_SOURCE_CREDIBILITY: dict[str, float] = {
    "tavily": 0.75,
    "prefetched": 0.70,
}
_DEFAULT_CREDIBILITY = 0.65


def _format_numbered_context_claims(context_claims: list[dict]) -> str:
    """Render context_claims as a numbered list for the LLM — no credibility scores exposed."""
    lines: list[str] = []
    for i, c in enumerate(context_claims, 1):
        if c["type"] == "memory":
            prior = f" (prior verdict: {c['verdict']})" if c.get("verdict") else ""
            lines.append(f"[{i}] MEMORY — prior verified claim{prior}")
            lines.append(f'    "{c["content"]}"')
        else:
            tag = "FACTUAL" if c["type"] == "factual" else "COUNTER-FACTUAL"
            lines.append(f"[{i}] {tag} ({c['source']})")
            if c.get("question"):
                lines.append(f"    Q: {c['question']}")
            lines.append(f"    Evidence: {c['content']}")
        lines.append("")
    return "\n".join(lines).strip() or "No evidence available."


def _get_claim_credibility(claim: dict) -> float:
    if claim["source"] == "memory" and claim.get("confidence") is not None:
        return float(claim["confidence"])
    # Tavily/prefetched claims carry a pre-computed credibility from context_claim_agent_node
    if claim.get("credibility") is not None:
        return float(claim["credibility"])
    return _SOURCE_CREDIBILITY.get(claim["source"], _DEFAULT_CREDIBILITY)


_VALID_DEGREES = {1.0, 0.5, 0.0, -0.5, -1.0}


def _compute_verdict(
    context_claims: list[dict],
    degrees: list[float],
) -> tuple[str, int, float]:
    """Compute verdict, confidence, and evidence volume via the formula:

        V = Σ(Di × Ci) / Σ|Ci|

    Di: signed degree of support (-1.0 to 1.0) returned by the LLM.
    Ci: credibility of the source (memory confidence | tavily 0.75 | prefetched 0.70).

    Verdict thresholds:  V > 0.5 → supported | V < -0.5 → refuted | else → misleading.

    Confidence blends |V| (verdict strength) with a volume factor (Σ|Ci| / 2.5):
    a single credible source is capped around 60 %; three or more can reach 97 %.

    Returns: (verdict_label, confidence_0_100, evidence_volume=Σ|Ci|)
    """
    total_weighted = 0.0
    total_credibility = 0.0

    for i, claim in enumerate(context_claims):
        raw_d = degrees[i] if i < len(degrees) else 0.0
        d = min(1.0, max(-1.0, float(raw_d)))  # clamp; snap to nearest valid value
        c = _get_claim_credibility(claim)
        total_weighted += d * c
        total_credibility += abs(c)

    evidence_volume = total_credibility  # Σ|Ci| — how much evidence we have

    if total_credibility == 0:
        return "misleading", 50, 0.0

    V = total_weighted / total_credibility  # [-1.0, 1.0]

    if V > 0.5:
        verdict = "supported"
    elif V < -0.5:
        verdict = "refuted"
    else:
        verdict = "misleading"

    # Confidence = verdict strength × volume factor
    # volume_factor → 1.0 as Σ|Ci| → 2.5 (≈ 3 tavily-credibility sources)
    volume_factor = min(1.0, evidence_volume / 2.5)
    confidence = int(min(97, max(15, abs(V) * 100 * (0.4 + 0.6 * volume_factor))))

    return verdict, confidence, evidence_volume


def synthesize_verdict(state: FactCheckState, memory=None, settings=None) -> dict:
    """Synthesise a credibility-weighted verdict using V = Σ(Di×Ci) / Σ|Ci|.

    The LLM assigns a signed Degree of Support Di ∈ {-1,-0.5,0,0.5,1} per claim
    without seeing credibility scores. Python then computes V and derives the
    verdict label + confidence from the formula.
    """
    from fact_check_agent.src.id_utils import make_id

    inp = state["input"]
    context_claims = state.get("context_claims") or []
    numbered_block = _format_numbered_context_claims(context_claims)

    prompt = VERDICT_SYNTHESIS_PROMPT.format(
        claim_text=inp.claim_text,
        numbered_claims=numbered_block,
    )

    client = _llm_factory.make_llm_client()
    raw = ""
    try:
        response = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            # Ollama/Gemma3 sometimes returns empty content with json_object format
            logger.warning(
                "synthesize_verdict: empty response from LLM — retrying without json_object"
            )
            response = client.chat.completions.create(
                model=_llm_factory.llm_model_name(),
                messages=[{"role": "user", "content": prompt + "\n\nRespond in JSON."}],
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
        result = json.loads(raw.strip())
    except Exception as e:
        failure_type = "json_parse_error" if isinstance(e, (json.JSONDecodeError, ValueError)) else "llm_api_error"
        logger.error("Verdict synthesis failed: %s", e)
        log_failure(
            memory=memory,
            claim_id=inp.claim_id,
            node_name="synthesize_verdict",
            failure_type=failure_type,
            exception=e,
            raw_llm_response=raw,
        )
        result = {"degrees": [], "reasoning": str(e)}

    degrees = [float(x) for x in result.get("degrees", [])]
    reasoning = result.get("reasoning", "")
    verdict, confidence, evidence_volume = _compute_verdict(context_claims, degrees)

    logger.info(
        "synthesize_verdict: V-formula → %s (confidence=%d, evidence_volume=%.2f, degrees=%s)",
        verdict,
        confidence,
        evidence_volume,
        [round(d, 1) for d in degrees[:8]],
    )

    output = FactCheckOutput(
        verdict_id=make_id("vrd_"),
        claim_id=inp.claim_id,
        verdict=verdict,
        confidence_score=confidence,
        evidence_links=[],
        reasoning=reasoning,
        cross_modal_flag=False,
        vlm_assessment_block=None,
    )
    return {
        "output": output,
        "neutral_degrees": degrees,
        "neutral_reasoning": reasoning,
    }


# ── Node: multi_agent_debate (S4) ─────────────────────────────────────────────


def _format_neutral_scores_block(context_claims: list[dict], degrees: list[float]) -> str:
    """Render each evidence item with its Neutral Di for the Supporter/Skeptic/Judge."""
    lines: list[str] = []
    for i, claim in enumerate(context_claims, 1):
        d = degrees[i - 1] if i - 1 < len(degrees) else 0.0
        tag = {"memory": "MEMORY", "factual": "FACTUAL", "counter_factual": "COUNTER-FACTUAL"}.get(
            claim["type"], claim["type"].upper()
        )
        lines.append(f"[{i}] {tag} | Neutral Di = {d:+.1f}")
        if claim.get("question"):
            lines.append(f"    Q: {claim['question']}")
        lines.append(f'    "{claim["content"][:200]}"')
        lines.append("")
    return "\n".join(lines).strip()


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    return json.loads(raw)


def multi_agent_debate(state: FactCheckState, memory=None, settings=None) -> dict:
    """S4: 4-role structured debate for low-confidence verdicts and/or VLM image signal.

    Two entry modes:
      Full debate (use_debate=True):
        Role 2 — Supporter: proposes Di boosts where neutral was too conservative.
        Role 3 — Skeptic:   proposes Di penalties where neutral missed flaws.
        Role 4 — Judge:     combines all three + VLM assessment → final Di.
      VLM-only (use_debate=False, image present):
        Skips Supporter/Skeptic; Judge integrates only VLM signal into neutral Di.
    """
    inp = state["input"]
    output = state.get("output")
    context_claims = state.get("context_claims") or []
    neutral_degrees = state.get("neutral_degrees") or []
    neutral_reasoning = state.get("neutral_reasoning") or ""
    numbered_block = _format_numbered_context_claims(context_claims)
    neutral_block = _format_neutral_scores_block(context_claims, neutral_degrees)
    vlm_block = state.get("vlm_assessment_block") or "No image available."

    client = _llm_factory.make_llm_client()
    model = _llm_factory.llm_model_name()

    def _call(prompt_text: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return raw

    _last_raw = ""
    try:
        if settings.use_debate:
            # ── Full S4: Supporter + Skeptic + Judge ──────────────────────────
            supporter_raw = _call(
                SUPPORTER_PROMPT.format(
                    claim_text=inp.claim_text,
                    numbered_claims=numbered_block,
                    neutral_scores_block=neutral_block,
                )
            )
            _last_raw = supporter_raw
            supporter_result = json.loads(supporter_raw)
            supporter_adj = supporter_result.get("adjustments", [])

            skeptic_raw = _call(
                SKEPTIC_PROMPT.format(
                    claim_text=inp.claim_text,
                    numbered_claims=numbered_block,
                    neutral_scores_block=neutral_block,
                )
            )
            _last_raw = skeptic_raw
            skeptic_result = json.loads(skeptic_raw)
            skeptic_adj = skeptic_result.get("adjustments", [])

            logger.info(
                "multi_agent_debate: supporter proposed %d adjustments, skeptic proposed %d",
                len(supporter_adj),
                len(skeptic_adj),
            )
        else:
            # ── VLM-only: skip Supporter/Skeptic, Judge uses image signal only ─
            supporter_raw = "None — no debate was run"
            skeptic_raw = "None — no debate was run"
            supporter_adj = []
            skeptic_adj = []
            logger.info("multi_agent_debate: VLM-only judge path (use_debate=False)")

        # ── Fast-path: no debate + no image → skip LLM judge, use neutral verbatim ─
        if not settings.use_debate and vlm_block == "No image available.":
            final_degrees = list(neutral_degrees)
            verdict, confidence, _ = _compute_verdict(context_claims, final_degrees)
            reasoning = (
                neutral_reasoning
                or "No debate and no image assessment — using neutral agent's verdict."
            )
            updated_output = (
                output.model_copy(
                    update={
                        "verdict": verdict,
                        "confidence_score": confidence,
                        "reasoning": reasoning,
                    }
                )
                if output
                else output
            )
            logger.info(
                "multi_agent_debate: skipped judge (no debate + no image) — verdict=%s confidence=%d",
                verdict,
                confidence,
            )
            return {
                "output": updated_output,
                "debate_transcript": "No debate and no image — using neutral verdict unchanged.",
            }

        # ── Role 4: Judge (always runs when debate mode or image is present) ───
        judge_raw = _call(
            JUDGE_PROMPT.format(
                claim_text=inp.claim_text,
                numbered_claims=numbered_block,
                neutral_scores_block=neutral_block,
                supporter_adjustments=supporter_raw,
                skeptic_adjustments=skeptic_raw,
                vlm_assessment_block=vlm_block,
            )
        )
        _last_raw = judge_raw
        judge_result = json.loads(judge_raw)

        # Extract final Di per evidence item from Judge's output
        final_scores = {
            item["evidence_id"]: float(item["final_D"])
            for item in judge_result.get("final_scores", [])
        }
        stalemates = sum(
            1 for item in judge_result.get("final_scores", []) if item.get("stalemate")
        )

        # Map back to ordered list aligned with context_claims
        final_degrees = [
            final_scores.get(i + 1, neutral_degrees[i] if i < len(neutral_degrees) else 0.0)
            for i in range(len(context_claims))
        ]

        # ── Clamp: when no debate ran, judge may only nudge via VLM ────────────
        if not settings.use_debate:
            for i in range(len(context_claims)):
                if i >= len(final_degrees) or i >= len(neutral_degrees):
                    continue
                delta = abs(final_degrees[i] - neutral_degrees[i])
                if delta > 0.5:
                    logger.warning(
                        "multi_agent_debate: clamping item %d (judge=%.1f → neutral=%.1f, delta=%.1f > 0.5)",
                        i + 1,
                        final_degrees[i],
                        neutral_degrees[i],
                        delta,
                    )
                    final_degrees[i] = neutral_degrees[i]

        verdict, confidence, evid_volume = _compute_verdict(context_claims, final_degrees)

        # Lower confidence if many stalemates (full debate only)
        if stalemates > 0 and settings.use_debate:
            stalemate_penalty = min(15, stalemates * 5)
            confidence = max(15, confidence - stalemate_penalty)

        verdict_explanation = judge_result.get("verdict_explanation") or judge_result.get(
            "debate_summary", ""
        )
        if settings.use_debate:
            reasoning = (
                f"{neutral_reasoning}\n\n"
                f"{verdict_explanation}\n\n"
                f"[Debate: {len(supporter_adj)} boosts, {len(skeptic_adj)} penalties, {stalemates} stalemates]"
            )
            transcript = (
                f"=== NEUTRAL (initial Di) ===\n{neutral_block}\n\n"
                f"=== SUPPORTER ===\n{supporter_raw}\n\n"
                f"=== SKEPTIC ===\n{skeptic_raw}\n\n"
                f"=== JUDGE ===\n{judge_raw}"
            )
        else:
            reasoning = f"{neutral_reasoning}\n\n[Judge: {verdict_explanation}]"
            transcript = f"=== JUDGE ONLY ===\n{judge_raw}"

        updated_output = (
            output.model_copy(
                update={
                    "verdict": verdict,
                    "confidence_score": confidence,
                    "reasoning": reasoning,
                }
            )
            if output
            else output
        )

        logger.info(
            "multi_agent_debate: verdict=%s confidence=%d (stalemates=%d, "
            "supporter_adj=%d, skeptic_adj=%d, vlm=%s)",
            verdict,
            confidence,
            stalemates,
            len(supporter_adj),
            len(skeptic_adj),
            vlm_block != "No image available.",
        )
        return {"output": updated_output, "debate_transcript": transcript}

    except Exception as e:
        failure_type = "json_parse_error" if isinstance(e, (json.JSONDecodeError, ValueError)) else "llm_api_error"
        logger.error("multi_agent_debate failed: %s — keeping original verdict", e)
        log_failure(
            memory=memory,
            claim_id=inp.claim_id,
            node_name="multi_agent_debate",
            failure_type=failure_type,
            exception=e,
            raw_llm_response=_last_raw,
        )
        return {}


# ── Node: cross_modal_check ───────────────────────────────────────────────────


def cross_modal_check(state: FactCheckState, settings) -> dict:
    """Cross-modal consistency check.

    Fast path: when vlm_assessment_node already ran, derive the conflict flag
    directly from its assessment score — no second VLM/LLM call needed.
    The assessment score in the block encodes image-claim alignment:
      negative → image refutes / contradicts claim → conflict = True
      zero/positive → image supports or is irrelevant → conflict = False

    Fallback path: SigLIP → Gemma4 vision → GPT-4o vision → LLM caption text.
    Used when no vlm_assessment_block is available (no image_url on the claim).
    """
    inp = state["input"]
    vlm_block = state.get("vlm_assessment_block") or ""

    siglip_score = None

    if vlm_block and vlm_block != "No image available.":
        # Parse assessment score from the block produced by vlm_assessment_node.
        # Format: "Assessment: -0.10 (+0.25=strong visual support, ...)"
        conflict = False
        try:
            for line in vlm_block.splitlines():
                if line.startswith("Assessment:"):
                    score_str = line.split(":")[1].strip().split()[0]
                    score = float(score_str)
                    conflict = score < 0.0
                    break
        except Exception as exc:
            logger.warning("cross_modal_check: could not parse assessment score: %s", exc)

        logger.info(
            "cross_modal_check: derived from vlm_assessment_block conflict=%s", conflict
        )
    else:
        # No VLM assessment — fall back to the full check_cross_modal tool
        result = check_cross_modal(
            claim_text=inp.claim_text,
            image_caption=inp.image_caption,
            api_key=settings.openai_api_key,
            model=_llm_factory.llm_model_name(),
            image_url=getattr(inp, "image_url", None),
        )
        conflict = result["flag"]
        siglip_score = result.get("siglip_score")
        vlm_block = None

    current_output: Optional[FactCheckOutput] = state.get("output")
    updated_output = None
    if current_output:
        updated_output = current_output.model_copy(
            update={
                "cross_modal_flag": conflict,
                "vlm_assessment_block": vlm_block or None,
                "image_url": getattr(inp, "image_url", None),
            }
        )

    return {
        "cross_modal_flag": conflict,
        "clip_similarity_score": siglip_score,
        "output": updated_output or current_output,
    }


# ── Node: write_memory ────────────────────────────────────────────────────────


def write_memory(state: FactCheckState, memory: "MemoryAgent") -> dict:
    """Delegate all 5 post-verdict write-backs to the Reflection Agent."""
    from fact_check_agent.src.config import settings as _settings

    if _settings.dry_run or _settings.offline_mode:
        logger.info(
            "write_memory: %s — skipping DB write",
            "offline_mode" if _settings.offline_mode else "dry_run",
        )
        return {}

    output: Optional[FactCheckOutput] = state.get("output")
    if not output:
        logger.warning("write_memory called with no output — skipping")
        return {}

    record_verdict_outcome(
        output=output,
        claim_text=state["input"].claim_text,
        source_url=state["input"].source_url,
        topic_text=state.get("effective_topic", "") or state["input"].topic_text,
        memory=memory,
    )
    logger.info("write_memory: verdict %s written for claim %s", output.verdict, output.claim_id)
    return {}


# ── Node: emit_output ─────────────────────────────────────────────────────────


def emit_output(state: FactCheckState) -> dict:
    """Terminal node — stamps freshness metadata onto the output before returning."""
    current_output: Optional[FactCheckOutput] = state.get("output")
    if not current_output:
        logger.error("emit_output reached with no output in state")
        return {}

    # Derive last_verified_at from the most recent fresh memory claim
    fresh = state.get("fresh_context") or []
    last_verified_at = None
    for claim in fresh:
        ts = claim.get("verified_at")
        if ts and (last_verified_at is None or ts > last_verified_at):
            last_verified_at = ts

    updated = current_output.model_copy(update={"last_verified_at": last_verified_at})
    return {"output": updated}
