"""Fact-Check Agent adapter — wraps FakeNewsAgent's LangGraph pipeline.

Public API for Streamlit frontend and other PredictionAgent code:

    from agents.fact_check_agent import run_fact_check, run_fact_check_by_claim_ids

    # Option A: full pipeline from a fresh PreprocessingOutput (cold path)
    outputs = run_fact_check(preprocessing_output)

    # Option B: by already-ingested claim_ids (cache-hit path; fetches from DB)
    outputs = run_fact_check_by_claim_ids(["clm_abc...", "clm_def..."])

This file lives in the integrated monorepo and bridges the three subfolders
by adjusting sys.path so `src.*` (scraper) and `fact_check_agent.src.*`
(FakeNewsAgent) resolve from this process.
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make the sibling subfolders importable:
#   <repo>/scraper_preprocessing_memory       → `src.*` imports
#   <repo>/scraper_preprocessing_memory/src   → flat `id_utils`, `config`, `models.*`
#                                                imports (used by task3 code)
#   <repo>/FakeNewsAgent                      → `fact_check_agent.*` imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCAPPER = _REPO_ROOT / "scraper_preprocessing_memory"
for _p in (str(_SCAPPER), str(_SCAPPER / "src"), str(_REPO_ROOT / "FakeNewsAgent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fact_check_agent.src.llm_factory import get_langfuse_handler
from fact_check_agent.src.models.schemas import EntityRef, FactCheckInput, FactCheckOutput
from src.id_utils import make_id
from src.models.pipeline import PreprocessingOutput

from agents.memory_agent import get_memory

logger = logging.getLogger(__name__)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from fact_check_agent.src.graph.graph import build_graph
        _graph = build_graph(get_memory())
    return _graph


def _claim_to_input(
    output: PreprocessingOutput,
    claim_index: int,
    image_caption: Optional[str],
    image_url: Optional[str] = None,
) -> FactCheckInput:
    claim = output.claims[claim_index]
    return FactCheckInput(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        entities=[
            EntityRef(
                entity_id=e.entity_id,
                name=e.name,
                entity_type=e.entity_type,
                sentiment=e.sentiment,
            )
            for e in claim.entities
        ],
        source_url=output.article.url,
        article_id=claim.article_id,
        image_caption=image_caption,
        image_url=image_url,
        timestamp=claim.extracted_at,
        topic_text=getattr(claim, "topic_text", "") or "",
    )


def run_fact_check(output: PreprocessingOutput) -> list[FactCheckOutput]:
    """Run the fact-check graph on every claim in a PreprocessingOutput."""
    memory = get_memory()
    graph = _get_graph()

    caption_result = memory.get_caption_by_article(output.article.article_id)
    image_caption: Optional[str] = None
    image_url: Optional[str] = None
    if caption_result.get("documents"):
        image_caption = caption_result["documents"][0]
    if caption_result.get("metadatas") and caption_result["metadatas"]:
        image_url = caption_result["metadatas"][0].get("image_url")

    results: list[FactCheckOutput] = []
    for i in range(len(output.claims)):
        fc_input = _claim_to_input(output, i, image_caption, image_url)
        lf = get_langfuse_handler()
        state = graph.invoke({"input": fc_input}, config={"callbacks": [lf]} if lf else {})
        fc_output: Optional[FactCheckOutput] = state.get("output")
        if fc_output:
            results.append(fc_output)
        else:
            logger.error("Graph returned no output for claim %s", fc_input.claim_id)

    return results


def run_fact_check_by_claim_ids(claim_ids: list[str]) -> list[FactCheckOutput]:
    """Fact-check claims that are already ingested in the DB.

    For each claim_id this:
      1. Loads claim_text + metadata from ChromaDB (`get_claims_by_ids`)
      2. Loads entities from Neo4j (`get_entity_context`)
      3. Loads the parent Article URL from Neo4j (`get_article_url_by_id`)
      4. Loads the article's caption + image (if any)
      5. Builds a FactCheckInput with real (DB-backed) IDs and invokes the graph

    Use this when the caller already has claim_ids from `decompose_input(query)`
    (cache-hit path) and does NOT have the original PreprocessingOutput in scope.
    For the cold path where you DO have the output, prefer `run_fact_check(output)`.
    """
    memory = get_memory()
    graph = _get_graph()

    if not claim_ids:
        return []

    claims_result = memory.get_claims_by_ids(claim_ids)
    docs = claims_result.get("documents") or []
    metas = claims_result.get("metadatas") or []
    ids = claims_result.get("ids") or []

    # Map id → (text, meta) for stable iteration order matching `claim_ids` arg
    by_id = {cid: (docs[i], metas[i]) for i, cid in enumerate(ids)}

    results: list[FactCheckOutput] = []
    for claim_id in claim_ids:
        record = by_id.get(claim_id)
        if record is None:
            logger.error("run_fact_check_by_claim_ids: claim_id not in DB: %s", claim_id)
            continue
        claim_text, meta = record
        meta = meta or {}
        article_id = meta.get("article_id", "")
        topic_text = meta.get("topic_text", "") or ""
        extracted_at_iso = meta.get("extracted_at", "")

        # Entities
        entity_rows = memory.get_entity_context(claim_id) or []
        entity_refs = [
            EntityRef(
                entity_id=e.get("entity_id", ""),
                name=e.get("name", ""),
                entity_type=e.get("entity_type", "") or "unknown",
                sentiment=e.get("sentiment", "neutral") or "neutral",
            )
            for e in entity_rows
            if e.get("entity_id")
        ]

        # Source URL (article-level)
        source_url = ""
        if article_id:
            source_url = memory.get_article_url_by_id(article_id) or ""

        # Image caption (article-level)
        image_caption: Optional[str] = None
        image_url: Optional[str] = None
        if article_id:
            caption_result = memory.get_caption_by_article(article_id) or {}
            cap_docs = caption_result.get("documents") or []
            cap_metas = caption_result.get("metadatas") or []
            if cap_docs:
                image_caption = cap_docs[0]
            if cap_metas:
                image_url = (cap_metas[0] or {}).get("image_url")

        # Timestamp
        try:
            timestamp = (
                datetime.fromisoformat(extracted_at_iso)
                if extracted_at_iso
                else datetime.now(timezone.utc)
            )
        except ValueError:
            timestamp = datetime.now(timezone.utc)

        fc_input = FactCheckInput(
            claim_id=claim_id,
            claim_text=claim_text,
            entities=entity_refs,
            source_url=source_url or "https://unknown.source",
            article_id=article_id or make_id("art_"),
            image_caption=image_caption,
            image_url=image_url,
            timestamp=timestamp,
            topic_text=topic_text,
        )

        lf = get_langfuse_handler()
        state = graph.invoke({"input": fc_input}, config={"callbacks": [lf]} if lf else {})
        fc_output: Optional[FactCheckOutput] = state.get("output")
        if fc_output:
            results.append(fc_output)
        else:
            logger.error("Graph returned no output for claim %s", claim_id)

    return results


def fact_check_claim(
    claim_text: str,
    source_url: str = "https://unknown.source",
    image_url: Optional[str] = None,
    image_caption: Optional[str] = None,
) -> FactCheckOutput:
    """Synthetic single-claim shortcut — used by `evaluation/evaluation.py` for
    benchmark runs on raw claim text (no DB ingest needed).

    The Streamlit frontend no longer uses this path; user-typed queries now
    flow through `decompose_input` + `run_fact_check_by_claim_ids` so the DB
    is properly populated. Keep this function only for benchmark scripts that
    need a stateless, fast call on a list of raw claim strings.
    """
    graph = _get_graph()
    fc_input = FactCheckInput(
        claim_id=make_id("clm_"),
        claim_text=claim_text,
        entities=[],
        source_url=source_url,
        article_id=make_id("art_"),
        image_caption=image_caption,
        image_url=image_url or None,
        timestamp=datetime.now(timezone.utc),
    )
    state = graph.invoke({"input": fc_input})
    fc_output: Optional[FactCheckOutput] = state.get("output")
    if fc_output is None:
        logger.error("Graph returned no output for synthetic claim")
        fc_output = FactCheckOutput(
            verdict_id=make_id("vrd_"),
            claim_id=fc_input.claim_id,
            verdict="misleading",
            confidence_score=0,
            evidence_links=[],
            reasoning="Pipeline error — no output produced.",
        )
    return fc_output
