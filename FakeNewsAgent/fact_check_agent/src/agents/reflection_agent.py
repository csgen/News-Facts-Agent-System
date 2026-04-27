"""Reflection Agent — post-verdict write-back and source credibility tracking.

Two public write entry points:

    record_verdict_outcome(output, claim_text, source_url, topic_text, memory)
        Called by write_memory node after agent verdict. Performs all 5 write-backs.
        Alpha learning rate: 0.05

    record_hitl_correction(verdict_id, fb_label, fb_confidence, source_url, claim_text, memory)
        Called from the frontend HITL feedback path after a human corrects a verdict.
        The verdict override itself is handled by memory.update_verdict_with_feedback().
        This function only applies the credibility alpha-update with higher weight.
        Alpha learning rate: 0.08 (human signal is more reliable)

Agent verdict write-backs (record_verdict_outcome):
  1. Claim status → ChromaDB
  2. Claim status → Neo4j
  3. Verdict       → ChromaDB
  4. Verdict       → Neo4j
  5. (Source)-[:HAS_CREDIBILITY]->(Topic) alpha-update → Neo4j

Design:
  - No LLM calls. No loops. Pure data utility wrapping MemoryAgent.
  - New (source, topic) pairs are created automatically via MERGE — no seeding needed.
"""
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent

    from fact_check_agent.src.models.schemas import FactCheckOutput

logger = logging.getLogger(__name__)

try:
    from src.models.verdict import Verdict
except ImportError:
    Verdict = None  # resolved at runtime via sys.path bootstrap in fact_check_agent.py

_ALPHA      = 0.05   # agent verdict credibility learning rate
_ALPHA_HITL = 0.08   # HITL correction learning rate (human signal weighted higher)
_DEFAULT_CREDIBILITY = 0.65  # fallback when source is unknown


# ── Helpers ───────────────────────────────────────────────────────────────────

def source_id_from_url(source_url: str) -> str:
    """Derive source_id from a URL — mirrors preprocessing/agent.py convention.

    Example: "https://bbc.co.uk/news/1" → "src_bbc_co_uk"
    """
    domain = urlparse(source_url).netloc or source_url
    return f"src_{domain.replace('.', '_')}"


def credibility_signal(verdict_label: str, confidence_score: int) -> float:
    """Map a verdict + confidence to a credibility observation in [0, 1].

      supported  @ conf  → conf/100        (source published truth)
      refuted    @ conf  → 1 - conf/100    (source published falsehood)
      misleading          → 0.5            (neutral)
    """
    c = confidence_score / 100.0
    if verdict_label == "supported":
        return c
    if verdict_label == "refuted":
        return 1.0 - c
    return 0.5


# ── Read path ─────────────────────────────────────────────────────────────────

def query_source_credibility(
    claim_text: str,
    source_url: str,
    memory: "MemoryAgent",
    topic: str = "",
) -> dict:
    """Return current (source, topic) credibility from Neo4j.

    Returns {"credibility_mean": float|None, "sample_count": int}.
    credibility_mean is None when no record exists yet for this pair.
    """
    source_id = source_id_from_url(source_url)
    if not topic:
        return {"credibility_mean": None, "sample_count": 0}

    try:
        credibility = memory.get_source_topic_credibility(source_id, topic)
        if credibility is None:
            # No dynamic record yet — fall back to static base_credibility
            credibility = memory.get_base_credibility(source_id)

        return {
            "credibility_mean": round(credibility, 4) if credibility is not None else None,
            "sample_count":     1 if credibility is not None else 0,
        }
    except Exception as exc:
        logger.warning("query_source_credibility failed for %s/%s: %s", source_id, topic, exc)
        return {"credibility_mean": None, "sample_count": 0}


# ── Write path — all 5 post-verdict updates ───────────────────────────────────

def record_verdict_outcome(
    output: "FactCheckOutput",
    claim_text: str,
    source_url: str,
    topic_text: str,
    memory: "MemoryAgent",
) -> None:
    """Consolidate all 5 write-backs after a verdict is produced.

    Failures in individual steps are logged but never propagate — the verdict
    is already in state and must not be lost because of a write error.
    """
    # ── 1 & 2: claim status in ChromaDB + Neo4j ───────────────────────────────
    try:
        memory.update_claim_status(output.claim_id, "verified")
    except Exception as exc:
        logger.error("update_claim_status failed for %s: %s", output.claim_id, exc)

    # ── 3 & 4: verdict in ChromaDB + Neo4j ───────────────────────────────────
    evidence_summary = output.reasoning
    if output.evidence_links:
        evidence_summary += "\n\nSources: " + " | ".join(output.evidence_links)

    verdict = Verdict(
        verdict_id       = output.verdict_id,
        claim_id         = output.claim_id,
        label            = output.verdict,
        confidence       = output.confidence_score / 100,
        evidence_summary = evidence_summary,
        image_mismatch   = output.cross_modal_flag,
        verified_at      = datetime.now(timezone.utc),
    )
    try:
        memory.add_verdict(verdict)
    except Exception as exc:
        logger.error("add_verdict failed for %s: %s", output.verdict_id, exc)

    # ── 5: (Source)-[:HAS_CREDIBILITY]->(Topic) alpha-update ─────────────────
    _update_credibility(output, source_url, topic_text, memory)


def _update_credibility(
    output: "FactCheckOutput",
    source_url: str,
    topic_text: str,
    memory: "MemoryAgent",
) -> None:
    source_id = source_id_from_url(source_url)
    topic     = topic_text.strip() or "unknown"
    signal    = credibility_signal(output.verdict, output.confidence_score)

    try:
        current = memory.get_source_topic_credibility(source_id, topic)
        if current is None:
            current = memory.get_base_credibility(source_id) or _DEFAULT_CREDIBILITY

        new_c = float(max(0.0, min(1.0, current + _ALPHA * (signal - 0.5))))

        memory.upsert_source_topic_credibility(source_id, topic, new_c)
        logger.info(
            "credibility update: source=%s topic=%s %.3f → %.3f (signal=%.2f verdict=%s)",
            source_id, topic, current, new_c, signal, output.verdict,
        )
    except Exception as exc:
        logger.error("credibility update failed for %s/%s: %s", source_id, topic, exc)


# ── HITL write path ───────────────────────────────────────────────────────────

def record_hitl_correction(
    verdict_id: str,
    fb_label: str,
    fb_confidence: float,   # 0-1 float as received from the frontend slider
    source_url: str,
    memory: "MemoryAgent",
) -> None:
    """Apply source-topic credibility update from a human-in-the-loop correction.

    The verdict override (Neo4j + ChromaDB) is already handled by
    memory.update_verdict_with_feedback(). This function only applies the
    credibility alpha-update with a higher learning rate (0.08) since human
    signals are more reliable than agent verdicts.

    Topic is traced from the Claim linked to the verdict in Neo4j — no fallback
    to LLM classification needed.
    """
    try:
        topic = memory.get_topic_for_verdict(verdict_id).strip() or "unknown"
    except Exception as exc:
        logger.warning("get_topic_for_verdict failed for %s: %s", verdict_id, exc)
        topic = "unknown"

    source_id      = source_id_from_url(source_url)
    confidence_int = int(fb_confidence * 100)
    signal         = credibility_signal(fb_label, confidence_int)

    try:
        current = memory.get_source_topic_credibility(source_id, topic)
        if current is None:
            current = memory.get_base_credibility(source_id) or _DEFAULT_CREDIBILITY

        new_c = float(max(0.0, min(1.0, current + _ALPHA_HITL * (signal - 0.5))))

        memory.upsert_source_topic_credibility(source_id, topic, new_c)
        logger.info(
            "HITL credibility update: source=%s topic=%s %.3f → %.3f "
            "(signal=%.2f label=%s conf=%d%%)",
            source_id, topic, current, new_c, signal, fb_label, confidence_int,
        )
    except Exception as exc:
        logger.error(
            "HITL credibility update failed for %s/%s: %s", source_id, topic, exc
        )
