"""LangGraph state schema for the Fact-Check Agent graph."""
from typing import Optional, TypedDict

from fact_check_agent.src.models.schemas import (
    FactCheckInput,
    FactCheckOutput,
    MemoryQueryResponse,
)


class FactCheckState(TypedDict):
    # ── Input (required — set before graph.invoke()) ─────────────────────────
    input: FactCheckInput

    # ── Memory query results ──────────────────────────────────────────────────
    memory_results: Optional[MemoryQueryResponse]
    entity_context: list[dict]

    # ── Freshness-tagged memory context ──────────────────────────────────────
    fresh_context: list[dict]
    stale_context: list[dict]

    # ── Context claims (from context_claim_agent) ─────────────────────────────
    context_claims: list[dict]

    # ── Prefetched chunks (benchmark mode — Factify2 doc + OCR) ──────────────
    retrieved_chunks: list[str]

    # ── Neutral synthesis output (fed into Supporter + Skeptic) ──────────────
    neutral_degrees: list[float]
    neutral_reasoning: Optional[str]

    # ── Multi-agent debate ────────────────────────────────────────────────────
    debate_transcript: Optional[str]

    # ── VLM image assessment (feeds into Judge prompt) ────────────────────────
    vlm_assessment_block: Optional[str]

    # ── Source credibility (from Reflection Agent) ────────────────────────────
    source_credibility: Optional[dict]

    # ── Effective topic (classified by query_memory when topic_text is empty) ──
    effective_topic: str

    # ── Cross-modal ───────────────────────────────────────────────────────────
    cross_modal_flag: bool
    vlm_assessment_block: Optional[str]
    clip_similarity_score: Optional[float]

    # ── Final output ──────────────────────────────────────────────────────────
    output: Optional[FactCheckOutput]


# Default values passed alongside FactCheckInput in graph.invoke()
INITIAL_STATE: dict = {
    "memory_results":          None,
    "entity_context":          [],
    "fresh_context":           [],
    "stale_context":           [],
    "context_claims":          [],
    "retrieved_chunks":        [],
    "neutral_degrees":         [],
    "neutral_reasoning":       None,
    "debate_transcript":       None,
    "vlm_assessment_block":    None,
    "source_credibility":      None,
    "effective_topic":         "",
    "cross_modal_flag":        False,
    "clip_similarity_score":   None,
    "output":                  None,
}
