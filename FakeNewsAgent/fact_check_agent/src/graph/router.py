"""Routing functions for the LangGraph conditional edges."""

from fact_check_agent.src.config import settings
from fact_check_agent.src.models.state import FactCheckState

# Mirrors the constants in nodes.py — both must be kept in sync.
_CACHE_EXACT_DISTANCE = 0.05
_CACHE_MIN_CONFIDENCE = 0.70


def cache_hit_check(state: FactCheckState) -> str:
    """Route to cached verdict when a near-identical fresh claim exists in memory.

    Returns "hit" when ALL of the following hold:
      - At least one fresh_context entry has distance < _CACHE_EXACT_DISTANCE
        (same claim, not just a similar one)
      - That entry's verdict_confidence >= _CACHE_MIN_CONFIDENCE
      - offline_mode is not active (cache bypass not forced)

    Returns "miss" otherwise → normal pipeline continues.
    """
    if settings.offline_mode:
        return "miss"

    fresh = state.get("fresh_context") or []
    for chunk in fresh:
        dist = chunk.get("distance", 1.0)
        conf = chunk.get("verdict_confidence") or 0.0
        label = chunk.get("verdict_label")
        if dist < _CACHE_EXACT_DISTANCE and conf >= _CACHE_MIN_CONFIDENCE and label:
            return "hit"
    return "miss"


def debate_check(state: FactCheckState) -> str:
    """Decide whether to trigger multi-agent debate or VLM-only judge.

    Routes through multi_agent_debate when:
      - VLM image assessment is present (runs Judge with image signal even when
        use_debate=False so the image evidence reaches the final verdict), OR
      - use_debate=True AND verdict confidence is below the threshold.
    """
    vlm_block = state.get("vlm_assessment_block") or ""
    if vlm_block and vlm_block != "No image available.":
        return "debate"

    if settings.use_debate:
        output = state.get("output")
        if output and output.confidence_score < settings.debate_confidence_threshold:
            return "debate"
    return "skip"
