"""Routing functions for the LangGraph conditional edges."""
from fact_check_agent.src.config import settings
from fact_check_agent.src.models.state import FactCheckState


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
