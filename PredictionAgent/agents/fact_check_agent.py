"""Fact-Check Agent adapter — wraps FakeNewsAgent's LangGraph pipeline.

Public API for Streamlit frontend and other PredictionAgent code:

    from agents.fact_check_agent import run_fact_check, fact_check_claim

    # Option A: full pipeline from a PreprocessingOutput
    outputs = run_fact_check(preprocessing_output)

    # Option B: quick check from raw claim text (creates synthetic context)
    output = fact_check_claim("Tesla recalled 500k vehicles due to brake defects")

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
        state = graph.invoke({"input": fc_input})
        fc_output: Optional[FactCheckOutput] = state.get("output")
        if fc_output:
            results.append(fc_output)
        else:
            logger.error("Graph returned no output for claim %s", fc_input.claim_id)

    return results


def fact_check_claim(
    claim_text: str,
    source_url: str = "https://unknown.source",
    image_url: Optional[str] = None,
    image_caption: Optional[str] = None,
) -> FactCheckOutput:
    """Convenience wrapper: fact-check a single raw text claim.

    Creates a synthetic FactCheckInput (no article, no entities) and runs the graph.
    Used by the Streamlit frontend for direct user queries.
    Pass image_url (and optionally image_caption) to enable cross-modal checking.
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
