"""Fact-check pipeline: verify unverified claims written by the scraper.

Queries Neo4j for claims with status='pending' created within the lookback
window, builds a FactCheckInput for each, runs the LangGraph fact-check
pipeline, and writes the resulting verdict back to Neo4j + ChromaDB.

Usage (from FakeNewsAgent/):
    python fact_check_pipeline.py
    python fact_check_pipeline.py --lookback-hours 48 --sleep-between 3.0

GitHub Actions runs this as a second step after the scraper, with a default
lookback of 24 h to cover the daily scrape window.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

# _bootstrap wires scraper_preprocessing_memory onto sys.path
from fact_check_agent.src.memory_client import close_memory, get_memory
from fact_check_agent.src.graph.graph import build_graph
from fact_check_agent.src.models.schemas import EntityRef, FactCheckInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
pipeline_logger = logging.getLogger("pipeline")

_DEFAULT_LOOKBACK_HOURS = 24
_DEFAULT_SLEEP_BETWEEN_S = 2.0


def _build_input(row: dict) -> FactCheckInput:
    entities = [
        EntityRef(
            entity_id=e["entity_id"],
            name=e["name"],
            entity_type=e.get("entity_type") or "unknown",
            sentiment=e.get("sentiment") or "neutral",
        )
        for e in (row.get("entities") or [])
    ]

    extracted_at = row.get("extracted_at")
    if hasattr(extracted_at, "to_native"):
        extracted_at = extracted_at.to_native()
    if extracted_at is None:
        extracted_at = datetime.now(timezone.utc)
    if extracted_at.tzinfo is None:
        extracted_at = extracted_at.replace(tzinfo=timezone.utc)

    return FactCheckInput(
        claim_id=row["claim_id"],
        claim_text=row["claim_text"],
        entities=entities,
        source_url=row.get("article_url") or "",
        article_id=row.get("article_id") or "",
        image_url=row.get("image_url") or None,
        image_caption=None,
        timestamp=extracted_at,
        topic_text=row.get("topic_text") or "",
    )


def run(lookback_hours: int = _DEFAULT_LOOKBACK_HOURS, sleep_between: float = _DEFAULT_SLEEP_BETWEEN_S) -> dict:
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    logger.info(
        "Fact-check pipeline starting — lookback %dh (since %s UTC)",
        lookback_hours,
        since.strftime("%Y-%m-%d %H:%M"),
    )

    memory = get_memory()
    graph = build_graph(memory)

    claims = memory.get_unverified_claims_since(since)
    logger.info("Found %d unverified claims to process", len(claims))

    processed = 0
    failed = 0

    for i, row in enumerate(claims):
        claim_id = row.get("claim_id", "?")
        try:
            inp = _build_input(row)
            pipeline_logger.info(
                "[%d/%d] claim_id=%s  %s",
                i + 1,
                len(claims),
                claim_id,
                inp.claim_text[:100],
            )
            state = graph.invoke({"input": inp})
            output = state.get("output")
            if output:
                pipeline_logger.info(
                    "  verdict=%-12s  confidence=%d%%  claim_id=%s",
                    output.verdict,
                    output.confidence_score,
                    claim_id,
                )
            processed += 1
        except Exception as exc:
            logger.error("claim_id=%s failed: %s", claim_id, exc)
            failed += 1

        if i < len(claims) - 1:
            time.sleep(sleep_between)

    close_memory()

    summary = {"found": len(claims), "processed": processed, "failed": failed}
    logger.info("Fact-check pipeline complete: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fact-check unverified scraped claims")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=_DEFAULT_LOOKBACK_HOURS,
        help="Process claims extracted within this many hours (default: 24)",
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=_DEFAULT_SLEEP_BETWEEN_S,
        help="Seconds to sleep between claims to avoid rate limits (default: 2.0)",
    )
    args = parser.parse_args()
    run(lookback_hours=args.lookback_hours, sleep_between=args.sleep_between)


if __name__ == "__main__":
    main()
