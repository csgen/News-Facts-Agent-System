"""End-to-end pipeline: Scraper → Preprocessing → Memory Agent."""

import logging
import time

from src.config import settings
from src.memory.agent import MemoryAgent
from src.preprocessing.agent import PreprocessingAgent
from src.scraper.agent import ScraperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(max_per_source: int = 20) -> dict:
    """Run the full data pipeline once.

    Returns a summary dict with counts.
    """
    logger.info("Starting pipeline run...")

    scraper = ScraperAgent(settings)
    preprocessor = PreprocessingAgent(settings)
    memory = MemoryAgent(settings)

    try:
        # Step 1: Scrape
        raw_articles = scraper.scrape(max_per_source=max_per_source)
        logger.info("Scraped %d raw articles", len(raw_articles))

        # Step 2: Preprocess + Ingest
        ingested = 0
        skipped = 0
        failed = 0

        for raw in raw_articles:
            try:
                output = preprocessor.process(raw)
                was_new = memory.ingest_preprocessed(output)
                if was_new:
                    ingested += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error("Failed to process article '%s': %s", raw.title[:50], e)
                failed += 1

        # Step 3: Entity reconciliation (cluster variants → merge)
        # ── Quota-window cooldown ────────────────────────────────────────
        # Article ingest just consumed a chunk of Gemini's free-tier 100 RPM
        # quota (each text inside a batch counts as a separate quota unit
        # against `embed_content_free_tier_requests`). Reconcile then bursts
        # several `embed_batch` calls back-to-back (one per entity_type) — if
        # the rolling 60s window isn't drained first, we reliably trip 429.
        # 60 s gives the window time to fully clear before reconcile starts.
        # Safe on cron / GitHub Actions (no idle-killer; default job timeout
        # is 6 h).
        _RECONCILE_COOLDOWN_S = 60
        logger.info(
            "Sleeping %ds before entity reconciliation to drain the Gemini quota window...",
            _RECONCILE_COOLDOWN_S,
        )
        time.sleep(_RECONCILE_COOLDOWN_S)

        try:
            reconcile_summary = memory.reconcile_entities()
        except Exception as e:
            logger.error("Entity reconciliation failed: %s", e)
            reconcile_summary = {}

        # Step 4: Promote significant entities to canonical list
        try:
            promoted = memory.promote_canonical_candidates()
        except Exception as e:
            logger.error("Canonical promotion failed: %s", e)
            promoted = []

        summary = {
            "scraped": len(raw_articles),
            "ingested": ingested,
            "skipped_duplicates": skipped,
            "failed": failed,
            "reconcile": reconcile_summary,
            "promoted_canonical": len(promoted),
        }
        logger.info("Pipeline complete: %s", summary)
        return summary

    finally:
        memory.close()


def main():
    """Entry point for docker CMD."""
    run_pipeline()


if __name__ == "__main__":
    main()
