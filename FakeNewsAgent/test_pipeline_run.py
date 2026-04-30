"""Manual test: run fact-check graph on 2 unverified claims with full node I/O logging.

Usage (from FakeNewsAgent/):
    python test_pipeline_run.py
    python test_pipeline_run.py --limit 1 --lookback-hours 72
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load root .env before any project imports so cloud DB creds are available
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# Bootstrap must fire before any src.* imports from scraper_preprocessing_memory
import fact_check_agent.src._bootstrap  # noqa: F401
from fact_check_agent.src.graph.graph import build_graph
from fact_check_agent.src.memory_client import close_memory, get_memory
from fact_check_agent.src.models.schemas import EntityRef, FactCheckInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_run")


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


def _log_state(node_name: str, state: dict) -> None:
    """Log the fields that changed after a node completes."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  NODE: {node_name}")
    print(sep)

    # Log the most informative fields per node
    if "input" in state and state["input"]:
        inp = state["input"]
        print(f"  claim_id   : {inp.claim_id}")
        print(f"  claim_text : {inp.claim_text[:120]}")

    if "memory_results" in state and state["memory_results"]:
        mr = state["memory_results"]
        print(f"  memory_results: {len(mr.results)} hits, max_confidence={mr.max_confidence:.2f}")
        for r in mr.results[:3]:
            conf = f"{r.verdict_confidence:.0%}" if r.verdict_confidence is not None else "n/a"
            print(f"    · [{r.distance:.3f}] {r.verdict_label} ({conf})  {r.claim_text[:80]}")

    if "fresh_context" in state and state["fresh_context"]:
        fc = state["fresh_context"]
        print(f"  fresh_context: {len(fc)} entries")
        for f in fc[:3]:
            dist = f.get("distance")
            dist_str = f"{dist:.3f}" if dist is not None else "?"
            print(f"    · dist={dist_str}  {f.get('verdict_label','?')}  {str(f.get('claim_text',''))[:80]}")

    if "retrieved_chunks" in state and state["retrieved_chunks"]:
        print(f"  retrieved_chunks: {len(state['retrieved_chunks'])} chunks")

    if "sub_claims" in state and state["sub_claims"]:
        print(f"  sub_claims: {state['sub_claims']}")

    if "context_claims" in state and state["context_claims"]:
        cc = state["context_claims"]
        print(f"  context_claims: {len(cc)} total")
        for c in cc:
            src = c.get("source", "?")
            ctype = c.get("type", "?")
            score = f"  score={c['score']:.2f}" if c.get("score") else ""
            print(f"    · [{ctype}/{src}{score}] {str(c.get('content',''))[:90]}")

    if "vlm_assessment_block" in state and state["vlm_assessment_block"]:
        print(f"  vlm_assessment_block: {state['vlm_assessment_block'][:200]}")

    if "cross_modal_flag" in state:
        score = state.get("clip_similarity_score")
        score_str = f"  siglip={score:.3f}" if score is not None else ""
        print(f"  cross_modal_flag: {state['cross_modal_flag']}{score_str}")

    if "output" in state and state["output"]:
        o = state["output"]
        print(f"  OUTPUT verdict    : {o.verdict}")
        print(f"  OUTPUT confidence : {o.confidence_score}%")
        print(f"  OUTPUT reasoning  : {o.reasoning[:200]}")
        if o.image_url:
            print(f"  OUTPUT image_url  : {o.image_url}")

    print()


def run_claim(graph, inp: FactCheckInput) -> None:
    print(f"\n{'═' * 70}")
    print(f"  CLAIM: {inp.claim_id}")
    print(f"  TEXT : {inp.claim_text}")
    print(f"  TOPIC: {inp.topic_text}  SOURCE: {inp.source_url[:60]}")
    print(f"{'═' * 70}")

    for step in graph.stream({"input": inp}, stream_mode="updates"):
        for node_name, state_update in step.items():
            _log_state(node_name, state_update)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2, help="Max claims to test (default: 2)")
    parser.add_argument("--lookback-hours", type=int, default=72, help="Lookback window in hours (default: 72)")
    args = parser.parse_args()

    memory = get_memory()

    since = datetime.now(timezone.utc) - timedelta(hours=args.lookback_hours)
    claims = memory.get_unverified_claims_since(since)

    if not claims:
        logger.warning("No unverified claims found in the last %dh — try a wider --lookback-hours", args.lookback_hours)
        close_memory()
        sys.exit(0)

    logger.info("Found %d unverified claims — running first %d", len(claims), args.limit)
    claims = claims[:args.limit]

    graph = build_graph(memory)

    for row in claims:
        inp = _build_input(row)
        try:
            run_claim(graph, inp)
        except Exception as exc:
            logger.error("claim_id=%s failed: %s", row.get("claim_id"), exc, exc_info=True)

    close_memory()


if __name__ == "__main__":
    main()
