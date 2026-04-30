"""Tests for the ScrapeRun write path: MemoryAgent.add_scrape_run + GraphStore.create_scrape_run.

ScrapeRun nodes are how scheduled scraper runs (local + GitHub Actions cron)
land in Neo4j Aura, where Grafana / the Aura browser can read them. Breakage
in this path means the dashboard's "Scheduled scrapes" history goes silent
without anything else looking obviously wrong, so we lock the contract:

- The MemoryAgent facade forwards kwargs to GraphStore verbatim.
- GraphStore.create_scrape_run produces the expected CREATE Cypher with all
  9 properties.
- GraphStore.init_schema includes the constraint + time index for the new node
  label.

No real Neo4j connection — the driver is mocked.
"""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

# Required by pydantic-settings during transitive imports of src.config.
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test_unused")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-unused")

from src.memory.agent import MemoryAgent  # noqa: E402
from src.memory.graph_store import GraphStore  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _bare_graph_store() -> GraphStore:
    """Build a GraphStore without going through __init__ (which connects to Neo4j).

    Tests then replace `._driver` with a MagicMock and exercise methods directly.
    """
    gs = GraphStore.__new__(GraphStore)
    gs._driver = MagicMock()
    return gs


def _bare_memory_agent() -> MemoryAgent:
    """Build a MemoryAgent without going through __init__ (which opens DB connections)."""
    agent = MemoryAgent.__new__(MemoryAgent)
    agent._graph = MagicMock()
    agent._vector = MagicMock()
    agent._embeddings = MagicMock()
    return agent


def _captured_session_run(graph_store: GraphStore) -> list[tuple[str, dict]]:
    """Return the (cypher, params) tuples captured from session.run calls."""
    session = MagicMock()
    graph_store._driver.session.return_value.__enter__.return_value = session
    graph_store._driver.session.return_value.__exit__.return_value = False
    return session  # caller inspects session.run.call_args_list


# ──────────────────────────────────────────────────────────────────────────────
# MemoryAgent facade
# ──────────────────────────────────────────────────────────────────────────────


def test_add_scrape_run_forwards_kwargs_to_graph():
    agent = _bare_memory_agent()
    started = datetime(2026, 4, 30, 10, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 4, 30, 10, 4, 7, tzinfo=timezone.utc)

    agent.add_scrape_run(
        run_id="sr_abc123",
        started_at=started,
        finished_at=finished,
        duration_s=247.0,
        scraped=42,
        ingested=38,
        skipped=4,
        failed=0,
        source="local",
    )

    agent._graph.create_scrape_run.assert_called_once_with(
        run_id="sr_abc123",
        started_at=started,
        finished_at=finished,
        duration_s=247.0,
        scraped=42,
        ingested=38,
        skipped=4,
        failed=0,
        source="local",
    )


# ──────────────────────────────────────────────────────────────────────────────
# GraphStore.create_scrape_run — Cypher contract
# ──────────────────────────────────────────────────────────────────────────────


def test_create_scrape_run_emits_create_cypher_with_all_props():
    gs = _bare_graph_store()
    session = _captured_session_run(gs)

    started = datetime(2026, 4, 30, 6, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 4, 30, 6, 4, 7, tzinfo=timezone.utc)

    gs.create_scrape_run(
        run_id="sr_x1",
        started_at=started,
        finished_at=finished,
        duration_s=247.5,
        scraped=10,
        ingested=8,
        skipped=2,
        failed=0,
        source="github_actions",
    )

    session.run.assert_called_once()
    cypher = session.run.call_args.args[0]
    params = session.run.call_args.kwargs

    # Cypher pattern
    assert "CREATE (sr:ScrapeRun" in cypher

    # All 9 properties in params
    expected = {
        "run_id": "sr_x1",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": 247.5,
        "scraped": 10,
        "ingested": 8,
        "skipped": 2,
        "failed": 0,
        "source": "github_actions",
    }
    for k, v in expected.items():
        assert params[k] == v, f"param {k}: expected {v!r}, got {params.get(k)!r}"


def test_create_scrape_run_iso_formats_datetimes():
    """Datetimes must be ISO strings — Neo4j Cypher datetime() function expects them."""
    gs = _bare_graph_store()
    session = _captured_session_run(gs)

    started = datetime(2026, 4, 30, 0, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 4, 30, 0, 0, 1, tzinfo=timezone.utc)

    gs.create_scrape_run(
        run_id="sr_a",
        started_at=started,
        finished_at=finished,
        duration_s=1.0,
        scraped=0, ingested=0, skipped=0, failed=0,
        source="local",
    )

    params = session.run.call_args.kwargs
    assert params["started_at"] == "2026-04-30T00:00:00+00:00"
    assert params["finished_at"] == "2026-04-30T00:00:01+00:00"


# ──────────────────────────────────────────────────────────────────────────────
# GraphStore.init_schema — constraint + index for ScrapeRun
# ──────────────────────────────────────────────────────────────────────────────


def test_init_schema_adds_scrape_run_constraint_and_index():
    """When init_schema runs, both the uniqueness constraint and the time index
    for :ScrapeRun must be issued so Grafana time-range queries are fast and
    re-runs of pipeline.py don't insert duplicates."""
    gs = _bare_graph_store()
    session = _captured_session_run(gs)

    gs.init_schema()

    statements = [call.args[0] for call in session.run.call_args_list]

    # Constraint
    assert any(
        "CREATE CONSTRAINT" in s
        and ":ScrapeRun" in s
        and "run_id IS UNIQUE" in s
        for s in statements
    ), "Missing :ScrapeRun uniqueness constraint in init_schema"

    # Time index
    assert any(
        "CREATE INDEX" in s
        and ":ScrapeRun" in s
        and "started_at" in s
        for s in statements
    ), "Missing :ScrapeRun time index on started_at"
