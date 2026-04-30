"""Tests for the monitoring sidecar's collect_neo4j / collect_chroma functions.

Locks in two regression contracts:

1. **Tolerance to missing Chroma collections.** A previous bug referenced
   `vs._source_credibility` directly during dict construction; when that
   collection was later removed from VectorStore, `collect_chroma` raised
   AttributeError and aborted the entire poll round, silently blanking the
   "Chroma collection counts" Grafana panel. The fix uses
   `getattr(vs, name, None)` and skips missing collections — these tests
   guard that behaviour.

2. **Per-target failure isolation.** A single failing Cypher count or a single
   failing Chroma `.count()` must not stop the rest of the poll round.

No real Neo4j / Chroma connections; everything is mocked.
"""

import os
from unittest.mock import MagicMock

# pydantic-settings reads these at module-import time when src.config is loaded
# (transitively imported by metrics_collector). Placeholders are fine for unit
# tests — we mock all DB calls.
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test_unused")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-unused")

from scripts import metrics_collector as mc  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_neo4j_session_memory(run_side_effect):
    """Build a fake MemoryAgent whose Neo4j session.run uses `run_side_effect`."""
    memory = MagicMock()
    session = MagicMock()
    # session is acquired via `with memory._graph._driver.session() as session:`
    memory._graph._driver.session.return_value.__enter__.return_value = session
    memory._graph._driver.session.return_value.__exit__.return_value = False
    session.run.side_effect = run_side_effect
    return memory, session


def _ok_run(_query, *_args, **_kwargs):
    """Default session.run side-effect: returns a single record with c=42."""
    result = MagicMock()
    result.single.return_value = {"c": 42}
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Chroma — the regression-prone path
# ──────────────────────────────────────────────────────────────────────────────


def test_collect_chroma_skips_missing_collections():
    """Regression: previously raised AttributeError on `vs._source_credibility`.

    Now uses getattr-with-default and silently skips missing collections,
    so the rest of the poll round still runs.
    """
    memory = MagicMock()
    # spec=[] starts the mock with NO attributes — getattr on missing names
    # returns the default rather than auto-creating a MagicMock child.
    vs = MagicMock(spec=[])
    coll_claims = MagicMock()
    coll_claims.count.return_value = 5
    vs._claims = coll_claims  # only one collection exists on this VectorStore
    memory._vector = vs

    mc.collect_chroma(memory)  # MUST NOT raise

    # The one present collection's .count() was queried
    coll_claims.count.assert_called_once()


def test_collect_chroma_per_collection_failure_isolated():
    """One collection's .count() raising must not abort the others."""
    memory = MagicMock()
    vs = MagicMock(spec=[])

    failing = MagicMock()
    failing.count.side_effect = RuntimeError("simulated chroma failure")
    working_a = MagicMock()
    working_a.count.return_value = 1
    working_b = MagicMock()
    working_b.count.return_value = 2

    # Names must match entries in mc.CHROMA_COLLECTIONS so the loop visits them.
    vs._claims = failing
    vs._articles = working_a
    vs._verdicts = working_b
    memory._vector = vs

    mc.collect_chroma(memory)  # MUST NOT raise

    # Both surviving collections still got polled
    working_a.count.assert_called_once()
    working_b.count.assert_called_once()
    # The failing one was attempted exactly once
    failing.count.assert_called_once()


def test_collect_chroma_iterates_all_known_collections():
    """Sanity: collect_chroma walks the canonical CHROMA_COLLECTIONS list."""
    memory = MagicMock()
    vs = MagicMock(spec=[])
    handles = {}
    for name in mc.CHROMA_COLLECTIONS:
        c = MagicMock()
        c.count.return_value = 1
        handles[name] = c
        setattr(vs, f"_{name}", c)
    memory._vector = vs

    mc.collect_chroma(memory)

    for c in handles.values():
        c.count.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# Neo4j
# ──────────────────────────────────────────────────────────────────────────────


def test_collect_neo4j_runs_all_labels():
    """Sanity: collect_neo4j runs one count query per NEO4J_LABELS entry."""
    memory, session = _make_neo4j_session_memory(_ok_run)

    mc.collect_neo4j(memory)

    assert session.run.call_count == len(mc.NEO4J_LABELS)


def test_collect_neo4j_partial_failure_continues():
    """One failing label must not abort the rest of the poll round."""
    call_state = {"n": 0}

    def run(query, *_args, **_kwargs):
        call_state["n"] += 1
        if call_state["n"] == 3:
            raise RuntimeError("simulated neo4j failure on 3rd label")
        result = MagicMock()
        result.single.return_value = {"c": call_state["n"] * 10}
        return result

    memory, session = _make_neo4j_session_memory(run)

    mc.collect_neo4j(memory)  # MUST NOT raise

    # All labels were attempted even though one in the middle threw
    assert session.run.call_count == len(mc.NEO4J_LABELS)


def test_collect_neo4j_uses_count_query_pattern():
    """Each Cypher query is `MATCH (n:<Label>) RETURN count(n) AS c`."""
    memory, session = _make_neo4j_session_memory(_ok_run)

    mc.collect_neo4j(memory)

    queries = [call.args[0] for call in session.run.call_args_list]
    # Every query begins with the expected MATCH pattern
    assert all(q.startswith("MATCH (n:") for q in queries), queries
    assert all("RETURN count(n) AS c" in q for q in queries), queries

    # Each NEO4J_LABEL appears in exactly one query
    for label in mc.NEO4J_LABELS:
        assert any(f"MATCH (n:{label})" in q for q in queries), \
            f"Missing query for label {label!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Module-level invariants
# ──────────────────────────────────────────────────────────────────────────────


def test_db_node_count_metric_has_expected_labels():
    """Lock in the gauge's `(store, label)` schema so dashboards keep working."""
    assert mc.DB_NODE_COUNT._name == "nfs_db_node_count"
    assert tuple(mc.DB_NODE_COUNT._labelnames) == ("store", "label")


def test_neo4j_labels_includes_scrape_run():
    """ScrapeRun was added at the same time as monitoring; ensure collector polls it."""
    assert "ScrapeRun" in mc.NEO4J_LABELS
