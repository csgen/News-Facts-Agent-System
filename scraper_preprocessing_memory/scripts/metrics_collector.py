"""Metrics collector sidecar — exposes Neo4j + Chroma sizes as Prometheus metrics.

Long-running daemon. Reuses MemoryAgent so it speaks the same DBs as the rest of
the stack (cloud or local — depends on the .env it's started with).

Exports a single Gauge with labels (store, label) so all node/collection counts
live on one metric, easy to graph in Grafana:

    nfs_db_node_count{store="neo4j",  label="Article"}                 = 1234
    nfs_db_node_count{store="neo4j",  label="Claim"}                   = 5678
    nfs_db_node_count{store="chroma", label="claims"}                  = 5678
    nfs_db_node_count{store="chroma", label="source_credibility"}      = 42
    ...

Polls every COLLECT_INTERVAL_S seconds (default 300 = 5 min). Prometheus scrapes
this process' /metrics endpoint at the same cadence (set in prometheus.yml's
job_name=metrics-collector).

Failures during a poll round are logged and survived — the loop never crashes.
"""

import logging
import os
import sys
import time

from prometheus_client import Gauge, start_http_server

sys.path.insert(0, ".")

from src.config import settings
from src.memory.agent import MemoryAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

PORT = int(os.getenv("METRICS_COLLECTOR_PORT", "8001"))
COLLECT_INTERVAL_S = int(os.getenv("METRICS_COLLECT_INTERVAL_S", "300"))

NEO4J_LABELS = [
    "Source",
    "Article",
    "Claim",
    "Entity",
    "Verdict",
    "ImageCaption",
    "CredibilitySnapshot",
    "Prediction",
    "ScrapeRun",
    "PipelineFailure",
]

CHROMA_COLLECTIONS = [
    "claims",
    "articles",
    "verdicts",
    "image_captions",
    "source_credibility",
]

# ── Prometheus metric ───────────────────────────────────────────────────────

DB_NODE_COUNT = Gauge(
    "nfs_db_node_count",
    "Size of each Neo4j node label / Chroma collection. "
    "store ∈ {neo4j, chroma}; label ∈ {Article, Claim, ..., claims, ...}.",
    ["store", "label"],
)

# Track collector health itself.
COLLECT_FAILURES = Gauge(
    "nfs_metrics_collector_last_status",
    "1 if the most recent collection round succeeded, 0 if it failed.",
)


# ── Collection logic ────────────────────────────────────────────────────────

def collect_neo4j(memory: MemoryAgent) -> None:
    """Run one count query per node label, set the corresponding Gauge."""
    with memory._graph._driver.session() as session:
        for label in NEO4J_LABELS:
            try:
                # No parameter substitution for labels in Cypher; safe-list controls input.
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                count = result.single()["c"]
                DB_NODE_COUNT.labels(store="neo4j", label=label).set(count)
            except Exception as e:
                logger.warning("Neo4j count for %s failed: %s", label, e)


def collect_chroma(memory: MemoryAgent) -> None:
    """Read each Chroma collection's count via the existing VectorStore handles.

    Tolerant of collections that don't exist on this VectorStore — getattr-with-
    default skips them rather than aborting the whole poll round on AttributeError.
    """
    vs = memory._vector
    for name in CHROMA_COLLECTIONS:
        collection = getattr(vs, f"_{name}", None)
        if collection is None:
            # VectorStore doesn't expose this collection in its current build —
            # silently skip so the rest of the poll round still runs.
            continue
        try:
            count = collection.count()
            DB_NODE_COUNT.labels(store="chroma", label=name).set(count)
        except Exception as e:
            logger.warning("Chroma count for %s failed: %s", name, e)


def main() -> None:
    logger.info("Starting metrics_collector on port %d (interval %ds)", PORT, COLLECT_INTERVAL_S)
    start_http_server(PORT)

    # MemoryAgent holds a Neo4j driver + Chroma client; one instance for the
    # lifetime of the daemon. Re-instantiation on each loop would be wasteful.
    memory = MemoryAgent(settings)

    while True:
        try:
            collect_neo4j(memory)
            collect_chroma(memory)
            COLLECT_FAILURES.set(1)
            logger.info("Collected DB sizes; sleeping %ds", COLLECT_INTERVAL_S)
        except Exception as e:
            COLLECT_FAILURES.set(0)
            logger.error("Collection round failed: %s", e)
        time.sleep(COLLECT_INTERVAL_S)


if __name__ == "__main__":
    main()
