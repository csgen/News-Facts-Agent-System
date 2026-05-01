"""Central failure logger for the Fact-Check Agent pipeline.

Fires nfs_pipeline_failures_total{failure_type} (Prometheus) and writes a
PipelineFailure node to Neo4j on every handled failure.

Usage:
    from fact_check_agent.src.failure_logger import log_failure

    log_failure(
        memory=memory,               # None → skips Neo4j write; counter still fires
        claim_id=inp.claim_id,
        node_name="synthesize_verdict",
        failure_type="json_parse_error",
        exception=e,
        raw_llm_response=raw,        # text captured before json.loads()
    )

failure_type values: llm_api_error | tavily_error | pydantic_error |
                     json_parse_error | db_error
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import fact_check_agent.src.id_utils as _id_utils

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent

logger = logging.getLogger(__name__)

# ── Prometheus counter ────────────────────────────────────────────────────────

try:
    from prometheus_client import Counter

    _PIPELINE_FAILURES = Counter(
        "nfs_pipeline_failures_total",
        "Pipeline failures by type across the fact-check agent",
        ["failure_type"],
    )
except ImportError:
    class _Noop:  # type: ignore[no-redef]
        def labels(self, **_): return self
        def inc(self): pass

    _PIPELINE_FAILURES = _Noop()  # type: ignore[assignment]


# ── Public helper ─────────────────────────────────────────────────────────────


def log_failure(
    *,
    memory: Optional["MemoryAgent"],
    claim_id: str,
    node_name: str,
    failure_type: str,
    exception: Exception,
    raw_llm_response: str = "",
) -> None:
    """Increment the Prometheus counter and write a PipelineFailure node.

    The Neo4j write is skipped when memory is None (tool-level callers that
    have no DB access) or when the write itself fails — the counter always fires.
    """
    _PIPELINE_FAILURES.labels(failure_type=failure_type).inc()

    if memory is None:
        return

    try:
        memory.write_pipeline_failure(
            failure_id=_id_utils.make_id("fail_"),
            claim_id=claim_id,
            node_name=node_name,
            failure_type=failure_type,
            raw_llm_response=raw_llm_response,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            occurred_at=datetime.now(timezone.utc),
        )
    except Exception as write_exc:
        logger.warning(
            "log_failure: Neo4j write failed (%s) — Prometheus counter was still incremented",
            write_exc,
        )
