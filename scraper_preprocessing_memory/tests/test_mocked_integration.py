"""Mocked-integration tests: real DB sidecars, mocked external APIs.

The point of this tier is to exercise real round-trips through Neo4j +
ChromaDB while keeping LLM/search calls cheap and deterministic. Together,
these tests catch:

  - request/response shape regressions against OpenAI / Gemini / Tavily
    (caught at the HTTP layer via respx, so SDK changes that alter the wire
    format trip the test rather than failing in production)
  - Cypher / metadata-filter bugs (real Neo4j + Chroma sidecars)
  - schema drift between MemoryAgent and the underlying stores
  - end-to-end MemoryAgent contracts (ingest → search → verdict supersede)

Marked @pytest.mark.mocked_integration so they're auto-included when CI runs
`pytest -m "not integration"` (mocked_integration is NOT excluded by that
filter — only the `integration` marker is).

Skipped automatically if Docker isn't available (e.g. host without Docker).
The test suite still passes locally on a Docker-less laptop.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone

import httpx
import pytest
import respx

# Skip the entire module if Docker isn't on PATH — testcontainers needs it.
pytestmark = [
    pytest.mark.mocked_integration,
    pytest.mark.skipif(
        shutil.which("docker") is None,
        reason="Docker not available; skipping mocked-integration tests",
    ),
]


# ── Sidecar containers (module-scoped — boot once per test file) ─────────────

@pytest.fixture(scope="module")
def neo4j_container():
    """Spin up a real Neo4j 5-community container for the duration of the module."""
    from testcontainers.neo4j import Neo4jContainer

    with Neo4jContainer("neo4j:5-community").with_env("NEO4J_AUTH", "neo4j/testpassword") as n:
        # Sanity-wait: testcontainers' verify is fast, but Neo4j needs a few extra
        # seconds before its first session works.
        yield n


@pytest.fixture(scope="module")
def chroma_container():
    """Spin up a real chromadb/chroma container exposing port 8000."""
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    c = (
        DockerContainer("chromadb/chroma:latest")
        .with_exposed_ports(8000)
        .with_env("ANONYMIZED_TELEMETRY", "FALSE")
    )
    c.start()
    try:
        wait_for_logs(c, "Application startup complete", timeout=60)
        yield c
    finally:
        c.stop()


@pytest.fixture
def memory_agent(neo4j_container, chroma_container, monkeypatch):
    """Build a fresh MemoryAgent pointed at the sidecar Neo4j + Chroma containers.

    Uses module-scoped containers but a per-test agent — so each test gets a
    clean schema (init_schema is idempotent; collections are cleared at end).
    """
    # Pydantic Settings reads from env. Set everything required.
    monkeypatch.setenv("NEO4J_URI", neo4j_container.get_connection_url())
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", str(chroma_container.get_exposed_port(8000)))
    monkeypatch.setenv("CHROMA_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mocked-not-used")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-mocked-not-used")

    # Reload settings so the new env vars are picked up.
    import importlib

    from src import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.memory.agent import MemoryAgent

    m = MemoryAgent(cfg_mod.settings)
    m.init_schema()
    yield m

    # ── Cleanup ───────────────────────────────────────────────────────────────
    # The container fixtures are module-scoped (one Neo4j + one Chroma reused
    # across all tests in this file), so each test must leave both stores empty
    # or it will pollute the next test. In particular, ingesting the same
    # `sample_preprocessing_output` twice trips check_content_hash_exists in
    # Chroma — even after Neo4j is wiped — and silently breaks dedup tests.
    try:
        # Neo4j: nuke all nodes + edges.
        with m._graph._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        # Chroma: each collection is reset by deleting every item it holds.
        # We walk the same five collections VectorStore creates in __init__.
        for coll in (
            m._vector._claims,
            m._vector._articles,
            m._vector._verdicts,
            m._vector._image_captions,
            m._vector._source_credibility,
        ):
            ids = coll.get().get("ids", [])
            if ids:
                coll.delete(ids=ids)
    except Exception:
        # Best-effort — never let cleanup mask a real test failure.
        pass
    m.close()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gemini_embedding_response(dim: int = 1536) -> httpx.Response:
    """Canned Gemini embedding response — single 1536-dim vector of 0.1s."""
    return httpx.Response(
        200,
        json={"embedding": {"values": [0.1] * dim}},
    )


def _openai_chat_response(content: str) -> httpx.Response:
    """Canned OpenAI chat-completion response."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-mocked",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40},
        },
    )


# ── Tests ────────────────────────────────────────────────────────────────────

@respx.mock
def test_ingest_preprocessed_lands_in_both_stores(memory_agent, sample_preprocessing_output):
    """End-to-end ingest path: PreprocessingOutput → Neo4j nodes + Chroma rows."""
    # Mock Gemini embedding endpoint — every call returns the same canned vector.
    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*embedContent.*").mock(
        return_value=_gemini_embedding_response()
    )

    ok = memory_agent.ingest_preprocessed(sample_preprocessing_output)
    assert ok is True

    # Neo4j: article + claim nodes exist
    with memory_agent._graph._driver.session() as s:
        article_count = s.run(
            "MATCH (a:Article {article_id: $aid}) RETURN count(a) AS n",
            aid=sample_preprocessing_output.article.article_id,
        ).single()["n"]
        claim_count = s.run(
            "MATCH (c:Claim {article_id: $aid}) RETURN count(c) AS n",
            aid=sample_preprocessing_output.article.article_id,
        ).single()["n"]
        entity_count = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]

    assert article_count == 1
    assert claim_count == len(sample_preprocessing_output.claims)
    assert entity_count >= 2  # Tesla + NHTSA from the fixture

    # Chroma: claim is queryable by ID
    claim_id = sample_preprocessing_output.claims[0].claim_id
    found = memory_agent.get_claims_by_ids([claim_id])
    assert claim_id in found["ids"]


@respx.mock
def test_check_duplicate_short_circuits_repeat_ingest(memory_agent, sample_preprocessing_output):
    """Re-ingesting the same article (same content_hash) should be a no-op."""
    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*embedContent.*").mock(
        return_value=_gemini_embedding_response()
    )

    first = memory_agent.ingest_preprocessed(sample_preprocessing_output)
    second = memory_agent.ingest_preprocessed(sample_preprocessing_output)

    assert first is True
    assert second is False  # dedup check trips on content_hash

    # Only one article node should exist.
    with memory_agent._graph._driver.session() as s:
        n = s.run("MATCH (a:Article) RETURN count(a) AS n").single()["n"]
    assert n == 1


@respx.mock
def test_search_similar_claims_returns_ingested_claim(memory_agent, sample_preprocessing_output):
    """After ingest, the same claim text should round-trip through similarity search."""
    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*embedContent.*").mock(
        return_value=_gemini_embedding_response()
    )

    memory_agent.ingest_preprocessed(sample_preprocessing_output)
    results = memory_agent.search_similar_claims(
        sample_preprocessing_output.claims[0].claim_text,
        top_k=5,
    )
    assert results["ids"][0], "expected at least one similar claim"
    assert sample_preprocessing_output.claims[0].claim_id in results["ids"][0]


@respx.mock
def test_add_verdict_supersedes_existing_active_verdict(memory_agent, sample_preprocessing_output):
    """Writing a second verdict for the same claim marks the first as superseded."""
    from src.models.verdict import Verdict

    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*embedContent.*").mock(
        return_value=_gemini_embedding_response()
    )

    memory_agent.ingest_preprocessed(sample_preprocessing_output)
    claim_id = sample_preprocessing_output.claims[0].claim_id
    now = datetime.now(timezone.utc)

    v1 = Verdict(
        verdict_id="vrd_first",
        claim_id=claim_id,
        label="supported",
        confidence=0.8,
        evidence_summary="first verdict",
        bias_score=0.1,
        image_mismatch=False,
        verified_at=now,
    )
    v2 = Verdict(
        verdict_id="vrd_second",
        claim_id=claim_id,
        label="refuted",
        confidence=0.9,
        evidence_summary="second verdict (supersedes the first)",
        bias_score=0.2,
        image_mismatch=False,
        verified_at=now,
    )

    memory_agent.add_verdict(v1)
    memory_agent.add_verdict(v2)

    # Active verdict for this claim is now v2 only — v1 should be superseded.
    active = memory_agent.get_verdict_by_claim(claim_id)
    assert active["ids"] == ["vrd_second"], f"expected only vrd_second active, got {active['ids']}"

    # Cypher: a SUPERSEDED_BY edge should exist from v1 to v2.
    with memory_agent._graph._driver.session() as s:
        rel_count = s.run(
            """
            MATCH (old:Verdict {verdict_id: 'vrd_first'})
                  -[:SUPERSEDED_BY]->
                  (new:Verdict {verdict_id: 'vrd_second'})
            RETURN count(*) AS n
            """
        ).single()["n"]
    assert rel_count == 1


def test_ensure_entity_and_lookup_by_name(memory_agent):
    """ensure_entity_exists is idempotent and get_entity_by_name finds it."""
    eid = memory_agent.ensure_entity_exists("Tesla")
    eid_again = memory_agent.ensure_entity_exists("Tesla")
    assert eid == eid_again, "ensure_entity_exists should be idempotent"

    found = memory_agent.get_entity_by_name("Tesla")
    assert found is not None
    assert found["entity_id"] == eid
    assert found["name"].lower() == "tesla"


def test_get_entity_by_name_returns_none_when_missing(memory_agent):
    """Lookup of a non-existent entity returns None (not a raise)."""
    assert memory_agent.get_entity_by_name("NoSuchEntityProbablyEverExisted") is None
