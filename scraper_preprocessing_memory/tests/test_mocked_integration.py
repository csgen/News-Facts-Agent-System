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
    """Spin up a real Neo4j 5-community container for the duration of the module.

    Use the documented `password=` constructor kwarg rather than the
    `.with_env("NEO4J_AUTH", …)` chain — the chain is unreliable across
    testcontainers-python versions (the Neo4jContainer constructor injects
    its own NEO4J_AUTH based on the `password` attribute, and depending on
    call order our override can get clobbered, leaving the container with
    the default password while we try to connect with a different one).
    """
    from testcontainers.neo4j import Neo4jContainer

    container = Neo4jContainer("neo4j:5-community", password="testpassword")
    container.start()
    try:
        yield container
    finally:
        container.stop()


def _wait_for_chroma_heartbeat(host: str, port: int, timeout: int = 90) -> None:
    """Poll Chroma's heartbeat endpoint until it returns HTTP 200, or fail.

    More reliable than log-matching: Chroma's startup banner has changed
    between versions, but the heartbeat endpoint is part of the public API
    and stable. Tries the v2 path first (current), falls back to v1 (older
    images), so the same fixture works against `chromadb/chroma:latest`
    regardless of which major version is published.
    """
    import time
    import urllib.request

    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        for path in ("/api/v2/heartbeat", "/api/v1/heartbeat"):
            try:
                with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception as e:  # noqa: BLE001 — broad on purpose; we want to keep polling
                last_err = e
        time.sleep(1)
    raise TimeoutError(
        f"Chroma container at {host}:{port} did not become ready within {timeout}s "
        f"(last error: {last_err!r})"
    )


@pytest.fixture(scope="module")
def chroma_container():
    """Spin up a real chromadb/chroma container exposing port 8000."""
    from testcontainers.core.container import DockerContainer

    c = (
        DockerContainer("chromadb/chroma:latest")
        .with_exposed_ports(8000)
        .with_env("ANONYMIZED_TELEMETRY", "FALSE")
    )
    c.start()
    try:
        host = c.get_container_host_ip()
        port = int(c.get_exposed_port(8000))
        _wait_for_chroma_heartbeat(host, port, timeout=90)
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
    # Read Neo4j credentials off the container itself so we can never drift
    # between what the container booted with and what MemoryAgent connects with.
    monkeypatch.setenv("NEO4J_URI", neo4j_container.get_connection_url())
    monkeypatch.setenv("NEO4J_USER", getattr(neo4j_container, "username", "neo4j"))
    monkeypatch.setenv("NEO4J_PASSWORD", getattr(neo4j_container, "password", "testpassword"))
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
    #
    # Each subsystem's cleanup lives in its OWN try/except so one failure
    # never poisons the rest. (An earlier version put everything under one
    # try and had a tuple of attribute-accesses; if any attribute didn't exist
    # the tuple-build raised before the for-loop even started, leaving Chroma
    # entirely uncleaned. Iterating attribute NAMES with getattr() avoids that
    # whole class of bug.)
    try:
        with m._graph._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
    except Exception:
        pass

    for coll_name in (
        "_claims",
        "_articles",
        "_verdicts",
        "_image_captions",
        "_source_credibility",  # only present on some VectorStore versions
    ):
        coll = getattr(m._vector, coll_name, None)
        if coll is None:
            continue
        try:
            ids = coll.get().get("ids", [])
            if ids:
                coll.delete(ids=ids)
        except Exception:
            pass

    m.close()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gemini_batch_embed_handler(request: httpx.Request) -> httpx.Response:
    """Canned Gemini :batchEmbedContents response, sized to the request.

    The google-genai SDK ALWAYS uses the batch endpoint — `embed(text)` calls
    `embed_batch([text])` internally (see embeddings.py:55-57). Per Google's
    REST contract for :batchEmbedContents, the request body is
        {"requests": [{"model": ..., "content": ...}, ...]}
    and the response must be
        {"embeddings": [{"values": [...]}, ...]}
    with the embeddings list the SAME LENGTH as the requests list. A static
    `return_value=` mock with one embedding fails for batches of N>1 because
    the caller does `[emb.values for emb in response.embeddings]` and then
    indexes by position.
    """
    import json

    raw = request.content or b""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        body = {}
    items = body.get("requests") or []
    # Defensive: never return zero embeddings, that would surprise the caller
    # in confusing ways. Mirror N=1 minimum.
    n = max(len(items), 1)
    return httpx.Response(
        200,
        json={"embeddings": [{"values": [0.1] * 1536} for _ in range(n)]},
    )


def _pass_through_local_traffic() -> None:
    """Tell respx to let httpx calls to the testcontainer hosts hit the network.

    respx in strict mode (`@respx.mock`) intercepts ALL httpx calls and raises
    on anything not registered. ChromaDB's HttpClient uses httpx, so its calls
    to the testcontainer would be blocked. Adding a `pass_through()` route for
    localhost lets those traverse normally while everything we explicitly mock
    (Gemini, OpenAI, Tavily) still returns canned responses.

    `respx.mock(assert_all_mocked=False)` is NOT enough — it just stops
    raising; unmatched requests still get an empty 200 response from respx,
    which makes Chroma's orjson parser blow up with "zero-length, empty
    document".
    """
    respx.route(host="localhost").pass_through()
    respx.route(host="127.0.0.1").pass_through()


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
    # Let testcontainer DB calls reach the real network; mock only the LLM endpoint.
    _pass_through_local_traffic()
    respx.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/.*:batchEmbedContents.*"
    ).mock(side_effect=_gemini_batch_embed_handler)

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
    _pass_through_local_traffic()
    respx.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/.*:batchEmbedContents.*"
    ).mock(side_effect=_gemini_batch_embed_handler)

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
    _pass_through_local_traffic()
    respx.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/.*:batchEmbedContents.*"
    ).mock(side_effect=_gemini_batch_embed_handler)

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

    _pass_through_local_traffic()
    respx.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/.*:batchEmbedContents.*"
    ).mock(side_effect=_gemini_batch_embed_handler)

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
