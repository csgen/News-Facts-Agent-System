"""Unit and integration tests for the reranker module.

Covers:
  - reciprocal_rank_fusion: merge, deduplication, ordering
  - rerank_candidates: RRF merge, single list, empty input
  - query_memory node: USE_GRAPH_RAG flag, verdict revision
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from fact_check_agent.src.tools.reranker import (
    reciprocal_rank_fusion,
    rerank_candidates,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_claim(
    claim_id,
    claim_text="test claim",
    verdict_label="supported",
    verdict_confidence=0.8,
    distance=0.2,
):
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "verdict_label": verdict_label,
        "verdict_confidence": verdict_confidence,
        "distance": distance,
        "verified_at": None,
    }


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────


def test_rrf_single_list_preserves_order():
    items = [make_claim(f"c{i}") for i in range(3)]
    result = reciprocal_rank_fusion([items])
    assert [r["claim_id"] for r in result] == ["c0", "c1", "c2"]


def test_rrf_deduplicates_across_lists():
    list_a = [make_claim("c1"), make_claim("c2")]
    list_b = [make_claim("c1"), make_claim("c3")]
    result = reciprocal_rank_fusion([list_a, list_b])
    ids = [r["claim_id"] for r in result]
    assert len(ids) == len(set(ids)), "Duplicate claim_ids in RRF output"
    assert set(ids) == {"c1", "c2", "c3"}


def test_rrf_boosts_item_ranked_high_in_both_lists():
    # c1 is #1 in both lists → highest RRF score
    list_a = [make_claim("c1"), make_claim("c2"), make_claim("c3")]
    list_b = [make_claim("c1"), make_claim("c4"), make_claim("c5")]
    result = reciprocal_rank_fusion([list_a, list_b])
    assert result[0]["claim_id"] == "c1"


def test_rrf_attaches_rrf_score():
    items = [make_claim("c1"), make_claim("c2")]
    result = reciprocal_rank_fusion([items])
    for item in result:
        assert "rrf_score" in item
        assert item["rrf_score"] > 0


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_one_empty_one_populated():
    items = [make_claim("c1"), make_claim("c2")]
    result = reciprocal_rank_fusion([items, []])
    assert [r["claim_id"] for r in result] == ["c1", "c2"]


# ── rerank_candidates ─────────────────────────────────────────────────────────


def test_rerank_candidates_rrf_merge():
    vector = [make_claim("c1"), make_claim("c2"), make_claim("c3")]
    graph = [make_claim("c4"), make_claim("c1")]  # c1 appears in both
    result = rerank_candidates(
        query="test query",
        vector_results=vector,
        graph_results=graph,
        top_k=3,
    )
    ids = [r["claim_id"] for r in result]
    assert len(result) == 3
    assert "c1" in ids  # boosted by appearing in both lists
    assert len(ids) == len(set(ids))


def test_rerank_candidates_top_k_respected():
    vector = [make_claim(f"c{i}") for i in range(10)]
    result = rerank_candidates(
        query="q",
        vector_results=vector,
        graph_results=[],
        top_k=4,
    )
    assert len(result) == 4


def test_rerank_candidates_empty_inputs():
    result = rerank_candidates(
        query="q",
        vector_results=[],
        graph_results=[],
        top_k=5,
    )
    assert result == []


def test_rerank_candidates_graph_only():
    graph = [make_claim("g1"), make_claim("g2")]
    result = rerank_candidates(
        query="q",
        vector_results=[],
        graph_results=graph,
        top_k=5,
    )
    assert [r["claim_id"] for r in result] == ["g1", "g2"]


# ── query_memory node with GraphRAG flags ─────────────────────────────────────


def make_memory_mock_with_similar(claims=None):
    memory = MagicMock()
    claims = claims or []
    ids = [c["claim_id"] for c in claims]
    docs = [c["claim_text"] for c in claims]
    distances = [c.get("distance", 0.2) for c in claims]

    memory.search_similar_claims.return_value = {
        "ids": [ids],
        "documents": [docs],
        "distances": [distances],
        "metadatas": [[{} for _ in claims]],
    }
    memory.get_entity_context.return_value = []
    memory.get_verdict_by_claim.return_value = {"ids": [], "metadatas": []}
    memory.get_entity_ids_for_claims.return_value = []
    memory.get_graph_claims_for_entities.return_value = []
    memory.add_verdict.return_value = None
    return memory


def _make_settings(use_graph_rag=False, top_k=5):
    s = MagicMock()
    s.use_graph_rag = use_graph_rag
    s.reranker_top_k = top_k
    s.offline_mode = False
    s.dry_run = False
    return s


def make_fci(claim_id="clm_001", claim_text="vaccines cause autism"):
    from fact_check_agent.src.models.schemas import FactCheckInput

    return FactCheckInput(
        claim_id=claim_id,
        claim_text=claim_text,
        entities=[],
        source_url="https://example.com",
        article_id="art_001",
        image_caption=None,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def test_query_memory_graph_rag_disabled_no_graph_calls():
    """With USE_GRAPH_RAG=false, entity expansion methods are never called."""
    from fact_check_agent.src.graph.nodes import query_memory

    memory = make_memory_mock_with_similar()
    settings = _make_settings(use_graph_rag=False)

    state = {"input": make_fci()}
    with patch(
        "fact_check_agent.src.agents.reflection_agent.query_source_credibility",
        return_value={"sample_count": 0},
    ):
        query_memory(state, memory, settings)

    memory.get_entity_ids_for_claims.assert_not_called()
    memory.get_graph_claims_for_entities.assert_not_called()


def test_query_memory_graph_rag_enabled_calls_entity_expansion():
    """With USE_GRAPH_RAG=true and vector results present, entity expansion runs."""
    from fact_check_agent.src.graph.nodes import query_memory

    similar = [make_claim("c1", "vaccines cause autism")]
    memory = make_memory_mock_with_similar(similar)
    memory.get_entity_ids_for_claims.return_value = [{"entity_id": "ent_1", "name": "WHO"}]
    memory.get_graph_claims_for_entities.return_value = [
        make_claim("c2", "vaccines are safe", distance=0.0)
    ]

    settings = _make_settings(use_graph_rag=True)

    state = {"input": make_fci()}
    with patch(
        "fact_check_agent.src.agents.reflection_agent.query_source_credibility",
        return_value={"sample_count": 0},
    ):
        result = query_memory(state, memory, settings)

    memory.get_entity_ids_for_claims.assert_called_once_with(["c1"])
    memory.get_graph_claims_for_entities.assert_called_once_with(["ent_1"])
    ids = [r.claim_id for r in result["memory_results"].results]
    assert "c1" in ids
    assert "c2" in ids


def test_query_memory_graph_rag_no_vector_results_skips_expansion():
    """With USE_GRAPH_RAG=true but no vector results, graph expansion is skipped."""
    from fact_check_agent.src.graph.nodes import query_memory

    memory = make_memory_mock_with_similar([])
    settings = _make_settings(use_graph_rag=True)

    state = {"input": make_fci()}
    with patch(
        "fact_check_agent.src.agents.reflection_agent.query_source_credibility",
        return_value={"sample_count": 0},
    ):
        query_memory(state, memory, settings)

    memory.get_entity_ids_for_claims.assert_not_called()


# ── Verdict revision ──────────────────────────────────────────────────────────


def _ensure_memory_agent_env():
    import os

    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_PASSWORD", "fakenews123")
    os.environ.setdefault("OPENAI_API_KEY", "unused")


def _stub_missing_modules():
    import sys
    from unittest.mock import MagicMock as _M

    for mod in [
        "google.genai",
        "rapidfuzz",
        "rapidfuzz.fuzz",
        "rapidfuzz.process",
        "neo4j",
        "chromadb",
    ]:
        sys.modules.setdefault(mod, _M())
    google_mod = sys.modules.get("google", _M())
    if not hasattr(google_mod, "genai"):
        google_mod.genai = sys.modules["google.genai"]
        sys.modules["google"] = google_mod


def _make_verdict(verdict_id="v_new", claim_id="clm_001", label="refuted"):
    _ensure_memory_agent_env()
    _stub_missing_modules()
    import importlib

    Verdict = importlib.import_module("src.models.verdict").Verdict
    return Verdict(
        verdict_id=verdict_id,
        claim_id=claim_id,
        label=label,
        confidence=0.9,
        evidence_summary="evidence",
        image_mismatch=False,
        verified_at=datetime.now(timezone.utc),
    )


def _get_add_verdict_fn():
    _ensure_memory_agent_env()
    _stub_missing_modules()
    import importlib

    return importlib.import_module("src.memory.agent").MemoryAgent.add_verdict


def test_add_verdict_supersedes_existing():
    """add_verdict should supersede any existing active verdict for the same claim."""
    add_verdict = _get_add_verdict_fn()
    verdict = _make_verdict("new_verdict_id", "clm_001", "refuted")

    ma = MagicMock()
    ma._vector.get_verdict_by_claim.return_value = {
        "ids": ["old_verdict_id"],
        "metadatas": [{"label": "supported", "status": "active"}],
    }
    ma._vector.get_claims_by_ids.return_value = {"documents": ["test claim"]}
    ma._embeddings.embed.return_value = [0.1] * 768

    add_verdict(ma, verdict)

    ma._vector.supersede_verdict.assert_called_once_with("old_verdict_id", "new_verdict_id")
    ma._graph.supersede_verdict.assert_called_once_with("old_verdict_id", "new_verdict_id")


def test_add_verdict_no_supersede_when_no_existing():
    """add_verdict should NOT call supersede when no active verdict exists."""
    add_verdict = _get_add_verdict_fn()
    verdict = _make_verdict("v1", "clm_002", "supported")

    ma = MagicMock()
    ma._vector.get_verdict_by_claim.return_value = {"ids": []}
    ma._vector.get_claims_by_ids.return_value = {"documents": []}
    ma._embeddings.embed.return_value = [0.1] * 768

    add_verdict(ma, verdict)

    ma._vector.supersede_verdict.assert_not_called()
    ma._graph.supersede_verdict.assert_not_called()
