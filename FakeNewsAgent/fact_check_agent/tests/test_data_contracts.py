"""Data contract tests — verify exact input/output values at every agent boundary.

These tests answer: "given this specific input, do I get exactly this output?"
They catch:
  - Field name typos in .get("key") calls
  - Wrong default values in fallback paths
  - Truncation / formatting bugs that corrupt LLM prompts
  - Off-by-one errors in confidence scaling (int vs float, 0-1 vs 0-100)
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fact_check_agent.src.models.schemas import (
    EntityRef,
    FactCheckInput,
    FactCheckOutput,
    MemoryQueryResponse,
    SimilarClaim,
)
from fact_check_agent.src.tools.live_search_tool import format_search_context
from fact_check_agent.src.tools.rag_tool import format_rag_context, retrieve_similar_claims

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_input(**overrides):
    defaults = dict(
        claim_id="clm_test001",
        claim_text="Scientists say climate change is a hoax.",
        entities=[
            EntityRef(entity_id="e1", name="IPCC", entity_type="organization", sentiment="negative")
        ],
        source_url="https://example.com/article/1",
        article_id="art_001",
        image_caption=None,
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    return FactCheckInput(**{**defaults, **overrides})


def make_llm_response(content: dict):
    choice = MagicMock()
    choice.message.content = json.dumps(content)
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── format_rag_context: exact prompt text ────────────────────────────────────


def test_rag_context_includes_confidence_as_percentage():
    """Confidence 0.95 must appear in the prompt as '95%', not '0.95' or '0.950'."""
    claims = [
        {
            "claim_id": "c1",
            "claim_text": "Prior claim",
            "verdict_label": "refuted",
            "verdict_confidence": 0.95,
            "distance": 0.10,
        }
    ]
    result = format_rag_context(claims)
    assert "95%" in result


def test_rag_context_no_prior_verdict_text():
    """Claims without a verdict must say 'no prior verdict' so the LLM knows."""
    claims = [
        {
            "claim_id": "c2",
            "claim_text": "Unverified claim",
            "verdict_label": None,
            "verdict_confidence": None,
            "distance": 0.30,
        }
    ]
    result = format_rag_context(claims)
    assert "no prior verdict" in result
    # Must NOT fabricate a verdict label
    assert "supported" not in result
    assert "refuted" not in result
    assert "misleading" not in result


def test_rag_context_contains_claim_text_verbatim():
    """Claim text must appear in the context unchanged — never truncated or paraphrased."""
    claim_text = "The unemployment rate fell to 3.4% in January 2024."
    claims = [
        {
            "claim_id": "c3",
            "claim_text": claim_text,
            "verdict_label": "supported",
            "verdict_confidence": 0.80,
            "distance": 0.05,
        }
    ]
    result = format_rag_context(claims)
    assert claim_text in result


def test_rag_context_multiple_claims_all_present():
    """All retrieved claims must appear in the context (none silently dropped)."""
    claims = [
        {
            "claim_id": "c1",
            "claim_text": "Claim Alpha",
            "verdict_label": "supported",
            "verdict_confidence": 0.90,
            "distance": 0.05,
        },
        {
            "claim_id": "c2",
            "claim_text": "Claim Beta",
            "verdict_label": "refuted",
            "verdict_confidence": 0.70,
            "distance": 0.20,
        },
        {
            "claim_id": "c3",
            "claim_text": "Claim Gamma",
            "verdict_label": None,
            "verdict_confidence": None,
            "distance": 0.35,
        },
    ]
    result = format_rag_context(claims)
    assert "Claim Alpha" in result
    assert "Claim Beta" in result
    assert "Claim Gamma" in result


# ── format_search_context: exact content and link extraction ──────────────────


def test_search_context_truncates_at_300_chars():
    """Content longer than 300 chars must be cut — otherwise the prompt balloons."""
    long_content = "A" * 500
    results = [{"url": "https://bbc.co.uk/1", "title": "BBC Story", "content": long_content}]
    context, _ = format_search_context(results)
    # The truncated content in the context should not contain 500 A's
    assert "A" * 400 not in context


def test_search_context_link_list_matches_urls():
    """Every URL in results must appear in the returned links list exactly once."""
    results = [
        {"url": "https://reuters.com/story/1", "title": "T1", "content": "Evidence 1"},
        {"url": "https://apnews.com/article/2", "title": "T2", "content": "Evidence 2"},
        {"url": "https://bbc.co.uk/news/3", "title": "T3", "content": "Evidence 3"},
    ]
    _, links = format_search_context(results)
    assert links == [
        "https://reuters.com/story/1",
        "https://apnews.com/article/2",
        "https://bbc.co.uk/news/3",
    ]


def test_search_context_title_appears_in_context():
    """Article titles must appear in the context block so the LLM can reference them."""
    results = [
        {
            "url": "https://reuters.com/1",
            "title": "Reuters: Study debunks claim",
            "content": "Short content.",
        }
    ]
    context, _ = format_search_context(results)
    assert "Reuters: Study debunks claim" in context


def test_search_context_empty_url_excluded_from_links_but_content_included():
    """Result with empty URL: content may appear in context but URL not in link list."""
    results = [
        {"url": "", "title": "No URL", "content": "Content A"},
        {"url": "https://example.com/valid", "title": "Valid", "content": "Content B"},
    ]
    context, links = format_search_context(results)
    assert "https://example.com/valid" in links
    assert "" not in links


# ── retrieve_similar_claims: ChromaDB dict → typed dicts ────────────────────


def test_retrieve_maps_confidence_correctly():
    """Confidence from ChromaDB metadata (float 0-1) must pass through unchanged."""
    memory = MagicMock()
    memory.search_similar_claims.return_value = {
        "ids": [["clm_abc"]],
        "documents": [["Prior claim text"]],
        "distances": [[0.15]],
        "metadatas": [[]],
    }
    memory.get_verdict_by_claim.return_value = {
        "metadatas": [{"label": "supported", "confidence": 0.88}]
    }
    results = retrieve_similar_claims("query claim", memory)
    assert results[0]["verdict_confidence"] == 0.88  # NOT 88 or 0.0


def test_retrieve_maps_distance_correctly():
    """ChromaDB distance must pass through unchanged — used by router for max_confidence."""
    memory = MagicMock()
    memory.search_similar_claims.return_value = {
        "ids": [["clm_xyz"]],
        "documents": [["Some text"]],
        "distances": [[0.42]],
        "metadatas": [[]],
    }
    memory.get_verdict_by_claim.return_value = {"metadatas": []}
    results = retrieve_similar_claims("query", memory)
    assert results[0]["distance"] == 0.42


def test_retrieve_claim_text_is_document_not_id():
    """claim_text must be the document text, not the ID string."""
    memory = MagicMock()
    memory.search_similar_claims.return_value = {
        "ids": [["clm_001"]],
        "documents": [["The actual claim sentence."]],
        "distances": [[0.1]],
        "metadatas": [[]],
    }
    memory.get_verdict_by_claim.return_value = {"metadatas": []}
    results = retrieve_similar_claims("query", memory)
    assert results[0]["claim_text"] == "The actual claim sentence."
    assert results[0]["claim_id"] == "clm_001"


# ── FactCheckOutput: field constraints ───────────────────────────────────────


def test_factcheckoutput_rejects_confidence_above_100():
    """Pydantic must reject confidence_score > 100 — catches scaling bugs (e.g. 0-1 → 0-100 done twice)."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FactCheckOutput(
            verdict_id="vrd_1",
            claim_id="clm_1",
            verdict="supported",
            confidence_score=101,  # invalid
            evidence_links=[],
            reasoning="test",
        )


def test_factcheckoutput_accepts_boundary_values():
    """confidence_score=0 and confidence_score=100 must both be valid."""
    for score in (0, 100):
        o = FactCheckOutput(
            verdict_id="vrd_1",
            claim_id="clm_1",
            verdict="misleading",
            confidence_score=score,
            evidence_links=[],
            reasoning="x",
        )
        assert o.confidence_score == score


# ── synthesize_verdict: LLM response parsing ─────────────────────────────────


def test_synthesize_verdict_strong_support_gives_supported():
    """All Di=1.0 (strong support) must produce a 'supported' verdict."""
    from fact_check_agent.src.config import settings
    from fact_check_agent.src.graph.nodes import synthesize_verdict

    # One prefetched chunk → one context claim, Di=1.0 → V=1.0 → supported
    llm_payload = {"degrees": [1.0], "reasoning": "Strong peer-reviewed evidence."}
    state = {
        "input": make_input(prefetched_chunks=["Peer-reviewed evidence chunk."]),
        "context_claims": [
            {
                "type": "factual",
                "source": "prefetched",
                "content": "Peer-reviewed evidence chunk.",
                "question": None,
            }
        ],
        "memory_results": MemoryQueryResponse(results=[], max_confidence=0.0),
    }

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_llm_response(llm_payload)
        updates = synthesize_verdict(state, settings)

    out = updates["output"]
    assert out.verdict == "supported"
    assert out.confidence_score > 0
    assert out.reasoning == "Strong peer-reviewed evidence."


def test_synthesize_verdict_strong_refutation_gives_refuted():
    """All Di=-1.0 (strong refutation) must produce a 'refuted' verdict."""
    from fact_check_agent.src.config import settings
    from fact_check_agent.src.graph.nodes import synthesize_verdict

    llm_payload = {"degrees": [-1.0], "reasoning": "Direct contradiction found."}
    state = {
        "input": make_input(prefetched_chunks=["Direct contradiction chunk."]),
        "context_claims": [
            {
                "type": "counter_factual",
                "source": "tavily",
                "content": "Contradicting evidence.",
                "question": None,
            }
        ],
        "memory_results": MemoryQueryResponse(results=[], max_confidence=0.0),
    }

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_llm_response(llm_payload)
        updates = synthesize_verdict(state, settings)

    out = updates["output"]
    assert out.verdict == "refuted"
    assert out.confidence_score > 0


def test_synthesize_verdict_fallback_on_missing_degrees():
    """If LLM returns no degrees, defaults to misleading with low confidence."""
    from fact_check_agent.src.config import settings
    from fact_check_agent.src.graph.nodes import synthesize_verdict

    state = {
        "input": make_input(),
        "context_claims": [],
        "memory_results": MemoryQueryResponse(results=[], max_confidence=0.0),
    }

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_llm_response(
            {"reasoning": "no evidence"}  # missing degrees
        )
        updates = synthesize_verdict(state, settings)

    out = updates["output"]
    assert out.verdict == "misleading"  # no evidence → misleading
    assert out.evidence_links == []


def test_synthesize_verdict_fallback_on_invalid_json():
    """LLM returning non-JSON must not crash — must return misleading (no evidence = neutral)."""
    from fact_check_agent.src.config import settings
    from fact_check_agent.src.graph.nodes import synthesize_verdict

    state = {
        "input": make_input(),
        "context_claims": [],
        "memory_results": MemoryQueryResponse(results=[], max_confidence=0.0),
    }

    choice = MagicMock()
    choice.message.content = "I cannot answer that."  # not JSON
    resp = MagicMock()
    resp.choices = [choice]

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = resp
        updates = synthesize_verdict(state, settings)

    out = updates["output"]
    assert out.verdict == "misleading"  # no evidence → misleading
    # With no evidence/context claims, formula returns confidence=50 (neutral)
    assert out.confidence_score == 50


# ── write_memory: delegates to record_verdict_outcome (T2) ───────────────────
# Field mapping (confidence scale, image_mismatch, evidence_summary format) is
# now tested in test_reflection_agent.py via record_verdict_outcome directly.
# Here we only verify that write_memory passes the correct arguments through.


def test_write_memory_calls_record_verdict_outcome_with_output():
    """write_memory must call record_verdict_outcome with the FactCheckOutput."""
    from fact_check_agent.src.graph.nodes import write_memory

    output = FactCheckOutput(
        verdict_id="vrd_test",
        claim_id="clm_test",
        verdict="refuted",
        confidence_score=72,
        evidence_links=["https://reuters.com/1"],
        reasoning="Evidence contradicts claim.",
        cross_modal_flag=False,
    )
    state = {"input": make_input(), "output": output}
    memory = MagicMock()

    with patch("fact_check_agent.src.graph.nodes.record_verdict_outcome") as mock_rvo:
        write_memory(state, memory)

    mock_rvo.assert_called_once()
    call_kwargs = mock_rvo.call_args.kwargs
    assert call_kwargs["output"] is output
    assert call_kwargs["memory"] is memory


def test_write_memory_passes_source_url():
    """write_memory must pass the correct source_url from state input."""
    from fact_check_agent.src.graph.nodes import write_memory

    output = FactCheckOutput(
        verdict_id="vrd_su",
        claim_id="clm_su",
        verdict="supported",
        confidence_score=80,
        evidence_links=[],
        reasoning="Supported.",
    )
    state = {"input": make_input(), "output": output}
    memory = MagicMock()

    with patch("fact_check_agent.src.graph.nodes.record_verdict_outcome") as mock_rvo:
        write_memory(state, memory)

    call_kwargs = mock_rvo.call_args.kwargs
    assert call_kwargs["source_url"] == "https://example.com/article/1"


def test_write_memory_skipped_when_no_output():
    """write_memory must not call record_verdict_outcome when output is missing."""
    from fact_check_agent.src.graph.nodes import write_memory

    state = {"input": make_input(), "output": None}
    memory = MagicMock()

    with patch("fact_check_agent.src.graph.nodes.record_verdict_outcome") as mock_rvo:
        write_memory(state, memory)

    mock_rvo.assert_not_called()
