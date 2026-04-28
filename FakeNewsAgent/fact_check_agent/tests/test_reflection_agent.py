"""Tests for the Reflection Agent — no Neo4j or OpenAI API keys required.

Covers:
  - source_id_from_url: URL → source_id derivation
  - credibility_signal: verdict + confidence → [0,1] observation
  - query_source_credibility: Neo4j exact lookup with base_credibility fallback
  - record_verdict_outcome: all 5 write-backs delegated correctly
"""

from unittest.mock import MagicMock, call, patch

from fact_check_agent.src.agents.reflection_agent import (
    _ALPHA,
    _DEFAULT_CREDIBILITY,
    credibility_signal,
    query_source_credibility,
    record_verdict_outcome,
    source_id_from_url,
)

# ── source_id_from_url ────────────────────────────────────────────────────────


def test_source_id_from_standard_url():
    assert source_id_from_url("https://bbc.co.uk/news/1") == "src_bbc_co_uk"


def test_source_id_from_url_strips_path():
    sid = source_id_from_url("https://reuters.com/article/long/path?q=1")
    assert sid == "src_reuters_com"
    assert "/article" not in sid


def test_source_id_from_url_replaces_dots():
    sid = source_id_from_url("https://apnews.com/article/1")
    assert "." not in sid.replace("src_", "")


def test_source_id_same_for_same_domain():
    a = source_id_from_url("https://bbc.co.uk/news/1")
    b = source_id_from_url("https://bbc.co.uk/news/2")
    assert a == b


# ── credibility_signal ────────────────────────────────────────────────────────


def test_credibility_signal_supported_high_confidence():
    assert credibility_signal("supported", 90) == 0.90


def test_credibility_signal_supported_low_confidence():
    assert credibility_signal("supported", 40) == 0.40


def test_credibility_signal_refuted_high_confidence():
    assert abs(credibility_signal("refuted", 90) - 0.10) < 1e-9


def test_credibility_signal_refuted_low_confidence():
    assert abs(credibility_signal("refuted", 30) - 0.70) < 1e-9


def test_credibility_signal_misleading_is_neutral():
    assert credibility_signal("misleading", 0) == 0.5
    assert credibility_signal("misleading", 100) == 0.5
    assert credibility_signal("misleading", 55) == 0.5


def test_credibility_signal_boundary_values():
    assert 0.0 <= credibility_signal("supported", 0) <= 1.0
    assert 0.0 <= credibility_signal("supported", 100) <= 1.0
    assert 0.0 <= credibility_signal("refuted", 0) <= 1.0
    assert 0.0 <= credibility_signal("refuted", 100) <= 1.0


# ── query_source_credibility ──────────────────────────────────────────────────


def test_query_returns_neo4j_value():
    """When a Neo4j record exists, return it as credibility_mean."""
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = 0.78

    result = query_source_credibility("any claim", "https://bbc.co.uk", memory, topic="health")

    assert abs(result["credibility_mean"] - 0.78) < 1e-4
    assert result["sample_count"] == 1


def test_query_falls_back_to_base_credibility():
    """When no dynamic record exists, fall back to Source.base_credibility."""
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = None
    memory.get_base_credibility.return_value = 0.72

    result = query_source_credibility("any claim", "https://bbc.co.uk", memory, topic="health")

    assert abs(result["credibility_mean"] - 0.72) < 1e-4
    assert result["sample_count"] == 1


def test_query_returns_none_when_no_record_and_no_base():
    """Unknown source with no dynamic record → credibility_mean is None."""
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = None
    memory.get_base_credibility.return_value = None

    result = query_source_credibility("any claim", "https://unknown.xyz", memory, topic="health")

    assert result["credibility_mean"] is None
    assert result["sample_count"] == 0


def test_query_returns_none_when_topic_empty():
    """No topic → skip lookup entirely."""
    memory = MagicMock()

    result = query_source_credibility("any claim", "https://bbc.co.uk", memory, topic="")

    assert result["credibility_mean"] is None
    assert result["sample_count"] == 0
    memory.get_source_topic_credibility.assert_not_called()


def test_query_uses_correct_source_id():
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = 0.7

    query_source_credibility("any claim", "https://bbc.co.uk/news/1", memory, topic="health")

    memory.get_source_topic_credibility.assert_called_once_with("src_bbc_co_uk", "health")


def test_query_failure_returns_none_stats():
    memory = MagicMock()
    memory.get_source_topic_credibility.side_effect = Exception("Neo4j unavailable")

    result = query_source_credibility("any claim", "https://test.com", memory, topic="health")

    assert result["credibility_mean"] is None
    assert result["sample_count"] == 0


# ── record_verdict_outcome ────────────────────────────────────────────────────


def _make_output(verdict="supported", confidence=80):
    output = MagicMock()
    output.verdict_id = "vrd_test123"
    output.claim_id = "clm_test456"
    output.verdict = verdict
    output.confidence_score = confidence
    output.reasoning = "Test reasoning."
    output.evidence_links = []
    output.cross_modal_flag = False
    return output


def test_record_calls_update_claim_status():
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = 0.65

    with patch("fact_check_agent.src.agents.reflection_agent.Verdict"):
        record_verdict_outcome(_make_output(), "claim", "https://x.com", "health", memory)

    memory.update_claim_status.assert_called_once_with("clm_test456", "verified")


def test_record_calls_add_verdict():
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = 0.65

    with patch("fact_check_agent.src.agents.reflection_agent.Verdict"):
        record_verdict_outcome(_make_output(), "claim", "https://x.com", "health", memory)

    memory.add_verdict.assert_called_once()


def test_record_alpha_update_supported():
    """supported @80% → signal=0.80; new_c = old_c + 0.05*(0.80-0.5)."""
    memory = MagicMock()
    old_c = 0.65
    memory.get_source_topic_credibility.return_value = old_c
    expected = round(old_c + _ALPHA * (0.80 - 0.5), 6)

    with patch("fact_check_agent.src.agents.reflection_agent.Verdict"):
        record_verdict_outcome(
            _make_output("supported", 80), "claim", "https://x.com", "health", memory
        )

    call_args = memory.upsert_source_topic_credibility.call_args
    assert call_args is not None
    _, _, written_c = call_args.args
    assert abs(written_c - expected) < 1e-6


def test_record_alpha_update_uses_base_when_no_dynamic():
    """First verdict for this source/topic: initialise from base_credibility."""
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = None
    memory.get_base_credibility.return_value = 0.70

    with patch("fact_check_agent.src.agents.reflection_agent.Verdict"):
        record_verdict_outcome(
            _make_output("supported", 80), "claim", "https://x.com", "health", memory
        )

    memory.upsert_source_topic_credibility.assert_called_once()


def test_record_alpha_update_uses_default_when_no_source():
    """Unknown source + no dynamic record → initialise from _DEFAULT_CREDIBILITY."""
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = None
    memory.get_base_credibility.return_value = None

    with patch("fact_check_agent.src.agents.reflection_agent.Verdict"):
        record_verdict_outcome(
            _make_output("supported", 80), "claim", "https://unknown.xyz", "health", memory
        )

    call_args = memory.upsert_source_topic_credibility.call_args
    _, _, written_c = call_args.args
    expected = round(_DEFAULT_CREDIBILITY + _ALPHA * (0.80 - 0.5), 6)
    assert abs(written_c - expected) < 1e-6


def test_record_memory_failure_does_not_raise():
    """Any individual write failure must not propagate."""
    memory = MagicMock()
    memory.update_claim_status.side_effect = Exception("DB down")
    memory.add_verdict.side_effect = Exception("DB down")
    memory.get_source_topic_credibility.side_effect = Exception("DB down")

    with patch("fact_check_agent.src.agents.reflection_agent.Verdict"):
        record_verdict_outcome(_make_output(), "claim", "https://x.com", "health", memory)


def test_record_unknown_topic_when_empty():
    """Empty topic_text falls back to 'unknown' bucket."""
    memory = MagicMock()
    memory.get_source_topic_credibility.return_value = 0.65

    with patch("fact_check_agent.src.agents.reflection_agent.Verdict"):
        record_verdict_outcome(_make_output(), "claim", "https://x.com", "", memory)

    call_args = memory.upsert_source_topic_credibility.call_args
    _, topic, _ = call_args.args
    assert topic == "unknown"
