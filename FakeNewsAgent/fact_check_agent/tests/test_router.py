"""Tests for the graph routing functions — no API keys required."""
from fact_check_agent.src.graph.router import debate_check
from fact_check_agent.src.models.schemas import FactCheckOutput


# ── debate_check ──────────────────────────────────────────────────────────────

def test_debate_check_returns_skip_when_use_debate_false():
    """Baseline: use_debate=False → debate_check must always return 'skip'."""
    from unittest.mock import patch
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = False
        assert debate_check({}) == "skip"


def test_debate_check_returns_skip_regardless_of_confidence_when_disabled():
    """Even low-confidence outputs should route to skip when debate is disabled."""
    from unittest.mock import MagicMock, patch

    output = MagicMock(spec=FactCheckOutput)
    output.confidence_score = 50
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = False
        assert debate_check({"output": output}) == "skip"


def test_debate_check_returns_debate_when_enabled_and_low_confidence():
    """When use_debate=True and confidence < threshold → should route to 'debate'."""
    from unittest.mock import MagicMock, patch

    output = MagicMock(spec=FactCheckOutput)
    output.confidence_score = 45
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = True
        ms.debate_confidence_threshold = 70
        assert debate_check({"output": output}) == "debate"


def test_debate_check_returns_skip_when_enabled_and_high_confidence():
    """When use_debate=True but confidence >= threshold → skip debate."""
    from unittest.mock import MagicMock, patch

    output = MagicMock(spec=FactCheckOutput)
    output.confidence_score = 85
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = True
        ms.debate_confidence_threshold = 70
        assert debate_check({"output": output}) == "skip"


def test_debate_check_returns_skip_when_no_output():
    """Missing output key defaults to skip (no output = nothing to debate)."""
    from unittest.mock import patch
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = True
        ms.debate_confidence_threshold = 70
        assert debate_check({}) == "skip"
