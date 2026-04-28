"""Unit tests for prediction_agent.analyse_trend.

The trend analyser is pure math (linear regression + std deviation), so these
tests run fast and require no external services. They cover:
    - Empty / single-point input → STABLE
    - Monotonically rising series  → RISING
    - Monotonically falling series → FALLING
    - High-variance series         → VOLATILE
"""

from agents.prediction_agent import analyse_trend


def _snapshots(scores: list[float]) -> list[dict]:
    """Build snapshot dicts in the shape analyse_trend expects (oldest → newest)."""
    return [{"credibility_score": s} for s in scores]


def test_analyse_trend_empty_input_returns_stable_with_default_score():
    """No snapshots → STABLE direction, neutral 0.5 fallback."""
    out = analyse_trend([])
    assert out["direction"] == "STABLE"
    assert out["slope"] == 0.0
    assert out["current"] == 0.5


def test_analyse_trend_rising_series_classified_as_rising():
    """Monotonically increasing scores → RISING direction with positive slope."""
    out = analyse_trend(_snapshots([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]))
    assert out["direction"] == "RISING"
    assert out["slope"] > 0.015
    assert out["delta"] > 0


def test_analyse_trend_falling_series_classified_as_falling():
    """Monotonically decreasing scores → FALLING direction with negative slope."""
    out = analyse_trend(_snapshots([0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]))
    assert out["direction"] == "FALLING"
    assert out["slope"] < -0.015
    assert out["delta"] < 0


def test_analyse_trend_high_variance_classified_as_volatile():
    """Big swings around the same mean → VOLATILE direction (volatility > 0.12)."""
    out = analyse_trend(_snapshots([0.1, 0.9, 0.2, 0.85, 0.15, 0.95, 0.1]))
    assert out["direction"] == "VOLATILE"
    assert out["volatility"] > 0.12
