"""Unit tests for prediction_agent.analyse_trend + the rule-based fallbacks.

These functions are pure math / pure conditional logic, so the tests run fast
and require no external services. They cover:
    - analyse_trend                — empty / rising / falling / volatile
    - _rule_based_prediction       — the deterministic fallback used when the
                                     LLM call fails. Asserts entity name + key
                                     directional phrasing appears in the message.
    - _rule_based_confidence       — exact threshold lookup; values are pinned
                                     in the source (0.35 / 0.80 / 0.60 / 0.45).
"""

import pytest
from agents.prediction_agent import (
    _rule_based_confidence,
    _rule_based_prediction,
    analyse_trend,
)


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


# ─────────────────────────────────────────────────────────────────────────────
# _rule_based_prediction
# Asserts entity name + directional phrasing land in the message. We don't
# assert the exact wording since the production strings are descriptive prose.
# ─────────────────────────────────────────────────────────────────────────────


def _trend(direction: str, current: float, delta: float, slope: float = 0.0,
           volatility: float = 0.0, r_squared: float = 0.5) -> dict:
    return {
        "direction": direction,
        "current": current,
        "delta": delta,
        "slope": slope,
        "volatility": volatility,
        "r_squared": r_squared,
    }


def test_rule_based_prediction_rising_message_includes_entity_and_direction():
    msg = _rule_based_prediction("Tesla", _trend("RISING", 0.7, 0.2, slope=0.05))
    assert "Tesla" in msg
    assert "rising" in msg.lower()


def test_rule_based_prediction_falling_message_includes_entity_and_direction():
    msg = _rule_based_prediction("Tesla", _trend("FALLING", 0.3, -0.2, slope=-0.05))
    assert "Tesla" in msg
    assert "declining" in msg.lower() or "fall" in msg.lower()


def test_rule_based_prediction_volatile_message_mentions_volatility():
    msg = _rule_based_prediction("Tesla", _trend("VOLATILE", 0.5, 0.0, volatility=0.20))
    assert "Tesla" in msg
    assert "volatility" in msg.lower() or "unpredictable" in msg.lower()


def test_rule_based_prediction_stable_message_mentions_stability():
    msg = _rule_based_prediction("Tesla", _trend("STABLE", 0.6, 0.01))
    assert "Tesla" in msg
    assert "stable" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# _rule_based_confidence
# Source thresholds (prediction_agent.py:204):
#   VOLATILE              → 0.35
#   r_squared > 0.85      → 0.80
#   r_squared > 0.50      → 0.60
#   else                  → 0.45
# Note: comparisons are STRICT (>), so r²=0.50 falls into the 0.45 bucket.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("direction,r_squared,expected", [
    ("VOLATILE", 0.99, 0.35),    # VOLATILE always wins regardless of r²
    ("RISING",   0.90, 0.80),    # high r² (>0.85)
    ("RISING",   0.51, 0.60),    # mid r² (>0.50, ≤0.85)
    ("STABLE",   0.50, 0.45),    # boundary: 0.50 not greater than 0.50 → fallback
    ("FALLING",  0.10, 0.45),    # low r² → fallback
])
def test_rule_based_confidence_threshold_lookup(direction, r_squared, expected):
    """Exact-value table lookup — assertions are equality, not range."""
    conf = _rule_based_confidence({"direction": direction, "r_squared": r_squared})
    assert conf == expected, f"{direction}/r²={r_squared}: expected {expected}, got {conf}"
