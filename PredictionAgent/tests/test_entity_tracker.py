"""Unit tests for entity_tracker.compute_credibility_score.

The scoring function is pure math (no DB / LLM calls), so these tests run
fast and require no external services. They cover:
    - Empty / None input → neutral prior (0.5)
    - All-supported claims → high credibility, but capped by Bayesian volume shrinkage
    - All-refuted claims → low credibility
"""

from datetime import datetime, timedelta, timezone

import pytest
from agents.entity_tracker import compute_credibility_score


def _claim(label: str, confidence: float = 0.9, days_ago: float = 0.0) -> dict:
    """Build a minimal claim dict matching the shape compute_credibility_score expects."""
    return {
        "verdict_label": label,
        "verdict_confidence": confidence,
        "verified_at": (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(),
    }


def test_compute_credibility_empty_claims_returns_neutral_prior():
    """No evidence → score sits at the 0.5 neutral prior."""
    assert compute_credibility_score([]) == 0.5


def test_compute_credibility_all_supported_high_confidence_pulls_high():
    """10 supported claims with high confidence → score well above the prior.

    Volume shrinkage at n=10 is ~0.964, so the score should land in [0.9, 1.0].
    """
    claims = [_claim("supported", confidence=0.95) for _ in range(10)]
    score = compute_credibility_score(claims)
    assert 0.9 <= score <= 1.0, f"expected [0.9, 1.0], got {score:.4f}"


def test_compute_credibility_all_refuted_pulls_low():
    """10 refuted claims with high confidence → score well below the prior."""
    claims = [_claim("refuted", confidence=0.95) for _ in range(10)]
    score = compute_credibility_score(claims)
    assert 0.0 <= score <= 0.1, f"expected [0.0, 0.1], got {score:.4f}"


def test_compute_credibility_single_claim_volume_shrinkage_pulls_toward_prior():
    """One supported claim should NOT score 1.0 — Bayesian shrinkage pulls toward 0.5.

    Math: evidence=1.0, volume_factor = 1 - exp(-1/3) ≈ 0.283
          → 0.5*(1-0.283) + 1.0*0.283 ≈ 0.642
    """
    score = compute_credibility_score([_claim("supported", confidence=1.0)])
    assert 0.6 <= score <= 0.7, f"expected [0.6, 0.7] (shrinkage active), got {score:.4f}"


@pytest.mark.parametrize(
    "label,expected_range",
    [
        # 5 claims, conf=0.9, days_ago=0 → volume_factor ≈ 0.811, evidence = LABEL_SCORE
        # → score = 0.5*0.189 + LABEL_SCORE * 0.811
        ("supported", (0.85, 1.0)),  # 1.0   → ~0.905
        ("misleading", (0.30, 0.45)),  # 0.35  → ~0.378  (label_score below prior)
        ("refuted", (0.0, 0.15)),  # 0.0   → ~0.094
    ],
)
def test_compute_credibility_label_score_mapping(label, expected_range):
    """Each verdict label maps to a distinct credibility range."""
    claims = [_claim(label, confidence=0.9) for _ in range(5)]
    score = compute_credibility_score(claims)
    lo, hi = expected_range
    assert lo <= score <= hi, f"{label}: expected [{lo}, {hi}], got {score:.4f}"
