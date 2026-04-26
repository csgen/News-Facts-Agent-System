"""
Prediction Agent
Task 3 — Full-Stack & Evaluation Engineer

WHAT THIS DOES:
- Reads the last N CredibilitySnapshots for an entity from Neo4j
- Runs statistical trend analysis (slope, volatility, R²)
- Classifies the trend: RISING / FALLING / STABLE / VOLATILE
- Calls the LLM (OpenAI) with the trend data to generate a human-readable prediction
- Writes a Prediction node back to Neo4j via MemoryAgent

WHEN IT RUNS:
- Called automatically at the end of run_entity_tracker() if enough snapshots exist
- Requires at least MIN_SNAPSHOTS snapshots to generate a prediction

HOW IT CONNECTS:
- Reads from:  Neo4j — CredibilitySnapshot nodes  (via MemoryAgent)
- Writes to:   Neo4j — Prediction node            (via MemoryAgent)
- External:    OpenAI API for prediction text generation
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import settings
from id_utils import make_id
from models.credibility import Prediction

from agents.memory_agent import MemoryAgent, get_memory

logger = logging.getLogger(__name__)

MIN_SNAPSHOTS = 3
PREDICTION_HORIZON_DAYS = 30


# ─────────────────────────────────────────────
# TREND ANALYSIS
# ─────────────────────────────────────────────

def analyse_trend(snapshots: list[dict]) -> dict:
    """
    Given snapshots sorted oldest→newest, compute:
      slope       : credibility change per snapshot (linear regression)
      r_squared   : how well a line fits (0=noisy, 1=perfect)
      volatility  : std deviation of scores
      direction   : RISING / FALLING / STABLE / VOLATILE
      current     : most recent score
      start       : oldest score
      delta       : total change
    """
    scores = [float(s["credibility_score"]) for s in snapshots]
    n = len(scores)

    if n < 2:
        return {
            "slope": 0.0, "r_squared": 0.0, "volatility": 0.0,
            "direction": "STABLE",
            "current": scores[-1] if scores else 0.5,
            "start":   scores[0]  if scores else 0.5,
            "delta":   0.0,
        }

    x_mean = (n - 1) / 2.0
    y_mean = sum(scores) / n

    ss_xy = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((scores[i] - y_mean) ** 2 for i in range(n))

    slope     = ss_xy / ss_xx if ss_xx > 0 else 0.0
    r_squared = (ss_xy ** 2 / (ss_xx * ss_yy)) if (ss_xx > 0 and ss_yy > 0) else 0.0

    variance   = sum((s - y_mean) ** 2 for s in scores) / n
    volatility = round(math.sqrt(variance), 4)
    delta      = scores[-1] - scores[0]

    if volatility > 0.12:
        direction = "VOLATILE"
    elif slope > 0.015:
        direction = "RISING"
    elif slope < -0.015:
        direction = "FALLING"
    else:
        direction = "STABLE"

    return {
        "slope":      round(slope, 5),
        "r_squared":  round(r_squared, 4),
        "volatility": volatility,
        "direction":  direction,
        "current":    round(scores[-1], 4),
        "start":      round(scores[0],  4),
        "delta":      round(delta, 4),
    }


# ─────────────────────────────────────────────
# LLM PREDICTION GENERATOR
# ─────────────────────────────────────────────

def generate_prediction_text(entity_name: str, trend: dict,
                              snapshots: list[dict]) -> tuple[str, float]:
    """
    Call OpenAI with the trend data and get a prediction.
    Falls back to rule-based text if the API call fails.
    Returns (prediction_text, confidence_score 0-1).
    """
    scores_str = ", ".join(
        f"{float(s['credibility_score']):.2f}" for s in snapshots[-8:]
    )

    prompt = f"""You are a credibility analyst for a real-time fact-checking system.

Entity: {entity_name}
Recent credibility scores (oldest to newest): [{scores_str}]
Trend direction: {trend['direction']}
Slope per snapshot: {trend['slope']:+.4f}
Total drift: {trend['delta']:+.4f}  ({trend['start']:.2f} → {trend['current']:.2f})
Volatility (std dev): {trend['volatility']:.4f}
R² (trend reliability): {trend['r_squared']:.4f}

Write ONE concise prediction (2-3 sentences) about this entity's credibility \
over the next {PREDICTION_HORIZON_DAYS} days. Be specific about direction and \
magnitude. End with a confidence level: LOW, MEDIUM, or HIGH.

Respond in this exact format:
PREDICTION: <your prediction text>
CONFIDENCE: <LOW|MEDIUM|HIGH>"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        raw = response.choices[0].message.content.strip()

        prediction_text = ""
        confidence = 0.6
        for line in raw.splitlines():
            if line.startswith("PREDICTION:"):
                prediction_text = line.replace("PREDICTION:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                level = line.replace("CONFIDENCE:", "").strip().upper()
                confidence = {"LOW": 0.4, "MEDIUM": 0.65, "HIGH": 0.85}.get(level, 0.6)

        if not prediction_text:
            prediction_text = raw[:300]

        return prediction_text, confidence

    except Exception as e:
        logger.warning("LLM prediction failed, using rule-based fallback: %s", e)
        return _rule_based_prediction(entity_name, trend), _rule_based_confidence(trend)


def _rule_based_prediction(entity_name: str, trend: dict) -> str:
    d = trend["direction"]
    delta   = trend["delta"]
    current = trend["current"]
    horizon = PREDICTION_HORIZON_DAYS

    if d == "RISING":
        projected = min(current + abs(delta), 1.0)
        return (f"{entity_name}'s credibility has been rising steadily "
                f"(+{abs(delta):.0%} total drift, slope {trend['slope']:+.4f}/snapshot). "
                f"If this trend holds, credibility may reach {projected:.0%} "
                f"within {horizon} days.")
    elif d == "FALLING":
        projected = max(current - abs(delta), 0.0)
        return (f"{entity_name}'s credibility is declining "
                f"({delta:.0%} drift, slope {trend['slope']:+.4f}/snapshot). "
                f"Without a course correction, scores could fall to {projected:.0%} "
                f"within {horizon} days.")
    elif d == "VOLATILE":
        return (f"{entity_name} shows high credibility volatility "
                f"(std dev {trend['volatility']:.3f}). Scores are unpredictable — "
                f"expect continued fluctuation rather than a clear directional trend.")
    else:
        return (f"{entity_name}'s credibility has been stable at approximately "
                f"{current:.0%} (slope ≈ {trend['slope']:+.4f}/snapshot). "
                f"No significant change is expected in the next {horizon} days "
                f"unless major new claims emerge.")


def _rule_based_confidence(trend: dict) -> float:
    if trend["direction"] == "VOLATILE":
        return 0.35
    r2 = trend["r_squared"]
    if r2 > 0.85:
        return 0.80
    elif r2 > 0.50:
        return 0.60
    else:
        return 0.45


# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────

def run_prediction_agent(entity_name: str,
                         memory: Optional[MemoryAgent] = None) -> Optional[Prediction]:
    """
    Generate and store a credibility prediction for the given entity.

    1. Look up entity in Neo4j
    2. Fetch recent CredibilitySnapshots (need >= MIN_SNAPSHOTS)
    3. Run trend analysis (slope, R², volatility, direction)
    4. Generate prediction text via LLM (or rule-based fallback)
    5. Write Prediction node to Neo4j
    """
    if memory is None:
        memory = get_memory()

    print(f"\n[prediction_agent] Running for: '{entity_name}'")

    entity_dict = memory.get_entity_by_name(entity_name)
    if entity_dict is None:
        print(f"[prediction_agent] Entity '{entity_name}' not found — skipping")
        return None

    entity_id = entity_dict["entity_id"]

    snapshots = memory.get_entity_snapshots(entity_id, limit=20)
    print(f"[prediction_agent] Found {len(snapshots)} snapshots")

    if len(snapshots) < MIN_SNAPSHOTS:
        print(f"[prediction_agent] Need {MIN_SNAPSHOTS}, have {len(snapshots)} — skipping")
        return None

    # Sort oldest → newest for regression
    def _snap_dt(s):
        t = s.get("snapshot_at")
        if hasattr(t, "to_native"):
            return t.to_native()
        try:
            return datetime.fromisoformat(str(t))
        except Exception:
            return datetime.min

    snapshots_sorted = sorted(snapshots, key=_snap_dt)

    trend = analyse_trend(snapshots_sorted)
    print(f"[prediction_agent] Trend={trend['direction']}  "
          f"slope={trend['slope']:+.5f}  R²={trend['r_squared']:.3f}  "
          f"vol={trend['volatility']:.4f}  "
          f"{trend['start']:.2f}→{trend['current']:.2f}")

    prediction_text, confidence = generate_prediction_text(
        entity_name, trend, snapshots_sorted
    )
    print(f"[prediction_agent] confidence={confidence:.0%}  "
          f"text={prediction_text[:80]!r}...")

    deadline = datetime.now(timezone.utc) + timedelta(days=PREDICTION_HORIZON_DAYS)
    prediction = Prediction(
        prediction_id   = make_id("pred_"),
        entity_id       = entity_id,
        prediction_text = prediction_text,
        confidence      = confidence,
        predicted_at    = datetime.now(timezone.utc),
        deadline        = deadline,
        outcome         = None,
    )
    memory.add_prediction(prediction)
    print(f"[prediction_agent] Stored prediction {prediction.prediction_id}")

    return prediction
