"""
Entity Tracker Agent
Task 3 — Full-Stack & Evaluation Engineer

WHAT THIS DOES:
- Reads from Graph DB: all claims mentioning a specific entity + their verdicts
- Computes a credibility score and sentiment score for a time window
- Writes a new CredibilitySnapshot node back to Graph DB
- Runs periodically (e.g., every hour via cron or manual trigger)

HOW IT CONNECTS:
- Reads from:  Graph DB (Neo4j) — CLAIM, VERDICT, MENTIONS edges  (via MemoryAgent)
- Writes to:   Graph DB — new CREDIBILITY_SNAPSHOT node, updated ENTITY fields

RUN: python -m agents.entity_tracker
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from id_utils import make_id
from models.credibility import CredibilitySnapshot

from agents.memory_agent import MemoryAgent, get_memory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SCORING LOGIC (original from Task 3 — unchanged)
# ─────────────────────────────────────────────


def compute_credibility_score(claims: list[dict]) -> float:
    """
    Recency-weighted credibility score with Bayesian volume shrinkage.

    Formula (per claim i):
        label_score  : supported=1.0, misleading=0.35, refuted=0.0
        time_decay   : e^(-λ * days_ago),  λ = ln(2)/14  → half-life 14 days
        weight_i     : confidence_i × time_decay_i

        evidence     = Σ(label_score_i × weight_i) / Σ(weight_i)

    Volume shrinkage — few claims pull the score toward the neutral prior (0.5):
        volume_factor = 1 - e^(-n/3)   → ~0.63 at n=1, ~0.86 at n=3, ~0.99 at n=10
        credibility   = 0.5×(1-volume_factor) + evidence×volume_factor

    Result is always in [0, 1].
    """
    if not claims:
        return 0.5

    HALF_LIFE_DAYS = 14.0
    LAMBDA = math.log(2) / HALF_LIFE_DAYS

    LABEL_SCORES = {"supported": 1.0, "misleading": 0.35, "refuted": 0.0}

    now = datetime.now(timezone.utc)
    weighted_sum = 0.0
    total_weight = 0.0

    for c in claims:
        conf = float(c.get("verdict_confidence") or 0.5)
        label = (c.get("verdict_label") or "misleading").lower()
        score = LABEL_SCORES.get(label, 0.35)

        # Recency: convert neo4j DateTime or ISO string to Python datetime
        verified_at = c.get("verified_at")
        try:
            if verified_at is None:
                days_ago = 0.0
            elif hasattr(verified_at, "to_native"):
                verified_at = verified_at.to_native()
                if verified_at.tzinfo is None:
                    verified_at = verified_at.replace(tzinfo=timezone.utc)
                days_ago = (now - verified_at).total_seconds() / 86400
            elif isinstance(verified_at, str):
                verified_at = datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
                days_ago = (now - verified_at).total_seconds() / 86400
            else:
                days_ago = 0.0
        except Exception:
            days_ago = 0.0

        time_decay = math.exp(-LAMBDA * max(days_ago, 0.0))
        weight = conf * time_decay

        weighted_sum += score * weight
        total_weight += weight

    evidence = weighted_sum / total_weight if total_weight > 0 else 0.5

    # Bayesian shrinkage toward 0.5 when n is small
    n = len(claims)
    volume_factor = 1.0 - math.exp(-n / 3.0)
    credibility = 0.5 * (1 - volume_factor) + evidence * volume_factor

    return round(credibility, 4)


def compute_sentiment_score(claims: list[dict]) -> float:
    """
    Recency-weighted sentiment score in [-1, +1].

    Maps positive→+1, neutral→0, negative→-1.
    Recent claims weighted more (same half-life as credibility).
    """
    if not claims:
        return 0.0

    HALF_LIFE_DAYS = 14.0
    LAMBDA = math.log(2) / HALF_LIFE_DAYS
    SENTIMENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    now = datetime.now(timezone.utc)
    weighted_sum = 0.0
    total_weight = 0.0

    for c in claims:
        score = SENTIMENT_MAP.get(c.get("sentiment", "neutral"), 0.0)

        verified_at = c.get("verified_at")
        try:
            if verified_at is None:
                days_ago = 0.0
            elif hasattr(verified_at, "to_native"):
                verified_at = verified_at.to_native()
                if verified_at.tzinfo is None:
                    verified_at = verified_at.replace(tzinfo=timezone.utc)
                days_ago = (now - verified_at).total_seconds() / 86400
            elif isinstance(verified_at, str):
                verified_at = datetime.fromisoformat(verified_at.replace("Z", "+00:00"))
                days_ago = (now - verified_at).total_seconds() / 86400
            else:
                days_ago = 0.0
        except Exception:
            days_ago = 0.0

        weight = math.exp(-LAMBDA * max(days_ago, 0.0))
        weighted_sum += score * weight
        total_weight += weight

    return round(weighted_sum / total_weight if total_weight > 0 else 0.0, 4)


# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────


def run_entity_tracker(
    entity_name: str, window_hours: int = 24, memory: Optional[MemoryAgent] = None
) -> Optional[CredibilitySnapshot]:
    """
    Main function of the Entity Tracker Agent.

    Steps:
    1. Find the entity in Graph DB by name
    2. Pull all verified claims mentioning it (within the time window)
    3. Compute credibility score and sentiment score
    4. Write a new CredibilitySnapshot node to Graph DB
    5. Update the Entity node's aggregate fields

    Args:
        entity_name  : The name of the entity to track (e.g., "Tesla")
        window_hours : How far back to look for claims (default: last 24 hours)
        memory       : MemoryAgent instance (uses singleton if None)
    """
    if memory is None:
        memory = get_memory()

    print(f"\n{'=' * 60}")
    print(f"  Entity Tracker Agent — Running for: '{entity_name}'")
    print(f"  Window: last {window_hours} hours")
    print(f"{'=' * 60}")

    # ── Step 1: Find the entity ──
    print("\n[1/5] Looking up entity in Graph DB...")
    entity_dict = memory.get_entity_by_name(entity_name)
    if entity_dict is None:
        print(f"  ERROR: Entity '{entity_name}' not found in Graph DB. Skipping.")
        return None

    entity_id = entity_dict["entity_id"]
    current_credibility = entity_dict.get("current_credibility", 0.5)
    total_claims = entity_dict.get("total_claims") or 0
    accurate_claims = entity_dict.get("accurate_claims") or 0
    print(f"  Found: {entity_dict['name']} (id: {entity_id})")

    # ── Step 2: Pull recent claims ──
    since = datetime.now() - timedelta(hours=window_hours)
    print(f"\n[2/5] Querying claims since {since.strftime('%Y-%m-%d %H:%M')}...")
    # Returns: [{claim_id, claim_text, verdict_label, verdict_confidence, sentiment}, ...]
    claims = memory.get_entity_claims(entity_id, since=since)
    print(f"  Found {len(claims)} verified claims in this window")

    if not claims:
        print("  No claims in this window. No snapshot will be written.")
        return None

    label_counts = {"supported": 0, "refuted": 0, "misleading": 0}
    for c in claims:
        label = c.get("verdict_label", "")
        if label in label_counts:
            label_counts[label] += 1
    print(f"  Breakdown: {label_counts}")

    # ── Step 3: Compute scores ──
    print("\n[3/5] Computing scores...")
    credibility_score = compute_credibility_score(claims)
    sentiment_score = compute_sentiment_score(claims)
    print(f"  Credibility score : {credibility_score:.4f}")
    print(f"  Sentiment score   : {sentiment_score:.4f}")

    drift = credibility_score - current_credibility
    if abs(drift) > 0.1:
        direction = "DROP" if drift < 0 else "RISE"
        print(
            f"  ALERT: Significant credibility {direction} detected! "
            f"({current_credibility:.2f} → {credibility_score:.2f})"
        )

    # ── Step 4: Write snapshot ──
    print("\n[4/5] Writing CredibilitySnapshot to Graph DB...")
    snapshot = CredibilitySnapshot(
        snapshot_id=make_id("snap_"),
        entity_id=entity_id,
        credibility_score=credibility_score,
        sentiment_score=sentiment_score,
        snapshot_at=datetime.now(),
    )
    memory.add_credibility_snapshot(snapshot)
    print(f"  Snapshot {snapshot.snapshot_id} written.")

    # ── Step 5: Update entity aggregates ──
    print("\n[5/5] Updating Entity aggregate fields in Graph DB...")
    new_total = total_claims + len(claims)
    new_accurate = accurate_claims + label_counts["supported"]
    memory.update_entity(
        entity_id,
        total_claims=new_total,
        accurate_claims=new_accurate,
        current_credibility=credibility_score,
        last_seen=datetime.now().isoformat(),
    )
    print(
        f"  total_claims={new_total}, accurate_claims={new_accurate}, "
        f"current_credibility={credibility_score:.4f}"
    )

    print(f"\nEntity Tracker complete for '{entity_name}'")
    print(f"   Snapshot ID: {snapshot.snapshot_id}")
    print(f"   Final credibility: {credibility_score:.2%}")

    # ── Auto-run Prediction Agent if enough snapshots exist ──
    try:
        from agents.prediction_agent import run_prediction_agent

        all_snaps = memory.get_entity_snapshots(entity_id, limit=20)
        if len(all_snaps) >= 3:
            print(
                f"\n[entity_tracker] {len(all_snaps)} snapshots available — running prediction agent"
            )
            run_prediction_agent(entity_name, memory=memory)
        else:
            print(f"\n[entity_tracker] Only {len(all_snaps)} snapshot(s) — prediction needs 3+")
    except Exception as _pe:
        logger.warning("Prediction agent skipped: %s", _pe)

    return snapshot


def run_batch_tracker(
    entity_names: list[str], window_hours: int = 24, memory: Optional[MemoryAgent] = None
) -> dict:
    """Run the Entity Tracker for a list of entities."""
    if memory is None:
        memory = get_memory()

    print(f"\n{'#' * 60}")
    print(f"  BATCH TRACKER — {len(entity_names)} entities")
    print(f"{'#' * 60}")

    results = {}
    for name in entity_names:
        snapshot = run_entity_tracker(name, window_hours, memory)
        results[name] = snapshot

    print(f"\n{'#' * 60}")
    print("  BATCH COMPLETE")
    for name, snap in results.items():
        if snap:
            print(f"  OK {name}: credibility={snap.credibility_score:.2%}")
        else:
            print(f"  SKIP {name}: no data")
    print(f"{'#' * 60}\n")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_entity_tracker("Tesla", window_hours=24)
