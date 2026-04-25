"""Credibility snapshot and Prediction models (written by teammates)."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CredibilitySnapshot(BaseModel):
    """Point-in-time credibility snapshot for an entity.

    Written by Entity Tracker Agent. Append-only to preserve
    full history for trend detection.
    """

    snapshot_id: str
    entity_id: str
    credibility_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    snapshot_at: datetime


class Prediction(BaseModel):
    """Time-bound prediction about an entity.

    Written by Prediction Agent.
    """

    prediction_id: str
    entity_id: str
    prediction_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    predicted_at: datetime
    deadline: datetime
    outcome: Optional[str] = None  # "confirmed", "refuted", "inconclusive", or None
