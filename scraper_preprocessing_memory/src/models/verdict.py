"""Verdict data model (written by Fact-Check Agent)."""

from datetime import datetime

from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """Fact-check verdict for a claim.

    Written by the Fact-Check Agent (teammate's module).
    Model defined here so teammates can construct typed objects
    when calling memory.add_verdict().
    """

    verdict_id: str
    claim_id: str
    label: str  # "supported", "refuted", "misleading"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str
    bias_score: float = Field(ge=0.0, le=1.0)
    image_mismatch: bool = False
    verified_at: datetime
