from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReasoningAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str = Field(min_length=1)
    evidence: list[str]
    limitations: str = Field(min_length=1)
    clinical_review_required: bool

    @field_validator("evidence")
    @classmethod
    def _validate_evidence_items(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Evidence entries must be non-empty strings")
        return value
