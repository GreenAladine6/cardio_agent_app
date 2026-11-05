from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AudioAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Literal["normal", "pathological"]
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str = Field(min_length=1, max_length=200)
