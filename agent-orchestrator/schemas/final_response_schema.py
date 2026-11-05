from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .audio_schema import AudioAgentResult
from .knowledge_schema import RetrievedPassage
from .reasoning_schema import ReasoningAgentResult


class FinalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_result: AudioAgentResult
    retrieved_passages: list[RetrievedPassage] = Field(min_length=1)
    reasoning_result: ReasoningAgentResult
    final_summary: str = Field(min_length=1)
