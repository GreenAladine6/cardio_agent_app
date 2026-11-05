from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievedPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source_document: str = Field(min_length=1, max_length=300)
    page_number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1, max_length=200)
    similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)


class KnowledgeAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passages: list[RetrievedPassage] = Field(min_length=1)
