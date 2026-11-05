from __future__ import annotations

from schemas.audio_schema import AudioAgentResult
from schemas.knowledge_schema import KnowledgeAgentResult
from schemas.reasoning_schema import ReasoningAgentResult


def validate_audio_agent_input(payload: dict) -> AudioAgentResult:
    return AudioAgentResult.model_validate(payload)


def validate_knowledge_agent_input(payload: dict) -> KnowledgeAgentResult:
    return KnowledgeAgentResult.model_validate(payload)


def validate_reasoning_agent_input(payload: dict) -> ReasoningAgentResult:
    return ReasoningAgentResult.model_validate(payload)
