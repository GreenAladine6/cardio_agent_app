from __future__ import annotations

from agents.audio_agent import AudioAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.reasoning_agent import ReasoningAgent
from schemas.final_response_schema import FinalResponse


class CardioAgentOrchestrator:
    def __init__(self) -> None:
        self.audio_agent = AudioAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.reasoning_agent = ReasoningAgent()

    def run(self, *, audio_engine_result: dict, user_question: str) -> FinalResponse:
        audio_result = self.audio_agent.validate_audio_output(audio_engine_result)
        knowledge_result = self.knowledge_agent.retrieve_passages(user_question)
        return self.reasoning_agent.build_final_response(
            audio_result=audio_result,
            knowledge_result=knowledge_result,
            user_question=user_question,
        )
