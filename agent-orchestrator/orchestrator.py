from __future__ import annotations

from typing import Any, TypedDict

from agents.audio_agent import AudioAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.reasoning_agent import ReasoningAgent
from schemas.final_response_schema import FinalResponse


class _AgentState(TypedDict, total=False):
    audio_engine_result: dict[str, Any]
    user_question: str
    audio_result: Any
    knowledge_result: Any
    final_response: Any


class CardioAgentOrchestrator:
    def __init__(self) -> None:
        self.audio_agent = AudioAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.reasoning_agent = ReasoningAgent()

    def run(self, *, audio_engine_result: dict, user_question: str) -> FinalResponse:
        def validate_audio(state: _AgentState) -> _AgentState:
            return {"audio_result": self.audio_agent.validate_audio_output(state["audio_engine_result"])}

        def retrieve_knowledge(state: _AgentState) -> _AgentState:
            return {"knowledge_result": self.knowledge_agent.retrieve_passages(state["user_question"])}

        def reason(state: _AgentState) -> _AgentState:
            return {"final_response": self.reasoning_agent.build_final_response(
                audio_result=state["audio_result"],
                knowledge_result=state["knowledge_result"],
                user_question=state["user_question"],
            )}

        initial_state: _AgentState = {
            "audio_engine_result": audio_engine_result,
            "user_question": user_question,
        }
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            state = initial_state | validate_audio(initial_state)
            state |= retrieve_knowledge(state)
            state |= reason(state)
            return state["final_response"]

        graph = StateGraph(_AgentState)
        graph.add_node("audio_agent", validate_audio)
        graph.add_node("knowledge_agent", retrieve_knowledge)
        graph.add_node("reasoning_agent", reason)
        graph.add_edge(START, "audio_agent")
        graph.add_edge("audio_agent", "knowledge_agent")
        graph.add_edge("knowledge_agent", "reasoning_agent")
        graph.add_edge("reasoning_agent", END)
        return graph.compile().invoke(initial_state)["final_response"]
