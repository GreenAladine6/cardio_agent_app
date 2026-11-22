from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pytest

AGENT_ORCHESTRATOR_PATH = (
    Path(__file__).resolve().parents[1] / "agent-orchestrator"
)
if str(AGENT_ORCHESTRATOR_PATH) not in sys.path:
    sys.path.insert(0, str(AGENT_ORCHESTRATOR_PATH))

from agents.reasoning_agent import ReasoningAgent, ReasoningAgentError
from orchestrator import CardioAgentOrchestrator
from schemas.audio_schema import AudioAgentResult
from schemas.knowledge_schema import KnowledgeAgentResult, RetrievedPassage

API_PATH = Path(__file__).resolve().parents[1] / "agent-orchestrator" / "app.py"
API_SPEC = importlib.util.spec_from_file_location("orchestrator_http_api", API_PATH)
orchestrator_http_api = importlib.util.module_from_spec(API_SPEC)
assert API_SPEC.loader is not None
API_SPEC.loader.exec_module(orchestrator_http_api)

pytestmark = pytest.mark.unit


def test_orchestrator_flow_uses_three_agents(monkeypatch):
    orchestrator = CardioAgentOrchestrator()

    monkeypatch.setattr(
        orchestrator.knowledge_agent,
        "retrieve_passages",
        lambda _: KnowledgeAgentResult(
            passages=[
                RetrievedPassage(
                    text="Guideline passage.",
                    source_document="esc.pdf",
                    page_number=10,
                    chunk_id="c-10-1",
                    similarity_score=0.9,
                )
            ]
        ),
    )
    monkeypatch.setattr(
        orchestrator.reasoning_agent,
        "build_final_response",
        lambda audio_result, knowledge_result, user_question: {
            "audio_prediction": audio_result.prediction,
            "question": user_question,
            "passage_count": len(knowledge_result.passages),
        },
    )

    result = orchestrator.run(
        audio_engine_result={
            "label": "abnormal",
            "probability_abnormal": 0.92,
            "model_version": "pcg-onnx-v1",
        },
        user_question="What is the likely interpretation?",
    )

    assert result["audio_prediction"] == "pathological"
    assert result["passage_count"] == 1


def test_reasoning_agent_retries_once_for_invalid_llm_output(monkeypatch):
    agent = ReasoningAgent()
    calls: list[bool] = []

    def fake_call_llm(_):
        calls.append(True)
        if len(calls) == 1:
            return "Not JSON"
        return (
            '{"explanation":"Evidence suggests risk.","evidence":["ESC p.10"],'
            '"limitations":"Not a diagnosis.","clinical_review_required":true}'
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = agent.build_final_response(
        audio_result=AudioAgentResult(
            prediction="pathological",
            confidence=0.9,
            model_name="pcg-onnx-v1",
        ),
        knowledge_result=KnowledgeAgentResult(
            passages=[
                RetrievedPassage(
                    text="ESC recommendation passage.",
                    source_document="esc.pdf",
                    page_number=10,
                    chunk_id="c-10-1",
                    similarity_score=0.95,
                )
            ]
        ),
        user_question="What should be considered next?",
    )

    assert len(calls) == 2
    assert result.reasoning_result.clinical_review_required is True


def test_reasoning_agent_fails_after_one_retry(monkeypatch):
    agent = ReasoningAgent()
    monkeypatch.setattr(agent, "_call_llm", lambda _: "still invalid")

    with pytest.raises(ReasoningAgentError):
        agent.build_final_response(
            audio_result=AudioAgentResult(
                prediction="normal",
                confidence=0.7,
                model_name="pcg-onnx-v1",
            ),
            knowledge_result=KnowledgeAgentResult(
                passages=[
                    RetrievedPassage(
                        text="Guideline context.",
                        source_document="esc.pdf",
                        page_number=5,
                        chunk_id="c-5-1",
                        similarity_score=0.8,
                    )
                ]
            ),
            user_question="Any concerns?",
        )


def test_orchestrator_http_api_validates_and_returns_structured_response(monkeypatch):
    from pydantic import ValidationError

    monkeypatch.setattr(
        orchestrator_http_api.orchestrator,
        "run",
        lambda **_: {"status": "structured"},
    )
    payload = orchestrator_http_api.OrchestrationRequest.model_validate({
        "audio_engine_result": {"label": "normal", "probability_abnormal": 0.1},
        "user_question": "What does this result mean?",
    })
    endpoint = next(
        route.endpoint for route in orchestrator_http_api.app.routes
        if getattr(route, "path", None) == "/v1/orchestrate"
    )
    response = endpoint(payload)

    assert response == {"status": "structured"}
    with pytest.raises(ValidationError):
        orchestrator_http_api.OrchestrationRequest.model_validate(
            {"audio_engine_result": {}, "user_question": ""}
        )
