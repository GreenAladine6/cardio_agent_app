from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

AGENT_ORCHESTRATOR_PATH = (
    Path(__file__).resolve().parents[1] / "agent-orchestrator"
)
if str(AGENT_ORCHESTRATOR_PATH) not in sys.path:
    sys.path.insert(0, str(AGENT_ORCHESTRATOR_PATH))

from guardrails.output_guard import validate_final_response
from guardrails.safety_guard import SafetyGuardError, run_safety_checks
from schemas.audio_schema import AudioAgentResult
from schemas.knowledge_schema import KnowledgeAgentResult
from schemas.reasoning_schema import ReasoningAgentResult

pytestmark = pytest.mark.unit


def _valid_final_response_payload() -> dict:
    return {
        "audio_result": {
            "prediction": "pathological",
            "confidence": 0.82,
            "model_name": "pcg-onnx-v1",
        },
        "retrieved_passages": [
            {
                "text": "ESC guideline passage example.",
                "source_document": "esc-2023.pdf",
                "page_number": 21,
                "chunk_id": "chunk-21-1",
                "similarity_score": 0.93,
            }
        ],
        "reasoning_result": {
            "explanation": "The audio pattern and guideline context suggest caution.",
            "evidence": ["ESC guideline passage example."],
            "limitations": "This is a decision-support output, not a diagnosis.",
            "clinical_review_required": True,
        },
        "final_summary": (
            "Evidence from guideline context supports a pathological-risk signal. "
            "Clinical review is required."
        ),
    }


def test_audio_schema_validation_accepts_valid_payload():
    result = AudioAgentResult.model_validate(
        {
            "prediction": "normal",
            "confidence": 0.55,
            "model_name": "pcg-onnx-v1",
        }
    )
    assert result.prediction == "normal"


def test_knowledge_schema_validation_accepts_valid_payload():
    result = KnowledgeAgentResult.model_validate(
        {
            "passages": [
                {
                    "text": "Clinical context.",
                    "source_document": "esc.pdf",
                    "page_number": 3,
                    "chunk_id": "c-3-1",
                    "similarity_score": 0.89,
                }
            ]
        }
    )
    assert result.passages[0].source_document == "esc.pdf"


def test_reasoning_schema_validation_accepts_valid_payload():
    result = ReasoningAgentResult.model_validate(
        {
            "explanation": "Model output and passages were cross-checked.",
            "evidence": ["Passage A", "Passage B"],
            "limitations": "Limited by model uncertainty.",
            "clinical_review_required": True,
        }
    )
    assert result.clinical_review_required is True


def test_invalid_confidence_rejected():
    with pytest.raises(ValidationError):
        AudioAgentResult.model_validate(
            {
                "prediction": "normal",
                "confidence": 1.4,
                "model_name": "pcg-onnx-v1",
            }
        )


def test_empty_explanation_rejected():
    with pytest.raises(ValidationError):
        ReasoningAgentResult.model_validate(
            {
                "explanation": "",
                "evidence": ["Passage A"],
                "limitations": "Limited by model uncertainty.",
                "clinical_review_required": True,
            }
        )


def test_final_response_validation_and_safety_rules():
    payload = _valid_final_response_payload()
    final_response = validate_final_response(payload)
    run_safety_checks(final_response)


def test_safety_guard_rejects_diagnosis_claim():
    payload = _valid_final_response_payload()
    payload["final_summary"] = "This is a confirmed diagnosis of severe pathology."
    final_response = validate_final_response(payload)
    with pytest.raises(SafetyGuardError):
        run_safety_checks(final_response)
