from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import ValidationError

from config import LLM_SERVER_URL, LLM_TIMEOUT_SECONDS
from guardrails.output_guard import validate_final_response
from guardrails.safety_guard import run_safety_checks
from schemas.audio_schema import AudioAgentResult
from schemas.final_response_schema import FinalResponse
from schemas.knowledge_schema import KnowledgeAgentResult
from schemas.reasoning_schema import ReasoningAgentResult


class ReasoningAgentError(RuntimeError):
    """Raised when the reasoning agent cannot produce a valid response."""


class ReasoningAgent:
    def build_final_response(
        self,
        audio_result: AudioAgentResult,
        knowledge_result: KnowledgeAgentResult,
        user_question: str,
    ) -> FinalResponse:
        first_response = self._call_llm(
            self._build_messages(audio_result, knowledge_result, user_question, retry=False)
        )
        reasoning_result = self._parse_and_validate_reasoning(first_response)
        if reasoning_result is None:
            second_response = self._call_llm(
                self._build_messages(
                    audio_result, knowledge_result, user_question, retry=True
                )
            )
            reasoning_result = self._parse_and_validate_reasoning(second_response)
            if reasoning_result is None:
                raise ReasoningAgentError(
                    "LLM returned invalid structured output after one retry"
                )

        final_payload = {
            "audio_result": audio_result.model_dump(),
            "retrieved_passages": [item.model_dump() for item in knowledge_result.passages],
            "reasoning_result": reasoning_result.model_dump(),
            "final_summary": reasoning_result.explanation,
        }
        final_response = validate_final_response(final_payload)
        run_safety_checks(final_response)
        return final_response

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 700,
        }
        try:
            response = httpx.post(LLM_SERVER_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ReasoningAgentError("LLM request timed out") from exc
        except httpx.HTTPError as exc:
            raise ReasoningAgentError("LLM request failed") from exc

        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ReasoningAgentError("LLM response did not contain text content")
        return content

    def _parse_and_validate_reasoning(self, response_text: str) -> ReasoningAgentResult | None:
        payload = _parse_json_object(response_text)
        if payload is None:
            return None
        try:
            return ReasoningAgentResult.model_validate(payload)
        except ValidationError:
            return None

    def _build_messages(
        self,
        audio_result: AudioAgentResult,
        knowledge_result: KnowledgeAgentResult,
        user_question: str,
        *,
        retry: bool,
    ) -> list[dict[str, str]]:
        passages_text = "\n".join(
            f"- {item.source_document} p.{item.page_number}: {item.text}"
            for item in knowledge_result.passages
        )
        instruction = (
            "Return only JSON with keys: explanation (string), evidence (list of strings), "
            "limitations (string), clinical_review_required (boolean)."
        )
        if retry:
            instruction = (
                "Previous answer was invalid. "
                + instruction
                + " Do not include markdown or extra keys."
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are the Cardio reasoning agent. Never provide confirmed diagnoses. "
                    "Use cautious language."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {user_question}\n"
                    f"Audio prediction: {audio_result.prediction} "
                    f"(confidence={audio_result.confidence:.3f})\n"
                    f"Evidence passages:\n{passages_text}\n\n{instruction}"
                ),
            },
        ]


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
