from __future__ import annotations

from pydantic import ValidationError

from schemas.audio_schema import AudioAgentResult


class AudioAgent:
    def validate_audio_output(self, payload: dict) -> AudioAgentResult:
        normalized = self._normalize_audio_payload(payload)
        try:
            return AudioAgentResult.model_validate(normalized)
        except ValidationError as exc:
            raise ValueError("Invalid Audio Engine output format") from exc

    def _normalize_audio_payload(self, payload: dict) -> dict:
        label = str(payload.get("label", "")).strip().lower()
        if label not in {"normal", "abnormal", "pathological"}:
            raise ValueError("Audio Engine label must be normal, abnormal, or pathological")

        probability_abnormal = payload.get("probability_abnormal")
        if probability_abnormal is None:
            confidence = 0.5
        else:
            confidence = float(probability_abnormal)

        if label in {"abnormal", "pathological"}:
            prediction = "pathological"
            final_confidence = confidence
        else:
            prediction = "normal"
            final_confidence = 1.0 - confidence

        return {
            "prediction": prediction,
            "confidence": final_confidence,
            "model_name": str(payload.get("model_version") or "unknown-model"),
        }
