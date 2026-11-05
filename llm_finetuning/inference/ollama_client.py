from __future__ import annotations

import math
import os
from typing import Any

import httpx


class OllamaClientError(RuntimeError):
    """Base exception for Ollama upstream failures."""


class OllamaUnavailableError(OllamaClientError):
    """Ollama could not be reached."""


class OllamaTimeoutError(OllamaClientError):
    """Ollama did not respond before the configured timeout."""


class OllamaResponseError(OllamaClientError):
    """Ollama returned an HTTP error or an invalid response."""


class OllamaModelUnavailableError(OllamaClientError):
    """The configured model has not been imported into Ollama."""


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must use http:// or https://")
        if not model.strip():
            raise ValueError("Ollama model name must not be empty")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Ollama request timeout must be a positive finite number")

        self.base_url = normalized_url
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> OllamaClient:
        timeout = os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120")
        try:
            timeout_seconds = float(timeout)
        except ValueError as exc:
            raise ValueError(
                "OLLAMA_REQUEST_TIMEOUT_SECONDS must be a positive finite number"
            ) from exc
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL_NAME", "cardio-llm"),
            timeout_seconds=timeout_seconds,
        )

    async def health(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama health check timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                f"Ollama health check returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError("Ollama health check failed") from exc
        except ValueError as exc:
            raise OllamaResponseError("Ollama health check returned invalid JSON") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise OllamaResponseError("Ollama health check returned an invalid payload")
        model_names = {
            entry.get("name")
            for entry in payload["models"]
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
        if self.model not in model_names and f"{self.model}:latest" not in model_names:
            raise OllamaModelUnavailableError(
                f"Configured Ollama model '{self.model}' is not installed"
            )

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        request_payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=request_payload,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama inference request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                f"Ollama inference returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError("Ollama inference request failed") from exc
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama returned an invalid response")
        response_model = payload.get("model")
        message = payload.get("message")
        if (
            not isinstance(response_model, str)
            or not response_model.strip()
            or not isinstance(message, dict)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
            or not isinstance(payload.get("done"), bool)
        ):
            raise OllamaResponseError("Ollama returned an invalid chat response")

        return {
            "model": response_model,
            "message": {
                "role": "assistant",
                "content": message["content"],
            },
            "done": payload["done"],
        }
