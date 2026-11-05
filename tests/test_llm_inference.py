from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_finetuning.inference import ollama_client
from llm_finetuning.inference.api import create_app
from llm_finetuning.inference.ollama_client import (
    OllamaClient,
    OllamaModelUnavailableError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

pytestmark = pytest.mark.integration


class StubAsyncClient:
    response: httpx.Response
    error: Exception | None = None
    request: tuple[str, dict | None] | None = None

    def __init__(self, *, timeout: float):
        assert timeout == 5

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def get(self, url: str):
        type(self).request = (url, None)
        if self.error is not None:
            raise self.error
        return self.response

    async def post(self, url: str, *, json: dict):
        type(self).request = (url, json)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def ollama(monkeypatch):
    StubAsyncClient.error = None
    StubAsyncClient.request = None
    monkeypatch.setattr(ollama_client.httpx, "AsyncClient", StubAsyncClient)
    return OllamaClient(
        base_url="http://ollama.test/",
        model="cardio-llm",
        timeout_seconds=5,
    )


def test_client_sends_non_streaming_chat_with_configured_model(ollama):
    StubAsyncClient.response = httpx.Response(
        200,
        json={
            "model": "cardio-llm",
            "message": {"role": "assistant", "content": "Hello."},
            "done": True,
        },
        request=httpx.Request("POST", "http://ollama.test/api/chat"),
    )

    result = asyncio.run(
        ollama.chat(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.4,
            max_tokens=128,
        )
    )

    assert result == {
        "model": "cardio-llm",
        "message": {"role": "assistant", "content": "Hello."},
        "done": True,
    }
    assert StubAsyncClient.request == (
        "http://ollama.test/api/chat",
        {
            "model": "cardio-llm",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 128},
        },
    )


def test_client_health_checks_ollama_tags_endpoint(ollama):
    StubAsyncClient.response = httpx.Response(
        200,
        json={"models": [{"name": "cardio-llm:latest"}]},
        request=httpx.Request("GET", "http://ollama.test/api/tags"),
    )

    asyncio.run(ollama.health())

    assert StubAsyncClient.request == ("http://ollama.test/api/tags", None)


def test_client_health_rejects_missing_model(ollama):
    StubAsyncClient.response = httpx.Response(
        200,
        json={"models": [{"name": "another-model:latest"}]},
        request=httpx.Request("GET", "http://ollama.test/api/tags"),
    )

    with pytest.raises(OllamaModelUnavailableError, match="cardio-llm"):
        asyncio.run(ollama.health())


@pytest.mark.parametrize(
    ("error", "expected_error"),
    [
        (httpx.ConnectTimeout("timed out"), OllamaTimeoutError),
        (httpx.ConnectError("unavailable"), OllamaUnavailableError),
    ],
)
def test_client_maps_transport_errors(ollama, error, expected_error):
    StubAsyncClient.error = error

    with pytest.raises(expected_error):
        asyncio.run(
            ollama.chat(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.2,
                max_tokens=64,
            )
        )


def test_client_rejects_malformed_ollama_response(ollama):
    StubAsyncClient.response = httpx.Response(
        200,
        json={"model": "cardio-llm"},
        request=httpx.Request("POST", "http://ollama.test/api/chat"),
    )

    with pytest.raises(OllamaResponseError, match="invalid chat response"):
        asyncio.run(
            ollama.chat(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.2,
                max_tokens=64,
            )
        )


def test_client_surfaces_upstream_http_status(ollama):
    StubAsyncClient.response = httpx.Response(
        503,
        request=httpx.Request("POST", "http://ollama.test/api/chat"),
    )

    with pytest.raises(OllamaResponseError, match="HTTP 503"):
        asyncio.run(
            ollama.chat(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.2,
                max_tokens=64,
            )
        )


class FakeOllamaClient:
    model = "cardio-llm"

    def __init__(self):
        self.chat_request = None
        self.health_error = None
        self.chat_error = None

    async def health(self):
        if self.health_error:
            raise self.health_error

    async def chat(self, *, messages, temperature, max_tokens):
        self.chat_request = (messages, temperature, max_tokens)
        if self.chat_error:
            raise self.chat_error
        return {
            "model": self.model,
            "message": {"role": "assistant", "content": "Please consult a clinician."},
            "done": True,
        }


@pytest.fixture
def api_client():
    fake = FakeOllamaClient()
    return fake, TestClient(create_app(fake))


def test_chat_endpoint_forwards_request_and_returns_completion(api_client):
    fake, client = api_client

    response = client.post(
        "/v1/chat",
        json={
            "messages": [
                {"role": "system", "content": "Be cautious."},
                {"role": "user", "content": "What does this sound mean?"},
            ],
            "temperature": 0.1,
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "cardio-llm",
        "message": {
            "role": "assistant",
            "content": "Please consult a clinician.",
        },
        "done": True,
    }
    assert fake.chat_request == (
        [
            {"role": "system", "content": "Be cautious."},
            {"role": "user", "content": "What does this sound mean?"},
        ],
        0.1,
        64,
    )


def test_chat_endpoint_validates_inputs(api_client):
    _, client = api_client

    assert client.post("/v1/chat", json={"messages": []}).status_code == 422
    assert (
        client.post(
            "/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 3,
            },
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (OllamaTimeoutError("Ollama inference request timed out"), 504),
        (OllamaUnavailableError("Ollama inference request failed"), 503),
        (OllamaResponseError("Ollama returned an invalid response"), 502),
    ],
)
def test_chat_endpoint_maps_upstream_errors(api_client, error, status_code):
    fake, client = api_client
    fake.chat_error = error

    response = client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == status_code
    assert response.json()["detail"] == str(error)


def test_health_endpoint_checks_ollama(api_client):
    _, client = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "cardio-llm"}


def test_health_endpoint_is_unavailable_until_model_is_installed(api_client):
    fake, client = api_client
    fake.health_error = OllamaModelUnavailableError(
        "Configured Ollama model 'cardio-llm' is not installed"
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == str(fake.health_error)
