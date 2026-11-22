from __future__ import annotations

import io
import wave

import httpx
import pytest
from fastapi.testclient import TestClient

from api_getway.app import main

pytestmark = pytest.mark.integration


def wav_payload() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(2000)
        wav_file.writeframes(b"\0\0" * 1000)
    return buffer.getvalue()


class StubResponse:
    def __init__(self, status_code: int, *, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("Response is not JSON")
        return self._json_data


class StubAsyncClient:
    response = StubResponse(200, json_data={"prediction": {"label": "normal"}})
    agent_response = StubResponse(200, json_data={"final_summary": "Review with a clinician."})
    error: httpx.HTTPError | None = None
    request = None
    agent_request = None

    def __init__(self, *, timeout: float):
        assert timeout in {30.0, 120.0}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def post(self, url: str, *, files=None, json=None):
        if self.error is not None:
            raise self.error
        if json is not None:
            type(self).agent_request = (url, json)
            return self.agent_response
        type(self).request = (url, files)
        return self.response


@pytest.fixture
def client(monkeypatch):
    StubAsyncClient.response = StubResponse(200, json_data={"prediction": {"label": "normal"}})
    StubAsyncClient.error = None
    StubAsyncClient.request = None
    StubAsyncClient.agent_request = None
    monkeypatch.setattr(main.httpx, "AsyncClient", StubAsyncClient)
    monkeypatch.setattr(main, "AUDIO_ENGINE_URL", "http://audio-engine.test")
    monkeypatch.setattr(main, "AUDIO_ENGINE_INFER_PATH", "/v1/infer")
    return TestClient(main.app)


def test_rejects_non_wav_and_empty_uploads(client):
    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.txt", b"audio", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only .wav files are accepted"

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Audio upload is empty"

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", b"not a wav", "audio/wav")},
    )
    assert response.status_code == 400


def test_forwards_valid_upload_and_returns_engine_response(client):
    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", wav_payload(), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": {"label": "normal"}}
    assert StubAsyncClient.request == (
        "http://audio-engine.test/v1/infer",
        {"file": ("recording.wav", wav_payload(), "audio/wav")},
    )


def test_rejects_upload_over_configured_size_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", wav_payload(), "audio/wav")},
    )

    assert response.status_code == 413


def test_optional_question_invokes_agent_orchestrator(client, monkeypatch):
    monkeypatch.setattr(main, "ORCHESTRATOR_URL", "http://agent-orchestrator.test")
    monkeypatch.setattr(main, "ORCHESTRATOR_PATH", "/v1/orchestrate")

    response = client.post(
        "/analyze_pcg",
        files={
            "file": ("recording.wav", wav_payload(), "audio/wav"),
            "question": (None, "What should I do next?"),
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": {"label": "normal"},
        "orchestration": {"final_summary": "Review with a clinician."},
    }
    assert StubAsyncClient.agent_request == (
        "http://agent-orchestrator.test/v1/orchestrate",
        {
            "audio_engine_result": {"label": "normal"},
            "user_question": "What should I do next?",
        },
    )


def test_stream_endpoint_emits_progress_and_final_result(client, monkeypatch):
    async def fake_analysis(file, question):
        return main.AnalyzePCGResponse(prediction={"label": "normal"})

    monkeypatch.setattr(main, "analyze_pcg", fake_analysis)
    response = client.post(
        "/analyze_pcg/stream",
        files={"file": ("recording.wav", wav_payload(), "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: progress\ndata: {"status": "analysis_started"}' in response.text
    assert 'event: complete\ndata: {"prediction": {"label": "normal"}, "orchestration": null}' in response.text


def test_preserves_engine_error_response(client):
    StubAsyncClient.response = StubResponse(
        503,
        json_data={"detail": "Model unavailable"},
    )

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", wav_payload(), "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model unavailable"}


def test_returns_engine_text_when_response_is_not_json(client):
    StubAsyncClient.response = StubResponse(502, text="upstream unavailable")

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", wav_payload(), "audio/wav")},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream unavailable"}


def test_maps_engine_connection_errors_to_bad_gateway(client):
    StubAsyncClient.error = httpx.ConnectError("connection refused")

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", wav_payload(), "audio/wav")},
    )

    assert response.status_code == 502
    assert "Audio engine request failed" in response.json()["detail"]


def test_serves_frontend_pages(client):
    assert client.get("/").status_code == 200
    assert client.get("/test").status_code == 200
