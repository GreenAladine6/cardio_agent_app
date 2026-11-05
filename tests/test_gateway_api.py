from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from api_getway.app import main

pytestmark = pytest.mark.integration


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
    error: httpx.HTTPError | None = None
    request = None

    def __init__(self, *, timeout: float):
        assert timeout == 30.0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def post(self, url: str, *, files):
        type(self).request = (url, files)
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def client(monkeypatch):
    StubAsyncClient.response = StubResponse(200, json_data={"prediction": {"label": "normal"}})
    StubAsyncClient.error = None
    StubAsyncClient.request = None
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


def test_forwards_valid_upload_and_returns_engine_response(client):
    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", b"RIFF-audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": {"label": "normal"}}
    assert StubAsyncClient.request == (
        "http://audio-engine.test/v1/infer",
        {"file": ("recording.wav", b"RIFF-audio", "audio/wav")},
    )


def test_preserves_engine_error_response(client):
    StubAsyncClient.response = StubResponse(
        503,
        json_data={"detail": "Model unavailable"},
    )

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", b"RIFF-audio", "audio/wav")},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model unavailable"}


def test_returns_engine_text_when_response_is_not_json(client):
    StubAsyncClient.response = StubResponse(502, text="upstream unavailable")

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", b"RIFF-audio", "audio/wav")},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream unavailable"}


def test_maps_engine_connection_errors_to_bad_gateway(client):
    StubAsyncClient.error = httpx.ConnectError("connection refused")

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", b"RIFF-audio", "audio/wav")},
    )

    assert response.status_code == 502
    assert "Audio engine request failed" in response.json()["detail"]


def test_serves_frontend_pages(client):
    assert client.get("/").status_code == 200
    assert client.get("/test").status_code == 200
