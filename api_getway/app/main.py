from pathlib import Path
from typing import Any
import json
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://localhost:8001").rstrip("/")
AUDIO_ENGINE_INFER_PATH = os.getenv("AUDIO_ENGINE_INFER_PATH", "/v1/infer")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8003").rstrip("/")
ORCHESTRATOR_PATH = os.getenv("ORCHESTRATOR_PATH", "/v1/orchestrate")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))


class AnalyzePCGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: dict[str, Any]
    orchestration: dict[str, Any] | None = None

app = FastAPI(
    title="API Gateway",
    description="Gateway for audio analysis requests",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/test")
async def test_page():
    return FileResponse(FRONTEND_DIR / "test.html")


@app.post("/analyze_pcg", response_model=AnalyzePCGResponse, response_model_exclude_none=True)
async def analyze_pcg(file: UploadFile = File(...), question: str | None = Form(default=None, max_length=4000)):
    if question is not None and not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Audio upload is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {MAX_UPLOAD_BYTES} bytes")
    if len(payload) < 12 or payload[:4] not in {b"RIFF", b"RF64"} or payload[8:12] != b"WAVE":
        raise HTTPException(status_code=400, detail="Uploaded file content is not a WAV recording")

    url = f"{AUDIO_ENGINE_URL}{AUDIO_ENGINE_INFER_PATH}"
    files = {"file": (file.filename, payload, file.content_type or "audio/wav")}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, files=files)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Audio engine request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text}

    if 200 <= response.status_code < 300:
        try:
            audio_response = AnalyzePCGResponse.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(status_code=502, detail="Audio engine returned an invalid response") from exc
        if question is None:
            return audio_response
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                agent_response = await client.post(
                    f"{ORCHESTRATOR_URL}{ORCHESTRATOR_PATH}",
                    json={
                        "audio_engine_result": audio_response.prediction,
                        "user_question": question.strip(),
                    },
                )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Agent orchestrator request failed: {exc}") from exc
        try:
            agent_data = agent_response.json()
        except ValueError:
            agent_data = {"detail": agent_response.text}
        if not 200 <= agent_response.status_code < 300:
            return JSONResponse(status_code=agent_response.status_code, content=agent_data)
        if not isinstance(agent_data, dict):
            raise HTTPException(status_code=502, detail="Agent orchestrator returned an invalid response")
        return AnalyzePCGResponse(prediction=audio_response.prediction, orchestration=agent_data)
    return JSONResponse(status_code=response.status_code, content=data)


@app.post("/analyze_pcg/stream", response_class=StreamingResponse)
async def analyze_pcg_stream(file: UploadFile = File(...), question: str | None = Form(default=None, max_length=4000)):
    async def events():
        yield f"event: progress\ndata: {json.dumps({'status': 'analysis_started'})}\n\n"
        try:
            result = await analyze_pcg(file, question)
        except HTTPException as exc:
            payload = {"detail": exc.detail, "status_code": exc.status_code}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"
            return
        if isinstance(result, JSONResponse):
            payload = json.loads(result.body.decode("utf-8"))
            event_name = "error" if result.status_code >= 400 else "complete"
        else:
            payload = result.model_dump(mode="json")
            event_name = "complete"
        yield f"event: {event_name}\ndata: {json.dumps(payload, allow_nan=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
