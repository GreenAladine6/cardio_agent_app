from pathlib import Path
import os

import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="API Gateway",
    description="Gateway for audio analysis requests",
    version="1.0.0"
)

AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://audio-engine:8000").rstrip("/")


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/test")
async def test_page():
    return FileResponse(FRONTEND_DIR / "test.html")


@app.post("/analyze_pcg")
async def analyze_pcg(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")

    try:
        content = await file.read(20 * 1024 * 1024 + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Audio upload is empty")
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio upload exceeds 20 MiB")
        timeout = httpx.Timeout(45.0, connect=3.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{AUDIO_ENGINE_URL}/v1/infer",
                files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            )
        if response.status_code >= 400:
            # Keep downstream diagnostics private and preserve useful HTTP semantics.
            detail = response.json().get("detail", "Audio analysis failed")
            raise HTTPException(status_code=response.status_code, detail=detail)
        return JSONResponse(response.json())
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Audio engine timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="Audio engine is unavailable") from exc
