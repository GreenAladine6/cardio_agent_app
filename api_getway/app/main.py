from pathlib import Path
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://localhost:8001").rstrip("/")
AUDIO_ENGINE_INFER_PATH = os.getenv("AUDIO_ENGINE_INFER_PATH", "/v1/infer")

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


@app.post("/analyze_pcg")
async def analyze_pcg(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Audio upload is empty")

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

    return JSONResponse(status_code=response.status_code, content=data)
