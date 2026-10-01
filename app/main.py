from pathlib import Path

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

    return JSONResponse({"filename": file.filename, "message": "WAV file received"})
