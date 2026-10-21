from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile

from .features import compute_feature_vector
from .schemas import FeatureSummary, HealthResponse, PredictionResponse


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/artifacts/cardio_model.onnx"))
META_PATH = MODEL_PATH.with_suffix(".json")


app = FastAPI(title="CardioAudioEngine", version="0.1.0")
app.state.session = None
app.state.meta = {}
app.state.model_loaded = False
app.state.startup_error = None


def _load_model_and_meta() -> tuple[ort.InferenceSession, dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"ONNX model not found at {MODEL_PATH}")
    if not META_PATH.exists():
        raise RuntimeError(f"Model metadata not found at {META_PATH}")
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return session, meta


def _ensure_wav(data: bytes) -> tuple[np.ndarray, int]:
    if not data:
        raise HTTPException(status_code=400, detail="Audio payload is empty")
    try:
        with wave.open(file="", mode="rb") as _:
            pass
    except Exception:
        pass
    try:
        buffer = __import__("io").BytesIO(data)
        with wave.open(buffer, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
            if sample_width not in (1, 2, 3, 4):
                raise ValueError("Unsupported WAV sample width")
            if channels > 0:
                import array

                values = array.array("h")
                values.frombytes(frames)
                if sample_width == 1:
                    values = array.array("h", [(b - 128) for b in frames])
                elif sample_width == 2:
                    arr = np.frombuffer(frames, dtype=np.int16)
                    mono = arr.reshape(-1, channels).mean(axis=1).astype(np.float32)
                    return mono, sample_rate
                elif sample_width == 4:
                    arr = np.frombuffer(frames, dtype=np.int32)
                    mono = arr.reshape(-1, channels).mean(axis=1).astype(np.float32)
                    return mono, sample_rate
            arr = np.frombuffer(frames, dtype=np.int16 if sample_width == 2 else np.uint8)
            mono = arr.reshape(-1, channels).mean(axis=1).astype(np.float32) if channels > 1 else arr.astype(np.float32)
            return mono, sample_rate
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio decode failed: {exc}") from exc


@app.on_event("startup")
def _startup() -> None:
    try:
        app.state.session, app.state.meta = _load_model_and_meta()
        app.state.model_loaded = True
    except Exception as exc:  # pragma: no cover - startup guard
        app.state.session = None
        app.state.meta = {}
        app.state.model_loaded = False
        app.state.startup_error = str(exc)


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
        "model_version": str(app.state.meta.get("model_version", "unknown")),
        "service": "audio-engine",
    }


@app.post("/features")
def features(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.filename is None or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are accepted")
    data = file.file.read()
    audio, sample_rate = _ensure_wav(data)
    if audio.size == 0:
        raise HTTPException(status_code=422, detail="Decoded audio contains no samples")
    if sample_rate != 2000:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=2000)
    audio = audio.astype(np.float32)
    vector = compute_feature_vector(audio, sr=2000)
    summary = FeatureSummary(
        mfcc_mean=[float(x) for x in vector[:20]],
        zcr_mean=float(vector[20]),
        rms_mean=float(vector[-2]),
        duration_s=float(len(audio) / 2000.0),
        estimated_heart_rate_bpm=None,
    )
    return {"features_summary": summary.model_dump()}


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.filename is None or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are accepted")
    start = time.perf_counter()
    data = file.file.read()
    audio, sample_rate = _ensure_wav(data)
    if len(audio) < 2000:
        raise HTTPException(status_code=422, detail="Audio is too short; provide at least 1 second of data")
    if sample_rate != 2000:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=2000)
    audio = audio.astype(np.float32)
    features = compute_feature_vector(audio, sr=2000)
    features = np.asarray([features], dtype=np.float32)
    session = app.state.session
    if session is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: features})
    prob_values = output[1]
    if isinstance(prob_values, dict):
        prob_values = [list(prob_values.values())]
    if isinstance(prob_values, list) and prob_values and isinstance(prob_values[0], dict):
        prob_values = [list(row.values()) for row in prob_values]
    probability = float(np.asarray(prob_values, dtype=np.float32).reshape(-1)[1]) if len(output) > 1 else 0.5
    threshold = float(app.state.meta.get("threshold", 0.5))
    label = "pathological" if probability >= threshold else "normal"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    summary = FeatureSummary(
        mfcc_mean=[float(x) for x in features[0][:20]],
        zcr_mean=float(features[0][20]),
        rms_mean=float(features[0][-2]),
        duration_s=float(len(audio) / 2000.0),
        estimated_heart_rate_bpm=None,
    )
    return {
        "label": label,
        "probability_pathological": probability,
        "threshold": threshold,
        "n_segments": 1,
        "segment_probabilities": [probability],
        "duration_s": float(len(audio) / 2000.0),
        "model_version": str(app.state.meta.get("model_version", "unknown")),
        "inference_ms": elapsed_ms,
        "features_summary": summary,
    }
