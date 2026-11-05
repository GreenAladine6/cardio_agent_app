"""HTTP interface consumed directly by the AudioAgent."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import AudioPreprocessor


MODEL_PATH = Path(os.getenv("ONNX_MODEL_PATH", "/models/pcg_classifier.onnx"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))
preprocessor = AudioPreprocessor()
logger = logging.getLogger("audio_engine")

app = FastAPI(
    title="PCG Audio Engine",
    description="PCG preprocessing, model benchmarking, and ONNX inference service for AudioAgent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LabeledAudio(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    audio_base64: str = Field(min_length=1)


class BenchmarkRequest(BaseModel):
    samples: list[LabeledAudio] = Field(min_length=8, max_length=256)


def _normalize_label(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _binary_diagnosis(raw_label: str, class_index: int) -> str:
    token = _normalize_label(raw_label)
    if token in {"1", "abnormal", "pathological", "murmur", "extrasystole", "extrahls", "positive", "disease"}:
        return "abnormal"
    if token in {"0", "normal", "negative", "healthy"}:
        return "normal"
    return "abnormal" if int(class_index) == 1 else "normal"


def _to_probability_map(values, labels: list[str]) -> dict[str, float]:
    if isinstance(values, list) and values and isinstance(values[0], dict):
        row = values[0]
        return {str(key): float(value) for key, value in row.items()}
    if isinstance(values, dict):
        return {str(key): float(value) for key, value in values.items()}

    arr = np.asarray(values).reshape(-1)
    return {
        (labels[i] if i < len(labels) else str(i)): float(value)
        for i, value in enumerate(arr)
    }


def _abnormal_probability(prob_map: dict[str, float]) -> float | None:
    if not prob_map:
        return None
    abnormal_total = 0.0
    normal_total = 0.0
    for key, value in prob_map.items():
        diagnosis = _binary_diagnosis(str(key), 1 if _normalize_label(str(key)) in {"1", "abnormal", "pathological", "murmur", "extrasystole", "extrahls", "positive", "disease"} else 0)
        if diagnosis == "abnormal":
            abnormal_total += float(value)
        else:
            normal_total += float(value)
    total = abnormal_total + normal_total
    if total <= 0:
        return None
    return abnormal_total / total


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="No trained ONNX model is available yet")
    try:
        return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    except Exception as exc:
        logger.exception("Configured ONNX model could not be loaded")
        raise HTTPException(status_code=503, detail="Configured ONNX model is unavailable or invalid") from exc


def _decode_upload(data: bytes):
    if not data:
        raise HTTPException(status_code=400, detail="Audio upload is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {MAX_UPLOAD_BYTES} bytes")
    try:
        return preprocessor.from_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _inference_result(features: np.ndarray):
    import time

    started = time.perf_counter()
    session = _load_model()
    input_meta = session.get_inputs()[0]
    declared_width = input_meta.shape[-1]
    if isinstance(declared_width, int) and declared_width != features.shape[0]:
        raise HTTPException(status_code=503, detail="ONNX model feature width does not match the audio pipeline")
    if input_meta.type != "tensor(float)":
        raise HTTPException(status_code=503, detail="ONNX model input must be float32")
    outputs = session.run(None, {input_meta.name: features.reshape(1, -1).astype(np.float32)})
    prediction = int(np.asarray(outputs[0]).reshape(-1)[0])
    labels = []
    metadata = {}
    metadata_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = metadata.get("classes", [])
        except (OSError, json.JSONDecodeError):
            labels = []
    raw_label = labels[prediction] if 0 <= prediction < len(labels) else str(prediction)
    diagnosis = _binary_diagnosis(raw_label, prediction)
    result = {
        "label": diagnosis,
        "diagnosis": diagnosis,
        "model_label": raw_label,
        "class_index": prediction,
        "is_abnormal": diagnosis == "abnormal",
    }
    if len(outputs) > 1:
        probabilities = outputs[1]
        prob_map = _to_probability_map(probabilities, labels)
        result["probabilities"] = prob_map
        result["probability_abnormal"] = _abnormal_probability(prob_map)
    return result


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "audio-engine",
        "sample_rate": preprocessor.config.sample_rate,
        "mfcc_count": preprocessor.config.n_mfcc,
        "feature_count": preprocessor.feature_count,
        "onnx_model_ready": MODEL_PATH.is_file(),
    }


@app.get("/v1/config")
def pipeline_config():
    return {**preprocessor.config.to_dict(), "feature_count": preprocessor.feature_count}


@app.post("/v1/infer")
async def infer(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    features = _decode_upload(data)
    return {"prediction": _inference_result(features)}


@app.post("/v1/benchmark")
def benchmark(request: BenchmarkRequest):
    # Keep sklearn/MLflow conversion dependencies out of the inference import path.
    from .benchmark import run_benchmark

    features = []
    labels = []
    for index, sample in enumerate(request.samples):
        try:
            data = base64.b64decode(sample.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Sample {index}: invalid base64 audio") from exc
        features.append(_decode_upload(data))
        labels.append(sample.label)
    try:
        report = run_benchmark(
            np.stack(features), labels, model_path=MODEL_PATH,
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "pcg-benchmark"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Audio benchmark failed")
        raise HTTPException(status_code=500, detail="Benchmark failed; see service logs") from exc
    _load_model.cache_clear()
    return report
