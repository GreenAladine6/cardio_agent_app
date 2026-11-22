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
from fastapi.responses import FileResponse

from .pipeline import AudioPreprocessor


MODEL_PATH = Path(os.getenv("ONNX_MODEL_PATH", "/models/pcg_classifier.onnx"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "/frontend"))
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
    recording_id: str | None = Field(default=None, min_length=1, max_length=300)


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


def _decode_upload_waveform(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="Audio upload is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {MAX_UPLOAD_BYTES} bytes")
    try:
        waveform, _ = preprocessor.waveform_from_bytes(data)
        return waveform
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _inference_result(features: np.ndarray, waveform: np.ndarray | None = None):
    import time

    started = time.perf_counter()
    session = _load_model()
    input_meta = session.get_inputs()[0]
    if input_meta.type != "tensor(float)":
        raise HTTPException(status_code=503, detail="ONNX model input must be float32")
    labels = []
    metadata = {}
    metadata_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = metadata.get("classes", [])
        except (OSError, json.JSONDecodeError):
            labels = []
    input_type = metadata.get("input_type", "mfcc_statistics")
    if input_type == "log_mel":
        if waveform is None:
            raise HTTPException(status_code=422, detail="Waveform is required by the configured mel-input model")
        import librosa

        preprocessing = metadata.get("preprocessing", {})
        mel_config = preprocessing.get("mel", {}) if isinstance(preprocessing, dict) else {}
        sample_rate = int(preprocessing.get("sample_rate", preprocessor.config.sample_rate)) if isinstance(preprocessing, dict) else preprocessor.config.sample_rate
        if sample_rate != preprocessor.config.sample_rate:
            raise HTTPException(status_code=503, detail="ONNX model sample rate does not match the audio service")
        mel_power = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=int(mel_config.get("n_fft", preprocessor.config.n_fft)),
            hop_length=int(mel_config.get("hop_length", preprocessor.config.hop_length)),
            n_mels=int(mel_config.get("n_mels", 64)),
            fmin=float(mel_config.get("fmin", 20)),
            fmax=float(mel_config.get("fmax", 1000)),
        )
        log_mel = librosa.power_to_db(mel_power, ref=np.max).astype(np.float32)
        model_input = log_mel[np.newaxis, np.newaxis, :, :]
        if len(input_meta.shape) != model_input.ndim or any(
            isinstance(expected, int) and expected != actual
            for expected, actual in zip(input_meta.shape, model_input.shape)
        ):
            raise HTTPException(status_code=503, detail="ONNX mel input shape does not match the audio preprocessing configuration")
        outputs = session.run(None, {input_meta.name: model_input.astype(np.float32)})
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(1, -1)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        prediction = int(np.argmax(probabilities[0]))
        probability_map = {
            (labels[i] if i < len(labels) else str(i)): float(value)
            for i, value in enumerate(probabilities[0])
        }
    else:
        declared_width = input_meta.shape[-1]
        if isinstance(declared_width, int) and declared_width != features.shape[0]:
            raise HTTPException(status_code=503, detail="ONNX model feature width does not match the audio pipeline")
        outputs = session.run(None, {input_meta.name: features.reshape(1, -1).astype(np.float32)})
        prediction = int(np.asarray(outputs[0]).reshape(-1)[0])
        probability_map = None
    raw_label = labels[prediction] if 0 <= prediction < len(labels) else str(prediction)
    diagnosis = _binary_diagnosis(raw_label, prediction)
    result = {
        "label": diagnosis,
        "diagnosis": diagnosis,
        "model_label": raw_label,
        "class_index": prediction,
        "is_abnormal": diagnosis == "abnormal",
        "inference_ms": (time.perf_counter() - started) * 1000.0,
    }
    if probability_map is not None:
        result["probabilities"] = probability_map
        result["probability_abnormal"] = _abnormal_probability(probability_map)
    elif len(outputs) > 1:
        probabilities = outputs[1]
        prob_map = _to_probability_map(probabilities, labels)
        result["probabilities"] = prob_map
        result["probability_abnormal"] = _abnormal_probability(prob_map)
    return result


def _make_windows(waveform: np.ndarray) -> list[np.ndarray]:
    """Create normalized, overlapping windows matching offline training."""
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    window_length = int(preprocessor.config.sample_rate * preprocessor.config.target_seconds)
    hop_length = max(1, int(window_length * 0.5))
    starts = list(range(0, max(1, audio.size - window_length + 1), hop_length))
    if starts[-1] + window_length < audio.size:
        trailing_start = starts[-1] + hop_length
        if trailing_start < audio.size:
            starts.append(trailing_start)

    windows: list[np.ndarray] = []
    for start in starts:
        window = audio[start : start + window_length]
        if window.size < window_length:
            window = np.pad(window, (0, window_length - window.size))
        energy_db = 10.0 * np.log10(float(np.mean(np.square(window))) + 1e-10)
        if energy_db < -45.0:
            continue
        peak = float(np.max(np.abs(window)))
        if peak > 1e-8:
            window = window / peak
        windows.append(np.asarray(window, dtype=np.float32))

    if not windows:
        first = audio[:window_length]
        if first.size < window_length:
            first = np.pad(first, (0, window_length - first.size))
        windows.append(np.asarray(first, dtype=np.float32))
    return windows


def _aggregate_segment_results(results: list[dict], duration_s: float) -> dict:
    probabilities = [item.get("probability_abnormal") for item in results]
    valid_probabilities = [float(value) for value in probabilities if value is not None]
    if valid_probabilities:
        probability = float(np.mean(valid_probabilities))
        diagnosis = "abnormal" if probability >= 0.5 else "normal"
        probability_map = {}
        keys = set.intersection(*[
            set(item.get("probabilities", {})) for item in results
            if isinstance(item.get("probabilities"), dict)
        ]) if any(isinstance(item.get("probabilities"), dict) for item in results) else set()
        for key in keys:
            values = [item["probabilities"][key] for item in results if key in item.get("probabilities", {})]
            probability_map[key] = float(np.mean(values))
    else:
        diagnosis = max(
            (item["label"] for item in results),
            key=lambda label: sum(result["label"] == label for result in results),
        )
        probability = None
        probability_map = {}
    chosen = next((item for item in results if item["label"] == diagnosis), results[0])
    output = {
        **chosen,
        "label": diagnosis,
        "diagnosis": diagnosis,
        "is_abnormal": diagnosis == "abnormal",
        "n_segments": len(results),
        "segment_probabilities": [float(value) for value in probabilities if value is not None],
        "duration_s": float(duration_s),
        "inference_ms": float(sum(item.get("inference_ms", 0.0) for item in results)),
    }
    if probability is not None:
        output["probability_abnormal"] = probability
    if probability_map:
        output["probabilities"] = probability_map
    return output


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


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/v1/config")
def pipeline_config():
    return {**preprocessor.config.to_dict(), "feature_count": preprocessor.feature_count}


@app.post("/v1/infer")
async def infer(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    waveform = _decode_upload_waveform(data)
    windows = _make_windows(waveform)
    segment_results = []
    for window in windows:
        features = preprocessor.from_signal(window, preprocessor.config.sample_rate)
        segment_results.append(_inference_result(features, window))
    duration = len(waveform) / preprocessor.config.sample_rate
    return {"prediction": _aggregate_segment_results(segment_results, duration)}


@app.post("/v1/benchmark")
def benchmark(request: BenchmarkRequest):
    # Keep sklearn/MLflow conversion dependencies out of the inference import path.
    from .benchmark import run_benchmark

    features = []
    labels = []
    recording_ids = []
    for index, sample in enumerate(request.samples):
        try:
            data = base64.b64decode(sample.audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Sample {index}: invalid base64 audio") from exc
        features.append(_decode_upload(data))
        labels.append(sample.label)
        recording_ids.append(sample.recording_id or f"sample-{index}")
    try:
        report = run_benchmark(
            np.stack(features), labels, model_path=MODEL_PATH,
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "pcg-benchmark"),
            recording_ids=recording_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Audio benchmark failed")
        raise HTTPException(status_code=500, detail="Benchmark failed; see service logs") from exc
    _load_model.cache_clear()
    return report
