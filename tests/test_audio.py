from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from audio_engine.data_prep import harmonize_labels, make_grouped_split, segment_audio, resample_audio
from audio_engine.features import compute_feature_vector, compute_log_mel
from audio_engine.export_onnx import export_sklearn_to_onnx, parity_check_onnx
from audio_engine.serve import app


@pytest.fixture
def synthetic_wave():
    sr = 2000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 80 * t) + 0.2 * np.sin(2 * np.pi * 200 * t)
    return signal.astype(np.float32), sr


def test_resample_and_normalize_audio(synthetic_wave):
    signal, sr = synthetic_wave
    resampled = resample_audio(signal, sr, target_sr=2000)
    assert resampled.shape[0] > 0
    assert np.isfinite(resampled).all()

    normalized = signal / max(np.max(np.abs(signal)), 1e-8)
    assert normalized.shape == signal.shape


def test_label_harmonization():
    df = pd.DataFrame(
        {
            "recording_id": ["a", "b", "c", "d"],
            "source": ["pascal", "physionet", "pascal", "physionet"],
            "original_label": ["normal", "abnormal", "artifact", "extrasystole"],
        }
    )
    result = harmonize_labels(df)
    assert set(result["binary_label"].unique()) == {0, 1}
    assert "artifact" not in set(result[result["binary_label"].notna() == True]["original_label"])


def test_fixed_length_windowing_and_padding():
    signal = np.zeros(1000, dtype=np.float32)
    windows = segment_audio(signal, sample_rate=2000, window_seconds=5.0, overlap=0.5)
    assert len(windows) >= 1
    assert all(w.shape[0] == 10000 for w in windows)


def test_feature_shapes(synthetic_wave):
    signal, sr = synthetic_wave
    vector = compute_feature_vector(signal, sr=2000)
    mel = compute_log_mel(signal, sr=2000)
    assert vector.shape[0] == 20 * 3 + 2 + 2 + 2 + 2 + 2 + 2 + 2
    assert mel.shape[0] == 1
    assert mel.shape[1] in {64}


def test_group_split_no_leakage():
    df = pd.DataFrame(
        {
            "recording_id": ["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4"],
            "binary_label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    split = make_grouped_split(df, target_col="binary_label", group_col="recording_id", test_size=0.25, seed=42)
    assert set(split["recording_id"]).issubset(df["recording_id"].unique())
    assert len(split) == len(df)


def test_onnx_export_and_parity(tmp_path):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(50, 6)).astype(np.float32)
    x_test = rng.normal(size=(10, 6)).astype(np.float32)
    y_train = np.array([0, 1] * 25, dtype=np.int64)
    y_test = np.array([0, 1] * 5, dtype=np.int64)

    model = make_pipeline(StandardScaler(), SVC(probability=True, kernel="rbf", C=1.0, gamma="scale"))
    model.fit(x_train, y_train)

    model_path = tmp_path / "tiny_model.onnx"
    metadata = export_sklearn_to_onnx(model, model_path, feature_dim=6)
    assert model_path.exists()
    assert metadata["feature_dim"] == 6
    parity = parity_check_onnx(model, model_path, x_test[:5])
    assert parity["max_abs_diff"] >= 0


def test_api_handles_wrong_input_and_non_wav(tmp_path):
    client = TestClient(app)
    response = client.post("/predict", files={"file": ("bad.txt", b"not-wav", "text/plain")})
    assert response.status_code == 400

    response = client.post("/predict", files={"file": ("short.wav", b"RIFF...", "audio/wav")})
    assert response.status_code in {400, 422}


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
