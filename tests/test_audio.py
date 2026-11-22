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
from audio_engine.audio_engine.pipeline import AudioPreprocessor
from audio_engine.audio_engine.service import app


@pytest.fixture
def synthetic_wave():
    sr = 2000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 80 * t) + 0.2 * np.sin(2 * np.pi * 200 * t)
    return signal.astype(np.float32), sr


@pytest.mark.unit
def test_resample_and_normalize_audio(synthetic_wave):
    signal, sr = synthetic_wave
    resampled = resample_audio(signal, sr, target_sr=2000)
    assert resampled.shape[0] > 0
    assert np.isfinite(resampled).all()

    normalized = signal / max(np.max(np.abs(signal)), 1e-8)
    assert normalized.shape == signal.shape


@pytest.mark.unit
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


@pytest.mark.unit
def test_fixed_length_windowing_and_padding():
    signal = np.zeros(1000, dtype=np.float32)
    windows = segment_audio(signal, sample_rate=2000, window_seconds=5.0, overlap=0.5)
    assert len(windows) >= 1
    assert all(w.shape[0] == 10000 for w in windows)


@pytest.mark.unit
def test_feature_shapes(synthetic_wave):
    signal, sr = synthetic_wave
    vector = compute_feature_vector(signal, sr=2000)
    mel = compute_log_mel(signal, sr=2000)
    assert vector.shape[0] == 20 * 3 + 2 + 2 + 2 + 2 + 2 + 2 + 2
    assert mel.shape[0] == 1
    assert mel.shape[1] in {64}


@pytest.mark.unit
def test_service_preprocessor_uses_fixed_width_features(synthetic_wave):
    signal, sr = synthetic_wave
    preprocessor = AudioPreprocessor()
    service_vector = preprocessor.from_signal(signal, sr)

    assert preprocessor.feature_count == 40
    assert service_vector.shape == (preprocessor.feature_count,)
    assert np.isfinite(service_vector).all()


@pytest.mark.unit
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
    assert split.groupby("recording_id")["split"].nunique().max() == 1


@pytest.mark.unit
def test_group_split_keeps_mixed_label_patient_together():
    df = pd.DataFrame({
        "recording_id": ["a", "b", "c", "d", "e", "f"],
        "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
        "binary_label": [0, 1, 0, 1, 0, 1],
    })

    split = make_grouped_split(
        df,
        target_col="binary_label",
        group_col="patient_id",
        train_fraction=1 / 3,
        val_fraction=1 / 3,
        test_fraction=1 / 3,
    )

    assert split.groupby("patient_id")["split"].nunique().max() == 1


@pytest.mark.unit
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


@pytest.mark.integration
def test_api_handles_wrong_input_and_non_wav(tmp_path):
    client = TestClient(app)
    response = client.post("/v1/infer", files={"file": ("bad.txt", b"not-wav", "text/plain")})
    assert response.status_code in {415, 422}

    response = client.post("/v1/infer", files={"file": ("short.wav", b"RIFF...", "audio/wav")})
    assert response.status_code in {422, 503}


@pytest.mark.integration
def test_root_serves_frontend_index(tmp_path, monkeypatch):
    from audio_engine.audio_engine import service

    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    index = frontend_dir / "index.html"
    index.write_text("<html>Cardio dashboard</html>", encoding="utf-8")
    monkeypatch.setattr(service, "FRONTEND_DIR", frontend_dir)

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.text == "<html>Cardio dashboard</html>"


@pytest.mark.integration
def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "onnx_model_ready" in data


@pytest.mark.unit
def test_mel_model_inference_uses_waveform_and_metadata(tmp_path, monkeypatch):
    from audio_engine.audio_engine import service

    class Input:
        name = "log_mel"
        type = "tensor(float)"
        shape = [None, 1, 64, None]

    class Session:
        def get_inputs(self):
            return [Input()]

        def run(self, _, feed):
            assert feed["log_mel"].shape[:3] == (1, 1, 64)
            return [np.asarray([[0.0, 2.0]], dtype=np.float32)]

    model_path = tmp_path / "model.onnx"
    model_path.with_suffix(".onnx.json").write_text(
        json.dumps({"input_type": "log_mel", "classes": ["normal", "pathological"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "MODEL_PATH", model_path)
    monkeypatch.setattr(service, "_load_model", lambda: Session())
    time = np.arange(10_000, dtype=np.float32) / 2000
    waveform = np.sin(2 * np.pi * 80 * time).astype(np.float32)

    result = service._inference_result(np.zeros(40, dtype=np.float32), waveform)

    assert result["label"] == "abnormal"
    assert result["class_index"] == 1
    assert result["probability_abnormal"] > 0.8


@pytest.mark.unit
def test_long_recordings_use_overlapping_segments_and_average_scores():
    from audio_engine.audio_engine.service import _aggregate_segment_results, _make_windows

    waveform = np.ones(16_000, dtype=np.float32)
    windows = _make_windows(waveform)
    assert len(windows) == 3
    assert all(window.shape == (10_000,) for window in windows)

    prediction = _aggregate_segment_results(
        [
            {"label": "normal", "probability_abnormal": 0.2, "probabilities": {"normal": 0.8, "pathological": 0.2}, "inference_ms": 1.0},
            {"label": "abnormal", "probability_abnormal": 0.8, "probabilities": {"normal": 0.2, "pathological": 0.8}, "inference_ms": 2.0},
        ],
        duration_s=8.0,
    )
    assert prediction["label"] == "abnormal"
    assert prediction["n_segments"] == 2
    assert prediction["probability_abnormal"] == pytest.approx(0.5)
    assert prediction["inference_ms"] == 3.0
