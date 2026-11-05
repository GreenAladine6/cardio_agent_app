from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from audio_engine.audio_engine.pipeline import AudioPreprocessor, FeatureConfig

pytestmark = pytest.mark.unit


def wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    return buffer.getvalue()


def test_wav_pipeline_resamples_downmixes_and_returns_finite_fixed_features():
    source_rate = 4000
    time = np.arange(source_rate, dtype=np.float32) / source_rate
    mono = 0.2 * np.sin(2 * np.pi * 90 * time)
    stereo = np.column_stack([mono, mono * 0.5])
    pipeline = AudioPreprocessor()

    features = pipeline.from_bytes(wav_bytes(stereo, source_rate))

    assert features.shape == (40,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_pipeline_rejects_non_wav_malformed_and_nonfinite_audio():
    pipeline = AudioPreprocessor()
    with pytest.raises(ValueError, match="WAV"):
        pipeline.from_bytes(b"not a wav")
    with pytest.raises(ValueError, match="finite"):
        pipeline.from_signal(np.array([0.0, np.nan], dtype=np.float32), 2000)
    with pytest.raises(ValueError, match="shorter"):
        pipeline.from_signal(np.zeros(100, dtype=np.float32), 2000)


def test_silent_recording_is_finite_and_short_window_is_padded():
    config = FeatureConfig(target_seconds=2.0, min_duration_seconds=0.1)
    pipeline = AudioPreprocessor(config)
    features = pipeline.from_signal(np.zeros(200, dtype=np.float32), 2000)
    assert features.shape == (40,)
    assert np.isfinite(features).all()
