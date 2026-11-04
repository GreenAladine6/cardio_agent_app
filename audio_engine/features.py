from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - optional at runtime
    yaml = None


def load_feature_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else Path(__file__).resolve().with_name("config.yaml")
    if path.exists() and yaml and hasattr(yaml, "safe_load"):
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict) and "feature" in loaded:
            return loaded["feature"]
    return {
        "sample_rate": 2000,
        "n_mfcc": 20,
        "n_fft": 256,
        "hop_length": 64,
        "n_mels": 64,
        "mel_fmin": 20,
        "mel_fmax": 1000,
        "window_seconds": 5.0,
        "overlap_ratio": 0.5,
        "normalization": "peak",
        "bandpass": {"low_hz": 25, "high_hz": 400, "order": 4},
        "energy_threshold_db": -45.0,
    }


def _as_f32(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=np.float32).reshape(-1)


def compute_feature_vector(signal: np.ndarray, *, sr: int = 2000, config: dict[str, Any] | None = None) -> np.ndarray:
    """Compute the canonical 1D feature vector used for classical models and inference."""
    cfg = load_feature_config() if config is None else config
    audio = _as_f32(signal)
    if audio.size == 0:
        raise ValueError("Audio signal is empty")
    if sr <= 0:
        raise ValueError("Sampling rate must be positive")

    audio = librosa.util.normalize(audio)
    target_seconds = float(cfg.get("target_seconds", 8.0))
    target_length = int(sr * target_seconds)
    if audio.size < target_length:
        audio = np.pad(audio, (0, target_length - audio.size))
    else:
        audio = audio[:target_length]

    n_fft = int(cfg.get("n_fft", 256))
    hop = int(cfg.get("hop_length", 64))
    n_mfcc = int(cfg.get("n_mfcc", 20))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    mfcc_delta = np.diff(mfcc, axis=1, prepend=mfcc[:, :1]).mean(axis=1)

    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop).mean()
    zcr_std = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop).std()
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop)

    stats = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            mfcc_delta,
            np.array([zcr, zcr_std], dtype=np.float32),
            np.array([centroid.mean(), centroid.std()], dtype=np.float32),
            np.array([bandwidth.mean(), bandwidth.std()], dtype=np.float32),
            np.array([rolloff.mean(), rolloff.std()], dtype=np.float32),
            np.array([rms.mean(), rms.std()], dtype=np.float32),
            np.array([flatness.mean(), flatness.std()], dtype=np.float32),
            np.array([audio.mean(), audio.std()], dtype=np.float32),
        ]
    )
    return stats.astype(np.float32)


def compute_log_mel(signal: np.ndarray, *, sr: int = 2000, config: dict[str, Any] | None = None) -> np.ndarray:
    """Compute single-channel log-Mel spectrogram with the canonical parameters used in training and serving."""
    cfg = load_feature_config() if config is None else config
    audio = _as_f32(signal)
    if audio.size == 0:
        raise ValueError("Audio signal is empty")
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=int(cfg.get("n_fft", 256)),
        hop_length=int(cfg.get("hop_length", 32)),
        n_mels=int(cfg.get("n_mels", 64)),
        fmin=float(cfg.get("mel_fmin", 20)),
        fmax=float(cfg.get("mel_fmax", 1000)),
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.expand_dims(log_mel.astype(np.float32), axis=0)
