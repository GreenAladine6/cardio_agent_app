"""Librosa based PCG feature extraction shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io

import librosa
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 2000
    n_mfcc: int = 20
    n_fft: int = 256
    hop_length: int = 64
    target_seconds: float = 8.0

    def to_dict(self):
        return asdict(self)


class AudioPreprocessor:
    """Decode, resample, peak-normalize, and summarize MFCCs from PCG audio."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    @property
    def feature_count(self) -> int:
        return self.config.n_mfcc * 2

    def from_bytes(self, data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("Audio upload is empty")
        try:
            signal, source_rate = librosa.load(io.BytesIO(data), sr=None, mono=True)
        except Exception as exc:
            raise ValueError(f"Could not decode audio: {exc}") from exc
        return self.from_signal(signal, source_rate)

    def from_signal(self, signal, source_rate: int) -> np.ndarray:
        if source_rate <= 0:
            raise ValueError("Sample rate must be positive")
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        if signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError("Audio must contain finite, non-empty samples")
        if source_rate != self.config.sample_rate:
            signal = librosa.resample(
                signal, orig_sr=source_rate, target_sr=self.config.sample_rate
            )
        peak = float(np.max(np.abs(signal)))
        if peak <= 1e-8:
            raise ValueError("Audio is silent and cannot be normalized")
        signal = signal / peak

        target_length = int(self.config.sample_rate * self.config.target_seconds)
        if signal.size < target_length:
            signal = np.pad(signal, (0, target_length - signal.size))
        else:
            signal = signal[:target_length]

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        features = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1))).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("MFCC extraction produced invalid values")
        return features
