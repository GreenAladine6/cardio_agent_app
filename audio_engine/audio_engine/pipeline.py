"""Audio decoding and PCG features shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 2000
    n_mfcc: int = 20
    n_fft: int = 256
    hop_length: int = 64
    target_seconds: float = 5.0
    min_duration_seconds: float = 0.25
    max_duration_seconds: float = 120.0

    def to_dict(self):
        return asdict(self)


class AudioPreprocessor:
    """Decode WAV audio, resample/normalize it, then summarize a fixed window.

    Multichannel recordings are downmixed by arithmetic channel mean. Recordings
    shorter than ``target_seconds`` are zero padded; longer ones are cropped.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    @property
    def feature_count(self) -> int:
        return self.config.n_mfcc * 2

    def from_bytes(self, data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("Audio upload is empty")
        try:
            with sf.SoundFile(io.BytesIO(data)) as audio_file:
                if audio_file.format != "WAV":
                    raise ValueError("Only WAV audio is supported")
                if audio_file.channels < 1 or audio_file.channels > 8:
                    raise ValueError("Audio must have between one and eight channels")
                duration = audio_file.frames / float(audio_file.samplerate)
                if duration < self.config.min_duration_seconds:
                    raise ValueError(f"Audio is shorter than {self.config.min_duration_seconds:g} seconds")
                if duration > self.config.max_duration_seconds:
                    raise ValueError(f"Audio exceeds {self.config.max_duration_seconds:g} seconds")
                source_rate = audio_file.samplerate
                signal = audio_file.read(dtype="float32", always_2d=False)
        except (RuntimeError, ValueError, OSError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("Audio "):
                raise
            raise ValueError("Audio is not a supported, readable WAV recording") from exc
        return self.from_signal(signal, source_rate)

    def from_signal(self, signal, source_rate: int) -> np.ndarray:
        if not isinstance(source_rate, (int, np.integer)) or source_rate <= 0:
            raise ValueError("Sample rate must be positive")
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 2:
            # soundfile yields frames x channels. Reject implausible channel layouts.
            if signal.shape[1] < 1 or signal.shape[1] > 8:
                raise ValueError("Audio must have between one and eight channels")
            signal = signal.mean(axis=1)
        elif signal.ndim != 1:
            raise ValueError("Audio waveform must be mono or frames-by-channels")
        if signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError("Audio must contain finite, non-empty samples")
        duration = signal.size / float(source_rate)
        if duration < self.config.min_duration_seconds:
            raise ValueError(f"Audio is shorter than {self.config.min_duration_seconds:g} seconds")
        if duration > self.config.max_duration_seconds:
            raise ValueError(f"Audio exceeds {self.config.max_duration_seconds:g} seconds")
        if source_rate != self.config.sample_rate:
            signal = librosa.resample(
                signal, orig_sr=source_rate, target_sr=self.config.sample_rate
            )
        peak = float(np.max(np.abs(signal)))
        if peak > 1e-8:
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
