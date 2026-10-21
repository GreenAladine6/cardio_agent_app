"""Small, dependency-free tools for inspecting phonocardiogram (PCG) WAV files.

The measurements in this module describe audio only. They are not a medical
diagnosis, and the event/heart-rate estimate is intentionally conservative.
"""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import io
import math
import os
import struct
import wave
from typing import BinaryIO

__all__ = [
    "AudioEngine",
    "AudioEngineError",
    "AudioMetadata",
    "AudioAnalysis",
    "read_wav",
]


class AudioEngineError(ValueError):
    """Raised when an audio file cannot be safely decoded or analyzed."""


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class AudioAnalysis:
    metadata: AudioMetadata
    peak_amplitude: float
    rms_amplitude: float
    clipped_fraction: float
    zero_crossing_rate: float
    filtered_rms_amplitude: float
    detected_event_count: int
    estimated_heart_rate_bpm: float | None
    event_times_seconds: tuple[float, ...]

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation of the analysis."""
        return asdict(self)


def _source_stream(source: str | os.PathLike | bytes | bytearray | memoryview | BinaryIO):
    if isinstance(source, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(source))
    if isinstance(source, (str, os.PathLike)):
        try:
            return open(source, "rb")
        except OSError as exc:
            raise AudioEngineError(f"Could not open WAV file: {exc}") from exc
    if hasattr(source, "read"):
        return source
    raise TypeError("source must be a path, bytes, or binary file-like object")


def _decode_pcm(raw: bytes, width: int) -> list[float]:
    """Decode little-endian PCM to normalized floats in [-1, 1]."""
    if width == 1:  # WAV 8-bit PCM is unsigned.
        return [(sample - 128) / 128.0 for sample in raw]
    if width == 2:
        values = array("h")
        values.frombytes(raw)
        if struct.pack("=H", 1) != struct.pack("<H", 1):
            values.byteswap()
        return [sample / 32768.0 for sample in values]
    if width == 3:
        usable = len(raw) - len(raw) % 3
        samples = []
        for offset in range(0, usable, 3):
            value = raw[offset] | raw[offset + 1] << 8 | raw[offset + 2] << 16
            if value & 0x800000:
                value -= 1 << 24
            samples.append(value / 8388608.0)
        return samples
    if width == 4:
        values = array("i")
        values.frombytes(raw)
        if struct.pack("=I", 1) != struct.pack("<I", 1):
            values.byteswap()
        return [sample / 2147483648.0 for sample in values]
    raise AudioEngineError(f"Unsupported PCM sample width: {width} bytes")


def read_wav(source, *, max_duration_seconds: float = 300.0):
    """Read PCM WAV audio and return ``(mono_samples, metadata)``.

    Multi-channel inputs are downmixed by averaging channels. Only uncompressed
    PCM at 8, 16, 24, or 32 bits is accepted.
    """
    stream = _source_stream(source)
    close_stream = stream is not source
    try:
        try:
            with wave.open(stream, "rb") as wav:
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                frames = wav.getnframes()
                if wav.getcomptype() != "NONE":
                    raise AudioEngineError("Compressed WAV audio is not supported")
                if channels < 1 or channels > 8:
                    raise AudioEngineError("WAV must contain between 1 and 8 channels")
                if width not in (1, 2, 3, 4):
                    raise AudioEngineError("WAV must use 8, 16, 24, or 32-bit PCM")
                if rate < 1000 or rate > 192000:
                    raise AudioEngineError("WAV sample rate must be between 1 kHz and 192 kHz")
                if frames == 0:
                    raise AudioEngineError("WAV contains no audio frames")
                duration = frames / rate
                if duration > max_duration_seconds:
                    raise AudioEngineError(
                        f"WAV duration exceeds the {max_duration_seconds:g} second limit"
                    )
                decoded = _decode_pcm(wav.readframes(frames), width)
        except (wave.Error, EOFError, struct.error) as exc:
            raise AudioEngineError(f"Invalid or truncated WAV file: {exc}") from exc
    finally:
        if close_stream:
            stream.close()

    if len(decoded) != frames * channels:
        raise AudioEngineError("WAV frame data is incomplete")
    if channels == 1:
        mono = decoded
    else:
        mono = [sum(decoded[i:i + channels]) / channels for i in range(0, len(decoded), channels)]
    metadata = AudioMetadata(rate, channels, width, frames, duration)
    return mono, metadata


def _biquad_filter(samples: list[float], rate: int, low_hz: float, high_hz: float) -> list[float]:
    """Cascade a second-order Butterworth high-pass and low-pass filter."""
    def apply(data, cutoff, highpass):
        omega = 2.0 * math.pi * cutoff / rate
        cosine = math.cos(omega)
        alpha = math.sin(omega) / math.sqrt(2.0)
        if highpass:
            b0, b1, b2 = (1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2
        else:
            b0, b1, b2 = (1 - cosine) / 2, 1 - cosine, (1 - cosine) / 2
        a0, a1, a2 = 1 + alpha, -2 * cosine, 1 - alpha
        b0, b1, b2, a1, a2 = b0/a0, b1/a0, b2/a0, a1/a0, a2/a0
        x1 = x2 = y1 = y2 = 0.0
        output = []
        for x0 in data:
            y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2
            output.append(y0)
            x2, x1, y2, y1 = x1, x0, y1, y0
        return output

    nyquist = rate / 2
    low_hz = min(max(20.0, low_hz), nyquist * 0.45)
    high_hz = min(max(low_hz + 1, high_hz), nyquist * 0.9)
    return apply(apply(samples, low_hz, True), high_hz, False)


class AudioEngine:
    """Analyze PCM WAV recordings for basic PCG signal characteristics."""

    def __init__(self, *, max_duration_seconds: float = 300.0,
                 band_hz: tuple[float, float] = (25.0, 400.0)):
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        if len(band_hz) != 2 or band_hz[0] <= 0 or band_hz[1] <= band_hz[0]:
            raise ValueError("band_hz must be a positive (low, high) frequency pair")
        self.max_duration_seconds = float(max_duration_seconds)
        self.band_hz = (float(band_hz[0]), float(band_hz[1]))

    def analyze(self, source) -> AudioAnalysis:
        """Analyze a WAV path, bytes value, or binary file-like object."""
        samples, metadata = read_wav(source, max_duration_seconds=self.max_duration_seconds)
        count = len(samples)
        peak = max(abs(sample) for sample in samples)
        rms = math.sqrt(sum(sample * sample for sample in samples) / count)
        clipped = sum(abs(sample) >= 0.999 for sample in samples) / count
        crossings = sum((samples[i - 1] < 0) != (samples[i] < 0) for i in range(1, count))
        zcr = crossings / max(1, count - 1)

        filtered = _biquad_filter(samples, metadata.sample_rate, *self.band_hz)
        filtered_rms = math.sqrt(sum(sample * sample for sample in filtered) / count)
        event_times = self._detect_events(filtered, metadata.sample_rate)
        intervals = [b - a for a, b in zip(event_times, event_times[1:])]
        heart_rate = None
        if len(intervals) >= 2:
            heart_rate = round(60.0 / (sum(intervals) / len(intervals)), 1)
        return AudioAnalysis(
            metadata=metadata,
            peak_amplitude=round(peak, 6),
            rms_amplitude=round(rms, 6),
            clipped_fraction=round(clipped, 6),
            zero_crossing_rate=round(zcr, 6),
            filtered_rms_amplitude=round(filtered_rms, 6),
            detected_event_count=len(event_times),
            estimated_heart_rate_bpm=heart_rate,
            event_times_seconds=tuple(round(t, 3) for t in event_times),
        )

    @staticmethod
    def _detect_events(samples: list[float], rate: int) -> list[float]:
        # Envelope from short-term RMS windows; keep peaks at least 300 ms apart.
        window = max(1, int(rate * 0.05))
        hop = max(1, window // 2)
        envelope = []
        for start in range(0, len(samples) - window + 1, hop):
            chunk = samples[start:start + window]
            envelope.append((start / rate, math.sqrt(sum(x*x for x in chunk) / window)))
        if len(envelope) < 3:
            return []
        floor = sorted(value for _, value in envelope)[len(envelope) // 2]
        ceiling = max(value for _, value in envelope)
        threshold = max(floor * 1.8, ceiling * 0.18, 1e-5)
        min_gap = 0.30
        events = []
        event_levels = []
        for i in range(1, len(envelope) - 1):
            time, value = envelope[i]
            if value >= threshold and value >= envelope[i - 1][1] and value > envelope[i + 1][1]:
                if not events or time - events[-1] >= min_gap:
                    events.append(time)
                    event_levels.append(value)
                else:
                    # Prefer the stronger peak when two candidates share the refractory period.
                    if value > event_levels[-1]:
                        events[-1] = time
                        event_levels[-1] = value
        return events
