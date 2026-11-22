from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.signal as scipy_signal

try:
    import yaml
except Exception:  # pragma: no cover - optional at runtime
    yaml = None


def _default_config() -> dict[str, Any]:
    return {
        "feature": {
            "sample_rate": 2000,
            "window_seconds": 5.0,
            "overlap_ratio": 0.5,
            "bandpass": {"low_hz": 25, "high_hz": 400, "order": 4},
            "normalization": "peak",
            "energy_threshold_db": -45.0,
        },
        "split": {"train_fraction": 0.70, "val_fraction": 0.15, "test_fraction": 0.15, "random_state": 42},
        "label": {"normal": 0, "pathological": 1},
    }


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration, falling back to the defaults if the file is missing."""
    yaml_path = Path(config_path) if config_path else Path(__file__).with_name("config.yaml")
    if yaml_path.exists() and yaml and hasattr(yaml, "safe_load"):
        with yaml_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            return loaded
    return _default_config()


def harmonize_labels(df: pd.DataFrame, label_column: str = "original_label") -> pd.DataFrame:
    """Map raw labels to binary 0/1 while preserving the original label text."""
    if label_column not in df.columns:
        raise KeyError(f"Column '{label_column}' was not found in the metadata table")

    output = df.copy()
    output["original_label"] = output[label_column].astype(str).str.strip().str.lower()
    label_map = {
        "normal": 0,
        "healthy": 0,
        "pathological": 1,
        "abnormal": 1,
        "murmur": 1,
        "extrahls": 1,
        "extrasystole": 1,
        "benign": 1,
    }
    output["binary_label"] = output["original_label"].map(label_map).astype("Int64")
    output = output[output["binary_label"].notna()].copy()
    output = output[output["original_label"] != "artifact"].copy()
    output = output[output["original_label"] != ""].copy()
    output = output[output["original_label"] != "nan"].copy()
    return output.reset_index(drop=True)


def resample_audio(signal: np.ndarray, orig_sr: int, target_sr: int = 2000, *, axis: int = -1) -> np.ndarray:
    """Resample audio to a target sampling rate while preserving float32 dtype."""
    audio = np.asarray(signal, dtype=np.float32)
    if orig_sr <= 0:
        raise ValueError("Original sample rate must be positive")
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32)
    resampled = signal.resample(audio, orig_sr, target_sr) if hasattr(signal, "resample") else None
    if resampled is not None:
        return np.asarray(resampled, dtype=np.float32)

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = scipy_signal.resample_poly(audio, up, down)
    return np.asarray(resampled, dtype=np.float32)


def apply_bandpass_filter(signal: np.ndarray, sample_rate: int, low_hz: float = 25.0, high_hz: float = 400.0, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter to isolate heart-sound frequencies."""
    if signal.size == 0:
        return signal.astype(np.float32)
    nyquist = 0.5 * sample_rate
    low = low_hz / nyquist
    high = high_hz / nyquist
    if high >= 1.0:
        high = 0.99
    if low <= 0 or high <= low:
        raise ValueError("Band-pass cutoff frequencies are invalid")
    b, a = scipy_signal.butter(order, [low, high], btype="bandpass")
    filtered = scipy_signal.filtfilt(b, a, np.asarray(signal, dtype=np.float32))
    return np.asarray(filtered, dtype=np.float32)


def normalize_audio(signal: np.ndarray, method: str = "peak") -> np.ndarray:
    """Normalize an audio array by peak amplitude or z-score."""
    data = np.asarray(signal, dtype=np.float32)
    if data.size == 0:
        return data
    if method == "peak":
        scale = np.max(np.abs(data))
        if scale < 1e-8:
            return data.astype(np.float32)
        return data / scale
    if method == "zscore":
        mean = float(np.mean(data))
        std = float(np.std(data))
        if std < 1e-8:
            return data.astype(np.float32)
        return (data - mean) / std
    raise ValueError(f"Unsupported normalization mode: {method}")


def segment_audio(
    signal: np.ndarray,
    sample_rate: int = 2000,
    window_seconds: float = 5.0,
    overlap: float = 0.5,
    energy_threshold_db: float = -45.0,
    pad: bool = True,
) -> list[np.ndarray]:
    """Split an audio recording into overlapping fixed-length windows and discard near-silent segments."""
    audio = np.asarray(signal, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return []
    if sample_rate <= 0 or window_seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError("Sample rate, window duration, or overlap is invalid")
    if not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite samples")
    window_length = max(1, int(round(sample_rate * window_seconds)))
    hop_length = max(1, int(round(window_length * (1.0 - overlap))))

    starts = list(range(0, max(1, audio.size - window_length + 1), hop_length))
    if pad and starts[-1] + window_length < audio.size:
        trailing_start = starts[-1] + hop_length
        if trailing_start < audio.size:
            starts.append(trailing_start)

    windows: list[np.ndarray] = []
    for start in starts:
        segment = audio[start : start + window_length]
        if segment.size < window_length:
            segment = np.pad(segment, (0, window_length - segment.size), mode="constant")
        energy_db = 10.0 * np.log10(np.mean(np.square(segment)) + 1e-10)
        if energy_db < energy_threshold_db:
            continue
        windows.append(segment.astype(np.float32))

    if not windows:
        padded = np.pad(audio[:window_length], (0, max(0, window_length - audio.size)), mode="constant")
        windows.append(padded.astype(np.float32))
    return windows


def make_grouped_split(
    df: pd.DataFrame,
    *,
    target_col: str = "binary_label",
    group_col: str = "recording_id",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    test_size: float | None = None,
) -> pd.DataFrame:
    """Assign grouped records to train/validation/test splits without leakage across a recording."""
    if test_size is not None:
        test_fraction = float(test_size)
        val_fraction = 0.0
        train_fraction = max(0.0, 1.0 - test_fraction)
    if not {target_col, group_col}.issubset(df.columns):
        raise KeyError(f"'{target_col}' and '{group_col}' must both exist in the dataframe")
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    fractions = {"train": train_fraction, "val": val_fraction, "test": test_fraction}
    active_splits = [name for name, fraction in fractions.items() if fraction > 0]
    if df[target_col].isna().any() or df[group_col].isna().any():
        raise ValueError("Target labels and group identifiers must be present for grouped splitting")
    group_sizes = df.groupby(group_col, sort=False).size()
    if len(group_sizes) < len(active_splits):
        raise ValueError("There are fewer unique groups than requested non-empty splits")

    labels = sorted(pd.unique(df[target_col]).tolist())
    group_vectors = {
        group_id: np.asarray(
            [int((group[target_col] == label).sum()) for label in labels], dtype=np.float64
        )
        for group_id, group in df.groupby(group_col, sort=False)
    }
    rng = np.random.default_rng(seed)
    group_ids = list(group_vectors)
    rng.shuffle(group_ids)
    random_order = {group_id: index for index, group_id in enumerate(group_ids)}
    group_ids.sort(key=lambda group_id: (-int(group_sizes[group_id]), random_order[group_id]))
    total_by_label = np.sum(list(group_vectors.values()), axis=0)
    targets = {
        name: total_by_label * fraction
        for name, fraction in fractions.items()
    }
    assigned = {name: np.zeros(len(labels), dtype=np.float64) for name in fractions}
    assigned_groups = {name: 0 for name in fractions}
    split_map = {}

    def balance_cost(counts: dict[str, np.ndarray], group_counts: dict[str, int]) -> float:
        class_cost = 0.0
        for name in active_splits:
            scale = np.maximum(targets[name], 1.0)
            class_cost += float(np.sum(((counts[name] - targets[name]) / scale) ** 2))
        size_cost = sum(
            ((group_counts[name] - len(group_ids) * fractions[name]) / max(len(group_ids) * fractions[name], 1.0)) ** 2
            for name in active_splits
        )
        return class_cost + float(size_cost)

    for group_position, group_id in enumerate(group_ids):
        eligible_splits = active_splits
        if group_position < len(active_splits):
            # Seed every requested split with one whole recording/patient group.
            eligible_splits = [active_splits[group_position]]
        best_split = min(
            eligible_splits,
            key=lambda name: balance_cost(
                assigned | {name: assigned[name] + group_vectors[group_id]},
                assigned_groups | {name: assigned_groups[name] + 1},
            ),
        )
        assigned[best_split] += group_vectors[group_id]
        assigned_groups[best_split] += 1
        split_map[group_id] = best_split

    result = df.copy()
    result["split"] = result[group_col].map(split_map)
    if result["split"].isna().any():
        raise RuntimeError("Grouped split did not assign every record")
    return result.reset_index(drop=True)
