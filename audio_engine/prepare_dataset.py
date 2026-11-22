"""Create recording-grouped MFCC and log-Mel arrays from WAV folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from audio_engine.audio_engine.pipeline import AudioPreprocessor, FeatureConfig
from audio_engine.data_prep import make_grouped_split, normalize_audio, resample_audio, segment_audio
from audio_engine.features import compute_log_mel


LABELS = {"normal": 0, "abnormal": 1}


def _recordings(dataset_root: Path, metadata_path: Path | None) -> pd.DataFrame:
    rows = []
    for label_name, label in LABELS.items():
        label_dir = dataset_root / label_name
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory: {label_dir}")
        for path in sorted(item for item in label_dir.rglob("*") if item.is_file() and item.suffix.lower() == ".wav"):
            rows.append({
                "path": str(path.resolve()),
                "relative_path": path.relative_to(dataset_root).as_posix(),
                "recording_id": path.stem,
                "source": "unspecified",
                "original_label": label_name,
                "binary_label": label,
            })
    if not rows:
        raise ValueError(f"No WAV files found beneath {dataset_root}")
    records = pd.DataFrame(rows)

    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path)
        required = {"relative_path"}
        if not required.issubset(metadata.columns):
            raise ValueError("Metadata CSV must include a relative_path column")
        if metadata["relative_path"].duplicated().any():
            raise ValueError("Metadata CSV contains duplicate relative_path values")
        metadata = metadata.set_index("relative_path")
        for column in ("recording_id", "patient_id", "source"):
            if column in metadata.columns:
                mapped = records["relative_path"].map(metadata[column])
                if column == "patient_id":
                    records[column] = mapped
                else:
                    records[column] = mapped.fillna(records.get(column, "unspecified"))
        if "group_id" in metadata.columns:
            records["group_id"] = records["relative_path"].map(metadata["group_id"])

    if "group_id" not in records:
        if "patient_id" in records:
            patient_ids = records["patient_id"]
            records["group_id"] = patient_ids.where(patient_ids.notna(), records["recording_id"])
        else:
            records["group_id"] = records["recording_id"]
    if records["group_id"].isna().any() or (records["group_id"].astype(str).str.strip() == "").any():
        raise ValueError("Group identifiers must be non-empty when supplied")
    return records


def prepare_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    seed: int = 42,
) -> Path:
    root = Path(dataset_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty export directory: {destination}")
    records = _recordings(root, Path(metadata_csv).resolve() if metadata_csv else None)
    split_records = make_grouped_split(
        records,
        target_col="binary_label",
        group_col="group_id",
        seed=seed,
    )
    preprocessor = AudioPreprocessor(FeatureConfig())
    split_rows: dict[str, list[tuple[np.ndarray, np.ndarray, int, str]]] = {
        "train": [], "val": [], "test": [],
    }
    for row in split_records.itertuples(index=False):
        waveform, sample_rate = sf.read(row.path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if not np.isfinite(waveform).all():
            raise ValueError(f"Recording contains non-finite samples: {row.relative_path}")
        waveform = normalize_audio(resample_audio(waveform, int(sample_rate), 2000))
        windows = segment_audio(
            waveform,
            sample_rate=2000,
            window_seconds=5.0,
            overlap=0.5,
            energy_threshold_db=-45.0,
        )
        for window_index, window in enumerate(windows):
            window = normalize_audio(window)
            mfcc = preprocessor.from_signal(window, 2000)
            mel = compute_log_mel(window, sr=2000)[0]
            split_rows[row.split].append((mfcc, mel, int(row.binary_label), f"{row.recording_id}:{window_index}"))

    destination.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_rows.items():
        if not rows:
            raise ValueError(f"The {split_name} split has no audio windows; add more recording groups")
        np.save(destination / f"{split_name}_features.npy", np.stack([item[0] for item in rows]))
        np.save(destination / f"{split_name}_mel.npy", np.stack([item[1] for item in rows]))
        np.save(destination / f"{split_name}_labels.npy", np.asarray([item[2] for item in rows], dtype=np.int64))
        pd.DataFrame({
            "window_id": [item[3] for item in rows],
            "label": [item[2] for item in rows],
        }).to_csv(destination / f"{split_name}_manifest.csv", index=False)

    split_records[["relative_path", "recording_id", "group_id", "source", "original_label", "binary_label", "split"]].to_csv(
        destination / "recordings_manifest.csv", index=False
    )
    (destination / "preprocessing.json").write_text(json.dumps({
        "sample_rate": 2000,
        "target_seconds": 5.0,
        "window_seconds": 5.0,
        "overlap_ratio": 0.5,
        "normalization": "peak",
        "mfcc_features": "mean_and_standard_deviation_of_20_mfcc_coefficients",
        "mfcc_feature_count": preprocessor.feature_count,
        "mel": {"n_mels": 64, "n_fft": 256, "hop_length": 64, "fmin": 20, "fmax": 1000},
        "label_mapping": LABELS,
        "seed": seed,
        "grouping": "patient_id or recording_id from optional metadata CSV; recording filename stem otherwise",
        "dataset_provenance": "user-supplied; not inferred from folder names",
    }, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Folder containing normal/ and abnormal/ subfolders")
    parser.add_argument("--metadata-csv", type=Path, help="Optional metadata CSV with relative_path and recording_id/patient_id/group_id/source columns")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare_dataset(args.dataset_root, args.output_dir, metadata_csv=args.metadata_csv, seed=args.seed))


if __name__ == "__main__":
    main()
