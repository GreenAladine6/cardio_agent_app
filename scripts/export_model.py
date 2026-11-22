"""Run the required five-model benchmark with untouched validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_engine.train_benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "audio_engine" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "exported_models")
    parser.add_argument("--mlflow-uri", default=None)
    args = parser.parse_args()
    arrays = {name: np.load(args.data_dir / f"{name}.npy") for name in (
        "train_features", "train_labels", "val_features", "val_labels",
        "test_features", "test_labels", "train_mel", "val_mel", "test_mel",
    )}
    summary = run_benchmark(
        arrays["train_features"], arrays["train_labels"],
        arrays["val_features"], arrays["val_labels"],
        arrays["test_features"], arrays["test_labels"],
        X_train_mel=arrays["train_mel"],
        X_val_mel=arrays["val_mel"],
        X_test_mel=arrays["test_mel"],
        mlflow_uri=args.mlflow_uri,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
