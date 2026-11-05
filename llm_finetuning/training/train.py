"""Command-line entry point for explicit QLoRA training runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import TrainingConfig
from .trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal language model with QLoRA and track it in MLflow"
    )
    parser.add_argument("--dataset", type=Path, help="Override LLM_DATASET_PATH")
    parser.add_argument("--base-model", help="Override LLM_BASE_MODEL")
    arguments = parser.parse_args()

    config = TrainingConfig.from_env()
    overrides = {}
    if arguments.dataset is not None:
        overrides["dataset_path"] = arguments.dataset
    if arguments.base_model is not None:
        overrides["base_model_id"] = arguments.base_model
    if overrides:
        config = replace(config, **overrides)

    result = train_model(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
