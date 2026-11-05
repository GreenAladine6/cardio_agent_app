"""Validated, environment-driven configuration for QLoRA fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Mapping

from .dataset_loader import DEFAULT_DATASET_PATH


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _non_negative_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must not be negative")
    return parsed


def _positive_float(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return parsed


@dataclass(frozen=True)
class TrainingConfig:
    base_model_id: str = DEFAULT_MODEL_ID
    dataset_path: Path = DEFAULT_DATASET_PATH
    dataset_identifier: str = "configured-jsonl"
    output_dir: Path = Path("models/adapters/qwen2.5-7b-instruct")
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "Cardio-LLM-FineTuning"
    registered_model_name: str = "Cardio-LLM-Adapter"
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    validation_fraction: float = 0.2
    seed: int = 42
    minimum_free_gpu_memory_gb: float = 12.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    def __post_init__(self) -> None:
        for field_name in (
            "base_model_id",
            "dataset_identifier",
            "mlflow_tracking_uri",
            "mlflow_experiment_name",
            "registered_model_name",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and greater than zero")
        if self.num_train_epochs <= 0 or not math.isfinite(self.num_train_epochs):
            raise ValueError("num_train_epochs must be finite and greater than zero")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be greater than zero")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be greater than zero")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be greater than zero")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_rank and lora_alpha must be greater than zero")
        if not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be between 0 (inclusive) and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if (
            not math.isfinite(self.minimum_free_gpu_memory_gb)
            or self.minimum_free_gpu_memory_gb <= 0
        ):
            raise ValueError(
                "minimum_free_gpu_memory_gb must be finite and greater than zero"
            )
        if not self.target_modules or any(not module.strip() for module in self.target_modules):
            raise ValueError("target_modules must contain non-empty module names")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> TrainingConfig:
        env = os.environ if environ is None else environ
        target_modules = tuple(
            module.strip()
            for module in env.get(
                "LLM_LORA_TARGET_MODULES", ",".join(DEFAULT_TARGET_MODULES)
            ).split(",")
            if module.strip()
        )
        validation_fraction = _positive_float(
            "LLM_VALIDATION_FRACTION", env.get("LLM_VALIDATION_FRACTION", "0.2")
        )
        if validation_fraction >= 1:
            raise ValueError("LLM_VALIDATION_FRACTION must be less than 1")

        return cls(
            base_model_id=env.get("LLM_BASE_MODEL", DEFAULT_MODEL_ID),
            dataset_path=Path(
                env.get("LLM_DATASET_PATH", str(DEFAULT_DATASET_PATH))
            ),
            dataset_identifier=env.get("LLM_DATASET_ID", "configured-jsonl"),
            output_dir=Path(
                env.get(
                    "LLM_OUTPUT_DIR",
                    "models/adapters/qwen2.5-7b-instruct",
                )
            ),
            mlflow_tracking_uri=env.get(
                "MLFLOW_TRACKING_URI", "http://localhost:5000"
            ),
            mlflow_experiment_name=env.get(
                "MLFLOW_EXPERIMENT_NAME", "Cardio-LLM-FineTuning"
            ),
            registered_model_name=env.get(
                "MLFLOW_REGISTERED_MODEL_NAME", "Cardio-LLM-Adapter"
            ),
            learning_rate=_positive_float(
                "LLM_LEARNING_RATE", env.get("LLM_LEARNING_RATE", "0.0002")
            ),
            num_train_epochs=_positive_float(
                "LLM_NUM_TRAIN_EPOCHS", env.get("LLM_NUM_TRAIN_EPOCHS", "3")
            ),
            per_device_train_batch_size=_positive_int(
                "LLM_BATCH_SIZE", env.get("LLM_BATCH_SIZE", "1")
            ),
            gradient_accumulation_steps=_positive_int(
                "LLM_GRADIENT_ACCUMULATION_STEPS",
                env.get("LLM_GRADIENT_ACCUMULATION_STEPS", "8"),
            ),
            max_seq_length=_positive_int(
                "LLM_MAX_SEQ_LENGTH", env.get("LLM_MAX_SEQ_LENGTH", "2048")
            ),
            lora_rank=_positive_int("LLM_LORA_RANK", env.get("LLM_LORA_RANK", "16")),
            lora_alpha=_positive_int(
                "LLM_LORA_ALPHA", env.get("LLM_LORA_ALPHA", "32")
            ),
            lora_dropout=_positive_float(
                "LLM_LORA_DROPOUT", env.get("LLM_LORA_DROPOUT", "0.05")
            ),
            validation_fraction=validation_fraction,
            seed=_non_negative_int("LLM_SEED", env.get("LLM_SEED", "42")),
            minimum_free_gpu_memory_gb=_positive_float(
                "LLM_MIN_FREE_GPU_MEMORY_GB",
                env.get("LLM_MIN_FREE_GPU_MEMORY_GB", "12"),
            ),
            target_modules=target_modules,
        )

    def to_mlflow_params(self) -> dict[str, str | int | float]:
        """Return safe metadata only; never include dataset paths or records."""
        params = asdict(self)
        params.pop("dataset_path")
        params.pop("output_dir")
        params.pop("mlflow_tracking_uri")
        params["target_modules"] = ",".join(self.target_modules)
        params["dataset_identifier"] = self.dataset_identifier
        return {
            key: value
            for key, value in params.items()
            if isinstance(value, (str, int, float))
        }

    def to_artifact_dict(self) -> dict[str, str | int | float | list[str]]:
        return {
            **self.to_mlflow_params(),
            "target_modules": list(self.target_modules),
        }
