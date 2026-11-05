"""GPU-checked QLoRA supervised fine-tuning with MLflow tracking."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import time
from typing import Any

from .config import TrainingConfig
from .dataset_loader import InstructionExample, load_dataset, split_dataset


logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    "Use only the supplied information. Do not make a medical diagnosis or "
    "invent clinical guidance. Clearly state when information is insufficient."
)


def select_cuda_device(torch: Any, minimum_free_memory_gb: float) -> dict[str, Any]:
    """Choose the CUDA device with most free memory or fail before model loading."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires a working NVIDIA CUDA device; none is available. "
            "Install a CUDA-enabled PyTorch build and configure the NVIDIA driver "
            "and container runtime before training."
        )

    memory_gb = 1024**3
    available_devices = []
    for device_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        available_devices.append((free_bytes, total_bytes, device_index))
    if not available_devices:
        raise RuntimeError("CUDA is available but no CUDA devices were reported")

    free_bytes, total_bytes, device_index = max(available_devices)
    free_memory_gb = free_bytes / memory_gb
    if free_memory_gb < minimum_free_memory_gb:
        raise RuntimeError(
            f"QLoRA requires at least {minimum_free_memory_gb:.1f} GiB of free GPU "
            f"memory by configuration; CUDA device {device_index} has "
            f"{free_memory_gb:.1f} GiB free of {total_bytes / memory_gb:.1f} GiB. "
            "Close other GPU workloads or lower the configured memory threshold "
            "only after checking the chosen model and sequence length."
        )

    torch.cuda.set_device(device_index)
    return {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "free_memory_gb": free_memory_gb,
        "total_memory_gb": total_bytes / memory_gb,
        "compute_dtype": (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        ),
    }


def _format_example(example: InstructionExample, tokenizer: Any) -> str:
    user_content = example.instruction
    if example.input:
        user_content = f"{user_content}\n\nContext:\n{example.input}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example.output},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _load_training_dependencies():
    try:
        import mlflow
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA training dependencies are missing. Install "
            "llm_finetuning/requirements-training.txt in a CUDA-capable "
            "environment before starting training."
        ) from exc
    return {
        "mlflow": mlflow,
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def _log_training_history(mlflow: Any, history: list[dict[str, Any]]) -> None:
    for index, record in enumerate(history):
        metrics = {
            key: float(value)
            for key, value in record.items()
            if key not in {"step", "epoch"}
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
        if metrics:
            step = int(record.get("step", index))
            mlflow.log_metrics(metrics, step=max(step, 0))


def _save_loss_plot(
    output_dir: Path,
    history: list[dict[str, Any]],
    training_metrics: dict[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training_points = [
        (float(record.get("epoch", record.get("step", index))), float(record["loss"]))
        for index, record in enumerate(history)
        if isinstance(record.get("loss"), (int, float))
        and math.isfinite(float(record["loss"]))
    ]
    if not training_points and "train_loss" in training_metrics:
        training_points = [
            (float(training_metrics.get("epoch", 1)), training_metrics["train_loss"])
        ]
    validation_points = [
        (
            float(record.get("epoch", record.get("step", index))),
            float(record["eval_loss"]),
        )
        for index, record in enumerate(history)
        if isinstance(record.get("eval_loss"), (int, float))
        and math.isfinite(float(record["eval_loss"]))
    ]
    if not training_points:
        raise RuntimeError("Training completed without a finite training loss to plot")

    figure, axis = plt.subplots()
    axis.plot(
        [point[0] for point in training_points],
        [point[1] for point in training_points],
        label="Training loss",
    )
    if validation_points:
        axis.plot(
            [point[0] for point in validation_points],
            [point[1] for point in validation_points],
            label="Validation loss",
        )
    axis.set_xlabel("Epoch / training step")
    axis.set_ylabel("Loss")
    axis.set_title("QLoRA training and validation loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(figure)


def _save_adapter_files(
    *,
    trainer: Any,
    tokenizer: Any,
    output_dir: Path,
    config: TrainingConfig,
    run_id: str,
    training_metrics: dict[str, float],
    evaluation_metrics: dict[str, float],
    history: list[dict[str, Any]],
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Adapter output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if not (output_dir / "adapter_config.json").is_file():
        raise RuntimeError("PEFT adapter save did not produce adapter_config.json")
    if not any(output_dir.glob("adapter_model.*")):
        raise RuntimeError("PEFT adapter save did not produce adapter model weights")
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "spiece.model",
        "vocab.json",
    )
    if not any((output_dir / filename).is_file() for filename in tokenizer_files):
        raise RuntimeError("Tokenizer save did not produce a recognized tokenizer file")

    (output_dir / "training_config.json").write_text(
        json.dumps(config.to_artifact_dict(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_log_history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(evaluation_metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "registered_model_name": config.registered_model_name,
                "base_model_id": config.base_model_id,
                "adapter_format": "peft-lora",
                "quantization": "4-bit NF4 with double quantization",
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "dataset_identifier": config.dataset_identifier,
                "training_metrics": training_metrics,
                "evaluation_metrics": evaluation_metrics,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _save_loss_plot(output_dir, history, training_metrics)
    return output_dir


def _register_adapter(mlflow: Any, config: TrainingConfig, run_id: str) -> str:
    client = mlflow.tracking.MlflowClient(
        tracking_uri=config.mlflow_tracking_uri
    )
    existing_models = client.search_registered_models()
    if not any(model.name == config.registered_model_name for model in existing_models):
        client.create_registered_model(config.registered_model_name)

    version = client.create_model_version(
        name=config.registered_model_name,
        source=f"runs:/{run_id}/lora_adapter",
        run_id=run_id,
        description=(
            "PEFT LoRA adapter artifact. Load it with its recorded base model "
            "and PEFT; it is not a standalone full-model checkpoint."
        ),
        tags={
            "artifact_format": "peft-lora-adapter",
            "base_model_id": config.base_model_id,
        },
    )
    return str(version.version)


def train_model(config: TrainingConfig) -> dict[str, Any]:
    """Run one explicit QLoRA training experiment and log non-sensitive metadata."""
    dependencies = _load_training_dependencies()
    torch = dependencies["torch"]
    gpu = select_cuda_device(torch, config.minimum_free_gpu_memory_gb)

    examples = load_dataset(config.dataset_path)
    split = split_dataset(
        examples,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    mlflow = dependencies["mlflow"]
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    logger.info(
        "Starting QLoRA run for %s on CUDA device %s with %d training and %d validation records",
        config.base_model_id,
        gpu["device_name"],
        len(split.train),
        len(split.validation),
    )
    run_started = time.perf_counter()
    with mlflow.start_run(run_name="qlora-finetuning") as run:
        mlflow.log_params(config.to_mlflow_params())
        mlflow.log_params(
            {
                "quantization": "4-bit NF4 with double quantization",
                "quantization_bits": 4,
                "quantization_type": "nf4",
                "double_quantization": True,
                "gradient_checkpointing": True,
                "registered_model_name": config.registered_model_name,
                "compute_dtype": str(gpu["compute_dtype"]),
                "gpu_name": gpu["device_name"],
                "gpu_total_memory_gb": round(gpu["total_memory_gb"], 2),
                "train_examples": len(split.train),
                "validation_examples": len(split.validation),
            }
        )
        mlflow.log_dict(config.to_artifact_dict(), "training/training_config.json")

        model_id = config.base_model_id
        tokenizer = dependencies["AutoTokenizer"].from_pretrained(
            model_id,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        compute_dtype = gpu["compute_dtype"]
        quantization_config = dependencies["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            device_map={"": gpu["device_index"]},
        )
        model.config.use_cache = False
        model = dependencies["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=True,
        )
        adapter_config = dependencies["LoraConfig"](
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = dependencies["get_peft_model"](model, adapter_config)

        train_dataset = dependencies["Dataset"].from_list(
            [{"text": _format_example(example, tokenizer)} for example in split.train]
        )
        validation_dataset = dependencies["Dataset"].from_list(
            [
                {"text": _format_example(example, tokenizer)}
                for example in split.validation
            ]
        )
        adapter_dir = config.output_dir / run.info.run_id
        training_args = dependencies["SFTConfig"](
            output_dir=str(adapter_dir),
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_seq_length=config.max_seq_length,
            dataset_text_field="text",
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=10,
            gradient_checkpointing=True,
            bf16=compute_dtype == torch.bfloat16,
            fp16=compute_dtype == torch.float16,
            optim="paged_adamw_8bit",
            report_to=[],
            seed=config.seed,
        )
        trainer = dependencies["SFTTrainer"](
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
        )
        result = trainer.train()
        runtime_seconds = time.perf_counter() - run_started
        training_metrics = {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        training_metrics["training_runtime_seconds"] = runtime_seconds
        mlflow.log_metrics(training_metrics)
        evaluation_metrics = {
            key: float(value)
            for key, value in trainer.evaluate(metric_key_prefix="final_eval").items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        mlflow.log_metrics(
            evaluation_metrics,
            step=trainer.state.global_step,
        )
        history = list(trainer.state.log_history)
        _log_training_history(mlflow, history)

        adapter_dir = _save_adapter_files(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=adapter_dir,
            config=config,
            run_id=run.info.run_id,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            history=history,
        )
        mlflow.log_dict(
            evaluation_metrics,
            "evaluation/evaluation_results.json",
        )
        mlflow.log_dict(history, "training/trainer_log_history.json")
        mlflow.log_artifacts(str(adapter_dir), artifact_path="lora_adapter")
        model_version = _register_adapter(mlflow, config, run.info.run_id)
        logger.info(
            "QLoRA training completed for %s in %.1f seconds; registered version %s",
            config.base_model_id,
            runtime_seconds,
            model_version,
        )
        return {
            "experiment_name": config.mlflow_experiment_name,
            "base_model_id": config.base_model_id,
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_metrics,
            "adapter_path": str(adapter_dir),
            "registered_model_name": config.registered_model_name,
            "registered_model_version": model_version,
        }
