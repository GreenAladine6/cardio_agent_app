"""Merge a PEFT adapter and convert it to an Ollama-compatible GGUF model."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


SUPPORTED_QUANTIZATIONS = ("Q4_K_M", "Q5_K_M", "Q8_0")
DEFAULT_OLLAMA_MODEL_NAME = "cardio-llm"
OLLAMA_SYSTEM_PROMPT = (
    "Use only the supplied information. Do not make a medical diagnosis or "
    "invent clinical guidance. State when information is insufficient."
)
logger = logging.getLogger(__name__)


def _read_adapter_metadata(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    metadata_path = adapter_dir / "model_metadata.json"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"PEFT adapter config not found: {config_path}")
    if not any(adapter_dir.glob("adapter_model.*")):
        raise FileNotFoundError(f"PEFT adapter weights not found in {adapter_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Adapter model metadata not found: {metadata_path}; "
            "export requires the base model recorded by the training run"
        )

    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Adapter metadata is unreadable: {exc}") from exc
    if not isinstance(adapter_config, dict) or not isinstance(metadata, dict):
        raise ValueError("Adapter config and model metadata must contain JSON objects")

    base_model_id = metadata.get("base_model_id")
    adapter_base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ValueError("Adapter metadata does not identify its base_model_id")
    if (
        isinstance(adapter_base_model, str)
        and adapter_base_model.strip()
        and adapter_base_model.strip() != base_model_id.strip()
    ):
        raise ValueError(
            "Adapter config base model does not match model_metadata.json; "
            "refusing to merge the adapter with a different base model"
        )
    return metadata


def _validate_model_name(model_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", model_name):
        raise ValueError(
            "Ollama model name must be 1-64 characters using letters, numbers, "
            "dots, underscores, or hyphens"
        )
    return model_name


def _write_modelfile(path: Path, gguf_name: str, ollama_model_name: str) -> None:
    contents = (
        f"# Ollama model definition for {ollama_model_name}\n"
        f"FROM /models/{gguf_name}\n"
        "PARAMETER temperature 0.2\n"
        "PARAMETER num_ctx 4096\n"
        f'SYSTEM """\n{OLLAMA_SYSTEM_PROMPT}\n"""\n'
    )
    path.write_text(contents, encoding="utf-8")


def _load_export_dependencies() -> dict[str, Any]:
    try:
        import mlflow
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "GGUF export dependencies are missing. Install "
            "llm_finetuning/requirements-training.txt in the export environment."
        ) from exc
    return {
        "mlflow": mlflow,
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def export_adapter(
    *,
    adapter_dir: Path,
    output_dir: Path,
    llama_cpp_dir: Path,
    quantize_binary: Path | None = None,
    quantization: str = "Q4_K_M",
    ollama_model_name: str = DEFAULT_OLLAMA_MODEL_NAME,
    allow_model_download: bool = False,
) -> dict[str, str]:
    """Merge a local PEFT adapter and export a quantized GGUF plus Modelfile."""
    if quantization not in SUPPORTED_QUANTIZATIONS:
        raise ValueError(
            f"Unsupported GGUF quantization '{quantization}'; choose one of "
            f"{', '.join(SUPPORTED_QUANTIZATIONS)}"
        )
    ollama_model_name = _validate_model_name(ollama_model_name)
    adapter_dir = adapter_dir.resolve()
    output_dir = output_dir.resolve()
    llama_cpp_dir = llama_cpp_dir.resolve()
    adapter_metadata = _read_adapter_metadata(adapter_dir)

    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantizer = quantize_binary or (
        llama_cpp_dir / "build" / "bin" / "llama-quantize"
    )
    if not converter.is_file():
        raise FileNotFoundError(f"llama.cpp converter not found: {converter}")
    if not quantizer.is_file() or not os.access(quantizer, os.X_OK):
        raise FileNotFoundError(
            f"Executable llama.cpp quantizer not found: {quantizer}; "
            "build llama-quantize before exporting"
        )

    base_model_id = adapter_metadata["base_model_id"]
    run_id = str(adapter_metadata.get("run_id", "unknown"))
    gguf_name = f"{ollama_model_name}-{quantization.lower()}.gguf"
    gguf_path = output_dir / gguf_name
    modelfile_path = output_dir / "Modelfile"
    metadata_path = output_dir / "export_metadata.json"
    existing_outputs = [
        path for path in (gguf_path, modelfile_path, metadata_path) if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing export file(s): "
            + ", ".join(str(path) for path in existing_outputs)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dependencies = _load_export_dependencies()
    mlflow = dependencies["mlflow"]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv(
        "MLFLOW_EXPORT_EXPERIMENT_NAME", "Cardio-LLM-GGUF-Exports"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(
        "Merging PEFT adapter from training run %s for base model %s",
        run_id,
        base_model_id,
    )
    with mlflow.start_run(run_name=f"gguf-export-{quantization}") as run:
        mlflow.log_params(
            {
                "base_model_id": base_model_id,
                "adapter_training_run_id": run_id,
                "quantization": quantization,
                "export_format": "GGUF",
                "ollama_model_name": ollama_model_name,
            }
        )
        with tempfile.TemporaryDirectory(
            prefix=".gguf-export-", dir=output_dir
        ) as temporary_dir:
            staging = Path(temporary_dir)
            merged_dir = staging / "merged-model"
            merged_dir.mkdir()
            local_files_only = not allow_model_download

            model = dependencies["AutoModelForCausalLM"].from_pretrained(
                base_model_id,
                torch_dtype=dependencies["torch"].float16,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            )
            peft_model = dependencies["PeftModel"].from_pretrained(
                model,
                str(adapter_dir),
                is_trainable=False,
            )
            merged_model = peft_model.merge_and_unload(safe_merge=True)
            merged_model.save_pretrained(
                merged_dir,
                safe_serialization=True,
                max_shard_size="5GB",
            )
            tokenizer = dependencies["AutoTokenizer"].from_pretrained(
                str(adapter_dir),
                use_fast=True,
                local_files_only=True,
            )
            tokenizer.save_pretrained(merged_dir)
            del model, peft_model, merged_model, tokenizer

            unquantized_gguf = staging / f"{ollama_model_name}-f16.gguf"
            quantized_gguf = staging / gguf_name
            subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    str(merged_dir),
                    "--outfile",
                    str(unquantized_gguf),
                    "--outtype",
                    "f16",
                ],
                check=True,
            )
            subprocess.run(
                [
                    str(quantizer),
                    str(unquantized_gguf),
                    str(quantized_gguf),
                    quantization,
                ],
                check=True,
            )
            if (
                not quantized_gguf.is_file()
                or quantized_gguf.stat().st_size < 4
                or quantized_gguf.read_bytes()[:4] != b"GGUF"
            ):
                raise RuntimeError(
                    "llama.cpp quantization did not produce a valid, non-empty GGUF file"
                )

            _write_modelfile(staging / "Modelfile", gguf_name, ollama_model_name)
            export_metadata = {
                "base_model_id": base_model_id,
                "adapter_training_run_id": run_id,
                "adapter_format": "peft-lora-merged",
                "export_format": "GGUF",
                "quantization": quantization,
                "gguf_file": gguf_name,
                "ollama_model_name": ollama_model_name,
                "model_registry_name": adapter_metadata.get(
                    "registered_model_name", "Cardio-LLM-Adapter"
                ),
                "ollama_model_path": f"/models/{gguf_name}",
            }
            (staging / "export_metadata.json").write_text(
                json.dumps(export_metadata, indent=2),
                encoding="utf-8",
            )

            shutil.move(str(quantized_gguf), gguf_path)
            shutil.move(str(staging / "Modelfile"), modelfile_path)
            shutil.move(str(staging / "export_metadata.json"), metadata_path)

        for artifact in (gguf_path, modelfile_path, metadata_path):
            mlflow.log_artifact(str(artifact), artifact_path="gguf_export")

    logger.info(
        "Exported %s as %s (%s)",
        base_model_id,
        gguf_path.name,
        quantization,
    )
    return {
        "gguf_path": str(gguf_path),
        "modelfile_path": str(modelfile_path),
        "ollama_model_name": ollama_model_name,
        "quantization": quantization,
        "mlflow_run_id": run.info.run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a trained PEFT adapter and export a quantized GGUF model"
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=os.getenv("LLM_ADAPTER_PATH"),
        required=os.getenv("LLM_ADAPTER_PATH") is None,
        help="Saved PEFT adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("LLM_GGUF_OUTPUT_DIR", "models/gguf")),
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path(os.getenv("LLAMA_CPP_DIR", "vendor/llama.cpp")),
    )
    parser.add_argument(
        "--quantize-binary",
        type=Path,
        default=(
            Path(os.environ["LLAMA_QUANTIZE_BIN"])
            if os.getenv("LLAMA_QUANTIZE_BIN")
            else None
        ),
    )
    parser.add_argument(
        "--quantization",
        choices=SUPPORTED_QUANTIZATIONS,
        default=os.getenv("LLM_GGUF_QUANTIZATION", "Q4_K_M"),
    )
    parser.add_argument(
        "--ollama-model-name",
        default=os.getenv("OLLAMA_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME),
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Explicitly allow downloading the base model if it is not cached locally",
    )
    arguments = parser.parse_args()

    result = export_adapter(
        adapter_dir=arguments.adapter_dir,
        output_dir=arguments.output_dir,
        llama_cpp_dir=arguments.llama_cpp_dir,
        quantize_binary=arguments.quantize_binary,
        quantization=arguments.quantization,
        ollama_model_name=arguments.ollama_model_name,
        allow_model_download=arguments.allow_model_download,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
