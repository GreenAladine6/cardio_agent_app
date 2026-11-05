from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_finetuning.training.config import DEFAULT_MODEL_ID, TrainingConfig
from llm_finetuning.training.dataset_loader import InstructionExample
from llm_finetuning.training import trainer
from llm_finetuning.training.trainer import (
    _format_example,
    _register_adapter,
    _save_adapter_files,
    select_cuda_device,
)

pytestmark = pytest.mark.unit


def test_training_config_defaults_to_qwen_and_existing_mlflow():
    config = TrainingConfig.from_env({})

    assert config.base_model_id == DEFAULT_MODEL_ID
    assert config.base_model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.mlflow_tracking_uri == "http://localhost:5000"
    assert config.mlflow_experiment_name == "Cardio-LLM-FineTuning"
    assert config.dataset_path.name == "medical_instructions.jsonl"


def test_training_config_reads_environment_and_excludes_local_paths_from_mlflow():
    config = TrainingConfig.from_env(
        {
            "LLM_BASE_MODEL": "org/test-model",
            "LLM_DATASET_PATH": "/private/local/path.jsonl",
            "LLM_DATASET_ID": "authorized-demo",
            "LLM_OUTPUT_DIR": "/private/output",
            "MLFLOW_TRACKING_URI": "http://user:secret@mlflow:5000",
            "LLM_SEED": "0",
        }
    )

    assert config.base_model_id == "org/test-model"
    assert config.dataset_path == Path("/private/local/path.jsonl")
    assert config.seed == 0
    logged = config.to_mlflow_params()
    assert logged["dataset_identifier"] == "authorized-demo"
    assert "dataset_path" not in logged
    assert "output_dir" not in logged
    assert "mlflow_tracking_uri" not in logged
    assert "/private" not in str(logged)
    assert "secret" not in str(logged)


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"LLM_BATCH_SIZE": "0"}, "LLM_BATCH_SIZE must be greater than zero"),
        ({"LLM_LORA_DROPOUT": "1"}, "lora_dropout must be between"),
        ({"LLM_VALIDATION_FRACTION": "1"}, "must be less than 1"),
        ({"LLM_SEED": "-1"}, "LLM_SEED must not be negative"),
        ({"LLM_MIN_FREE_GPU_MEMORY_GB": "nan"}, "finite number"),
    ],
)
def test_training_config_rejects_invalid_environment(environ, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig.from_env(environ)


def test_training_prompt_uses_context_but_never_split_group():
    class StubTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is False
            self.messages = messages
            return "formatted"

    tokenizer = StubTokenizer()
    result = _format_example(
        InstructionExample(
            "Summarize this result",
            "Synthetic context",
            "Synthetic answer",
            split_group="private-record-id",
        ),
        tokenizer,
    )

    assert result == "formatted"
    assert tokenizer.messages[1]["content"] == (
        "Summarize this result\n\nContext:\nSynthetic context"
    )
    assert "private-record-id" not in str(tokenizer.messages)


def test_gpu_check_fails_before_model_loading_when_cuda_is_unavailable():
    class CpuOnlyCuda:
        @staticmethod
        def is_available():
            return False

    class CpuOnlyTorch:
        cuda = CpuOnlyCuda()

    with pytest.raises(RuntimeError, match="requires a working NVIDIA CUDA device"):
        select_cuda_device(CpuOnlyTorch, minimum_free_memory_gb=12)


def test_gpu_check_reports_insufficient_free_vram():
    class LowMemoryCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def mem_get_info(device_index):
            return 4 * 1024**3, 8 * 1024**3

    class LowMemoryTorch:
        cuda = LowMemoryCuda()

    with pytest.raises(RuntimeError, match="has 4.0 GiB free of 8.0 GiB"):
        select_cuda_device(LowMemoryTorch, minimum_free_memory_gb=12)


def test_training_rejects_missing_cuda_before_dataset_or_model_load(monkeypatch, tmp_path):
    class CpuOnlyCuda:
        @staticmethod
        def is_available():
            return False

    class CpuOnlyTorch:
        cuda = CpuOnlyCuda()

    class ModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("Model loading must not happen without CUDA")

    monkeypatch.setattr(
        trainer,
        "_load_training_dependencies",
        lambda: {"torch": CpuOnlyTorch, "AutoModelForCausalLM": ModelLoader},
    )
    config = TrainingConfig(dataset_path=tmp_path / "not-loaded.jsonl")

    with pytest.raises(RuntimeError, match="requires a working NVIDIA CUDA device"):
        trainer.train_model(config)


def test_saves_and_validates_adapter_tokenizer_and_safe_metadata(
    monkeypatch, tmp_path
):
    class StubTrainer:
        def save_model(self, output_dir):
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (output_path / "adapter_model.safetensors").write_bytes(b"adapter")

    class StubTokenizer:
        def save_pretrained(self, output_dir):
            (Path(output_dir) / "tokenizer_config.json").write_text(
                "{}", encoding="utf-8"
            )

    monkeypatch.setattr(
        trainer,
        "_save_loss_plot",
        lambda output_dir, history, metrics: (
            output_dir / "loss_curve.png"
        ).write_bytes(b"plot"),
    )
    config = TrainingConfig(
        mlflow_tracking_uri="http://user:secret@mlflow:5000",
        dataset_path=Path("/private/patient-data.jsonl"),
    )

    saved_path = _save_adapter_files(
        trainer=StubTrainer(),
        tokenizer=StubTokenizer(),
        output_dir=tmp_path / "adapter-run-123",
        config=config,
        run_id="run-123",
        training_metrics={"train_loss": 0.5},
        evaluation_metrics={"final_eval_loss": 0.4},
        history=[{"loss": 0.5, "epoch": 1.0}],
    )

    assert (saved_path / "adapter_model.safetensors").is_file()
    assert (saved_path / "tokenizer_config.json").is_file()
    assert (saved_path / "loss_curve.png").is_file()
    metadata = json.loads((saved_path / "model_metadata.json").read_text())
    assert metadata["adapter_format"] == "peft-lora"
    assert "secret" not in str(metadata)
    assert "/private" not in str(metadata)


def test_registers_adapter_artifact_as_a_versioned_registry_entry():
    class FakeRegistryClient:
        def __init__(self):
            self.models = []
            self.created_model = None
            self.version_request = None

        def search_registered_models(self):
            return self.models

        def create_registered_model(self, name):
            self.created_model = name
            self.models.append(SimpleNamespace(name=name))

        def create_model_version(self, **kwargs):
            self.version_request = kwargs
            return SimpleNamespace(version=3)

    client = FakeRegistryClient()
    fake_mlflow = SimpleNamespace(
        tracking=SimpleNamespace(MlflowClient=lambda **kwargs: client)
    )
    config = TrainingConfig(mlflow_tracking_uri="http://mlflow:5000")

    version = _register_adapter(fake_mlflow, config, "run-123")

    assert version == "3"
    assert client.created_model == "Cardio-LLM-Adapter"
    assert client.version_request["source"] == "runs:/run-123/lora_adapter"
    assert client.version_request["run_id"] == "run-123"
