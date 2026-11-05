from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_finetuning.export import export_gguf

pytestmark = pytest.mark.unit


def create_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
    (adapter_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
                "run_id": "training-run-1",
                "registered_model_name": "Cardio-LLM-Adapter",
            }
        ),
        encoding="utf-8",
    )


def test_export_rejects_unsupported_quantization_before_loading_dependencies(
    tmp_path, monkeypatch
):
    adapter_dir = tmp_path / "adapter"
    create_adapter(adapter_dir)
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()

    def dependencies_must_not_load():
        raise AssertionError("Export dependencies must not load for invalid options")

    monkeypatch.setattr(
        export_gguf, "_load_export_dependencies", dependencies_must_not_load
    )

    with pytest.raises(ValueError, match="Unsupported GGUF quantization"):
        export_gguf.export_adapter(
            adapter_dir=adapter_dir,
            output_dir=tmp_path / "output",
            llama_cpp_dir=llama_cpp_dir,
            quantization="Q2_K",
        )


def test_export_merges_peft_adapter_converts_quantizes_and_logs_artifacts(
    tmp_path, monkeypatch
):
    adapter_dir = tmp_path / "adapter"
    create_adapter(adapter_dir)
    llama_cpp_dir = tmp_path / "llama.cpp"
    llama_cpp_dir.mkdir()
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    converter.write_text("# synthetic converter stub\n", encoding="utf-8")
    quantizer = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    quantizer.parent.mkdir(parents=True)
    quantizer.write_text("#!/bin/sh\n", encoding="utf-8")
    quantizer.chmod(0o755)

    class FakeMergedModel:
        def save_pretrained(self, output_dir, **kwargs):
            (Path(output_dir) / "config.json").write_text("{}", encoding="utf-8")

    class FakePeftModel:
        def merge_and_unload(self, *, safe_merge):
            assert safe_merge is True
            return FakeMergedModel()

    class FakeBaseModelLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            assert model_id == "Qwen/Qwen2.5-7B-Instruct"
            assert kwargs["local_files_only"] is True
            return object()

    class FakeTokenizer:
        def save_pretrained(self, output_dir):
            (Path(output_dir) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(adapter_path, **kwargs):
            assert Path(adapter_path) == adapter_dir.resolve()
            assert kwargs["local_files_only"] is True
            return FakeTokenizer()

    class FakePeftLoader:
        @staticmethod
        def from_pretrained(model, adapter_path, *, is_trainable):
            assert model is not None
            assert adapter_path == str(adapter_dir.resolve())
            assert is_trainable is False
            return FakePeftModel()

    class FakeRun:
        info = SimpleNamespace(run_id="export-run-1")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    class FakeMlflow:
        def __init__(self):
            self.logged_artifacts = []
            self.logged_params = {}
            self.tracking = SimpleNamespace()

        def set_tracking_uri(self, uri):
            assert uri == "http://localhost:5000"

        def set_experiment(self, name):
            assert name == "Cardio-LLM-GGUF-Exports"

        def start_run(self, *, run_name):
            assert run_name == "gguf-export-Q4_K_M"
            return FakeRun()

        def log_params(self, params):
            self.logged_params.update(params)

        def log_artifact(self, path, *, artifact_path):
            self.logged_artifacts.append((Path(path).name, artifact_path))

    fake_mlflow = FakeMlflow()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.delenv("MLFLOW_EXPORT_EXPERIMENT_NAME", raising=False)
    monkeypatch.setattr(
        export_gguf,
        "_load_export_dependencies",
        lambda: {
            "mlflow": fake_mlflow,
            "torch": SimpleNamespace(float16="float16"),
            "AutoModelForCausalLM": FakeBaseModelLoader,
            "PeftModel": FakePeftLoader,
            "AutoTokenizer": FakeTokenizerLoader,
        },
    )

    def fake_run(command, *, check):
        assert check is True
        output = (
            Path(command[command.index("--outfile") + 1])
            if "--outfile" in command
            else Path(command[2])
        )
        output.write_bytes(b"GGUFsynthetic")

    monkeypatch.setattr(export_gguf.subprocess, "run", fake_run)

    result = export_gguf.export_adapter(
        adapter_dir=adapter_dir,
        output_dir=tmp_path / "output",
        llama_cpp_dir=llama_cpp_dir,
        quantization="Q4_K_M",
    )

    gguf_path = Path(result["gguf_path"])
    modelfile_path = Path(result["modelfile_path"])
    assert gguf_path.read_bytes().startswith(b"GGUF")
    assert result["mlflow_run_id"] == "export-run-1"
    assert result["ollama_model_name"] == "cardio-llm"
    assert f"FROM /models/{gguf_path.name}" in modelfile_path.read_text()
    metadata = json.loads(
        (gguf_path.parent / "export_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["adapter_training_run_id"] == "training-run-1"
    assert fake_mlflow.logged_params["quantization"] == "Q4_K_M"
    assert {path for path, _ in fake_mlflow.logged_artifacts} == {
        gguf_path.name,
        "Modelfile",
        "export_metadata.json",
    }


def test_export_refuses_base_model_mismatch(tmp_path):
    adapter_dir = tmp_path / "adapter"
    create_adapter(adapter_dir)
    metadata_path = adapter_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["base_model_id"] = "some/other-model"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match model_metadata"):
        export_gguf._read_adapter_metadata(adapter_dir)


def test_modelfile_uses_internal_model_volume_and_validates_name(tmp_path):
    modelfile = tmp_path / "Modelfile"

    export_gguf._write_modelfile(
        modelfile,
        "cardio-llm-q4_k_m.gguf",
        "cardio-llm",
    )

    contents = modelfile.read_text(encoding="utf-8")
    assert "FROM /models/cardio-llm-q4_k_m.gguf" in contents
    assert "Do not make a medical diagnosis" in contents
    with pytest.raises(ValueError, match="Ollama model name"):
        export_gguf._validate_model_name("../unexpected")
