"""Export the selected trained model and its input contract as a portable bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def export_model_bundle(
    *,
    algorithm: str,
    model: Any,
    output_dir: str | Path,
    classes: list[str],
    metrics: dict[str, Any],
    preprocessing: dict[str, Any],
    training_seconds: float,
    inference_ms_per_sample: float,
    model_parameters: int,
    mel_input: bool = False,
    example_input: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write a no-overwrite model directory; ONNX is best-effort for torch models."""
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite model bundle: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "algorithm": algorithm,
        "classes": [str(value) for value in classes],
        "label_mapping": {str(value): index for index, value in enumerate(classes)},
        "metrics": metrics,
        "training_seconds": float(training_seconds),
        "inference_ms_per_sample": float(inference_ms_per_sample),
        "model_parameters": int(model_parameters),
        "preprocessing": preprocessing,
        "input_type": "log_mel" if mel_input else "mfcc_statistics",
    }

    if mel_input:
        import torch

        checkpoint_path = destination / "model.pt"
        cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({"algorithm": algorithm, "state_dict": cpu_state}, checkpoint_path)
        if example_input is not None:
            try:
                model.to("cpu")
                model.eval()
                input_tensor = torch.from_numpy(np.asarray(example_input, dtype=np.float32))
                if input_tensor.ndim == 3:
                    input_tensor = input_tensor.unsqueeze(1)
                torch.onnx.export(
                    model,
                    input_tensor[:1],
                    destination / "model.onnx",
                    input_names=["log_mel"],
                    output_names=["logits"],
                    dynamic_axes={"log_mel": {0: "batch"}, "logits": {0: "batch"}},
                    opset_version=17,
                )
                metadata["onnx_export"] = "success"
            except Exception as exc:  # The checkpoint remains a valid portable artifact.
                (destination / "model.onnx").unlink(missing_ok=True)
                metadata["onnx_export"] = "failed"
                metadata["onnx_export_error"] = f"{type(exc).__name__}: {exc}"
        else:
            metadata["onnx_export"] = "skipped: no representative mel input supplied"
    else:
        import joblib

        joblib.dump(model, destination / "model.joblib")
        try:
            try:
                from .export_onnx import export_sklearn_to_onnx, parity_check_onnx
            except ImportError:
                from export_onnx import export_sklearn_to_onnx, parity_check_onnx

            feature_count = int(preprocessing["feature_count"])
            export_sklearn_to_onnx(model, destination / "model.onnx", feature_dim=feature_count)
            parity = parity_check_onnx(model, destination / "model.onnx", example_input)
            metadata["onnx_export"] = "success"
            metadata["onnx_parity"] = parity
        except Exception as exc:
            (destination / "model.onnx").unlink(missing_ok=True)
            (destination / "model.json").unlink(missing_ok=True)
            metadata["onnx_export"] = "failed"
            metadata["onnx_export_error"] = f"{type(exc).__name__}: {exc}"

    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata
