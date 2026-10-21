from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_sklearn_to_onnx(estimator: Any, out_path: str | Path, *, feature_dim: int, target_opset: int = 17) -> dict[str, Any]:
    """Convert a scikit-learn estimator to ONNX and return metadata for the export."""
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx_model = convert_sklearn(
        estimator,
        initial_types=[("features", FloatTensorType([None, feature_dim]))],
        target_opset=target_opset,
    )
    destination.write_bytes(onnx_model.SerializeToString())
    metadata = {
        "model_type": type(estimator).__name__,
        "feature_dim": int(feature_dim),
        "target_opset": int(target_opset),
        "path": str(destination),
    }
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parity_check_onnx(model: Any, onnx_path: str | Path, x_test: np.ndarray, *, atol: float = 1e-4) -> dict[str, Any]:
    """Run a parity check against ONNX Runtime to ensure exported model outputs match the original estimator."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.asarray(x_test, dtype=np.float32)})

    original_labels = np.asarray(model.predict(np.asarray(x_test, dtype=np.float32)))
    if len(outputs) == 1:
        onnx_out = np.asarray(outputs[0], dtype=np.float32)
        if onnx_out.ndim == 2 and onnx_out.shape[1] == 2:
            original = np.asarray(model.predict_proba(np.asarray(x_test, dtype=np.float32)), dtype=np.float32)
            max_abs_diff = float(np.max(np.abs(original - onnx_out)))
            if max_abs_diff > atol:
                raise ValueError(f"ONNX parity check failed: max_abs_diff={max_abs_diff} exceeds atol={atol}")
            return {"max_abs_diff": max_abs_diff, "atol": atol, "passed": True}
        if onnx_out.ndim == 1:
            max_abs_diff = float(np.max(np.abs(original_labels - onnx_out)))
            if max_abs_diff > 0:
                raise ValueError(f"ONNX label parity check failed: max_abs_diff={max_abs_diff} exceeds 0")
            return {"max_abs_diff": max_abs_diff, "atol": atol, "passed": True}

    if len(outputs) > 1:
        onnx_prob = outputs[1]
        if isinstance(onnx_prob, dict):
            onnx_prob = np.asarray([list(onnx_prob.values())], dtype=np.float32)
        elif isinstance(onnx_prob, list) and onnx_prob and isinstance(onnx_prob[0], dict):
            onnx_prob = np.asarray([list(row.values()) for row in onnx_prob], dtype=np.float32)
        else:
            onnx_prob = np.asarray(onnx_prob, dtype=np.float32)

        original = np.asarray(model.predict_proba(np.asarray(x_test, dtype=np.float32)), dtype=np.float32)
        max_abs_diff = float(np.max(np.abs(original - onnx_prob)))
        if max_abs_diff > atol:
            raise ValueError(f"ONNX parity check failed: max_abs_diff={max_abs_diff} exceeds atol={atol}")
        return {"max_abs_diff": max_abs_diff, "atol": atol, "passed": True}

    raise ValueError(f"Unsupported ONNX export output structure: {[(np.asarray(o).shape) for o in outputs]}")


def write_model_meta(meta_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    destination = Path(meta_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


__all__ = ["export_sklearn_to_onnx", "parity_check_onnx", "write_model_meta"]
