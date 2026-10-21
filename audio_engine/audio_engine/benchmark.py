"""Train, compare, track, and export the five supported PCG classifiers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


_MLFLOW_LOCK = threading.Lock()


def candidate_models():
    """Return a fresh set of four classical ML models and a two-layer MLP."""
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "mlp_2_layer": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
    }


def run_benchmark(features, labels, *, model_path: str | Path, experiment_name: str = "pcg-benchmark"):
    """Fit/evaluate candidates, log every run to MLflow, and publish best ONNX.

    `features` must be the fixed-width output of AudioPreprocessor and `labels`
    one target per recording. The holdout split is stratified and reproducible.
    """
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels).astype(str)
    if x.ndim != 2 or len(x) != len(y) or len(x) < 8:
        raise ValueError("Provide at least 8 samples as a 2D feature matrix with one label each")
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains NaN or infinite values")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 2:
        raise ValueError("At least two classes are required, with at least two samples per class")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError as exc:
        raise ValueError(f"Could not create stratified holdout split: {exc}") from exc

    encoder = LabelEncoder().fit(y_train)
    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns"))
    mlflow.set_experiment(experiment_name)
    results = []
    fitted_models = {}
    with _MLFLOW_LOCK:
        with mlflow.start_run(run_name="pcg-five-model-benchmark") as parent:
            mlflow.log_params({"samples": len(y), "features": x.shape[1], "classes": len(classes), "test_fraction": 0.25, "seed": 42})
            for name, model in candidate_models().items():
                start = time.perf_counter()
                with mlflow.start_run(run_name=name, nested=True) as child:
                    model.fit(x_train, y_train_encoded)
                    train_seconds = time.perf_counter() - start
                    inference_start = time.perf_counter()
                    predictions = model.predict(x_test).astype(int)
                    inference_ms = (time.perf_counter() - inference_start) * 1000 / len(y_test)
                    accuracy = float(accuracy_score(y_test_encoded, predictions))
                    weighted_f1 = float(f1_score(y_test_encoded, predictions, average="weighted", zero_division=0))
                    mlflow.log_params({"algorithm": name, "train_samples": len(y_train), "test_samples": len(y_test)})
                    mlflow.log_metrics({"accuracy": accuracy, "weighted_f1": weighted_f1,
                                        "fit_seconds": train_seconds, "inference_ms_per_sample": inference_ms})
                    results.append({"algorithm": name, "accuracy": accuracy, "weighted_f1": weighted_f1,
                                    "fit_seconds": round(train_seconds, 4), "inference_ms_per_sample": round(inference_ms, 4),
                                    "mlflow_run_id": child.info.run_id})
                    fitted_models[name] = model

            winner = max(results, key=lambda result: (result["weighted_f1"], result["accuracy"]))
            winning_model = fitted_models[winner["algorithm"]]
            final_estimator = winning_model.steps[-1][1] if hasattr(winning_model, "steps") else winning_model
            onnx = convert_sklearn(
                winning_model,
                initial_types=[("features", FloatTensorType([None, x.shape[1]]))],
                options={id(final_estimator): {"zipmap": False}},
                target_opset=17,
            )
            destination = Path(model_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".onnx", delete=False) as temp:
                temp_path = Path(temp.name)
                temp.write(onnx.SerializeToString())
            os.replace(temp_path, destination)
            metadata_path = destination.with_suffix(destination.suffix + ".json")
            metadata_path.write_text(json.dumps({
                "algorithm": winner["algorithm"], "classes": encoder.classes_.tolist(),
                "feature_count": int(x.shape[1]),
            }, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(destination), artifact_path="winner_model")
            mlflow.log_artifact(str(metadata_path), artifact_path="winner_model")
            mlflow.log_param("winner", winner["algorithm"])
            mlflow.log_metric("winner_weighted_f1", winner["weighted_f1"])

    results.sort(key=lambda result: (result["weighted_f1"], result["accuracy"]), reverse=True)
    return {"experiment": experiment_name, "winner": winner["algorithm"], "model_path": str(destination),
            "classes": encoder.classes_.tolist(), "results": results}
