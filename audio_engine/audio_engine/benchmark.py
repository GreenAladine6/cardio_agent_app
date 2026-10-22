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
import onnxruntime as ort
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
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
    one target per recording. The reproducible stratified 60/20/20 split uses
    validation recall for candidate selection and evaluates the winner once on
    the held-out test set.
    """
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels).astype(str)
    if x.ndim != 2 or len(x) != len(y) or len(x) < 8:
        raise ValueError("Provide at least 8 samples as a 2D feature matrix with one label each")
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains NaN or infinite values")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or np.min(counts) < 5:
        raise ValueError("At least two classes and five samples per class are required for train/validation/test")
    positive_names = {"pathological", "abnormal", "murmur", "extrahls", "extrasystole", "1"}
    positive = os.getenv("PATHOLOGICAL_LABEL", "").strip()
    if positive:
        matches = [value for value in classes if str(value).casefold() == positive.casefold()]
    else:
        matches = [value for value in classes if str(value).casefold() in positive_names]
    if len(matches) != 1:
        raise ValueError("Could not identify one pathological class; set PATHOLOGICAL_LABEL to a label in the dataset")
    pathological_class = int(np.flatnonzero(classes == matches[0])[0])
    try:
        x_train, x_holdout, y_train, y_holdout = train_test_split(
            x, y, test_size=0.4, random_state=42, stratify=y
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_holdout, y_holdout, test_size=0.5, random_state=42, stratify=y_holdout
        )
    except ValueError as exc:
        raise ValueError(f"Could not create stratified holdout split: {exc}") from exc

    encoder = LabelEncoder().fit(y_train)
    y_train_encoded = encoder.transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)
    pathological_encoded = int(encoder.transform([matches[0]])[0])
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns"))
    mlflow.set_experiment(experiment_name)
    results = []
    fitted_models = {}
    with _MLFLOW_LOCK:
        with mlflow.start_run(run_name="pcg-five-model-benchmark") as parent:
            mlflow.log_params({"samples": len(y), "features": x.shape[1], "classes": len(classes), "train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "seed": 42, "pathological_class": str(matches[0])})
            for name, model in candidate_models().items():
                start = time.perf_counter()
                with mlflow.start_run(run_name=name, nested=True) as child:
                    model.fit(x_train, y_train_encoded)
                    train_seconds = time.perf_counter() - start
                    inference_start = time.perf_counter()
                    predictions = model.predict(x_val).astype(int)
                    inference_ms = (time.perf_counter() - inference_start) * 1000 / len(y_val)
                    accuracy = float(accuracy_score(y_val_encoded, predictions))
                    weighted_f1 = float(f1_score(y_val_encoded, predictions, average="weighted", zero_division=0))
                    pathological_recall = float(recall_score(y_val_encoded, predictions, pos_label=pathological_encoded, zero_division=0))
                    pathological_precision = float(precision_score(y_val_encoded, predictions, pos_label=pathological_encoded, zero_division=0))
                    tn, fp, fn, tp = confusion_matrix(y_val_encoded == pathological_encoded, predictions == pathological_encoded, labels=[False, True]).ravel()
                    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
                    mlflow.log_params({"algorithm": name, "train_samples": len(y_train), "validation_samples": len(y_val), "test_samples": len(y_test)})
                    mlflow.log_metrics({"val_accuracy": accuracy, "val_weighted_f1": weighted_f1,
                                        "val_pathological_recall": pathological_recall,
                                        "val_pathological_precision": pathological_precision,
                                        "val_specificity": specificity,
                                        "fit_seconds": train_seconds, "inference_ms_per_sample": inference_ms})
                    results.append({"algorithm": name, "val_accuracy": accuracy, "val_weighted_f1": weighted_f1,
                                    "val_pathological_recall": pathological_recall,
                                    "val_pathological_precision": pathological_precision,
                                    "val_specificity": specificity,
                                    "fit_seconds": round(train_seconds, 4), "inference_ms_per_sample": round(inference_ms, 4),
                                    "mlflow_run_id": child.info.run_id})
                    fitted_models[name] = model

            winner = max(results, key=lambda result: (
                result["val_pathological_recall"], result["val_weighted_f1"],
                result["val_specificity"], result["val_accuracy"], result["algorithm"],
            ))
            winning_model = fitted_models[winner["algorithm"]]
            test_start = time.perf_counter()
            test_predictions = winning_model.predict(x_test).astype(int)
            winner["test_accuracy"] = float(accuracy_score(y_test_encoded, test_predictions))
            winner["test_weighted_f1"] = float(f1_score(y_test_encoded, test_predictions, average="weighted", zero_division=0))
            winner["test_pathological_recall"] = float(recall_score(y_test_encoded, test_predictions, pos_label=pathological_encoded, zero_division=0))
            winner["test_confusion_matrix"] = confusion_matrix(y_test_encoded == pathological_encoded, test_predictions == pathological_encoded, labels=[False, True]).tolist()
            winner["test_inference_ms_per_sample"] = (time.perf_counter() - test_start) * 1000.0 / len(y_test)
            mlflow.log_metrics({key: winner[key] for key in ("test_accuracy", "test_weighted_f1", "test_pathological_recall", "test_inference_ms_per_sample")})
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
            # Verify loadability and label parity before replacing a serving model.
            try:
                session = ort.InferenceSession(str(temp_path), providers=["CPUExecutionProvider"])
                input_name = session.get_inputs()[0].name
                parity_features = x_val[: min(8, len(x_val))].astype(np.float32)
                onnx_prediction = np.asarray(session.run(None, {input_name: parity_features})[0]).reshape(-1)
                source_prediction = winning_model.predict(parity_features).reshape(-1)
                if not np.array_equal(onnx_prediction, source_prediction):
                    raise ValueError("Exported ONNX predictions do not match the selected source model")
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
            os.replace(temp_path, destination)
            metadata_path = destination.with_suffix(destination.suffix + ".json")
            metadata_path.write_text(json.dumps({
                "algorithm": winner["algorithm"], "classes": encoder.classes_.tolist(),
                "feature_count": int(x.shape[1]), "model_version": parent.info.run_id,
                "sample_rate": 2000, "target_seconds": 5.0, "n_mfcc": 20,
                "normalization": "peak", "pathological_class": str(matches[0]),
                "validation_selection": "pathological_recall, weighted_f1, specificity, accuracy, algorithm",
                "train_samples": int(len(y_train)), "validation_samples": int(len(y_val)),
                "test_samples": int(len(y_test)),
            }, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(destination), artifact_path="winner_model")
            mlflow.log_artifact(str(metadata_path), artifact_path="winner_model")
            mlflow.log_param("winner", winner["algorithm"])
            mlflow.log_metric("winner_val_pathological_recall", winner["val_pathological_recall"])

    results.sort(key=lambda result: (result["val_pathological_recall"], result["val_weighted_f1"], result["val_specificity"], result["val_accuracy"], result["algorithm"]), reverse=True)
    return {"experiment": experiment_name, "winner": winner["algorithm"], "model_path": str(destination),
            "pathological_class": str(matches[0]), "classes": encoder.classes_.tolist(), "results": results}
