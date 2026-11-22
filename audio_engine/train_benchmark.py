from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .models import build_classical_1d_models, build_deep_2d_models, set_seed
from .model_bundle import export_model_bundle


@dataclass
class BenchmarkResult:
    algorithm: str
    val_recall: float
    val_specificity: float
    val_f1: float
    test_accuracy: float
    test_roc_auc: float | None
    test_pr_auc: float | None
    training_time_s: float
    inference_ms_per_window: float
    params: int
    model_size_mb: float
    metadata: dict[str, Any]


def _safe_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)


def _safe_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0


def _safe_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fp) if (tp + fp) else 0.0


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    recall = _safe_recall(y_true, y_pred)
    specificity = _safe_specificity(y_true, y_pred)
    precision = _safe_precision(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.0
    return {
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
    }


def select_best_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select by pathological validation recall, with deterministic metric tie-breaks."""
    if not results:
        raise ValueError("At least one evaluated model is required")
    return max(results, key=lambda item: (
        item.get("val_recall", 0.0), item.get("val_f1", 0.0),
        item.get("val_specificity", 0.0), item.get("val_accuracy", 0.0), item.get("algorithm", ""),
    ))


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _estimate_model_size_mb(model: torch.nn.Module) -> float:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return (parameters * 4) / (1024**2)


def _classical_model_size(model: Any) -> tuple[int, float]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "tree_"):
        parameter_count = int(estimator.tree_.node_count)
    elif hasattr(estimator, "support_"):
        parameter_count = int(estimator.support_.size + estimator.dual_coef_.size)
    else:
        parameter_count = 0
    return parameter_count, len(pickle.dumps(model)) / (1024**2)


def _train_1d_model(name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    models = build_classical_1d_models()
    model = models[name]
    start = __import__("time").perf_counter()
    model.fit(X_train, y_train)
    training_time = __import__("time").perf_counter() - start
    inference_start = __import__("time").perf_counter()
    val_prob = model.predict_proba(X_val)[:, 1]
    inference_ms = (__import__("time").perf_counter() - inference_start) * 1000 / max(1, len(X_val))
    val_metrics = _compute_metrics(y_val, val_prob)
    val_metrics["confusion_matrix"] = confusion_matrix(y_val, (val_prob >= 0.5).astype(int), labels=[0, 1]).tolist()
    parameter_count, model_size_mb = _classical_model_size(model)
    return {
        "algorithm": name,
        "model": model,
        "val_metrics": val_metrics,
        "training_time_s": training_time,
        "inference_ms_per_window": inference_ms,
        "params": parameter_count,
        "model_size_mb": model_size_mb,
    }


def _train_2d_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    pretrained_resnet: bool = False,
) -> dict[str, Any]:
    model_map = build_deep_2d_models(pretrained_resnet=pretrained_resnet)
    model = model_map[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_t = torch.from_numpy(X_train.astype(np.float32)).unsqueeze(1)
    y_t = torch.from_numpy(y_train.astype(np.int64))
    batches = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    start = __import__("time").perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    class_counts = np.bincount(y_train.astype(np.int64), minlength=2).astype(np.float32)
    if np.any(class_counts == 0):
        raise ValueError("Training split must contain both classes")
    class_weights = class_counts.sum() / (2.0 * class_counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    training_time = __import__("time").perf_counter() - start
    model.eval()
    inference_start = __import__("time").perf_counter()
    with torch.no_grad():
        val_tensor = torch.from_numpy(X_val.astype(np.float32)).unsqueeze(1)
        val_prob = np.concatenate([
            torch.softmax(model(batch.to(device)), dim=1)[:, 1].cpu().numpy()
            for batch in torch.utils.data.DataLoader(val_tensor, batch_size=batch_size)
        ])
    inference_ms = (__import__("time").perf_counter() - inference_start) * 1000 / max(1, len(X_val))
    val_metrics = _compute_metrics(y_val, val_prob)
    val_metrics["confusion_matrix"] = confusion_matrix(y_val, (val_prob >= 0.5).astype(int), labels=[0, 1]).tolist()
    return {
        "algorithm": name,
        "model": model,
        "val_metrics": val_metrics,
        "training_time_s": training_time,
        "inference_ms_per_window": inference_ms,
        "params": _count_parameters(model),
        "model_size_mb": _estimate_model_size_mb(model),
    }


def run_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    X_train_mel: np.ndarray | None = None,
    X_val_mel: np.ndarray | None = None,
    X_test_mel: np.ndarray | None = None,
    mlflow_uri: str | None = None,
    experiment_name: str = "cardio_audio_benchmark",
    output_dir: str | Path = "./exported_models",
    epochs: int = 10,
    batch_size: int = 32,
    pretrained_resnet: bool = False,
) -> dict[str, Any]:
    """Run the benchmark with the required 1D and 2D model families and return a summary dictionary."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must both be positive")
    for split_name, labels in (("train", y_train), ("validation", y_val), ("test", y_test)):
        values = set(np.asarray(labels).tolist())
        if not values.issubset({0, 1}) or values != {0, 1}:
            raise ValueError(f"{split_name} labels must contain both binary classes: 0 normal and 1 pathological")
    set_seed(42)
    if any(value is None for value in (X_train_mel, X_val_mel, X_test_mel)):
        raise ValueError("Provide separate train/validation/test log-Mel arrays for the CNN, ResNet18, and Transformer-CNN models")
    mel_splits = (np.asarray(X_train_mel), np.asarray(X_val_mel), np.asarray(X_test_mel))
    for name, values, labels in zip(("train", "validation", "test"), mel_splits, (y_train, y_val, y_test)):
        if values.ndim != 3 or values.shape[0] != len(labels) or not np.isfinite(values).all():
            raise ValueError(f"{name} log-Mel data must be finite (samples, mel_bins, frames) and align with labels")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    results: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="benchmark", nested=False) as parent_run:
        mlflow.log_params({
            "experiment_name": experiment_name,
            "seed": 42,
            "deep_epochs": epochs,
            "batch_size": batch_size,
            "pretrained_resnet": pretrained_resnet,
            "selection_metric": "pathological_validation_recall",
        })
        for algorithm in ["decision_tree", "rbf_svm", "cnn", "resnet18", "transformer_cnn"]:
            with mlflow.start_run(run_name=algorithm, nested=True):
                if algorithm in {"decision_tree", "rbf_svm"}:
                    record = _train_1d_model(algorithm, X_train, y_train, X_val, y_val, X_test, y_test)
                else:
                    record = _train_2d_model(
                        algorithm, X_train_mel, y_train, X_val_mel, y_val, X_test_mel, y_test,
                        epochs=epochs, batch_size=batch_size, pretrained_resnet=pretrained_resnet,
                    )
                mlflow.log_metrics({
                    "val_recall": record["val_metrics"]["recall"],
                    "val_specificity": record["val_metrics"]["specificity"],
                    "val_precision": record["val_metrics"]["precision"],
                    "val_f1": record["val_metrics"]["f1"],
                    "val_accuracy": record["val_metrics"]["accuracy"],
                    "val_roc_auc": record["val_metrics"]["roc_auc"],
                    "training_time_s": record["training_time_s"],
                    "inference_ms_per_window": record["inference_ms_per_window"],
                    "model_parameters": record["params"],
                    "model_size_mb": record["model_size_mb"],
                })
                mlflow.log_dict(record["val_metrics"], "validation_metrics.json")
                if hasattr(record["model"], "get_params"):
                    mlflow.log_dict(record["model"].get_params(deep=True), "model_hyperparameters.json")
                else:
                    mlflow.log_dict({"architecture": algorithm, "parameters": record["params"]}, "model_hyperparameters.json")
                results.append(record)
        winner = select_best_model([
            {
                "algorithm": item["algorithm"],
                "val_recall": item["val_metrics"]["recall"],
                "val_specificity": item["val_metrics"]["specificity"],
                "val_f1": item["val_metrics"]["f1"],
                "val_accuracy": item["val_metrics"]["accuracy"],
            }
            for item in results
        ])
        selected = next(item for item in results if item["algorithm"] == winner["algorithm"])
        model = selected["model"]
        if winner["algorithm"] in {"decision_tree", "rbf_svm"}:
            test_probability = model.predict_proba(X_test)[:, 1]
        else:
            device = next(model.parameters()).device
            model.eval()
            test_tensor = torch.from_numpy(X_test_mel.astype(np.float32)).unsqueeze(1)
            with torch.no_grad():
                test_probability = np.concatenate([
                    torch.softmax(model(batch.to(device)), dim=1)[:, 1].cpu().numpy()
                    for batch in torch.utils.data.DataLoader(test_tensor, batch_size=batch_size)
                ])
        winner_test_metrics = _compute_metrics(y_test, test_probability)
        winner_test_metrics["confusion_matrix"] = confusion_matrix(
            y_test, (test_probability >= 0.5).astype(int), labels=[0, 1]
        ).tolist()
        mlflow.log_param("winner_algorithm", winner["algorithm"])
        mlflow.log_metric("winner_val_recall", winner["val_recall"])
        mlflow.log_metrics({f"winner_test_{key}": value for key, value in winner_test_metrics.items() if isinstance(value, (int, float))})
        is_mel_model = winner["algorithm"] not in {"decision_tree", "rbf_svm"}
        preprocessing = {
            "sample_rate": 2000,
            "target_seconds": 5.0,
            "window_seconds": 5.0,
            "overlap_ratio": 0.5,
            "normalization": "peak",
            "feature_count": int(X_train.shape[1]),
            "mfcc_feature_count": int(X_train.shape[1]),
            "mfcc": {"n_mfcc": 20, "n_fft": 256, "hop_length": 64, "aggregation": ["mean", "std"]},
            "mel_shape": list(np.asarray(X_train_mel).shape[1:]),
            "mel": {"n_mels": 64, "n_fft": 256, "hop_length": 64, "fmin": 20, "fmax": 1000},
            "label_mapping": {"normal": 0, "pathological": 1},
        }
        bundle_dir = Path(output_dir) / parent_run.info.run_id
        bundle_metadata = export_model_bundle(
            algorithm=winner["algorithm"],
            model=model,
            output_dir=bundle_dir,
            classes=["normal", "pathological"],
            metrics={"validation": selected["val_metrics"], "test": winner_test_metrics},
            preprocessing=preprocessing,
            training_seconds=selected["training_time_s"],
            inference_ms_per_sample=selected["inference_ms_per_window"],
            model_parameters=selected["params"],
            mel_input=is_mel_model,
            example_input=(X_test_mel[:1] if is_mel_model else X_test[: min(8, len(X_test))]),
        )
        mlflow.log_artifacts(str(bundle_dir), artifact_path="selected_model_bundle")
    result_summary = [
        {key: value for key, value in record.items() if key != "model"}
        for record in results
    ]
    return {
        "winner": winner["algorithm"],
        "results": result_summary,
        "winner_test_metrics": winner_test_metrics,
        "parent_run_id": parent_run.info.run_id,
        "artifact_directory": str(bundle_dir),
        "artifact_metadata": bundle_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CardioAgent-Ops PCG benchmark")
    parser.add_argument("--train-features", type=str, default="./data/train_features.npy")
    parser.add_argument("--train-labels", type=str, default="./data/train_labels.npy")
    parser.add_argument("--val-features", type=str, default="./data/val_features.npy")
    parser.add_argument("--val-labels", type=str, default="./data/val_labels.npy")
    parser.add_argument("--test-features", type=str, default="./data/test_features.npy")
    parser.add_argument("--test-labels", type=str, default="./data/test_labels.npy")
    parser.add_argument("--train-mel", type=str, default="./data/train_mel.npy")
    parser.add_argument("--val-mel", type=str, default="./data/val_mel.npy")
    parser.add_argument("--test-mel", type=str, default="./data/test_mel.npy")
    parser.add_argument("--mlflow-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    parser.add_argument("--experiment-name", type=str, default="cardio_audio_benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("./exported_models"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrained-resnet", action="store_true", help="Allow torchvision to download pretrained ResNet18 weights")
    args = parser.parse_args()

    X_train = np.load(args.train_features)
    y_train = np.load(args.train_labels)
    X_val = np.load(args.val_features)
    y_val = np.load(args.val_labels)
    X_test = np.load(args.test_features)
    y_test = np.load(args.test_labels)
    X_train_mel = np.load(args.train_mel)
    X_val_mel = np.load(args.val_mel)
    X_test_mel = np.load(args.test_mel)

    summary = run_benchmark(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_mel=X_train_mel,
        X_val_mel=X_val_mel,
        X_test_mel=X_test_mel,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pretrained_resnet=args.pretrained_resnet,
    )
    output_path = Path("benchmark_summary.json")
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"winner": summary["winner"], "results_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
