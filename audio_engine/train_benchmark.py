from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .models import build_classical_1d_models, build_deep_2d_models, set_seed


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


def select_best_model(results: list[dict[str, Any]], *, min_specificity: float = 0.5) -> dict[str, Any]:
    """Choose the best model by validation recall, subject to specificity constraint."""
    eligible = [result for result in results if result.get("val_specificity", 0.0) >= min_specificity]
    if not eligible:
        return max(results, key=lambda item: item.get("val_recall", 0.0))
    return max(eligible, key=lambda item: (item.get("val_recall", 0.0), item.get("val_f1", 0.0)))


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _estimate_model_size_mb(model: torch.nn.Module) -> float:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return (parameters * 4) / (1024**2)


def _train_1d_model(name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    models = build_classical_1d_models()
    model = models[name]
    start = __import__("time").perf_counter()
    model.fit(X_train, y_train)
    training_time = __import__("time").perf_counter() - start
    val_prob = model.predict_proba(X_val)[:, 1]
    val_metrics = _compute_metrics(y_val, val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_prob)
    return {
        "algorithm": name,
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_time_s": training_time,
        "inference_ms_per_window": 0.0,
        "params": 0,
        "model_size_mb": 0.0,
    }


def _train_2d_model(name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    model_map = build_deep_2d_models()
    model = model_map[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    x_t = torch.tensor(X_train.astype(np.float32), device=device).unsqueeze(1)
    y_t = torch.tensor(y_train.astype(np.int64), device=device)
    start = __import__("time").perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(4):
        optimizer.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
    training_time = __import__("time").perf_counter() - start
    model.eval()
    with torch.no_grad():
        val_prob = torch.softmax(model(torch.tensor(X_val.astype(np.float32), device=device).unsqueeze(1)), dim=1)[:, 1].cpu().numpy()
        test_prob = torch.softmax(model(torch.tensor(X_test.astype(np.float32), device=device).unsqueeze(1)), dim=1)[:, 1].cpu().numpy()
    val_metrics = _compute_metrics(y_val, val_prob)
    test_metrics = _compute_metrics(y_test, test_prob)
    return {
        "algorithm": name,
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_time_s": training_time,
        "inference_ms_per_window": 0.0,
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
    mlflow_uri: str | None = None,
    experiment_name: str = "cardio_audio_benchmark",
) -> dict[str, Any]:
    """Run the benchmark with the required 1D and 2D model families and return a summary dictionary."""
    set_seed(42)
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    results: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="benchmark", nested=False) as parent_run:
        mlflow.log_param("experiment_name", experiment_name)
        for algorithm in ["decision_tree", "rbf_svm", "cnn", "resnet18", "transformer_cnn"]:
            with mlflow.start_run(run_name=algorithm, nested=True):
                if algorithm in {"decision_tree", "rbf_svm"}:
                    record = _train_1d_model(algorithm, X_train, y_train, X_val, y_val, X_test, y_test)
                else:
                    record = _train_2d_model(algorithm, X_train, y_train, X_val, y_val, X_test, y_test)
                mlflow.log_metrics({
                    "val_recall": record["val_metrics"]["recall"],
                    "val_specificity": record["val_metrics"]["specificity"],
                    "val_f1": record["val_metrics"]["f1"],
                    "test_accuracy": record["test_metrics"]["accuracy"],
                    "test_roc_auc": record["test_metrics"]["roc_auc"],
                })
                results.append(record)
        winner = select_best_model([
            {
                "algorithm": item["algorithm"],
                "val_recall": item["val_metrics"]["recall"],
                "val_specificity": item["val_metrics"]["specificity"],
                "val_f1": item["val_metrics"]["f1"],
            }
            for item in results
        ])
        mlflow.log_param("winner_algorithm", winner["algorithm"])
        mlflow.log_metric("winner_val_recall", winner["val_recall"])
    return {
        "winner": winner["algorithm"],
        "results": results,
        "parent_run_id": parent_run.info.run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CardioAgent-Ops PCG benchmark")
    parser.add_argument("--train-features", type=str, default="./data/train_features.npy")
    parser.add_argument("--train-labels", type=str, default="./data/train_labels.npy")
    parser.add_argument("--val-features", type=str, default="./data/val_features.npy")
    parser.add_argument("--val-labels", type=str, default="./data/val_labels.npy")
    parser.add_argument("--test-features", type=str, default="./data/test_features.npy")
    parser.add_argument("--test-labels", type=str, default="./data/test_labels.npy")
    parser.add_argument("--mlflow-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    parser.add_argument("--experiment-name", type=str, default="cardio_audio_benchmark")
    args = parser.parse_args()

    X_train = np.load(args.train_features)
    y_train = np.load(args.train_labels)
    X_val = np.load(args.val_features)
    y_val = np.load(args.val_labels)
    X_test = np.load(args.test_features)
    y_test = np.load(args.test_labels)

    summary = run_benchmark(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
    )
    output_path = Path("benchmark_summary.json")
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"winner": summary["winner"], "results_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
