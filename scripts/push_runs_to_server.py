from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow


def push_runs_to_server(*, tracking_uri: str, model_path: str | Path | None = None, metrics_path: str | Path | None = None) -> dict[str, str]:
    """Re-log a benchmark summary and champion model to a remote MLflow server."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "cardio_audio_benchmark")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="champion-push") as run:
        if metrics_path:
            path = Path(metrics_path)
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                mlflow.log_dict(payload, "benchmark_summary.json")
        if model_path:
            candidate = Path(model_path)
            if candidate.exists():
                mlflow.log_artifact(str(candidate), artifact_path="champion_model")
        mlflow.set_tag("source", "colab-export")
    return {"run_id": run.info.run_id, "tracking_uri": tracking_uri}


def main() -> None:
    parser = argparse.ArgumentParser(description="Push benchmark metrics and model artifacts to the configured MLflow server")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--model-path", default="./audio_engine/artifacts/cardio_model.onnx")
    parser.add_argument("--metrics-path", default="./benchmark_summary.json")
    args = parser.parse_args()
    result = push_runs_to_server(tracking_uri=args.tracking_uri, model_path=args.model_path, metrics_path=args.metrics_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
