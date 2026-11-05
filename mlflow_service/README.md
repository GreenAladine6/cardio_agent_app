# MLflow tracking service

This folder contains the standalone MLflow tracking-server image. The server
stores tracking metadata in SQLite at `/mlflow/data/mlflow.db` and artifacts
under `/mlflow/artifacts`. Compose mounts separate persistent volumes at both
locations. The server proxies artifact requests so clients only need the
tracking URI; the artifact volume does not need to be mounted in the clients.

From the repository root, build and start the service with the rest of the
platform:

```sh
docker compose up --build mlflow
```

The tracking UI and API are available at `http://localhost:5000`; Compose
clients use `http://mlflow:5000`. Check readiness with
`curl -f http://localhost:5000/health`. Metadata and artifacts survive
container recreation. `docker compose down -v` removes the named data volumes
and permanently deletes that persisted data.

To build the image independently:

```sh
docker build -t cardio-mlflow:latest mlflow_service
```
