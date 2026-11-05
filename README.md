# CardioAgent-Ops

CardioAgent-Ops is a microservice-based phonocardiogram (PCG) analysis platform for heart sound classification and inference. The project is organized into separate services for the API gateway and the audio-processing engine, with a Dockerized deployment pattern for local or cloud execution.

## Overview

The repository contains:

- `api_getway`: lightweight FastAPI gateway that receives uploads and routes requests
- `audio_engine`: audio preprocessing, feature extraction, model training, export, and inference service
- `mlflow_service`: standalone MLflow tracking server and container image
- `docker-compose.yml`: service orchestration for local multi-container setup
- `tests`: validation checks for the audio preprocessing and service behavior

## Architecture

### API gateway
The API gateway is a small FastAPI service that accepts uploaded WAV files and exposes a web/test entrypoint.
It forwards valid uploads to the audio engine inference endpoint and returns the engine response.

### Audio engine
The audio engine handles:

- audio resampling and filtering
- normalization and segmentation
- MFCC and spectral feature extraction
- model training and benchmarking
- ONNX export and parity validation
- inference over audio files

## Project structure

```text
cardio_app/
├── api_getway/
│   ├── app/
│   ├── frontend/
│   └── Dockerfile
├── audio_engine/
│   ├── audio_engine/
│   ├── artifacts/
│   ├── models/
│   ├── Dockerfile
│   ├── README.md
│   ├── config.yaml
│   ├── data_prep.py
│   ├── export_onnx.py
│   ├── features.py
│   ├── models.py
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── schemas.py
│   ├── serve.py
│   ├── train_benchmark.py
│   └── requirements-runtime.txt
├── docker/
├── mlflow_service/
│   ├── Dockerfile
│   ├── README.md
│   └── requirements.txt
├── docker-compose.yml
├── tests/
├── .gitignore
├── README.md
└── scripts/
```

## Requirements

### Python
- Python 3.10+

### Core Python dependencies
The project uses packages like:

- FastAPI
- Uvicorn
- NumPy
- SciPy
- librosa
- scikit-learn
- onnxruntime
- skl2onnx
- MLflow
- pandas

You can install them from the project environment or the local virtualenv used during development.

## Local setup

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r audio_engine/requirements.txt
```

If you want the runtime-only dependencies for the service:

```bash
pip install -r audio_engine/requirements-runtime.txt
```

## Run the services

### Option 1: Docker Compose

From the repository root:

```bash
docker compose up --build
```

This starts the separate services defined in `docker-compose.yml`.

The Compose stack also starts the LLM inference API at
`http://localhost:8002`, backed by Ollama on `http://localhost:11434`.
Ollama's model data persists in the `ollama-data` volume. The exported GGUF
directory is mounted read-only at `/models`; the stack does not download or
import a model automatically.

After exporting a GGUF and its `Modelfile` into `models/gguf/`, import it with:

```bash
docker compose exec ollama ollama create cardio-llm -f /models/Modelfile
```

The inference API's `/health` endpoint stays unavailable until `cardio-llm` is
present in Ollama. Chat requests use `POST /v1/chat`.

It also starts an MLflow tracking server at `http://localhost:5000`.
The server is built from `mlflow_service/`, stores its SQLite metadata and
artifacts in separate Docker-managed volumes, and exposes `/health` for
readiness checks. Compose waits for MLflow to become healthy before starting
the audio engine. Data persists across container recreation; `docker compose
down -v` removes the MLflow and Ollama volumes and their data.

Gateway forwarding is controlled by:

- `AUDIO_ENGINE_URL` (default local: `http://localhost:8001`, docker-compose: `http://audio-engine:8000`)
- `AUDIO_ENGINE_INFER_PATH` (default: `/v1/infer`)

MLflow tracking is controlled by:

- `MLFLOW_TRACKING_URI` (docker-compose default: `http://mlflow:5000` inside containers)
- Host UI URL: `http://localhost:5000`

### Option 2: Run the audio engine directly

```bash
cd audio_engine
uvicorn audio_engine.service:app --host 0.0.0.0 --port 8000
```

### Option 3: Run the gateway directly

```bash
cd api_getway
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API usage

### Health check

```bash
curl http://localhost:8001/health
```

### Prediction endpoint

```bash
curl -X POST "http://localhost:8001/v1/infer" -F "file=@sample.wav"
```

### Gateway upload endpoint

```bash
curl -X POST "http://localhost:8000/analyze_pcg" -F "file=@sample.wav"
```

## Model workflow

The project’s intended workflow is:

1. Prepare and clean recordings
2. Extract features using the shared preprocessing code
3. Train or benchmark models
4. Export the selected model to ONNX
5. Load the ONNX model in the prediction service

When you call `POST /v1/benchmark` on the audio engine, benchmark runs and metrics are logged to MLflow.
Open the MLflow UI at `http://localhost:5000` to inspect experiments, runs, metrics, and artifacts.

The ONNX export and parity-check logic is implemented in `audio_engine/export_onnx.py`.

## Testing

Install the audio-engine development dependencies and API gateway dependencies
from the repository root:

```bash
python -m pip install -e "audio_engine[dev]"
python -m pip install -r api_getway/app/requirements.txt
```

Run the unit and API integration tests with a coverage report:

```bash
python -m pytest -m "unit or integration"
```

This covers:

- audio preprocessing and feature extraction
- grouped split logic
- ONNX export parity checks
- audio-engine health and validation behavior
- API gateway upload validation, forwarding, and upstream error handling

Pytest prints a missing-lines coverage summary and writes the Cobertura report to
`coverage.xml`. GitHub Actions runs these tests on pushes and pull requests and
stores the JUnit and coverage reports as workflow artifacts.

The LLM inference API and Ollama integration are available through Compose.
The separate LLM conformance job remains disabled by default; enable the
repository Actions variable `LLM_CONFORMANCE_ENABLED` with the value `true`
to run tests marked `llm_conformance`.

## Notes

- This project is intended for research and service prototype use, not clinical diagnosis.
- The inference service expects valid WAV input and a trained model artifact.
- Docker Desktop or an equivalent Docker engine is required for containerized execution.

## License

This project is intended for educational and research use. Add your preferred license here if you plan to publish the repository publicly.
