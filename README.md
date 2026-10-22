# CardioAgent-Ops

CardioAgent-Ops is a microservice-based phonocardiogram (PCG) analysis platform for heart sound classification and inference. The project is organized into separate services for the API gateway and the audio-processing engine, with a Dockerized deployment pattern for local or cloud execution.

## Overview

The repository contains:

- `api_getway`: lightweight FastAPI gateway that receives uploads and routes requests
- `audio_engine`: audio preprocessing, feature extraction, model training, export, and inference service
- `docker-compose.yml`: service orchestration for local multi-container setup
- `tests`: validation checks for the audio preprocessing and service behavior

## Architecture

### API gateway
The API gateway accepts WAV uploads and forwards them to the audio engine over
the shared Docker Compose network.

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

The ONNX export and parity-check logic is implemented in `audio_engine/export_onnx.py`.

## Testing

Run the project test suite from the repository root:

```bash
python -m pytest -q
```

This covers:

- audio preprocessing
- feature extraction
- grouped split logic
- ONNX export parity checks
- service health and validation behavior

## Notes

- This project is intended for research and service prototype use, not clinical diagnosis.
- The inference service expects valid WAV input and a trained model artifact.
- Docker Desktop or an equivalent Docker engine is required for containerized execution.

## License

This project is intended for educational and research use. Add your preferred license here if you plan to publish the repository publicly.
