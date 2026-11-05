# Audio Engine

FastAPI microservice for WAV validation, PCG preprocessing, ONNX inference, and
an online small-data classical-model benchmark. The Compose service is
`audio_engine.service:app`; it exposes `/health`, `/v1/config`, `/v1/infer`,
and `/v1/benchmark`. The API gateway forwards uploads to
`http://audio-engine:8000/v1/infer` on Compose's default network.

## Layout

```text
audio_engine/
├── audio_engine/       # Compose service and shared fixed-vector preprocessor
├── artifacts/          # test ONNX fixture only; no production model is shipped
├── config.yaml         # dataset preprocessing defaults
├── data_prep.py        # label harmonization, resampling, grouped splits, windows
├── features.py         # MFCC, ZCR, spectral statistics and log-Mel features
├── models.py           # Decision Tree, RBF SVM, CNN, ResNet18, CNN/Transformer
├── train_benchmark.py  # offline five-architecture training/evaluation entrypoint
├── export_onnx.py      # sklearn export and ONNX Runtime parity helper
└── requirements*.txt   # full, runtime, and notebook dependency sets
```

## Setup and configuration

Python 3.10 or newer is required. From the repository root, install the
service and online classical-benchmark dependencies with
`pip install -r audio_engine/requirements.txt`. Install the larger pandas and
PyTorch data-training stack with `pip install -r audio_engine/requirements-training.txt`.
For Compose, `audio_engine/Dockerfile` installs the service and online
benchmark dependencies. Runtime-only dependencies are listed in
`requirements-runtime.txt` and omit MLflow and scikit-learn.

Compose variables:

| Variable | Default | Purpose |
|---|---|---|
| `ONNX_MODEL_PATH` | `/models/pcg_classifier.onnx` | ONNX candidate loaded for inference |
| `MAX_AUDIO_BYTES` | `20971520` | Upload size limit (20 MiB) |
| `MLFLOW_TRACKING_URI` | `file:/mlruns` | MLflow tracking backend; Compose points to `http://mlflow:5000` |
| `MLFLOW_EXPERIMENT_NAME` | `pcg-benchmark` | Online benchmark experiment |
| `AUDIO_ENGINE_URL` | `http://audio-engine:8000` in Compose | Gateway's internal service URL |

The service validates file contents with SoundFile, accepts WAV encoding,
downmixes multichannel audio by channel mean, resamples to 2,000 Hz, peak
normalizes non-silent recordings, then pads/crops to a fixed five-second input
window. MFCC mean and standard deviation produce a 40-value float32 vector.
The preprocessing is deterministic and shared by the online benchmark and
inference service. Files shorter than 0.25 seconds, longer than 120 seconds,
empty, malformed, or non-finite are rejected. Silences remain valid zero
signals.

The dataset configuration in `config.yaml` uses five-second windows at 50%
overlap and includes the separate MFCC/ZCR/log-Mel data-preparation functions.
`data_prep.make_grouped_split` assigns recordings by group when metadata is
available; always use patient/source IDs as group keys where supplied.

## Dataset and training

No PASCAL or PhysioNet records are present in this checkout. Supply legally
obtained recordings and metadata yourself; do not commit raw recordings.
`data_prep.harmonize_labels` maps common normal and abnormal label names, and
the offline benchmark entrypoint currently reads precomputed NumPy arrays:

```sh
cd audio_engine
python -m audio_engine.train_benchmark \
  --train-features data/train_features.npy --train-labels data/train_labels.npy \
  --val-features data/val_features.npy --val-labels data/val_labels.npy \
  --test-features data/test_features.npy --test-labels data/test_labels.npy
```

`models.py` contains the required Decision Tree and scaled RBF SVM for 1D
features, a custom log-Mel CNN, ResNet18, and a CNN stem plus Transformer
encoder. The neural models expect single-channel `(batch, 1, mel, frames)`
inputs. ResNet18 uses no weights by default; `PCGResNet18(pretrained=True)` is
an explicit training-time request that may download torchvision weights.
`train_benchmark.py` takes separate MFCC/feature and log-Mel arrays. The latter
must have `(samples, mel_bins, frames)` shape and align with the same labels.
It selects by validation pathological recall, then F1, specificity, accuracy,
and algorithm name; a minimum specificity of 0.5 is required. Only the winner
is evaluated on the held-out test split. This five-model runner currently does
not export the selected neural model to ONNX; the online classical benchmark
does export and checks label parity before replacing its artifact.

The `/v1/benchmark` convenience endpoint accepts 8–256 base64 WAV samples and
trains the lightweight models in `audio_engine/audio_engine/benchmark.py`.
Each class needs at least five recordings. It selects by validation
pathological recall with deterministic F1/specificity/accuracy/name tie-breaks,
then evaluates only the winner on the test split. It tracks runs through MLflow
and exports a scikit-learn model after ONNX Runtime parity verification. It is a small
online utility and is not the five-architecture dataset benchmark. Inference
does not contact MLflow; a tracking outage affects benchmarking, not normal
prediction.

## API

Start locally from the repository root:

```sh
cd audio_engine
uvicorn audio_engine.service:app --host 0.0.0.0 --port 8000
```

Or start the whole stack with `docker compose up --build`. The gateway is
published at port 8000 and the engine at port 8001 on the host.

```sh
curl http://localhost:8001/health
curl http://localhost:8001/v1/config
curl -F 'file=@recording.wav' http://localhost:8000/analyze_pcg
```

`POST /v1/infer` accepts multipart field `file` with WAV bytes and responds
with `{"prediction":{"label":"...","class_index":0,"probabilities":{},"model_version":"...","inference_ms":...}}`.
Scores are emitted as model scores; they are not calibrated unless the model
training pipeline has calibrated them. Missing or unreadable model artifacts
produce HTTP 503. `/health` reports whether the configured model file exists.

`POST /v1/benchmark` accepts JSON of the form
`{"samples":[{"label":"normal","audio_base64":"..."}]}`. It requires
multiple examples of at least two classes.

## Docker, tests, and verified status

Compose uses Docker service DNS (`audio-engine`, `mlflow`) and the default
shared network. Model files are mounted at `/models`. MLflow metadata and
artifacts are stored in the separate `mlflow-data` and `mlflow-artifacts`
Docker volumes and are served through the tracking server.
The checked-in `artifacts/test_onnx.onnx` is a six-input test fixture, not a
production model, and is not mounted as the Compose model. Consequently a
fresh deployment correctly returns 503 from inference until a compatible
trained artifact and metadata file are supplied.

### Build and smoke-test the image

Run these commands from the repository root. The build context must be
`audio_engine` because the Dockerfile copies the service package from that
directory:

```sh
docker build -f audio_engine/Dockerfile \
  -t cardio-audio-engine:latest audio_engine
```

Start the image with the same model path used by Compose. `./models` should
contain a compatible trained ONNX model named `pcg_classifier.onnx` and its
metadata sidecar `pcg_classifier.onnx.json` for successful inference:

```sh
mkdir -p models mlruns
docker run --rm --name cardio-audio-engine-test \
  -p 8001:8000 \
  -e ONNX_MODEL_PATH=/models/pcg_classifier.onnx \
  -e MLFLOW_TRACKING_URI=file:/mlruns \
  -v "$PWD/models:/models" \
  -v "$PWD/mlruns:/mlruns" \
  cardio-audio-engine:latest
```

In another terminal, check that the service starts and reports its configured
pipeline:

```sh
curl -f http://localhost:8001/health
curl -f http://localhost:8001/v1/config
curl -i -F 'file=@recording.wav' http://localhost:8001/v1/infer
```

With a compatible model mounted, inference should return HTTP 200 and a
`prediction` object. Without one, `/health` still responds but reports
`onnx_model_ready: false`, while inference returns HTTP 503. That confirms
service startup and the missing-model behavior; it does not verify model
inference. The six-feature file in `artifacts/` is only a test fixture and
does not match the service's 40-feature input.

To build and start both microservices through Compose instead:

```sh
docker compose up --build
curl -f http://localhost:8001/health
curl -i -F 'file=@recording.wav' http://localhost:8000/analyze_pcg
```

The gateway forwards the upload to the engine. It returns the engine's
successful prediction when a compatible model is available, or propagates the
engine's HTTP error otherwise. Stop Compose with `docker compose down`.

Install test dependencies with
`pip install -r audio_engine/requirements-dev.txt`. Run tests from the root
with `python -m pytest -q`. Tests use small synthetic signals; no dataset, GPU,
or pretrained download is required. This checkout's
current execution environment did not have pytest or the service dependencies
installed, so tests and container builds could not be executed here. No model
was trained or validated in this environment. These components are research
software and do not provide a medical diagnosis.
