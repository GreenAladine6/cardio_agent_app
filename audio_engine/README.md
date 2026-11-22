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
downmixes multichannel audio by channel mean, resamples to 2,000 Hz, and peak
normalizes non-silent recordings. Inference segments longer recordings into
five-second windows with 50% overlap and pads shorter windows; each window
produces a 40-value float32 MFCC mean/std vector or a log-Mel tensor according
to model metadata. Files shorter than 0.25 seconds, longer than 120 seconds,
empty, malformed, or non-finite are rejected. Silences remain valid zero
signals.

The dataset configuration uses five-second windows at 50% overlap.
`prepare_dataset.py` assigns whole recordings or patient groups to
train/validation/test before creating windows. An optional metadata CSV can
provide `recording_id`, `patient_id`, `group_id`, and `source`; otherwise the
recording filename stem is the group key. Folder names are not treated as
evidence of dataset provenance.

## Dataset and training

No PCG recordings or professor TP notebooks are present in this checkout.
Supply authorized recordings and metadata in Google Drive; do not commit raw
recordings. In Colab, mount Drive and prepare the arrays with:

```sh
%cd /content/CardioAgent-Ops
!pip install -r audio_engine/requirements-colab.txt
!python -m audio_engine.prepare_dataset \
  --dataset-root /content/drive/MyDrive/ourdataset/train \
  --metadata-csv /content/drive/MyDrive/ourdataset/metadata.csv \
  --output-dir /content/pcg-export
```

The metadata CSV is optional. If supplied, it must include `relative_path`
matching a path beneath the dataset root; it can also contain patient/recording
identifiers and a configurable source value. The prepared export contains MFCC
and log-Mel arrays, labels, per-window manifests, a recording split manifest,
and preprocessing/label configuration.

Train in Colab with:

```sh
!python -m audio_engine.train_benchmark \
  --train-features /content/pcg-export/train_features.npy --train-labels /content/pcg-export/train_labels.npy \
  --val-features /content/pcg-export/val_features.npy --val-labels /content/pcg-export/val_labels.npy \
  --test-features /content/pcg-export/test_features.npy --test-labels /content/pcg-export/test_labels.npy \
  --train-mel /content/pcg-export/train_mel.npy --val-mel /content/pcg-export/val_mel.npy \
  --test-mel /content/pcg-export/test_mel.npy --output-dir /content/pcg-model-exports \
  --mlflow-uri file:/content/pcg-mlruns
```

`models.py` contains the Decision Tree, scaled RBF SVM, custom log-Mel CNN,
ResNet18, and CNN/Transformer hybrid. Neural models use CUDA when available
and CPU otherwise. Selection uses pathological validation recall first, then
F1, specificity, accuracy, and algorithm name. The winner is evaluated on the
held-out test set and written into a unique MLflow run directory. Classical
bundles include joblib and ONNX with a parity report. Neural bundles include a
PyTorch checkpoint and an ONNX export attempt; failures are recorded in the
metadata while preserving the checkpoint. Metrics, confusion matrices,
training/inference durations, parameter count, label mapping, and preprocessing
settings accompany the artifact. No training metrics are supplied until a
real dataset has been used.

Zip the selected run directory and the `pcg-export` manifest for transfer from
Colab. To install a winner locally, copy its `model.onnx` to
`models/pcg_classifier.onnx` and copy its `metadata.json` to
`models/pcg_classifier.onnx.json`. The serving endpoint reads `input_type` and
the label list from that sidecar; MFCC models receive the 40-value fixed
feature vector and neural models receive the configured log-Mel tensor. Keep
`model.pt` for PyTorch recovery when ONNX export is unavailable. Then start the
audio engine with `docker compose up --build audio-engine api-gateway`.

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
with a recording-level prediction, mean abnormal-class probability, segment
count, per-segment probabilities, and model timing.
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
