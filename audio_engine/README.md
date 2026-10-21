# Service 3: PCG audio engine

FastAPI service intended for direct calls from `AudioAgent`. It resamples
recordings to 2,000 Hz with Librosa, peak-normalizes each clip, pads or crops to
eight seconds, and turns 20 MFCCs into a fixed 40-value vector (mean and
standard deviation per coefficient). Training and inference use the same
preprocessor.

## Run

Build from the repository root so Docker can copy the package directory:

```sh
docker build -f audio_engine/Dockerfile -t cardio-audio-engine .
docker run --rm -p 8001:8000 \
  -e MLFLOW_TRACKING_URI=file:/mlruns \
  -v ./models:/models -v ./mlruns:/mlruns \
  cardio-audio-engine
```

The service listens on port 8000 in the container. Set `ONNX_MODEL_PATH` to
change the model path, `MLFLOW_TRACKING_URI` to use a shared MLflow server, or
`MLFLOW_EXPERIMENT_NAME` to choose the tracking experiment. `/health` reports
whether a trained model is present, and `/v1/config` describes the feature
vector expected by models.

## API

- `POST /v1/infer`: multipart field `file` containing WAV audio. Returns the
  predicted class and available class probabilities. A trained model must be
  present first.
- `POST /v1/benchmark`: JSON body with 8 to 256 labeled audio samples:

  ```json
  {"samples": [{"label": "normal", "audio_base64": "<base64 WAV>"}]}
  ```

  Each class needs examples in the stratified training and holdout sets. The
  service compares logistic regression, RBF SVM, random forest, k-nearest
  neighbors, and a two-layer MLP. Accuracy, weighted F1, fit time, and inference
  time are tracked in MLflow. The best weighted-F1 model is exported to ONNX at
  `ONNX_MODEL_PATH`; class labels are stored next to it in a `.json` file.

Benchmarking needs a labeled dataset and can take time. No classifier is
shipped in this repository, so inference returns HTTP 503 until `/v1/benchmark`
has successfully trained and exported a model. Model outputs are predictions
from the supplied labels and data; they are not a clinical diagnosis.
