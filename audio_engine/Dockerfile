FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ONNX_MODEL_PATH=/models/pcg_classifier.onnx \
    MLFLOW_TRACKING_URI=file:/mlruns \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

COPY audio_engine/requirements.txt ./requirements.txt
# Docker caches this requirements layer after a successful install. Long
# timeouts/retries help with large scientific wheels on slow package mirrors.
RUN for i in 1 2 3 4 5; do \
      pip install --no-cache-dir --disable-pip-version-check \
        --timeout 120 --retries 10 -r requirements.txt && exit 0; \
      echo "pip install failed (attempt $i), retrying..."; sleep 5; \
    done; exit 1

COPY audio_engine/audio_engine ./audio_engine
COPY frontend/ /frontend/

RUN mkdir -p /models /mlruns

EXPOSE 8000

CMD ["uvicorn", "audio_engine.service:app", "--host", "0.0.0.0", "--port", "8000"]
