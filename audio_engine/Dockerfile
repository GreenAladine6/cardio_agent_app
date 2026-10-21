FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ONNX_MODEL_PATH=/models/pcg_classifier.onnx \
    MLFLOW_TRACKING_URI=file:/mlruns

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY audio_engine ./audio_engine

RUN mkdir -p /models /mlruns

EXPOSE 8000

CMD ["uvicorn", "audio_engine.service:app", "--host", "0.0.0.0", "--port", "8000"]
