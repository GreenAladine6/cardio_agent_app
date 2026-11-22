FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY agent-orchestrator/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY agent-orchestrator/ ./agent-orchestrator/
COPY rag_engine/ ./rag_engine/
ENV PYTHONPATH=/app/agent-orchestrator:/app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
