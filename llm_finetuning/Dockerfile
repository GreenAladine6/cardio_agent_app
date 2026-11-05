FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY llm_finetuning/requirements-inference.txt ./requirements-inference.txt
RUN python -m pip install --no-cache-dir -r requirements-inference.txt

COPY llm_finetuning/ ./llm_finetuning/

RUN useradd --system --uid 10001 --create-home llm
USER llm

EXPOSE 8000

CMD ["uvicorn", "llm_finetuning.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
