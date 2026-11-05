from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from llm_finetuning.inference.ollama_client import (
    OllamaClient,
    OllamaModelUnavailableError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=16_384)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str


def create_app(ollama_client: OllamaClient | None = None) -> FastAPI:
    app = FastAPI(
        title="Cardio LLM Inference API",
        description="Chat inference API backed by the Ollama model service",
        version="1.0.0",
    )
    app.state.ollama_client = (
        ollama_client if ollama_client is not None else OllamaClient.from_env()
    )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        client: OllamaClient = request.app.state.ollama_client
        try:
            await client.health()
        except OllamaModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OllamaTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except OllamaUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OllamaResponseError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return HealthResponse(model=client.model)

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        client: OllamaClient = request.app.state.ollama_client
        try:
            result = await client.chat(
                messages=[message.model_dump() for message in payload.messages],
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            )
        except OllamaTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except OllamaUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OllamaResponseError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ChatResponse.model_validate(result)

    return app


app = create_app()
