from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agents.knowledge_agent import CollectionNotFoundError, QdrantUnavailableError
from agents.reasoning_agent import ReasoningAgentError
from orchestrator import CardioAgentOrchestrator


class OrchestrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_engine_result: dict
    user_question: str = Field(min_length=1, max_length=4000)


app = FastAPI(
    title="CardioAgent Orchestrator",
    description="Coordinates audio validation, guideline retrieval, and cautious reasoning.",
    version="1.0.0",
)
orchestrator = CardioAgentOrchestrator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent-orchestrator"}


@app.post("/v1/orchestrate")
def orchestrate(payload: OrchestrationRequest):
    try:
        result = orchestrator.run(
            audio_engine_result=payload.audio_engine_result,
            user_question=payload.user_question,
        )
    except (QdrantUnavailableError, CollectionNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ReasoningAgentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result.model_dump() if hasattr(result, "model_dump") else result
