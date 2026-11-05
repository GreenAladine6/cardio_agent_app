# Agent Orchestrator Guardrails

This folder contains a simple, three-agent orchestration flow for CardioAgent-Ops:

1. Audio Agent  
   Validates and normalizes Audio Engine output with Pydantic.
2. Knowledge Agent  
   Retrieves guideline passages from Qdrant through `rag_engine`.
3. Reasoning Agent  
   Builds final reasoning with the LLM server and applies final consistency and safety checks.

## What Pydantic guardrails do (and do not do)

Pydantic validates **structure and programmed rules**:
- required fields
- value ranges
- allowed labels
- non-empty explanations/limitations

Pydantic does **not** verify medical truth. It cannot prove whether a clinical statement is correct.

## Schemas

- `schemas/audio_schema.py`
- `schemas/knowledge_schema.py`
- `schemas/reasoning_schema.py`
- `schemas/final_response_schema.py`

## Guardrails

- `guardrails/input_guard.py`: input validation wrappers
- `guardrails/output_guard.py`: final response validation (+ one retry helper)
- `guardrails/safety_guard.py`: custom safety checks

Safety checks include:
- explanation must not be empty
- limitations are required
- output must not claim a confirmed diagnosis from ML prediction
- evidence must be present when evidence-based claims are made

## One-retry policy for invalid LLM output

The Reasoning Agent tries once to parse/validate structured JSON from the LLM.  
If invalid, it asks the LLM one more time with stricter instructions.  
If still invalid, it returns a controlled error.

## Environment variables

- `LLM_SERVER_URL` (default: `http://localhost:8002/v1/chat`)
- `LLM_TIMEOUT_SECONDS` (default: `30`)
- `QDRANT_COLLECTION_NAME` (default: `clinical_guidelines`)

## Run tests

From repository root:

```bash
/home/greenala/Documents/Cardio/api_getway/apigetwayenv/bin/python -m pytest -m unit tests/test_guardrails_schemas.py tests/test_agent_orchestrator.py
```
