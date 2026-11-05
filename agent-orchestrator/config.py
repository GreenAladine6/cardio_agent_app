import os


LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://localhost:8002/v1/chat")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "clinical_guidelines")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
