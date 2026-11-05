from __future__ import annotations

from pydantic import ValidationError

from config import QDRANT_COLLECTION_NAME
from rag_engine.embeddings import embed_text
from rag_engine.vector_store import (
    CollectionNotFoundError,
    QdrantUnavailableError,
    connect_to_qdrant,
    search_similar_documents,
)
from schemas.knowledge_schema import KnowledgeAgentResult


class KnowledgeAgent:
    def retrieve_passages(self, question: str, limit: int = 5) -> KnowledgeAgentResult:
        if not question.strip():
            raise ValueError("Question for retrieval must not be empty")

        client = connect_to_qdrant()
        try:
            query_vector = embed_text(question)
            raw_results = search_similar_documents(
                client=client,
                query_vector=query_vector,
                limit=limit,
                collection_name=QDRANT_COLLECTION_NAME,
            )
        except (QdrantUnavailableError, CollectionNotFoundError):
            raise
        finally:
            client.close()

        payload = {
            "passages": [
                {
                    "text": item["text"],
                    "source_document": item["source_document"],
                    "page_number": item["page_number"],
                    "chunk_id": item["chunk_id"],
                    "similarity_score": item.get("score"),
                }
                for item in raw_results
            ]
        }
        try:
            return KnowledgeAgentResult.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("Invalid retrieved passage payload") from exc
