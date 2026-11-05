from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .config import COLLECTION_NAME, EMBEDDING_DIMENSION, QDRANT_URL


class QdrantUnavailableError(RuntimeError):
    """Qdrant could not be reached or returned a server error."""


class CollectionNotFoundError(RuntimeError):
    """The requested Qdrant collection does not exist."""


class EmbeddingDimensionError(ValueError):
    """An embedding does not match the configured model dimension."""


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    embedding: list[float]
    text: str
    source_document: str
    page_number: int
    chunk_id: str


def connect_to_qdrant() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, timeout=5)
    try:
        client.get_collections()
    except (httpx.HTTPError, ResponseHandlingException, UnexpectedResponse) as exc:
        client.close()
        raise QdrantUnavailableError(
            "Could not connect to Qdrant; check QDRANT_URL and service health"
        ) from exc
    return client


def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    try:
        if client.collection_exists(collection_name):
            return
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )
    except (httpx.HTTPError, ResponseHandlingException, UnexpectedResponse) as exc:
        raise QdrantUnavailableError("Could not create the Qdrant collection") from exc


def insert_documents(
    client: QdrantClient,
    documents: Sequence[DocumentChunk],
    collection_name: str = COLLECTION_NAME,
) -> None:
    if not documents:
        return

    points = []
    for document in documents:
        _validate_embedding(document.embedding)
        points.append(
            models.PointStruct(
                id=document.id,
                vector=document.embedding,
                payload={
                    "text": document.text,
                    "source_document": document.source_document,
                    "page_number": document.page_number,
                    "chunk_id": document.chunk_id,
                },
            )
        )

    _ensure_collection_exists(client, collection_name)
    try:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    except (httpx.HTTPError, ResponseHandlingException, UnexpectedResponse) as exc:
        raise QdrantUnavailableError("Could not insert documents into Qdrant") from exc


def search_similar_documents(
    client: QdrantClient,
    query_vector: list[float],
    limit: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    _validate_embedding(query_vector)
    if limit < 1:
        raise ValueError("Search limit must be at least 1")

    _ensure_collection_exists(client, collection_name)
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
    except (httpx.HTTPError, ResponseHandlingException, UnexpectedResponse) as exc:
        raise QdrantUnavailableError("Could not search the Qdrant collection") from exc

    return [
        {
            "id": point.id,
            **(point.payload or {}),
            "score": point.score,
        }
        for point in response.points
    ]


def delete_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    _ensure_collection_exists(client, collection_name)
    try:
        client.delete_collection(collection_name=collection_name)
    except (httpx.HTTPError, ResponseHandlingException, UnexpectedResponse) as exc:
        raise QdrantUnavailableError("Could not delete the Qdrant collection") from exc


def _ensure_collection_exists(client: QdrantClient, collection_name: str) -> None:
    try:
        exists = client.collection_exists(collection_name)
    except (httpx.HTTPError, ResponseHandlingException, UnexpectedResponse) as exc:
        raise QdrantUnavailableError(
            "Could not check the Qdrant collection; check service health"
        ) from exc
    if not exists:
        raise CollectionNotFoundError(
            f"Qdrant collection '{collection_name}' does not exist"
        )


def _validate_embedding(embedding: list[float]) -> None:
    if len(embedding) != EMBEDDING_DIMENSION:
        raise EmbeddingDimensionError(
            f"Expected an embedding with {EMBEDDING_DIMENSION} values, "
            f"received {len(embedding)}"
        )
    if not all(math.isfinite(value) for value in embedding):
        raise ValueError("Embedding values must be finite numbers")
