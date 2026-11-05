from uuid import uuid4

import pytest

from rag_engine.vector_store import (
    DocumentChunk,
    connect_to_qdrant,
    create_collection,
    delete_collection,
    insert_documents,
    search_similar_documents,
)

pytestmark = pytest.mark.integration


def test_qdrant_vector_store_round_trip():
    client = connect_to_qdrant()
    collection_name = f"test_guidelines_{uuid4().hex}"

    try:
        create_collection(client, collection_name)
        create_collection(client, collection_name)

        insert_documents(
            client,
            [
                DocumentChunk(
                    id=str(uuid4()),
                    embedding=[1.0] + [0.0] * 383,
                    text="Example guideline passage",
                    source_document="example-guideline.pdf",
                    page_number=2,
                    chunk_id="chunk-1",
                ),
                DocumentChunk(
                    id=str(uuid4()),
                    embedding=[0.0, 1.0] + [0.0] * 382,
                    text="Unrelated example passage",
                    source_document="other-guideline.pdf",
                    page_number=1,
                    chunk_id="chunk-2",
                ),
            ],
            collection_name,
        )

        results = search_similar_documents(
            client,
            query_vector=[1.0] + [0.0] * 383,
            collection_name=collection_name,
        )

        assert results[0]["text"] == "Example guideline passage"
        assert results[0]["source_document"] == "example-guideline.pdf"
        assert results[0]["page_number"] == 2
        assert results[0]["chunk_id"] == "chunk-1"
    finally:
        if client.collection_exists(collection_name):
            delete_collection(client, collection_name)
        client.close()
