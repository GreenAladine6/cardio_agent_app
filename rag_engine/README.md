# RAG Vector Store

Qdrant stores and searches vectors; it does not create embeddings. The RAG
engine generates text embeddings with
`sentence-transformers/all-MiniLM-L6-v2`, a compact sentence-transformers
model that returns 384-dimensional vectors. `embeddings.py` handles model
loading and embedding generation, while `vector_store.py` only stores and
retrieves vectors and document metadata.

## Start Qdrant

From the repository root:

```bash
docker compose up -d qdrant
curl http://localhost:6333/healthz
```

Qdrant's REST API is available on port `6333` and gRPC on `6334`. Its data is
persisted in the `qdrant-data` Docker volume.

## Configure and use

Install the RAG dependencies with:

```bash
python -m pip install -r rag_engine/requirements.txt
```

Configuration is read from `QDRANT_URL` (default
`http://localhost:6333`) and `QDRANT_COLLECTION_NAME` (default
`clinical_guidelines`). Generate vectors with `embed_text(text)`, then pass
those vectors with the original passage, source document, page number, and
chunk ID to `insert_documents`. `search_similar_documents` returns matching
passages with their metadata and similarity scores.

The Qdrant collection uses cosine similarity and the same 384-vector
dimension as the embedding model.

## Test

With Qdrant running, from the repository root run:

```bash
python -m pytest -m integration tests/test_vector_store.py
```

The integration test inserts dummy vectors and metadata, checks the nearest
match, and removes only its temporary test collection. It does not download
clinical documents or load the embedding model.
