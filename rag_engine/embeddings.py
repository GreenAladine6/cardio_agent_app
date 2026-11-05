from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _load_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Text to embed must not be empty")

    vector = _load_model().encode(text, normalize_embeddings=True).tolist()
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding model returned {len(vector)} values; "
            f"expected {EMBEDDING_DIMENSION}"
        )
    return vector
