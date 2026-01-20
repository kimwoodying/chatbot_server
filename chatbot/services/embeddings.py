from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from chatbot.config import get_settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embeddings for the given texts as Python lists."""
    if not texts:
        return []
    model = _load_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()
