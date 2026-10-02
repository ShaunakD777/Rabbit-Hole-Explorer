"""Singleton sentence-transformer encoder with pgvector helpers."""
from __future__ import annotations
import logging
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    logger.info("Loading sentence-transformer model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def encode(texts: Sequence[str]) -> list[list[float]]:
    """Encode a list of texts → list of embedding vectors."""
    model = get_encoder()
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def encode_one(text: str) -> list[float]:
    return encode([text])[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks
