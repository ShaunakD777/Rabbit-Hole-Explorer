"""Base class and shared types for all source connectors."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


SourceType = Literal["wikipedia", "youtube", "reddit", "paper", "web"]


@dataclass
class RawDocument:
    source_type: SourceType
    url: str
    title: str
    raw_text: str
    author_or_channel: Optional[str] = None
    published_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_cached_dict(cls, d: dict) -> "RawDocument":
        """
        Reconstruct from a dict round-tripped through Redis (every connector's
        _to_dict() serializes published_at with .isoformat()). Plain
        RawDocument(**d) leaves published_at as that ISO string on a cache hit --
        a real datetime on a fresh fetch, a string after any cache hit -- so
        callers that assume it's always a datetime (as the type hint promises)
        crash on cached results. aggregator.persist_documents() already works
        around this with its own isinstance(str) check; this fixes it at the
        source so every other caller gets a real datetime | None too.
        """
        published_at = d.get("published_at")
        if isinstance(published_at, str):
            published_at = datetime.fromisoformat(published_at)
        return cls(**{**d, "published_at": published_at})


class BaseConnector:
    """All connectors implement fetch(query) → list[RawDocument]."""

    source_type: SourceType = "web"

    async def fetch(self, query: str, max_results: int = 5) -> list[RawDocument]:
        raise NotImplementedError
