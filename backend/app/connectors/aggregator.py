"""Aggregates all source connectors, runs them in parallel, deduplicates results."""
import asyncio
import logging
from collections import defaultdict
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.base import RawDocument
from app.connectors.wikipedia import WikipediaConnector
from app.connectors.youtube import YouTubeConnector
from app.connectors.reddit import RedditConnector
from app.connectors.papers import PapersConnector
from app.connectors.web import WebConnector
from app.db.models import SourceDocument

logger = logging.getLogger(__name__)

_CONNECTORS = [
    WikipediaConnector(),
    YouTubeConnector(),
    RedditConnector(),
    PapersConnector(),
    WebConnector(),
]


async def fetch_all_sources(query: str, max_per_source: int = 5) -> list[RawDocument]:
    """Run all connectors in parallel, merge, deduplicate by URL."""
    tasks = [c.fetch(query, max_per_source) for c in _CONNECTORS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    docs: list[RawDocument] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("A connector raised an exception: %s", result)
            continue
        for doc in result:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                docs.append(doc)

    logger.info("Aggregated %d unique documents for query '%s'", len(docs), query)
    return docs


async def persist_documents(docs: list[RawDocument], db: AsyncSession) -> list[SourceDocument]:
    """Store RawDocuments in the DB, skip duplicates by URL."""
    from sqlalchemy import select
    persisted: list[SourceDocument] = []
    for doc in docs:
        # Check if URL already exists
        stmt = select(SourceDocument).where(SourceDocument.url == doc.url)
        existing = (await db.execute(stmt)).scalars().first()
        if existing:
            persisted.append(existing)
            continue

        pub_at: datetime | None = None
        if isinstance(doc.published_at, str):
            try:
                pub_at = datetime.fromisoformat(doc.published_at)
            except ValueError:
                pub_at = None
        else:
            pub_at = doc.published_at

        sd = SourceDocument(
            source_type=doc.source_type,
            url=doc.url,
            title=doc.title,
            author_or_channel=doc.author_or_channel,
            published_at=pub_at,
            raw_text=doc.raw_text,
            doc_metadata=doc.metadata,
        )
        db.add(sd)
        persisted.append(sd)

    await db.flush()
    return persisted


def build_combined_context(
    docs: list[SourceDocument],
    per_doc_chars: int = 800,
    max_total_chars: int = 6000,
) -> str:
    """
    Concatenate source text for an LLM prompt, round-robining one slice per
    source type at a time instead of taking documents in whatever order they
    happen to sit in `docs`.

    fetch_all_sources() always appends connector results in the fixed order
    Wikipedia -> YouTube -> Reddit -> Papers -> Web, and every LLM-facing caller
    (concept extraction, relation-classification context) can only afford to
    send a few thousand characters in one prompt. A flat `" ".join(doc.raw_text
    [:n] for doc in docs[:m])` means Wikipedia's articles silently monopolize
    that whole budget -- every other source (video transcripts, forum threads,
    papers, web snippets) never reaches the LLM at all, no matter how much of
    it was actually fetched. For a topic whose sources genuinely span distinct
    senses (e.g. an unflagged ambiguous query), this is what lets whichever
    source happened to load first silently define the whole topic.

    Round-robining one chunk per source type guarantees every source that
    returned anything gets a slice of the budget before any single source gets
    a second slice.
    """
    by_type: dict[str, list[SourceDocument]] = defaultdict(list)
    for doc in docs:
        if doc.raw_text:
            by_type[doc.source_type].append(doc)

    queues = [q for q in by_type.values() if q]
    chunks: list[str] = []
    total = 0
    i = 0
    while queues and total < max_total_chars:
        idx = i % len(queues)
        doc = queues[idx].pop(0)
        piece = doc.raw_text[:per_doc_chars]
        chunks.append(piece)
        total += len(piece)
        if queues[idx]:
            i += 1
        else:
            queues.pop(idx)

    return " ".join(chunks)[:max_total_chars]
