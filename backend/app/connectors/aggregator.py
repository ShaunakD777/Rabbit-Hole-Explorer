"""Aggregates all source connectors, runs them in parallel, deduplicates results."""
import asyncio
import logging
import time
from collections import Counter, defaultdict
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
    async def _timed_fetch(connector) -> list[RawDocument]:
        name = type(connector).__name__
        t0 = time.monotonic()
        try:
            result = await connector.fetch(query, max_per_source)
        except Exception as exc:
            logger.warning("Source %s FAILED after %.2fs: %s: %s", name,
                           time.monotonic() - t0, type(exc).__name__, exc)
            return []
        # 0 docs is normal for a connector with no key configured (it skips silently),
        # so say so explicitly rather than leaving it indistinguishable from a failure.
        logger.info("Source %s returned %d doc(s) (%d chars) in %.2fs", name, len(result),
                    sum(len(d.raw_text or "") for d in result), time.monotonic() - t0)
        return result

    results = await asyncio.gather(*(_timed_fetch(c) for c in _CONNECTORS))

    seen_urls: set[str] = set()
    docs: list[RawDocument] = []
    duplicates = 0
    for result in results:
        for doc in result:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                docs.append(doc)
            else:
                duplicates += 1

    by_type = Counter(d.source_type for d in docs)
    logger.info("Aggregated %d unique documents for query '%s' (%d duplicate URLs dropped): %s",
                len(docs), query, duplicates, dict(by_type))
    if not docs:
        logger.error("No source documents were fetched for '%s' -- the pipeline cannot build a "
                     "graph without any", query)
    return docs


async def persist_documents(docs: list[RawDocument], db: AsyncSession) -> list[SourceDocument]:
    """Store RawDocuments in the DB, skip duplicates by URL."""
    from sqlalchemy import select
    persisted: list[SourceDocument] = []
    reused = 0
    for doc in docs:
        # Check if URL already exists
        stmt = select(SourceDocument).where(SourceDocument.url == doc.url)
        existing = (await db.execute(stmt)).scalars().first()
        if existing:
            # A row stored by an older, degraded fetch (e.g. a Wikipedia article saved
            # as its one-sentence summary before the redirects fix) would otherwise be
            # reused forever, since URLs never change. Take the fresh text when it's
            # longer.
            if doc.raw_text and len(doc.raw_text) > len(existing.raw_text or ""):
                logger.info("Refreshing stored text for %s (%d -> %d chars)", doc.url,
                            len(existing.raw_text or ""), len(doc.raw_text))
                existing.raw_text = doc.raw_text
            persisted.append(existing)
            reused += 1
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
    logger.info("Persisted %d source document(s) (%d new, %d already in DB by URL)",
                len(persisted), len(persisted) - reused, reused)
    return persisted


# Order in which source types take their turn in build_combined_context's
# round-robin: introductory material first, research abstracts last. Unknown
# types slot in before "paper".
_SOURCE_PRIORITY = ["wikipedia", "youtube", "reddit", "web", "paper"]
# How many per_doc_chars-sized slices one turn takes. Encyclopedia articles are
# the one source written to introduce a topic, so a Wikipedia turn takes the
# first ~2400 chars of an article (its lead + first sections) instead of 800.
_SLICE_MULTIPLIER = {"wikipedia": 3}
# Docs shorter than this (search-result snippets, stub extracts) are left out of
# the context whenever anything longer is available -- a 150-char SERP snippet
# carries almost no teachable content but still costs a full round-robin turn.
MIN_CONTEXT_DOC_CHARS = 300


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

    Equal turns are not equal value for a learner, though: with pure round-robin
    a topic like "association football" got ~70% of its context from arXiv
    abstracts (robot vision, concussion mouthguards), and the extracted "concepts"
    were those papers' subjects. So turns go in _SOURCE_PRIORITY order (papers
    last, so they only fill what's left of the budget), Wikipedia turns are
    larger (_SLICE_MULTIPLIER), and snippet-sized docs are skipped when longer
    material exists (MIN_CONTEXT_DOC_CHARS).
    """
    with_text = [d for d in docs if d.raw_text]
    substantial = [d for d in with_text if len(d.raw_text) >= MIN_CONTEXT_DOC_CHARS]
    usable = substantial or with_text  # never return nothing just for being short

    by_type: dict[str, list[SourceDocument]] = defaultdict(list)
    for doc in usable:
        by_type[doc.source_type].append(doc)
    available = {t: len(v) for t, v in by_type.items()}

    def _priority(source_type: str) -> float:
        if source_type in _SOURCE_PRIORITY:
            return _SOURCE_PRIORITY.index(source_type)
        return _SOURCE_PRIORITY.index("paper") - 0.5  # unknown types: just before papers

    ordered_types = sorted(by_type, key=_priority)
    queues = [(t, by_type[t]) for t in ordered_types if by_type[t]]
    chunks: list[str] = []
    total = 0
    i = 0
    while queues and total < max_total_chars:
        idx = i % len(queues)
        source_type, queue = queues[idx]
        doc = queue.pop(0)
        piece = doc.raw_text[:per_doc_chars * _SLICE_MULTIPLIER.get(source_type, 1)]
        chunks.append(piece)
        total += len(piece)
        if queue:
            i += 1
        else:
            queues.pop(idx)

    combined = " ".join(chunks)[:max_total_chars]
    logger.debug("LLM context built: %d chars (budget %d) from %d slice(s); docs with text by "
                 "source: %s", len(combined), max_total_chars, len(chunks), available)
    return combined
