"""
Phase 5: Multi-depth grounded summarization (RAG-style).
Retrieves top-k relevant source chunks via pgvector, then calls Claude to summarize.
"""
from __future__ import annotations
import logging
import re
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.config import get_settings
from app.cache import cache_get, cache_set, llm_summary_key
from app.db.models import Node, Summary, SourceChunk, NodeSource
from app.nlp.llm import complete_text, AllProvidersFailed

logger = logging.getLogger(__name__)
settings = get_settings()

SummaryDepth = Literal["2min", "10min", "deepdive"]

DEPTH_CONFIG = {
    "2min":     {"max_chunks": 3,  "max_words": 150,  "label": "2-minute overview"},
    "10min":    {"max_chunks": 6,  "max_words": 500,  "label": "10-minute explanation"},
    "deepdive": {"max_chunks": 12, "max_words": 1200, "label": "deep-dive with citations"},
}

SUMMARY_PROMPT = """\
You are a clear, accurate technical writer. Write a {label} of the concept "{concept}".

Use ONLY the provided source passages to ground your answer. \
Do not invent facts not present in the passages.
For deepdive summaries, add inline citation markers like [1], [2] matching the passage numbers.

Target length: ~{max_words} words.

--- Source Passages ---
{passages}
--- End Passages ---

Write the summary now:
"""


async def get_or_generate_summary(
    node: Node,
    depth: SummaryDepth,
    db: AsyncSession,
) -> Summary:
    """Return cached summary or generate + persist a new one."""
    # Check DB first
    stmt = select(Summary).where(Summary.node_id == node.id, Summary.depth == depth)
    existing = (await db.execute(stmt)).scalars().first()
    if existing:
        return existing

    # Check Redis
    cache_k = llm_summary_key(str(node.id), depth)
    cached = await cache_get(cache_k)
    if cached:
        s = Summary(node_id=node.id, depth=depth,
                    content=cached["content"], citations=cached["citations"])
        db.add(s)
        await db.flush()
        return s

    # Generate
    content, citations = await _generate_summary(node, depth, db)

    s = Summary(node_id=node.id, depth=depth, content=content, citations=citations)
    db.add(s)
    await db.flush()
    await cache_set(cache_k, {"content": content, "citations": citations},
                    ttl=settings.ttl_llm_cache)
    return s


async def _generate_summary(
    node: Node,
    depth: SummaryDepth,
    db: AsyncSession,
) -> tuple[str, list[dict]]:
    cfg = DEPTH_CONFIG[depth]

    # Retrieve relevant chunks via pgvector cosine distance
    if node.embedding is not None:
        vec_str = "[" + ",".join(str(x) for x in node.embedding) + "]"
        # NOTE: the space before "::vector" is load-bearing -- see search.py's
        # identical query for the full explanation. Without it, SQLAlchemy's
        # text() bind-parameter parser silently reads the bind name as "ve"
        # instead of "vec" and sends malformed SQL to asyncpg.
        raw_sql = text("""
            SELECT sc.id, sc.chunk_text, sc.source_document_id,
                   1 - (sc.embedding <=> :vec ::vector) AS similarity
            FROM source_chunks sc
            JOIN node_sources ns ON ns.source_document_id = sc.source_document_id
            WHERE ns.node_id = :node_id
            ORDER BY similarity DESC
            LIMIT :k
        """)
        result = await db.execute(raw_sql, {"node_id": str(node.id), "k": cfg["max_chunks"], "vec": vec_str})
        rows = result.fetchall()
    else:
        # Fallback: grab any chunks from linked source docs
        rows = []

    passages = []
    citations: list[dict] = []
    for i, row in enumerate(rows, 1):
        passages.append(f"[{i}] {row.chunk_text}")
        citations.append({"index": i, "source_document_id": str(row.source_document_id)})

    if not passages:
        # Last resort: use node description
        content = node.description_short or f"No content available for {node.label}."
        return content, []

    prompt = SUMMARY_PROMPT.format(
        label=cfg["label"],
        concept=node.label,
        max_words=cfg["max_words"],
        passages="\n\n".join(passages),
    )
    try:
        content = await complete_text(prompt, max_tokens=cfg["max_words"] * 2)
        logger.info("Generated %s summary for node '%s' (%d chars)", depth, node.label, len(content))
        return content, citations
    except AllProvidersFailed as exc:
        logger.warning("No LLM provider available for summary, using raw passages: %s", exc)
        content = f"**{node.label}**\n\n" + "\n\n".join(passages[:3])
        return content, citations
