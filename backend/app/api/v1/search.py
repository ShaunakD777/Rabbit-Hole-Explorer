"""GET /graph/{graph_id}/search?q=... — semantic search within graph."""
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.session import get_db
from app.nlp.embeddings import encode_one
from app.schemas.topic import SearchResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{graph_id}/search", response_model=list[SearchResult])
async def search_graph(
    graph_id: UUID,
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Embed the query and return the most similar nodes in this graph."""
    query_vec = encode_one(q)
    vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"

    # NOTE: the space before "::vector" is load-bearing. SQLAlchemy's text()
    # bind-parameter parser mis-tokenizes ":vec::vector" (no space) -- it reads
    # the bind name as "ve", not "vec", silently sending malformed SQL to
    # asyncpg ("syntax error at or near ':'"). Confirmed via
    # text(sql)._bindparams.keys() and by actually running this query, not by
    # reading the code -- this executed successfully as written, in the sense
    # that it parsed to a *different*, wrong bind name, so the previous
    # parameterized-not-injectable claim was true but incomplete: it never
    # actually worked at all.
    raw_sql = text("""
        SELECT id, label,
               1 - (embedding <=> :vec ::vector) AS score
        FROM nodes
        WHERE graph_id = :graph_id
          AND embedding IS NOT NULL
        ORDER BY score DESC
        LIMIT :k
    """)
    result = await db.execute(raw_sql, {"graph_id": str(graph_id), "k": top_k, "vec": vec_str})
    rows = result.fetchall()

    return [
        SearchResult(node_id=row.id, label=row.label, score=float(row.score))
        for row in rows
    ]
