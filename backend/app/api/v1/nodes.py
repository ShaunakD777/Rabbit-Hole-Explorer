"""GET /nodes/{id}/summary, POST /nodes/{id}/expand"""
import logging
from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Node, Graph
from app.schemas.topic import SummaryOut
from app.nlp.summarizer import get_or_generate_summary
from app.tasks import expand_node

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{node_id}/summary", response_model=SummaryOut)
async def get_summary(
    node_id: UUID,
    depth: Literal["2min", "10min", "deepdive"] = Query("2min"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Node).where(Node.id == node_id)
    node = (await db.execute(stmt)).scalars().first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    summary = await get_or_generate_summary(node, depth, db)
    await db.commit()
    return summary


@router.post("/{node_id}/expand", status_code=202)
async def trigger_expansion(
    node_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Queue a rabbit-hole expansion for this node."""
    stmt = select(Node).where(Node.id == node_id)
    node = (await db.execute(stmt)).scalars().first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    expand_node.delay(str(node_id), str(node.graph_id))
    return {"status": "queued", "node_id": str(node_id), "graph_id": str(node.graph_id)}


@router.get("/{node_id}/sources")
async def get_node_sources(node_id: UUID, db: AsyncSession = Depends(get_db)):
    """Return source documents linked to this node."""
    from app.db.models import NodeSource, SourceDocument
    stmt = (
        select(SourceDocument)
        .join(NodeSource, NodeSource.source_document_id == SourceDocument.id)
        .where(NodeSource.node_id == node_id)
        .order_by(NodeSource.relevance_score.desc())
    )
    docs = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": str(d.id),
            "source_type": d.source_type,
            "url": d.url,
            "title": d.title,
            "author_or_channel": d.author_or_channel,
            "published_at": d.published_at.isoformat() if d.published_at else None,
            "metadata": d.doc_metadata,
        }
        for d in docs
    ]
