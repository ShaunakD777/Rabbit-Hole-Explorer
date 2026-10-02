"""POST /progress/{node_id} — upsert user progress."""
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import UserProgress, Node
from app.schemas.topic import ProgressUpsert, ProgressOut

logger = logging.getLogger(__name__)
router = APIRouter()

# NOTE: In a real auth setup this would come from JWT. For the prototype
# we accept user_id as a path param to keep things simple.


@router.post("/{user_id}/{node_id}", response_model=ProgressOut)
async def upsert_progress(
    user_id: UUID,
    node_id: UUID,
    payload: ProgressUpsert,
    db: AsyncSession = Depends(get_db),
):
    node_check = (await db.execute(select(Node).where(Node.id == node_id))).scalars().first()
    if not node_check:
        raise HTTPException(status_code=404, detail="Node not found")

    stmt = select(UserProgress).where(
        UserProgress.user_id == user_id,
        UserProgress.node_id == node_id,
    )
    prog = (await db.execute(stmt)).scalars().first()

    if prog:
        prog.status = payload.status
    else:
        prog = UserProgress(user_id=user_id, node_id=node_id, status=payload.status)
        db.add(prog)

    await db.flush()
    await db.commit()
    return prog


@router.get("/{user_id}/graph/{graph_id}")
async def get_graph_progress(
    user_id: UUID,
    graph_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Return all progress records for a user on a given graph."""
    from app.db.models import Node
    stmt = (
        select(UserProgress)
        .join(Node, Node.id == UserProgress.node_id)
        .where(UserProgress.user_id == user_id, Node.graph_id == graph_id)
    )
    records = (await db.execute(stmt)).scalars().all()
    return [
        {"node_id": str(r.node_id), "status": r.status,
         "updated_at": r.updated_at.isoformat()}
        for r in records
    ]
