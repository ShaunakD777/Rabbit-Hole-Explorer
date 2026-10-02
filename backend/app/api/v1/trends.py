"""GET /trends/{topic_id} — trend snapshots."""
import logging
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import TrendSnapshot
from app.schemas.topic import TrendOut

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{topic_id}", response_model=list[TrendOut])
async def get_trends(topic_id: UUID, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(TrendSnapshot)
        .where(TrendSnapshot.topic_id == topic_id)
        .order_by(TrendSnapshot.growth_rate.desc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [TrendOut.model_validate(r) for r in rows]
