"""POST /topics, POST /topics/{id}/clarify, GET /topics/{id}/graph"""
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import Topic, Graph, Node, Edge
from app.schemas.topic import TopicCreate, ClarifyChoice, TopicOut, GraphOut, NodeOut, EdgeOut
from app.tasks import build_topic_graph
from app.nlp.intent import check_ambiguity

logger = logging.getLogger(__name__)
router = APIRouter()


async def _start_pipeline(topic: Topic, prior_knowledge: list[str], db: AsyncSession) -> None:
    """Mark a topic processing and fire the Celery build task. Shared by the
    unambiguous POST /topics path and the post-clarification path."""
    topic.status = "processing"
    topic.clarification_options = None
    await db.flush()
    await db.commit()

    build_topic_graph.delay(str(topic.id), prior_knowledge)
    logger.info("Queued build_topic_graph for topic %s", topic.id)


@router.post("", response_model=TopicOut, status_code=status.HTTP_202_ACCEPTED)
async def create_topic(
    payload: TopicCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a topic. If the query is unambiguous, kicks off the async
    graph-building pipeline immediately. If it has multiple distinct, unrelated
    common meanings (see nlp/intent.py), returns with status="needs_clarification"
    and candidate senses instead of guessing -- the pipeline only starts once the
    caller resolves it via POST /topics/{id}/clarify.
    """
    candidates = await check_ambiguity(payload.raw_query)

    topic = Topic(
        raw_query=payload.raw_query,
        normalized_label=payload.raw_query.strip().title(),
        status="needs_clarification" if candidates else "processing",
        clarification_options=candidates,
    )
    db.add(topic)
    await db.flush()

    if candidates:
        await db.commit()
        logger.info("Topic %s needs clarification (%d candidates)", topic.id, len(candidates))
        return topic

    await _start_pipeline(topic, payload.prior_knowledge, db)
    return topic


@router.post("/{topic_id}/clarify", response_model=TopicOut, status_code=status.HTTP_202_ACCEPTED)
async def clarify_topic(
    topic_id: UUID,
    payload: ClarifyChoice,
    db: AsyncSession = Depends(get_db),
):
    """Resolve a topic stuck in needs_clarification to one chosen sense, then
    start the pipeline exactly as POST /topics does for an unambiguous query."""
    stmt = select(Topic).where(Topic.id == topic_id)
    topic = (await db.execute(stmt)).scalars().first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    if topic.status != "needs_clarification":
        raise HTTPException(
            status_code=409,
            detail=f"Topic is not awaiting clarification (status={topic.status!r})",
        )

    topic.raw_query = payload.chosen_query
    topic.normalized_label = payload.chosen_query.strip().title()
    await _start_pipeline(topic, payload.prior_knowledge, db)
    return topic


@router.get("/{topic_id}", response_model=TopicOut)
async def get_topic(topic_id: UUID, db: AsyncSession = Depends(get_db)):
    stmt = select(Topic).where(Topic.id == topic_id)
    topic = (await db.execute(stmt)).scalars().first()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    return topic


@router.get("/{topic_id}/graph", response_model=GraphOut)
async def get_graph(topic_id: UUID, db: AsyncSession = Depends(get_db)):
    """Return the latest graph version with all nodes and edges."""
    # Get latest graph for this topic
    graph_stmt = (
        select(Graph)
        .where(Graph.topic_id == topic_id)
        .order_by(Graph.version.desc())
    )
    graph = (await db.execute(graph_stmt)).scalars().first()
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not ready yet")

    nodes = (await db.execute(
        select(Node).where(Node.graph_id == graph.id)
    )).scalars().all()

    edges = (await db.execute(
        select(Edge).where(Edge.graph_id == graph.id)
    )).scalars().all()

    # Build learning path (topological sort of prereq edges, tie-broken from the root)
    from app.nlp.graph_builder import _topological_sort
    learning_path = _topological_sort(list(nodes), list(edges), graph.root_node_id)

    return GraphOut(
        graph_id=graph.id,
        topic_id=topic_id,
        version=graph.version,
        nodes=[NodeOut.model_validate(n) for n in nodes],
        edges=[EdgeOut.model_validate(e) for e in edges],
        learning_path=[str(nid) for nid in learning_path],
    )
