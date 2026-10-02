"""
Celery tasks for the async pipeline.

Task flow for a new topic:
  build_topic_graph
    ├── fetch_all_sources   (connectors/aggregator)
    ├── extract_concepts    (nlp/extraction)
    └── build_graph         (nlp/graph_builder)
          └── persist nodes/edges + embeddings
"""
import asyncio
import logging

from app.worker import celery_app

logger = logging.getLogger(__name__)


def _run(coro):
    """
    Run an async coroutine from a sync Celery task, in a fresh event loop.

    The SQLAlchemy async engine (app/db/session.py) is a module-level singleton
    whose pooled asyncpg connections are bound to whichever event loop was running
    when they were opened. asyncio.run() creates a brand-new loop on every call, so
    the *second* time this process runs a task -- including a Celery self.retry(),
    which lands back on the same forked worker process -- reusing that engine's
    pooled connections against the new loop raises "Future ... attached to a
    different loop" and the task fails outright, before its own logic ever runs.

    Disposing the engine's pool forces the next call to open fresh connections
    against whichever loop is current -- but the dispose() itself must happen
    *inside* the same loop that owns those connections, before that loop closes.
    Calling it after asyncio.run(coro) returns (in a second, separate asyncio.run())
    is too late: that loop is already gone, so the connections can't be closed
    cleanly either -- it just trades one "different loop" error for another.
    """
    async def _run_and_dispose():
        try:
            return await coro
        finally:
            from app.db.session import engine
            await engine.dispose()

    return asyncio.run(_run_and_dispose())


@celery_app.task(bind=True, name="tasks.build_topic_graph", max_retries=2)
def build_topic_graph(self, topic_id: str, prior_knowledge: list[str] | None = None):
    """
    Full pipeline: sources → extraction → graph construction.
    Updates topic.status on completion or failure.

    `prior_knowledge` (concepts the user says they already know) is passed as a task
    argument rather than persisted on Topic -- it only matters for biasing this one
    build, not as a durable record, so it avoids a schema change.
    """
    async def _run_pipeline():
        from app.db.session import AsyncSessionLocal
        from app.db.models import Topic
        from sqlalchemy import select
        from app.connectors.aggregator import fetch_all_sources, persist_documents, build_combined_context
        from app.nlp.extraction import extract_concepts
        from app.nlp.graph_builder import build_graph
        from app.api.ws import broadcast

        async with AsyncSessionLocal() as db:
            # Load topic
            stmt = select(Topic).where(Topic.id == topic_id)
            topic = (await db.execute(stmt)).scalars().first()
            if not topic:
                logger.error("Topic %s not found", topic_id)
                return

            try:
                # Notify frontend: step 1
                await broadcast(topic_id, {"event": "progress", "step": 1,
                                           "message": "Searching sources..."})

                # Step 1: Fetch sources
                raw_docs = await fetch_all_sources(topic.raw_query, max_per_source=5)
                source_docs = await persist_documents(raw_docs, db)

                # Notify frontend: step 2
                await broadcast(topic_id, {"event": "progress", "step": 2,
                                           "message": "Extracting concepts..."})

                # Step 2: Extract concepts
                combined_text = build_combined_context(source_docs)
                concepts = await extract_concepts(
                    topic.raw_query, combined_text, prior_knowledge=prior_knowledge
                )

                # Notify frontend: step 3
                await broadcast(topic_id, {"event": "progress", "step": 3,
                                           "message": "Building knowledge graph..."})

                # Step 3: Build graph
                graph, learning_path = await build_graph(
                    topic, concepts, source_docs, db
                )

                topic.status = "ready"
                await db.commit()

                # Notify frontend: done
                await broadcast(topic_id, {
                    "event": "graph_ready",
                    "topic_id": topic_id,
                    "graph_id": str(graph.id),
                    "learning_path": learning_path,
                })
                logger.info("Graph built for topic %s (graph %s)", topic_id, graph.id)

                # Kick off trend detection in background
                detect_trends.delay(topic_id)

            except Exception as exc:
                logger.exception("Pipeline failed for topic %s: %s", topic_id, exc)
                topic.status = "failed"
                await db.commit()
                await broadcast(topic_id, {"event": "error", "message": str(exc)})
                raise self.retry(exc=exc, countdown=10)

    _run(_run_pipeline())


@celery_app.task(bind=True, name="tasks.expand_node", max_retries=2)
def expand_node(self, node_id: str, graph_id: str):
    """
    Rabbit-hole expansion: fetch more sources scoped to a node,
    extract new concepts, merge into existing graph.
    """
    async def _run_expansion():
        from app.db.session import AsyncSessionLocal
        from app.db.models import Node, Graph, Topic, Edge
        from sqlalchemy import select
        from app.connectors.aggregator import fetch_all_sources, persist_documents, build_combined_context
        from app.nlp.extraction import extract_concepts
        from app.nlp.graph_builder import apply_degree_centrality
        from app.nlp.embeddings import encode, cosine_similarity
        from app.api.ws import broadcast
        from app.config import get_settings

        settings = get_settings()

        async with AsyncSessionLocal() as db:
            node_stmt = select(Node).where(Node.id == node_id)
            node = (await db.execute(node_stmt)).scalars().first()
            if not node:
                return

            graph_stmt = select(Graph).where(Graph.id == graph_id)
            graph = (await db.execute(graph_stmt)).scalars().first()

            topic_stmt = select(Topic).where(Topic.id == graph.topic_id)
            topic = (await db.execute(topic_stmt)).scalars().first()
            topic_id = str(topic.id) if topic else None

            async def _broadcast_both(payload: dict) -> None:
                # The frontend only ever subscribes to the topic_id channel
                # (GraphWorkspace.tsx), so publish there too, not just graph_id.
                await broadcast(graph_id, payload)
                if topic_id:
                    await broadcast(topic_id, payload)

            try:
                # Enforce the expansion depth cap -- previously defined in config
                # (max_expansion_depth) but never checked, so expansion could
                # recurse indefinitely.
                current_depth = node.depth_level or 0
                if current_depth >= settings.max_expansion_depth:
                    await _broadcast_both({
                        "event": "expansion_complete",
                        "node_id": node_id,
                        "message": f"Maximum expansion depth "
                                   f"({settings.max_expansion_depth}) reached.",
                    })
                    return

                # Enforce a total graph-size cap -- the depth cap alone bounds
                # recursion depth, not total node count, since each expansion click
                # can add up to 6 nodes with no limit on how many times it's clicked.
                # Checked before fetching sources so an already-full graph doesn't
                # waste an API call.
                existing_nodes = (await db.execute(
                    select(Node).where(Node.graph_id == graph_id)
                )).scalars().all()
                remaining_budget = settings.max_graph_nodes - len(existing_nodes)
                if remaining_budget <= 0:
                    await _broadcast_both({
                        "event": "expansion_complete",
                        "node_id": node_id,
                        "message": f"Maximum graph size "
                                   f"({settings.max_graph_nodes} nodes) reached.",
                    })
                    return

                await _broadcast_both({"event": "expanding", "node_id": node_id,
                                       "message": f"Expanding '{node.label}'..."})

                raw_docs = await fetch_all_sources(node.label, max_per_source=3)
                source_docs = await persist_documents(raw_docs, db)

                combined_text = build_combined_context(source_docs, per_doc_chars=400, max_total_chars=2500)
                new_concepts = await extract_concepts(node.label, combined_text)

                # Filter out concepts that are near-duplicates of any existing node,
                # by embedding cosine similarity -- exact lowercase label matching
                # let semantically identical concepts (e.g. "Neural Network" vs.
                # "Neural Networks") both into the graph.
                candidate_labels = [c["label"] for c in new_concepts]
                candidate_vecs = encode(candidate_labels) if candidate_labels else []

                filtered: list[dict] = []
                filtered_vecs: list[list] = []
                per_call_cap = min(6, remaining_budget)
                for c, vec in zip(new_concepts, candidate_vecs):
                    is_dup = any(
                        existing.embedding is not None
                        and cosine_similarity(vec, existing.embedding)
                        >= settings.node_dedup_threshold
                        for existing in existing_nodes
                    )
                    if not is_dup:
                        filtered.append(c)
                        filtered_vecs.append(vec)
                    if len(filtered) >= per_call_cap:
                        break

                if filtered:
                    # Add new nodes at depth = parent depth + 1
                    new_nodes = []
                    for c, vec in zip(filtered, filtered_vecs):
                        new_node = Node(
                            graph_id=graph_id,
                            label=c["label"],
                            aliases=c.get("aliases", []),
                            description_short=c.get("description", ""),
                            embedding=vec,
                            category=c.get("category", "Other"),
                            depth_level=current_depth + 1,
                        )
                        db.add(new_node)
                        new_nodes.append(new_node)

                    await db.flush()

                    # Connect to parent
                    new_edges = []
                    for new_node in new_nodes:
                        edge = Edge(
                            graph_id=graph_id,
                            source_node_id=node.id,
                            target_node_id=new_node.id,
                            relation_type="subtopic_of",
                            confidence=0.8,
                        )
                        db.add(edge)
                        new_edges.append(edge)

                    await db.flush()

                    # Recompute centrality over the whole graph, not just the new
                    # nodes -- otherwise every expanded node keeps importance_score's
                    # default of 1.0 forever, and the parent's score never reflects
                    # its new children.
                    all_nodes = existing_nodes + new_nodes
                    all_edges = (await db.execute(
                        select(Edge).where(Edge.graph_id == graph_id)
                    )).scalars().all()
                    apply_degree_centrality(all_nodes, all_edges)

                    await db.commit()

                    await _broadcast_both({
                        "event": "nodes_added",
                        "parent_node_id": node_id,
                        "new_nodes": [
                            {"id": str(n.id), "label": n.label,
                             "description_short": n.description_short,
                             "depth_level": n.depth_level,
                             "importance_score": n.importance_score}
                            for n in new_nodes
                        ],
                        "new_edges": [
                            {"id": str(e.id), "source_node_id": str(e.source_node_id),
                             "target_node_id": str(e.target_node_id), "relation_type": e.relation_type,
                             "confidence": e.confidence}
                            for e in new_edges
                        ],
                    })
                else:
                    await _broadcast_both({
                        "event": "expansion_complete",
                        "node_id": node_id,
                        "message": "No new concepts found.",
                    })

            except Exception as exc:
                logger.exception("Expansion failed for node %s: %s", node_id, exc)
                raise self.retry(exc=exc, countdown=10)

    _run(_run_expansion())


@celery_app.task(bind=True, name="tasks.detect_trends", max_retries=1)
def detect_trends(self, topic_id: str):
    """
    Run BERTopic trend detection in the background.
    """
    async def _run_trends():
        from app.db.session import AsyncSessionLocal
        from app.nlp.trend_detector import detect_topic_trends

        async with AsyncSessionLocal() as db:
            try:
                await detect_topic_trends(topic_id, db)
            except Exception as exc:
                logger.exception("Trend detection failed for topic %s: %s", topic_id, exc)
                raise self.retry(exc=exc, countdown=30)

    _run(_run_trends())
