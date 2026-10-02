"""
Trend detection using BERTopic over recent papers.
"""
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from bertopic import BERTopic

from app.db.models import Graph, Node, NodeSource, SourceDocument, TrendSnapshot
from app.nlp.embeddings import encode_one, cosine_similarity

# Minimum cosine similarity to link a trend cluster to a node -- matches the
# similarity floor NodeSource linking uses elsewhere (graph_builder.py) so a
# cluster with no genuinely close node stays unlinked rather than mislinked.
_LINK_SIMILARITY_FLOOR = 0.25

logger = logging.getLogger(__name__)

async def detect_topic_trends(topic_id: str, db: AsyncSession):
    """
    1. Collect all paper SourceDocuments linked to the topic's nodes.
    2. Filter to last 24 months.
    3. Run BERTopic.
    4. Compute growth rate over the last 6 months vs previous 6 months.
    5. Save to TrendSnapshot.
    """
    # 1. Get the latest graph for the topic
    stmt_graph = select(Graph).where(Graph.topic_id == topic_id).order_by(Graph.version.desc())
    graph = (await db.execute(stmt_graph)).scalars().first()
    if not graph:
        logger.warning("No graph found for topic %s to detect trends.", topic_id)
        return

    # 2. Get all nodes in the graph
    stmt_nodes = select(Node).where(Node.graph_id == graph.id)
    nodes = (await db.execute(stmt_nodes)).scalars().all()
    node_ids = [n.id for n in nodes]
    if not node_ids:
        return

    # 3. Get all paper SourceDocuments linked to these nodes
    # We join NodeSource and SourceDocument
    stmt_docs = (
        select(SourceDocument)
        .join(NodeSource, NodeSource.source_document_id == SourceDocument.id)
        .where(NodeSource.node_id.in_(node_ids))
        .where(SourceDocument.source_type == "paper")
        .where(SourceDocument.published_at.is_not(None))
    )
    docs = (await db.execute(stmt_docs)).scalars().unique().all()
    
    if len(docs) < 10:
        logger.info("Not enough papers (%d) to run trend detection for topic %s", len(docs), topic_id)
        return

    # 4. Filter to last 24 months
    now = datetime.now(timezone.utc)
    two_years_ago = now - timedelta(days=730)
    six_months_ago = now - timedelta(days=180)
    
    recent_docs = [d for d in docs if d.published_at and d.published_at >= two_years_ago]
    if len(recent_docs) < 10:
        logger.info("Not enough recent papers (%d) for topic %s", len(recent_docs), topic_id)
        return

    # Extract abstracts (raw_text contains title + abstract usually)
    abstracts = [d.raw_text for d in recent_docs]

    # 5. Run BERTopic
    # We use a small embedding model or let BERTopic use its default (all-MiniLM-L6-v2)
    try:
        topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=False)
        topics, _ = topic_model.fit_transform(abstracts)
    except Exception as exc:
        logger.error("BERTopic failed for topic %s: %s", topic_id, exc)
        return

    # 6. Analyze clusters to compute growth rates
    topic_info = topic_model.get_topic_info()
    
    # Pre-calculate counts per topic in the two windows
    # Window 1: two_years_ago to six_months_ago
    # Window 2: six_months_ago to now
    counts_w1 = defaultdict(int)
    counts_w2 = defaultdict(int)
    
    for doc, topic_idx in zip(recent_docs, topics):
        if topic_idx == -1:
            continue # outlier
        if doc.published_at >= six_months_ago:
            counts_w2[topic_idx] += 1
        else:
            counts_w1[topic_idx] += 1
            
    # 7. Create TrendSnapshots
    trend_snapshots = []
    # Skip topic -1 (outliers)
    for _, row in topic_info.iterrows():
        t_id = row["Topic"]
        if t_id == -1:
            continue
            
        c1 = counts_w1[t_id]
        c2 = counts_w2[t_id]
        
        # Growth rate: (c2 - c1) / c1, default to 0 if no prior baseline, or a large number if c1=0 and c2>0
        if c1 == 0 and c2 > 0:
            growth_rate = 1.0 # arbitrary 100% growth if new
        elif c1 == 0 and c2 == 0:
            growth_rate = 0.0
        else:
            growth_rate = (c2 - c1) / c1
            
        # Get label from top words
        top_words = topic_model.get_topic(t_id)
        cluster_label = ", ".join([word for word, _ in top_words[:3]])

        snapshot = TrendSnapshot(
            topic_id=topic_id,
            cluster_label=cluster_label,
            growth_rate=growth_rate,
            window_start=six_months_ago.date(),
            window_end=now.date(),
            linked_node_id=None,  # resolved below, once we know which snapshots survive
        )
        trend_snapshots.append(snapshot)

    # Sort by growth rate descending, take top 5
    trend_snapshots.sort(key=lambda x: x.growth_rate, reverse=True)
    survivors = trend_snapshots[:5]

    # Link each surviving cluster to its nearest node by embedding similarity, and
    # flag that node as trending -- previously TrendSnapshot rows were created with
    # linked_node_id always NULL and Node.is_trending never set, so the frontend's
    # "Trending" badge (which reads node.is_trending) could never appear.
    nodes_with_embeddings = [(n, n.embedding) for n in nodes if n.embedding is not None]
    for ts in survivors:
        best_node, best_sim = None, 0.0
        if nodes_with_embeddings:
            cluster_vec = encode_one(ts.cluster_label)
            for n, vec in nodes_with_embeddings:
                sim = cosine_similarity(cluster_vec, vec)
                if sim > best_sim:
                    best_node, best_sim = n, sim
        if best_node is not None and best_sim >= _LINK_SIMILARITY_FLOOR:
            ts.linked_node_id = best_node.id
            best_node.is_trending = True
        db.add(ts)

    await db.commit()
    logger.info("Generated %d trend snapshots for topic %s", len(survivors), topic_id)
