"""
Phase 2–3: Build and persist a knowledge graph from extracted concepts.

Steps:
  1. Deduplicate concepts via embedding cosine similarity.
  2. Classify relations between all concept pairs (with shared context).
  3. Persist nodes + edges; return graph ID.
  4. Compute topological sort for learning path.
"""
from __future__ import annotations
import logging
import uuid
from collections import defaultdict, deque

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Graph, Node, Edge, Topic
from app.nlp.embeddings import encode, cosine_similarity, chunk_text, encode_one
from app.nlp.extraction import classify_relation, classify_relations_ensemble, classify_orphan_edges
from app.connectors.aggregator import build_combined_context
from app.db.models import SourceDocument, SourceChunk, NodeSource

logger = logging.getLogger(__name__)
settings = get_settings()


def deduplicate_concepts(concepts: list[dict], threshold: float) -> list[dict]:
    """
    Merge concepts whose label embeddings are cosine-similar above `threshold`.
    Each surviving concept dict gains "aliases" (labels merged into it) and "_vec"
    (its embedding, reused when persisting the Node). Order-preserving: a concept
    only ever merges into an *earlier* survivor, never a later one.

    Each cluster's comparison vector is a running centroid of everything merged
    into it so far, not the original anchor concept's embedding alone -- pure
    single-linkage-to-the-anchor is prone to chaining (e.g. A merges B, B would
    have merged C, but C never gets compared against B, only against A). Updating
    the centroid on every merge means later candidates are judged against the
    cluster's drift so far, which is less sensitive to extraction order.

    Pulled out of build_graph() (which also does DB writes) so this pure,
    order-preserving-merge logic can be unit tested without a database.
    """
    labels = [c["label"] for c in concepts]
    vectors = encode(labels)

    deduped: list[dict] = []
    centroids: list[list[float]] = []
    cluster_sizes: list[int] = []

    for concept, vec in zip(concepts, vectors):
        merged = False
        for j, centroid in enumerate(centroids):
            if cosine_similarity(vec, centroid) >= threshold:
                deduped[j].setdefault("aliases", []).append(concept["label"])
                n = cluster_sizes[j]
                centroids[j] = [(cv * n + v) / (n + 1) for cv, v in zip(centroid, vec)]
                cluster_sizes[j] += 1
                merged = True
                break
        if not merged:
            deduped.append({**concept, "aliases": []})
            centroids.append(list(vec))
            cluster_sizes.append(1)

    for c, centroid in zip(deduped, centroids):
        c["_vec"] = centroid

    return deduped


def _rank_by_textual_salience(concepts: list[dict], text: str) -> list[dict]:
    """
    Order concepts by how often their label (or any alias merged into it during
    dedup) appears in the source text. Used only when capping to
    initial_graph_node_cap, so the nodes kept are the ones the sources actually
    talk about most -- extraction order alone carries no importance signal.
    """
    text_lower = text.lower()

    def _salience(c: dict) -> int:
        return sum(text_lower.count(label.lower()) for label in [c["label"], *c.get("aliases", [])])

    return sorted(concepts, key=_salience, reverse=True)


async def build_graph(
    topic: Topic,
    concepts: list[dict],
    source_docs: list[SourceDocument],
    db: AsyncSession,
) -> tuple[Graph, list[str]]:
    """
    Build and persist the knowledge graph.
    Returns (Graph, ordered_node_ids_for_learning_path).
    """
    if not concepts:
        raise ValueError("No concepts extracted — cannot build graph.")

    # ------------------------------------------------------------------ #
    # 1. Embed concepts and deduplicate
    # ------------------------------------------------------------------ #
    deduped = deduplicate_concepts(concepts, settings.node_dedup_threshold)
    logger.info("After dedup: %d concepts (from %d raw)", len(deduped), len(concepts))

    # Salience ranking is pure local string-counting (no LLM call), so it can and
    # should use every fetched doc's full text, not just the first 5 in whatever
    # order the aggregator happened to return them in (fixed connector order --
    # see build_combined_context's docstring for why that order is misleading).
    salience_text = " ".join(doc.raw_text or "" for doc in source_docs)

    # Enforce the initial-graph node cap (previously defined in config but never
    # applied, and when first applied just took deduped[:cap] -- i.e. whichever
    # concepts the LLM happened to list first, with no importance signal at all).
    # Rank by textual salience in the actual source material first, so capping
    # drops the least-discussed concepts rather than an arbitrary suffix.
    if len(deduped) > settings.initial_graph_node_cap:
        deduped = _rank_by_textual_salience(deduped, salience_text)[: settings.initial_graph_node_cap]
        logger.info("Capped graph to %d nodes (initial_graph_node_cap, ranked by textual salience)",
                    settings.initial_graph_node_cap)

    # ------------------------------------------------------------------ #
    # 2. Create Graph record
    # ------------------------------------------------------------------ #
    graph = Graph(topic_id=topic.id, version=1)
    db.add(graph)
    await db.flush()  # get graph.id

    # ------------------------------------------------------------------ #
    # 3. Persist Nodes
    # ------------------------------------------------------------------ #
    nodes: list[Node] = []
    for i, c in enumerate(deduped):
        node = Node(
            graph_id=graph.id,
            label=c["label"],
            aliases=c.get("aliases", []),
            description_short=c.get("description", ""),
            embedding=c["_vec"],
            category=c.get("category", "Other"),
            importance_score=1.0,
            depth_level=0,
        )
        db.add(node)
        nodes.append(node)

    await db.flush()  # get node IDs

    # Set root node: whichever persisted node's embedding is closest to the topic
    # query itself, rather than assuming the first extracted concept represents
    # the topic -- extraction order carries no such guarantee.
    topic_vec = encode_one(topic.raw_query)
    root_node = max(nodes, key=lambda n: cosine_similarity(topic_vec, n.embedding))
    graph.root_node_id = root_node.id
    await db.flush()

    # ------------------------------------------------------------------ #
    # 4. Classify relations
    # ------------------------------------------------------------------ #
    edges: list[Edge] = []
    # LLM-facing, unlike salience_text above -- must stay within a prompt-sized
    # budget, so it goes through the same round-robin-by-source-type sampling as
    # concept extraction rather than a flat slice of whichever docs came first.
    context_snippet = build_combined_context(source_docs, per_doc_chars=400, max_total_chars=1500)

    if settings.relation_mode == "batched":
        # Default: a single LLM call (or, when multiple providers are configured,
        # a cross-provider consensus call -- see classify_relations_ensemble's
        # docstring) classifies every related pair. See llm.py's module docstring
        # for why the pairwise mode below is unworkable on free tiers.
        batched = await classify_relations_ensemble([n.label for n in nodes], context_snippet)
        for item in batched:
            if item["confidence"] < settings.relation_confidence_threshold:
                continue
            edge = Edge(
                graph_id=graph.id,
                source_node_id=nodes[item["source_index"]].id,
                target_node_id=nodes[item["target_index"]].id,
                relation_type=item["relation"],
                confidence=item["confidence"],
            )
            db.add(edge)
            edges.append(edge)

        # Repair orphans: any node relation classification left with zero edges
        # gets one focused follow-up call rather than staying permanently
        # disconnected in the persisted graph. Only in batched mode -- pairwise
        # exists solely so the ablation study can measure the *unmodified*
        # legacy behavior, and this would contaminate that comparison.
        connected_ids = {e.source_node_id for e in edges} | {e.target_node_id for e in edges}
        orphan_indices = [i for i, n in enumerate(nodes) if n.id not in connected_ids]
        if orphan_indices:
            orphan_edges = await classify_orphan_edges(
                [n.label for n in nodes], orphan_indices, context_snippet
            )
            for item in orphan_edges:
                if item["confidence"] < settings.relation_confidence_threshold:
                    continue
                edge = Edge(
                    graph_id=graph.id,
                    source_node_id=nodes[item["source_index"]].id,
                    target_node_id=nodes[item["target_index"]].id,
                    relation_type=item["relation"],
                    confidence=item["confidence"],
                )
                db.add(edge)
                edges.append(edge)
    else:
        # Legacy pairwise mode: O(n^2 - n) LLM calls. Kept only so the ablation study
        # (backend/eval/) can compare it against the batched mode above.
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                rel_result = await classify_relation(
                    nodes[i].label, nodes[j].label, context_snippet
                )
                confidence = rel_result.get("confidence", 0.5)
                if confidence < settings.relation_confidence_threshold:
                    continue

                edge = Edge(
                    graph_id=graph.id,
                    source_node_id=nodes[i].id,
                    target_node_id=nodes[j].id,
                    relation_type=rel_result.get("relation", "related_to"),
                    confidence=confidence,
                )
                db.add(edge)
                edges.append(edge)

    apply_degree_centrality(nodes, edges)
    await db.flush()

    # ------------------------------------------------------------------ #
    # 5. Chunk + embed source docs and link to nodes
    # ------------------------------------------------------------------ #
    await _embed_and_link_sources(nodes, source_docs, db)

    # ------------------------------------------------------------------ #
    # 6. Topological sort for learning path
    # ------------------------------------------------------------------ #
    learning_path = _topological_sort(nodes, edges)
    logger.info("Learning path computed: %d nodes", len(learning_path))

    return graph, [str(nid) for nid in learning_path]


async def _embed_and_link_sources(
    nodes: list[Node],
    source_docs: list[SourceDocument],
    db: AsyncSession,
) -> None:
    """Chunk source docs, embed, and create NodeSource links via similarity."""
    for doc in source_docs:
        if not doc.raw_text:
            continue
        chunks = chunk_text(doc.raw_text, chunk_size=300, overlap=30)
        chunk_vecs = encode(chunks)

        for idx, (chunk, vec) in enumerate(zip(chunks, chunk_vecs)):
            sc = SourceChunk(
                source_document_id=doc.id,
                chunk_text=chunk,
                chunk_index=idx,
                embedding=vec,
            )
            db.add(sc)

        # Link document to every node it's meaningfully similar to, not just the
        # single closest one -- node_sources is a many-to-many table, and a
        # source that discusses several concepts should inform RAG summaries
        # for all of them, not just whichever one happened to score highest.
        doc_vec = encode_one(doc.title or doc.raw_text[:200])
        for node in nodes:
            sim = cosine_similarity(doc_vec, node.embedding)
            if sim > 0.3:
                ns = NodeSource(
                    node_id=node.id,
                    source_document_id=doc.id,
                    relevance_score=sim,
                )
                db.add(ns)

    await db.flush()


def apply_degree_centrality(nodes: list[Node], edges: list[Edge]) -> None:
    """
    Set importance_score = (in_degree + out_degree) / (N - 1) on each node, in place.

    Extracted so tasks.py can recompute centrality over the *full* node set after a
    rabbit-hole expansion adds nodes -- previously this only ran once at initial build,
    so every expanded node kept importance_score's default of 1.0 forever.
    """
    if len(nodes) <= 1:
        return
    in_degree: dict = defaultdict(int)
    out_degree: dict = defaultdict(int)
    for e in edges:
        out_degree[e.source_node_id] += 1
        in_degree[e.target_node_id] += 1

    n_minus_1 = len(nodes) - 1
    for node in nodes:
        node.importance_score = (in_degree[node.id] + out_degree[node.id]) / n_minus_1


def _topological_sort(nodes: list[Node], edges: list[Edge]) -> list[uuid.UUID]:
    """
    Kahn's algorithm over 'prerequisite_of' edges.
    Falls back to importance-sorted list if the subgraph is empty or cyclic.
    """
    prereq_edges = [e for e in edges if e.relation_type == "prerequisite_of"]
    node_ids = [n.id for n in nodes]

    if not prereq_edges:
        # No prerequisites defined — sort by importance descending
        return [n.id for n in sorted(nodes, key=lambda n: n.importance_score, reverse=True)]

    # Build adjacency
    in_degree: dict = defaultdict(int)
    successors: dict = defaultdict(list)
    for nid in node_ids:
        in_degree[nid] = 0

    for e in prereq_edges:
        successors[e.source_node_id].append(e.target_node_id)
        in_degree[e.target_node_id] += 1

    queue = deque([nid for nid in node_ids if in_degree[nid] == 0])
    order: list[uuid.UUID] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for successor in successors[nid]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    # If cycle detected (order shorter than nodes), append remaining
    remaining = [nid for nid in node_ids if nid not in set(order)]
    order.extend(remaining)

    return order
