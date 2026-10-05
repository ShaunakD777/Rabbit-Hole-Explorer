"""
Phase 2–3: Build and persist a knowledge graph from extracted concepts.

Steps:
  1. Deduplicate concepts via embedding cosine similarity.
  2. Drop clear off-topic outliers (embedding similarity to the topic query).
  3. Cap to initial_graph_node_cap.
  4. Derive relations: the curriculum prerequisites extraction already returned
     (default), or an LLM relation-classification call.
  5. Keep only the root concept's connected component.
  6. Persist nodes + edges; compute the learning path from the root.
"""
from __future__ import annotations
import heapq
import logging
import uuid
from collections import defaultdict, deque

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.logging_utils import degrade
from app.db.models import Graph, Node, Edge, Topic
from app.nlp.embeddings import encode, cosine_similarity, chunk_text, encode_one
from app.nlp.extraction import classify_relation, classify_relations_ensemble, is_spacy_fallback
from app.connectors.aggregator import build_combined_context
from app.db.models import SourceDocument, SourceChunk, NodeSource

logger = logging.getLogger(__name__)
settings = get_settings()

# The off-topic filter never shrinks a graph below this many concepts.
MIN_GRAPH_NODES = 5
# The connectivity filter only drops nodes outside the root's component when that
# component has at least this many nodes; otherwise it keeps everything rather than
# reducing the graph to a stub.
MIN_ROOT_COMPONENT = 3
# Curriculum extraction doesn't report a per-edge confidence (asking for one adds
# tokens and the number wouldn't be calibrated anyway), so its prerequisite edges
# get this fixed value -- above relation_confidence_threshold, below the 1.0 that
# would claim certainty.
CURRICULUM_EDGE_CONFIDENCE = 0.8


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

    A merged concept's "prerequisites" (curriculum extraction) are folded into the
    survivor's, so merging never silently deletes a learning-path edge.

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
            sim = cosine_similarity(vec, centroid)
            if sim >= threshold:
                logger.info("Dedup: merged '%s' into '%s' (cosine %.3f >= %.2f)",
                            concept["label"], deduped[j]["label"], sim, threshold)
                deduped[j].setdefault("aliases", []).append(concept["label"])
                if concept.get("prerequisites"):
                    deduped[j]["prerequisites"] = [
                        *(deduped[j].get("prerequisites") or []), *concept["prerequisites"],
                    ]
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
    dedup) appears in the source text. Used only to cap spaCy-fallback concepts to
    initial_graph_node_cap -- spaCy's candidate order carries no importance signal.
    (Curriculum extraction lists concepts most-foundational-first, so for LLM
    output the cap keeps that order instead.)
    """
    text_lower = text.lower()

    def _salience(c: dict) -> int:
        return sum(text_lower.count(label.lower()) for label in [c["label"], *c.get("aliases", [])])

    return sorted(concepts, key=_salience, reverse=True)


def relevant_indices(concepts: list[dict], topic_query: str, threshold: float) -> list[int]:
    """
    Indices (order-preserving) of concepts whose embedding is at least `threshold`
    cosine-similar to the topic query, never fewer than MIN_GRAPH_NODES (the most
    similar ones are kept if the threshold would cut deeper).

    LLM concepts are compared as "label: description" -- a bare label like "Game
    Overview" says little on its own. spaCy-fallback concepts all share one
    templated description ("A key concept related to X") that would make every
    one look relevant, so they're compared by label only.
    """
    if len(concepts) <= MIN_GRAPH_NODES:
        return list(range(len(concepts)))

    label_only = is_spacy_fallback(concepts)
    texts = [c["label"] if label_only else f"{c['label']}: {c.get('description') or ''}"
             for c in concepts]
    topic_vec = encode_one(topic_query)
    scores = [cosine_similarity(topic_vec, v) for v in encode(texts)]

    keep = {i for i, s in enumerate(scores) if s >= threshold}
    if len(keep) < MIN_GRAPH_NODES:
        keep = set(sorted(range(len(concepts)), key=lambda i: scores[i], reverse=True)[:MIN_GRAPH_NODES])

    dropped = [(concepts[i]["label"], round(scores[i], 2)) for i in range(len(concepts)) if i not in keep]
    if dropped:
        logger.info("Relevance filter: dropped %d off-topic concept(s) below %.2f similarity to "
                    "'%s': %s", len(dropped), threshold, topic_query, dropped)
    return [i for i in range(len(concepts)) if i in keep]


def _norm_label(label: str) -> str:
    return " ".join(label.lower().split())


def curriculum_edges(concepts: list[dict]) -> list[dict]:
    """
    Turn each concept's "prerequisites" (labels, from curriculum extraction) into
    prerequisite_of edges in the same 0-based shape classify_relations_batched()
    returns: {source_index, target_index, relation, confidence}, where the source
    must be learned before the target. Labels match case/whitespace-insensitively,
    including any alias merged in during dedup; unknown labels and self-references
    are ignored. Returns [] when no concept carries prerequisites (spaCy fallback,
    or an older cached extraction result) -- callers fall back to classification.
    """
    index_of: dict[str, int] = {}
    for i, c in enumerate(concepts):
        for name in [c["label"], *c.get("aliases", [])]:
            index_of.setdefault(_norm_label(name), i)

    edges: list[dict] = []
    seen: set[tuple[int, int]] = set()
    unmatched = 0
    for i, c in enumerate(concepts):
        prereqs = c.get("prerequisites")
        if not isinstance(prereqs, list):
            continue
        for p in prereqs:
            j = index_of.get(_norm_label(p)) if isinstance(p, str) else None
            if j is None:
                unmatched += 1
                continue
            if j == i or (j, i) in seen:
                continue
            seen.add((j, i))
            edges.append({"source_index": j, "target_index": i,
                          "relation": "prerequisite_of", "confidence": CURRICULUM_EDGE_CONFIDENCE})
    if unmatched:
        logger.info("Curriculum edges: %d prerequisite label(s) didn't match any concept and "
                    "were ignored", unmatched)
    return edges


def project_edges(edges: list[dict], kept: list[int]) -> list[dict]:
    """
    Re-index edges onto the kept subset of concepts (kept[k] = old index of new
    index k). A prerequisite edge whose source was dropped (off-topic filter, cap,
    connectivity) is bridged through to that source's own kept prerequisites, so
    A -> B -> C with B dropped still yields A -> C rather than silently breaking
    the chain. Other relation types are simply dropped with their endpoint.
    """
    new_index = {old: new for new, old in enumerate(kept)}
    prereqs_of: dict[int, set[int]] = defaultdict(set)
    for e in edges:
        if e["relation"] == "prerequisite_of":
            prereqs_of[e["target_index"]].add(e["source_index"])

    def _kept_ancestors(j: int, seen: set[int]) -> set[int]:
        if j in new_index:
            return {j}
        found: set[int] = set()
        for k in prereqs_of[j]:
            if k not in seen:
                seen.add(k)
                found |= _kept_ancestors(k, seen)
        return found

    projected: list[dict] = []
    seen_pairs: set[tuple[int, int, str]] = set()
    for e in edges:
        s, t, rel = e["source_index"], e["target_index"], e["relation"]
        if t not in new_index:
            continue
        sources = _kept_ancestors(s, {s}) if rel == "prerequisite_of" else ({s} & new_index.keys())
        for src in sources:
            key = (new_index[src], new_index[t], rel)
            if src == t or key in seen_pairs:
                continue
            seen_pairs.add(key)
            projected.append({**e, "source_index": key[0], "target_index": key[1]})
    return projected


def one_edge_per_pair(edges: list[dict]) -> list[dict]:
    """
    Keep at most one edge per unordered concept pair: the highest-confidence one,
    preferring prerequisite_of on a tie (it's the relation the learning path uses).
    The batched classifier routinely returned two contradictory edges for one pair
    (e.g. "Game Overview prerequisite_of Pitch" AND "Pitch subtopic_of Game
    Overview"), which rendered as a tangle and carried no extra meaning.
    """
    best: dict[frozenset, dict] = {}
    for e in edges:
        pair = frozenset((e["source_index"], e["target_index"]))
        current = best.get(pair)
        rank = (e["confidence"], e["relation"] == "prerequisite_of")
        if current is None or rank > (current["confidence"], current["relation"] == "prerequisite_of"):
            best[pair] = e
    return list(best.values())


def connected_component(root: int, n: int, edges: list[dict]) -> set[int]:
    """Indices reachable from `root` treating every edge as undirected."""
    adj: dict[int, set[int]] = defaultdict(set)
    for e in edges:
        adj[e["source_index"]].add(e["target_index"])
        adj[e["target_index"]].add(e["source_index"])
    seen = {root}
    queue = deque([root])
    while queue:
        for nxt in adj[queue.popleft()]:
            if nxt not in seen and 0 <= nxt < n:
                seen.add(nxt)
                queue.append(nxt)
    return seen


async def _classify_pairwise(labels: list[str], context: str) -> list[dict]:
    """Legacy O(n^2 - n) mode, kept only for the ablation study (backend/eval/)."""
    items: list[dict] = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            result = await classify_relation(labels[i], labels[j], context)
            items.append({"source_index": i, "target_index": j,
                          "relation": result.get("relation", "related_to"),
                          "confidence": result.get("confidence", 0.5)})
    return items


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
    all_concepts = deduplicate_concepts(concepts, settings.node_dedup_threshold)
    logger.info("After dedup: %d concepts (from %d raw)", len(all_concepts), len(concepts))
    spacy_fallback = is_spacy_fallback(all_concepts)

    # Prerequisites over the full deduped set, before anything is dropped, so
    # project_edges() can bridge chains through dropped concepts.
    all_prereq_edges = curriculum_edges(all_concepts)

    # ------------------------------------------------------------------ #
    # 2. Off-topic filter + node cap
    # ------------------------------------------------------------------ #
    kept = relevant_indices(all_concepts, topic.raw_query, settings.concept_relevance_threshold)

    if len(kept) > settings.initial_graph_node_cap:
        cap = settings.initial_graph_node_cap
        if spacy_fallback:
            # Salience ranking is pure local string-counting (no LLM call), so it
            # can use every fetched doc's full text.
            salience_text = " ".join(doc.raw_text or "" for doc in source_docs)
            position = {id(all_concepts[i]): i for i in kept}
            ranked = [position[id(c)] for c in _rank_by_textual_salience(
                [all_concepts[i] for i in kept], salience_text)]
        else:
            ranked = kept  # curriculum order: most foundational first
        logger.info("Capped graph to %d nodes (initial_graph_node_cap); dropped: %s", cap,
                    [all_concepts[i]["label"] for i in ranked[cap:]])
        kept = sorted(ranked[:cap])

    selected = [all_concepts[i] for i in kept]
    labels = [c["label"] for c in selected]

    # ------------------------------------------------------------------ #
    # 3. Relations
    # ------------------------------------------------------------------ #
    mode = settings.relation_mode
    if mode == "curriculum" and all_prereq_edges:
        relations = project_edges(all_prereq_edges, kept)
        logger.info("Relations: %d prerequisite edge(s) from curriculum extraction (no extra "
                    "LLM call)", len(relations))
    else:
        # LLM-facing context: must stay within a prompt-sized budget, so it goes
        # through the same priority/round-robin sampling as concept extraction.
        context_snippet = build_combined_context(source_docs, per_doc_chars=400, max_total_chars=1500)
        if mode == "pairwise":
            # Kept only so the ablation study (backend/eval/) can compare against
            # the unmodified legacy behavior.
            relations = await _classify_pairwise(labels, context_snippet)
        else:
            if mode == "curriculum":
                logger.info("Relations: extraction returned no prerequisites (%s) -- falling "
                            "back to batched classification",
                            "spaCy fallback" if spacy_fallback else "older-format result")
            # A single LLM call (or a cross-provider consensus call -- see
            # classify_relations_ensemble's docstring) classifies every related
            # pair. See llm.py's module docstring for why pairwise is unworkable
            # on free tiers.
            relations = await classify_relations_ensemble(labels, context_snippet)

    proposed = len(relations)
    relations = [r for r in relations if r["confidence"] >= settings.relation_confidence_threshold]
    relations = one_edge_per_pair(relations)
    logger.info("Relations: %d proposed, %d kept (confidence >= %.2f, one edge per pair)",
                proposed, len(relations), settings.relation_confidence_threshold)

    # ------------------------------------------------------------------ #
    # 4. Root + connectivity filter
    # ------------------------------------------------------------------ #
    # Root: whichever concept's embedding is closest to the topic query itself,
    # rather than assuming the first extracted concept represents the topic.
    topic_vec = encode_one(topic.raw_query)
    root_idx = max(range(len(selected)), key=lambda i: cosine_similarity(topic_vec, selected[i]["_vec"]))

    if relations:
        component = connected_component(root_idx, len(selected), relations)
        if len(component) < len(selected):
            if len(component) >= min(MIN_ROOT_COMPONENT, len(selected)):
                # Nodes the learner can't reach from the topic's root are tangents
                # (e.g. a cluster of robotics papers in a football graph). Dropping
                # them replaces the old LLM "orphan repair" pass, which invented
                # edges to pull such tangents back in.
                logger.info("Connectivity: dropped %d node(s) outside the root's component: %s",
                            len(selected) - len(component),
                            [selected[i]["label"] for i in range(len(selected)) if i not in component])
                keep_local = sorted(component)
                relations = project_edges(relations, keep_local)
                root_idx = keep_local.index(root_idx)
                selected = [selected[i] for i in keep_local]
            else:
                degrade("graph", f"root '{selected[root_idx]['label']}' is in a component of only "
                                 f"{len(component)} node(s); keeping all {len(selected)} nodes, "
                                 f"some of them disconnected")
    else:
        degrade("graph", "graph has zero edges -- it will render as disconnected nodes and the "
                         "learning path falls back to distance/importance ordering")

    # ------------------------------------------------------------------ #
    # 5. Persist Graph, Nodes, Edges
    # ------------------------------------------------------------------ #
    graph = Graph(topic_id=topic.id, version=1)
    db.add(graph)
    await db.flush()  # get graph.id

    nodes: list[Node] = []
    for c in selected:
        node = Node(
            graph_id=graph.id,
            label=c["label"],
            aliases=c.get("aliases", []),
            description_short=c.get("description", ""),
            embedding=c["_vec"],
            category=c.get("category", "Context"),
            importance_score=1.0,
            depth_level=0,
        )
        db.add(node)
        nodes.append(node)
    await db.flush()  # get node IDs

    graph.root_node_id = nodes[root_idx].id
    logger.info("Persisted %d nodes (graph %s); root node = '%s'; nodes: %s",
                len(nodes), graph.id, nodes[root_idx].label, [n.label for n in nodes])

    edges: list[Edge] = []
    for r in relations:
        edge = Edge(
            graph_id=graph.id,
            source_node_id=nodes[r["source_index"]].id,
            target_node_id=nodes[r["target_index"]].id,
            relation_type=r["relation"],
            confidence=r["confidence"],
        )
        db.add(edge)
        edges.append(edge)

    apply_degree_centrality(nodes, edges)
    await db.flush()
    by_type: dict[str, int] = defaultdict(int)
    for e in edges:
        by_type[e.relation_type] += 1
    logger.info("Edges persisted: %d total %s; centrality applied", len(edges), dict(by_type))

    # ------------------------------------------------------------------ #
    # 6. Chunk + embed source docs and link to nodes
    # ------------------------------------------------------------------ #
    await _embed_and_link_sources(nodes, source_docs, db)

    # ------------------------------------------------------------------ #
    # 7. Learning path
    # ------------------------------------------------------------------ #
    learning_path = _topological_sort(nodes, edges, graph.root_node_id)
    logger.info("Learning path computed: %d nodes", len(learning_path))

    return graph, [str(nid) for nid in learning_path]


async def _embed_and_link_sources(
    nodes: list[Node],
    source_docs: list[SourceDocument],
    db: AsyncSession,
) -> None:
    """Chunk source docs, embed, and create NodeSource links via similarity."""
    n_chunks = n_links = n_skipped = 0
    for doc in source_docs:
        if not doc.raw_text:
            n_skipped += 1
            continue
        chunks = chunk_text(doc.raw_text, chunk_size=300, overlap=30)
        chunk_vecs = encode(chunks)
        n_chunks += len(chunks)

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
                n_links += 1

    await db.flush()
    logger.info("RAG index: %d chunks embedded from %d docs (%d without text skipped), "
                "%d doc->node links (similarity > 0.3)",
                n_chunks, len(source_docs) - n_skipped, n_skipped, n_links)
    if n_links == 0:
        # Without links, node summaries have no passages and fall back to the node
        # description (see summarizer.py).
        degrade("rag", "no source document linked to any node; summaries will have no grounding")


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


def _topological_sort(
    nodes: list[Node], edges: list[Edge], root_id: uuid.UUID | None = None,
) -> list[uuid.UUID]:
    """
    Kahn's algorithm over 'prerequisite_of' edges, with a deterministic priority
    deciding which *available* node comes next.

    Prerequisite edges only constrain some pairs -- in a sparse graph most nodes are
    unconstrained, and a plain FIFO Kahn's emitted those in whatever order the DB
    returned them (a football graph started with "Terminology Origins" and put
    "Rules and Regulations" 11th). Among the nodes whose prerequisites are all
    satisfied, the next one is chosen by:
      1. depth_level -- core concepts before rabbit-hole expansions (depth >= 1),
      2. distance from the root node over *all* edges (undirected) -- concepts
         closest to the topic itself come first,
      3. importance_score (degree centrality), descending,
      4. label, then id -- only so ties are stable.
    With no prerequisite edges at all, this is simply a sort by that key. Nodes
    stuck in a prerequisite cycle are appended at the end in the same key order.
    """
    adjacency: dict = defaultdict(set)
    for e in edges:
        adjacency[e.source_node_id].add(e.target_node_id)
        adjacency[e.target_node_id].add(e.source_node_id)

    node_ids = {n.id for n in nodes}
    distance: dict = {}
    if root_id is not None and root_id in node_ids:
        distance[root_id] = 0
        queue = deque([root_id])
        while queue:
            current = queue.popleft()
            for nxt in adjacency[current]:
                if nxt not in distance:
                    distance[nxt] = distance[current] + 1
                    queue.append(nxt)
    unreachable = len(nodes)

    def _key(n: Node) -> tuple:
        return (n.depth_level or 0, distance.get(n.id, unreachable),
                -(n.importance_score or 0.0), (n.label or "").lower(), str(n.id))

    prereq_edges = [e for e in edges if e.relation_type == "prerequisite_of"
                    and e.source_node_id in node_ids and e.target_node_id in node_ids]
    if not prereq_edges:
        logger.info("Learning path: no 'prerequisite_of' edges, ordering by depth, distance from "
                    "root and importance")
        return [n.id for n in sorted(nodes, key=_key)]

    by_id = {n.id: n for n in nodes}
    in_degree: dict = {nid: 0 for nid in node_ids}
    successors: dict = defaultdict(list)
    for e in prereq_edges:
        successors[e.source_node_id].append(e.target_node_id)
        in_degree[e.target_node_id] += 1

    heap = [(_key(n), n.id) for n in nodes if in_degree[n.id] == 0]
    heapq.heapify(heap)
    order: list[uuid.UUID] = []
    while heap:
        _, nid = heapq.heappop(heap)
        order.append(nid)
        for successor in successors[nid]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                heapq.heappush(heap, (_key(by_id[successor]), successor))

    placed = set(order)
    remaining = sorted((n for n in nodes if n.id not in placed), key=_key)
    if remaining:
        logger.warning("Learning path: cycle among 'prerequisite_of' edges -- %d node(s) could "
                       "not be topologically ordered and were appended at the end", len(remaining))
    logger.info("Learning path: topological order over %d prerequisite edge(s)", len(prereq_edges))
    return order + [n.id for n in remaining]
