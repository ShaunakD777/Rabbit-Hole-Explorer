"""
Unit tests for the pure logic in app/nlp/graph_builder.py -- no database needed.
Node/Edge are plain SQLAlchemy declarative models; instantiating them with keyword
args (never flushing/committing) works with no DB connection at all.
"""
import uuid

import pytest

from app.db.models import Node, Edge
from app.nlp import graph_builder
from app.nlp.graph_builder import (
    deduplicate_concepts, apply_degree_centrality, _topological_sort,
    _rank_by_textual_salience, curriculum_edges, project_edges, one_edge_per_pair,
    connected_component, relevant_indices, CURRICULUM_EDGE_CONFIDENCE, MIN_GRAPH_NODES,
)


def _node(importance: float = 1.0, label: str = "", depth: int = 0) -> Node:
    return Node(id=uuid.uuid4(), importance_score=importance, label=label, depth_level=depth)


def _rel(source: int, target: int, relation: str = "prerequisite_of", confidence: float = 0.8) -> dict:
    return {"source_index": source, "target_index": target, "relation": relation,
            "confidence": confidence}


def _edge(source: Node, target: Node, relation_type: str = "related_to", confidence: float = 1.0) -> Edge:
    return Edge(source_node_id=source.id, target_node_id=target.id,
                relation_type=relation_type, confidence=confidence)


# --------------------------------------------------------------------------- #
# deduplicate_concepts
# --------------------------------------------------------------------------- #

def test_deduplicate_concepts_merges_exact_duplicate_labels():
    # The same label twice encodes to the same vector, so cosine similarity is
    # always exactly 1.0 -- deterministic, unlike asserting a specific similarity
    # for near-synonyms (e.g. singular/plural), which depends on model behavior
    # this test shouldn't need to assume.
    concepts = [
        {"label": "Neural Network", "description": "A", "category": "Fundamentals"},
        {"label": "Neural Network", "description": "B", "category": "Fundamentals"},
        {"label": "Quantum Computing", "description": "C", "category": "Fundamentals"},
    ]
    deduped = deduplicate_concepts(concepts, threshold=0.92)

    assert len(deduped) == 2
    survivor = next(c for c in deduped if c["label"] == "Neural Network")
    assert "Neural Network" in survivor["aliases"]
    assert all("_vec" in c for c in deduped)


def test_deduplicate_concepts_keeps_distinct_concepts_separate():
    concepts = [
        {"label": "Photosynthesis", "description": "A", "category": "Fundamentals"},
        {"label": "Blockchain Consensus", "description": "B", "category": "Fundamentals"},
    ]
    deduped = deduplicate_concepts(concepts, threshold=0.92)
    assert len(deduped) == 2
    assert all(c["aliases"] == [] for c in deduped)


def test_deduplicate_concepts_only_merges_into_an_earlier_survivor():
    # A concept must never merge into one that appears later in the input list.
    concepts = [{"label": f"Concept {i}"} for i in range(3)]
    deduped = deduplicate_concepts(concepts, threshold=1.1)  # impossible threshold -> no merges
    assert [c["label"] for c in deduped] == [c["label"] for c in concepts]


def test_deduplicate_concepts_centroid_averages_every_merged_member(monkeypatch):
    # A cluster's comparison vector should be the running average of everything
    # merged into it, not the original anchor's embedding alone -- otherwise a
    # later candidate close to member 2 but not to the anchor never gets a fair
    # comparison (single-linkage-to-the-anchor "chaining" problem).
    vectors = {"A": [1.0, 0.0], "B": [0.9, 0.1], "C": [0.8, 0.2]}
    monkeypatch.setattr(
        "app.nlp.graph_builder.encode",
        lambda labels: [vectors[label] for label in labels],
    )
    monkeypatch.setattr("app.nlp.graph_builder.cosine_similarity", lambda a, b: 1.0)

    concepts = [{"label": "A"}, {"label": "B"}, {"label": "C"}]
    deduped = deduplicate_concepts(concepts, threshold=0.5)

    assert len(deduped) == 1
    assert deduped[0]["_vec"] == pytest.approx([(1.0 + 0.9 + 0.8) / 3, (0.0 + 0.1 + 0.2) / 3])


# --------------------------------------------------------------------------- #
# _rank_by_textual_salience
# --------------------------------------------------------------------------- #

def test_rank_by_textual_salience_orders_by_mention_frequency():
    concepts = [
        {"label": "Rare Concept", "aliases": []},
        {"label": "Common Concept", "aliases": []},
    ]
    text = "Common Concept appears here. Common Concept again. Rare Concept appears once."

    ranked = _rank_by_textual_salience(concepts, text)

    assert [c["label"] for c in ranked] == ["Common Concept", "Rare Concept"]


def test_rank_by_textual_salience_counts_merged_aliases_too():
    concepts = [
        {"label": "Main Label", "aliases": ["Alt Name"]},
        {"label": "Other Concept", "aliases": []},
    ]
    text = "Alt Name shows up twice. Alt Name again. Other Concept shows up once."

    ranked = _rank_by_textual_salience(concepts, text)

    assert ranked[0]["label"] == "Main Label"


# --------------------------------------------------------------------------- #
# apply_degree_centrality
# --------------------------------------------------------------------------- #

def test_apply_degree_centrality_hub_scores_highest():
    hub = _node()
    leaves = [_node() for _ in range(3)]
    nodes = [hub, *leaves]
    edges = [_edge(hub, leaf) for leaf in leaves]  # hub -> each leaf

    apply_degree_centrality(nodes, edges)

    # hub: out-degree 3, in-degree 0 -> 3 / (4-1) = 1.0
    assert hub.importance_score == pytest.approx(1.0)
    # each leaf: in-degree 1, out-degree 0 -> 1 / 3
    for leaf in leaves:
        assert leaf.importance_score == pytest.approx(1 / 3)


def test_apply_degree_centrality_single_node_is_a_noop():
    solo = _node(importance=1.0)
    apply_degree_centrality([solo], [])
    assert solo.importance_score == 1.0  # untouched, not divided by zero


def test_apply_degree_centrality_isolated_node_scores_zero():
    hub = _node()
    isolated = _node()
    other = _node()
    apply_degree_centrality([hub, isolated, other], [_edge(hub, other)])
    assert isolated.importance_score == 0.0


# --------------------------------------------------------------------------- #
# _topological_sort
# --------------------------------------------------------------------------- #

def test_topological_sort_orders_by_prerequisite_edges():
    a, b, c = _node(), _node(), _node()
    # a must precede b, b must precede c
    edges = [_edge(a, b, "prerequisite_of"), _edge(b, c, "prerequisite_of")]

    order = _topological_sort([a, b, c], edges)

    assert order.index(a.id) < order.index(b.id) < order.index(c.id)
    assert set(order) == {a.id, b.id, c.id}


def test_topological_sort_falls_back_to_importance_when_no_prerequisites():
    high = _node(importance=0.9)
    low = _node(importance=0.1)
    # Only a related_to edge -- no prerequisite_of edges at all.
    edges = [_edge(high, low, "related_to")]

    order = _topological_sort([low, high], edges)

    assert order == [high.id, low.id]  # importance-sorted descending


def test_topological_sort_handles_a_cycle_without_dropping_nodes():
    a, b = _node(), _node()
    # a -> b -> a: a genuine cycle in the prerequisite subgraph.
    edges = [_edge(a, b, "prerequisite_of"), _edge(b, a, "prerequisite_of")]

    order = _topological_sort([a, b], edges)

    # Kahn's algorithm can't order a cycle; both nodes must still appear exactly once.
    assert sorted(order) == sorted([a.id, b.id])
    assert len(order) == 2


def test_topological_sort_breaks_ties_by_distance_from_root():
    # Only one prerequisite edge constrains the order; the rest used to come out in
    # arbitrary DB order. Unconstrained nodes must now follow distance from the root.
    root, near, far, prereq_target = (_node(label=x) for x in ("root", "near", "far", "target"))
    edges = [
        _edge(root, near, "related_to"),
        _edge(near, far, "related_to"),
        _edge(root, prereq_target, "prerequisite_of"),
    ]

    order = _topological_sort([far, prereq_target, near, root], edges, root.id)

    assert order[0] == root.id
    assert order.index(near.id) < order.index(far.id)
    assert order.index(root.id) < order.index(prereq_target.id)


def test_topological_sort_puts_expanded_nodes_after_core_nodes():
    core = _node(importance=0.1, label="core")
    expanded = _node(importance=0.9, label="expanded", depth=1)

    order = _topological_sort([expanded, core], [])

    assert order == [core.id, expanded.id]


# --------------------------------------------------------------------------- #
# curriculum_edges / project_edges / one_edge_per_pair / connected_component
# --------------------------------------------------------------------------- #

def test_curriculum_edges_match_labels_case_insensitively_and_via_aliases():
    concepts = [
        {"label": "Ball Control", "aliases": [], "prerequisites": []},
        {"label": "Passing", "aliases": ["Pass"], "prerequisites": ["ball  control"]},
        {"label": "Offside Rule", "aliases": [],
         "prerequisites": ["pass", "Unknown Thing", "Offside Rule", 42]},
    ]

    edges = curriculum_edges(concepts)

    assert edges == [
        {"source_index": 0, "target_index": 1, "relation": "prerequisite_of",
         "confidence": CURRICULUM_EDGE_CONFIDENCE},
        {"source_index": 1, "target_index": 2, "relation": "prerequisite_of",
         "confidence": CURRICULUM_EDGE_CONFIDENCE},
    ]  # unknown labels, self-references and non-strings ignored


def test_curriculum_edges_empty_without_prerequisites():
    assert curriculum_edges([{"label": "A"}, {"label": "B"}]) == []


def test_project_edges_bridges_prerequisite_chain_through_a_dropped_concept():
    # A -> B -> C, B dropped: A must still precede C.
    edges = [_rel(0, 1), _rel(1, 2)]

    projected = project_edges(edges, kept=[0, 2])

    assert [(e["source_index"], e["target_index"]) for e in projected] == [(0, 1)]


def test_project_edges_drops_non_prerequisite_edges_with_their_endpoint():
    edges = [_rel(0, 1, "related_to"), _rel(1, 2, "related_to")]
    assert project_edges(edges, kept=[0, 2]) == []


def test_one_edge_per_pair_keeps_strongest_and_prefers_prerequisite_on_tie():
    edges = [
        _rel(0, 1, "prerequisite_of", 0.5),
        _rel(1, 0, "subtopic_of", 0.5),     # same pair, tie -> prerequisite_of wins
        _rel(2, 3, "related_to", 0.6),
        _rel(3, 2, "enables", 0.9),         # same pair, higher confidence wins
    ]

    kept = {(e["source_index"], e["target_index"], e["relation"]) for e in one_edge_per_pair(edges)}

    assert kept == {(0, 1, "prerequisite_of"), (3, 2, "enables")}


def test_connected_component_is_undirected():
    edges = [_rel(1, 0), _rel(1, 2), _rel(3, 4)]
    assert connected_component(0, 5, edges) == {0, 1, 2}


# --------------------------------------------------------------------------- #
# relevant_indices (off-topic filter)
# --------------------------------------------------------------------------- #

def _fake_scores(monkeypatch, scores: dict[str, float]):
    """Make encode()/cosine_similarity() return a fixed similarity per text."""
    monkeypatch.setattr(graph_builder, "encode_one", lambda text: "TOPIC")
    monkeypatch.setattr(graph_builder, "encode", lambda texts: list(texts))
    monkeypatch.setattr(graph_builder, "cosine_similarity", lambda a, b: scores[b])


def test_relevant_indices_drops_clear_outliers(monkeypatch):
    concepts = [{"label": f"C{i}", "description": "d"} for i in range(7)]
    scores = {f"C{i}: d": 0.5 for i in range(7)}
    scores["C3: d"] = 0.05
    _fake_scores(monkeypatch, scores)

    assert relevant_indices(concepts, "topic", threshold=0.2) == [0, 1, 2, 4, 5, 6]


def test_relevant_indices_never_drops_below_minimum(monkeypatch):
    concepts = [{"label": f"C{i}", "description": "d"} for i in range(MIN_GRAPH_NODES + 2)]
    scores = {f"C{i}: d": 0.01 * i for i in range(MIN_GRAPH_NODES + 2)}  # all below threshold
    _fake_scores(monkeypatch, scores)

    kept = relevant_indices(concepts, "topic", threshold=0.9)

    assert len(kept) == MIN_GRAPH_NODES
    assert kept == sorted(kept)  # order preserved
    assert 0 not in kept and 1 not in kept  # the two least similar went


def test_relevant_indices_uses_label_only_for_spacy_fallback(monkeypatch):
    # spaCy concepts share a templated description that would make every one look
    # relevant, so only the label is compared.
    concepts = [{"label": f"C{i}", "description": "A key concept related to X.",
                 "_source": "spacy_fallback"} for i in range(6)]
    scores = {f"C{i}": 0.5 for i in range(6)}
    scores["C0"] = 0.0
    _fake_scores(monkeypatch, scores)

    assert relevant_indices(concepts, "X", threshold=0.2) == [1, 2, 3, 4, 5]
