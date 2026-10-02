"""
Tests for app/nlp/extraction.py's batched relation classifier: JSON parsing,
index/relation validation, and graceful degradation when no LLM provider is
available. classify_relations_batched() must never raise -- graph_builder.py
treats an empty edge list as a valid (if sparse) outcome, not a failure.
"""
import pytest

from app.nlp import extraction
from app.nlp.llm import AllProvidersFailed


async def test_classify_relations_batched_parses_well_formed_response(monkeypatch):
    concepts = ["Qubit", "Quantum Gate", "Quantum Algorithm"]

    async def fake_complete_json(prompt, max_tokens=2048):
        return [
            {"source": 1, "target": 2, "relation": "prerequisite_of", "confidence": 0.9},
            {"source": 2, "target": 3, "relation": "prerequisite_of", "confidence": 0.8},
        ]

    monkeypatch.setattr(extraction, "complete_json", fake_complete_json)
    edges = await extraction.classify_relations_batched(concepts, "context")

    assert edges == [
        {"source_index": 0, "target_index": 1, "relation": "prerequisite_of", "confidence": 0.9},
        {"source_index": 1, "target_index": 2, "relation": "prerequisite_of", "confidence": 0.8},
    ]


async def test_classify_relations_batched_drops_invalid_entries(monkeypatch):
    concepts = ["A", "B"]

    async def fake_complete_json(prompt, max_tokens=2048):
        return [
            {"source": 1, "target": 2, "relation": "related_to", "confidence": 0.7},  # valid
            {"source": 1, "target": 1, "relation": "related_to", "confidence": 0.7},  # self-loop
            {"source": 1, "target": 99, "relation": "related_to", "confidence": 0.7},  # out of range
            {"source": 1, "target": 2, "relation": "not_a_real_relation", "confidence": 0.7},  # bad type
            {"source": 1, "relation": "related_to", "confidence": 0.7},  # missing target
        ]

    monkeypatch.setattr(extraction, "complete_json", fake_complete_json)
    edges = await extraction.classify_relations_batched(concepts, "context")

    assert len(edges) == 1
    assert edges[0]["relation"] == "related_to"


async def test_classify_relations_batched_returns_empty_when_not_a_list(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=2048):
        return {"not": "a list"}

    monkeypatch.setattr(extraction, "complete_json", fake_complete_json)
    edges = await extraction.classify_relations_batched(["A", "B"], "context")
    assert edges == []


async def test_classify_relations_batched_degrades_gracefully_with_no_provider(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=2048):
        raise AllProvidersFailed("no keys configured")

    monkeypatch.setattr(extraction, "complete_json", fake_complete_json)
    edges = await extraction.classify_relations_batched(["A", "B"], "context")
    assert edges == []  # never raises -- graph_builder treats this as a sparse graph


async def test_classify_relation_falls_back_to_related_to_on_any_failure(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=256):
        raise AllProvidersFailed("no keys configured")

    monkeypatch.setattr(extraction, "complete_json", fake_complete_json)
    result = await extraction.classify_relation("A", "B", "context")
    assert result["relation"] == "related_to"
    assert result["confidence"] < 0.5  # below graph_builder's default edge threshold
