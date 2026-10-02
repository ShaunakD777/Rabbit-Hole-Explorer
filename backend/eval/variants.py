"""
Four relation-classification variants compared by the ablation study, all scored
against the same fixed concept set and gold labels (eval/labels/{topic}.csv):

  1. keyword       -- co-occurrence baseline. Can only say "related_to" or nothing.
  2. embedding     -- cosine-similarity baseline over local sentence embeddings.
                      Can only say "related_to" or nothing.
  3. llm_pairwise  -- the legacy one-call-per-pair classifier (extraction.classify_relation).
  4. llm_batched   -- the default one-call-for-the-whole-graph classifier
                      (extraction.classify_relations_batched).

Variants 1-2 are deliberately restricted to relatedness only (no typed relation),
matching the "keyword co-occurrence baseline" / "embedding-only baseline" wording in
the project's own implementation plan -- report.py scores them on a binary
"any relation detected" axis in addition to the 5-way typed metric, so they aren't
penalized purely for a distinction they were never designed to make.

NO_RELATION is used as an explicit placeholder prediction (and can appear in gold
labels for a pair a human annotator marked as unrelated), so every prediction is a
member of the same closed label set and every variant's output can be scored with
one shared metric.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.nlp.embeddings import encode, encode_one, cosine_similarity
from app.nlp.extraction import classify_relation, classify_relations_batched

NO_RELATION = "no_relation"
EMBEDDING_SIMILARITY_THRESHOLD = 0.45
VARIANTS = ["keyword", "embedding", "llm_pairwise", "llm_batched"]


@dataclass
class VariantResult:
    predictions: dict[tuple[str, str], str]  # (concept_a, concept_b) -> relation
    api_calls: int
    wall_clock_seconds: float
    extra: dict = field(default_factory=dict)


def run_keyword(pairs: list[tuple[str, str]], corpus_text: str) -> VariantResult:
    """related_to if both concepts co-occur in the same ~300-word window of the
    corpus (the same chunk size graph_builder.py/summarizer.py use elsewhere),
    else no_relation. Zero API calls, zero embedding cost."""
    t0 = time.perf_counter()
    words = corpus_text.split()
    window = 300
    chunks = [" ".join(words[i:i + window]).lower() for i in range(0, len(words), window)]

    predictions: dict[tuple[str, str], str] = {}
    for a, b in pairs:
        a_l, b_l = a.lower(), b.lower()
        co_occurs = any(a_l in c and b_l in c for c in chunks)
        predictions[(a, b)] = "related_to" if co_occurs else NO_RELATION

    return VariantResult(predictions, api_calls=0, wall_clock_seconds=time.perf_counter() - t0)


def run_embedding(pairs: list[tuple[str, str]]) -> VariantResult:
    """related_to if cosine similarity of the two concept-label embeddings clears
    EMBEDDING_SIMILARITY_THRESHOLD, else no_relation. Local model, zero API calls."""
    t0 = time.perf_counter()
    labels = sorted({c for pair in pairs for c in pair})
    vecs = dict(zip(labels, encode(labels)))

    predictions: dict[tuple[str, str], str] = {}
    for a, b in pairs:
        sim = cosine_similarity(vecs[a], vecs[b])
        predictions[(a, b)] = "related_to" if sim >= EMBEDDING_SIMILARITY_THRESHOLD else NO_RELATION

    return VariantResult(predictions, api_calls=0, wall_clock_seconds=time.perf_counter() - t0)


async def run_llm_pairwise(pairs: list[tuple[str, str]], context: str) -> VariantResult:
    """The legacy O(n^2 - n) approach: one LLM call per pair."""
    t0 = time.perf_counter()
    predictions: dict[tuple[str, str], str] = {}
    for a, b in pairs:
        result = await classify_relation(a, b, context)
        relation = result.get("relation", NO_RELATION)
        confidence = result.get("confidence", 0.0)
        predictions[(a, b)] = relation if confidence >= 0.5 else NO_RELATION

    return VariantResult(predictions, api_calls=len(pairs), wall_clock_seconds=time.perf_counter() - t0)


async def run_llm_batched(pairs: list[tuple[str, str]], concepts: list[str], context: str) -> VariantResult:
    """The default approach: one LLM call classifies every related pair in the
    whole concept set at once."""
    t0 = time.perf_counter()
    edges = await classify_relations_batched(concepts, context)

    by_pair: dict[tuple[int, int], dict] = {(e["source_index"], e["target_index"]): e for e in edges}
    index_of = {c: i for i, c in enumerate(concepts)}

    predictions: dict[tuple[str, str], str] = {}
    for a, b in pairs:
        edge = by_pair.get((index_of[a], index_of[b]))
        predictions[(a, b)] = edge["relation"] if edge else NO_RELATION

    return VariantResult(predictions, api_calls=1, wall_clock_seconds=time.perf_counter() - t0)
