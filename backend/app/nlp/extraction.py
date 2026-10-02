"""
Topic Understanding / Slot Extraction — Phase 2, Step 1.

Strategy:
  1. Try the free-tier LLM provider chain (app/nlp/llm.py: Gemini -> Groq -> Cerebras ->
     Mistral -> optional Anthropic).
  2. Fall back to spaCy NER + noun-chunk extraction if every provider is unconfigured
     or fails.

Relation classification has two modes (see graph_builder.py, settings.relation_mode):
  - classify_relations_batched(): one LLM call for the whole graph (the default -- see
    the module docstring in llm.py for why the O(n^2) pairwise approach is unworkable
    against free-tier rate limits).
  - classify_relation(): the legacy one-call-per-pair approach, kept only so the
    ablation study (backend/eval/) can compare batched vs. pairwise quality.

classify_relations_ensemble() wraps classify_relations_batched() with cross-provider
consensus when settings.relation_ensemble_size > 1 and more than one provider is
configured: it sends the same prompt to several distinct providers concurrently and
uses how many of them agree on a given edge as the confidence signal, instead of
trusting one model's self-reported (uncalibrated) confidence number, and tolerates any
single provider failing outright. graph_builder.py calls this instead of
classify_relations_batched() directly; it degrades to the identical single-call
behavior automatically when ensembling isn't possible, so it's a safe default even
with only one provider configured.

classify_orphan_edges() is a small follow-up pass: after relation classification, any
node left with zero edges gets one focused, cheap call (proportional to orphan count,
not O(n^2)) asking specifically for its best connection into the rest of the graph.
"""
from __future__ import annotations
import asyncio
import json
import logging
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import spacy

from app.config import get_settings
from app.cache import cache_get, cache_set, llm_extract_key
from app.nlp.llm import (
    complete_json, complete_json_from, complete_text, strip_json_fences,
    configured_providers, AllProvidersFailed,
)

logger = logging.getLogger(__name__)
settings = get_settings()

EXTRACTION_PROMPT = """\
You are an expert knowledge graph builder. Given a topic query and supporting text, \
extract the 6–10 most important concepts/sub-topics as a JSON list.

Each concept must have:
- "label": canonical short name (2–5 words max)
- "description": one sentence explaining the concept
- "category": one of [Fundamentals, Algorithms, Hardware, Applications, \
Research, Ethics, History, Tools, Other]
{prior_knowledge_clause}
Return ONLY a valid JSON array. No explanation, no markdown fences.

Topic: {topic}
Supporting text (first 6000 chars):
{text}
"""

PRIOR_KNOWLEDGE_CLAUSE = """
The user already knows: {items}. Do not re-extract these as top-level concepts; \
focus on what builds on or goes beyond them.
"""

RELATION_PROMPT = """\
You are an expert knowledge graph builder. Classify the relationship between these two concepts.

Concept A: {concept_a}
Concept B: {concept_b}
Context: {context}

Choose EXACTLY ONE relation from:
  - prerequisite_of  (A must be understood before B)
  - subtopic_of      (A is a specific sub-area of B)
  - enables          (understanding A makes B much easier)
  - related_to       (A and B are related but no strict order)
  - breaks           (A challenges or contradicts B)

Respond with JSON only: {{"relation": "<relation>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}
"""

BATCHED_RELATION_PROMPT = """\
You are an expert knowledge graph builder. Below is a numbered list of concepts from the \
same topic, followed by shared context. Classify the relationship for EVERY pair of \
concepts that has a meaningful, directed relation. Skip pairs with no meaningful relation \
instead of forcing one.

Concepts:
{numbered_concepts}

Context: {context}

For each related pair, choose EXACTLY ONE relation from:
  - prerequisite_of  (the first concept must be understood before the second)
  - subtopic_of      (the first concept is a specific sub-area of the second)
  - enables          (understanding the first makes the second much easier)
  - related_to       (they are related but with no strict order)
  - breaks           (the first challenges or contradicts the second)

Return ONLY a valid JSON array, no markdown fences, of objects shaped exactly like:
{{"source": <1-based index of first concept>, "target": <1-based index of second concept>, \
"relation": "<relation>", "confidence": <0.0-1.0>}}
"""

ORPHAN_PROMPT = """\
You are an expert knowledge graph builder. The concepts below were extracted for the same \
topic. The ones listed as "orphans" ended up with no relation to any other concept, which \
is almost always a coverage gap rather than genuine isolation.

All concepts:
{numbered_concepts}

Orphan concepts (each needs at least one connection to another concept above): {orphan_list}

Context: {context}

For each orphan, choose the ONE other concept from the full list it relates to most \
strongly, and classify that relationship using EXACTLY ONE relation from:
  - prerequisite_of  (the first concept must be understood before the second)
  - subtopic_of      (the first concept is a specific sub-area of the second)
  - enables          (understanding the first makes the second much easier)
  - related_to       (they are related but with no strict order)
  - breaks           (the first challenges or contradicts the second)

Return ONLY a valid JSON array, no markdown fences, of objects shaped exactly like:
{{"source": <1-based index of the orphan>, "target": <1-based index of the concept it \
connects to>, "relation": "<relation>", "confidence": <0.0-1.0>}}
"""


@lru_cache(maxsize=1)
def get_spacy():
    return spacy.load("en_core_web_sm")


async def extract_concepts(
    topic: str,
    supporting_text: str,
    prior_knowledge: list[str] | None = None,
) -> list[dict]:
    """
    Extract key concepts from the topic + supporting documents.

    `prior_knowledge` (concepts the user says they already know, from
    TopicCreate.prior_knowledge) biases extraction toward what builds on that --
    previously accepted by the API but never reaching this function.
    """
    prior_knowledge = prior_knowledge or []
    cache_k = llm_extract_key(f"{topic}::{','.join(sorted(prior_knowledge))}::{supporting_text[:500]}")
    cached = await cache_get(cache_k)
    if cached:
        return cached

    prior_knowledge_clause = (
        PRIOR_KNOWLEDGE_CLAUSE.format(items=", ".join(prior_knowledge))
        if prior_knowledge else ""
    )
    prompt = EXTRACTION_PROMPT.format(
        topic=topic, text=supporting_text[:6000], prior_knowledge_clause=prior_knowledge_clause
    )
    try:
        # 6-10 concepts x {label, description, category} plus reasoning-model
        # "thinking" tokens (which count against max_tokens but aren't part of
        # the visible answer -- see llm.py) can overrun a tighter budget and
        # truncate the JSON mid-string; 1024 was observed doing exactly that.
        concepts = await complete_json(prompt, max_tokens=3072)
        if not isinstance(concepts, list) or not concepts:
            raise ValueError(f"Expected a non-empty JSON array, got: {type(concepts)}")
        logger.info("LLM extracted %d concepts for topic '%s'", len(concepts), topic)
        await cache_set(cache_k, concepts, ttl=settings.ttl_llm_cache)
        return concepts
    except AllProvidersFailed as exc:
        logger.warning("No LLM provider available, falling back to spaCy: %s", exc)
    except Exception as exc:
        logger.warning("LLM extraction failed, falling back to spaCy: %s", exc)

    concepts = _spacy_extract(topic, supporting_text)
    await cache_set(cache_k, concepts, ttl=settings.ttl_llm_cache)
    return concepts


def _spacy_extract(topic: str, text: str) -> list[dict]:
    """Fallback: noun chunks + named entities. Local, free, never fails."""
    nlp = get_spacy()
    doc = nlp(text[:5000])

    seen: set[str] = set()
    candidates: list[str] = []

    # Named entities
    for ent in doc.ents:
        label = ent.text.strip()
        if len(label) > 2 and label.lower() not in seen:
            seen.add(label.lower())
            candidates.append(label)

    # Noun chunks
    for chunk in doc.noun_chunks:
        label = chunk.text.strip()
        if 2 < len(label.split()) <= 5 and label.lower() not in seen:
            seen.add(label.lower())
            candidates.append(label)

    # Score by frequency
    text_lower = text.lower()
    scored = sorted(candidates, key=lambda c: text_lower.count(c.lower()), reverse=True)
    top = scored[:10]

    return [
        {"label": c, "description": f"A key concept related to {topic}.", "category": "Other"}
        for c in top
    ]


async def classify_relation(
    concept_a: str,
    concept_b: str,
    context: str,
) -> dict:
    """
    Legacy pairwise classifier: one call per ordered pair, O(n^2 - n) total for a graph.
    Kept for the ablation study (backend/eval/) -- classify_relations_batched() is the
    default path used by graph_builder.py.
    """
    prompt = RELATION_PROMPT.format(
        concept_a=concept_a,
        concept_b=concept_b,
        context=context[:800],
    )
    try:
        result = await complete_json(prompt, max_tokens=256)
        if not isinstance(result, dict) or "relation" not in result:
            raise ValueError(f"Malformed relation response: {result!r}")
        return result
    except AllProvidersFailed as exc:
        return {"relation": "related_to", "confidence": 0.3, "reasoning": f"LLM unavailable: {exc}"}
    except Exception as exc:
        logger.warning("Relation classification failed (%s vs %s): %s", concept_a, concept_b, exc)
        return {"relation": "related_to", "confidence": 0.3, "reasoning": str(exc)}


VALID_RELATIONS = {"prerequisite_of", "subtopic_of", "enables", "related_to", "breaks"}


async def classify_relations_batched(labels: list[str], context: str) -> list[dict]:
    """
    Single-call relation classification for a whole node set: one LLM call returns the
    complete typed edge list as {source, target, relation, confidence} with 1-based
    indices into `labels`. This is the default relation_mode -- it takes a 12-node graph
    build from ~132 LLM calls to 1, which is what makes free-tier rate limits workable.

    Returns [] (an edgeless graph -- graph_builder falls back to importance ordering)
    if every provider is unconfigured or fails; callers should not treat that as fatal.
    """
    numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, 1))
    prompt = BATCHED_RELATION_PROMPT.format(numbered_concepts=numbered, context=context[:1500])

    try:
        raw = await complete_json(prompt, max_tokens=2048)
    except AllProvidersFailed as exc:
        logger.warning("Batched relation classification unavailable (no LLM provider): %s", exc)
        return []
    except Exception as exc:
        logger.warning("Batched relation classification failed to parse: %s", exc)
        return []

    edges = _parse_relation_items(raw, len(labels))
    logger.info("Batched relation classification produced %d edges for %d nodes (1 LLM call)",
                len(edges), len(labels))
    return edges


def _parse_relation_items(raw: object, n: int) -> list[dict]:
    """
    Validate and convert a raw {"source", "target", "relation", "confidence"}
    (1-based) JSON array into the 0-based edge-dict shape graph_builder.py
    expects. Shared by every relation-classification path that talks to the LLM
    in this batched shape (single-call, ensemble, orphan-repair).
    """
    if not isinstance(raw, list):
        logger.warning("Relation response was not a JSON array: %r", raw)
        return []

    edges: list[dict] = []
    for item in raw:
        try:
            src, tgt = int(item["source"]), int(item["target"])
            relation = item["relation"]
            confidence = float(item.get("confidence", 0.5))
        except (KeyError, TypeError, ValueError):
            continue
        if not (1 <= src <= n and 1 <= tgt <= n) or src == tgt:
            continue
        if relation not in VALID_RELATIONS:
            continue
        edges.append({
            "source_index": src - 1,
            "target_index": tgt - 1,
            "relation": relation,
            "confidence": confidence,
        })
    return edges


async def classify_relations_ensemble(labels: list[str], context: str) -> list[dict]:
    """
    Cross-provider ensemble wrapper around the same batched-relation prompt used
    by classify_relations_batched(): sends it to up to `relation_ensemble_size`
    distinct configured providers concurrently, then merges their answers using
    how many of them agree on a given (source, target, relation) as the
    confidence signal -- a self-reported number from a single model isn't a
    calibrated probability, and different providers self-report on different
    scales, so agreement across independently-queried models is a more honest
    signal than trusting any one of them.

    Also more resilient than the single-call path: one provider failing outright
    no longer means the whole relation-classification step returns [] (an
    edgeless graph) -- any other provider that answered still contributes edges.

    Degrades automatically to plain classify_relations_batched() when ensembling
    isn't possible (only one provider configured, or relation_ensemble_size <= 1),
    so this is safe to call unconditionally regardless of how many keys are set.
    """
    ensemble_size = max(1, settings.relation_ensemble_size)
    # Anthropic has no free tier -- prefer free providers for the extra opinions
    # and only reach for it if it's literally the only thing configured.
    providers = configured_providers()
    free_providers = [p for p in providers if p != "anthropic"]
    chosen = (free_providers or providers)[:ensemble_size]

    if len(chosen) <= 1:
        return await classify_relations_batched(labels, context)

    numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, 1))
    prompt = BATCHED_RELATION_PROMPT.format(numbered_concepts=numbered, context=context[:1500])

    async def _ask(provider_name: str) -> list[dict]:
        try:
            raw = await complete_json_from(provider_name, prompt, max_tokens=2048)
        except Exception as exc:
            logger.warning("Ensemble relation call to '%s' failed: %s", provider_name, exc)
            return []
        return _parse_relation_items(raw, len(labels))

    responses = await asyncio.gather(*(_ask(name) for name in chosen))
    non_empty_responses = [r for r in responses if r]
    if not non_empty_responses:
        logger.warning("Every provider in the relation ensemble failed or returned nothing")
        return []

    votes: dict[tuple[int, int, str], int] = defaultdict(int)
    for response in responses:
        seen_this_response: set[tuple[int, int, str]] = set()
        for item in response:
            key = (item["source_index"], item["target_index"], item["relation"])
            if key not in seen_this_response:  # one vote per provider per edge
                votes[key] += 1
                seen_this_response.add(key)

    total_responses = len(responses)
    edges = [
        {
            "source_index": src_idx,
            "target_index": tgt_idx,
            "relation": relation,
            "confidence": count / total_responses,  # cross-model agreement, not a self-report
        }
        for (src_idx, tgt_idx, relation), count in votes.items()
    ]

    logger.info(
        "Ensemble relation classification: %d/%d providers responded, %d edges (agreement-weighted)",
        len(non_empty_responses), len(chosen), len(edges),
    )
    return edges


async def classify_orphan_edges(
    labels: list[str], orphan_indices: list[int], context: str,
) -> list[dict]:
    """
    Focused repair pass for nodes relation classification left with zero edges.
    One small follow-up call proportional to orphan count (typically 0-3 nodes),
    not another O(n^2) pass -- an orphan is almost always a coverage gap in the
    main classification step rather than a genuinely unrelated concept, so it's
    worth a second, narrower shot at connecting it before persisting the graph.

    Returns [] on any failure (no provider available, malformed response) --
    callers should treat that as "the node stays unconnected," not fatal.
    """
    if not orphan_indices:
        return []

    numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, 1))
    orphan_list = ", ".join(f"{i + 1}. {labels[i]}" for i in orphan_indices)
    prompt = ORPHAN_PROMPT.format(
        numbered_concepts=numbered, orphan_list=orphan_list, context=context[:1500]
    )

    try:
        raw = await complete_json(prompt, max_tokens=1024)
    except AllProvidersFailed as exc:
        logger.warning("Orphan-repair classification unavailable (no LLM provider): %s", exc)
        return []
    except Exception as exc:
        logger.warning("Orphan-repair classification failed to parse: %s", exc)
        return []

    edges = _parse_relation_items(raw, len(labels))
    logger.info("Orphan repair: %d candidate connections for %d orphan node(s)",
                len(edges), len(orphan_indices))
    return edges
