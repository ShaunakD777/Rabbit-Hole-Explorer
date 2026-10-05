"""
Topic Understanding / Slot Extraction — Phase 2, Step 1.

Strategy:
  1. Try the free-tier LLM provider chain (app/nlp/llm.py: Gemini -> Groq -> Cerebras ->
     Mistral -> optional Anthropic).
  2. Fall back to spaCy NER + noun-chunk extraction if every provider is unconfigured
     or fails.

Relations come from one of three modes (see graph_builder.py, settings.relation_mode):
  - "curriculum" (default): no separate relation call at all. extract_concepts() asks
    for each concept's direct prerequisites, and those become prerequisite_of edges.
  - classify_relations_batched(): one LLM call for the whole graph ("batched"; also the
    fallback when curriculum extraction produced no prerequisites, e.g. a cached
    older-format result -- see the module docstring in llm.py for why the O(n^2)
    pairwise approach is unworkable against free-tier rate limits).
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

There is deliberately no "orphan repair" pass any more. It used to ask the LLM to
connect every node left without edges, and it obliged even for off-topic nodes
(e.g. "Injury Measurement Tools related_to Game Overview" in a football graph), which
made tangents look like they belonged. graph_builder.py now drops nodes outside the
root's connected component instead.
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
from app.logging_utils import degrade, last_llm_provider

logger = logging.getLogger(__name__)
settings = get_settings()

# Marker key on concepts produced by the spaCy fallback; survives the Redis cache so a
# later cache hit can be recognised as degraded (see extract_concepts).
SPACY_SOURCE = "spacy_fallback"

# Curriculum-style extraction: asks what a *beginner needs to learn*, in learning
# order, with each concept's direct prerequisites -- not "the most important
# concepts in this text". The old text-centric prompt turned whatever the sources
# happened to discuss (e.g. arXiv abstracts on soccer-playing robots) into nodes,
# produced section headings ("Game Overview", "Terminology Origins") instead of
# learnable concepts, and its tech-only category list pushed "Algorithms"/"Ethics"
# nodes into topics like football. The prerequisites double as the graph's
# prerequisite_of edges (relation_mode="curriculum"), so one call yields both nodes
# and learning-path ordering.
EXTRACTION_PROMPT = """\
You are designing a short curriculum for a complete beginner who wants to learn "{topic}". \
Using the supporting text below only as reference material, list the concepts that \
beginner must learn to understand {topic}: at most {max_concepts}, fewer if the topic is \
narrow. Do not pad the list to reach the maximum.

Rules:
- Each concept must be something a learner studies and can come to understand (e.g. \
for a sport: "Offside Rule", "Player Positions"; for a science: "Chlorophyll", \
"Light-Dependent Reactions"). Do NOT use document-section or meta headings such as \
"Overview", "Introduction", "Terminology", "Ethical Considerations" or "Applications" \
unless that is genuinely core subject matter of {topic}.
- Stay on the core of {topic}. Ignore niche research projects, specific papers, \
products or tools that appear in the supporting text but that a beginner does not need.
- List the concepts in the order a beginner should learn them, most foundational first.
- "prerequisites" lists the labels -- copied exactly from your own list -- of the \
concepts that must be understood directly BEFORE this one. Usually the first concept \
has none, and every later concept has at least one.

Each concept must have:
- "label": canonical short name (2-5 words)
- "description": one sentence a beginner can understand
- "category": one of [Foundation, Core Concept, Technique, Application, Context]
- "prerequisites": a list of labels from your own list (may be empty)
{prior_knowledge_clause}
Return ONLY a valid JSON array. No explanation, no markdown fences.

Topic: {topic}
Supporting text (first 6000 chars):
{text}
"""
# Bumped whenever EXTRACTION_PROMPT's wording or output shape changes. Part of the
# extraction cache key, so results cached under an older prompt (30-day TTL) can't
# mask the new one -- the same trap run_ambiguity_eval.py guards against.
EXTRACTION_PROMPT_VERSION = "v2-curriculum"

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
    max_concepts = settings.initial_graph_node_cap
    cache_k = llm_extract_key(
        f"{EXTRACTION_PROMPT_VERSION}::{topic}::{max_concepts}::"
        f"{','.join(sorted(prior_knowledge))}::{supporting_text[:500]}"
    )
    cached = await cache_get(cache_k)
    if cached:
        if any(c.get("_source") == SPACY_SOURCE for c in cached):
            # The cache outlives the outage that produced this entry (30-day TTL), so
            # re-querying the same topic keeps serving generic spaCy output even after
            # the LLM providers recover. Flag it so that's visible, not mysterious.
            degrade("extraction", f"cache HIT served a stale spaCy-fallback result for "
                                  f"'{topic}' (key {cache_k}); the LLM was NOT consulted")
        else:
            logger.info("Extraction cache HIT for '%s': %d concepts (key %s)",
                        topic, len(cached), cache_k)
        return cached
    logger.info("Extraction cache MISS for '%s' (key %s)", topic, cache_k)

    prior_knowledge_clause = (
        PRIOR_KNOWLEDGE_CLAUSE.format(items=", ".join(prior_knowledge))
        if prior_knowledge else ""
    )
    prompt = EXTRACTION_PROMPT.format(
        topic=topic, text=supporting_text[:6000], prior_knowledge_clause=prior_knowledge_clause,
        max_concepts=max_concepts,
    )
    logger.info("Extracting concepts for '%s': prompt context=%d chars, prior_knowledge=%d, cap=%d",
                topic, len(supporting_text[:6000]), len(prior_knowledge), max_concepts)
    reason: str
    try:
        # Up to max_concepts x {label, description, category} plus reasoning-model
        # "thinking" tokens (which count against max_tokens but aren't part of
        # the visible answer -- see llm.py) can overrun a tighter budget and
        # truncate the JSON mid-string; 1024 was observed doing exactly that.
        # The prerequisites lists add output tokens on top of label/description/category.
        concepts = await complete_json(prompt, max_tokens=4096)
        if not isinstance(concepts, list) or not concepts:
            raise ValueError(f"Expected a non-empty JSON array, got: {type(concepts)}")
        concepts = [c for c in concepts if isinstance(c, dict) and c.get("label")]
        if not concepts:
            raise ValueError("JSON array contained no concept objects with a label")
        logger.info("LLM extracted %d concepts for '%s' via provider=%s: %s", len(concepts), topic,
                    last_llm_provider.get(), [c.get("label") for c in concepts])
        await cache_set(cache_k, concepts, ttl=settings.ttl_llm_cache)
        return concepts
    except AllProvidersFailed as exc:
        reason = f"no LLM provider available ({exc})"
    except Exception as exc:
        reason = f"LLM answered but extraction failed ({type(exc).__name__}: {exc})"

    degrade("extraction", f"falling back to spaCy for '{topic}': {reason}. Nodes will have generic "
                          f"descriptions, category 'Other', and no relation edges")
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
    n_ents = 0
    for ent in doc.ents:
        label = ent.text.strip()
        if len(label) > 2 and label.lower() not in seen:
            seen.add(label.lower())
            candidates.append(label)
            n_ents += 1

    # Noun chunks
    n_chunks = 0
    for chunk in doc.noun_chunks:
        label = chunk.text.strip()
        if 2 < len(label.split()) <= 5 and label.lower() not in seen:
            seen.add(label.lower())
            candidates.append(label)
            n_chunks += 1

    # Score by frequency
    text_lower = text.lower()
    scored = sorted(candidates, key=lambda c: text_lower.count(c.lower()), reverse=True)
    top = scored[:10]
    logger.info("spaCy fallback extraction: %d chars analysed, %d entities + %d noun chunks "
                "-> %d candidates, kept top %d: %s",
                min(len(text), 5000), n_ents, n_chunks, len(candidates), len(top), top)

    return [
        {"label": c, "description": f"A key concept related to {topic}.", "category": "Context",
         "_source": SPACY_SOURCE}
        for c in top
    ]


def is_spacy_fallback(concepts: list[dict]) -> bool:
    """True if these concepts came from _spacy_extract (directly or via the cache)."""
    return any(c.get("_source") == SPACY_SOURCE for c in concepts)


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

    logger.info("Relation classification (batched, single call): %d concepts, context=%d chars",
                len(labels), len(context[:1500]))
    try:
        raw = await complete_json(prompt, max_tokens=2048)
    except AllProvidersFailed as exc:
        degrade("relations", f"no LLM provider for batched classification, graph will have no "
                             f"edges ({exc})")
        return []
    except Exception as exc:
        degrade("relations", f"batched classification response unusable, graph will have no "
                             f"edges ({type(exc).__name__}: {exc})")
        return []

    edges = _parse_relation_items(raw, len(labels))
    logger.info("Batched relation classification via provider=%s: %d valid edges for %d nodes "
                "(1 LLM call)", last_llm_provider.get(), len(edges), len(labels))
    return edges


def _parse_relation_items(raw: object, n: int) -> list[dict]:
    """
    Validate and convert a raw {"source", "target", "relation", "confidence"}
    (1-based) JSON array into the 0-based edge-dict shape graph_builder.py
    expects. Shared by every relation-classification path that talks to the LLM
    in this batched shape (single-call, ensemble).
    """
    if not isinstance(raw, list):
        logger.warning("Relation response was not a JSON array: %r", raw)
        return []

    edges: list[dict] = []
    malformed = out_of_range = bad_relation = 0
    for item in raw:
        try:
            src, tgt = int(item["source"]), int(item["target"])
            relation = item["relation"]
            confidence = float(item.get("confidence", 0.5))
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        if not (1 <= src <= n and 1 <= tgt <= n) or src == tgt:
            out_of_range += 1
            continue
        if relation not in VALID_RELATIONS:
            bad_relation += 1
            continue
        edges.append({
            "source_index": src - 1,
            "target_index": tgt - 1,
            "relation": relation,
            "confidence": confidence,
        })
    dropped = malformed + out_of_range + bad_relation
    if dropped:
        logger.warning("Relation parse: kept %d of %d items; dropped %d malformed, %d with "
                       "bad/self indices, %d with unknown relation type",
                       len(edges), len(raw), malformed, out_of_range, bad_relation)
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
        logger.info("Relation classification: ensemble not possible (providers=%s, "
                    "relation_ensemble_size=%d) -- using single-provider batched call",
                    providers, settings.relation_ensemble_size)
        return await classify_relations_batched(labels, context)
    logger.info("Relation classification: ensemble across providers=%s (%d concepts)",
                chosen, len(labels))

    numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, 1))
    prompt = BATCHED_RELATION_PROMPT.format(numbered_concepts=numbered, context=context[:1500])

    async def _ask(provider_name: str) -> list[dict]:
        try:
            raw = await complete_json_from(provider_name, prompt, max_tokens=2048)
        except Exception as exc:
            logger.warning("Ensemble relation call to '%s' failed: %s: %s", provider_name,
                           type(exc).__name__, exc)
            return []
        parsed = _parse_relation_items(raw, len(labels))
        logger.info("Ensemble member '%s' proposed %d valid edges", provider_name, len(parsed))
        return parsed

    responses = await asyncio.gather(*(_ask(name) for name in chosen))
    non_empty_responses = [r for r in responses if r]
    if not non_empty_responses:
        degrade("relations", f"every provider in the relation ensemble {chosen} failed or "
                             f"returned nothing, graph will have no edges")
        return []
    if len(non_empty_responses) < len(chosen):
        degrade("relations", f"ensemble partially failed: only {len(non_empty_responses)}/"
                             f"{len(chosen)} providers answered, edge confidences are computed "
                             f"over all {len(chosen)} so they are depressed")

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
