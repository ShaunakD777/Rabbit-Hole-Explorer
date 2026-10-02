"""
Topic Understanding / Ambiguity Check — runs synchronously inside POST /topics,
before any connector fetch or Celery task is fired.

Problem: fetch_all_sources() has no notion of "which sense of this query did the
user mean" -- it fans out to 5 connectors and whichever documents survive the
per-connector result cap and the 2000-char extraction window implicitly define
the topic. For a genuinely ambiguous query ("Mercury", "Java") that produces a
graph skewed toward whichever sense happened to dominate the surviving sources,
with no signal to the user that a choice was made for them.

This module only detects true ambiguity (multiple distinct, unrelated common
meanings) -- not "this topic is broad" or "sources are sparse", which is a
separate problem this does not attempt to fix. Uses the same free-tier LLM
chain as extraction.py; if no provider is configured or the call fails, it
degrades to "not ambiguous" rather than blocking topic creation.
"""
from __future__ import annotations
import logging
from typing import Optional

from app.config import get_settings
from app.cache import cache_get, cache_set, llm_ambiguity_key
from app.nlp.llm import complete_json, AllProvidersFailed

logger = logging.getLogger(__name__)
settings = get_settings()

AMBIGUITY_PROMPT = """\
You are deciding whether a topic query names ONE coherent subject to build a knowledge graph \
around, or whether it actually names several mutually exclusive subjects that would produce an \
incoherent, unrelated mix of concepts if explored together.

Ask yourself: if someone built one knowledge graph of concepts for this exact query, would those \
concepts cohere into a single explorable subject -- or would it actually be several separate, \
non-overlapping subjects stapled together, each with its own rules/vocabulary/history that \
doesn't meaningfully connect to the others? Being in the same broad category (e.g. "a sport", \
"a field of science") is NOT enough to make something one coherent subject if the specific senses \
don't actually share rules, concepts, or history.

Examples that ARE ambiguous (separate subjects, must ask which one):
- "Mercury" -> the planet, the chemical element, the Roman god, the car brand (different fields entirely)
- "Java" -> the programming language, the Indonesian island, coffee (different fields entirely)
- "Football" -> American football, association football (soccer), rugby football, Australian \
rules, Gaelic football (all "a sport", but each is a separate game with its own rules, players, \
and history -- merging them produces an incoherent graph, not a richer one)
- "Hockey" -> ice hockey, field hockey (same reasoning as Football: one broad category, \
mutually exclusive specific games)

Examples that are NOT ambiguous (broad, technical, or with many sub-topics, but everything builds \
toward one coherent subject -- do not flag):
- "Quantum Computing", "Machine Learning", "CRISPR Gene Editing", "Blockchain" (many sub-topics, \
but all inform and connect to the same underlying subject)
- "Basketball" (positions, rules, history, famous players all belong to one coherent sport)

Topic: {topic}

Return ONLY valid JSON, no markdown fences, in exactly one of these shapes:
  {{"ambiguous": false}}
  {{"ambiguous": true, "candidates": [{{"label": "<short label>", \
"clarifying_query": "<disambiguated search query>", "hint": "<one sentence>"}}, ...]}}

If ambiguous, return 2-4 candidates covering the genuinely distinct subjects.
"""


async def check_ambiguity(raw_query: str) -> Optional[list[dict]]:
    """
    Returns None if the topic is unambiguous, or if no LLM provider is available
    / the call fails -- callers should treat None as "proceed as normal", never
    as an error. Returns a list of 2-4 candidate senses if genuinely ambiguous.
    """
    cache_k = llm_ambiguity_key(raw_query)
    cached = await cache_get(cache_k)
    if cached is not None:
        return cached or None

    prompt = AMBIGUITY_PROMPT.format(topic=raw_query)
    try:
        result = await complete_json(prompt, max_tokens=512)
    except AllProvidersFailed as exc:
        logger.info("Ambiguity check skipped (no LLM provider): %s", exc)
        return None
    except Exception as exc:
        logger.warning("Ambiguity check failed for '%s': %s", raw_query, exc)
        return None

    if not isinstance(result, dict):
        return None

    if not result.get("ambiguous"):
        await cache_set(cache_k, [], ttl=settings.ttl_llm_cache)
        return None

    candidates = result.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None

    valid = [
        c for c in candidates
        if isinstance(c, dict) and c.get("label") and c.get("clarifying_query")
    ]
    if not valid:
        return None

    await cache_set(cache_k, valid, ttl=settings.ttl_llm_cache)
    logger.info("Topic '%s' flagged ambiguous with %d candidates", raw_query, len(valid))
    return valid
