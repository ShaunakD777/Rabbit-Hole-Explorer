"""
Tests for app/nlp/intent.py's ambiguity check: JSON parsing, candidate
validation, and graceful degradation when no LLM provider is available.
check_ambiguity() must never raise -- POST /topics treats None as "proceed
as normal", not as a failure.
"""
from app.nlp import intent
from app.nlp.llm import AllProvidersFailed


async def _no_cache_get(key):
    return None


async def _noop_cache_set(key, value, ttl):
    return None


async def test_check_ambiguity_returns_none_when_not_ambiguous(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=512):
        return {"ambiguous": False}

    monkeypatch.setattr(intent, "complete_json", fake_complete_json)
    monkeypatch.setattr(intent, "cache_get", _no_cache_get)
    monkeypatch.setattr(intent, "cache_set", _noop_cache_set)

    result = await intent.check_ambiguity("Quantum Computing")
    assert result is None


async def test_check_ambiguity_returns_candidates_when_ambiguous(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=512):
        return {
            "ambiguous": True,
            "candidates": [
                {"label": "Planet", "clarifying_query": "Mercury (planet)", "hint": "..."},
                {"label": "Element", "clarifying_query": "Mercury (chemical element)", "hint": "..."},
            ],
        }

    monkeypatch.setattr(intent, "complete_json", fake_complete_json)
    monkeypatch.setattr(intent, "cache_get", _no_cache_get)
    monkeypatch.setattr(intent, "cache_set", _noop_cache_set)

    result = await intent.check_ambiguity("Mercury")
    assert result is not None
    assert len(result) == 2
    assert result[0]["clarifying_query"] == "Mercury (planet)"


async def test_check_ambiguity_drops_malformed_candidates(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=512):
        return {
            "ambiguous": True,
            "candidates": [
                {"label": "Planet", "clarifying_query": "Mercury (planet)"},  # valid
                {"label": "Missing query"},  # missing clarifying_query
                "not even a dict",
            ],
        }

    monkeypatch.setattr(intent, "complete_json", fake_complete_json)
    monkeypatch.setattr(intent, "cache_get", _no_cache_get)
    monkeypatch.setattr(intent, "cache_set", _noop_cache_set)

    result = await intent.check_ambiguity("Mercury")
    assert result is not None
    assert len(result) == 1


async def test_check_ambiguity_returns_none_when_candidates_all_invalid(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=512):
        return {"ambiguous": True, "candidates": ["nope"]}

    monkeypatch.setattr(intent, "complete_json", fake_complete_json)
    monkeypatch.setattr(intent, "cache_get", _no_cache_get)
    monkeypatch.setattr(intent, "cache_set", _noop_cache_set)

    result = await intent.check_ambiguity("Mercury")
    assert result is None


async def test_check_ambiguity_degrades_gracefully_with_no_provider(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=512):
        raise AllProvidersFailed("no keys configured")

    monkeypatch.setattr(intent, "complete_json", fake_complete_json)
    monkeypatch.setattr(intent, "cache_get", _no_cache_get)
    monkeypatch.setattr(intent, "cache_set", _noop_cache_set)

    result = await intent.check_ambiguity("anything")
    assert result is None  # never raises -- POST /topics proceeds as unambiguous


async def test_check_ambiguity_degrades_gracefully_on_malformed_response(monkeypatch):
    async def fake_complete_json(prompt, max_tokens=512):
        return ["not", "a", "dict"]

    monkeypatch.setattr(intent, "complete_json", fake_complete_json)
    monkeypatch.setattr(intent, "cache_get", _no_cache_get)
    monkeypatch.setattr(intent, "cache_set", _noop_cache_set)

    result = await intent.check_ambiguity("anything")
    assert result is None


async def test_check_ambiguity_uses_cached_result(monkeypatch):
    calls = []

    async def fake_complete_json(prompt, max_tokens=512):
        calls.append(prompt)
        return {"ambiguous": False}

    async def fake_cache_get(key):
        return []  # previously cached as "not ambiguous"

    monkeypatch.setattr(intent, "complete_json", fake_complete_json)
    monkeypatch.setattr(intent, "cache_get", fake_cache_get)

    result = await intent.check_ambiguity("Quantum Computing")
    assert result is None
    assert calls == []  # cache hit -- no LLM call made
