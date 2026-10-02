"""
Tests for app/nlp/llm.py's provider fallthrough chain and JSON-fence parsing.
Uses httpx.MockTransport so no real network calls are made -- see the fixture
below for how AsyncClient is redirected to it.
"""
import httpx
import pytest

from app.nlp import llm

# Captured before any test monkeypatches httpx.AsyncClient -- the mock factory
# below must construct the *real* client, not recurse into its own patched name
# (llm.py's `import httpx` is the same module object this file imports).
_RealAsyncClient = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _isolate_llm_state(monkeypatch):
    """Every test starts with no providers configured and a fresh rate-limiter
    state -- _buckets is a module-level dict that would otherwise leak between
    tests within the same pytest process."""
    for provider in llm._OPENAI_COMPAT_PROVIDERS.values():
        monkeypatch.setattr(llm.settings, provider.api_key_setting, "")
    monkeypatch.setattr(llm.settings, "anthropic_api_key", "")
    monkeypatch.setattr(llm, "_buckets", {})
    monkeypatch.setattr(llm, "_last_used", {})


def _mock_client_factory(handler):
    """Redirect every httpx.AsyncClient() constructed inside llm.py to a
    MockTransport driven by `handler`, without changing production code.
    llm.py always constructs it with no positional args, so **kwargs suffices."""
    def _factory(**kwargs):
        kwargs.pop("transport", None)
        return _RealAsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    return _factory


# --------------------------------------------------------------------------- #
# strip_json_fences
# --------------------------------------------------------------------------- #

def test_strip_json_fences_removes_markdown_fence():
    raw = '```json\n[{"label": "Qubit"}]\n```'
    assert llm.strip_json_fences(raw) == '[{"label": "Qubit"}]'


def test_strip_json_fences_passes_through_plain_json():
    raw = '{"relation": "related_to", "confidence": 0.8}'
    assert llm.strip_json_fences(raw) == raw


# --------------------------------------------------------------------------- #
# Provider chain
# --------------------------------------------------------------------------- #

async def test_complete_text_raises_when_no_provider_configured():
    with pytest.raises(llm.AllProvidersFailed):
        await llm.complete_text("hello")


async def test_complete_text_uses_the_only_configured_provider(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", "fake-gemini-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "generativelanguage.googleapis.com" in str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello from gemini"}}]})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
    result = await llm.complete_text("hi")
    assert result == "hello from gemini"


async def test_complete_text_falls_through_to_next_provider_on_failure(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", "fake-gemini-key")
    monkeypatch.setattr(llm.settings, "groq_api_key", "fake-groq-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if "generativelanguage.googleapis.com" in str(request.url):
            return httpx.Response(500, json={"error": "server error"})
        if "api.groq.com" in str(request.url):
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello from groq"}}]})
        raise AssertionError(f"unexpected URL: {request.url}")

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
    result = await llm.complete_text("hi")
    assert result == "hello from groq"


async def test_complete_text_raises_after_every_provider_fails(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", "fake-gemini-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
    with pytest.raises(llm.AllProvidersFailed):
        await llm.complete_text("hi", max_tokens=10)


async def test_complete_json_parses_fenced_response(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", "fake-gemini-key")

    def handler(request: httpx.Request) -> httpx.Response:
        content = '```json\n[{"label": "Qubit", "category": "Fundamentals"}]\n```'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr(httpx, "AsyncClient", _mock_client_factory(handler))
    result = await llm.complete_json("extract concepts")
    assert result == [{"label": "Qubit", "category": "Fundamentals"}]


def test_configured_chain_skips_providers_without_a_key(monkeypatch):
    monkeypatch.setattr(llm.settings, "groq_api_key", "fake-groq-key")
    monkeypatch.setattr(llm.settings, "llm_provider_chain", "gemini,groq,cerebras")
    assert llm._configured_chain() == ["groq"]


def test_configured_chain_includes_anthropic_only_when_keyed(monkeypatch):
    monkeypatch.setattr(llm.settings, "anthropic_api_key", "fake-anthropic-key")
    monkeypatch.setattr(llm.settings, "llm_provider_chain", "gemini,anthropic")
    assert llm._configured_chain() == ["anthropic"]


# --------------------------------------------------------------------------- #
# Multi-key least-recently-used selection
# --------------------------------------------------------------------------- #

async def test_acquire_key_rotates_across_keys_with_capacity():
    """With several keys all under capacity, _acquire_key must not just keep
    returning index 0 every time -- that's the exact bug that let one key
    absorb all traffic and hit its account-level quota while the rest sat
    idle. Each successive call (with capacity available) should prefer a key
    that hasn't been used yet over one that has."""
    provider = llm._OPENAI_COMPAT_PROVIDERS["gemini"]
    keys = ["key-a", "key-b", "key-c"]

    seen = [await llm._acquire_key(provider, keys, est_tokens=10) for _ in range(3)]
    assert set(seen) == set(keys)  # all three keys got used, none skipped

    # Once every key has been used at least once, the next pick must be
    # whichever was used longest ago -- i.e. the first one from this batch.
    fourth = await llm._acquire_key(provider, keys, est_tokens=10)
    assert fourth == seen[0]
