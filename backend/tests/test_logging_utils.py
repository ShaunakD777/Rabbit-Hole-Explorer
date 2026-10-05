"""
Tests for the pipeline observability layer (app/logging_utils.py) and the places
that now report through it: llm.py (truncation, JSON parse failures) and
extraction.py (spaCy fallback flagged as degraded, stale spaCy cache hits).
"""
import logging

import httpx
import pytest

from app import logging_utils
from app.logging_utils import (
    current_trace, degrade, finish_trace, start_trace, step,
)
from app.nlp import extraction, llm

_RealAsyncClient = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _reset_trace():
    logging_utils._trace.set(None)
    yield
    logging_utils._trace.set(None)


# --------------------------------------------------------------------------- #
# Trace + step()
# --------------------------------------------------------------------------- #

def test_step_records_duration_and_info(caplog):
    caplog.set_level(logging.INFO)
    trace = start_trace("topic", "abcdef123456")
    with step("fetch_sources", query="'x'") as info:
        info["docs"] = 7
    assert [s[0] for s in trace.steps] == ["fetch_sources"]
    assert trace.steps[0][2] == "ok"
    assert "STEP fetch_sources DONE" in caplog.text and "docs=7" in caplog.text


def test_step_logs_failure_and_reraises(caplog):
    caplog.set_level(logging.INFO)
    trace = start_trace("topic", "abcdef123456")
    with pytest.raises(ValueError):
        with step("build_graph"):
            raise ValueError("boom")
    assert trace.steps[0][2] == "failed"
    assert "STEP build_graph FAIL" in caplog.text and "boom" in caplog.text


def test_degrade_is_recorded_and_makes_summary_a_warning(caplog):
    caplog.set_level(logging.INFO)
    trace = start_trace("topic", "abcdef123456")
    degrade("extraction", "falling back to spaCy")
    level, line = trace.summary("ready")
    assert level == logging.WARNING
    assert "extraction: falling back to spaCy" in line
    assert "DEGRADED stage=extraction" in caplog.text


def test_clean_run_summary_is_info():
    trace = start_trace("topic", "abcdef123456")
    level, line = trace.summary("ready")
    assert level == logging.INFO
    assert "degraded: no" in line


def test_finish_trace_clears_current_trace():
    start_trace("topic", "abcdef123456")
    assert current_trace() is not None
    finish_trace("ready")
    assert current_trace() is None


def test_degrade_without_a_trace_still_logs(caplog):
    caplog.set_level(logging.INFO)
    degrade("summary", "no provider")  # e.g. API request path, no trace started
    assert "DEGRADED stage=summary" in caplog.text


def test_preview_truncates_long_text_keeping_head_and_tail():
    text = "A" * 500 + "TAIL_MARKER"
    out = logging_utils.preview(text, head=20, tail=11)
    assert out.startswith("A" * 20) and out.endswith("TAIL_MARKER")
    assert "chars" in out
    assert logging_utils.preview("short") == "short"


# --------------------------------------------------------------------------- #
# llm.py reporting
# --------------------------------------------------------------------------- #

@pytest.fixture
def _one_provider(monkeypatch):
    for provider in llm._OPENAI_COMPAT_PROVIDERS.values():
        monkeypatch.setattr(llm.settings, provider.api_key_setting, "")
    monkeypatch.setattr(llm.settings, "anthropic_api_key", "")
    monkeypatch.setattr(llm.settings, "gemini_api_key", "fake-gemini-key")
    monkeypatch.setattr(llm, "_buckets", {})
    monkeypatch.setattr(llm, "_last_used", {})


def _patch_transport(monkeypatch, handler):
    def _factory(**kwargs):
        kwargs.pop("transport", None)
        return _RealAsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(httpx, "AsyncClient", _factory)


async def test_truncated_response_is_flagged(monkeypatch, caplog, _one_provider):
    caplog.set_level(logging.INFO)

    def handler(request):
        return httpx.Response(200, json={"choices": [
            {"message": {"content": '[{"label": "Qub'}, "finish_reason": "length"}]})

    _patch_transport(monkeypatch, handler)
    await llm.complete_text("hi", max_tokens=10)
    assert "LLM TRUNCATED provider=gemini" in caplog.text


async def test_successful_call_records_provider_and_trace(monkeypatch, caplog, _one_provider):
    caplog.set_level(logging.INFO)
    trace = start_trace("topic", "abcdef123456")

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })

    _patch_transport(monkeypatch, handler)
    await llm.complete_text("hi")
    assert logging_utils.last_llm_provider.get() == "gemini"
    assert trace.llm_calls and trace.llm_calls[0]["ok"] and trace.llm_calls[0]["provider"] == "gemini"
    assert "LLM OK provider=gemini" in caplog.text
    assert "key=#1/1" in caplog.text
    assert "fake-gemini-key" not in caplog.text  # API keys must never reach the logs


async def test_failed_provider_logs_status_and_body(monkeypatch, caplog, _one_provider):
    caplog.set_level(logging.INFO)
    trace = start_trace("topic", "abcdef123456")

    def handler(request):
        return httpx.Response(404, json={"error": {"message": "model_not_found"}})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(llm.AllProvidersFailed):
        await llm.complete_text("hi")
    assert "LLM FAIL provider=gemini" in caplog.text
    assert "HTTP 404" in caplog.text and "model_not_found" in caplog.text
    assert "no providers left" in caplog.text
    assert trace.llm_calls[0]["ok"] is False


async def test_json_parse_failure_logs_raw_tail(monkeypatch, caplog, _one_provider):
    caplog.set_level(logging.INFO)

    def handler(request):
        return httpx.Response(200, json={"choices": [
            {"message": {"content": '[{"label": "Unterminated'}, "finish_reason": "stop"}]})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(Exception):
        await llm.complete_json("x")
    assert "LLM JSON PARSE FAIL source=gemini" in caplog.text
    assert "Unterminated" in caplog.text


# --------------------------------------------------------------------------- #
# extraction.py fallback visibility
# --------------------------------------------------------------------------- #

@pytest.fixture
def _fake_cache(monkeypatch):
    store: dict = {}

    async def fake_get(key):
        return store.get(key)

    async def fake_set(key, value, ttl=None):
        store[key] = value

    monkeypatch.setattr(extraction, "cache_get", fake_get)
    monkeypatch.setattr(extraction, "cache_set", fake_set)
    return store


async def test_spacy_fallback_is_flagged_degraded_and_tagged(monkeypatch, caplog, _fake_cache):
    caplog.set_level(logging.INFO)
    trace = start_trace("topic", "abcdef123456")

    async def failing_complete_json(prompt, max_tokens=2048):
        raise llm.AllProvidersFailed("gemini: HTTP 404")

    monkeypatch.setattr(extraction, "complete_json", failing_complete_json)
    monkeypatch.setattr(extraction, "_spacy_extract",
                        lambda topic, text: [{"label": "Qubit", "description": "d",
                                              "category": "Other",
                                              "_source": extraction.SPACY_SOURCE}])

    concepts = await extraction.extract_concepts("quantum", "some text")
    assert concepts[0]["_source"] == extraction.SPACY_SOURCE
    assert any(d.startswith("extraction: falling back to spaCy") for d in trace.degradations)
    assert "no LLM provider available" in trace.degradations[0]


async def test_stale_spacy_cache_hit_is_flagged(monkeypatch, caplog, _fake_cache):
    caplog.set_level(logging.INFO)
    trace = start_trace("topic", "abcdef123456")

    async def must_not_be_called(prompt, max_tokens=2048):
        raise AssertionError("LLM should not be consulted on a cache hit")

    monkeypatch.setattr(extraction, "complete_json", must_not_be_called)

    # First call populates the cache with a degraded entry via the fallback...
    async def failing(prompt, max_tokens=2048):
        raise llm.AllProvidersFailed("down")

    monkeypatch.setattr(extraction, "complete_json", failing)
    monkeypatch.setattr(extraction, "_spacy_extract",
                        lambda topic, text: [{"label": "Qubit", "category": "Other",
                                              "_source": extraction.SPACY_SOURCE}])
    await extraction.extract_concepts("quantum", "some text")
    trace.degradations.clear()

    # ...second call hits that cache and must say so.
    monkeypatch.setattr(extraction, "complete_json", must_not_be_called)
    await extraction.extract_concepts("quantum", "some text")
    assert any("cache HIT served a stale spaCy-fallback" in d for d in trace.degradations)


async def test_healthy_llm_extraction_is_not_degraded(monkeypatch, _fake_cache):
    trace = start_trace("topic", "abcdef123456")

    async def ok(prompt, max_tokens=2048):
        return [{"label": "Qubit", "description": "A quantum bit", "category": "Fundamentals"}]

    monkeypatch.setattr(extraction, "complete_json", ok)
    concepts = await extraction.extract_concepts("quantum", "some text")
    assert concepts[0]["label"] == "Qubit"
    assert trace.degradations == []
