"""
Provider-agnostic LLM client for free-tier chat completion APIs.

Tries providers in `settings.llm_provider_chain` order (comma-separated), skipping any
whose API key is empty, and raises `AllProvidersFailed` only once every configured
provider has been tried and failed. Callers use that signal to fall back to a fully
local path (spaCy extraction, passage concatenation, etc.) that never fails and never
costs anything -- see extraction.py / summarizer.py.

Gemini, Groq, Cerebras, and Mistral all expose an OpenAI-compatible `/chat/completions`
endpoint, so one httpx-based caller covers all four -- no per-provider SDK. Anthropic
(no free tier) is wired in as an optional last hosted step using the native SDK, imported
lazily so a missing/old `anthropic` package can never block application startup; it is
only exercised if ANTHROPIC_API_KEY is actually set.

Each hosted provider is rate-limited client-side against its published free-tier RPM/TPM
so the app respects the quota rather than discovering it via 429s -- see graph_builder.py
and its O(n^2) relation-classification history for why this matters here specifically.

Each provider's *_api_key setting accepts one key or several comma-separated keys (e.g.
"key1,key2,key3") -- multiple free-tier accounts for the same provider, each with its own
RPM/TPM bucket, so more keys genuinely raise that provider's effective throughput rather
than being a spare. complete_text_from()/complete_json_from() call one specific named
provider directly (no fallback), for callers that want independent answers from several
providers at once -- see extraction.py's classify_relations_ensemble().
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def strip_json_fences(raw: str) -> str:
    """LLMs often wrap JSON in ```json ... ``` fences despite instructions not to."""
    return _JSON_FENCE_RE.sub("", raw.strip())


class AllProvidersFailed(Exception):
    """Every configured provider failed, or none is configured at all."""


@dataclass(frozen=True)
class _Provider:
    name: str
    base_url: str          # OpenAI-compatible base; POST {base_url}/chat/completions
    default_model: str
    api_key_setting: str    # attribute name on Settings holding the API key
    model_setting: str      # attribute name on Settings holding a model override
    rpm: int                 # free-tier requests/minute ceiling
    tpm: int                 # free-tier tokens/minute ceiling (heuristic-estimated)
    extra_body: dict[str, Any] | None = None  # merged into the request JSON body


_OPENAI_COMPAT_PROVIDERS: dict[str, _Provider] = {
    "gemini": _Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        # "gemini-2.5-flash" now 404s ("no longer available to new users") even
        # though it still shows up in /models -- use the rolling alias so this
        # doesn't go stale again as Google retires dated model names.
        default_model="gemini-flash-latest",
        api_key_setting="gemini_api_key",
        model_setting="gemini_model",
        rpm=10, tpm=250_000,
        # gemini-flash-latest reasons by default, and its (invisible) thinking
        # tokens count against max_tokens -- at max_tokens=1024 that left as
        # little as ~180 chars for the actual answer, truncating our JSON
        # mid-string. Disabling it gives the whole budget to real output.
        extra_body={"reasoning_effort": "none"},
    ),
    "groq": _Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        # "llama-3.3-70b-versatile" has been decommissioned from Groq's catalog
        # (404 model_not_found); gpt-oss-20b is the closest still-available
        # general-purpose instruct model on the free tier.
        default_model="openai/gpt-oss-20b",
        api_key_setting="groq_api_key",
        model_setting="groq_model",
        rpm=30, tpm=12_000,
        # Also a reasoning model, so its thinking tokens count against
        # max_tokens too (same failure mode as Gemini -- see above). Groq's
        # API rejects "none" outright (400: must be low/medium/high), so
        # "low" is the closest equivalent to cut that overhead.
        extra_body={"reasoning_effort": "low"},
    ),
    "cerebras": _Provider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        default_model="gpt-oss-120b",
        api_key_setting="cerebras_api_key",
        model_setting="cerebras_model",
        rpm=30, tpm=30_000,
    ),
    "mistral": _Provider(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        api_key_setting="mistral_api_key",
        model_setting="mistral_model",
        rpm=60, tpm=50_000,
    ),
}


# --------------------------------------------------------------------------- #
# Client-side rate limiting
# --------------------------------------------------------------------------- #
class _TokenBucket:
    """
    Thread-safe request+token bucket, continuously refilled.

    Celery runs each task via `asyncio.run(coro)` inside a thread-pool worker
    (concurrency=4 in worker.py), so a fresh event loop can appear per call on any
    thread. The bucket's own state is guarded by a plain `threading.Lock` (correct
    across threads and event loops); callers wait for capacity with `asyncio.sleep`
    so they never block the event loop they're running on.
    """

    def __init__(self, rpm: int, tpm: int):
        self._rpm = rpm
        self._tpm = tpm
        self._lock = threading.Lock()
        self._req_tokens = float(rpm)
        self._tok_tokens = float(tpm)
        self._last = time.monotonic()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._req_tokens = min(self._rpm, self._req_tokens + elapsed * self._rpm / 60.0)
        self._tok_tokens = min(self._tpm, self._tok_tokens + elapsed * self._tpm / 60.0)

    def try_acquire(self, est_tokens: int) -> bool:
        """Non-blocking: consume capacity and return True only if it was available right now."""
        with self._lock:
            self._refill_locked()
            if self._req_tokens >= 1 and self._tok_tokens >= est_tokens:
                self._req_tokens -= 1
                self._tok_tokens -= est_tokens
                return True
            return False

    async def acquire(self, est_tokens: int) -> None:
        while not self.try_acquire(est_tokens):
            await asyncio.sleep(0.5)


_buckets: dict[str, _TokenBucket] = {}
_buckets_lock = threading.Lock()

# Last-used timestamp per "{provider}:{key_index}", for least-recently-used key
# selection (see _acquire_key). Missing entries sort first (never used yet).
_last_used: dict[str, float] = {}
_last_used_lock = threading.Lock()


def _bucket_for(bucket_key: str, provider: _Provider) -> _TokenBucket:
    with _buckets_lock:
        bucket = _buckets.get(bucket_key)
        if bucket is None:
            bucket = _TokenBucket(provider.rpm, provider.tpm)
            _buckets[bucket_key] = bucket
        return bucket


def _touch_last_used(bucket_key: str) -> None:
    with _last_used_lock:
        _last_used[bucket_key] = time.monotonic()


def _estimate_tokens(text: str) -> int:
    """Cheap heuristic (no tokenizer dependency): ~4 characters per token."""
    return max(1, len(text) // 4)


def _keys_for(provider: _Provider) -> list[str]:
    """
    A provider's *_api_key setting may hold one key or several comma-separated
    keys. Multiple keys are multiple free-tier accounts for the same provider --
    each has its own RPM/TPM ceiling, so more keys genuinely raise the effective
    throughput for that provider rather than just being a spare.
    """
    raw = getattr(settings, provider.api_key_setting, "") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


async def _acquire_key(provider: _Provider, keys: list[str], est_tokens: int) -> str:
    """
    Pick whichever of this provider's keys has capacity available right now,
    instead of funneling every call through a single shared bucket -- that's what
    makes multiple keys for one provider actually add parallel throughput.

    Among keys with capacity, prefer whichever was used longest ago (least-
    recently-used), rather than always preferring the first key in the list.
    With a flat RPM/TPM bucket, "always try index 0 first" lets key 0 soak up
    almost every call (its bucket keeps refilling between requests), so it hits
    the provider's real account-level quota while keys 1-4 sit idle. LRU spreads
    calls evenly across all configured keys for a provider instead.

    Falls back to waiting on the least-recently-used key if every key for this
    provider is currently at capacity.
    """
    bucket_keys = [f"{provider.name}:{i}" for i in range(len(keys))]
    buckets = [_bucket_for(bk, provider) for bk in bucket_keys]

    with _last_used_lock:
        order = sorted(range(len(keys)), key=lambda i: _last_used.get(bucket_keys[i], 0.0))

    for i in order:
        if buckets[i].try_acquire(est_tokens):
            _touch_last_used(bucket_keys[i])
            return keys[i]

    lru_index = order[0]
    await buckets[lru_index].acquire(est_tokens)
    _touch_last_used(bucket_keys[lru_index])
    return keys[lru_index]


# --------------------------------------------------------------------------- #
# Provider chain resolution
# --------------------------------------------------------------------------- #
def _configured_chain() -> list[str]:
    """Provider names from settings.llm_provider_chain that have at least one API key set."""
    names = [n.strip().lower() for n in settings.llm_provider_chain.split(",") if n.strip()]
    chain: list[str] = []
    for name in names:
        if name == "anthropic":
            if settings.anthropic_api_key:
                chain.append(name)
            continue
        provider = _OPENAI_COMPAT_PROVIDERS.get(name)
        if provider is None:
            logger.warning("Unknown LLM provider '%s' in LLM_PROVIDER_CHAIN -- skipping", name)
            continue
        if _keys_for(provider):
            chain.append(name)
    return chain


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


async def _post_chat_completion(
    client: httpx.AsyncClient, provider: _Provider, api_key: str, prompt: str, max_tokens: int
) -> str:
    model = getattr(settings, provider.model_setting, "") or provider.default_model
    url = f"{provider.base_url}/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        **(provider.extra_body or {}),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    ):
        with attempt:
            resp = await client.post(url, json=body, headers=headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    raise AssertionError("unreachable")  # AsyncRetrying always returns or raises


async def _call_anthropic(prompt: str, max_tokens: int) -> str:
    """Optional, no free tier. Lazily imported so a missing/old SDK never blocks startup."""
    import anthropic  # local import by design

    def _sync_call() -> str:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=settings.anthropic_model or "claude-3-5-haiku-20241022",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    # anthropic's SDK client is synchronous; run it off the event loop.
    return await asyncio.to_thread(_sync_call)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
async def complete_text(prompt: str, *, max_tokens: int = 1024) -> str:
    """Return raw text from the first provider in the chain that succeeds."""
    chain = _configured_chain()
    if not chain:
        raise AllProvidersFailed("No LLM provider is configured (all API keys empty).")

    errors: list[str] = []
    async with httpx.AsyncClient() as client:
        for name in chain:
            try:
                if name == "anthropic":
                    text = await _call_anthropic(prompt, max_tokens)
                else:
                    provider = _OPENAI_COMPAT_PROVIDERS[name]
                    keys = _keys_for(provider)
                    api_key = await _acquire_key(provider, keys, _estimate_tokens(prompt) + max_tokens)
                    text = await _post_chat_completion(client, provider, api_key, prompt, max_tokens)
                return text.strip()
            except Exception as exc:
                logger.warning("LLM provider '%s' failed: %s", name, exc)
                errors.append(f"{name}: {exc}")

    raise AllProvidersFailed("; ".join(errors))


async def complete_json(prompt: str, *, max_tokens: int = 1024) -> Any:
    """Like complete_text, but parses the (fence-stripped) result as JSON."""
    raw = await complete_text(prompt, max_tokens=max_tokens)
    return json.loads(strip_json_fences(raw))


async def complete_text_from(provider_name: str, prompt: str, *, max_tokens: int = 1024) -> str:
    """
    Call exactly one named provider, with no fallback to the rest of the chain.

    For callers that want independent answers from several *specific* providers
    at once (e.g. cross-provider ensemble classification) rather than "whichever
    provider answers first" -- complete_text()'s fallback semantics would just
    return the first success and never reach the others.
    """
    provider_name = provider_name.lower()
    if provider_name == "anthropic":
        if not settings.anthropic_api_key:
            raise AllProvidersFailed("anthropic has no configured key")
        return (await _call_anthropic(prompt, max_tokens)).strip()

    provider = _OPENAI_COMPAT_PROVIDERS.get(provider_name)
    if provider is None:
        raise AllProvidersFailed(f"unknown provider '{provider_name}'")
    keys = _keys_for(provider)
    if not keys:
        raise AllProvidersFailed(f"provider '{provider_name}' has no configured key")

    async with httpx.AsyncClient() as client:
        api_key = await _acquire_key(provider, keys, _estimate_tokens(prompt) + max_tokens)
        return (await _post_chat_completion(client, provider, api_key, prompt, max_tokens)).strip()


async def complete_json_from(provider_name: str, prompt: str, *, max_tokens: int = 1024) -> Any:
    """Like complete_text_from, but parses the (fence-stripped) result as JSON."""
    raw = await complete_text_from(provider_name, prompt, max_tokens=max_tokens)
    return json.loads(strip_json_fences(raw))


def configured_providers() -> list[str]:
    """
    Public wrapper around _configured_chain() for callers that need to pick
    specific providers themselves (e.g. ensemble classification) rather than use
    the automatic single-answer fallback chain.
    """
    return _configured_chain()
