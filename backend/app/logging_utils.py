"""
Pipeline observability: one logging setup shared by FastAPI and Celery, plus a
lightweight per-run trace.

Why this exists: several fallbacks in this project (LLM chain -> spaCy extraction,
LLM summary -> raw passages, ensemble -> single provider) are *silent by design* --
the graph still builds, just worse. Before this module, a dead provider, a truncated
JSON response, or a poisoned cache entry produced a perfectly healthy-looking
"ready" topic and nothing in the logs tied the symptom back to its cause (see
CODEBASE_DOCUMENTATION.md §12). This module makes every step, every LLM attempt and
every degradation visible, correlated per topic/node run.

Pieces:
  - configure_logging(): idempotent root-logger setup (stdout, optional rotating file);
    every record carries a `run_id` (e.g. "topic:1a2b3c4d") so one build can be grepped
    out of interleaved worker output.
  - PipelineTrace / start_trace() / finish_trace(): collects step timings, LLM call
    outcomes and degradations for one run and emits a single SUMMARY line at the end.
  - step(): context manager that logs START / DONE (with duration) / FAIL for a stage.
  - degrade(): flag "this run fell back to a lower-quality path", at WARNING level.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import logging.handlers
import sys
import time
from collections import Counter
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(run_id)s | %(name)s | %(message)s"
_HANDLER_TAG = "_rh_handler"

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("rh_run_id", default="-")
_trace: contextvars.ContextVar["PipelineTrace | None"] = contextvars.ContextVar(
    "rh_trace", default=None
)
# Provider that served the most recent successful complete_text() in this context.
# Lets callers (extraction.py) report *who* answered without changing the
# complete_json() signature that tests and other call sites rely on.
last_llm_provider: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rh_last_llm_provider", default=None
)


# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
def configure_logging() -> None:
    """
    Configure the root logger. Safe to call repeatedly and from every process
    (FastAPI at import time, Celery via the `setup_logging` signal).
    """
    from app.config import get_settings

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    old_factory = logging.getLogRecordFactory()
    if not getattr(old_factory, "_rh_factory", False):
        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)
            record.run_id = _run_id.get()
            return record

        factory._rh_factory = True  # type: ignore[attr-defined]
        logging.setLogRecordFactory(factory)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if settings.log_file:
        handlers.append(logging.handlers.RotatingFileHandler(
            settings.log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8",
        ))
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)

    # httpx logs every request at INFO; llm.py logs a richer line per call already.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Per-run trace
# --------------------------------------------------------------------------- #
class PipelineTrace:
    def __init__(self, kind: str, subject_id: str):
        self.kind = kind
        self.subject_id = subject_id
        self.started = time.monotonic()
        self.steps: list[tuple[str, float, str]] = []   # (name, seconds, "ok"|"failed")
        self.llm_calls: list[dict[str, Any]] = []
        self.degradations: list[str] = []

    def summary(self, status: str) -> tuple[int, str]:
        """(log level, one-line summary) for this run."""
        total = time.monotonic() - self.started
        steps = ", ".join(
            f"{name}={secs:.1f}s{'' if st == 'ok' else '(FAILED)'}" for name, secs, st in self.steps
        ) or "none"
        ok = [c for c in self.llm_calls if c["ok"]]
        failed = len(self.llm_calls) - len(ok)
        served_by = dict(Counter(c["provider"] for c in ok)) or "none"
        level = logging.WARNING if (self.degradations or status != "ready") else logging.INFO
        line = (
            f"SUMMARY {self.kind} status={status} total={total:.1f}s | steps: {steps} | "
            f"llm_calls: {len(ok)} ok / {failed} failed, served_by={served_by} | "
            f"degraded: {self.degradations or 'no'}"
        )
        return level, line


def start_trace(kind: str, subject_id: str) -> PipelineTrace:
    """Begin a run: every log line emitted in this context now carries `kind:subject`."""
    trace = PipelineTrace(kind, str(subject_id))
    _run_id.set(f"{kind}:{str(subject_id)[:8]}")
    _trace.set(trace)
    last_llm_provider.set(None)
    logger.info("START %s %s", kind, subject_id)
    return trace


def finish_trace(status: str) -> None:
    trace = _trace.get()
    if trace is None:
        return
    level, line = trace.summary(status)
    logger.log(level, line)
    _trace.set(None)


def current_trace() -> PipelineTrace | None:
    return _trace.get()


def degrade(stage: str, reason: str) -> None:
    """Record that `stage` fell back to a lower-quality path. Always logged at WARNING."""
    trace = _trace.get()
    if trace is not None:
        trace.degradations.append(f"{stage}: {reason}")
    logger.warning("DEGRADED stage=%s reason=%s", stage, reason)


def record_llm_call(provider: str, ok: bool, seconds: float, **fields: Any) -> None:
    trace = _trace.get()
    if trace is not None:
        trace.llm_calls.append({"provider": provider, "ok": ok, "seconds": seconds, **fields})


@contextlib.contextmanager
def step(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """
    Log START / DONE (with duration) / FAIL around a pipeline stage. The yielded
    dict can be filled in by the body; its contents are appended to the DONE line
    (e.g. `info["docs"] = 12`).
    """
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("STEP %s START %s", name, extra)
    info: dict[str, Any] = {}
    t0 = time.monotonic()
    try:
        yield info
    except BaseException as exc:
        secs = time.monotonic() - t0
        trace = _trace.get()
        if trace is not None:
            trace.steps.append((name, secs, "failed"))
        logger.error("STEP %s FAIL after %.2fs: %s: %s", name, secs, type(exc).__name__, exc)
        raise
    secs = time.monotonic() - t0
    trace = _trace.get()
    if trace is not None:
        trace.steps.append((name, secs, "ok"))
    done = " ".join(f"{k}={v}" for k, v in info.items())
    logger.info("STEP %s DONE in %.2fs %s", name, secs, done)


def preview(text: str, head: int = 200, tail: int = 120) -> str:
    """Compact head...tail excerpt of a long string for log lines."""
    text = text.replace("\n", "\\n")
    if len(text) <= head + tail:
        return text
    return f"{text[:head]} ...[{len(text) - head - tail} chars]... {text[-tail:]}"
