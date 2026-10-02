"""
API smoke tests, run against the real FastAPI app via httpx's ASGI transport (no
running server process needed) but a real Postgres connection (DATABASE_URL) --
run these from inside the backend container, where that's already configured:

    docker compose exec backend pytest tests/test_api_smoke.py

build_topic_graph.delay() is mocked out so creating a topic doesn't actually
enqueue and run the full source-fetching/LLM pipeline on every test run.
"""
import httpx
import pytest

from app.main import app
from app.db.session import init_db, engine
import app.api.v1.topics as topics_module


@pytest.fixture(autouse=True)
async def _ensure_db_ready():
    # httpx.ASGITransport doesn't run FastAPI's lifespan (which normally calls
    # init_db()), so table creation / ANON_USER seeding would otherwise only
    # have happened if the real `backend` service happened to start first.
    # Calling it directly here makes the test suite self-sufficient either way.
    #
    # Function-scoped (not session-scoped) deliberately: the async engine's
    # pooled connections bind to whichever event loop opened them, and
    # pytest-asyncio gives each test function its own loop by default. A
    # session-scoped DB fixture would open connections on the *first* test's
    # loop, then every later test's own loop would hit the same "attached to a
    # different loop" RuntimeError fixed in tasks.py's _run(). init_db() is
    # idempotent (CREATE TABLE IF NOT EXISTS + check-then-insert ANON_USER), so
    # re-running it per test is cheap. Disposing afterwards, in this same loop,
    # is the other half of that fix -- see tasks.py's _run() for the full story.
    await init_db()
    yield
    await engine.dispose()


@pytest.fixture(autouse=True)
def _mock_celery_dispatch(monkeypatch):
    monkeypatch.setattr(topics_module.build_topic_graph, "delay", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _mock_ambiguity_check(monkeypatch):
    # Default every test to "not ambiguous" -- without this, POST /topics would
    # make a real synchronous LLM call per test whenever API keys are configured
    # in the environment pytest runs in. Tests exercising the ambiguous path
    # override this via their own monkeypatch.setattr call.
    async def fake_check_ambiguity(raw_query):
        return None

    monkeypatch.setattr(topics_module, "check_ambiguity", fake_check_ambiguity)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_create_topic_returns_202(client):
    resp = await client.post("/api/v1/topics", json={"raw_query": "test smoke topic"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["raw_query"] == "test smoke topic"
    assert body["status"] == "processing"
    assert "id" in body


async def test_create_topic_rejects_too_short_query(client):
    resp = await client.post("/api/v1/topics", json={"raw_query": "a"})
    assert resp.status_code == 422  # min_length=2 on TopicCreate.raw_query


async def test_get_topic_404_on_unknown_id(client):
    resp = await client.get("/api/v1/topics/00000000-0000-0000-0000-000000000099")
    assert resp.status_code == 404


async def test_get_graph_404_before_any_graph_exists(client):
    create_resp = await client.post("/api/v1/topics", json={"raw_query": "no graph yet topic"})
    topic_id = create_resp.json()["id"]

    resp = await client.get(f"/api/v1/topics/{topic_id}/graph")
    assert resp.status_code == 404


async def test_create_topic_returns_needs_clarification_for_ambiguous_query(client, monkeypatch):
    async def fake_check_ambiguity(raw_query):
        return [
            {"label": "Planet", "clarifying_query": "Mercury (planet)", "hint": "..."},
            {"label": "Element", "clarifying_query": "Mercury (chemical element)", "hint": "..."},
        ]

    monkeypatch.setattr(topics_module, "check_ambiguity", fake_check_ambiguity)

    resp = await client.post("/api/v1/topics", json={"raw_query": "Mercury"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "needs_clarification"
    assert len(body["clarification_options"]) == 2


async def test_create_topic_does_not_enqueue_when_ambiguous(client, monkeypatch):
    calls = []
    monkeypatch.setattr(topics_module.build_topic_graph, "delay", lambda *a, **k: calls.append(a))

    async def fake_check_ambiguity(raw_query):
        return [{"label": "Planet", "clarifying_query": "Mercury (planet)", "hint": "..."}]

    monkeypatch.setattr(topics_module, "check_ambiguity", fake_check_ambiguity)

    resp = await client.post("/api/v1/topics", json={"raw_query": "Mercury"})
    assert resp.status_code == 202
    assert calls == []  # pipeline must not start until /clarify resolves it


async def test_clarify_topic_starts_pipeline(client, monkeypatch):
    calls = []
    monkeypatch.setattr(topics_module.build_topic_graph, "delay", lambda *a, **k: calls.append(a))

    async def fake_check_ambiguity(raw_query):
        return [{"label": "Planet", "clarifying_query": "Mercury (planet)", "hint": "..."}]

    monkeypatch.setattr(topics_module, "check_ambiguity", fake_check_ambiguity)

    create_resp = await client.post("/api/v1/topics", json={"raw_query": "Mercury"})
    topic_id = create_resp.json()["id"]

    clarify_resp = await client.post(
        f"/api/v1/topics/{topic_id}/clarify",
        json={"chosen_query": "Mercury (planet)"},
    )
    assert clarify_resp.status_code == 202
    body = clarify_resp.json()
    assert body["status"] == "processing"
    assert body["raw_query"] == "Mercury (planet)"
    assert body["clarification_options"] is None
    assert len(calls) == 1  # pipeline fired exactly once, after clarification


async def test_clarify_topic_404_on_unknown_id(client):
    resp = await client.post(
        "/api/v1/topics/00000000-0000-0000-0000-000000000099/clarify",
        json={"chosen_query": "anything"},
    )
    assert resp.status_code == 404


async def test_clarify_topic_409_when_not_awaiting_clarification(client):
    create_resp = await client.post("/api/v1/topics", json={"raw_query": "already unambiguous"})
    topic_id = create_resp.json()["id"]

    resp = await client.post(
        f"/api/v1/topics/{topic_id}/clarify",
        json={"chosen_query": "anything"},
    )
    assert resp.status_code == 409


async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
