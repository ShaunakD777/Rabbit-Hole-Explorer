# 🐇 AI Rabbit Hole Explorer

An NLP-powered knowledge graph explorer, built as a Natural Language Processing
course project (Semester VII). Enter any topic and get an interactive, expandable
knowledge graph with learning paths, multi-depth summaries, YouTube extraction,
trend detection, and progress tracking.

**Goal:** let you learn a topic *without* falling into a rabbit hole. You get a
bounded map of the topic, an ordered "Start Here" path through it, a clear next
step and visible progress. Expanding a node lets you take a detour, but detours are
capped so you can always get back to the main path. The name describes the problem
the app solves, not what it encourages. A secondary goal is the NLP research
behind it: an ablation of relation-classification methods (see `backend/eval/`).

**Stack:** React + TypeScript (Vite, Cytoscape) · FastAPI · Celery · PostgreSQL + pgvector · Redis

---

## Features

- **Topic → knowledge graph:** fetches sources (Wikipedia, arXiv / Semantic Scholar,
  YouTube, Reddit, web search), extracts up to 12 key concepts, and classifies the
  relations between them (`prerequisite_of`, `subtopic_of`, `enables`, `related_to`, `breaks`).
- **Ambiguity check:** queries with several unrelated meanings ("Mercury", "Java")
  ask you to pick a sense before the graph is built.
- **Learning path:** a topological ordering of the graph's prerequisites.
- **RAG-grounded summaries:** each node summarized at three depths (`2min`, `10min`,
  `deepdive`) from retrieved source passages (pgvector similarity search).
- **Rabbit-hole expansion:** expand any node into its own sub-concepts, deduplicated
  against the existing graph by embedding similarity.
- **Semantic search, trend detection (BERTopic), YouTube extraction, progress tracking.**
- **Real-time updates:** pipeline progress is streamed to the browser over WebSockets.

### Free-tier only

No paid API is required. LLM calls go through an ordered fallback chain:
**Gemini → Groq → Cerebras → Mistral → (optional) Anthropic → local spaCy**.
Any provider whose key is blank is skipped, and if none is configured, concept
extraction falls back to local spaCy. Results are better with at least one LLM key.

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Optional: free-tier API keys (see below)

### 1. Configure environment

```bash
cp backend/.env.example backend/.env
# then edit backend/.env and fill in whichever keys you have
```

> ⚠️ `backend/.env` holds your real API keys. It is listed in `.gitignore`.
> **Never commit it.** Only `.env.example` (placeholders) belongs in git.

Every key is optional. Leave a key blank to skip that provider or source.

| Variable | Used for | Notes |
|---|---|---|
| `GEMINI_API_KEY` | LLM (1st in chain) | Recommended. Free key from [Google AI Studio](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | LLM (2nd in chain) | Recommended. Free key from [Groq Console](https://console.groq.com/keys) |
| `CEREBRAS_API_KEY`, `MISTRAL_API_KEY` | Additional LLM fallbacks | Optional |
| `ANTHROPIC_API_KEY` | LLM, last in chain | Optional, paid; not needed |
| `LLM_PROVIDER_CHAIN` | Provider order | e.g. `gemini,groq` |
| `YOUTUBE_API_KEY` | YouTube source + video extraction | YouTube Data API v3 |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit source | Optional |
| `SEMANTIC_SCHOLAR_API_KEY` | Papers source | Optional; arXiv is used as a keyless fallback |
| `SERPAPI_KEY`, `SERPER_API_KEY` | Web search source | Optional; SerpAPI is tried first, then Serper |

Each LLM and search key accepts several comma-separated keys (`key1,key2`). Each
key gets its own rate-limit bucket, which raises throughput on free tiers.

Wikipedia and arXiv need no key, so the app produces a graph even with an empty `.env`.

### 2. Start everything

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8001 |
| API docs (Swagger) | http://localhost:8001/docs |
| Postgres | localhost:5432 |
| Redis | localhost:6379 |

The backend listens on port 8000 inside its container; Docker maps it to **8001**
on the host.

### 3. Local development (without Docker for the app)

Start only the databases with Docker:

```bash
docker compose up -d db redis
```

**Backend** (serves on port 8000, which the frontend uses by default outside Docker):
```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --reload --port 8000
```

**Celery worker** (separate terminal, same venv):
```bash
cd backend
celery -A app.worker worker --loglevel=info
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev                       # http://localhost:3000
```

---

## Testing

```bash
# Backend (needs Postgres + Redis running)
docker compose exec backend pytest -v

# Frontend type-check + production build, and lint
cd frontend
npm run build
npm run lint
```

---

## Ablation Study

The project's research component compares four relation-classification variants
on hand-labeled concept pairs:

| Variant | Approach | LLM calls per graph |
|---|---|---|
| `keyword` | co-occurrence in the same source chunk | 0 |
| `embedding` | cosine similarity of sentence embeddings | 0 |
| `llm_pairwise` | one LLM call per concept pair | O(n² − n) |
| `llm_batched` | one LLM call for the whole graph (the app's default) | 1 |

```bash
docker compose exec backend python -m eval.capture_fixtures --topic "quantum computing" --slug quantum_computing
docker compose exec backend python -m eval.run_ablation --topic quantum_computing --variant all
docker compose exec backend python -m eval.report    # -> eval/out/summary.csv, confusion matrices, f1_comparison.png
```

The labels live in `backend/eval/labels/`. The harness and a 14-pair starter set
are in place, but the full 150–300-pair dataset is still to be labeled. See
[backend/eval/README.md](backend/eval/README.md) for details.

---

## Project Structure

```
rabbit-hole-explorer/
├── backend/
│   ├── app/
│   │   ├── api/v1/           # REST endpoints (topics, nodes, search, progress, trends, videos)
│   │   ├── api/ws.py         # WebSocket channel (Redis pub/sub bridge from Celery)
│   │   ├── connectors/       # Wikipedia, YouTube, Reddit, Papers (Semantic Scholar/arXiv), Web (SerpAPI/Serper)
│   │   ├── db/               # SQLAlchemy models + session
│   │   ├── nlp/              # LLM chain, extraction, intent/ambiguity, embeddings,
│   │   │                     #   graph builder, summarizer, trend detector
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── cache.py          # Redis helpers
│   │   ├── config.py         # Settings (pydantic-settings, reads backend/.env)
│   │   ├── main.py           # FastAPI app
│   │   ├── tasks.py          # Celery tasks
│   │   └── worker.py         # Celery app definition
│   ├── eval/                 # Ablation + ambiguity evaluation harness
│   ├── tests/                # pytest suite
│   ├── migrations/init.sql   # Schema + pgvector extension
│   ├── .env.example          # Template for backend/.env
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/client.ts     # Axios API client
│   │   ├── components/       # Graph canvas, node drawer, panels, pipeline progress
│   │   ├── hooks/            # useWebSocket
│   │   ├── pages/            # LandingPage, GraphWorkspace
│   │   ├── store/appStore.ts # Zustand global state
│   │   └── types/index.ts    # TypeScript types
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.ts
├── docs/                     # PRD, TRD, app flow, UI/UX, schema, implementation plan
├── CODEBASE_DOCUMENTATION.md # Detailed, verified reference for every subsystem
└── docker-compose.yml
```

---

## Implementation Phases

| Phase | Status | Description |
|---|---|---|
| 0 — Scaffold | ✅ | Docker Compose, FastAPI, React/TS, Postgres+pgvector |
| 1 — Connectors | ✅ | Wikipedia, YouTube, Reddit, Papers (Semantic Scholar/arXiv), Web (SerpAPI/Serper) |
| 2 — NLP Pipeline | ✅ | Free-tier LLM chain, extraction, batched relation classification, dedup |
| 3 — Graph + Path | ✅ | Centrality scoring, topological-sort learning path, persistence |
| 4 — Frontend MVP | ✅ | Cytoscape canvas, learning path, node drawer, WebSocket |
| 5 — Summarization + Expansion | ✅ | RAG-grounded summaries, rabbit-hole expansion |
| 6 — Advanced Features | ✅ | Ambiguity check, YouTube extraction, trends, progress, semantic search |
| 7 — Ablation Study | 🟡 | Harness and starter label set done; full labeled dataset pending |
| 8 — Polish + Write-up | 🔜 | Final UI polish, load testing, research paper |

---

## API Reference

Full interactive docs are at http://localhost:8001/docs while the app runs under Docker.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/topics` | Submit a topic (202; may return `needs_clarification`) |
| `POST` | `/api/v1/topics/{topic_id}/clarify` | Pick a sense for an ambiguous topic |
| `GET` | `/api/v1/topics/{topic_id}` | Topic status |
| `GET` | `/api/v1/topics/{topic_id}/graph` | Nodes, edges and learning path |
| `GET` | `/api/v1/nodes/{node_id}/summary?depth=2min\|10min\|deepdive` | Node summary |
| `POST` | `/api/v1/nodes/{node_id}/expand` | Trigger rabbit-hole expansion |
| `GET` | `/api/v1/nodes/{node_id}/sources` | Source documents behind a node |
| `GET` | `/api/v1/graph/{graph_id}/search?q=...` | Semantic search |
| `POST` | `/api/v1/progress/{user_id}/{node_id}` | Update learning progress |
| `GET` | `/api/v1/progress/{user_id}/graph/{graph_id}` | Progress for a graph |
| `GET` | `/api/v1/trends/{topic_id}` | Research trends |
| `POST` | `/api/v1/videos/{source_document_id}/extract` | YouTube extraction |
| `WS` | `/ws/channel/{channel_id}` | Real-time pipeline updates |
| `GET` | `/health` | Health check |

> There is no authentication. This is a deliberate scope decision for an academic
> project: every request runs as a single seeded anonymous user.

---

## Documentation

- [CODEBASE_DOCUMENTATION.md](CODEBASE_DOCUMENTATION.md): architecture, data model,
  every subsystem, and known/fixed issues
- [docs/](docs/): product requirements, technical design, app flow, UI/UX, schema,
  and implementation plan
