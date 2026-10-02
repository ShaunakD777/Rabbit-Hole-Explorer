# Technical Requirements Document (TRD)
## AI Internet Rabbit-Hole Explorer

**Version:** 1.0
**Companion to:** PRD v1.0

---

## 1. System Architecture (High Level)

```
┌────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Frontend  │◄───►│   API Gateway /   │◄───►│   NLP Service Layer │
│ (React +   │     │   Backend (FastAPI│     │  (extraction, embed,│
│  D3/Cytoscape│    │   / Node.js)      │     │  summarize, graph)  │
└────────────┘     └──────────────────┘     └────────────────────┘
                              │                          │
                              ▼                          ▼
                     ┌────────────────┐        ┌──────────────────┐
                     │  Relational DB  │        │  Vector Store     │
                     │  (Postgres)     │        │  (pgvector/FAISS) │
                     └────────────────┘        └──────────────────┘
                              │
                              ▼
                     ┌────────────────────────────┐
                     │  Source Connectors          │
                     │  Wikipedia / YouTube /       │
                     │  Reddit / arXiv / Web Search │
                     └────────────────────────────┘
```

## 2. Component Breakdown

### 2.1 Frontend
- **Framework:** React + TypeScript
- **Graph rendering:** Cytoscape.js (preferred for large interactive graphs) or D3-force as an alternative
- **State management:** React Query (server cache) + Zustand (UI state)
- **Styling:** Tailwind CSS

### 2.2 Backend / API Layer
- **Framework:** FastAPI (Python) — chosen because the NLP pipeline is Python-native, avoids cross-language serialization overhead
- **Async task queue:** Celery + Redis (for long-running expansion/summarization jobs)
- **API style:** REST for CRUD, WebSocket channel for streaming graph-expansion updates to the frontend in real time

### 2.3 NLP Service Layer
| Sub-component | Technique | Notes |
|---|---|---|
| Topic Understanding | LLM-based keyword/entity extraction (few-shot prompt) + spaCy NER as fallback | Produces initial concept seed list |
| Topic Modeling | BERTopic (embedding-based clustering) over aggregated documents | Used for source categorization and trend detection |
| Semantic Search | Sentence-transformer embeddings (e.g., `all-MiniLM-L6-v2` or larger) stored in vector DB | Powers "related concept" retrieval and in-graph search |
| Relation Extraction | LLM prompt-based relation classification (`prerequisite_of`, `related_to`, `subtopic_of`, `enables`, `breaks`) with confidence scoring | Ablation-tested against a rule-based baseline (this IS the research contribution) |
| Summarization | LLM abstractive summarization, 3 prompt templates (2-min / 10-min / deep-dive) | Always grounded with source citations (RAG-style, not free generation) |
| Learning Path Detection | Topological sort (Kahn's algorithm) over the `prerequisite_of` subgraph | Falls back to relevance ranking if graph has cycles (cycle-breaking heuristic) |
| YouTube Extraction | YouTube Data API (metadata) + `youtube-transcript-api` (transcript) → LLM note/quiz generation | Cached per video ID to avoid recomputation |
| Trend Detection | Time-windowed topic modeling over arXiv/Semantic Scholar abstracts (last 12–24 months) | Ranks clusters by growth rate of paper count |

### 2.4 Data Layer
- **Relational DB (PostgreSQL):** users, topics, nodes, edges, summaries, progress, source_documents
- **Vector store:** `pgvector` extension on Postgres (simplifies infra — avoids a separate FAISS service for the academic prototype) storing embeddings for nodes and source chunks
- **Cache:** Redis for (a) API response caching from external sources, (b) Celery broker, (c) session/progress cache

### 2.5 Source Connectors
| Source | API | Auth | Rate Limit Handling |
|---|---|---|---|
| Wikipedia | MediaWiki REST API | None required | Cache aggressively (topics rarely change) |
| YouTube | YouTube Data API v3 | API Key | Quota-aware queue, batch requests |
| Reddit | Reddit API (PRAW) | OAuth app credentials | Backoff + local caching |
| Papers | Semantic Scholar API / arXiv API | None / rate-limited | Cache, and batch by topic |
| General web | SerpAPI or Bing Web Search API (or Tavily) | API Key | Used sparingly, primarily for blogs/news |

## 3. NLP Pipeline Detail (Core Research Component)

### 3.1 Pipeline Stages
1. **Slot Extraction** — extract candidate entities/phrases from raw query and retrieved documents.
2. **Ontology/Embedding Matching** — map extracted phrases to canonical concept nodes using embedding similarity + a curated domain ontology (avoids duplicate nodes like "AI" vs "Artificial Intelligence").
3. **Relation Classification** — LLM classifies the relationship type between concept pairs with a confidence score.
4. **Clarifying / Disambiguation Step** — if confidence is below threshold, the system either asks a lightweight clarifying question or defaults to `related_to` with lower edge weight.
5. **Graph Assembly** — nodes + typed, weighted edges are persisted; duplicate/near-duplicate nodes are merged via embedding cosine similarity threshold (e.g., > 0.92).

### 3.2 Evaluation / Ablation Design
Since this is an NLP course project, the pipeline should be evaluated with an ablation study:
- **Baseline A:** Keyword co-occurrence only (no embeddings, no LLM).
- **Baseline B:** Embedding similarity only (no relation classification).
- **Full system:** Embedding + ontology matching + LLM relation classification.
- **Metrics:** precision/recall on a hand-labeled test set of concept pairs (~150–300 pairs across 3 seed topics), plus human-rated path quality (Likert 1–5).
- This ablation table is the empirical results section of the accompanying research paper.

## 4. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | Initial graph (5–10 nodes) generated in < 20s cold, < 8s cached |
| Scalability | Academic prototype: support ~50 concurrent users; architecture should not preclude horizontal scaling (stateless API workers) |
| Reliability | Graceful degradation if a single source API is down (skip source, don't fail whole request) |
| Caching | All external API responses and LLM outputs cached by (topic, source) key with TTL (7–30 days depending on source volatility) |
| Cost control | LLM calls batched where possible; summarization only computed on-demand (lazy), not pre-generated for unexplored nodes |
| Security | API keys stored server-side only; no client-side exposure; basic auth/session for progress tracking |
| Observability | Structured logging of pipeline stage latencies + LLM token usage per request |

## 5. Tech Stack Summary

| Layer | Choice |
|---|---|
| Frontend | React, TypeScript, Cytoscape.js, Tailwind CSS |
| Backend API | Python, FastAPI |
| Async jobs | Celery + Redis |
| Database | PostgreSQL + pgvector |
| NLP/LLM | Anthropic Claude API (extraction, relation classification, summarization) + sentence-transformers (embeddings) + BERTopic (topic modeling) |
| Source APIs | Wikipedia, YouTube Data API v3, Reddit API, Semantic Scholar/arXiv API, SerpAPI/Bing |
| Deployment | Docker Compose (dev); single VM or small container cluster for demo |

## 6. External Dependencies & API Keys Required
- YouTube Data API v3 key
- Reddit OAuth app (client ID/secret)
- Semantic Scholar API key (optional, higher rate limit)
- SerpAPI/Bing Search key (or Tavily) for general web/blog search
- LLM provider API key (Anthropic Claude)

## 7. Data Retention & Privacy
- Only public content is ingested; no scraping behind logins/paywalls.
- User accounts store: email/username, progress state, saved graphs. No sensitive personal data collected.
- Summaries store a link back to the original source; no full copyrighted text is stored verbatim.
