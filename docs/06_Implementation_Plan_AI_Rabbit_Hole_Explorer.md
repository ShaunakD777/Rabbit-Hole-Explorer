# Implementation Plan
## AI Internet Rabbit-Hole Explorer

**Context:** NLP course project — combines a working software system with material suitable for a research write-up (the relation-extraction pipeline + ablation study is the research contribution).

---

## 1. Suggested Timeline (12–14 Week Semester Plan)

### Phase 0 — Setup (Week 1)
- Repo scaffolding: frontend (React/TS), backend (FastAPI), Postgres + pgvector via Docker Compose.
- Register API keys: YouTube Data API, Reddit API, Semantic Scholar/arXiv, web search provider, LLM provider.
- Define initial DB schema (from Backend Schema doc) and run first migration.

### Phase 1 — Source Connectors (Weeks 2–3)
- Build individual connector modules per source (Wikipedia, YouTube, Reddit, papers, web search).
- Each connector: query → normalize → dedupe → store into `source_documents`.
- Add Redis caching layer for all external calls.
- **Deliverable:** given a topic string, system can fetch and store raw documents from all 5 sources.

### Phase 2 — NLP Core Pipeline v1 (Weeks 4–6)
- Implement Step 1 (Topic Understanding / slot extraction) using LLM few-shot prompting + spaCy fallback.
- Implement embedding generation (sentence-transformers) for nodes and source chunks; store via pgvector.
- Implement ontology/ embedding-based concept matching and deduplication.
- Implement relation classification (LLM-based) producing typed, weighted edges.
- **Deliverable:** given raw documents, system outputs a structured node/edge list (this is the core research artifact — start logging outputs for later evaluation).

### Phase 3 — Graph Construction & Learning Path (Weeks 6–7)
- Persist nodes/edges into `graphs`/`nodes`/`edges` tables.
- Implement topological sort for `prerequisite_of` edges → "Start Here" path.
- Handle cycle-breaking heuristic (drop lowest-confidence edge in a detected cycle).
- **Deliverable:** `/topics` → `/topics/{id}/graph` endpoint returns a usable graph + ordered path.

### Phase 4 — Frontend Graph Workspace (Weeks 7–9)
- Build landing page + topic submission flow.
- Integrate Cytoscape.js graph canvas with node states, learning-path sidebar, and node-detail drawer (per App Flow / UI-UX docs).
- Wire up WebSocket channel for streaming node updates during processing/expansion.
- **Deliverable:** end-to-end flow from topic input to interactive graph in the browser.

### Phase 5 — Summarization, Sources Tab, Rabbit-Hole Expansion (Weeks 9–10)
- Implement 3-depth grounded summarization (RAG over `source_chunks`).
- Implement `/nodes/{id}/expand` endpoint reusing Phase 1–3 pipeline scoped to a node.
- Implement expansion depth caps + "Focus Mode" in frontend.
- **Deliverable:** clicking any node produces summaries and can expand into new nodes.

### Phase 6 — Advanced Features (Weeks 10–12)
- YouTube transcript extraction → notes + quiz generation (`video_extractions`).
- Trend detection job (BERTopic over recent papers, time-windowed clustering) → `trend_snapshots`.
- Progress tracking (`user_progress`) + "suggested next node" recommendation logic.
- Semantic search-within-graph.

### Phase 7 — Evaluation / Ablation Study (Weeks 11–13, overlaps Phase 6)
- Hand-label a test set (~150–300 concept pairs across 3 seed topics, e.g., Quantum Computing, AGI, Blockchain).
- Run three pipeline variants: keyword co-occurrence baseline, embedding-only baseline, full pipeline.
- Compute precision/recall/F1 for relation classification; collect human Likert ratings (1–5) for learning-path quality and node-expansion relevance.
- Compile results into tables/plots for the paper ("black book").

### Phase 8 — Polish, Testing, Write-up (Weeks 13–14)
- UI polish pass (empty/loading/error states, accessibility check, responsive behavior).
- Load-test with concurrent topic submissions; verify caching effectiveness.
- Finalize research paper using ablation results as the empirical section.
- Prepare demo script + slides for evaluation/defense.

---

## 2. Milestone Summary Table

| Milestone | Week | Definition of Done |
|---|---|---|
| M1: Source ingestion working | 3 | All 5 connectors return and store documents for a test topic |
| M2: NLP extraction pipeline v1 | 6 | Given documents, system outputs structured nodes/edges with confidence scores |
| M3: Graph + learning path API | 7 | `/topics/{id}/graph` returns valid graph with topological "Start Here" order |
| M4: Interactive frontend MVP | 9 | Full click-through demo: topic → graph → node detail → summary |
| M5: Rabbit-hole expansion live | 10 | Clicking "Expand" grows the graph in real time |
| M6: Advanced features complete | 12 | YouTube extraction, trends, progress tracking, semantic search all functional |
| M7: Ablation study complete | 13 | Results table finalized, ready to insert into paper |
| M8: Final polish + submission | 14 | App demo-ready, paper draft complete |

---

## 3. Team Role Split (if applicable / adaptable for solo work)

| Role | Responsibilities |
|---|---|
| NLP/Backend Lead | Extraction pipeline, relation classification, embeddings, ablation study |
| Full-stack/API Lead | FastAPI endpoints, DB schema, connectors, caching |
| Frontend Lead | Graph workspace, UI states, accessibility |
| Research/Write-up Lead | Test-set labeling, evaluation metrics, paper drafting |

*(If working solo, follow the phase order sequentially — the NLP pipeline in Phase 2 should be built first and most carefully, since it doubles as the research contribution.)*

## 4. Risk-Adjusted Contingencies
- **If LLM cost/rate limits become a blocker:** fall back to smaller/local embedding models for search and reserve LLM calls only for relation classification and summarization (the two steps that most need generative reasoning).
- **If a source API is unreliable during development (e.g., Reddit):** stub it with a fixture dataset so pipeline development isn't blocked; re-enable live calls before the final demo.
- **If graph UI performance degrades with large graphs:** enforce the node-count cap earlier than planned and ship "Focus Mode" ahead of schedule.
- **If ablation labeling takes longer than expected:** shrink test set to 2 seed topics (still statistically presentable) rather than cutting the study entirely — it is the core research deliverable.

## 5. Definition of "Done" for the Overall Project
1. A user can enter any reasonably well-known topic and get a coherent, navigable knowledge graph within the target latency.
2. The learning path is topologically valid and rated ≥ 4.0/5 by human evaluators on sample topics.
3. The ablation study demonstrates the full pipeline (ontology + embeddings + LLM relation classification) outperforms both baselines on precision/recall.
4. The research paper documents the pipeline design, evaluation methodology, and results in a form suitable for the "black book" submission.
