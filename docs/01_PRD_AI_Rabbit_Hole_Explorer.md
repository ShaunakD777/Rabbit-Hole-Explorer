# Product Requirements Document (PRD)
## AI Internet Rabbit-Hole Explorer

**Version:** 1.0
**Course:** NLP
**Document Type:** Product Requirements Document

---

## 1. Overview

### 1.1 Problem Statement
Learners exploring an unfamiliar domain (e.g., Artificial General Intelligence) face three core problems:
1. **No starting point** — they don't know which sub-topic to learn first.
2. **No structure** — information is scattered across Wikipedia, YouTube, Reddit, blogs, and papers with no connecting thread.
3. **No sense of progress** — there is no way to see what has been learned, what remains, or how topics relate to each other.

### 1.2 Proposed Solution
An AI-powered system that takes a single broad topic as input and:
- Extracts the key sub-concepts using NLP.
- Searches and aggregates content from multiple heterogeneous sources.
- Builds an interactive, visual knowledge graph connecting these concepts.
- Infers a beginner-friendly learning path (topological ordering of prerequisites).
- Lets the learner "fall down the rabbit hole" by clicking any node to expand it further, with the graph growing dynamically.
- Tracks what has been explored/completed and recommends the next node.

### 1.3 Target Users
| Persona | Description | Core Need |
|---|---|---|
| Curious Beginner | Wants to learn a new domain (e.g., "Quantum Computing") from zero | A guided roadmap with no prior context needed |
| Self-directed Researcher | Wants to track emerging trends in a field | Trend detection + graph of active research areas |
| Student preparing for exams/interviews | Needs structured notes fast | Auto-summarization + quiz generation from source material |

### 1.4 Goals & Non-Goals

**Goals**
- Turn one search term into a structured, explorable knowledge map within seconds.
- Automatically infer prerequisite ordering between concepts.
- Support incremental, click-driven graph expansion ("rabbit holes").
- Summarize any node at 3 levels of depth (2-min / 10-min / deep-dive).
- Track learner progress across sessions.

**Non-Goals (v1)**
- Not a replacement for structured courses (MOOCs) — it is a discovery/orientation layer.
- Not building a general-purpose search engine; it curates/aggregates from existing sources, it does not crawl the entire web.
- No real-time collaborative graph editing in v1 (single-user only).
- No support for non-English content in v1.

---

## 2. Core Features (Functional Scope)

### F1 — Topic Understanding (NLP Extraction)
- Input: free-text topic (e.g., "Artificial General Intelligence").
- System extracts 5–10 candidate key concepts using keyword extraction + entity linking.
- Output is shown to the user as an initial concept seed list before graph generation.

### F2 — Multi-Source Exploration
- System issues parallel queries to: Wikipedia API, YouTube Data API, Reddit API, arXiv/Semantic Scholar API, and general web search (news/blogs).
- Retrieved documents are deduplicated and passed into the NLP pipeline.

### F3 — Knowledge Graph Construction
- Concepts become graph nodes; relationships (`prerequisite_of`, `related_to`, `subtopic_of`, `breaks`/`enables` for causal links) become edges.
- Graph is hierarchical but not strictly a tree — cross-links are allowed.

### F4 — Learning Path Detection
- System performs a topological sort over `prerequisite_of` edges to produce a linear "Start Here" roadmap.
- Path adapts if the user indicates prior knowledge (skips known nodes).

### F5 — Rabbit Hole Expansion
- Clicking a node triggers on-demand expansion: new child/related nodes are fetched, summarized, and merged into the graph.
- Expansion is incremental — the graph is never fully pre-computed for arbitrarily deep topics.

### F6 — Multi-Level Summarization
- Every node offers three summary depths: 2-minute, 10-minute, and deep-dive (with citations to source documents).

### F7 — YouTube Lecture Extraction
- For video sources, the system pulls transcripts, generates a note outline, extracts key concepts, and produces 3–5 quiz questions per video.

### F8 — Research Trend Detection
- For research-heavy topics, the system clusters recent papers (last 12–24 months) by topic modeling and highlights emerging sub-areas.

### F9 — Progress Tracking
- User can mark nodes as "learned" / "in progress."
- System stores this per user and biases recommendations toward unexplored, high-relevance nodes.

### F10 — Semantic Search Within Graph
- A search bar lets the user query the existing graph using natural language; results are ranked by embedding similarity, not keyword match.

---

## 3. User Stories

1. As a beginner, I want to type "AGI" and immediately see a roadmap of what to learn first, so I don't waste time guessing.
2. As a learner, I want to click on any concept and see it expand with related ideas, so I can explore naturally instead of following a rigid course.
3. As a busy student, I want a 2-minute summary of a topic before committing to the full deep-dive.
4. As a researcher, I want to see which sub-areas of a field are trending in papers published this year.
5. As a returning user, I want the system to remember what I've already learned and suggest what's next.
6. As a visual learner, I want to watch a lecture video and get auto-generated notes and quiz questions instead of re-watching it.

---

## 4. Success Metrics

| Metric | Target |
|---|---|
| Time from query to first usable graph | < 8 seconds (cached), < 20 seconds (cold) |
| Learning path accuracy (human-rated, 1–5) | ≥ 4.0 average |
| Node expansion relevance (precision@5, human-rated) | ≥ 80% |
| Summary factual accuracy (spot-checked against source) | ≥ 90% |
| Returning-user session rate | Tracked, no hard target in academic prototype |

---

## 5. Assumptions & Constraints
- Prototype scope is limited to public/free-tier APIs (Wikipedia, YouTube Data API v3, Reddit API, arXiv/Semantic Scholar API).
- LLM calls (extraction, summarization, relation inference) are rate- and cost-constrained; caching is mandatory.
- Single-user academic deployment; no enterprise auth/multitenancy required.
- English-language content only in v1.

## 6. Out of Scope for v1
- Mobile native apps (web-responsive only).
- Real-time multi-user collaboration on the same graph.
- Paid/subscription content ingestion (e.g., paywalled courses).
- Non-English NLP pipeline.

## 7. Risks
| Risk | Impact | Mitigation |
|---|---|---|
| API rate limits (YouTube/Reddit) throttle exploration | Broken/slow UX | Aggressive caching, backoff, source rotation |
| LLM hallucination in relation extraction ("breaks", "enables") | Wrong prerequisite ordering | Human-readable citations per edge; confidence scores; ablation-tested extraction pipeline |
| Graph grows unbounded and becomes unreadable | Cognitive overload | Depth/breadth caps per expansion; collapse/expand UI |
| Copyright concerns in summarization | Legal/academic integrity issue | Summaries are paraphrased, never verbatim; always cite source link |
