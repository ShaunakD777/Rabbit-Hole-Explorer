# App Flow Document
## AI Internet Rabbit-Hole Explorer

---

## 1. High-Level User Flow

```
[Landing Page]
      │
      ▼
[User enters topic: "AGI"]
      │
      ▼
[Loading state: "Understanding your topic..."]
      │
      ▼
[Step 1: Topic Understanding — concept seed list shown]
      │
      ▼
[Step 2: Explore Sources — background fetch from Wikipedia/YouTube/Reddit/Papers/Web]
      │
      ▼
[Step 3: Knowledge Graph rendered — initial 8-12 nodes]
      │
      ▼
[Step 4: "Start Here" learning path highlighted/overlaid on graph]
      │
      ├──► [User clicks a node] ──► [Node detail panel opens]
      │                                   │
      │                                   ├──► [View 2-min / 10-min / deep-dive summary]
      │                                   ├──► [View sources: articles, videos, papers, threads]
      │                                   ├──► [Mark as "Learned" / "In Progress"]
      │                                   └──► [Expand Further → Step 5: Rabbit Hole]
      │
      ├──► [User searches within graph] ──► [Semantic search results highlighted on graph]
      │
      └──► [User opens Trends panel] ──► [Emerging sub-topics shown, clickable into graph]
```

## 2. Detailed Screen-by-Screen Flow

### 2.1 Landing / Entry
- Single prominent search bar: "What do you want to learn?"
- Optional: "I already know..." chips for prior-knowledge input (used to prune the learning path).
- Example topic chips for inspiration (Quantum Computing, AGI, Blockchain, etc.).

**Actions available:**
- Submit topic → proceeds to processing state.
- Select example chip → pre-fills and submits.

### 2.2 Processing State
- Multi-step progress indicator reflecting backend pipeline stages:
  1. Understanding topic
  2. Searching sources (Wikipedia, YouTube, Reddit, papers, web)
  3. Extracting concepts & relationships
  4. Building your knowledge graph
- Partial results may stream in via WebSocket (graph nodes appear progressively rather than a single blocking spinner).

### 2.3 Main Graph View (Core Screen)
- **Center:** Interactive force-directed / hierarchical graph.
- **Left sidebar:** "Start Here" learning path list (numbered, matches highlighted path on graph).
- **Top bar:** Search-within-graph (semantic search), topic title, "New Topic" button.
- **Right sidebar (collapsed by default):** Trends panel, Progress panel.
- **Node states (visual encoding):**
  - Not explored (default color)
  - In progress (highlighted border)
  - Learned (checkmark + muted color)
  - Currently selected (highlighted glow)

**Interactions:**
- Click node → opens Node Detail Panel (right-side drawer).
- Double-click / "Expand" button on node → triggers rabbit-hole expansion (Step 5).
- Drag nodes → manual layout adjustment (persisted per user per topic).
- Zoom/pan → standard graph navigation.

### 2.4 Node Detail Panel
- Node title + short description.
- Tabs: **Summary | Sources | Related**
  - **Summary tab:** toggle between 2-min / 10-min / deep-dive; deep-dive includes inline citations linking to source list.
  - **Sources tab:** list of contributing sources grouped by type (Wikipedia, YouTube videos w/ thumbnails, Reddit threads, papers, blogs/news), each opens externally.
  - **Related tab:** shows directly connected nodes with relationship labels (prerequisite of / related to / enables / breaks).
- Bottom actions: "Mark as Learned," "Mark as In Progress," "Expand Further."

### 2.5 Rabbit Hole Expansion Flow
1. User clicks "Expand Further" on a node.
2. Backend fetches additional sources scoped to that node + its immediate context.
3. New candidate nodes/edges are extracted, deduplicated against existing graph.
4. Graph animates new nodes appearing, connected to the origin node.
5. If expansion depth exceeds a soft cap (e.g., 3 levels deep in one session), user is prompted: "This is getting deep — want to keep going or start a focused path here?"

### 2.6 YouTube Lecture Extraction Flow
1. From Sources tab, user selects a YouTube video.
2. System checks cache; if not cached, fetches transcript and runs extraction.
3. Modal opens showing: generated notes outline, key concepts (linked back to graph nodes if they match), and a quiz (3–5 questions) with "Check Answers."

### 2.7 Trend Detection Panel
- Shows a ranked list of emerging sub-topics for research-heavy queries (e.g., "Quantum Error Correction — 240% growth in papers, last 12 months").
- Clicking a trend adds it as a new node connected to the relevant parent, with a "trending" badge.

### 2.8 Progress Tracking Flow
- Progress panel shows: % of graph explored, list of "Learned" nodes, suggested next node (highest-relevance unexplored node adjacent to already-learned nodes).
- On return visit, system re-opens the last active topic graph with saved progress state pre-applied.

### 2.9 Semantic Search-Within-Graph Flow
1. User types a natural-language query into the top search bar (e.g., "how do machines understand images").
2. Query is embedded and compared against existing node embeddings.
3. Matching nodes are highlighted/pulsed on the graph; if no strong match exists, system offers "Explore this as a new rabbit hole?" which triggers expansion seeded by the query.

## 3. Error / Edge-Case Flows
- **No sources found for a niche topic:** system falls back to a broader parent topic and informs the user ("Limited direct sources found — showing results for the closest matching field: X").
- **External API failure (e.g., YouTube quota exceeded):** that source type is silently skipped in the sources list with a small "temporarily unavailable" note; graph generation continues with remaining sources.
- **Graph becomes too large/cluttered:** system auto-collapses branches beyond a configurable node count and offers a "Focus Mode" that hides everything except the path from root to the selected node.
