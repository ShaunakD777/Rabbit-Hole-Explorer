# Backend Schema Document
## AI Internet Rabbit-Hole Explorer

---

## 1. Entity-Relationship Overview

```
users ──< user_progress >── nodes
  │                            │
  └──< topics >── graphs ──< nodes >── < edges >
                                │
                                ├──< summaries
                                └──< node_sources >── source_documents

source_documents ──< source_chunks (embeddings)

video_extractions (linked to source_documents where type = youtube)
trend_snapshots (linked to topics)
```

## 2. Table Definitions

### 2.1 `users`
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| email | TEXT UNIQUE | |
| display_name | TEXT | |
| prior_knowledge_tags | TEXT[] | free-text topics user says they already know |
| created_at | TIMESTAMPTZ | |

### 2.2 `topics`
Represents a root query the user started (e.g., "AGI").
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| user_id | UUID (FK → users.id) | nullable for anonymous sessions |
| raw_query | TEXT | original user input |
| normalized_label | TEXT | canonicalized topic label |
| status | ENUM('processing','ready','failed') | |
| created_at | TIMESTAMPTZ | |

### 2.3 `graphs`
A graph is a versioned snapshot container tied to a topic (allows regeneration without destroying history).
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| topic_id | UUID (FK → topics.id) | |
| version | INTEGER | increments on major regeneration |
| root_node_id | UUID (FK → nodes.id, nullable) | |
| created_at | TIMESTAMPTZ | |

### 2.4 `nodes`
Each node = one concept.
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| graph_id | UUID (FK → graphs.id) | |
| label | TEXT | canonical concept name |
| aliases | TEXT[] | alternate phrasings merged into this node |
| description_short | TEXT | 1–2 line description |
| embedding | VECTOR(384 or 1536) | pgvector column, for semantic search/dedup |
| category | TEXT | e.g., "Hardware," "Algorithms," "Security" (from topic modeling) |
| importance_score | FLOAT | degree centrality or relevance score, drives node sizing |
| is_trending | BOOLEAN | set by trend detection job |
| depth_level | INTEGER | distance from root node, used for expansion caps |
| created_at | TIMESTAMPTZ | |

### 2.5 `edges`
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| graph_id | UUID (FK → graphs.id) | |
| source_node_id | UUID (FK → nodes.id) | |
| target_node_id | UUID (FK → nodes.id) | |
| relation_type | ENUM('prerequisite_of','related_to','subtopic_of','enables','breaks') | |
| confidence | FLOAT | 0–1, from relation classification step |
| created_at | TIMESTAMPTZ | |

Unique constraint: (`graph_id`, `source_node_id`, `target_node_id`, `relation_type`) to avoid duplicate edges.

### 2.6 `summaries`
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| node_id | UUID (FK → nodes.id) | |
| depth | ENUM('2min','10min','deepdive') | |
| content | TEXT | generated summary |
| citations | JSONB | array of `{source_document_id, span}` references |
| generated_at | TIMESTAMPTZ | |

Unique constraint: (`node_id`, `depth`) — one cached summary per depth per node.

### 2.7 `source_documents`
Raw retrieved documents from external sources, before/after processing.
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| source_type | ENUM('wikipedia','youtube','reddit','paper','web') | |
| url | TEXT | |
| title | TEXT | |
| author_or_channel | TEXT | nullable |
| published_at | TIMESTAMPTZ | nullable |
| raw_text | TEXT | extracted/cleaned text (transcript, article body, thread text) |
| fetched_at | TIMESTAMPTZ | |
| metadata | JSONB | source-specific fields (e.g., subreddit, view count, citation count) |

### 2.8 `source_chunks`
Chunked + embedded pieces of `source_documents`, used for RAG-style grounded summarization and semantic search.
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| source_document_id | UUID (FK → source_documents.id) | |
| chunk_text | TEXT | |
| chunk_index | INTEGER | |
| embedding | VECTOR(384 or 1536) | pgvector |

### 2.9 `node_sources`
Many-to-many mapping between nodes and the documents that contributed to them.
| Column | Type | Notes |
|---|---|---|
| node_id | UUID (FK → nodes.id) | |
| source_document_id | UUID (FK → source_documents.id) | |
| relevance_score | FLOAT | |
| PRIMARY KEY | (node_id, source_document_id) | |

### 2.10 `video_extractions`
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| source_document_id | UUID (FK → source_documents.id, source_type='youtube') | |
| notes_outline | TEXT | generated structured notes |
| key_concepts | TEXT[] | |
| quiz | JSONB | array of `{question, options[], correct_index}` |
| generated_at | TIMESTAMPTZ | |

### 2.11 `trend_snapshots`
| Column | Type | Notes |
|---|---|---|
| id | UUID (PK) | |
| topic_id | UUID (FK → topics.id) | |
| cluster_label | TEXT | emerging sub-topic name |
| growth_rate | FLOAT | % change in paper/document count over window |
| window_start | DATE | |
| window_end | DATE | |
| linked_node_id | UUID (FK → nodes.id, nullable) | set once user adds trend to graph |

### 2.12 `user_progress`
| Column | Type | Notes |
|---|---|---|
| user_id | UUID (FK → users.id) | |
| node_id | UUID (FK → nodes.id) | |
| status | ENUM('not_started','in_progress','learned') | |
| updated_at | TIMESTAMPTZ | |
| PRIMARY KEY | (user_id, node_id) | |

## 3. Indexing Strategy
- `nodes.embedding` — IVFFlat or HNSW index (pgvector) for approximate nearest-neighbor semantic search.
- `source_chunks.embedding` — same, used for grounded summarization retrieval.
- `edges (source_node_id)`, `edges (target_node_id)` — B-tree indexes for fast graph traversal queries.
- `nodes (graph_id, depth_level)` — supports expansion-depth-capped queries.
- `user_progress (user_id)` — fast lookup of a user's full progress state on login.

## 4. Caching Layer (Redis)
| Key pattern | Value | TTL |
|---|---|---|
| `source:{source_type}:{query_hash}` | raw API response | 7–30 days |
| `llm:extract:{content_hash}` | extraction result | 30 days |
| `llm:summary:{node_id}:{depth}` | generated summary | 30 days (invalidated on regeneration) |
| `graph:{topic_id}:layout` | last saved manual node positions per user | indefinite (until user resets) |

## 5. API Surface (for reference, maps to schema above)
- `POST /topics` → creates a `topics` row, kicks off Step 1–4 pipeline asynchronously, returns `topic_id`.
- `GET /topics/{id}/graph` → returns current `nodes` + `edges` for the latest `graphs` version.
- `POST /nodes/{id}/expand` → triggers rabbit-hole expansion job, streams new nodes/edges via WebSocket.
- `GET /nodes/{id}/summary?depth=2min|10min|deepdive` → returns cached or freshly generated summary.
- `GET /graph/{id}/search?q=...` → semantic search against `nodes.embedding`.
- `POST /nodes/{id}/progress` → upserts `user_progress`.
- `GET /topics/{id}/trends` → returns `trend_snapshots`.
- `POST /videos/{source_document_id}/extract` → triggers/returns `video_extractions`.

## 6. Data Volume Estimates (Academic Prototype Scale)
- ~50 concurrent topics, each graph averaging 30–80 nodes after moderate exploration.
- ~5–15 source documents per node → estimated 10K–50K `source_documents` rows for a class demo dataset.
- Embedding storage: 384-dim vectors × ~50K rows ≈ manageable within a single Postgres instance with pgvector; no need for a dedicated vector DB at this scale.
