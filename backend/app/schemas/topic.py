from __future__ import annotations
from uuid import UUID
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class TopicCreate(BaseModel):
    raw_query: str = Field(..., min_length=2, max_length=500)
    prior_knowledge: list[str] = Field(default_factory=list)


class ClarifyChoice(BaseModel):
    """Body for POST /topics/{id}/clarify -- resolves an ambiguous topic to one sense."""
    chosen_query: str = Field(..., min_length=2, max_length=500)
    # Not persisted on Topic (see tasks.py's build_topic_graph docstring), so the
    # client must resend whatever prior_knowledge it originally submitted.
    prior_knowledge: list[str] = Field(default_factory=list)


class ClarificationOption(BaseModel):
    label: str
    clarifying_query: str
    hint: Optional[str] = None


class TopicOut(BaseModel):
    id: UUID
    raw_query: str
    normalized_label: str
    status: Literal["processing", "ready", "failed", "needs_clarification"]
    clarification_options: Optional[list[ClarificationOption]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class NodeOut(BaseModel):
    id: UUID
    label: str
    aliases: list[str]
    description_short: Optional[str]
    category: Optional[str]
    importance_score: float
    is_trending: bool
    depth_level: int

    model_config = {"from_attributes": True}


class EdgeOut(BaseModel):
    id: UUID
    source_node_id: UUID
    target_node_id: UUID
    relation_type: str
    confidence: float

    model_config = {"from_attributes": True}


class GraphOut(BaseModel):
    graph_id: UUID
    topic_id: UUID
    version: int
    nodes: list[NodeOut]
    edges: list[EdgeOut]
    learning_path: list[str]   # ordered node IDs (as strings) from topological sort


class SummaryOut(BaseModel):
    node_id: UUID
    depth: Literal["2min", "10min", "deepdive"]
    content: str
    citations: list[dict]
    generated_at: datetime

    model_config = {"from_attributes": True}


class ProgressUpsert(BaseModel):
    status: Literal["not_started", "in_progress", "learned"]


class ProgressOut(BaseModel):
    user_id: UUID
    node_id: UUID
    status: str
    updated_at: datetime

    model_config = {"from_attributes": True}


class SearchResult(BaseModel):
    node_id: UUID
    label: str
    score: float


class TrendOut(BaseModel):
    id: UUID
    cluster_label: str
    growth_rate: Optional[float]
    window_start: Optional[str]
    window_end: Optional[str]
    linked_node_id: Optional[UUID]

    model_config = {"from_attributes": True}
