"""SQLAlchemy ORM models — mirrors the schema in 05_Backend_Schema doc."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Enum, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.session import Base
from app.config import get_settings

settings = get_settings()
EMBED_DIM = settings.embedding_dim


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, unique=True, nullable=False)
    display_name = Column(Text)
    hashed_password = Column(Text, nullable=False)
    prior_knowledge_tags = Column(ARRAY(Text), default=list)
    created_at = Column(DateTime(timezone=True), default=_now)

    topics = relationship("Topic", back_populates="user")
    progress = relationship("UserProgress", back_populates="user")


# ---------------------------------------------------------------------------
# topics
# ---------------------------------------------------------------------------
class Topic(Base):
    __tablename__ = "topics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    raw_query = Column(Text, nullable=False)
    normalized_label = Column(Text, nullable=False)
    status = Column(
        Enum("processing", "ready", "failed", "needs_clarification", name="topic_status"),
        default="processing",
        nullable=False,
    )
    # Populated only while status == "needs_clarification"; the candidate senses
    # returned by nlp/intent.py's ambiguity check, for the frontend to render as a
    # picker. Cleared once POST /topics/{id}/clarify resolves the topic.
    clarification_options = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    user = relationship("User", back_populates="topics")
    graphs = relationship("Graph", back_populates="topic")
    trend_snapshots = relationship("TrendSnapshot", back_populates="topic")


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------
class Graph(Base):
    __tablename__ = "graphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id = Column(UUID(as_uuid=True), ForeignKey("topics.id"), nullable=False)
    version = Column(Integer, default=1)
    root_node_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    topic = relationship("Topic", back_populates="graphs")
    nodes = relationship("Node", back_populates="graph")
    edges = relationship("Edge", back_populates="graph")


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
class Node(Base):
    __tablename__ = "nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), ForeignKey("graphs.id"), nullable=False)
    label = Column(Text, nullable=False)
    aliases = Column(ARRAY(Text), default=list)
    description_short = Column(Text)
    embedding = Column(Vector(EMBED_DIM))
    category = Column(Text)
    importance_score = Column(Float, default=1.0)
    is_trending = Column(Boolean, default=False)
    depth_level = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=_now)

    graph = relationship("Graph", back_populates="nodes")
    summaries = relationship("Summary", back_populates="node")
    node_sources = relationship("NodeSource", back_populates="node")
    progress = relationship("UserProgress", back_populates="node")
    trend_snapshots = relationship("TrendSnapshot", back_populates="linked_node")


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------
class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (
        UniqueConstraint("graph_id", "source_node_id", "target_node_id", "relation_type"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), ForeignKey("graphs.id"), nullable=False)
    source_node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    relation_type = Column(
        Enum("prerequisite_of", "related_to", "subtopic_of", "enables", "breaks",
             name="relation_type"),
        nullable=False,
    )
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), default=_now)

    graph = relationship("Graph", back_populates="edges")


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
class Summary(Base):
    __tablename__ = "summaries"
    __table_args__ = (UniqueConstraint("node_id", "depth"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    depth = Column(
        Enum("2min", "10min", "deepdive", name="summary_depth"),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    citations = Column(JSONB, default=list)
    generated_at = Column(DateTime(timezone=True), default=_now)

    node = relationship("Node", back_populates="summaries")


# ---------------------------------------------------------------------------
# source_documents
# ---------------------------------------------------------------------------
class SourceDocument(Base):
    __tablename__ = "source_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_type = Column(
        Enum("wikipedia", "youtube", "reddit", "paper", "web", name="source_type"),
        nullable=False,
    )
    url = Column(Text, nullable=False)
    title = Column(Text)
    author_or_channel = Column(Text)
    published_at = Column(DateTime(timezone=True), nullable=True)
    raw_text = Column(Text)
    fetched_at = Column(DateTime(timezone=True), default=_now)
    # NOTE: 'metadata' is reserved on a declarative base (Base.metadata is the MetaData
    # object), so the attribute is named doc_metadata while the DB column stays "metadata".
    doc_metadata = Column("metadata", JSONB, default=dict)

    chunks = relationship("SourceChunk", back_populates="source_document")
    node_sources = relationship("NodeSource", back_populates="source_document")
    video_extraction = relationship("VideoExtraction", back_populates="source_document",
                                   uselist=False)


# ---------------------------------------------------------------------------
# source_chunks
# ---------------------------------------------------------------------------
class SourceChunk(Base):
    __tablename__ = "source_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("source_documents.id"),
                                nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(EMBED_DIM))

    source_document = relationship("SourceDocument", back_populates="chunks")


# ---------------------------------------------------------------------------
# node_sources (many-to-many)
# ---------------------------------------------------------------------------
class NodeSource(Base):
    __tablename__ = "node_sources"

    node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), primary_key=True)
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("source_documents.id"),
                                primary_key=True)
    relevance_score = Column(Float, default=1.0)

    node = relationship("Node", back_populates="node_sources")
    source_document = relationship("SourceDocument", back_populates="node_sources")


# ---------------------------------------------------------------------------
# video_extractions
# ---------------------------------------------------------------------------
class VideoExtraction(Base):
    __tablename__ = "video_extractions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("source_documents.id"),
                                nullable=False)
    notes_outline = Column(Text)
    key_concepts = Column(ARRAY(Text), default=list)
    quiz = Column(JSONB, default=list)
    generated_at = Column(DateTime(timezone=True), default=_now)

    source_document = relationship("SourceDocument", back_populates="video_extraction")


# ---------------------------------------------------------------------------
# trend_snapshots
# ---------------------------------------------------------------------------
class TrendSnapshot(Base):
    __tablename__ = "trend_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id = Column(UUID(as_uuid=True), ForeignKey("topics.id"), nullable=False)
    cluster_label = Column(Text, nullable=False)
    growth_rate = Column(Float)
    window_start = Column(Date)
    window_end = Column(Date)
    linked_node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=True)

    topic = relationship("Topic", back_populates="trend_snapshots")
    linked_node = relationship("Node", back_populates="trend_snapshots")


# ---------------------------------------------------------------------------
# user_progress
# ---------------------------------------------------------------------------
class UserProgress(Base):
    __tablename__ = "user_progress"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), primary_key=True)
    status = Column(
        Enum("not_started", "in_progress", "learned", name="progress_status"),
        default="not_started",
        nullable=False,
    )
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    user = relationship("User", back_populates="progress")
    node = relationship("Node", back_populates="progress")
