from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://rabbit:rabbit_secret@localhost:5432/rabbit_hole"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # LLM -- free-tier provider chain (tried in this order; each skipped if its key
    # is empty). See app/nlp/llm.py. Anthropic has no free tier, so it is optional
    # and normally placed last. Each *_api_key accepts one key or several
    # comma-separated keys ("key1,key2") -- multiple free-tier accounts for that
    # provider, each with its own rate-limit bucket.
    gemini_api_key: str = ""
    gemini_model: str = ""
    groq_api_key: str = ""
    groq_model: str = ""
    cerebras_api_key: str = ""
    cerebras_model: str = ""
    mistral_api_key: str = ""
    mistral_model: str = ""
    # Cerebras needs a billed account and Mistral has no key registered, so
    # neither is in the default chain right now -- both remain valid provider
    # names in _OPENAI_COMPAT_PROVIDERS and can be re-added here once usable.
    llm_provider_chain: str = "gemini,groq"
    anthropic_model: str = ""

    # "curriculum" (default): no relation call -- prerequisite_of edges come from the
    #   prerequisites concept extraction already returns (one LLM call per build total).
    #   Falls back to "batched" when extraction produced no prerequisites (spaCy
    #   fallback, or an older cached result).
    # "batched": one extra LLM call classifies all typed relations for a graph build.
    # "pairwise": the legacy O(n^2) one-call-per-pair behaviour, kept only for the
    #   ablation study (see backend/eval/).
    relation_mode: str = "curriculum"

    # Number of distinct providers to consult concurrently for relation classification
    # (see extraction.py::classify_relations_ensemble). >1 requires that many providers
    # actually configured with keys; otherwise it silently degrades to a single call.
    relation_ensemble_size: int = 2

    anthropic_api_key: str = ""

    # Source APIs
    youtube_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "RabbitHoleExplorer/1.0"
    semantic_scholar_api_key: str = ""
    # Web search: SerpAPI tried first, then Serper, each skipped if empty. Like the
    # LLM *_api_key settings above, each accepts one key or several comma-separated
    # keys -- multiple free-tier accounts, tried in order so one account's exhausted
    # quota falls through to the next instead of failing the search.
    serpapi_key: str = ""
    serper_api_key: str = ""

    # App
    secret_key: str = "change-me-in-production"
    environment: str = "development"
    log_level: str = "INFO"
    log_file: str = ""   # optional rotating log file path (stdout is always on); see logging_utils.py

    # NLP settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    node_dedup_threshold: float = 0.92   # cosine similarity above which nodes are merged
    relation_confidence_threshold: float = 0.5
    # Concepts whose "label: description" embedding is less similar than this to the
    # topic query are dropped before the graph is built (never below
    # graph_builder.MIN_GRAPH_NODES). Deliberately conservative: calibrated on stored
    # graphs (2026-10-03), on-topic concepts scored as low as ~0.34 while clear
    # tangents scored ~0.08-0.16, and the two ranges overlap above ~0.3, so this
    # only removes clear outliers. It is a safety net, not the main relevance fix.
    concept_relevance_threshold: float = 0.20
    max_expansion_depth: int = 3
    initial_graph_node_cap: int = 12
    # Total nodes a single graph may ever reach via expand_node -- max_expansion_depth
    # alone bounds recursion depth, not total size (each expansion can add up to 6
    # nodes per click, unbounded across clicks).
    max_graph_nodes: int = 60

    # Cache TTLs (seconds)
    ttl_source_cache: int = 7 * 24 * 3600      # 7 days
    ttl_llm_cache: int = 30 * 24 * 3600        # 30 days

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
