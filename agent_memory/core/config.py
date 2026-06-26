"""Centralized configuration via Pydantic BaseSettings."""

from pydantic_settings import BaseSettings


class MCPConfig(BaseSettings):
    """Memory-MCP configuration.

    All values can be overridden via environment variables (case-insensitive).
    ``mongodb_connection_string`` is the only required field.
    """

    # Server
    app_name: str = "memory-mcp"
    app_version: str = "3.2.0"
    port: int = 8000
    transport: str = "streamable-http"
    debug: bool = False

    # MongoDB
    mongodb_connection_string: str
    mongodb_database_name: str = "agent_memory"
    mongodb_max_pool_size: int = 20
    mongodb_min_pool_size: int = 2

    # Embedding Provider
    embedding_provider: str = "bedrock"
    embedding_model: str = "amazon.titan-embed-text-v1"
    embedding_dimension: int = 1536

    # LLM Provider
    llm_provider: str = "bedrock"
    llm_model: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    # AWS (Bedrock)
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # Voyage AI
    # voyage_base_url selects the endpoint: the public Voyage API (default) or
    # the MongoDB Atlas embeddings gateway (https://ai.mongodb.com/v1/embeddings),
    # both via the VOYAGE_BASE_URL env var. The payload format is identical.
    voyage_api_key: str | None = None
    voyage_base_url: str = "https://api.voyageai.com/v1/embeddings"
    voyage_model: str = "voyage-3"

    # Memory Lifecycle
    stm_ttl_hours: int = 24
    ltm_retention_critical_days: int = 365
    ltm_retention_reference_days: int = 180
    ltm_retention_standard_days: int = 90
    ltm_retention_temporary_days: int = 7

    # Memory Evolution Thresholds
    reinforce_threshold: float = 0.85
    merge_threshold: float = 0.70

    # Retrieval Ranking Weights
    ranking_alpha: float = 0.2
    ranking_beta: float = 0.3
    ranking_gamma: float = 0.5

    # RRF Parameters
    rrf_k: int = 60
    rrf_vector_weight: float = 1.0
    rrf_text_weight: float = 0.7

    # Query Limits
    max_results_per_query: int = 100
    max_response_bytes: int = 16_777_216

    # Cache
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.95

    # Consolidation (Phase 1)
    consolidation_interval_hours: int = 24
    stm_compression_age_hours: int = 24
    forgetting_score_threshold: float = 0.1
    promotion_importance_threshold: float = 0.6
    promotion_access_threshold: int = 2
    promotion_age_minutes: int = 60

    # Enrichment
    enrichment_interval_seconds: int = 30
    enrichment_batch_size: int = 50
    enrichment_concurrency: int = 5
    enrichment_max_retries: int = 3

    # Audit
    audit_buffer_size: int = 10
    audit_flush_interval_seconds: int = 60
    audit_flush_on_write: bool = False
    audit_retention_days: int = 365

    # Soft Delete
    soft_delete_purge_days: int = 30

    # Identity & Auth (Phase 2)
    auth_enabled: bool = False
    auth_token_header: str = "Authorization"
    auth_secret: str = ""
    auth_token_expiry_seconds: int = 86400
    auth_user_id_claim: str = "sub"
    auth_role_claim: str = "role"
    auth_default_role: str = "end_user"

    # Governance (Phase 2)
    governance_enabled: bool = False
    governance_default_profile: str = "default"
    governance_cache_ttl_seconds: int = 300
    rate_limit_enabled: bool = False
    rate_limit_window_seconds: int = 60
    rate_limit_max_requests: int = 100

    # Prompt Library (Phase 2)
    prompt_experiment_enabled: bool = True
    prompt_cache_ttl_seconds: int = 300

    # Auto-Capture (Phase 2)
    auto_capture_enabled: bool = True
    auto_capture_tools: list[str] = [
        "recall_memory", "hybrid_search",
        "store_decision", "recall_decision",
    ]
    auto_capture_min_length: int = 30
    auto_capture_max_content_length: int = 2000

    # Decision Stickiness (Phase 2)
    decision_stickiness_enabled: bool = False
    decision_default_ttl_days: int = 90

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }
