"""Tests for MCPConfig (Pydantic Settings)."""

import os
import pytest
from agent_memory.core.config import MCPConfig


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove env vars that could leak from .env files into tests."""
    for key in list(os.environ):
        upper = key.upper()
        if upper.startswith("MONGODB_") or upper.startswith("AWS_") or upper in (
            "PORT", "DEBUG", "EMBEDDING_DIMENSION", "ENRICHMENT_CONCURRENCY",
            "ENRICHMENT_BATCH_SIZE", "LLM_MODEL_ID",
            "EMBEDDING_MODEL_ID", "LOGGER_SERVICE_URL", "AI_MEMORY_SERVICE_URL",
            "SEMANTIC_CACHE_SERVICE_URL", "VECTOR_DIMENSION",
        ):
            monkeypatch.delenv(key, raising=False)


def _make_config(**overrides) -> MCPConfig:
    """Create config without loading .env files."""
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


class TestMCPConfigDefaults:
    """TC-001: MCPConfig loads with correct default values."""

    def test_server_defaults(self):
        config = _make_config()
        assert config.app_name == "memory-mcp"
        assert config.app_version == "3.2.0"
        assert config.port == 8000
        assert config.transport == "streamable-http"
        assert config.debug is False

    def test_transport_override(self):
        config = _make_config(transport="stdio")
        assert config.transport == "stdio"

    def test_mongodb_defaults(self):
        config = _make_config()
        assert config.mongodb_database_name == "agent_memory"
        assert config.mongodb_max_pool_size == 20
        assert config.mongodb_min_pool_size == 2

    def test_embedding_defaults(self):
        config = _make_config()
        assert config.embedding_provider == "bedrock"
        assert config.embedding_model == "amazon.titan-embed-text-v1"
        assert config.embedding_dimension == 1536

    def test_llm_defaults(self):
        config = _make_config()
        assert config.llm_provider == "bedrock"

    def test_memory_lifecycle_defaults(self):
        config = _make_config()
        assert config.stm_ttl_hours == 24
        assert config.ltm_retention_critical_days == 365
        assert config.ltm_retention_reference_days == 180
        assert config.ltm_retention_standard_days == 90
        assert config.ltm_retention_temporary_days == 7

    def test_evolution_threshold_defaults(self):
        config = _make_config()
        assert config.reinforce_threshold == 0.85
        assert config.merge_threshold == 0.70

    def test_ranking_weight_defaults(self):
        config = _make_config()
        assert config.ranking_alpha == 0.2
        assert config.ranking_beta == 0.3
        assert config.ranking_gamma == 0.5

    def test_rrf_defaults(self):
        config = _make_config()
        assert config.rrf_k == 60
        assert config.rrf_vector_weight == 1.0
        assert config.rrf_text_weight == 0.7
        assert not hasattr(config, "rrf_single_pipeline")

    def test_query_limit_defaults(self):
        config = _make_config()
        assert config.max_results_per_query == 100
        assert config.max_response_bytes == 16_777_216

    def test_cache_defaults(self):
        config = _make_config()
        assert config.cache_ttl_seconds == 3600
        assert config.cache_similarity_threshold == 0.95

    def test_enrichment_defaults(self):
        config = _make_config()
        assert config.enrichment_interval_seconds == 30
        assert config.enrichment_batch_size == 50
        assert config.enrichment_concurrency == 5
        assert config.enrichment_max_retries == 3

    def test_audit_defaults(self):
        config = _make_config()
        assert config.audit_buffer_size == 10
        assert config.audit_flush_interval_seconds == 60
        assert config.audit_flush_on_write is False
        assert config.audit_retention_days == 365

    def test_soft_delete_defaults(self):
        config = _make_config()
        assert config.soft_delete_purge_days == 30

    def test_feature_flags_removed(self):
        config = _make_config()
        assert not hasattr(config, "use_new_memory_service")
        assert not hasattr(config, "use_new_cache_service")
        assert not hasattr(config, "use_new_search_service")

    def test_auth_defaults_disabled(self):
        config = _make_config()
        assert config.auth_enabled is False
        assert config.auth_secret == ""
        assert config.auth_token_expiry_seconds == 86400
        assert not hasattr(config, "auth_jwks_url")
        assert config.governance_enabled is False
        assert config.rate_limit_enabled is False


class TestMCPConfigEnvOverride:
    """TC-002: MCPConfig values can be overridden from environment."""

    def test_override_port(self):
        config = _make_config(port=9090)
        assert config.port == 9090

    def test_override_mongodb_connection(self):
        config = _make_config(mongodb_connection_string="mongodb://custom:27017/mydb")
        assert config.mongodb_connection_string == "mongodb://custom:27017/mydb"

    def test_override_embedding_dimension(self):
        config = _make_config(embedding_dimension=768)
        assert config.embedding_dimension == 768

    def test_override_enrichment_settings(self):
        config = _make_config(enrichment_concurrency=10, enrichment_batch_size=100)
        assert config.enrichment_concurrency == 10
        assert config.enrichment_batch_size == 100


class TestMCPConfigValidation:
    """TC-003: MCPConfig validates required fields."""

    def test_mongodb_connection_string_required(self):
        with pytest.raises(Exception):
            MCPConfig(_env_file=None)

    def test_aws_keys_optional(self):
        config = _make_config()
        assert config.aws_access_key_id is None
        assert config.aws_secret_access_key is None

class TestMCPConfigAutoCapture:
    """TC-E: Auto-capture config defaults and overrides."""

    def test_auto_capture_defaults(self):
        config = _make_config()
        assert config.auto_capture_enabled is True
        assert "recall_memory" in config.auto_capture_tools
        assert "hybrid_search" in config.auto_capture_tools
        assert "store_decision" in config.auto_capture_tools
        assert "recall_decision" in config.auto_capture_tools
        assert config.auto_capture_min_length == 30
        assert config.auto_capture_max_content_length == 2000

    def test_auto_capture_disabled(self):
        config = _make_config(auto_capture_enabled=False)
        assert config.auto_capture_enabled is False

    def test_auto_capture_custom_tools(self):
        config = _make_config(auto_capture_tools=["recall_memory"])
        assert config.auto_capture_tools == ["recall_memory"]

    def test_prompt_experiment_enabled_default_true(self):
        """REQ-E-024: prompt_experiment_enabled default changed to true."""
        config = _make_config()
        assert config.prompt_experiment_enabled is True
