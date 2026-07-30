"""Tests for MemoryConfig (programmatic config + from_env). REQ-E-010..012."""

import os

import pytest

from agent_memory.config import MemoryConfig


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        upper = key.upper()
        if (
            upper.startswith("MONGODB_")
            or upper.startswith("AWS_")
            or upper.startswith("OPENAI_")
            or upper.startswith("ANTHROPIC_")
            or upper in ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "WORKERS_IN_PROCESS")
        ):
            monkeypatch.delenv(key, raising=False)


def _make_config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


class TestMemoryConfigNewFields:
    """TC-CFG-001: new provider + worker fields exist with documented defaults."""

    def test_workers_in_process_defaults_true(self):
        assert _make_config().workers_in_process is True

    def test_openai_fields_present(self):
        cfg = _make_config(
            openai_api_key="sk-test",
            openai_base_url="https://grove.example/v1",
            openai_model="gpt-4o",
            openai_embedding_model="text-embedding-3-small",
        )
        assert cfg.openai_api_key == "sk-test"
        assert cfg.openai_base_url == "https://grove.example/v1"
        assert cfg.openai_model == "gpt-4o"
        assert cfg.openai_embedding_model == "text-embedding-3-small"

    def test_anthropic_fields_present(self):
        cfg = _make_config(
            anthropic_api_key="ak-test",
            anthropic_base_url="https://grove.example/anthropic",
            anthropic_model="claude-sonnet-4-6",
        )
        assert cfg.anthropic_api_key == "ak-test"
        assert cfg.anthropic_base_url == "https://grove.example/anthropic"
        assert cfg.anthropic_model == "claude-sonnet-4-6"

    def test_inherits_substrate_fields(self):
        cfg = _make_config()
        # backward-compatible with memory-mcp config surface
        assert cfg.embedding_dimension == 1536
        assert cfg.stm_ttl_hours == 24


class TestMemoryConfigFromEnv:
    """TC-CFG-002 / TC-CFG-003: from_env reads env and defaults to bedrock."""

    def test_from_env_reads_connection_string(self, monkeypatch):
        monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://envhost:27017")
        cfg = MemoryConfig.from_env(_env_file=None)
        assert cfg.mongodb_connection_string == "mongodb://envhost:27017"

    def test_from_env_defaults_providers_to_bedrock(self, monkeypatch):
        monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://envhost:27017")
        cfg = MemoryConfig.from_env(_env_file=None)
        assert cfg.llm_provider == "bedrock"
        assert cfg.embedding_provider == "bedrock"

    def test_from_env_reads_new_provider_vars(self, monkeypatch):
        monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://envhost:27017")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("WORKERS_IN_PROCESS", "false")
        cfg = MemoryConfig.from_env(_env_file=None)
        assert cfg.openai_api_key == "sk-env"
        assert cfg.workers_in_process is False
