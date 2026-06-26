"""ProviderManager arms for openai/anthropic + missing-extra ConfigError.

REQ-E-053, REQ-E-054 (premortem #5, boundary #5).
"""

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.exceptions import ConfigError
from agent_memory.providers.manager import ProviderManager


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


class TestNewArms:
    """TC-PROV-004: manager builds openai/anthropic providers."""

    def test_openai_llm_and_embedding(self):
        from agent_memory.providers.openai import (
            OpenAIEmbeddingProvider,
            OpenAILLMProvider,
        )

        cfg = _config(llm_provider="openai", embedding_provider="openai", openai_api_key="sk")
        mgr = ProviderManager(cfg)
        assert isinstance(mgr.llm, OpenAILLMProvider)
        assert isinstance(mgr.embedding, OpenAIEmbeddingProvider)

    def test_anthropic_llm_with_openai_embeddings(self):
        from agent_memory.providers.anthropic import AnthropicLLMProvider
        from agent_memory.providers.openai import OpenAIEmbeddingProvider

        cfg = _config(
            llm_provider="anthropic", embedding_provider="openai",
            anthropic_api_key="ak", openai_api_key="sk",
        )
        mgr = ProviderManager(cfg)
        assert isinstance(mgr.llm, AnthropicLLMProvider)
        assert isinstance(mgr.embedding, OpenAIEmbeddingProvider)


class TestMissingExtra:
    """TC-PROV-005: selecting a provider whose SDK is absent → ConfigError + hint.

    Absence is simulated by setting the module-level SDK symbol to None, which is
    exactly the state the ``try/except ImportError`` import block leaves when the
    extra is not installed.
    """

    def test_openai_missing_sdk_raises_config_error(self, monkeypatch):
        import agent_memory.providers.openai as mod
        monkeypatch.setattr(mod, "AsyncOpenAI", None)
        cfg = _config(embedding_provider="openai", openai_api_key="sk")
        with pytest.raises(ConfigError, match=r"agent-memory\[openai\]"):
            ProviderManager(cfg)

    def test_anthropic_missing_sdk_raises_config_error(self, monkeypatch):
        import agent_memory.providers.anthropic as mod
        monkeypatch.setattr(mod, "AsyncAnthropic", None)
        cfg = _config(
            llm_provider="anthropic", embedding_provider="openai",
            anthropic_api_key="ak", openai_api_key="sk",
        )
        with pytest.raises(ConfigError, match=r"agent-memory\[anthropic\]"):
            ProviderManager(cfg)
