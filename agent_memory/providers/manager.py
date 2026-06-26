"""Provider initialization — created once at startup, not lazily.

Adding a provider is a new ``match`` arm. Non-default providers (OpenAI,
Anthropic) ship as opt-in SDK extras; if the SDK is missing, the provider raises
``ConfigError`` with the install hint instead of a deep ``ImportError``/
``TypeError``.
"""

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import EmbeddingProvider, LLMProvider


class ProviderManager:
    """Initialized once at startup. No lazy initialization."""

    def __init__(self, config: MCPConfig) -> None:
        self.embedding: EmbeddingProvider = self._create_embedding_provider(config)
        self.llm: LLMProvider = self._create_llm_provider(config)

    def _create_embedding_provider(self, config: MCPConfig) -> EmbeddingProvider:
        match config.embedding_provider:
            case "bedrock":
                from agent_memory.providers.bedrock import BedrockEmbeddingProvider
                return BedrockEmbeddingProvider(config)
            case "voyage":
                from agent_memory.providers.voyage import VoyageEmbeddingProvider
                # Sync the canonical embedding_model from the Voyage-specific
                # config so documents record the correct model name.
                config.embedding_model = config.voyage_model
                return VoyageEmbeddingProvider(config)
            case "openai":
                from agent_memory.providers.openai import OpenAIEmbeddingProvider
                config.embedding_model = config.openai_embedding_model
                return OpenAIEmbeddingProvider(config)
            case _:
                raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")

    def _create_llm_provider(self, config: MCPConfig) -> LLMProvider:
        match config.llm_provider:
            case "bedrock":
                from agent_memory.providers.bedrock import BedrockLLMProvider
                return BedrockLLMProvider(config)
            case "openai":
                from agent_memory.providers.openai import OpenAILLMProvider
                return OpenAILLMProvider(config)
            case "anthropic":
                from agent_memory.providers.anthropic import AnthropicLLMProvider
                return AnthropicLLMProvider(config)
            case _:
                raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
