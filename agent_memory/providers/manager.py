"""Provider initialization — created once at startup, not lazily."""

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
            case _:
                raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")

    def _create_llm_provider(self, config: MCPConfig) -> LLMProvider:
        match config.llm_provider:
            case "bedrock":
                from agent_memory.providers.bedrock import BedrockLLMProvider
                return BedrockLLMProvider(config)
            case _:
                raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
