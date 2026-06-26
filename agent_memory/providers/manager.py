"""Provider initialization — created once at startup, not lazily.

Adding a provider is a new ``match`` arm. Non-default providers (OpenAI,
Anthropic) ship as opt-in SDK extras; if the SDK is missing, the provider raises
``ConfigError`` with the install hint instead of a deep ``ImportError``/
``TypeError``.
"""

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import EmbeddingProvider, LLMProvider

# Native output dimensions for known Voyage models, used to keep
# ``embedding_dimension`` (and thus the Atlas vector index numDimensions) in
# sync with the selected model. Both the public Voyage API and the MongoDB
# Atlas embeddings gateway serve these same models.
_VOYAGE_MODEL_DIMS = {
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "voyage-3-lite": 512,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-code-3": 1024,
    "voyage-2": 1024,
}

# Default embedding_dimension inherited from MCPConfig — lets us detect whether
# the operator explicitly pinned a dimension (never overridden) vs left default.
_DEFAULT_EMBEDDING_DIMENSION = 1536


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
                # Align embedding_dimension to the model's native dimension,
                # unless the operator explicitly pinned a non-default value.
                known_dim = _VOYAGE_MODEL_DIMS.get(config.voyage_model)
                if known_dim is not None and config.embedding_dimension == _DEFAULT_EMBEDDING_DIMENSION:
                    config.embedding_dimension = known_dim
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
