"""Provider initialization — created once at startup, not lazily.

Adding a provider is a new ``match`` arm. Non-default providers (OpenAI,
Anthropic) ship as opt-in SDK extras; if the SDK is missing, the provider raises
``ConfigError`` with the install hint instead of a deep ``ImportError``/
``TypeError``.
"""

import logging

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import EmbeddingProvider, LLMProvider
from agent_memory.services.importance import (
    ImportanceScorer,
    LLMScorer,
    LocalScorer,
    bundled_artifact_path,
    load_artifact,
)

logger = logging.getLogger(__name__)

# Native output dimensions for known Voyage models, used to keep
# ``embedding_dimension`` (and thus the Atlas vector index numDimensions) in
# sync with the selected model. Both the public Voyage API and the MongoDB
# Atlas embeddings gateway serve these same models.
_VOYAGE_MODEL_DIMS = {
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "voyage-4-lite": 1024,
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

# Native output dimensions for the other providers' embedding models. Same purpose
# as `_VOYAGE_MODEL_DIMS`, but these are *not* auto-applied to config — Voyage is
# special-cased because the MongoDB-issued key targets a gateway whose voyage-3
# emits 1024 rather than the 1536 default, so leaving that unaligned rejects a
# correct setup. These are used only by the startup guard, to check a declared
# dimension against a known model without needing the network.
_BEDROCK_MODEL_DIMS = {
    "amazon.titan-embed-text-v1": 1536,
    "amazon.titan-embed-text-v2:0": 1024,
    "amazon.titan-embed-image-v1": 1024,
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
}

_OPENAI_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def known_embedding_dimension(config: MCPConfig) -> int | None:
    """The configured model's documented output size, or None if unknown.

    Offline counterpart to probing the embedder. The startup guard needs an answer
    even when the embedder is unreachable, because the failure it exists to prevent
    — writing 1024-dim vectors into a 1536-dim index — is silent and corrupts the
    collection rather than erroring. An unknown model returns None and the guard
    falls back to the probe.
    """
    match config.embedding_provider:
        case "voyage":
            return _VOYAGE_MODEL_DIMS.get(config.voyage_model)
        case "bedrock":
            return _BEDROCK_MODEL_DIMS.get(config.embedding_model)
        case "openai":
            return _OPENAI_MODEL_DIMS.get(config.openai_embedding_model)
        case _:
            return None


# Bundled artifacts, keyed by the (provider, model, dimension) triple they were
# trained on. Keyed on the triple rather than the provider because dimension is
# the part that silently breaks: voyage-3 is 1024 and voyage-3-lite is 512, and
# loading 1024 coefficients against a 512-vector is the mismatch `LocalScorer`
# exists to refuse. Better to refuse here, by name, than per-call.
_BUNDLED_ARTIFACTS = {
    ("bedrock", "amazon.titan-embed-text-v1", 1536): "titan-1536",
    ("voyage", "voyage-3", 1024): "voyage-3-1024",
}

# Provider-independent fallback. Weaker than a trained embedding head, and much
# better than returning a constant for every memory.
_FALLBACK_ARTIFACT = "lexical"


def select_artifact_name(config: MCPConfig) -> str:
    """The bundled artifact matching this config's embedder, or the fallback.

    Must be called *after* the embedding provider is constructed: the Voyage arm
    of ``_create_embedding_provider`` rewrites ``embedding_model`` and
    ``embedding_dimension`` on the config object, and reading them before that
    yields Titan's defaults on a Voyage deployment.
    """
    key = (
        config.embedding_provider,
        config.embedding_model,
        config.embedding_dimension,
    )
    return _BUNDLED_ARTIFACTS.get(key, _FALLBACK_ARTIFACT)


class ProviderManager:
    """Initialized once at startup. No lazy initialization."""

    def __init__(self, config: MCPConfig) -> None:
        self.embedding: EmbeddingProvider = self._create_embedding_provider(config)
        self.llm: LLMProvider = self._create_llm_provider(config)
        # Last, and not by accident. The Voyage arm of
        # `_create_embedding_provider` mutates `config.embedding_model` and
        # `config.embedding_dimension`; selecting an artifact before that runs
        # reads Titan's defaults, matches nothing, and quietly downgrades a Voyage
        # deployment to lexical scoring. See
        # `test_scorer_built_after_embedding_provider`.
        self.scorer: ImportanceScorer = self._create_scorer(config)

    def _create_scorer(self, config: MCPConfig) -> ImportanceScorer:
        """Build the importance scorer named by config.

        ``config.importance_scorer`` is already validated and normalized by
        ``MCPConfig._importance_scorer_must_be_known``, so the ``case _`` arm is
        unreachable in practice — kept because a future field change should fail
        loudly here rather than return None.

        The prompt getter is deliberately absent: ``ProviderManager`` has no prompt
        library. ``AsyncMemory._maybe_start_workers`` attaches the worker's
        ``_get_prompt`` when it injects the scorer, which is the only place where
        both exist.
        """
        match config.importance_scorer:
            case "llm":
                return LLMScorer(self.llm)
            case "local":
                if config.importance_model_path:
                    # An operator who names a file gets that file or a startup
                    # failure. Falling back to a bundled artifact would load
                    # different coefficients than they asked for and report
                    # healthy.
                    artifact = load_artifact(config.importance_model_path)
                else:
                    name = select_artifact_name(config)
                    artifact = load_artifact(bundled_artifact_path(name))
                    logger.info(
                        "Local importance scoring enabled: artifact %r "
                        "(kind=%s, trained on %s samples)",
                        name,
                        artifact.kind,
                        (artifact.training or {}).get("n_samples", "unknown"),
                    )
                return LocalScorer(artifact)
            case _:
                raise ValueError(
                    f"Unknown importance scorer: {config.importance_scorer}"
                )

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
