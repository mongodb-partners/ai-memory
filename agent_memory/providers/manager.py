"""Provider initialization — created once at startup, not lazily.

Adding a provider is a new ``match`` arm. Non-default providers (OpenAI,
Anthropic) ship as opt-in SDK extras; if the SDK is missing, the provider raises
``ConfigError`` with the install hint instead of a deep ``ImportError``/
``TypeError``.
"""

import logging
from dataclasses import dataclass

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

# `_DEFAULT_EMBEDDING_DIMENSION = 1536` used to live here, as the way to tell a
# pinned dimension from an untouched one: anything != 1536 was assumed deliberate.
# That could not distinguish "left alone" from "set to 1536 on purpose", so an
# operator who pinned the default on a Voyage deployment had it silently rewritten
# to 1024 — and they are the operator most likely to already have a 1536-dim index
# with vectors in it. `resolve_embedding` reads `model_fields_set` instead, which
# records what was actually supplied.

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


@dataclass(frozen=True)
class ResolvedEmbedding:
    """The embedding model and dimension actually in force.

    Two of the config's declared values are *derived* rather than authoritative:
    on a Voyage or OpenAI deployment the canonical ``embedding_model`` comes from
    the provider-specific field, and Voyage's native dimension supersedes the
    inherited Titan default. That resolution used to be performed by writing back
    onto the config object inside ``_create_embedding_provider``, which made the
    correctness of migrations, the dimension guard, and artifact selection depend
    on all three running *after* the factory — an ordering nothing could enforce
    and that a reader had no way to see.

    Returning it instead makes the derived values a value: computable without
    constructing a provider, safe to compute twice, and impossible to read too
    early, because there is nothing to read until it has been computed.
    """

    model: str
    dimension: int


def resolve_embedding(config: MCPConfig) -> ResolvedEmbedding:
    """The canonical model name and vector dimension for this config.

    Pure: touches no network and mutates nothing, so the dimension guard can use
    it when the embedder is unreachable and a caller can ask twice without the
    second answer differing from the first.

    Voyage is the case that needs the dimension rule. The MongoDB-issued key
    targets a gateway whose models emit 1024 while ``embedding_dimension``
    inherits Titan's 1536 default, so a correct Voyage setup would be rejected by
    the startup guard — or worse, provision a 1536-dim index and write 1024-dim
    vectors that ``$vectorSearch`` silently never returns.

    An operator who *pinned* a dimension keeps it. Pinning is read from
    ``model_fields_set`` — pydantic's record of which fields the caller or the
    environment actually supplied — rather than by comparing against the default.
    The comparison version could not tell "left alone" from "deliberately set to
    1536", so an operator who pinned the default value on a Voyage deployment had
    it silently overwritten with 1024: exactly the operator most likely to have
    an existing 1536-dim index and existing vectors in it.
    """
    match config.embedding_provider:
        case "voyage":
            known = _VOYAGE_MODEL_DIMS.get(config.voyage_model)
            pinned = "embedding_dimension" in config.model_fields_set
            dimension = (
                config.embedding_dimension
                if known is None or pinned
                else known
            )
            return ResolvedEmbedding(model=config.voyage_model, dimension=dimension)
        case "openai":
            # `getattr`, because `openai_embedding_model` is declared on
            # `MemoryConfig` and not on the `MCPConfig` base this function accepts.
            # A bare `MCPConfig` naming the OpenAI provider cannot construct one
            # anyway — `OpenAIEmbeddingProvider` reads the same field — but a pure
            # resolver should answer rather than raise, so it falls back to the
            # generic name.
            return ResolvedEmbedding(
                model=getattr(
                    config, "openai_embedding_model", config.embedding_model
                ),
                dimension=config.embedding_dimension,
            )
        case _:
            return ResolvedEmbedding(
                model=config.embedding_model,
                dimension=config.embedding_dimension,
            )


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


# Embedding-head artifacts, keyed by the (provider, model, dimension) triple they
# were trained on. Keyed on the triple rather than the provider because dimension
# is the part that silently breaks: voyage-4 is 1024 and voyage-3-lite is 512, and
# loading 1024 coefficients against a 512-vector is the mismatch `LocalScorer`
# exists to refuse. Better to refuse here, by name, than per-call.
#
# Deliberately empty: we ship no trained embedding head. Two placeholders used to
# live here, and both scored every memory an identical 0.5 — which reads as
# working (no error, a plausible number) while disabling promotion entirely, since
# consolidation promotes at >= 0.6. Measurement said training them was not worth
# it either: held-out Spearman for an embedding head tops out around 0.45, the
# in-sample ceiling is 0.70, and `assess_importance` emits only 9 distinct values
# for a 1024-coefficient fit to aim at. So every deployment gets `lexical`, which
# is trained, until a head beats it on held-out calibration.
#
# To add one: train it against your embedder (`scripts/train_importance.py
# --space embedding --out agent_memory/data/importance/<name>.json`), then add its
# triple here.
_BUNDLED_ARTIFACTS: dict[tuple[str, str, int], str] = {}

# Provider-independent fallback, and currently the only artifact we ship. Weaker
# than a *trained* embedding head would be, and much better than the constant an
# untrained one returns.
_FALLBACK_ARTIFACT = "lexical"


def select_artifact_name(config: MCPConfig) -> str:
    """The bundled artifact matching this config's embedder, or the fallback.

    Returns ``"lexical"`` for every config today, because `_BUNDLED_ARTIFACTS` is
    empty — see the note there. That is by design, not a lookup miss: a lexical
    model trained on real labels beats an untrained embedding head, whatever the
    embedder.

    Reads the *resolved* model and dimension rather than the config's declared
    ones. It used to read the declared fields and therefore had to run after
    ``_create_embedding_provider`` had rewritten them — an ordering constraint on
    a lookup, enforced by a comment. Resolving here removes the constraint: this
    is now correct whenever it is called.
    """
    resolved = resolve_embedding(config)
    key = (config.embedding_provider, resolved.model, resolved.dimension)
    return _BUNDLED_ARTIFACTS.get(key, _FALLBACK_ARTIFACT)


class ProviderManager:
    """Initialized once at startup. No lazy initialization."""

    def __init__(self, config: MCPConfig) -> None:
        # The derived model/dimension, computed once and published. Callers that
        # need the dimension actually in force — the startup guard, index
        # provisioning — read this rather than `config.embedding_dimension`, which
        # is the *declared* value and is Titan's default on a Voyage deployment.
        self.embedding_spec: ResolvedEmbedding = resolve_embedding(config)
        self.embedding: EmbeddingProvider = self._create_embedding_provider(config)
        self.llm: LLMProvider = self._create_llm_provider(config)
        # Construction order no longer carries meaning. It used to: the Voyage arm
        # of `_create_embedding_provider` rewrote two config fields, so a scorer
        # built before it read Titan's defaults and silently downgraded a Voyage
        # deployment to lexical scoring. Now every one of these reads a resolved
        # value, so any order gives the same answer.
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
                # No write-back onto `config`. The canonical model name and the
                # native dimension are derived values, and they are derived by
                # `resolve_embedding`; the provider reads `voyage_model` directly.
                return VoyageEmbeddingProvider(config)
            case "openai":
                from agent_memory.providers.openai import OpenAIEmbeddingProvider
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
