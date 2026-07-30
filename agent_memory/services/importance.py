"""Pluggable importance scoring — LLM or a local pre-trained linear model.

``importance`` decides whether a memory is forgotten
(``ConsolidationWorker._forget_low_importance``), promoted (``_promote_to_ltm``),
and how it ranks (``MemoryService._calibrated_rank``). It is produced by exactly
one call today: ``providers.llm.assess_importance`` — one LLM round trip per
long-term memory.

This module makes that call swappable. The local path is viable only because the
embedding already exists before scoring happens — ``EnrichmentWorker`` holds
``memory["embedding"]`` and passes it to ``evolve_memory`` — so a linear head on
it is a dot product rather than an encoder forward pass. Pure Python, no numpy:
the library ships 8 runtime dependencies and a scientific stack is not going to
be the ninth.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from agent_memory.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Bumped only for a breaking artifact-format change. An unknown version must
# refuse to load rather than guess: misreading coefficients produces plausible
# numbers, and a plausible wrong importance is invisible until memories start
# disappearing.
SCHEMA_VERSION = 1

_KINDS = ("embedding_linear", "lexical")
_SQUASHES = ("logistic",)

# Keep in lockstep with `lexical_features`. Asserted against every lexical
# artifact at load time, so a feature added without retraining fails loudly
# instead of silently shifting every weight by one position.
LEXICAL_FEATURE_COUNT = 7

# The floor is 0.1, never 0.0. See `providers/base.py:44-51`: 0.0 is at
# `forgetting_score_threshold`, so emitting it is ordering a deletion.
MIN_IMPORTANCE = 0.1
MAX_IMPORTANCE = 1.0


@dataclass(frozen=True)
class Artifact:
    """A loaded, validated scoring model.

    Frozen and tuple-backed: one instance is shared by every scoring call for the
    process's lifetime, so mutability would be a cross-request bug waiting to
    happen.
    """

    kind: str
    coefficients: tuple[float, ...]
    intercept: float
    provider: str | None = None
    model: str | None = None
    dimension: int | None = None
    training: dict = field(default_factory=dict)


def artifact_dir() -> Path:
    """Directory holding the bundled artifacts."""
    return Path(__file__).resolve().parent.parent / "data" / "importance"


def bundled_artifact_path(name: str) -> Path:
    """Path to a bundled artifact by bare name, e.g. ``"titan-1536"``."""
    return artifact_dir() / f"{name}.json"


def load_artifact(path: str | Path) -> Artifact:
    """Load and validate a scoring artifact.

    Raises ``ConfigError`` naming the file and the specific problem. Every message
    is written to be actionable on its own: an operator reading "coefficient count
    1024 does not match declared dimension 1536" knows what to fix, where "failed
    to load model" sends them to read source.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Importance model artifact not found: {path}")

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not read importance artifact {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Importance artifact {path} must be a JSON object")

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"Importance artifact {path} has schema_version {version!r}; "
            f"this build understands {SCHEMA_VERSION}"
        )

    kind = raw.get("kind")
    if kind not in _KINDS:
        raise ConfigError(
            f"Importance artifact {path} has unknown kind {kind!r}; "
            f"expected one of {_KINDS}"
        )

    squash = raw.get("squash", "logistic")
    if squash not in _SQUASHES:
        raise ConfigError(
            f"Importance artifact {path} has unknown squash {squash!r}; "
            f"expected one of {_SQUASHES}"
        )

    coefficients = raw.get("coefficients")
    if not isinstance(coefficients, list) or not coefficients:
        raise ConfigError(
            f"Importance artifact {path} must have a non-empty coefficients list"
        )
    # `bool` is an `int` subclass, so it would pass a naive isinstance check and
    # then behave as 0/1 — excluded explicitly rather than tolerated.
    for i, c in enumerate(coefficients):
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise ConfigError(
                f"Importance artifact {path} coefficient {i} is not numeric: {c!r}"
            )

    intercept = raw.get("intercept", 0.0)
    if isinstance(intercept, bool) or not isinstance(intercept, (int, float)):
        raise ConfigError(
            f"Importance artifact {path} intercept is not numeric: {intercept!r}"
        )

    provider = model = None
    dimension = None

    if kind == "embedding_linear":
        emb = raw.get("embedding")
        if not isinstance(emb, dict):
            raise ConfigError(
                f"Importance artifact {path} of kind 'embedding_linear' requires "
                "an 'embedding' object naming provider, model, and dimension"
            )
        provider = emb.get("provider")
        model = emb.get("model")
        dimension = emb.get("dimension")
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise ConfigError(
                f"Importance artifact {path} embedding.dimension must be an "
                f"integer, got {dimension!r}"
            )
        if dimension != len(coefficients):
            raise ConfigError(
                f"Importance artifact {path} coefficient count "
                f"{len(coefficients)} does not match declared dimension {dimension}"
            )
    else:  # lexical
        if len(coefficients) != LEXICAL_FEATURE_COUNT:
            raise ConfigError(
                f"Importance artifact {path} coefficient count "
                f"{len(coefficients)} does not match the lexical feature count "
                f"{LEXICAL_FEATURE_COUNT}"
            )

    return Artifact(
        kind=kind,
        coefficients=tuple(float(c) for c in coefficients),
        intercept=float(intercept),
        provider=provider,
        model=model,
        dimension=dimension,
        # Normalized to {} rather than left as None: callers read
        # `artifact.training.get(...)`, and a hand-written artifact that omits the
        # block would otherwise raise AttributeError at startup.
        training=raw.get("training") or {},
    )
