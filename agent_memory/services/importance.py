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
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

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


# --------------------------------------------------------------------------
# Lexical features
#
# The fallback feature space, used when no artifact matches the configured
# embedder. Deliberately crude: seven interpretable numbers a regex can produce,
# not a replacement for the embedding head. Its job is to be *better than a
# constant* on a deployment we have not trained an artifact for.
#
# The ORDER IS THE WIRE FORMAT. Coefficients in a trained artifact are
# positional, so inserting a feature in the middle silently reassigns every
# weight and the model goes on returning plausible numbers. Append only, and bump
# LEXICAL_FEATURE_COUNT and SCHEMA_VERSION when you do.
# --------------------------------------------------------------------------

LEXICAL_FEATURE_NAMES = (
    "length",
    "digit_ratio",
    "preference",
    "identity",
    "temporal",
    "interrogative",
    "entity",
)

# Standing rules and stated preferences — the archetypal long-term memory.
_PREFERENCE_TERMS = frozenset({
    "prefer", "prefers", "preferred", "preference", "preferences",
    "always", "never", "must", "should", "avoid", "avoids",
    "favorite", "favourite", "hate", "hates", "dislike", "dislikes",
    "policy", "convention", "standard", "rule", "requires", "required",
})

# Facts about the user rather than about the task at hand.
_IDENTITY_TERMS = frozenset({
    "my", "mine", "our", "ours", "i'm", "im", "we're",
    "myself", "ourselves", "me",
})

# Scoped, expiring facts — the archetypal *short*-term memory. Expected to train
# with a negative weight.
_TEMPORAL_TERMS = frozenset({
    "today", "tomorrow", "yesterday", "tonight", "now", "currently",
    "temporarily", "temporary", "meanwhile", "sprint", "standup", "asap",
})

# Length above which more text stops meaning more signal. A 4000-character
# memory is not four times as important as a 1000-character one, and without a
# cap this feature would dominate the linear combination on outliers.
_LENGTH_SATURATION = 1000

# Marker counts saturate too: three preference words is as strong a signal as
# ten, and an unbounded count would make one ranting memory an outlier.
_MARKER_SATURATION = 3

# Word tokens, apostrophes kept so "I'm" survives as one token.
_WORD_RE = re.compile(r"[A-Za-z']+")


def _marker_ratio(tokens: list[str], terms: frozenset[str]) -> float:
    hits = sum(1 for t in tokens if t in terms)
    return min(hits, _MARKER_SATURATION) / _MARKER_SATURATION


def lexical_features(content: str | None) -> list[float]:
    """Extract the fallback feature vector from raw text.

    Returns exactly ``LEXICAL_FEATURE_COUNT`` values, each in ``[0.0, 1.0]``.
    Bounded on purpose: an unbounded feature crossed with a trained weight can
    push the pre-squash sum far enough that ``logistic`` saturates, and a scorer
    that returns 1.0 for everything long is worse than one that returns 0.5.
    """
    text = content or ""
    n = len(text)
    if n == 0:
        return [0.0] * LEXICAL_FEATURE_COUNT

    raw_words = _WORD_RE.findall(text)
    tokens = [w.lower() for w in raw_words]

    length = min(n, _LENGTH_SATURATION) / _LENGTH_SATURATION
    digit_ratio = sum(1 for c in text if c.isdigit()) / n
    preference = _marker_ratio(tokens, _PREFERENCE_TERMS)
    identity = _marker_ratio(tokens, _IDENTITY_TERMS)
    temporal = _marker_ratio(tokens, _TEMPORAL_TERMS)
    interrogative = 1.0 if "?" in text else 0.0

    # Skip index 0: a sentence-initial capital is grammar, not an entity. Without
    # this, every memory that starts with "The" scores an entity it does not have.
    if len(raw_words) > 1:
        rest = raw_words[1:]
        entity = sum(1 for w in rest if w[:1].isupper()) / len(rest)
    else:
        entity = 0.0

    return [length, digit_ratio, preference, identity, temporal, interrogative, entity]


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
    """Path to a bundled artifact by bare name, e.g. ``"lexical"``."""
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


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def logistic(x: float) -> float:
    """Numerically stable logistic squash.

    The naive ``1 / (1 + exp(-x))`` raises ``OverflowError`` for ``x < -710``,
    which a trained model on an outlier embedding can reach. A crash here is not
    cosmetic: it propagates out of ``_enrich_memory`` and the memory ends up
    ``enrichment_status: "failed"``. So the branch is on the sign, keeping the
    exponent negative either way.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clamp(value: float) -> float:
    """Confine a score to ``[MIN_IMPORTANCE, MAX_IMPORTANCE]``.

    Applied outside the model arithmetic on purpose. ``logistic`` already lands in
    ``(0, 1)``, so this looks redundant — but it is the only thing standing between
    a badly trained artifact and ``importance: 0.0``, which
    ``ConsolidationWorker._forget_low_importance`` reads as "delete this". Cheap
    insurance against a silent, irreversible failure.
    """
    return max(MIN_IMPORTANCE, min(MAX_IMPORTANCE, value))


@runtime_checkable
class ImportanceScorer(Protocol):
    """The seam. One method, called once per long-term memory during enrichment.

    The signature is the union of what both implementations need, not the
    intersection: ``LLMScorer`` uses ``content`` alone and ignores the rest,
    ``LocalScorer``'s embedding kind uses ``embedding`` alone. Passing everything
    the caller has means adding a feature to the local model later does not change
    the call site. ``tags`` and ``message_type`` are unused by both today and are
    reserved for exactly that.
    """

    async def score(
        self,
        content: str,
        embedding: Sequence[float] | None = None,
        *,
        tags: Sequence[str] | None = None,
        message_type: str | None = None,
    ) -> float:
        """Return importance in ``[0.1, 1.0]``."""
        ...


class LLMScorer:
    """Today's behaviour, unchanged, behind the protocol.

    Wrapping rather than rewriting is the point: this is what makes the default
    path a no-op refactor. The prompt handling reproduces
    ``EnrichmentWorker._process_standard_enrichment`` exactly, including omitting
    the ``prompt`` kwarg entirely when no custom prompt is configured — passing
    ``prompt=None`` would be a different call, and the providers' defaults are
    keyed on the kwarg's absence.
    """

    def __init__(
        self,
        llm,
        prompt_getter: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> None:
        self._llm = llm
        self._prompt_getter = prompt_getter

    async def score(
        self,
        content: str,
        embedding: Sequence[float] | None = None,
        *,
        tags: Sequence[str] | None = None,
        message_type: str | None = None,
    ) -> float:
        prompt = None
        if self._prompt_getter is not None:
            prompt = await self._prompt_getter("importance_assessment")
        if prompt:
            return await self._llm.assess_importance(content, prompt=prompt)
        return await self._llm.assess_importance(content)


class LocalScorer:
    """A pre-trained linear model evaluated in-process.

    No network call, no tokens, microseconds instead of a round trip. Viable only
    because the embedding is already computed by the time enrichment runs, so the
    "inference" is a dot product over a vector we were holding anyway.
    """

    def __init__(self, artifact: Artifact) -> None:
        # Public: "which model is actually loaded" is the first question anyone
        # debugging importance drift asks, and the Artifact is frozen, so exposing
        # it costs nothing.
        self.artifact = artifact

    def _validated_embedding(self, embedding: Sequence[float] | None) -> Sequence[float]:
        """Reject a vector this artifact cannot score. REQ-E-165.

        Raising is the deliberate choice over degrading. A silent fallback to the
        intercept would return *the same plausible number* for every memory in the
        store, and the only symptom would be recall quality drifting weeks later.
        Raising lands in ``_enrich_memory``'s retry path, which parks the affected
        memories in ``enrichment_status: "failed"`` — countable, queryable, fixable.
        """
        expected = self.artifact.dimension or len(self.artifact.coefficients)
        if not embedding:
            raise ConfigError(
                "Local importance model of kind 'embedding_linear' requires an "
                f"embedding of dimension {expected}, but none was provided. "
                "Set IMPORTANCE_SCORER=llm or configure a matching embedder."
            )
        if len(embedding) != expected:
            raise ConfigError(
                f"Embedding dimension {len(embedding)} does not match importance "
                f"model dimension {expected} "
                f"(model trained for {self.artifact.provider}/{self.artifact.model}). "
                "Scoring a truncated vector would return a plausible wrong number, "
                "so this refuses instead."
            )
        for i, v in enumerate(embedding):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(
                    f"Embedding value at index {i} is not numeric: {v!r}"
                )
        return embedding

    async def score(
        self,
        content: str,
        embedding: Sequence[float] | None = None,
        *,
        tags: Sequence[str] | None = None,
        message_type: str | None = None,
    ) -> float:
        coefficients = self.artifact.coefficients

        if self.artifact.kind == "embedding_linear":
            features: Sequence[float] = self._validated_embedding(embedding)
        else:
            # The lexical head never touches the embedding: its coefficients are
            # indexed by feature position, so a 1536-vector would be scored against
            # the wrong weights and return a plausible number.
            features = lexical_features(content)

        total = self.artifact.intercept
        for weight, value in zip(coefficients, features):
            total += weight * value

        return _clamp(logistic(total))
