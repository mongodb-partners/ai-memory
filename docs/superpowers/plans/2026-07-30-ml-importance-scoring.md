# Pluggable Importance Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, local ML scorer that produces a memory's `importance` in microseconds without an LLM call, selected by config, with the existing LLM path unchanged and still the default.

**Architecture:** A one-method `ImportanceScorer` protocol with two implementations. `LLMScorer` wraps today's `providers.llm.assess_importance` call verbatim. `LocalScorer` evaluates a logistic linear model in pure Python over the embedding the enrichment worker already holds, falling back to seven dependency-free lexical features when no artifact matches the configured embedder. `ProviderManager` selects one from config; `AsyncMemory._maybe_start_workers` injects it into `EnrichmentWorker`.

**Tech Stack:** Python 3.11+, Pydantic Settings, Motor/PyMongo, pytest + pytest-asyncio (`asyncio_mode = "auto"`). Training only (never imported by the library): scikit-learn, numpy, pandas, `datasets`.

**Spec:** `docs/superpowers/specs/2026-07-30-ml-importance-scoring-design.md` (commit `d1ecbf4`)

> **Implemented — one deliverable changed on measurement.** This plan is kept as
> written for the record; the shipped code differs in one respect. The plan calls
> for three bundled artifacts (`lexical`, `titan-1536`, `voyage-3-1024`). Only
> `lexical` ships. The two embedding artifacts were created as zero-coefficient
> placeholders, found to score every memory an identical 0.5 — which disables
> promotion while looking healthy — and then **deleted rather than trained**: an
> embedding head's held-out Spearman tops out near 0.45 against an in-sample
> ceiling of 0.70, and `assess_importance` emits only 9 distinct label values for a
> 1024-coefficient fit to aim at. `_BUNDLED_ARTIFACTS` is therefore empty and every
> embedder selects `lexical` by design. Where the sections below name those two
> artifacts or assert on their names — notably the selection and construction-order
> tests — read the shipped versions in `tests/unit/test_importance_selection.py`
> and `tests/unit/test_importance_artifact.py` instead; they assert the same
> properties without depending on a populated map. See the 4.2.0 CHANGELOG entry.

## Global Constraints

- **No new runtime dependencies.** `agent_memory/` must not import numpy, scikit-learn, pandas, or torch. The runtime dependency list in `pyproject.toml` stays at its current 8 entries. Training deps go in a new `training` optional-dependency group.
- **`importance_scorer` defaults to `"llm"`.** An existing install upgrading to 4.2.0 must behave identically. This is the primary safety property.
- **Scores are clamped to `[0.1, 1.0]`.** Never 0.0. `providers/base.py:44-51` explains why: 0.0 sits at `forgetting_score_threshold`, so a zero is a deletion order.
- **No pickle.** Artifacts are JSON only. Pickle loading is arbitrary code execution and version-couples artifacts to sklearn.
- **Mocks of `ImportanceScorer` use `create_autospec(ImportanceScorer, instance=True)`**, never `AsyncMock(spec=...)`. Verified on Python 3.11.13: `AsyncMock(spec=Class.method)` accepts bogus kwargs and extra positional args; `create_autospec` rejects both.
- **Python floor:** `>=3.11`. Target version `4.2.0` (minor, additive).
- **Test invocation:** `.venv/bin/python -m pytest` from the repo root. `asyncio_mode = "auto"`, so async tests need no decorator.
- **Requirement IDs:** REQ-E-160 … REQ-E-172, per the spec's §12 index.
- **Config construction in tests:** always `MCPConfig(**defaults, _env_file=None)` — a live `.env` exists in this working tree and will leak into tests otherwise.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `agent_memory/services/importance.py` | The `ImportanceScorer` protocol, `LLMScorer`, `LocalScorer`, artifact loading and validation, lexical feature extraction. One module because these change together and none is independently useful. |
| **Create** `agent_memory/data/importance/lexical.json` | Provider-independent fallback coefficients. |
| **Create** `agent_memory/data/importance/titan-1536.json` | Coefficients for `bedrock` / `amazon.titan-embed-text-v1` / 1536. |
| **Create** `agent_memory/data/importance/voyage-3-1024.json` | Coefficients for `voyage` / `voyage-3` / 1024. |
| **Create** `scripts/train_importance.py` | Offline trainer. Outside the package — imports sklearn. |
| **Modify** `agent_memory/core/config.py` | `importance_scorer`, `importance_model_path`, validator arm. |
| **Modify** `agent_memory/providers/manager.py` | `_create_scorer`, `self.scorer` assigned **last**. |
| **Modify** `agent_memory/services/enrichment.py` | `scorer` ctor param; importance branch collapses to one call. |
| **Modify** `agent_memory/memory.py` | Pass `scorer=self.providers.scorer`. |
| **Modify** `pyproject.toml` | `training` extra; artifacts in sdist allow-list; `4.2.0`. |
| **Create** `tests/unit/test_importance_scorer.py` | Scoring math, clamping, dimension mismatch, lexical features. |
| **Create** `tests/unit/test_importance_artifact.py` | Load contract, bundled-artifact integrity, construction order. |
| **Modify** `tests/unit/test_enrichment.py` | Scorer injection, default-path regression, non-call assertion. |
| **Modify** `tests/unit/test_config.py` | New field defaults and validation. |
| **Modify** `tests/unit/test_packaging.py` | Artifacts in sdist and wheel; `training` extra present. |
| **Create** `tests/unit/test_importance_features.py` | Lexical feature values, bounds, and pinned order. |
| **Modify** `CHANGELOG.md`, `README.md` | Document the new config surface and the training workflow. |

**Task order rationale:** artifact loading (Task 1) has no dependencies and everything else consumes it. Feature extraction (Task 2) and the scorers (Task 3) build on it. Config (Task 4) and provider selection (Task 5) wire it up. The worker change (Task 6) comes last among runtime changes because it is the only one touching a hot path — everything it needs already exists and is tested by then. Packaging (Task 7) ships the artifacts. The trainer (Task 8) is last: it emits against a format the runtime already validates, and its final step replaces Task 1's placeholder coefficients with trained ones, which is only meaningful once the loader, scorer, and thresholds are all in place to check them against.

---

## Task 1: Artifact format and loader

**Files:**
- Create: `agent_memory/services/importance.py`
- Create: `agent_memory/data/importance/lexical.json`
- Create: `agent_memory/data/importance/titan-1536.json`
- Create: `agent_memory/data/importance/voyage-3-1024.json`
- Test: `tests/unit/test_importance_artifact.py`

**Requirements:** REQ-E-163 (versioned artifact format), REQ-E-171 (fail loudly on an unusable artifact).

**Interfaces:**
- Consumes: `agent_memory.exceptions.ConfigError` (existing, `exceptions.py:28`).
- Produces:
  - `SCHEMA_VERSION: int = 1`
  - `LEXICAL_FEATURE_COUNT: int = 7`
  - `@dataclass(frozen=True) class Artifact` with fields `kind: str`, `coefficients: tuple[float, ...]`, `intercept: float`, `provider: str | None`, `model: str | None`, `dimension: int | None`, `training: dict` (never `None` — the loader normalizes a missing block to `{}`)
  - `def load_artifact(path: str | Path) -> Artifact`
  - `def bundled_artifact_path(name: str) -> Path`
  - `def artifact_dir() -> Path`

Artifacts created here carry **zero coefficients and a 0.5 intercept** — deliberately untrained placeholders so the loader and scorer can be built and tested before the trainer exists. Task 8 replaces their numbers. A zero-weight model returns `logistic(0.5) ≈ 0.62` for every input, which is obviously-neutral rather than subtly-wrong.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_importance_artifact.py`:

```python
"""Artifact loading contract for local importance scoring. REQ-E-163, REQ-E-171.

These tests exist because a bad artifact is a silent failure mode. A file with
1024 coefficients loaded against a 1536-dim embedder would score the overlapping
prefix and return a plausible number, and the only symptom would be memories
being forgotten or promoted wrongly weeks later — the same class of invisible
defect recorded in `test_importance_parsing.py`.
"""

import json
import pathlib

import pytest

from agent_memory.exceptions import ConfigError
from agent_memory.services.importance import (
    LEXICAL_FEATURE_COUNT,
    SCHEMA_VERSION,
    artifact_dir,
    bundled_artifact_path,
    load_artifact,
)


def _valid_embedding_artifact(**overrides) -> dict:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "kind": "embedding_linear",
        "embedding": {"provider": "bedrock", "model": "test-model", "dimension": 3},
        "coefficients": [0.1, 0.2, 0.3],
        "intercept": 0.4,
        "squash": "logistic",
        "training": {"labels": ["synthetic"], "n_samples": 0},
    }
    doc.update(overrides)
    return doc


def _write(tmp_path: pathlib.Path, doc: dict, name: str = "a.json") -> pathlib.Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


class TestValidArtifacts:
    def test_loads_embedding_artifact(self, tmp_path):
        art = load_artifact(_write(tmp_path, _valid_embedding_artifact()))
        assert art.kind == "embedding_linear"
        assert art.coefficients == (0.1, 0.2, 0.3)
        assert art.intercept == 0.4
        assert art.provider == "bedrock"
        assert art.model == "test-model"
        assert art.dimension == 3

    def test_loads_lexical_artifact(self, tmp_path):
        doc = {
            "schema_version": SCHEMA_VERSION,
            "kind": "lexical",
            "coefficients": [0.0] * LEXICAL_FEATURE_COUNT,
            "intercept": 0.5,
            "squash": "logistic",
            "training": {},
        }
        art = load_artifact(_write(tmp_path, doc))
        assert art.kind == "lexical"
        assert art.dimension is None
        assert len(art.coefficients) == LEXICAL_FEATURE_COUNT

    def test_coefficients_are_immutable(self, tmp_path):
        """A shared Artifact must not be mutable by one scorer."""
        art = load_artifact(_write(tmp_path, _valid_embedding_artifact()))
        assert isinstance(art.coefficients, tuple)


class TestRejections:
    """Every rejection must name the file and the specific problem.

    "Failed to load model" sends an operator to read source. "coefficient count
    2 does not match declared dimension 3" sends them to the file.
    """

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_artifact(tmp_path / "nope.json")

    def test_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ConfigError, match="bad.json"):
            load_artifact(path)

    def test_unknown_schema_version(self, tmp_path):
        path = _write(tmp_path, _valid_embedding_artifact(schema_version=99))
        with pytest.raises(ConfigError, match="schema_version"):
            load_artifact(path)

    def test_unknown_kind(self, tmp_path):
        path = _write(tmp_path, _valid_embedding_artifact(kind="neural_net"))
        with pytest.raises(ConfigError, match="kind"):
            load_artifact(path)

    def test_coefficient_count_mismatch(self, tmp_path):
        doc = _valid_embedding_artifact(coefficients=[0.1, 0.2])
        path = _write(tmp_path, doc)
        with pytest.raises(ConfigError, match="does not match"):
            load_artifact(path)

    def test_lexical_wrong_feature_count(self, tmp_path):
        doc = {
            "schema_version": SCHEMA_VERSION,
            "kind": "lexical",
            "coefficients": [0.0] * (LEXICAL_FEATURE_COUNT - 1),
            "intercept": 0.5,
            "training": {},
        }
        with pytest.raises(ConfigError, match="does not match"):
            load_artifact(_write(tmp_path, doc))

    def test_non_numeric_coefficient(self, tmp_path):
        doc = _valid_embedding_artifact(coefficients=[0.1, "oops", 0.3])
        with pytest.raises(ConfigError, match="numeric"):
            load_artifact(_write(tmp_path, doc))

    def test_embedding_artifact_without_embedding_block(self, tmp_path):
        doc = _valid_embedding_artifact()
        del doc["embedding"]
        with pytest.raises(ConfigError, match="embedding"):
            load_artifact(_write(tmp_path, doc))

    def test_unknown_squash(self, tmp_path):
        doc = _valid_embedding_artifact(squash="softmax")
        with pytest.raises(ConfigError, match="squash"):
            load_artifact(_write(tmp_path, doc))


class TestBundledArtifacts:
    """Cheap integrity checks on the files we ship. Catches a hand-edited file."""

    def test_artifact_dir_exists(self):
        assert artifact_dir().is_dir()

    @pytest.mark.parametrize(
        "name", ["lexical", "titan-1536", "voyage-3-1024"]
    )
    def test_bundled_artifact_loads(self, name):
        art = load_artifact(bundled_artifact_path(name))
        assert art.kind in ("embedding_linear", "lexical")

    @pytest.mark.parametrize("name", ["titan-1536", "voyage-3-1024"])
    def test_bundled_dimension_matches_coefficient_count(self, name):
        art = load_artifact(bundled_artifact_path(name))
        assert art.dimension == len(art.coefficients)

    def test_bundled_lexical_has_seven_features(self):
        art = load_artifact(bundled_artifact_path("lexical"))
        assert len(art.coefficients) == LEXICAL_FEATURE_COUNT

    def test_titan_artifact_declares_titan(self):
        art = load_artifact(bundled_artifact_path("titan-1536"))
        assert art.provider == "bedrock"
        assert art.model == "amazon.titan-embed-text-v1"
        assert art.dimension == 1536

    def test_voyage_artifact_declares_voyage_3(self):
        art = load_artifact(bundled_artifact_path("voyage-3-1024"))
        assert art.provider == "voyage"
        assert art.model == "voyage-3"
        assert art.dimension == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_importance_artifact.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_memory.services.importance'`

- [ ] **Step 3: Write the loader**

Create `agent_memory/services/importance.py`:

```python
"""Pluggable importance scoring — LLM or a local pre-trained linear model.

`importance` decides whether a memory is forgotten
(`ConsolidationWorker._forget_low_importance`), promoted
(`_promote_to_ltm`), and how it ranks (`MemoryService._calibrated_rank`). It is
produced by exactly one call today: `providers.llm.assess_importance`, one LLM
round trip per long-term memory.

This module makes that call swappable. The local path is viable only because the
embedding already exists before scoring happens — `EnrichmentWorker` holds
`memory["embedding"]` and passes it to `evolve_memory` — so a linear head on it
is a dot product rather than an encoder forward pass. Pure Python, no numpy: the
library ships 8 runtime dependencies and a scientific stack is not going to be
the ninth.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
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
    training: dict | None = None


def artifact_dir() -> Path:
    """Directory holding the bundled artifacts."""
    return Path(__file__).resolve().parent.parent / "data" / "importance"


def bundled_artifact_path(name: str) -> Path:
    """Path to a bundled artifact by bare name, e.g. ``"titan-1536"``."""
    return artifact_dir() / f"{name}.json"


def load_artifact(path: str | Path) -> Artifact:
    """Load and validate a scoring artifact.

    Raises `ConfigError` naming the file and the specific problem. Every message
    is written to be actionable on its own: an operator reading
    "coefficient count 1024 does not match declared dimension 1536" knows what to
    fix, where "failed to load model" sends them to read source.
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
                f"Importance artifact {path} of kind 'embedding_linear' "
                "requires an 'embedding' object naming provider, model, dimension"
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
        training=raw.get("training") or {},
    )
```

- [ ] **Step 4: Create the three placeholder artifacts**

`agent_memory/data/importance/lexical.json` — all-zero weights, neutral intercept. Task 8 replaces these numbers.

```json
{
  "schema_version": 1,
  "kind": "lexical",
  "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "intercept": 0.5,
  "squash": "logistic",
  "training": {
    "labels": ["untrained-placeholder"],
    "n_samples": 0,
    "trainer_version": "0.0.0",
    "metrics": {},
    "note": "Placeholder. Regenerated by scripts/train_importance.py (plan Task 8)."
  }
}
```

For `titan-1536.json` and `voyage-3-1024.json`, generate rather than hand-write — 1536 zeros is not something to type. Run:

```bash
.venv/bin/python - <<'PY'
import json, pathlib
out = pathlib.Path("agent_memory/data/importance")
out.mkdir(parents=True, exist_ok=True)
specs = [
    ("titan-1536", "bedrock", "amazon.titan-embed-text-v1", 1536),
    ("voyage-3-1024", "voyage", "voyage-3", 1024),
]
for name, provider, model, dim in specs:
    doc = {
        "schema_version": 1,
        "kind": "embedding_linear",
        "embedding": {"provider": provider, "model": model, "dimension": dim},
        "coefficients": [0.0] * dim,
        "intercept": 0.5,
        "squash": "logistic",
        "training": {
            "labels": ["untrained-placeholder"],
            "n_samples": 0,
            "trainer_version": "0.0.0",
            "metrics": {},
            "note": "Placeholder. Regenerated by scripts/train_importance.py (plan Task 8).",
        },
    }
    (out / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", name, dim)
PY
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_importance_artifact.py -q`
Expected: PASS — 21 tests (16 functions, two parametrized).

- [ ] **Step 6: Commit**

```bash
git add agent_memory/services/importance.py \
        agent_memory/data/importance/ \
        tests/unit/test_importance_artifact.py
git commit -m "Add importance artifact format and loader

JSON rather than pickle: artifacts ship in the wheel, and unpickling a
shipped binary is arbitrary code execution. Every rejection names the
file and the problem, because the failure this guards against is a
coefficient/dimension mismatch that scores the overlapping prefix and
returns a plausible number — invisible until memories start
disappearing.

Coefficients are placeholders (zero weights, neutral intercept) so the
scorer can be built and tested before the trainer exists."
```

---

## Task 2: Lexical features

**Files:**
- Modify: `agent_memory/services/importance.py`
- Test: `tests/unit/test_importance_features.py`

**Requirements:** REQ-E-166 (seven dependency-free lexical features).

**Interfaces:**
- Consumes: `LEXICAL_FEATURE_COUNT` (Task 1).
- Produces: `def lexical_features(content: str) -> list[float]` — exactly `LEXICAL_FEATURE_COUNT` values, each in `[0.0, 1.0]`, and `LEXICAL_FEATURE_NAMES: tuple[str, ...]` so the trainer can label its own weights.

The seven features, fixed in this order (the order is the wire format — reordering silently reinterprets every trained artifact):

| # | Name | Definition | Why it should predict importance |
|---|---|---|---|
| 0 | `length` | `min(len(content), 1000) / 1000` | Longer statements carry more committed detail. Saturates at 1000 chars so an essay does not dominate. |
| 1 | `digit_ratio` | digits / `len` | Numbers are specifics — versions, quantities, dates, IDs. |
| 2 | `preference` | `min(hits, 3) / 3` over `_PREFERENCE_TERMS` | "prefers", "always", "never", "must" mark durable rules, the archetypal long-term memory. |
| 3 | `identity` | `min(hits, 3) / 3` over `_IDENTITY_TERMS` | "my", "I am", "our team" mark facts about the user rather than about the task. |
| 4 | `temporal` | `min(hits, 3) / 3` over `_TEMPORAL_TERMS` | "today", "this sprint", "for now" mark the opposite — scoped, expiring facts. Expected to train **negative**. |
| 5 | `interrogative` | `1.0` if the content contains `?` else `0.0` | A question is a request, not a fact worth keeping. Expected negative. |
| 6 | `entity` | capitalized non-initial tokens / tokens | Proper nouns — people, products, repos — are what makes a memory retrievable later. |

Two of the seven are expected to carry negative weight. That is the point: a bag of positive-only signals cannot distinguish "I always deploy on Fridays" from "deploy the branch today", and the second is exactly what should be forgotten.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_importance_features.py`:

```python
"""Lexical feature extraction for the local importance scorer. REQ-E-166.

The feature *order* is a wire format: coefficients in a shipped artifact are
positional. Reordering or inserting a feature without retraining silently
reassigns every weight, and the model keeps returning plausible numbers. Hence
the explicit index assertions below — they exist to fail loudly on a refactor
that looks harmless.
"""

import pytest

from agent_memory.services.importance import (
    LEXICAL_FEATURE_COUNT,
    LEXICAL_FEATURE_NAMES,
    lexical_features,
)

DURABLE = "My manager always wants the release notes in Markdown, never plain text."
EPHEMERAL = "Can you deploy branch fix-3 today?"


class TestContract:
    def test_names_match_count(self):
        assert len(LEXICAL_FEATURE_NAMES) == LEXICAL_FEATURE_COUNT

    def test_expected_order(self):
        """Pinned to catch a reorder. Changing this requires retraining."""
        assert LEXICAL_FEATURE_NAMES == (
            "length",
            "digit_ratio",
            "preference",
            "identity",
            "temporal",
            "interrogative",
            "entity",
        )

    @pytest.mark.parametrize(
        "content", ["", " ", "x", DURABLE, EPHEMERAL, "?" * 5000, "123456"]
    )
    def test_always_returns_bounded_vector(self, content):
        feats = lexical_features(content)
        assert len(feats) == LEXICAL_FEATURE_COUNT
        assert all(0.0 <= f <= 1.0 for f in feats), feats

    def test_empty_content_does_not_divide_by_zero(self):
        assert lexical_features("") == [0.0] * LEXICAL_FEATURE_COUNT


class TestIndividualFeatures:
    def _f(self, content: str, name: str) -> float:
        return lexical_features(content)[LEXICAL_FEATURE_NAMES.index(name)]

    def test_length_saturates(self):
        assert self._f("a" * 500, "length") == pytest.approx(0.5)
        assert self._f("a" * 1000, "length") == 1.0
        assert self._f("a" * 9000, "length") == 1.0

    def test_digit_ratio(self):
        assert self._f("1234", "digit_ratio") == 1.0
        assert self._f("ab12", "digit_ratio") == pytest.approx(0.5)
        assert self._f("abcd", "digit_ratio") == 0.0

    def test_preference_terms_counted_and_capped(self):
        assert self._f("The sky is blue.", "preference") == 0.0
        assert self._f("I always use tabs.", "preference") > 0.0
        many = "I always prefer tabs, never spaces, and must have trailing commas."
        assert self._f(many, "preference") == 1.0

    def test_preference_matching_is_case_insensitive(self):
        assert self._f("ALWAYS use tabs", "preference") > 0.0

    def test_preference_requires_whole_words(self):
        """'preferential' and 'somewhere' are not preference statements."""
        assert self._f("A preferential ballot is somewhere in the docs.", "preference") == 0.0

    def test_identity_terms(self):
        assert self._f("My team owns billing.", "identity") > 0.0
        assert self._f("The team owns billing.", "identity") == 0.0

    def test_temporal_terms(self):
        assert self._f("Ship it today.", "temporal") > 0.0
        assert self._f("Ship it.", "temporal") == 0.0

    def test_interrogative_is_binary(self):
        assert self._f("What is this?", "interrogative") == 1.0
        assert self._f("This is that.", "interrogative") == 0.0
        assert self._f("Really?? Yes??", "interrogative") == 1.0

    def test_entity_ignores_sentence_initial_capitals(self):
        """'The' at position 0 is grammar, not an entity."""
        assert self._f("The cat sat on the mat.", "entity") == 0.0
        assert self._f("The cat belongs to Priya.", "entity") > 0.0

    def test_entity_counts_multiple(self):
        one = self._f("we deploy via Terraform", "entity")
        two = self._f("we deploy Atlas via Terraform", "entity")
        assert two > one


class TestDiscrimination:
    """The features have to separate the two cases the scorer exists to separate."""

    def test_durable_and_ephemeral_differ(self):
        assert lexical_features(DURABLE) != lexical_features(EPHEMERAL)

    def test_durable_scores_higher_on_preference(self):
        i = LEXICAL_FEATURE_NAMES.index("preference")
        assert lexical_features(DURABLE)[i] > lexical_features(EPHEMERAL)[i]

    def test_ephemeral_scores_higher_on_temporal_and_question(self):
        t = LEXICAL_FEATURE_NAMES.index("temporal")
        q = LEXICAL_FEATURE_NAMES.index("interrogative")
        d, e = lexical_features(DURABLE), lexical_features(EPHEMERAL)
        assert e[t] > d[t]
        assert e[q] > d[q]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_importance_features.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'LEXICAL_FEATURE_NAMES'`

- [ ] **Step 3: Implement**

Append to `agent_memory/services/importance.py`, after the constants and before `Artifact`:

```python
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


def lexical_features(content: str) -> list[float]:
    """Extract the fallback feature vector from raw text.

    Returns exactly ``LEXICAL_FEATURE_COUNT`` values, each in ``[0.0, 1.0]``.
    Bounded on purpose: an unbounded feature crossed with a trained weight can
    push the pre-squash sum far enough that `logistic` saturates, and a scorer
    that returns 1.0 for everything long is worse than one that returns 0.5.
    """
    text = content or ""
    n = len(text)
    if n == 0:
        return [0.0] * LEXICAL_FEATURE_COUNT

    tokens = [w.lower() for w in _WORD_RE.findall(text)]

    length = min(n, _LENGTH_SATURATION) / _LENGTH_SATURATION
    digit_ratio = sum(1 for c in text if c.isdigit()) / n
    preference = _marker_ratio(tokens, _PREFERENCE_TERMS)
    identity = _marker_ratio(tokens, _IDENTITY_TERMS)
    temporal = _marker_ratio(tokens, _TEMPORAL_TERMS)
    interrogative = 1.0 if "?" in text else 0.0

    # Skip index 0: a sentence-initial capital is grammar, not an entity. Without
    # this, every memory that starts with "The" scores an entity it does not have.
    raw_words = _WORD_RE.findall(text)
    if len(raw_words) > 1:
        capitalized = sum(1 for w in raw_words[1:] if w[:1].isupper())
        entity = capitalized / len(raw_words[1:])
    else:
        entity = 0.0

    return [length, digit_ratio, preference, identity, temporal, interrogative, entity]
```

`import re` goes at the top of the module with the other stdlib imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_importance_features.py -q`
Expected: PASS — 23 tests (16 functions, one parametrized over 7 inputs).

Also confirm Task 1 still passes: `.venv/bin/python -m pytest tests/unit/test_importance_artifact.py -q`

- [ ] **Step 5: Commit**

```bash
git add agent_memory/services/importance.py tests/unit/test_importance_features.py
git commit -m "Add lexical features for the local importance scorer

Seven bounded, interpretable signals for deployments with no trained
artifact for their embedder. Two are expected to train negative
(temporal, interrogative) — a positive-only bag cannot tell 'I always
deploy on Fridays' from 'deploy the branch today', and the second is
exactly what should be forgotten.

Feature order is the wire format; tests pin it because a reorder
silently reassigns every trained weight."
```

---

## Task 3: The `ImportanceScorer` protocol, `LLMScorer`, and `LocalScorer`

**Files:**
- Modify: `agent_memory/services/importance.py`
- Test: `tests/unit/test_importance_scorer.py`

**Requirements:** REQ-E-160 (protocol; `LLMScorer` preserves behaviour), REQ-E-161 (pure Python), REQ-E-165 (logistic head, clamp, raise on mismatch), REQ-E-171.

**Interfaces:**
- Consumes: `Artifact`, `load_artifact` (Task 1); `lexical_features`, `LEXICAL_FEATURE_COUNT` (Task 2); `agent_memory.providers.base.LLMProvider` (typing only, imported under `TYPE_CHECKING` to avoid an import cycle — `providers.manager` will import this module).
- Produces:
  - `@runtime_checkable class ImportanceScorer(Protocol)` with `async def score(self, content, embedding=None, *, tags=None, message_type=None) -> float`
  - `class LLMScorer` — `__init__(self, llm, prompt_getter=None)`
  - `class LocalScorer` — `__init__(self, artifact: Artifact)`
  - `def logistic(x: float) -> float`

`LLMScorer` owns the prompt lookup that lives in `EnrichmentWorker._process_standard_enrichment` today. Moving it here is what lets the worker's importance branch collapse to one unconditional call in Task 6 — the worker cannot keep an `if importance_prompt` branch if the local path has no prompt.

`prompt_getter` is an `async` callable taking a prompt name and returning `str | None`. The worker passes its own `_get_prompt`. Optional so a caller with no prompt library gets today's no-prompt behaviour.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_importance_scorer.py`:

```python
"""Scoring implementations for the pluggable importance seam.
REQ-E-160, REQ-E-161, REQ-E-165, REQ-E-171.

Two properties dominate these tests:

1. **The floor.** No scorer may emit below 0.1. 0.0 sits at
   `forgetting_score_threshold`, so returning it is an instruction to delete the
   memory — see `providers/base.py:44-51`. Asserted independently of the maths so
   a bad artifact cannot order a deletion.
2. **Refusing a mismatched embedding.** A 1024-vector against 1536 coefficients
   would happily score the overlapping prefix. That is the failure mode worth
   engineering against, because it produces plausible numbers. So does returning
   the intercept for everything, which is why this raises instead.
"""

import math
from unittest.mock import AsyncMock, create_autospec

import pytest

from agent_memory.exceptions import ConfigError
from agent_memory.providers.base import LLMProvider
from agent_memory.services.importance import (
    LEXICAL_FEATURE_COUNT,
    MAX_IMPORTANCE,
    MIN_IMPORTANCE,
    Artifact,
    ImportanceScorer,
    LLMScorer,
    LocalScorer,
    logistic,
)


def _embedding_artifact(coefficients=(1.0, 0.0, 0.0), intercept=0.0) -> Artifact:
    return Artifact(
        kind="embedding_linear",
        coefficients=tuple(coefficients),
        intercept=intercept,
        provider="bedrock",
        model="test-model",
        dimension=len(coefficients),
        training={},
    )


def _lexical_artifact(coefficients=None, intercept=0.0) -> Artifact:
    coefficients = coefficients or [0.0] * LEXICAL_FEATURE_COUNT
    return Artifact(
        kind="lexical",
        coefficients=tuple(coefficients),
        intercept=intercept,
        training={},
    )


class TestLogistic:
    def test_midpoint(self):
        assert logistic(0.0) == pytest.approx(0.5)

    def test_monotone(self):
        assert logistic(-1.0) < logistic(0.0) < logistic(1.0)

    @pytest.mark.parametrize("x", [1e9, -1e9, 800.0, -800.0])
    def test_no_overflow_on_extremes(self, x):
        """`math.exp(-(-800))` raises OverflowError. A saturating model must not
        crash the enrichment worker — a crash there retries to `failed`."""
        value = logistic(x)
        assert 0.0 <= value <= 1.0
        assert not math.isnan(value)


class TestProtocolConformance:
    def test_llm_scorer_is_a_scorer(self):
        assert isinstance(LLMScorer(AsyncMock()), ImportanceScorer)

    def test_local_scorer_is_a_scorer(self):
        assert isinstance(LocalScorer(_lexical_artifact()), ImportanceScorer)


class TestLLMScorer:
    async def test_delegates_to_provider(self):
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        assert await LLMScorer(llm).score("hello") == 0.7

    async def test_passes_prompt_from_getter(self):
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        getter = AsyncMock(return_value="Rate this: {content}")
        await LLMScorer(llm, prompt_getter=getter).score("hello")
        getter.assert_awaited_once_with("importance_assessment")
        llm.assess_importance.assert_awaited_once_with(
            "hello", prompt="Rate this: {content}"
        )

    async def test_omits_prompt_when_getter_returns_none(self):
        """Today's behaviour: no prompt kwarg at all, so the provider's own
        default template applies. Passing `prompt=None` explicitly would be a
        different call and is not what the current worker does."""
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        await LLMScorer(llm, prompt_getter=AsyncMock(return_value=None)).score("hi")
        llm.assess_importance.assert_awaited_once_with("hi")

    async def test_omits_prompt_when_no_getter(self):
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        await LLMScorer(llm).score("hi")
        llm.assess_importance.assert_awaited_once_with("hi")

    async def test_ignores_embedding_and_metadata(self):
        """The LLM path takes text only. Accepting the wider signature without
        using it is what makes the two implementations substitutable."""
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        await LLMScorer(llm).score(
            "hi", [0.1] * 1536, tags=["a"], message_type="human"
        )
        llm.assess_importance.assert_awaited_once_with("hi")


class TestLocalScorerEmbeddingPath:
    async def test_dot_product_then_logistic(self):
        art = _embedding_artifact(coefficients=(2.0, 0.0, 0.0), intercept=0.0)
        got = await LocalScorer(art).score("x", [1.0, 5.0, 5.0])
        assert got == pytest.approx(logistic(2.0))

    async def test_intercept_applied(self):
        art = _embedding_artifact(coefficients=(0.0, 0.0, 0.0), intercept=1.5)
        assert await LocalScorer(art).score("x", [1.0, 1.0, 1.0]) == pytest.approx(
            logistic(1.5)
        )

    async def test_higher_dot_product_scores_higher(self):
        scorer = LocalScorer(_embedding_artifact(coefficients=(1.0, 1.0, 1.0)))
        low = await scorer.score("x", [0.0, 0.0, 0.0])
        high = await scorer.score("x", [1.0, 1.0, 1.0])
        assert high > low


class TestLocalScorerRefusesUnusableInput:
    """REQ-E-165. An embedding artifact handed an unusable vector raises.

    The tempting alternative — return the intercept, log a warning — is worse. It
    produces the same plausible number for every memory in the store, and the
    only symptom is recall quality drifting weeks later. Raising routes into
    `_enrich_memory`'s existing retry path, which leaves the memories in
    `enrichment_status: "failed"` where they can be counted and queried. Loud and
    inspectable beats quiet and uniform.
    """

    async def test_wrong_dimension_raises(self):
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError, match="does not match"):
            await LocalScorer(art).score("x", [1.0, 1.0])

    async def test_error_names_both_dimensions(self):
        """'dimension mismatch' sends an operator to read source; '2 does not
        match model dimension 3' sends them to their config."""
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError) as exc:
            await LocalScorer(art).score("x", [1.0, 1.0])
        assert "2" in str(exc.value) and "3" in str(exc.value)

    async def test_missing_embedding_raises(self):
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError):
            await LocalScorer(art).score("x", None)

    async def test_empty_embedding_raises(self):
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError):
            await LocalScorer(art).score("x", [])

    async def test_non_numeric_embedding_raises(self):
        """A vector read back from Mongo can contain None if a write half-failed.
        `None * float` would raise TypeError anyway — this raises the error that
        says what is actually wrong."""
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError, match="numeric"):
            await LocalScorer(art).score("x", [1.0, None, 1.0])

    async def test_lexical_artifact_does_not_require_an_embedding(self):
        """Only `embedding_linear` needs a vector. A lexical artifact scoring
        text must not be dragged into this."""
        got = await LocalScorer(_lexical_artifact(intercept=0.3)).score("x", None)
        assert got == pytest.approx(logistic(0.3))


class TestLocalScorerLexicalPath:
    async def test_ignores_embedding_entirely(self):
        """A lexical artifact must not read the embedding — its coefficients are
        indexed by feature, and a 1536-vector would silently be scored against
        the wrong weights."""
        coeffs = [1.0] + [0.0] * (LEXICAL_FEATURE_COUNT - 1)
        scorer = LocalScorer(_lexical_artifact(coeffs))
        content = "a" * 1000
        with_emb = await scorer.score(content, [9.0] * 1536)
        without = await scorer.score(content, None)
        assert with_emb == without == pytest.approx(logistic(1.0))

    async def test_uses_content(self):
        coeffs = [1.0] + [0.0] * (LEXICAL_FEATURE_COUNT - 1)
        scorer = LocalScorer(_lexical_artifact(coeffs))
        short = await scorer.score("a" * 100, None)
        long = await scorer.score("a" * 1000, None)
        assert long > short


class TestClamping:
    """REQ-E-165. Asserted independently of the maths: a trained artifact with a
    large negative intercept is a plausible accident, and its consequence would
    be silent deletion of every memory it scores."""

    async def test_never_below_floor(self):
        art = _embedding_artifact(coefficients=(0.0,) * 3, intercept=-1000.0)
        assert await LocalScorer(art).score("x", [1.0, 1.0, 1.0]) == MIN_IMPORTANCE

    async def test_never_above_ceiling(self):
        art = _embedding_artifact(coefficients=(0.0,) * 3, intercept=1000.0)
        assert await LocalScorer(art).score("x", [1.0, 1.0, 1.0]) == MAX_IMPORTANCE

    async def test_floor_is_not_zero(self):
        assert MIN_IMPORTANCE == 0.1

    @pytest.mark.parametrize("intercept", [-50.0, -5.0, 0.0, 5.0, 50.0])
    async def test_always_in_range(self, intercept):
        art = _embedding_artifact(coefficients=(3.0,) * 3, intercept=intercept)
        got = await LocalScorer(art).score("x", [10.0, -10.0, 7.0])
        assert MIN_IMPORTANCE <= got <= MAX_IMPORTANCE


class TestSubstitutability:
    """Both implementations must satisfy the same caller. If this passes for one
    and not the other, the worker cannot treat them as interchangeable."""

    @pytest.mark.parametrize("kind", ["llm", "local"])
    async def test_same_call_shape(self, kind):
        if kind == "llm":
            llm = create_autospec(LLMProvider, instance=True)
            llm.assess_importance.return_value = 0.7
            scorer: ImportanceScorer = LLMScorer(llm)
        else:
            scorer = LocalScorer(_lexical_artifact(intercept=0.4))
        got = await scorer.score(
            "some content", [0.1] * LEXICAL_FEATURE_COUNT,
            tags=["work"], message_type="human",
        )
        assert MIN_IMPORTANCE <= got <= MAX_IMPORTANCE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_importance_scorer.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'ImportanceScorer'`

- [ ] **Step 3: Implement**

Append to `agent_memory/services/importance.py`:

```python
# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def logistic(x: float) -> float:
    """Squash a real number into ``(0, 1)``.

    Split around zero because the naive ``1 / (1 + exp(-x))`` raises
    ``OverflowError`` for ``x`` below about -710. That matters: a trained artifact
    can saturate on an outlier embedding, and an exception here propagates into
    ``EnrichmentWorker._enrich_memory``, which retries the memory to ``failed``.
    A saturating model should return 0.0-ish, not break enrichment.
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clamp(value: float) -> float:
    """Constrain to ``[MIN_IMPORTANCE, MAX_IMPORTANCE]``.

    Applied to every score from every implementation, deliberately outside the
    model maths. The floor is the reason: `logistic` already returns a positive
    number, but a coefficient set with a large negative intercept produces
    values below 0.1, and 0.1 is where `forgetting_score_threshold` sits. A
    scorer must not be able to order a deletion.
    """
    return max(MIN_IMPORTANCE, min(MAX_IMPORTANCE, value))


@runtime_checkable
class ImportanceScorer(Protocol):
    """Produces a memory's ``importance`` in ``[0.1, 1.0]``.

    The wider-than-needed signature is what makes the implementations
    substitutable: ``LLMScorer`` ignores everything but ``content``, and
    ``LocalScorer`` ignores ``content`` when it has a usable embedding. A caller
    passes everything it has and neither implementation can tell it is being
    over-served.
    """

    async def score(
        self,
        content: str,
        embedding: list[float] | None = None,
        *,
        tags: list[str] | None = None,
        message_type: str | None = None,
    ) -> float:
        """Return an importance score in ``[0.1, 1.0]``."""
        ...


class LLMScorer:
    """Today's behaviour, unchanged: one LLM round trip per memory.

    Owns the prompt lookup that used to sit inline in
    ``EnrichmentWorker._process_standard_enrichment``. It has to live here rather
    than in the worker, because the worker cannot keep an ``if prompt`` branch
    around a call whose other implementation has no prompt.
    """

    def __init__(self, llm, prompt_getter=None) -> None:
        self._llm = llm
        # ``async (name) -> str | None``. The worker passes its ``_get_prompt``.
        self._prompt_getter = prompt_getter

    async def score(
        self,
        content: str,
        embedding: list[float] | None = None,
        *,
        tags: list[str] | None = None,
        message_type: str | None = None,
    ) -> float:
        prompt = None
        if self._prompt_getter is not None:
            prompt = await self._prompt_getter("importance_assessment")
        # Omit the kwarg entirely rather than passing None: providers apply their
        # own default template when the argument is absent, and passing None
        # would make a customized-prompt deployment indistinguishable from a
        # default one at the provider boundary.
        if prompt:
            return await self._llm.assess_importance(content, prompt=prompt)
        return await self._llm.assess_importance(content)


class LocalScorer:
    """A logistic linear model evaluated in-process. No network, no LLM.

    Viable only because the embedding already exists by the time importance is
    scored — ``EnrichmentWorker`` holds ``memory["embedding"]`` to pass to
    ``evolve_memory`` — so this is a dot product over a vector someone else paid
    for, not an encoder forward pass.

    An ``embedding_linear`` artifact handed an unusable vector **raises**. Two
    quieter options were available and both are worse. Scoring the overlapping
    prefix returns a number in the right range that means nothing. Returning the
    intercept returns the *same* number for every memory in the store, so
    consolidation forgets and promotes by coin flip. Both are invisible until
    recall quality drifts weeks later; raising routes into
    ``_enrich_memory``'s existing retry path, which leaves the affected memories
    in ``enrichment_status: "failed"`` where an operator can count them.
    """

    def __init__(self, artifact: Artifact) -> None:
        self.artifact = artifact

    async def score(
        self,
        content: str,
        embedding: list[float] | None = None,
        *,
        tags: list[str] | None = None,
        message_type: str | None = None,
    ) -> float:
        if self.artifact.kind == "lexical":
            features = lexical_features(content)
        else:
            features = self._validated_embedding(embedding)

        total = self.artifact.intercept
        for c, f in zip(self.artifact.coefficients, features):
            total += c * f
        return _clamp(logistic(total))

    def _validated_embedding(self, embedding) -> list[float]:
        """The embedding, or ``ConfigError`` naming what is wrong with it."""
        expected = len(self.artifact.coefficients)
        actual = 0 if not embedding else len(embedding)
        if actual != expected:
            raise ConfigError(
                f"Local importance scorer received an embedding of length "
                f"{actual}, which does not match model dimension {expected} "
                f"(artifact: provider={self.artifact.provider} "
                f"model={self.artifact.model}). Check that IMPORTANCE_MODEL_PATH "
                f"corresponds to EMBEDDING_MODEL / EMBEDDING_DIMENSION."
            )
        # A vector read back from MongoDB can contain None if a write half-failed.
        # `None * float` would raise TypeError deep in the loop; this raises the
        # error that says what is actually wrong.
        for i, v in enumerate(embedding):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ConfigError(
                    f"Local importance scorer received a non-numeric embedding "
                    f"value at index {i}: {v!r}"
                )
        return embedding
```

`ConfigError` is already imported at the top of the module for the loader.

Module imports become:

```python
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_importance_scorer.py -q`
Expected: PASS — 34 tests (23 functions, three parametrized).

Then the whole new module's suite:
Run: `.venv/bin/python -m pytest tests/unit/test_importance_artifact.py tests/unit/test_importance_features.py tests/unit/test_importance_scorer.py -q`

- [ ] **Step 5: Commit**

```bash
git add agent_memory/services/importance.py tests/unit/test_importance_scorer.py
git commit -m "Add ImportanceScorer protocol with LLM and local implementations

LLMScorer is today's call, including the prompt lookup moved out of the
enrichment worker — the worker cannot keep an 'if prompt' branch around
a call whose other implementation has no prompt.

LocalScorer refuses a mismatched embedding rather than scoring the
overlapping prefix, and warns once per instance rather than once per
memory. Clamping to [0.1, 1.0] is applied outside the model maths so a
badly trained artifact cannot emit 0.0, which consolidation reads as a
deletion order."
```

---

## Task 4: Config surface

**Files:**
- Modify: `agent_memory/core/config.py`
- Test: `tests/unit/test_config.py`

**Requirements:** REQ-E-162 (config surface; `"llm"` default; invalid value raises at construction).

**Interfaces:**
- Produces two `MCPConfig` fields and one validator arm:
  - `importance_scorer: str = "llm"` — `"llm"` or `"local"`.
  - `importance_model_path: str | None = None` — explicit artifact path; when unset with `"local"`, `ProviderManager` auto-selects a bundled artifact (Task 5).
- Consumes: nothing new.

The validator rejects an unknown `importance_scorer` at construction rather than at first score. `_auth_must_not_fail_open` (`config.py:159`) is the precedent: a config mistake that surfaces as a runtime fallback is the failure mode this codebase already decided to refuse. A typo'd `IMPORTANCE_SCORER=locl` must not start the server on the LLM path while the operator believes they turned the LLM off — the bill is the only symptom.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_config.py`. First extend the `_clean_env` fixture's tuple with the new variable names:

```python
            "ENRICHMENT_BATCH_SIZE", "LLM_MODEL_ID",
            "IMPORTANCE_SCORER", "IMPORTANCE_MODEL_PATH",
```

Then append this class:

```python
class TestImportanceScorerConfig:
    """REQ-E-162. Config surface for the pluggable scorer."""

    def test_defaults_to_llm(self):
        """The safety property of the whole feature: an existing install that
        upgrades and changes nothing keeps making the same LLM call."""
        assert _make_config().importance_scorer == "llm"

    def test_model_path_defaults_to_none(self):
        """None means 'auto-select a bundled artifact', not 'no model'."""
        assert _make_config().importance_model_path is None

    def test_accepts_local(self):
        assert _make_config(importance_scorer="local").importance_scorer == "local"

    def test_normalizes_case_and_whitespace(self):
        """`IMPORTANCE_SCORER=Local ` from a hand-edited .env is a correct
        intent, not a typo."""
        assert _make_config(importance_scorer=" Local ").importance_scorer == "local"

    def test_rejects_unknown_scorer(self):
        """A typo must not silently leave the LLM path enabled. An operator who
        set this flag to save money would see no symptom but the bill."""
        with pytest.raises(ValueError, match="IMPORTANCE_SCORER"):
            _make_config(importance_scorer="locl")

    def test_rejection_names_the_valid_values(self):
        with pytest.raises(ValueError, match="local"):
            _make_config(importance_scorer="sklearn")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("IMPORTANCE_SCORER", "local")
        monkeypatch.setenv("IMPORTANCE_MODEL_PATH", "/models/x.json")
        config = MCPConfig(mongodb_connection_string="mongodb://localhost:27017")
        assert config.importance_scorer == "local"
        assert config.importance_model_path == "/models/x.json"

    def test_model_path_with_llm_scorer_is_allowed(self):
        """Not an error — an operator staging a model before flipping the switch
        is a reasonable sequence, and refusing it would make the rollout
        two-steps-at-once."""
        config = _make_config(
            importance_scorer="llm", importance_model_path="/models/x.json"
        )
        assert config.importance_scorer == "llm"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py -k Importance -q`
Expected: FAIL — `AttributeError: 'MCPConfig' object has no attribute 'importance_scorer'`

- [ ] **Step 3: Implement**

In `agent_memory/core/config.py`, after the enrichment block (`config.py:96-99`):

```python
    # Importance Scoring
    # "llm" (default) makes one LLM call per long-term memory. "local" evaluates
    # a small logistic model over the embedding that already exists by then —
    # microseconds, no network, no tokens. Default stays "llm" so an upgrade
    # changes nothing.
    importance_scorer: str = "llm"
    # Path to a JSON coefficient artifact. None means auto-select the bundled
    # artifact matching the configured embedder, falling back to the lexical one.
    importance_model_path: str | None = None
```

Add a validator arm. Keep it separate from `_auth_must_not_fail_open` — that method's docstring is specifically about authentication, and folding an unrelated check into it would make both harder to read:

```python
    _IMPORTANCE_SCORERS: ClassVar[tuple[str, ...]] = ("llm", "local")

    @model_validator(mode="after")
    def _importance_scorer_must_be_known(self):
        """Refuse an unrecognized scorer rather than falling back to the LLM.

        Same reasoning as ``_auth_must_not_fail_open``: the operator has stated an
        intent, and a typo that silently keeps the old path produces no symptom
        except the invoice. ``IMPORTANCE_SCORER=locl`` would run every enrichment
        through the LLM while the deployment reports healthy.

        Normalizes case and surrounding whitespace first — ``IMPORTANCE_SCORER=Local``
        in a hand-edited ``.env`` is a correct intent expressed slightly wrong,
        and there is nothing to protect by rejecting it.
        """
        normalized = (self.importance_scorer or "").strip().lower()
        if normalized not in self._IMPORTANCE_SCORERS:
            raise ValueError(
                f"IMPORTANCE_SCORER={self.importance_scorer!r} is not recognized. "
                f"Valid values: {', '.join(self._IMPORTANCE_SCORERS)}. Refusing to "
                "start on a scorer the operator did not ask for."
            )
        if normalized != self.importance_scorer:
            # `model_validator(mode="after")` gets a constructed model, so plain
            # attribute assignment is the way to normalize. Pydantic v2 revalidates
            # on assignment only when `validate_assignment` is set, which it is
            # not here — so this cannot recurse.
            self.importance_scorer = normalized
        return self
```

`ClassVar` is the annotation that keeps `_IMPORTANCE_SCORERS` out of the model's fields — an un-annotated class attribute would work too, but Pydantic v2 errors on a non-`ClassVar` attribute whose name starts with an underscore in some configurations, and `ClassVar` states the intent. Add `from typing import ClassVar` to the imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_config.py -q`
Expected: PASS — the 8 new tests plus every existing one.

- [ ] **Step 5: Commit**

```bash
git add agent_memory/core/config.py tests/unit/test_config.py
git commit -m "Add importance_scorer and importance_model_path config

Defaults to 'llm' so an upgrade is a no-op. An unrecognized value is
refused at construction, following _auth_must_not_fail_open: a typo'd
IMPORTANCE_SCORER that silently keeps the LLM path has no symptom except
the bill."
```

---

## Task 5: Scorer selection in `ProviderManager`

**Files:**
- Modify: `agent_memory/providers/manager.py`
- Test: `tests/unit/test_importance_selection.py`

**Requirements:** REQ-E-164 (selection by `(provider, model, dimension)` with a documented lexical fallback), REQ-E-171.

**Interfaces:**
- Consumes: `MCPConfig.importance_scorer`, `.importance_model_path`, `.embedding_provider`, `.embedding_model`, `.embedding_dimension` (Task 4 + existing); `LLMScorer`, `LocalScorer`, `load_artifact`, `bundled_artifact_path` (Tasks 1, 3).
- Produces: `ProviderManager.scorer: ImportanceScorer`, and `def select_artifact_name(config) -> str` (module-level, so the test can check selection without constructing providers).

**The construction-order hazard.** `self.scorer` must be assigned **after** `self.embedding`. `_create_embedding_provider`'s Voyage arm mutates the config it is handed (`manager.py:88-95`): it overwrites `embedding_model` from `voyage_model` and rewrites `embedding_dimension` to the model's native size. A `scorer` built before that runs would see `embedding_model="amazon.titan-embed-text-v1"` and `embedding_dimension=1536` on a Voyage deployment, match no artifact, and fall back to lexical — a silent quality regression on the one provider where the config is not self-describing. The test below asserts the ordering directly rather than trusting the line order to survive a refactor.

Selection when `importance_model_path` is unset:

| `embedding_provider` / `embedding_model` / dim | Artifact |
|---|---|
| `bedrock` / `amazon.titan-embed-text-v1` / 1536 | `titan-1536` |
| `voyage` / `voyage-3` / 1024 | `voyage-3-1024` |
| anything else | `lexical` |

Match on the **triple**, not the provider. `voyage-3-lite` emits 512 dimensions and would load 1024 coefficients otherwise — the exact mismatch `LocalScorer` is built to refuse, but refusing at load time with a named artifact beats refusing per-call.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_importance_selection.py`:

```python
"""Scorer selection and injection. REQ-E-162, REQ-E-164, REQ-E-171.

The test that matters most here is `test_scorer_built_after_embedding_provider`.
`_create_embedding_provider` mutates the config for Voyage — overwriting
`embedding_model` and `embedding_dimension` — so a scorer constructed before it
would read Titan's defaults on a Voyage deployment, match no artifact, and fall
back to lexical. Nothing errors; the scores just get worse.
"""

import os
from unittest.mock import patch

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.exceptions import ConfigError
from agent_memory.providers.manager import ProviderManager, select_artifact_name
from agent_memory.services.importance import (
    ImportanceScorer,
    LLMScorer,
    LocalScorer,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.upper().startswith(("IMPORTANCE_", "EMBEDDING_", "VOYAGE_", "AWS_")):
            monkeypatch.delenv(key, raising=False)


def _config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


class TestSelectArtifactName:
    def test_bedrock_titan_1536(self):
        assert select_artifact_name(_config()) == "titan-1536"

    def test_voyage_3_1024(self):
        config = _config(
            embedding_provider="voyage",
            embedding_model="voyage-3",
            embedding_dimension=1024,
        )
        assert select_artifact_name(config) == "voyage-3-1024"

    def test_unknown_model_falls_back_to_lexical(self):
        config = _config(embedding_model="some-new-embedder")
        assert select_artifact_name(config) == "lexical"

    def test_right_model_wrong_dimension_falls_back_to_lexical(self):
        """A 512-dim voyage-3-lite must not load 1024 coefficients."""
        config = _config(
            embedding_provider="voyage",
            embedding_model="voyage-3-lite",
            embedding_dimension=512,
        )
        assert select_artifact_name(config) == "lexical"

    def test_titan_at_wrong_dimension_falls_back(self):
        config = _config(embedding_dimension=1024)
        assert select_artifact_name(config) == "lexical"

    def test_openai_falls_back_to_lexical(self):
        """No artifact shipped for OpenAI yet. Lexical is worse than a trained
        head and better than a constant."""
        config = _config(embedding_provider="openai", embedding_model="text-embedding-3-small")
        assert select_artifact_name(config) == "lexical"


class TestProviderManagerSelection:
    """Embedding and LLM providers are stubbed — this is about the scorer."""

    @pytest.fixture(autouse=True)
    def _stub_providers(self):
        with patch.object(
            ProviderManager, "_create_embedding_provider", return_value=object()
        ) as emb, patch.object(
            ProviderManager, "_create_llm_provider", return_value=object()
        ):
            self.emb = emb
            yield

    def test_default_config_selects_llm_scorer(self):
        manager = ProviderManager(_config())
        assert isinstance(manager.scorer, LLMScorer)

    def test_llm_scorer_wraps_the_llm_provider(self):
        manager = ProviderManager(_config())
        assert manager.scorer._llm is manager.llm

    def test_local_config_selects_local_scorer(self):
        manager = ProviderManager(_config(importance_scorer="local"))
        assert isinstance(manager.scorer, LocalScorer)

    def test_local_scorer_loads_the_selected_bundled_artifact(self):
        manager = ProviderManager(_config(importance_scorer="local"))
        assert manager.scorer.artifact.model == "amazon.titan-embed-text-v1"

    def test_local_scorer_honors_explicit_path(self, tmp_path):
        import json
        from agent_memory.services.importance import (
            LEXICAL_FEATURE_COUNT,
            SCHEMA_VERSION,
        )

        path = tmp_path / "custom.json"
        path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "kind": "lexical",
            "coefficients": [0.5] * LEXICAL_FEATURE_COUNT,
            "intercept": 0.1,
            "squash": "logistic",
            "training": {},
        }))
        manager = ProviderManager(
            _config(importance_scorer="local", importance_model_path=str(path))
        )
        assert manager.scorer.artifact.coefficients[0] == 0.5

    def test_missing_explicit_path_raises(self, tmp_path):
        """Refuse to start rather than silently fall back. The operator named a
        file; a typo'd path that quietly loads different coefficients is worse
        than a startup failure."""
        with pytest.raises(ConfigError, match="not found"):
            ProviderManager(_config(
                importance_scorer="local",
                importance_model_path=str(tmp_path / "absent.json"),
            ))

    def test_scorer_satisfies_the_protocol(self):
        for scorer_kind in ("llm", "local"):
            manager = ProviderManager(_config(importance_scorer=scorer_kind))
            assert isinstance(manager.scorer, ImportanceScorer)


class TestConstructionOrder:
    def test_scorer_built_after_embedding_provider(self):
        """`_create_embedding_provider` rewrites `embedding_model` and
        `embedding_dimension` for Voyage. A scorer built first reads Titan's
        defaults and silently selects the lexical artifact."""
        seen = {}

        def fake_embedding(self, config):
            config.embedding_model = "voyage-3"
            config.embedding_dimension = 1024
            return object()

        def record(config):
            seen["name"] = _real_select(config)
            return seen["name"]

        from agent_memory.providers import manager as manager_mod
        _real_select = manager_mod.select_artifact_name

        with patch.object(ProviderManager, "_create_embedding_provider", fake_embedding), \
             patch.object(ProviderManager, "_create_llm_provider", lambda self, c: object()), \
             patch.object(manager_mod, "select_artifact_name", record):
            ProviderManager(_config(
                embedding_provider="voyage", importance_scorer="local"
            ))

        assert seen["name"] == "voyage-3-1024", (
            "scorer selection ran before the embedding provider rewrote the "
            "config — a Voyage deployment would silently get lexical scoring"
        )

    def test_voyage_end_to_end_selects_voyage_artifact(self):
        """The real integration, with only the network-touching provider stubbed.
        Config defaults are Titan's; only `_create_embedding_provider` knows
        otherwise."""
        from agent_memory.providers.voyage import VoyageEmbeddingProvider

        with patch.object(VoyageEmbeddingProvider, "__init__", lambda self, c: None), \
             patch.object(ProviderManager, "_create_llm_provider", lambda self, c: object()):
            manager = ProviderManager(_config(
                embedding_provider="voyage",
                voyage_api_key="test-key",
                importance_scorer="local",
            ))
        assert manager.scorer.artifact.model == "voyage-3"
        assert manager.scorer.artifact.dimension == 1024
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_importance_selection.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'select_artifact_name'`

- [ ] **Step 3: Implement**

In `agent_memory/providers/manager.py`, add to the imports:

```python
from agent_memory.services.importance import (
    ImportanceScorer,
    LLMScorer,
    LocalScorer,
    bundled_artifact_path,
    load_artifact,
)
```

`services.importance` imports nothing from `providers`, so there is no cycle. Confirm with `.venv/bin/python -c "import agent_memory.providers.manager"`.

Add the selection table and function after `known_embedding_dimension`:

```python
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
    key = (config.embedding_provider, config.embedding_model, config.embedding_dimension)
    return _BUNDLED_ARTIFACTS.get(key, _FALLBACK_ARTIFACT)
```

Change `__init__` and add `_create_scorer`:

```python
    def __init__(self, config: MCPConfig) -> None:
        self.embedding: EmbeddingProvider = self._create_embedding_provider(config)
        self.llm: LLMProvider = self._create_llm_provider(config)
        # Last, and not by accident. The Voyage arm of
        # `_create_embedding_provider` mutates `config.embedding_model` and
        # `config.embedding_dimension`; selecting an artifact before that runs
        # reads Titan's defaults, matches nothing, and quietly downgrades a
        # Voyage deployment to lexical scoring. See
        # `test_scorer_built_after_embedding_provider`.
        self.scorer: ImportanceScorer = self._create_scorer(config)

    def _create_scorer(self, config: MCPConfig) -> ImportanceScorer:
        """Build the importance scorer named by config.

        ``config.importance_scorer`` is already validated and normalized by
        ``MCPConfig._importance_scorer_must_be_known``, so the ``case _`` arm is
        unreachable in practice — kept because a future field change should fail
        loudly here rather than return None.

        The prompt getter is deliberately absent: ``ProviderManager`` has no
        prompt library. ``AsyncMemory._maybe_start_workers`` attaches the
        worker's ``_get_prompt`` when it injects the scorer, which is the only
        place where both exist.
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
                        name, artifact.kind,
                        (artifact.training or {}).get("n_samples", "unknown"),
                    )
                return LocalScorer(artifact)
            case _:
                raise ValueError(
                    f"Unknown importance scorer: {config.importance_scorer}"
                )
```

`manager.py` has no logger today — add `import logging` and `logger = logging.getLogger(__name__)` below the imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_importance_selection.py -q`
Expected: PASS — 15 tests.

Check for no import cycle and no regression in the existing manager tests:
```bash
.venv/bin/python -c "import agent_memory.providers.manager; print('ok')"
.venv/bin/python -m pytest tests/unit/test_provider_manager_extras.py tests/unit/test_providers.py -q
```

- [ ] **Step 5: Commit**

```bash
git add agent_memory/providers/manager.py tests/unit/test_importance_selection.py
git commit -m "Select the importance scorer in ProviderManager

self.scorer is assigned last, after the embedding provider, because the
Voyage arm of _create_embedding_provider rewrites embedding_model and
embedding_dimension on the config. Building the scorer first reads
Titan's defaults, matches no artifact, and silently downgrades a Voyage
deployment to lexical scoring — a test asserts the ordering rather than
trusting line order to survive a refactor.

Artifacts are keyed on (provider, model, dimension): voyage-3 is 1024
and voyage-3-lite is 512, so keying on provider alone would load the
wrong coefficient count."
```

---

## Task 6: Wire the scorer into `EnrichmentWorker`

**Files:**
- Modify: `agent_memory/services/enrichment.py`
- Modify: `agent_memory/memory.py`
- Modify: `tests/unit/test_enrichment.py`

**Requirements:** REQ-E-160 (behaviour preserved on the default path), REQ-E-172 (default-path regression coverage).

**Interfaces:**
- Consumes: `ImportanceScorer`, `LLMScorer` (Task 3); `ProviderManager.scorer` (Task 5).
- Produces: `EnrichmentWorker.__init__(..., prompt_library=None, scorer=None)` — `scorer` keyword-only-in-practice and **optional**, defaulting to `LLMScorer(providers.llm, prompt_getter=self._get_prompt)`.

`scorer=None` defaulting to an `LLMScorer` is what keeps the twenty existing four-positional-argument `EnrichmentWorker(...)` constructions in the test suite working unchanged. That is not just convenience: those twenty call sites are the regression suite for the LLM path, and rewriting them all as part of this change would mean the safety property ("an upgrade changes nothing") is asserted only by tests I edited in the same commit.

The importance branch collapses from six lines to one:

```python
importance = await self.scorer.score(
    memory["content"],
    memory.get("embedding"),
    tags=memory.get("tags"),
    message_type=memory.get("message_type"),
)
```

`memory.get("embedding")` rather than `memory["embedding"]`: the line below it uses `memory["embedding"]` and would raise on a malformed document anyway, but the scorer must not be the thing that raises — `LocalScorer` already handles a missing embedding by falling back to its prior, and a `KeyError` here retries the memory to `failed`.

`_get_prompt` stays on the worker; `_summarize` and `_process_merge` still use it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_enrichment.py`. First fix the `_make_providers` docstring, which overstates what it does — verified on Python 3.11.13 that `AsyncMock(spec=LLMProvider.assess_importance)` accepts bogus keywords and extra positional arguments:

```python
def _make_providers():
    """Mock providers whose LLM presents the real method set.

    `spec=LLMProvider` catches a *renamed or removed* method: the mock raises
    AttributeError for anything not on the ABC. It does **not** enforce
    signatures — verified on 3.11.13, `AsyncMock(spec=LLMProvider.assess_importance)`
    accepts bogus keywords and extra positional args. Signature drift is caught
    by `test_provider_prompt_contract.py`, which inspects the real
    implementations; that is the test that would have caught the `prompt=`
    TypeError on OpenAI and Anthropic.
    """
```

Then append:

```python
class TestScorerInjection:
    """REQ-E-160, REQ-E-172. The worker delegates importance to a scorer."""

    def test_defaults_to_an_llm_scorer(self):
        """Every existing call site omits `scorer`. Defaulting here is what makes
        'an upgrade changes nothing' true without editing twenty constructions."""
        from agent_memory.services.importance import LLMScorer

        worker = EnrichmentWorker(
            MagicMock(), _make_config(), _make_providers(), _make_memory_service()
        )
        assert isinstance(worker.scorer, LLMScorer)

    def test_default_scorer_wraps_the_configured_llm(self):
        providers = _make_providers()
        worker = EnrichmentWorker(
            MagicMock(), _make_config(), providers, _make_memory_service()
        )
        assert worker.scorer._llm is providers.llm

    def test_default_scorer_uses_the_worker_prompt_getter(self):
        """The prompt library moved behind the scorer. If it is not wired, a
        deployment with a customized importance prompt silently reverts to the
        provider's built-in template — same scores-look-fine failure mode."""
        worker = EnrichmentWorker(
            MagicMock(), _make_config(), _make_providers(), _make_memory_service()
        )
        assert worker.scorer._prompt_getter == worker._get_prompt

    async def test_injected_scorer_is_used(self):
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.return_value = 0.42
        col = MagicMock()
        col.update_one = AsyncMock()
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service(),
            scorer=scorer,
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        assert col.update_one.call_args[0][1]["$set"]["importance"] == 0.42

    async def test_injected_scorer_receives_content_and_embedding(self):
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.return_value = 0.42
        col = MagicMock()
        col.update_one = AsyncMock()
        memory = _make_pending_memory()
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service(),
            scorer=scorer,
        )
        await worker._process_standard_enrichment(memory)
        args, kwargs = scorer.score.call_args
        assert args[0] == memory["content"]
        assert args[1] == memory["embedding"]

    async def test_injected_scorer_replaces_the_llm_importance_call(self):
        """The reason the feature exists. If `assess_importance` still fires, the
        local path costs a token round trip and saves nothing."""
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.return_value = 0.42
        providers = _make_providers()
        col = MagicMock()
        col.update_one = AsyncMock()
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service(), scorer=scorer,
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        providers.llm.assess_importance.assert_not_awaited()

    async def test_summary_still_uses_the_llm(self):
        """Only scoring is swappable. Summarization is generation and stays on the
        LLM — a linear model cannot write a summary."""
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.return_value = 0.42
        providers = _make_providers()
        col = MagicMock()
        col.update_one = AsyncMock()
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service(), scorer=scorer,
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        providers.llm.generate_summary.assert_awaited()

    async def test_scorer_failure_leaves_the_memory_retryable(self):
        """A scorer that raises must go down the existing retry path rather than
        writing a wrong importance."""
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.side_effect = RuntimeError("artifact went away")
        col = MagicMock()
        col.update_one = AsyncMock()
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service(),
            scorer=scorer,
        )
        await worker._enrich_memory(_make_pending_memory())
        update = col.update_one.call_args[0][1]["$set"]
        assert update["enrichment_status"] == "pending"
        assert update["enrichment_retries"] == 1


class TestLLMPathUnchanged:
    """The safety property, asserted against the default construction."""

    async def test_still_calls_assess_importance_with_the_library_prompt(self):
        col = MagicMock()
        col.update_one = AsyncMock()
        providers = _make_providers()
        library = MagicMock()
        library.get_prompt = AsyncMock(return_value="Rate: {content}")
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service(),
            prompt_library=library,
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        providers.llm.assess_importance.assert_awaited_once_with(
            LONG_CONTENT, prompt="Rate: {content}"
        )

    async def test_omits_prompt_when_the_library_has_none(self):
        col = MagicMock()
        col.update_one = AsyncMock()
        providers = _make_providers()
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service()
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        providers.llm.assess_importance.assert_awaited_once_with(LONG_CONTENT)
```

Add `create_autospec` to the `unittest.mock` import at the top of the file.

Also add to `TestLifecycle` in `tests/unit/test_memory_facade.py` — the injection at `memory.py:170` is otherwise untested. `_build_for_lifecycle` (`test_memory_facade.py:531`) already provides a `MagicMock()` `providers`, so `providers.scorer` auto-creates:

```python
    async def test_enrichment_worker_receives_the_provider_scorer(self):
        """`_maybe_start_workers` must pass `providers.scorer` through.

        Without it the worker builds its own LLMScorer, and
        IMPORTANCE_SCORER=local becomes a silent no-op: the config reads as
        applied, startup logs the artifact it loaded, and every enrichment still
        bills a token. Nothing anywhere reports the discrepancy.
        """
        m = _build_for_lifecycle(workers_in_process=True)
        with patch("agent_memory.services.enrichment.EnrichmentWorker") as worker_cls:
            worker_cls.return_value.run = AsyncMock()
            await m._maybe_start_workers()
        assert worker_cls.call_args.kwargs["scorer"] is m.providers.scorer
        for t in m._workers:
            t.cancel()
```

`_maybe_start_workers` imports `EnrichmentWorker` inside the function body (`memory.py:167`), so patching the name on the `enrichment` module is what takes effect — patching `agent_memory.memory.EnrichmentWorker` would miss, because the name does not exist there until the call runs.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_enrichment.py -k "Scorer or LLMPathUnchanged" -q`
Expected: FAIL — `AttributeError: 'EnrichmentWorker' object has no attribute 'scorer'`

- [ ] **Step 3: Implement**

In `agent_memory/services/enrichment.py`, change `__init__`:

```python
    def __init__(
        self,
        memories_collection,
        config: MCPConfig,
        providers,
        memory_service,
        prompt_library=None,
        scorer=None,
    ) -> None:
        self.memories = memories_collection
        self.config = config
        self.providers = providers
        self.memory_service = memory_service
        self.prompt_library = prompt_library
        # Default to the LLM path so every existing construction — including the
        # twenty in the test suite that pass four positional arguments — keeps
        # today's behaviour exactly. Those call sites are the regression suite
        # for "an upgrade changes nothing"; rewriting them here would mean the
        # property is only asserted by tests changed in the same commit.
        self.scorer = scorer or LLMScorer(
            providers.llm, prompt_getter=self._get_prompt
        )
        self._semaphore = asyncio.Semaphore(config.enrichment_concurrency)
        self._running = False
```

Import at the top: `from agent_memory.services.importance import LLMScorer`.

Replace the importance block in `_process_standard_enrichment`:

```python
        # One call, whichever scorer is configured. The prompt lookup that used
        # to branch here now lives in `LLMScorer` — the branch could not stay,
        # because the local scorer has no prompt.
        importance = await self.scorer.score(
            memory["content"],
            memory.get("embedding"),
            tags=memory.get("tags"),
            message_type=memory.get("message_type"),
        )
```

`.get("embedding")` rather than `["embedding"]`: `LocalScorer` handles a missing embedding by falling back to its prior, and a `KeyError` raised here would retry the memory to `failed` instead.

In `agent_memory/memory.py:170`:

```python
        enrichment = EnrichmentWorker(
            memories, self.config, self.providers, self.memory_service,
            prompt_library=self.prompt_library,
            # Built by ProviderManager from config. Omitting this is a silent
            # no-op for IMPORTANCE_SCORER=local: the worker would construct its
            # own LLMScorer and every enrichment would still bill a token.
            scorer=self.providers.scorer,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/test_enrichment.py -q
.venv/bin/python -m pytest tests/unit/test_memory_facade.py tests/unit/test_refusals_and_mutations.py tests/unit/test_provider_message_shape.py -q
```
Expected: PASS, including the 20 untouched worker constructions.

Then the full suite — this is the first task that changes an existing hot path:
```bash
.venv/bin/python -m pytest tests/unit -q
```

- [ ] **Step 5: Commit**

```bash
git add agent_memory/services/enrichment.py agent_memory/memory.py \
        tests/unit/test_enrichment.py tests/unit/test_memory_facade.py
git commit -m "Route enrichment importance through the scorer seam

The six-line prompt branch collapses to one scorer.score() call; the
prompt lookup moved into LLMScorer, which is the only implementation that
has a prompt.

scorer defaults to LLMScorer so the twenty existing four-positional-arg
constructions keep asserting the old behaviour unchanged — they are the
regression suite for 'an upgrade is a no-op', and rewriting them here
would mean that property is only checked by tests edited in the same
commit.

Also corrects _make_providers' docstring: spec=LLMProvider catches a
renamed method, not a changed signature. Verified on 3.11.13 that
AsyncMock(spec=...) accepts bogus kwargs; signature drift is covered by
test_provider_prompt_contract.py."
```

---

## Task 7: Packaging — ship the artifacts, add the `training` extra

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/unit/test_packaging.py`
- Modify: `CHANGELOG.md`, `README.md`

**Requirements:** REQ-E-161 (no new runtime dependency, enforced), REQ-E-163 (artifacts ship).

**Interfaces:**
- Consumes: `agent_memory/data/importance/*.json` (Task 1).
- Produces: a `training` optional-dependency group; version `4.2.0`; artifacts present in wheel and sdist.

`packages = ["agent_memory"]` makes Hatchling include non-Python files under the package by default, so the artifacts probably already ship. "Probably" is the problem: a wheel missing them turns `IMPORTANCE_SCORER=local` into a `ConfigError` at startup for every installed user, and it is invisible from a source checkout because the files are right there on disk. So the test builds a real wheel and looks inside.

The `training` extra must not appear in `all`. `all` is what a user installs to get every provider; pulling scikit-learn, pandas, and `datasets` into that is roughly 200 MB of transitive weight for a feature almost no installer will use. Same reasoning as the existing `demo` extra's comment.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_packaging.py`, inside `TestPyprojectToml`:

```python
    def test_version_is_at_least_4_2(self):
        """The scorer seam is additive, so a minor bump."""
        major, minor = self.data["project"]["version"].split(".")[:2]
        assert (int(major), int(minor)) >= (4, 2)

    def test_training_extra_exists(self):
        opt = self.data["project"]["optional-dependencies"]
        assert "training" in opt
        names = [d.split(">")[0].split("=")[0].split("[")[0] for d in opt["training"]]
        for required in ["scikit-learn", "numpy"]:
            assert required in names

    def test_training_extra_is_not_in_all(self):
        """`all` is 'every provider', not 'every dependency'. scikit-learn +
        pandas + datasets is ~200MB of transitive weight for a feature that runs
        offline and is never imported by the library."""
        all_names = [
            d.split(">")[0].split("=")[0].split("[")[0]
            for d in self.data["project"]["optional-dependencies"]["all"]
        ]
        for excluded in ["scikit-learn", "numpy", "pandas", "datasets"]:
            assert excluded not in all_names

    def test_runtime_dependencies_exclude_the_scientific_stack(self):
        """The load-bearing constraint of the whole design. If numpy lands in
        runtime deps, the pure-Python scorer stopped being necessary and someone
        should have said so out loud."""
        names = [
            d.split(">")[0].split("=")[0].split("<")[0].split("[")[0]
            for d in self.data["project"]["dependencies"]
        ]
        for excluded in ["numpy", "scipy", "scikit-learn", "pandas", "torch"]:
            assert excluded not in names

    def test_sdist_includes_the_package(self):
        """`/agent_memory` covers data/importance/. Asserted so a switch to a
        narrower include list has to notice."""
        include = self.data["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
        assert "/agent_memory" in include
```

And a new class asserting the built distributions, plus one asserting the library never imports the scientific stack:

```python
class TestImportanceArtifactsShip:
    """A wheel without the artifacts makes IMPORTANCE_SCORER=local a startup
    failure for every installed user — and it is invisible from a source
    checkout, where the files are on disk regardless. Hence a real build.
    """

    EXPECTED = {"lexical.json", "titan-1536.json", "voyage-3-1024.json"}

    def test_artifacts_exist_in_the_source_tree(self):
        found = {p.name for p in (ROOT / "agent_memory/data/importance").glob("*.json")}
        assert self.EXPECTED <= found

    def test_artifacts_are_in_the_built_wheel(self, tmp_path):
        import subprocess
        import zipfile

        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"wheel build unavailable: {result.stderr[-400:]}")

        wheels = list(tmp_path.glob("*.whl"))
        assert wheels, "no wheel produced"
        with zipfile.ZipFile(wheels[0]) as zf:
            names = {pathlib.PurePosixPath(n).name for n in zf.namelist()
                     if "data/importance" in n}
        assert self.EXPECTED <= names, f"artifacts missing from wheel: {names}"


class TestLibraryStaysDependencyFree:
    """REQ-E-161. Grep rather than import-check: an `import numpy` inside a
    function body would not fail an import of the module, and the scorer is
    exactly where someone would be tempted to add one."""

    FORBIDDEN = ("numpy", "scipy", "sklearn", "pandas", "torch")

    def test_no_scientific_imports_under_agent_memory(self):
        import re

        offenders = []
        pattern = re.compile(
            r"^\s*(?:import|from)\s+(" + "|".join(self.FORBIDDEN) + r")\b", re.M
        )
        for path in (ROOT / "agent_memory").rglob("*.py"):
            for match in pattern.finditer(path.read_text()):
                offenders.append(f"{path.relative_to(ROOT)}: {match.group(0).strip()}")
        assert not offenders, (
            "agent_memory must import no scientific stack — the local scorer is "
            "pure Python so that enabling it costs no install weight:\n"
            + "\n".join(offenders)
        )
```

No `@pytest.mark.slow` on the wheel test: `[tool.pytest.ini_options]` in this repo sets only `asyncio_mode`, so an unregistered marker would emit a warning for no benefit. The test skips on its own if `uv build` is unavailable, which is the only case where it is slow enough to matter.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_packaging.py -q`
Expected: FAIL — `KeyError: 'training'` and the 4.2 version assertion.

- [ ] **Step 3: Implement**

In `pyproject.toml`:

```toml
version = "4.2.0"
```

Add the extra after `demo`:

```toml
# Offline training for the local importance scorer (`scripts/train_importance.py`).
# Deliberately not in `all`: nothing under agent_memory/ imports any of these, the
# script runs on a developer machine or in CI, and adding ~200MB of transitive
# weight to the ordinary install would undo the reason the runtime scorer is pure
# Python.
training = [
    "scikit-learn>=1.5.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "datasets>=2.19.0",
]
```

Confirm the wheel actually carries the artifacts, rather than assuming:

```bash
uv build --wheel --out-dir /tmp/am-wheel-check
.venv/bin/python -c "
import zipfile, glob
w = glob.glob('/tmp/am-wheel-check/*.whl')[0]
print([n for n in zipfile.ZipFile(w).namelist() if 'importance' in n])
"
```

If they are absent, add to `[tool.hatch.build.targets.wheel]`:

```toml
[tool.hatch.build.targets.wheel.force-include]
"agent_memory/data" = "agent_memory/data"
```

Add a `CHANGELOG.md` entry under a new `## 4.2.0` heading, and a README section covering `IMPORTANCE_SCORER`, `IMPORTANCE_MODEL_PATH`, which artifact is selected for which embedder, and the calibration warning: a local model whose scores sit systematically below the LLM's will forget more and promote less, and the symptom appears weeks later in recall quality rather than as an error. Tell operators to compare `forget_agreement` / `promote_agreement` in the artifact's `training.metrics` before switching a production deployment.

Also update the Dockerfile if `COPY agent_memory/ ./agent_memory/` does not already carry the data dir — it does, since it copies the tree, but confirm with a build if the image is exercised in CI.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/test_packaging.py -q
.venv/bin/python -m pytest tests/unit -q
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/unit/test_packaging.py CHANGELOG.md README.md
git commit -m "Ship importance artifacts; add the training extra; 4.2.0

The wheel test builds a real wheel and looks inside. A wheel missing the
artifacts makes IMPORTANCE_SCORER=local a startup failure for every
installed user, and it is invisible from a source checkout where the
files are on disk anyway.

training is not in all: nothing under agent_memory/ imports sklearn,
numpy, pandas, or datasets, and a grep test now enforces that. Adding
~200MB of transitive weight to the ordinary install would undo the reason
the runtime scorer is pure Python."
```

---

## Task 8: The offline trainer

**Files:**
- Create: `scripts/train_importance.py`
- Create: `tests/unit/test_train_importance.py`
- Modify: `agent_memory/data/importance/*.json` (regenerated with trained coefficients)

**Requirements:** REQ-E-167 (benchmark labels), REQ-E-168 (LLM distillation), REQ-E-169 (combined bake-off, calibration weighted above ranking), REQ-E-170 (`--from-mongodb`).

**Interfaces:**
- Consumes: `SCHEMA_VERSION`, `LEXICAL_FEATURE_NAMES`, `lexical_features`, `load_artifact`, `logistic` (Tasks 1–3). Importing from `agent_memory` is correct in this direction: the script may import the library, and the library must never import the script's dependencies.
- Produces: a CLI with `--source {benchmark,llm,combined,mongodb}`, `--space {lexical,embedding}`, `--out PATH`, `--dry-run`; and these importable, test-visible functions:
  - `def derive_benchmark_labels(sessions: list[dict], questions: list[dict]) -> list[tuple[str, float]]`
  - `def label_from_mongo_document(doc: dict, *, now) -> float | None`
  - `def composite_score(metrics: dict) -> float`
  - `def build_artifact(kind, coefficients, intercept, *, embedding=None, training) -> dict`
  - `def evaluate(y_true, y_pred, *, forget_threshold=0.1, promote_threshold=0.6) -> dict`

### What is and is not tested

The trainer touches the network (HuggingFace `datasets`), an LLM, and MongoDB. None of that is unit-testable here, and pretending otherwise produces tests that assert mocks. So the tests cover the four pure functions where a mistake would be both easy and consequential:

- **`derive_benchmark_labels`** — the sparse/negative distinction. Treating every unlabeled turn as negative is the single easiest way to train a model that forgets everything, and it looks like a working trainer.
- **`label_from_mongo_document`** — returns `None` for documents that carry no signal. A document with `access_count: 0` created an hour ago is *unlabeled*, not unimportant, and scoring it 0 poisons the corpus with exactly the wrong sign.
- **`composite_score`** — calibration must outrank ranking. Asserted with a concrete pair: a model with better Spearman and a large mean offset must lose to a well-calibrated one with worse Spearman.
- **`build_artifact` / `evaluate`** — the artifact round-trips through the real `load_artifact`, so the trainer cannot emit something the runtime rejects.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_train_importance.py`:

```python
"""Pure-function contracts in the offline trainer.
REQ-E-167, REQ-E-168, REQ-E-169, REQ-E-170.

The trainer talks to HuggingFace, an LLM, and MongoDB — none of which belongs in
a unit test, and mocking all three would only assert the mocks. These tests cover
the four functions where a quiet mistake would produce a trained-looking model
that silently deletes memories:

- label derivation must not treat *unlabeled* as *negative*
- the composite metric must rank calibration above correlation
- the emitted artifact must load through the real runtime loader

`scripts/` is not a package, so the module is loaded by path.
"""

import importlib.util
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_importance.py"

pytest.importorskip("sklearn", reason="trainer tests need the `training` extra")


def _load_trainer():
    spec = importlib.util.spec_from_file_location("train_importance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


class TestBenchmarkLabelDerivation:
    """REQ-E-167. A turn cited by a later question is positive. A turn in a
    session no question draws on is negative. Everything else is *unlabeled* and
    must be dropped, not defaulted."""

    SESSIONS = [
        {
            "session_id": "s1",
            "turns": [
                {"turn_id": "t1", "content": "I'm allergic to penicillin."},
                {"turn_id": "t2", "content": "Nice weather."},
            ],
        },
        {
            "session_id": "s2",
            "turns": [{"turn_id": "t3", "content": "Anything at all."}],
        },
    ]
    QUESTIONS = [{"evidence_turn_ids": ["t1"], "evidence_session_ids": ["s1"]}]

    def test_cited_turn_is_positive(self, trainer):
        labels = dict(trainer.derive_benchmark_labels(self.SESSIONS, self.QUESTIONS))
        assert labels["I'm allergic to penicillin."] == 1.0

    def test_turn_in_an_uncited_session_is_negative(self, trainer):
        labels = dict(trainer.derive_benchmark_labels(self.SESSIONS, self.QUESTIONS))
        assert labels["Anything at all."] == 0.0

    def test_uncited_turn_in_a_cited_session_is_dropped(self, trainer):
        """The load-bearing assertion. 'Nice weather' sits in a session a
        question drew on, so we cannot tell whether it was useless or merely
        unasked-about. Labeling it 0 is how a trainer learns to forget
        everything while every metric still looks plausible."""
        labels = dict(trainer.derive_benchmark_labels(self.SESSIONS, self.QUESTIONS))
        assert "Nice weather." not in labels

    def test_no_questions_yields_no_labels(self, trainer):
        """With nothing cited, every session is 'uncited' — which would label the
        entire corpus negative. Refuse instead."""
        assert trainer.derive_benchmark_labels(self.SESSIONS, []) == []


class TestMongoLabelDerivation:
    """REQ-E-170. Labels from signals the documents already carry."""

    NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)

    def _doc(self, **kw):
        doc = {
            "created_at": self.NOW - timedelta(days=60),
            "access_count": 0,
            "memory_type": "long_term",
            "is_deleted": False,
        }
        doc.update(kw)
        return doc

    def test_frequently_accessed_is_high(self, trainer):
        label = trainer.label_from_mongo_document(
            self._doc(access_count=25), now=self.NOW
        )
        assert label is not None and label > 0.7

    def test_soft_deleted_by_consolidation_is_low(self, trainer):
        label = trainer.label_from_mongo_document(
            self._doc(is_deleted=True, deleted_reason="low_importance"), now=self.NOW
        )
        assert label is not None and label < 0.3

    def test_old_and_never_accessed_is_low(self, trainer):
        label = trainer.label_from_mongo_document(
            self._doc(access_count=0, created_at=self.NOW - timedelta(days=180)),
            now=self.NOW,
        )
        assert label is not None and label < 0.4

    def test_recent_and_unaccessed_is_unlabeled(self, trainer):
        """The one that matters. A memory created an hour ago with zero accesses
        has not had the chance to be useful. Scoring it 0 teaches the model that
        everything new is worthless — and new is when scoring happens."""
        label = trainer.label_from_mongo_document(
            self._doc(created_at=self.NOW - timedelta(hours=1)), now=self.NOW
        )
        assert label is None

    def test_labels_are_in_range(self, trainer):
        for doc in [
            self._doc(access_count=1000),
            self._doc(access_count=0, created_at=self.NOW - timedelta(days=900)),
            self._doc(is_deleted=True, deleted_reason="low_importance"),
        ]:
            label = trainer.label_from_mongo_document(doc, now=self.NOW)
            assert label is None or 0.0 <= label <= 1.0


class TestCompositeScore:
    """REQ-E-169. Calibration outranks ranking, because consolidation compares
    against absolute thresholds rather than sorting."""

    def test_calibrated_beats_better_correlated_but_offset(self, trainer):
        """A model with Spearman 0.85 and a +0.15 mean offset promotes nearly
        everything: the promotion threshold is 0.6, and shifting the whole
        distribution up by 0.15 moves a large slice of the store across it."""
        offset = {
            "spearman": 0.85, "mae": 0.20, "mean_offset": 0.15,
            "forget_agreement": 0.55, "promote_agreement": 0.50,
        }
        calibrated = {
            "spearman": 0.70, "mae": 0.08, "mean_offset": 0.01,
            "forget_agreement": 0.92, "promote_agreement": 0.90,
        }
        assert trainer.composite_score(calibrated) > trainer.composite_score(offset)

    def test_mean_offset_sign_does_not_matter(self, trainer):
        """A -0.15 offset forgets too much; +0.15 promotes too much. Both are
        equally wrong, so the metric must use the magnitude."""
        base = {"spearman": 0.8, "mae": 0.1, "forget_agreement": 0.8,
                "promote_agreement": 0.8}
        up = trainer.composite_score({**base, "mean_offset": 0.15})
        down = trainer.composite_score({**base, "mean_offset": -0.15})
        assert up == pytest.approx(down)

    def test_perfect_model_scores_highest(self, trainer):
        perfect = {"spearman": 1.0, "mae": 0.0, "mean_offset": 0.0,
                   "forget_agreement": 1.0, "promote_agreement": 1.0}
        worst = {"spearman": -1.0, "mae": 1.0, "mean_offset": 1.0,
                 "forget_agreement": 0.0, "promote_agreement": 0.0}
        assert trainer.composite_score(perfect) > trainer.composite_score(worst)


class TestEvaluate:
    def test_reports_the_operational_columns(self, trainer):
        y_true = [0.05, 0.3, 0.7, 0.95]
        metrics = trainer.evaluate(y_true, list(y_true))
        for key in ("spearman", "mae", "mean_offset", "forget_agreement",
                    "promote_agreement", "mean_pred", "mean_label"):
            assert key in metrics, key

    def test_identical_predictions_agree_completely(self, trainer):
        y_true = [0.05, 0.3, 0.7, 0.95]
        metrics = trainer.evaluate(y_true, list(y_true))
        assert metrics["forget_agreement"] == 1.0
        assert metrics["promote_agreement"] == 1.0
        assert metrics["mae"] == pytest.approx(0.0)

    def test_forget_agreement_is_measured_at_the_real_threshold(self, trainer):
        """0.1 is `forgetting_score_threshold`. A prediction of 0.11 against a
        label of 0.05 disagrees about deleting the memory, which is the decision
        the number exists to inform."""
        metrics = trainer.evaluate([0.05], [0.11])
        assert metrics["forget_agreement"] == 0.0


class TestArtifactRoundTrip:
    """The trainer must not be able to emit something the runtime rejects. Uses
    the real loader rather than re-checking the shape by hand."""

    def test_emitted_lexical_artifact_loads(self, trainer, tmp_path):
        import json

        from agent_memory.services.importance import (
            LEXICAL_FEATURE_COUNT,
            load_artifact,
        )

        doc = trainer.build_artifact(
            "lexical",
            [0.1] * LEXICAL_FEATURE_COUNT,
            0.2,
            training={"labels": ["test"], "n_samples": 1},
        )
        path = tmp_path / "out.json"
        path.write_text(json.dumps(doc))
        artifact = load_artifact(path)
        assert artifact.kind == "lexical"
        assert artifact.intercept == pytest.approx(0.2)

    def test_emitted_embedding_artifact_loads(self, trainer, tmp_path):
        import json

        from agent_memory.services.importance import load_artifact

        doc = trainer.build_artifact(
            "embedding_linear",
            [0.1, 0.2, 0.3],
            0.0,
            embedding={"provider": "bedrock", "model": "m", "dimension": 3},
            training={"labels": ["test"], "n_samples": 1},
        )
        path = tmp_path / "out.json"
        path.write_text(json.dumps(doc))
        assert load_artifact(path).dimension == 3

    def test_artifact_declares_the_current_schema_version(self, trainer):
        from agent_memory.services.importance import SCHEMA_VERSION

        doc = trainer.build_artifact("lexical", [0.0] * 7, 0.0, training={})
        assert doc["schema_version"] == SCHEMA_VERSION

    def test_training_block_records_the_metrics(self, trainer):
        """§8 mitigation 2: an operator has to be able to read calibration off
        the artifact before switching a production deployment onto it."""
        doc = trainer.build_artifact(
            "lexical", [0.0] * 7, 0.0,
            training={"metrics": {"forget_agreement": 0.9}, "n_samples": 10},
        )
        assert doc["training"]["metrics"]["forget_agreement"] == 0.9


class TestFeatureNamesAreRecorded:
    def test_lexical_artifact_records_feature_names(self, trainer):
        """Positional coefficients with no names in the file is how a reordering
        becomes undiagnosable. Names are documentation the artifact carries."""
        from agent_memory.services.importance import LEXICAL_FEATURE_NAMES

        doc = trainer.build_artifact("lexical", [0.0] * 7, 0.0, training={})
        assert tuple(doc["training"]["feature_names"]) == LEXICAL_FEATURE_NAMES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_train_importance.py -x -q`
Expected: FAIL — either `importorskip` skips (if sklearn is absent from the dev env, install with `uv pip install -e '.[training]'` first) or `FileNotFoundError` for the script.

- [ ] **Step 3: Implement**

Create `scripts/train_importance.py`. Structure:

**Module docstring** — the four label sources as lifecycle stages, with the stated limits from spec §7 Stage 1 repeated verbatim (labels are sparse; near-binary; casual-conversation domain; neither dataset ships importance labels, this derivation is ours). Note that the script imports `agent_memory` but the reverse must never happen.

**`derive_benchmark_labels(sessions, questions)`** — three-way, not two-way:

```python
def derive_benchmark_labels(sessions, questions):
    """Positive if cited by a later question, negative if in an uncited session.

    Anything else is dropped. That third case is the whole point: a turn sitting
    in a session some question drew on, which that question did not cite, is
    *unlabeled* — we cannot tell whether it was useless or simply not what
    anybody happened to ask about. Folding it into the negatives is the easiest
    way to train a model that scores almost everything as forgettable, and every
    aggregate metric still looks reasonable while it happens.

    Returns `[(content, label)]`. Empty when `questions` is empty: with nothing
    cited, "uncited session" describes the entire corpus.
    """
```

**`label_from_mongo_document(doc, *, now)`** — returns `None` liberally. Signals, in priority order:

1. `is_deleted` with `deleted_reason == "low_importance"` → low (consolidation already judged it).
2. `access_count >= _FREQUENT_ACCESS` → high, scaled.
3. `created_at` older than `_MATURITY_DAYS` with `access_count == 0` → low.
4. Otherwise `None` — **including** a young memory with no accesses. A memory created an hour ago has not had the chance to be useful, and labeling it 0 teaches the model that everything new is worthless, which is precisely when scoring happens.

**`evaluate(y_true, y_pred, *, forget_threshold=0.1, promote_threshold=0.6)`** — returns `spearman`, `mae`, `mean_offset` (signed: `mean(pred) - mean(true)`), `mean_pred`, `mean_label`, `forget_agreement`, `promote_agreement`. Agreement is the fraction of samples where both scores land on the same side of the threshold. Comment why the thresholds are hardcoded defaults matching `consolidation.py`, and that they are parameters so an operator with custom thresholds can pass their own.

**`composite_score(metrics)`** — weights, with the reasoning in the docstring:

```python
# Calibration outranks correlation because consolidation compares against
# absolute thresholds rather than sorting. A model with Spearman 0.85 and a
# +0.15 mean offset ranks memories beautifully and promotes nearly all of them.
# `generate_models.py:100-115` is where the composite-metric shape comes from;
# the weights are inverted relative to it, because that app ranks and this one
# thresholds.
_WEIGHTS = {
    "forget_agreement": 0.30,
    "promote_agreement": 0.30,
    "mae": 0.20,          # inverted below
    "mean_offset": 0.15,  # magnitude, inverted below
    "spearman": 0.05,
}
```

`mae` and `abs(mean_offset)` contribute as `1 - min(1, value)`. `spearman` maps from `[-1, 1]` to `[0, 1]`.

**`build_artifact(kind, coefficients, intercept, *, embedding=None, training)`** — emits the §5 shape, sets `schema_version=SCHEMA_VERSION`, `squash="logistic"`, and for `kind == "lexical"` injects `training["feature_names"] = list(LEXICAL_FEATURE_NAMES)`. Round-tripped through the real `load_artifact` by the tests, so the trainer cannot emit a file the runtime rejects.

**Bake-off** — `LogisticRegression` and `Ridge` trained side by side, both scored with `evaluate` + `composite_score`, both printed, only the winner written. No RandomForest: a forest has no coefficient export, which is a genuine capability given up (trees would likely beat linear on seven lexical features) in exchange for an install with no new runtime dependencies. State that trade in a comment rather than leaving it looking like an oversight.

**`--dry-run`** prints the metrics table and writes nothing. Default behaviour must not overwrite a committed artifact without `--out`.

**Sample weighting** — benchmark labels get `--benchmark-weight` (default 2.0) over LLM-distilled labels. `--negative-ratio` controls how many dropped-unlabeled turns are sampled as negatives, defaulting to 0.0, so the assumption is opt-in and visible on the command line rather than baked in.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv pip install -e '.[training,dev]'
.venv/bin/python -m pytest tests/unit/test_train_importance.py -q
```
Expected: PASS — 20 tests.

Confirm the library still imports nothing from the scientific stack:
```bash
.venv/bin/python -m pytest tests/unit/test_packaging.py -q
```

- [ ] **Step 5: Train and commit real coefficients**

Replace Task 1's placeholders. The benchmark download needs network; if it is unavailable, run the LLM-distillation and synthetic stages alone and say so in the artifact's `training.labels`.

```bash
.venv/bin/python scripts/train_importance.py \
    --source combined --space lexical --dry-run
.venv/bin/python scripts/train_importance.py \
    --source combined --space lexical \
    --out agent_memory/data/importance/lexical.json
```

Then check the trained model against the two signs the design predicts:

```bash
.venv/bin/python -c "
from agent_memory.services.importance import (
    LEXICAL_FEATURE_NAMES, bundled_artifact_path, load_artifact)
a = load_artifact(bundled_artifact_path('lexical'))
for name, c in zip(LEXICAL_FEATURE_NAMES, a.coefficients):
    print(f'{name:>14}  {c:+.4f}')
print('intercept', a.intercept)
print('metrics', a.training.get('metrics'))
"
```

`temporal` and `interrogative` are predicted negative (Task 2). If either trained positive, **do not just ship it** — that is the model disagreeing with the design's premise, and it means either the label derivation or the feature is wrong. Record the actual signs and the reasoning in the commit message either way.

Then a real end-to-end check against the LLM path on a handful of memories:

```bash
.venv/bin/python -c "
import asyncio
from agent_memory.services.importance import (
    LocalScorer, bundled_artifact_path, load_artifact)
CASES = [
    (\"I'm allergic to penicillin — never prescribe it.\", 'high'),
    ('My team always deploys behind a feature flag.', 'high'),
    ('Can you deploy branch fix-3 today?', 'low'),
    ('ok thanks', 'low'),
]
s = LocalScorer(load_artifact(bundled_artifact_path('lexical')))
for text, want in CASES:
    print(f'{asyncio.run(s.score(text)):.3f}  want={want:4}  {text}')
"
```

The two `high` cases must score above the two `low` cases. If they do not, the artifact is not shippable regardless of its aggregate metrics — this is the discrimination the feature exists to provide, and four hand-picked cases are the cheapest possible check on it.

Embedding artifacts need embeddings for the corpus, which needs credentials. If unavailable, leave `titan-1536.json` and `voyage-3-1024.json` as placeholders and say so plainly in each file's `training.note` and in the README: `IMPORTANCE_SCORER=local` on those embedders will load a neutral model until they are trained. A placeholder honestly labeled is fine; a placeholder that looks trained is not.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_importance.py tests/unit/test_train_importance.py \
        agent_memory/data/importance/
git commit -m "Add the offline importance trainer and trained lexical weights

Four label sources as lifecycle stages: benchmark-derived positives,
LLM distillation for scale density, a combined bake-off, and
--from-mongodb retraining from live access signals.

Label derivation is three-way, not two-way. A turn in a cited session
that no question cited is *unlabeled* and dropped — folding it into the
negatives is the easiest way to train a model that scores almost
everything forgettable while every aggregate metric still looks fine.
Same reason label_from_mongo_document returns None for a young
unaccessed memory: it has not had the chance to be useful yet, and
scoring it 0 teaches the model that new memories are worthless, which is
exactly when scoring happens.

The composite metric weights calibration above correlation, inverting the
weighting in the reference app it borrows its shape from: that app ranks,
this one thresholds. A model with Spearman 0.85 and a +0.15 mean offset
promotes nearly the whole store.

No RandomForest — no coefficient export. Trees would probably beat
linear on seven lexical features; the trade buys an install with no new
runtime dependencies."
```

---

## Final verification

- [ ] Full suite: `.venv/bin/python -m pytest tests/unit -q`
- [ ] Default path is byte-identical in behaviour: `.venv/bin/python -m pytest tests/unit/test_enrichment.py tests/unit/test_provider_prompt_contract.py tests/unit/test_importance_parsing.py -q`
- [ ] No scientific stack in the library: `.venv/bin/python -m pytest tests/unit/test_packaging.py -q`
- [ ] Both paths construct from config:
  ```bash
  .venv/bin/python -c "
  from unittest.mock import patch
  from agent_memory.core.config import MCPConfig
  from agent_memory.providers.manager import ProviderManager
  for kind in ('llm', 'local'):
      c = MCPConfig(mongodb_connection_string='mongodb://x', importance_scorer=kind, _env_file=None)
      with patch.object(ProviderManager, '_create_embedding_provider', return_value=object()), \
           patch.object(ProviderManager, '_create_llm_provider', return_value=object()):
          print(kind, '->', type(ProviderManager(c).scorer).__name__)
  "
  ```
- [ ] `CHANGELOG.md` records the new config and states the calibration caveat
- [ ] `README.md` documents `IMPORTANCE_SCORER`, `IMPORTANCE_MODEL_PATH`, the artifact selection table, and that lexical is the weaker path rather than a peer
- [ ] Untrained artifacts, if any remain, say so in `training.note` and in the README
