# agent-memory — Pluggable Importance Scoring (local ML alternative to the LLM)

**Date:** 2026-07-30
**Status:** Approved design, pre-implementation
**Repo:** `mongodb-partners/ai-memory` (package `agent-memory`)
**Branch:** `revamp/agent-memory-v4`
**Requirement IDs:** REQ-E-160 … REQ-E-172
**Target version:** `4.2.0` (minor — additive, default behavior unchanged)

---

## 1. Context & Goal

Every long-term memory `agent-memory` stores is scored for importance by an LLM.
`EnrichmentWorker._process_standard_enrichment` calls
`providers.llm.assess_importance`, the provider prompts for a rating, and
`parse_importance` normalizes the reply to `0.1–1.0`. That score is not
cosmetic: `ConsolidationWorker` soft-deletes memories below
`forgetting_score_threshold` (0.1) and promotes STM above
`promotion_importance_threshold` (0.6), and `MemoryService._calibrated_rank`
weights it at `ranking_beta` (0.3) in every recall.

One LLM round trip per LTM memory, at `enrichment_concurrency=5`, is the
dominant cost and latency in enrichment. For short memories it is the *only*
LLM call in the standard path — `_summarize` is gated behind
`MIN_SUMMARIZABLE_CHARS = 120`, and the merge completion only fires on
near-duplicates.

**Goal:** add a second, opt-in scoring path backed by a small pre-trained model
that runs in-process in microseconds, and let the operator choose. The LLM path
remains the default and is unchanged.

### Why this is cheap here specifically

Importance scoring on text normally means running an encoder at inference time.
It does not here: **the embedding already exists before scoring happens.**
`_process_standard_enrichment` receives `memory["embedding"]` and passes it to
`evolve_memory` (`services/enrichment.py:127-132`). The feature vector is
already paid for, so a linear head on it is a dot product — no torch, no second
network call, no model download.

### Prior art in the two reference apps

Two sibling repos already do local ML, and they disagree in instructive ways.

| | `predictive-maintenance-aws-mongodb` | `maap-recommendation-engine-qs` |
|---|---|---|
| Model | LogisticRegression / RandomForest bake-off | Surprise SVD matrix factorization |
| Training data | committed CSVs | live MongoDB, CSV only as cold-start bootstrap |
| Artifacts | committed `.pkl` (2.8 MB) | generated at startup, synced to S3/volume |
| Retraining | offline script | live `POST /api/retrain_matrix_factorization` |
| Versioning | filename only | `training_info` dict inside the pickle |

What we take from each:

- **From predictive-maintenance:** the model bake-off with a weighted composite
  score (`generate_models.py:100-115`), and offline training that emits
  artifacts shipped with the app.
- **From the recommendation engine:** artifact metadata that inference
  dispatches on. `recommend_products` reads `training_info.model_type` and
  routes to the Surprise or legacy LightFM path (`utils/llm_utils.py:161-172`),
  so the format evolved without breaking installed models. Also its
  bootstrap → live-data → retrain shape (`extract_ratings_from_mongodb`,
  `core/services.py:125-134`).

What we deliberately reject:

- **Committing multi-megabyte pickles.** Viable for them, not for a library
  wheel — and pickle loading is arbitrary code execution.
- **Deleting the losing model as a training side effect**
  (`generate_models.py:190-193`). Their own code needed a `1e-6` guard and a
  tie-break branch to make it safe. Our trainer reports both and writes the
  winner.
- **Generating artifacts at startup.** The recommendation engine can afford a
  boot-time training pass; a library imported into someone else's process
  cannot.

---

## 2. Architecture

**A one-method seam with two implementations, selected by config.**

```text
                      MCPConfig.importance_scorer
                                 │
              ┌──────────────────┴──────────────────┐
              │ "llm" (default)                     │ "local"
              ▼                                     ▼
        LLMScorer                              LocalScorer
   providers.llm.assess_importance()       pure-Python dot product
   + prompt_library lookup                 over the existing embedding
              │                                     │
              └──────────────┬──────────────────────┘
                             ▼
                   ImportanceScorer protocol
                             │
                 EnrichmentWorker(scorer=…)
                             │
                    memory["importance"]
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  _forget_low_importance  _promote_to_ltm  _calibrated_rank
```

Selection logic lives in `ProviderManager` (which is already the
"pick an implementation from config" component); injection happens in
`AsyncMemory._maybe_start_workers`, so the dependency is visible in
`EnrichmentWorker`'s signature and substitutable in tests.

### Rejected alternatives

**`LocalScorer` subclasses `LLMProvider`.** Overriding `assess_importance` and
delegating `chat`/`generate_summary` to a wrapped real provider would need zero
changes to `enrichment.py`. Rejected: `LLMProvider`'s contract is "a thing that
talks to a language model," and an object that is local for one method and
remote for two others makes that contract a lie. It also corrupts provenance —
documents record which model produced a score, and the wrapper would report the
wrapped provider's name for a score it never computed.

**A third `ProviderManager` attribute the worker reaches through
(`self.providers.scorer`).** Mechanically equivalent, but hides the dependency
from the worker's signature. We keep the *selection* in `ProviderManager` and
pass the result in explicitly.

### New files

```text
agent_memory/services/importance.py      # protocol + both scorers + artifact loader
agent_memory/data/importance/
    titan-1536.json                      # bedrock amazon.titan-embed-text-v1
    voyage-3-1024.json                   # voyage-3 (public API + Atlas gateway)
    lexical.json                         # provider-independent fallback
scripts/train_importance.py              # offline trainer, never imported by the library
tests/unit/test_importance_scorer.py
tests/unit/test_importance_artifact.py
```

### Modified files

| File | Change |
|---|---|
| `core/config.py` | two new fields + validator arm |
| `providers/manager.py` | `_create_scorer` match arm, `self.scorer` |
| `services/enrichment.py` | `scorer` ctor param; importance branch → one call |
| `memory.py` | pass `scorer=self.providers.scorer` |
| `pyproject.toml` | `training` extra; artifacts in sdist allow-list |
| `tests/unit/test_enrichment.py` | scorer injection + default-path regression |
| `tests/unit/test_config.py` | new field defaults and validation |
| `tests/unit/test_packaging.py` | artifacts present in sdist and wheel |

---

## 3. The seam (REQ-E-160, REQ-E-161)

```python
@runtime_checkable
class ImportanceScorer(Protocol):
    async def score(
        self,
        content: str,
        embedding: list[float] | None = None,
        *,
        tags: list[str] | None = None,
        message_type: str | None = None,
    ) -> float:
        """Return an importance score in [0.1, 1.0]."""
        ...
```

`embedding` is a parameter because the caller already holds it. It is optional
because a scorer must work when it is absent, and because `LLMScorer` ignores
it entirely. `tags` and `message_type` are keyword-only and optional for the same
reason — the lexical model uses them, the embedding head and `LLMScorer` do not.

The seam takes fields, not the memory document. Passing the whole document would
couple every scorer to the document shape and let a scorer read `importance`
itself, which is the value it exists to produce.

`runtime_checkable` is declared so `isinstance` works against the protocol —
useful for a startup assertion and for test readability. It is **not** what makes
mocks bind the signature; see §10, which pins down what actually does.

`async` even though `LocalScorer` never awaits: the seam must accommodate the
LLM implementation, and a sync method would force the worker to branch on which
scorer it holds.

### `LLMScorer` (REQ-E-160)

Holds `providers.llm` and an optional `prompt_library`. Its `score` is the code
currently at `services/enrichment.py:112-118`, moved verbatim — including the
`_get_prompt("importance_assessment")` lookup and the fallback to calling
`assess_importance` without a `prompt` argument.

`_get_prompt` remains on `EnrichmentWorker`; `_summarize` and `_process_merge`
still use it. `LLMScorer` receives the bound method (or `None`) rather than the
library itself, so the prompt-resolution policy stays in one place.

### `LocalScorer` (REQ-E-161)

Holds a loaded artifact. `score` computes `logistic(Σ wᵢxᵢ + b)` and clamps to
`[0.1, 1.0]`. Pure Python — no numpy, no sklearn, no pickle.

**The floor is 0.1, not 0.0.** `providers/base.py:44-51` explains why: 0.0 is at
`forgetting_score_threshold`, so a score of zero is a deletion order. That
reasoning does not change because the number came from a dot product.
`LocalScorer` enforces the floor itself rather than routing through
`parse_importance` (there is no text to parse), and the invariant is asserted
independently in tests because it is a *different* code path reaching the same
guarantee.

---

## 4. Configuration (REQ-E-162)

Added to `MCPConfig` (inherited by `MemoryConfig`):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `importance_scorer` | `str` | `"llm"` | `"llm"` or `"local"` |
| `importance_model_path` | `str \| None` | `None` | Override the bundled artifact |

**`"llm"` is the default and that is the whole safety argument.** An existing
install upgrades to 4.2.0 and nothing about its behavior changes.

An unrecognized `importance_scorer` value raises at config construction, via a
new arm on the existing `@model_validator` — not at first score. The precedent
is `_auth_must_not_fail_open` (`core/config.py:159-186`): a config that cannot
do what the operator asked should refuse to start rather than degrade quietly.

`importance_model_path` is how deployment-specific retraining lands: the
operator runs the trainer against their own data and points this at the output.

---

## 5. Artifact format (REQ-E-163, REQ-E-164)

One JSON file per embedding space, plus one lexical fallback.

```json
{
  "schema_version": 1,
  "kind": "embedding_linear",
  "embedding": {
    "provider": "bedrock",
    "model": "amazon.titan-embed-text-v1",
    "dimension": 1536
  },
  "coefficients": [0.0031, -0.0117, "…1536 floats"],
  "intercept": 0.4812,
  "squash": "logistic",
  "training": {
    "labels": ["longmemeval", "locomo", "llm_distill"],
    "n_samples": 8400,
    "trainer_version": "1.0.0",
    "metrics": {
      "spearman": 0.71,
      "mae": 0.094,
      "mean_pred": 0.52,
      "mean_label": 0.53,
      "forget_agreement": 0.93,
      "promote_agreement": 0.88
    }
  }
}
```

The `training` block is the recommendation engine's `training_info` pattern
(`utils/llm_utils.py:161-172`) — metadata inference dispatches on, so the format
can evolve without breaking installed artifacts. `kind` is the dispatch key
(`"embedding_linear"` or `"lexical"`); `schema_version` means a future format
change refuses to load rather than misreading old coefficients.

`mean_pred` and `mean_label` are recorded because of the calibration risk in
§8 — a shifted model is then visible in the file itself, not only in training
logs nobody keeps.

### Selection (REQ-E-164)

`LocalScorer` matches on the tuple
`(embedding_provider, embedding_model, embedding_dimension)`, read **after**
`ProviderManager` aligns the Voyage dimension (`providers/manager.py:93-95`).
That alignment mutates `config.embedding_dimension`, which is precisely why the
configured value can differ from the declared default.

**This makes construction order load-bearing.** `_create_scorer` must run after
`_create_embedding_provider` in `ProviderManager.__init__`, because the Voyage
arm both rewrites `config.embedding_model` (to `config.voyage_model`) and may
rewrite `config.embedding_dimension`. A scorer built first would match against
`"amazon.titan-embed-text-v1"` / `1536` — the pre-alignment defaults — on a
correctly configured Voyage deployment, and silently fall back to lexical. The
implementation must add the `self.scorer` assignment *below* the existing two,
and a test must pin this by constructing a `ProviderManager` with a Voyage config
and asserting the embedding head was selected.

- **Match** → embedding head.
- **No match** → lexical artifact, logged once at INFO naming both the
  configured model and the reason. A silent downgrade to weaker features is
  exactly the failure that hides in production.
- **`embedding=None`** at score time → lexical path for that call.

### Packaging

Artifacts live in `agent_memory/data/importance/` and are added to hatch's
`sdist` `include` list and the wheel. That allow-list is explicit rather than an
exclude-list because a live `.env` exists in this working tree
(`pyproject.toml:91-93`), so it must be extended deliberately.

The specific risk being tested for: artifacts present in the repo but absent
from the wheel would make local mode fail only for installed users and never in
development.

---

## 6. Scoring math (REQ-E-165, REQ-E-166)

### Embedding head (REQ-E-165)

`logistic(Σ wᵢxᵢ + b)`, clamped to `[0.1, 1.0]`.

Logistic rather than a raw linear output because the target is bounded and a
linear head extrapolates past both ends; the squash makes clamping a guard
rather than routine operation. 1536 multiply-adds in pure Python is
roughly 50–100 µs — three to four orders of magnitude below the LLM round trip
it replaces.

A dimension mismatch between vector and artifact **raises** rather than scoring
the overlapping prefix. This is the class of failure
`known_embedding_dimension` exists to prevent (`providers/manager.py:54-62`),
reached by a new route.

### Lexical fallback (REQ-E-166)

Seven features, all computable from the document with no dependencies, no
tokenizer, and no vocabulary:

| Feature | Rationale |
|---|---|
| `log(len(content))` | Longer memories carry more; log because it saturates |
| `message_type == "human"` | Already the gate for LTM candidacy (`services/memory.py:139`) — see caveat below |
| digit density | Dates, amounts, IDs — the specifics worth keeping |
| `?` present / imperative opener | Questions and instructions vs. chatter |
| capitalized-token ratio | Crude proper-noun proxy: names, products, places |
| `len(tags)` | Caller-supplied tags are an explicit relevance signal |
| first-person-singular ratio | "I prefer…", "my…" — stated preferences are durable |

Same JSON shape with `"kind": "lexical"`. Having no vocabulary means it degrades
gracefully on non-English text rather than scoring it as noise.

**`message_type` is near-constant on the memories this scorer actually sees.**
LTM candidates are only created for `message_type == "human"`
(`services/memory.py:139`), so at enrichment time the feature is almost always
`1`. It is retained for two reasons: promoted STM reaches enrichment with
`enrichment_status="pending"` (`services/consolidation.py:164`) and can carry
`"ai"`, and `evolve_memory`'s reinforce path writes documents with
`message_type: None` (`services/memory.py:500`). The trainer must therefore
report this feature's fitted weight and sample variance, and drop the feature if
variance is degenerate on the training corpus — a feature that is constant in
training and constant in production contributes nothing but a bias term already
covered by the intercept.

**This is a floor, not a peer.** The lexical model cannot distinguish
"I'm allergic to penicillin" from "I'm allergic to bad puns" — both are
first-person, similar length, no digits. The embedding head can. Documentation
must state plainly that it is the weaker path rather than implying parity; it
exists so local mode remains usable on an embedder we did not ship coefficients
for.

---

## 7. Labels and the trainer (REQ-E-167 … REQ-E-170)

`scripts/train_importance.py` — **outside `agent_memory/`**, because it imports
sklearn, pandas, and `datasets`, which must never be importable from library
code. Guarded by a new extra:

```toml
training = ["scikit-learn>=1.3", "numpy>=1.26", "pandas>=2.1", "datasets>=2.14"]
```

Four label sources, combined as sequential stages of one lifecycle rather than
as alternatives.

### Stage 1 — benchmark anchor (REQ-E-167)

`xiaowu0162/longmemeval-cleaned` (the original `xiaowu0162/longmemeval` carries
a deprecation notice pointing here) and `adymaharana/locomo`. Both are
multi-session conversations with questions asked in later sessions.

Label derivation: a turn is **positive** if it is cited in the evidence for a
later question, **negative** if it sits in a session no question draws on. This
measures future usefulness directly, which is the target quantity — every other
source is a proxy for it.

Stated limits, to be repeated in the script's docstring:

- Labels are **sparse**. Most turns are unlabeled, not negative.
  Treating unlabeled as negative is an assumption, controlled by an explicit
  sampling ratio flag rather than left implicit.
- Labels are near-binary, so they cannot alone populate a continuous scale.
- The domain is casual conversation, not any given deployment's domain.
- Neither dataset ships importance labels. This derivation is ours.

### Stage 2 — LLM distillation for scale coverage (REQ-E-168)

Run the existing `assess_importance` over the same corpora plus a synthetic set;
take the returned scores as continuous regression targets.

This is what makes the local path *agree with* the LLM path rather than merely
correlate with it — which matters because `forgetting_score_threshold` and
`promotion_importance_threshold` are absolute cutoffs. They only mean the same
thing under both scorers if the two scales coincide.

### Stage 3 — combine and bake off (REQ-E-169)

Benchmark labels receive higher sample weight (grounded in observed utility);
LLM labels supply density across `0.1–1.0`. `LogisticRegression` and `Ridge` are
trained side by side and scored; the script reports both and writes only the
winner.

The composite-metric idea transfers from `generate_models.py:100-115`, but the
weights change: **calibration MAE and mean-offset outrank ranking accuracy
here**, because absolute thresholds consume the output. A model with excellent
Spearman and a `+0.15` offset would promote nearly everything.

No RandomForest — a forest cannot be exported as coefficients, which is a real
capability given up (trees would likely beat linear on the lexical features).
The trade is an install with no new runtime dependencies, which is the right
call for a library.

### Stage 4 — deployment retraining, script-driven (REQ-E-170)

A `--from-mongodb` flag pulls the operator's own memories and labels them from
signals the documents already carry: `access_count`, `last_accessed`, promotion
to LTM, and survival past consolidation. This is the recommendation engine's
`extract_ratings_from_mongodb` (`utils/utils.py:134-180`) in spirit — train from
live data, with the shipped artifact as the cold-start bootstrap their CSV
indexing provides.

Output is a JSON artifact the operator points `importance_model_path` at. This
is operator-driven rather than automatic: with artifacts committed to the repo
there is no in-app artifact store to write back to. An automatic retraining loop
would need a MongoDB-backed artifact collection, which is deferred (§11).

### Trainer output

The JSON artifacts of §5, plus a printed metrics table (the `tabulate` pattern
from `generate_models.py:229`) reporting Spearman, MAE, mean offset, and
**agreement rate with the LLM at both consolidation thresholds**. That last
column is the operationally meaningful one: *of the memories the LLM would
forget, what fraction does this model also forget?*

---

## 8. Consequences for consolidation

The consolidation worker is not modified, but it is affected, and this is the
sharpest risk in the design.

`_forget_low_importance` soft-deletes on `importance < 0.1`
(`services/consolidation.py:113-131`) and `_promote_to_ltm` promotes on
`importance >= 0.6` (`:133-148`). Both are **absolute** thresholds. A local
scorer whose calibration differs from the LLM's silently changes which memories
get deleted — and the symptom, as
`tests/unit/test_importance_parsing.py:1-16` records from a previous incident,
is a memory that stops being recalled weeks later.

Mitigations, all specified above rather than left to implementation:

1. The trainer optimizes for calibrated agreement, not ranking correlation
   (§7 Stage 3).
2. `mean_pred` / `mean_label` / `forget_agreement` / `promote_agreement` are
   recorded in the artifact itself (§5).
3. The `0.1` floor is enforced by `LocalScorer` and asserted independently
   (§3).

---

## 9. Error handling (REQ-E-171)

Three tiers, each chosen against a named failure:

| Condition | Behavior | Why |
|---|---|---|
| `local` + artifact missing / malformed / unknown `schema_version` / unknown `kind` / wrong coefficient count | Raise at `AsyncMemory.create()` | The operator asked for local scoring; silently serving LLM scoring makes the cost saving they are measuring fictional |
| `local` + artifact loads but embedder unmatched | Lexical fallback, INFO log | Designed degradation, still functional |
| Scorer raises at score time | Existing `_enrich_memory` retry path | Already correct — increments `enrichment_retries`, keeps status `pending`; duplicating it would create a second policy |

Load errors name the file and the specific problem. "Failed to load model" is
not actionable; "coefficient count 1024 does not match declared dimension 1536
in titan-1536.json" is.

---

## 10. Testing strategy (REQ-E-172)

Every scorer mock uses **`create_autospec(ImportanceScorer, instance=True)`**,
not `AsyncMock(spec=...)`.

This was verified on the project's Python (3.11.13) rather than assumed, and the
result contradicts what the existing suite believes. `tests/unit/test_enrichment.py:24-36`
uses `AsyncMock(spec=LLMProvider.assess_importance)` and its comment states that
the `spec=` is what catches signature drift. It does not:

| Mock construction | bogus kwarg | 4 positional args |
|---|---|---|
| `AsyncMock(spec=Class.method)` | **accepted** | **accepted** |
| `create_autospec(Proto, instance=True)` | `TypeError` | `TypeError` |

`AsyncMock(spec=some_function)` copies the spec's *async-ness* and restricts
*attributes*, but does not enforce the call signature. `create_autospec` does.
So the existing test's protection against the exact drift its docstring
describes — a provider that does not accept `prompt=` — is weaker than it claims;
it would catch a *missing method*, not a wrong signature.

`create_autospec` also yields awaitable methods for `async def` members, so it is
a drop-in for the async case (verified). Fixing the existing `LLMProvider` mocks
is out of scope here, but the new tests will not inherit the flaw, and this
finding should be filed as a separate defect.

**`test_importance_scorer.py`** — scoring math, no I/O:

- Known coefficients × known vector → hand-computed expected value. Pins the
  arithmetic so a future refactor of the loop cannot drift.
- Clamping at both ends: saturating positive ≤ 1.0; saturating negative
  == exactly 0.1, not 0.0.
- 1024-dim vector against a 1536-dim artifact raises rather than scoring the
  overlap.
- `embedding=None` with an embedding artifact → lexical path, logged.
- Each lexical feature isolated: two inputs differing only in digit density
  score differently, in the expected direction.
- Non-English input scores without raising.

**`test_importance_artifact.py`** — loading contract:

- Missing file, malformed JSON, unknown `schema_version`, unknown `kind`, wrong
  coefficient count → each raises, message names file and problem.
- Every bundled artifact loads, and its declared `dimension` equals its
  `coefficients` length. Cheap, and catches a hand-edited file.
- `importance_model_path` overrides the bundled artifact.
- **Construction order:** a `ProviderManager` built from a `voyage-3-lite` config
  (512 dims, so alignment definitely fires) selects the artifact matching the
  *aligned* model and dimension, not the pre-alignment defaults. See §5.

**`test_enrichment.py`** — extended, not rewritten:

- The `LLMScorer` default path produces writes identical to current behavior.
  This is the most important regression in the change: `"llm"` is the default,
  so an existing install must be unaffected.
- In local mode the injected scorer is called and
  `providers.llm.assess_importance` is **not** — asserted as a non-call, since
  "we stopped paying for this" is otherwise unfalsifiable.
- A scorer that raises increments `enrichment_retries` via the existing handler.

**`test_config.py`** — `importance_scorer` defaults to `"llm"`; an unrecognized
value raises at construction.

**`test_packaging.py`** — `data/importance/*.json` are in the sdist allow-list
and the wheel.

**Explicitly not tested:** accuracy of the shipped coefficients. That belongs to
the trainer's metrics output. A unit test asserting `spearman > 0.7` would
either require the network or be a tautology against committed fixtures.

---

## 11. Out of scope (deferred)

- **Inline scoring at write time.** A sub-millisecond scorer *could* run inside
  `store_stm`, so memories would land with a real importance instead of the
  `0.5` placeholder at `services/memory.py:117,150`. Deliberately excluded: it
  would make the local path behave differently from the LLM path rather than
  substitute for it, so the two modes would no longer be comparable. The
  placeholder gap is real but pre-existing.
- **MongoDB-backed artifact store with automatic retraining.** The
  recommendation engine's live `retrain` endpoint, adapted. Would give
  per-deployment self-improvement with no filesystem writes, following the
  `PromptLibrary` pattern of DB-stored, cached, versioned assets. Deferred in
  favor of committed artifacts that work on first boot.
- **Local `generate_summary` or merge.** Both are generation, not scoring; no
  small local model substitutes for them at acceptable quality.
- **RandomForest / gradient-boosted artifacts.** Requires a numeric runtime
  dependency, contradicting §6.
- **Rescoring existing memories after a model change.** An operator who swaps
  artifacts has a store scored under two models. A backfill command is
  plausible future work.

---

## 12. Requirement index

| ID | Requirement |
|---|---|
| REQ-E-160 | `ImportanceScorer` protocol; `LLMScorer` preserves current behavior exactly |
| REQ-E-161 | `LocalScorer` scores in pure Python with no new runtime dependency |
| REQ-E-162 | `importance_scorer` / `importance_model_path` config; `"llm"` default; invalid value raises at construction |
| REQ-E-163 | Versioned JSON artifact format with a `training` metadata block |
| REQ-E-164 | Artifact selection by `(provider, model, dimension)`; documented lexical fallback |
| REQ-E-165 | Embedding head: logistic linear model, `[0.1, 1.0]` clamp, raise on dimension mismatch |
| REQ-E-166 | Lexical fallback: seven dependency-free features |
| REQ-E-167 | Benchmark-derived labels from LongMemEval and LoCoMo |
| REQ-E-168 | LLM-distilled labels for scale coverage and calibration |
| REQ-E-169 | Combined training with a bake-off; calibration weighted above ranking |
| REQ-E-170 | `--from-mongodb` retraining from live feedback signals |
| REQ-E-171 | Fail loudly on an unusable artifact; fall back only where designed |
| REQ-E-172 | Test coverage per §10, including the default-path regression |
