# Changelog

All notable changes to `agent-memory` are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0] — 2026-07-30

### Added

**Local importance scoring — the LLM call for importance is now optional.**
Importance decides what gets forgotten, what gets promoted to long-term memory,
and how results rank. Until now it cost one LLM round trip per long-term memory,
on the enrichment worker's hot path. `IMPORTANCE_SCORER=local` replaces that call
with a small logistic model evaluated in-process: microseconds instead of a network
round trip, and no tokens.

The local path is viable only because of where it sits in the pipeline. The
embedding already exists by the time enrichment runs, so "inference" is a dot
product over a vector the worker was already holding — no encoder, no second
model, no new dependency. The scorer is pure Python; `numpy` is not a runtime
dependency and a packaging test enforces that nothing under `agent_memory/` ever
imports the scientific stack.

- `IMPORTANCE_SCORER` — `"llm"` (default) or `"local"`. An unrecognized value is
  refused at construction rather than falling back, following the
  `AUTH_ENABLED`/`AUTH_SECRET` precedent: a typo'd `locl` that silently keeps the
  LLM path has no symptom except the invoice.
- `IMPORTANCE_MODEL_PATH` — an explicit JSON coefficient artifact. Unset means
  auto-select the bundled artifact matching the configured embedder.
- `ImportanceScorer` protocol in `agent_memory/services/importance.py`, with
  `LLMScorer` (today's behaviour, wrapped) and `LocalScorer`.
- Bundled artifacts under `agent_memory/data/importance/`, selected on the
  `(provider, model, dimension)` triple — not the provider alone, because
  coefficients are positional and `voyage-3` emits 1024 dimensions where
  `voyage-3-lite` emits 512.
- A trained `lexical` artifact (seven bounded, interpretable text features),
  currently the only one bundled and therefore what every deployment loads. Weaker
  than a trained embedding head would be, and much better than the constant an
  untrained one returns.
- `training` optional-dependency group for the offline trainer. Deliberately not
  part of `all`.
- `scripts/train_importance.py` — the offline trainer, with four label sources that
  compose: the locomo/longmemeval long-term-memory benchmarks, distillation from the
  shipped `assess_importance`, a synthetic in-domain set spanning the scale, and
  `--source mongodb` for retraining on a deployment's own access and consolidation
  signals. `LogisticRegression` and `Ridge` are fitted side by side against two
  constant baselines, and every metric is computed through the *runtime's* arithmetic
  — the library's own `logistic` and clamp, not sklearn's `predict` — so a printed
  number is one the served scorer reproduces.

**Exactly one artifact ships, and it is trained.** `lexical` is provider-independent,
so every deployment gets a model fitted on real labels. `_BUNDLED_ARTIFACTS` is
empty by design and `select_artifact_name` returns `lexical` for every embedder —
that is the decision, not a lookup miss.

`titan-1536` and `voyage-3-1024` were briefly bundled as zero-coefficient
placeholders and have been **deleted rather than trained**. They scored every memory
0.5 — above the forgetting threshold, below the promotion threshold — so
`IMPORTANCE_SCORER=local` on those triples did not approximate the LLM, it silently
turned importance-based promotion off while reporting healthy. Deleting beat
training on measurement: an embedding head's held-out Spearman tops out near 0.45,
its in-sample ceiling on the same 1,234 rows is only 0.70, regularization only
shrinks it toward the constant (alpha 1 → 100,000 gives 0.45 → 0.34), and
`assess_importance` emits a 1–10 integer — 9 distinct values for a 1024-coefficient
fit to aim at. Paid labels would buy resolution the teacher does not have.
`test_no_bundled_artifact_is_a_constant` now rejects an all-zero artifact at commit
time, and the artifact invariants are keyed off directory discovery so they arm
automatically on any head added later.

**The benchmarks are collected but carry zero weight by default**, and that default
is a measurement rather than a preference. Their labels mark a turn positive when a
later question cited it as evidence, and the questions are time-anchored, so
`yesterday` carries a 21.7× lift toward positives and `today` 3.2×. Every nonzero
weight from 0.1 to 2.0 trains the `temporal` feature strongly positive and yields a
model that promotes *"let's pick this up after lunch, I'm busy today and tomorrow"*
(0.775, above the promotion threshold) over *"our policy is that customer data never
leaves the EU region"* (0.435). The stage is kept rather than deleted because the
confound is specific to seven lexical features that can only see the *word*
`yesterday`; an embedding head can tell "we deployed yesterday" from "call me
tomorrow", so the grounded-utility signal is still worth having there.

**A discrimination gate, because the calibration metrics cannot catch this.** The
trainer refuses to write a model that ranks any of four expiring task-chatter cases
above any of four standing-preference cases, held out by construction, regardless of
aggregate metrics. Not redundant: it rejected a candidate scoring 0.85 on the
composite, and the shipped model beats the hardest constant baseline by 0.0005 on
composite while separating the held-out cases by 0.0218. Calibration is satisfiable
by a model that predicts the training mean for everything and discriminates nothing,
which is precisely the model a threshold-based consolidator must not be handed.

Two limits are documented rather than papered over. The seven lexical features cannot
separate 31.5% of the label variance — measured as rows sharing a feature vector while
carrying different labels — and the learning curve is flat past ~460 rows, so a larger
training set will not improve this artifact; the ceiling is the feature space, not the
sample count. And `assess_importance` returns a 1-10 integer, so distilled labels take
only 9 distinct values: the local scorer cannot be more granular than its teacher.

**Calibration matters more than ranking here.** Consolidation compares importance
against *absolute* thresholds — below `forgetting_score_threshold` (0.1) a memory
is deleted, at or above `promotion_importance_threshold` (0.6) it is promoted. A
local model whose scores sit systematically below the LLM's forgets more and
promotes less, and the symptom shows up weeks later as degraded recall rather than
as an error. Before switching a production deployment, read
`forget_agreement`/`promote_agreement` from the artifact's `training.metrics`. The
`[0.1, 1.0]` clamp is applied outside the model arithmetic and tested
independently, so a badly trained artifact cannot emit the value that means
"delete this".

### Changed

- `EnrichmentWorker.__init__` accepts an optional `scorer`. Omitted, it builds an
  `LLMScorer` over the configured LLM provider, so every existing construction
  keeps today's behaviour exactly.
- The importance prompt lookup moved from the worker into `LLMScorer` — the local
  scorer has no prompt, so the worker could not keep the branch.

### Fixed

**Changing the embedding dimension silently orphaned every vector already
stored.** A vector index cannot have its `numDimensions` edited, so index
reconciliation dropped and recreated it. The documents beneath were untouched by
that — which sounds like the safe outcome and is the dangerous one: every stored
vector kept the old width, and the rebuilt index returned none of them from
`$vectorSearch`.

Nothing about that failure surfaces. No exception, no change in document count,
`find` still returning every memory. Recall goes empty for the entire history
while working perfectly for anything written afterwards, so it presents as "the
user has no memories about that" — indistinguishable from the truth. Recovery
means re-embedding every document with the *previous* provider, which is the
config the operator has just replaced. The cost is asymmetric: leaving the stale
index in place loses indexing of new writes, recoverable by fixing the config,
while rebuilding loses the history and is not.

So it is refused in two places, because one is not enough. `create()` runs a
preflight (`find_stranding_dimension_changes`) before any service is built and
raises `ConfigError` naming each affected index, both dimensions, the number of
embedded documents at risk, and the four ways forward. `ensure_search_indexes`
independently declines the destructive rebuild and logs at error level — it has
to, because stage 2 runs as a background task by default whose exceptions are
logged as non-fatal, so a guard that only raised upstream would be one the
default path routes around.

An empty collection is never affected: there are no vectors to strand, which is
the ordinary first-run case of an index left over from a previous config. The
count is of documents that actually carry an `embedding`, not rows, so unenriched
memories and counter documents do not trigger a refusal. A count that cannot be
read is treated as non-empty — "we could not check" is not "there is nothing
there".

- `ALLOW_EMBEDDING_DIMENSION_CHANGE` (`allow_embedding_dimension_change`,
  default `false`) proceeds anyway, for an operator who has read the refusal and
  accepts that the old vectors become unsearchable. It travels down to
  reconciliation as well, so the two guards cannot disagree.

Verified against 15 mutations, all killed: dropping unconditionally; no
preflight; findings logged instead of raised; the preflight moved after services
and workers start; the preflight comparing the declared rather than the resolved
dimension; an unreadable count assumed empty (in each of the two places
separately); empty collections refused too; the opt-in ignored by reconciliation;
the opt-in not forwarded by the facade; counting unembedded documents; the
refusal logged at debug; a declined rebuild pushing an incompatible update
instead; the flag defaulting to allowing; and an unchanged dimension reported as
a stranding.

**CI's lint job could not pass.** The workflow runs `uv run ruff check
agent_memory/`, but `ruff` was declared in neither `pyproject.toml` nor `uv.lock`.
Locally that resolved to whatever ruff happened to be on `PATH` — a developer's
Homebrew build — so the gate looked green; on a clean runner `uv run ruff` exits 2
with `Failed to spawn: ruff`. Verified by running the step with an empty `PATH`.
`ruff>=0.14,<0.17` is now in the `dev` extra, which `uv sync --all-extras` installs.

**The lint gate had no configuration, so its verdict depended on the linter's
release date rather than on the diff.** With no `[tool.ruff]` section, both the
enabled rule set and `target-version` came from whatever ruff was installed: 0.9.10
reported the package clean while 0.16.0 flagged 28 findings, no source change in
between. In the other direction, 0.9.10 demanded five UP038 rewrites of
`isinstance(x, (A, B))` to `A | B` — a rule newer ruff *removed*, because the
rewrite is slower at runtime. `select = ["E4","E7","E9","F","I","UP","RUF"]` and
`target-version = "py311"` are now explicit, and the 67 findings that surfaced are
fixed (`datetime.UTC` for `timezone.utc`, unquoted annotations, sorted imports, one
stale `noqa`). `BLE001`/`S110` are deliberately unselected: the `except Exception:
pass` blocks they flag are health-probe and migration handlers marked `# pragma: no
cover - a probe must never 500`, where the rule is pointing at the intent.
`B008` likewise — `Depends(...)` in an argument default is how FastAPI works.

`TestLintGateIsRunnable` in `test_packaging.py` now asserts ruff is declared, is
upper-bounded, has an explicit `select` and pinned `target-version`, and that the
package passes the gate — running the project's pinned ruff rather than PATH's, so
the test cannot start demanding what the pinned linter does not want. All four
guards were verified against mutations.

Not addressed: `tests/` and `scripts/` sit outside CI's lint scope and carry ~118
findings (mostly `UP017`, `F401`, `I001`). Widening the gate is a larger mechanical
change than this fix, and the three `F841` unused locals were checked by hand —
they are throwaway assignments in tests that assert on mock call args, not dropped
assertions.

**The demo's reset paths had not caught up with `wipe_user_data`.** Both predate
the wipe fix below, which brought `episodes` inside the library's user-data
contract. Two consequences, one of them visible on stage:

* `POST /reset` in `examples/memory-ui/server/app.py` spread the library's result
  and *then* set `episodes_deleted` from its own second delete of `episodes` —
  which necessarily found nothing, because the library had already cleared the
  collection. The count the presenter reads was structurally guaranteed to be
  zero; a reset that cleared nine episodes reported none. The duplicate delete is
  gone and the library's count is what the route returns.
* Neither caller handled `PartialWipeError`, which the wipe now raises instead of
  returning. `/reset` would have answered 500 — "the server is broken" when what
  happened is that some of the user's data is still there — and `demo/seed.py`
  would have printed a traceback. `/reset` now answers **409** carrying the
  per-collection counts and `failed_collections`; the seed script reports it
  through its existing `NOT READY TO PRESENT` channel and exits 1, because seeding
  on top of a half-cleared user leaves exactly the stale documents a recall beat
  can surface.

`test_demo_seed_reset.py` grew from 13 to 22 tests, including the `/reset` route
driven through the real app and lifespan rather than a re-implementation — the
defect was the order of two dict writes, which a test that restates that order
cannot see. Both fixes were verified against mutations that restore the old
behaviour.

**Importing a demo module leaked the live `.env` into the test session.**
`server.app` calls `load_dotenv` at module scope exactly as `demo/seed.py` does, so
importing it inside a test body — where the existing module-scope guard could not
apply — published the real `VOYAGE_API_KEY`, `VOYAGE_API_URL`, and
`EMBEDDING_DIMENSION` into `os.environ` for every test that followed. Ten tests in
three unrelated files began asserting the developer's actual configuration against
library defaults, and all ten passed in isolation. One of them printed the live API
key into the failure output.

The guard is now general (`_import_demo`) and covers both modules, and
`TestTheDemoModulesDoNotLeakTheEnvironment` asserts the property rather than
documenting it: that each demo module was imported through the guard, that each
still loads a `.env` (so the guard is not load-bearing for nothing), that the patch
is restored afterwards, and that no live credential is present in `os.environ`. The
last of those is what fails now — inside the file responsible, naming the variable
— instead of surfacing three files away.

### Security

A further review of the 4.1.0 hardening, which found that two of its fixes had a
reachable way around them and that three subsystems were failing silently. As
before, none of these crashed: each returned something plausible. Every fix below
was confirmed by mutation — the new tests were run against the vulnerable code and
observed to fail, then against the fix and observed to pass.

- **A refused MCP call could still write into another tenant.** `tools.py` resolved
  identity through `resolve_caller` and refused a request naming someone else — and
  then returned that refusal as `{"error": ...}` rather than raising, because that
  is the MCP convention. The auto-capture middleware wrapped the tool, saw a normal
  return, and persisted the turn using `params["user_id"]`: the attacker-supplied
  value the tool had just rejected. So the guarded path was guarded and the
  capture path beside it was not, and the write landed in the victim's tenant.
  Capture now resolves its own identity from the same token through the same
  function — `resolve_capture_identity` — and a mismatch drops the capture instead
  of writing it. The identity is passed to `spawn()` as its own argument rather
  than re-read from the params, so the unsafe value is not in reach at the write
  site. `tools.py`'s module docstring states the trap for the next author: a refusal
  here is a *return value*, not an exception, so "returned without raising" does
  not mean "was authorised".

- **Any authenticated caller could change retention for every tenant.**
  `set_activity_retention` is collection-wide by nature — a TTL index belongs to the
  collection — and was categorised `admin` to withhold it from `power_user`. But
  that categorisation is enforced by the governance service, which is
  `governance_enabled: bool = False`. On a stock multi-tenant deployment the
  `admin` category bought nothing, and the operation was reachable through a public
  REST endpoint. Shortening every other tenant's retention is the destructive
  direction and it happens quietly: Atlas expires the documents on the TTL
  monitor's own schedule, so the caller sees `{"scope": "collection"}` and the data
  goes away later. `_require_admin_for_global_mutation` is now a floor underneath
  the category, independent of governance being switched on. An authorisation rule
  that exists only when an optional subsystem is enabled is a default-open rule.

- **Every tag-filtered and type-filtered search silently returned the wrong
  results.** Three compounding faults, all silent:

  `memory_type` and `tags` were used as `$vectorSearch` pre-filters and declared in
  neither index. An undeclared filter path does not raise — the branch matches
  nothing — so a filtered recall returned zero results while the memories sat in
  the collection, and the failure read as "this user has no memories of that type".

  Tag filtering additionally used `{"tags": {"$all": [...]}}`. `$all` is not among
  the operators `$vectorSearch` supports, and an unsupported operator in a
  pre-filter also does not raise. Declaring the fields alone would have left tag
  search exactly as broken. All-of is now an `$and` of equalities, which works
  because filter fields accept arrays and match when any element matches. The
  supported operator set is pinned in a test, and a separate test rejects `$in`/`$or`
  — the mechanical "fix" that is supported and silently widens all-of to any-of.

  `hybrid_search` applied its narrowings to the vector branch only. `$rankFusion`
  merges two ranked lists, so a restriction on one branch is not a restriction: the
  unfiltered branch contributed matches that ignored it, and fusion mixed them in
  by relevance. A search scoped to one `memory_type` returned documents of every
  other type, indistinguishable from correct hits — wrong results rather than
  missing ones. `user_id` was always in both, so this was never an isolation bug.
  The new contract test asserts against the *built pipelines* and the *shipped
  index definitions* rather than a copied list, and asserts the generic invariant:
  both fusion branches restrict the same fields, with soft-delete the one
  documented asymmetry.

- **An erasure recreated the user it erased.** `wipe_user_data` deletes every
  `audit_log` row matching the user's id, and was then audited through `_run`,
  which writes its success record *after* the service call. The last thing the
  erasure did was write that identifier back into the collection it had just
  cleared — not a leftover it failed to catch, but a row it created, dated a
  millisecond later, which survived every subsequent wipe because each one
  recreated it. The audit buffer was a second path to the same place:
  `audit_flush_on_write` defaults to False, so this user's pending entries flushed
  to Atlas *after* `delete_many` had swept the collection.

  Deleting the record is not the fix — a total irreversible deletion is precisely
  the operation that must leave a trace. So the trace is kept and the subject
  dropped: the record is filed against a reserved `ERASURE_PRINCIPAL`, carries the
  per-collection counts, and names nobody. The buffer is flushed before the delete
  and the record after it. What the audit log deliberately cannot answer is "was
  user X erased?", which is not a question anyone who has genuinely stopped holding
  X can answer. The reserved principal is refused as a wipe target, so the erasure
  trail cannot be deleted by asking to be forgotten under that name. A *denied*
  wipe is still audited against the real user: it erased nothing, so there is no
  erasure to respect, and an attempt to wipe a tenant is exactly what an auditor
  needs attributed.

  Relatedly, a **partial wipe was audited as a success.** Per-collection failures
  were collected and *returned*; `_run` derives its status from whether the
  coroutine raised, so a wipe that cleared three collections and failed on four
  recorded `"success"` — and an operator who reads a success has no reason to
  retry. It now raises `PartialWipeError`, carrying the counts so the audit record
  and the MCP client can both see how far it got. Every remaining collection is
  still attempted; raising early would leave the most data behind.

- **A definition change to an Atlas index could never reach an existing cluster.**
  Stage 2 compared only `numDimensions` on vector indexes and `continue`d past
  every existing full-text index, so an index built by an earlier version kept that
  version's definition for the life of the deployment. Only a fresh cluster ever ran
  the current schema — which means the `memory_type`/`tags` fix above would have
  passed every test, because tests start empty, and changed nothing on the
  deployments that needed it. This fix is the delivery mechanism for that one.

  Existing indexes are now reconciled: a `numDimensions` change still drops and
  recreates, and anything else is an in-place `update_search_index`, which
  preserves the built index and keeps serving the old definition while Atlas
  rebuilds. Dropping would take vector search offline for minutes on every
  deployment, for a change that does not require it. The comparison is deliberately
  a *subset* check, because Atlas echoes definitions back enriched with its own
  defaults — an equality test would find a difference on a perfectly current
  cluster and rebuild every index on every startup. A failed update leaves the
  working index in place rather than dropping it.

- **MCP authentication failed open on a context error.** `tools.py` wrapped
  `get_access_token()` in a bare `except Exception` and treated *any* failure as
  "auth is off", which is the single-tenant path that honours the caller-supplied
  `user_id`. So on an auth-enabled deployment, a token the server could not read —
  an incompatible `AccessToken` type, which FastMCP converts to a `TypeError` —
  downgraded the request to the exact posture tenant binding exists to prevent.

  Whether auth is *configured* is now read from the config rather than inferred
  from whether a token turned up. The catch stays broad, because the import and
  the context lookup have several legitimate failure modes, but with auth on it
  raises `IdentityError` and logs at warning instead of falling through. With auth
  off the behaviour is unchanged. A `None` token with auth on is still accepted and
  deliberately so: over HTTP that state is unreachable — FastMCP rejects an
  unauthenticated request before any tool runs — so it means there is no request at
  all (stdio, or an in-process call), and refusing it would break a documented
  single-tenant transport without closing anything a remote caller can reach.

- **Auto-capture shutdown draining existed and was never wired up.** The
  middleware grew a `drain()` in 4.1.0 to wait for in-flight captures, and
  `mcp/server.py` discarded the middleware object after registering it, so nothing
  could call it. Captures in flight at shutdown were dropped exactly as before the
  fix. The server now retains it and drains on shutdown.

## [4.1.0] — 2026-07-29

### Added

**Episodic memory — the agent activity log.** A fourth memory tier recording what
the agent *did*, not just what it was told: messages, tool calls, todos, files
touched, and a correlation id per turn. Short-term state answers "what are we
doing", long-term semantic memory answers "what do I know about this user", and
episodic memory answers "what did we actually do last Tuesday" — a question
neither of the other two can answer.

- `log_activity(user_id, thread_id, messages, *, todos, agent_name,
  correlation_id, conversation_id, ts)` — non-blocking. Builds the document and
  enqueues; never awaits Atlas or the embedder on the caller's path.
- `recall_activity(user_id, query, *, thread_id, agent_name, since, limit)` —
  hybrid recall over logged turns via `$rankFusion` RRF.
- `get_thread(user_id, thread_id, *, limit, ascending)` — replay a thread in
  step order.
- `get_activity_by_correlation(user_id, correlation_id)` — every turn sharing a
  trace id. Accepts a W3C `traceparent`, so it joins to an existing tracing
  stack without a new convention.
- `flush_activity(timeout)` — bounded wait for queued turns to reach Atlas.
  Returns `bool`; never raises.
- `set_activity_retention(user_id, *, ttl_seconds)` — change retention in place
  via `collMod`. `None` drops the TTL and keeps the log permanently.
- `activity_stats()` — queue depth, throughput, and failure counters.
  Synchronous, so a health probe never waits on an event loop.

Every method has a blocking twin on `Memory`, five MCP tools, and five REST
routes. `GET /health` now reports the writer's counters, because a 200 with a
full queue and rising write failures is not health.

**New `episodes` collection** with five B-tree indexes, a TTL index (30 days by
default), a vector index, and a full-text index. Separate from `memories`: the
document shape, retention policy, and query patterns are all different, and the
deduplication and calibrated-ranking logic in the memories tier must not touch
it. Still one cluster and one database.

**Framework-neutral projection layer** (`core/projection.py`) accepting both
attribute-style message objects and plain dicts, so any agent framework — or
none — can feed it.

**Per-user scoping via `contextvars`** (`core/context.py`): `current_user_id()`
and `scoped_user()`, isolated per asyncio Task and per thread.

**`LLMProvider.chat_stream()` — token streaming.** `chat()` returns a complete
string, so there was no way to stream a response through the provider seam at
all. `chat_stream()` yields text deltas and is implemented natively for Bedrock
(`converse_stream`), Anthropic (`messages.stream`), and OpenAI
(`stream=True`) — no agent framework involved.

It is concrete rather than abstract: the default implementation awaits `chat()`
and yields one chunk, so a provider that cannot stream stays correct and no
existing implementation breaks. Text only — tool-call and usage events would make
the return type provider-shaped, which defeats the point of the seam.

### Changed

- `services/search_pipeline.py` extracts the `$rankFusion` builder that was
  inlined in `MemoryService.hybrid_search`. Both tiers now share one pipeline
  definition, so a fix to the fusion logic cannot land in only one of them.
- `_sanitize_doc` recurses into lists. Episodic documents carry `messages[]`,
  `todos[]`, and `files_touched[]`, which would otherwise return raw BSON.
- `GovernanceService.seed_defaults()` changed from skip-if-exists to an additive
  `$addToSet` backfill. Skip-if-exists left every already-deployed profile
  denying operations a new release added, and the symptom was an `AccessError`
  on the exact feature the user upgraded to get. Custom limits and
  operator-added operations survive; nothing is ever removed.
- `app_version` is now read from installed package metadata instead of a
  hardcoded literal — the literal had drifted to `3.2.0` while the package said
  `4.0.0`, and that value is served by `/health` and stamped on audit records.
- `app_name` default is now `agent-memory` (was `memory-mcp`).
- The container no longer runs as root. A `USER appuser` (uid 10001) owns `/app`,
  which is enough because nothing here needs privilege: port 8000 is
  unprivileged and every write goes to Atlas, not the filesystem.
- Every GitHub Actions `uses:` and the `ghcr.io/astral-sh/uv` build stage are
  pinned to immutable digests instead of `@v4` / `:latest`. A mutable tag can be
  repointed by its publisher — the mechanism behind the `trivy-action` and
  `kics-github-action` compromises.
- The Dockerfile copies `README.md` and `LICENSE` into the build context. Adding
  `readme` and `license-files` to `pyproject.toml` made them build-time
  requirements, and hatchling fails outright without them — the image build was
  broken and nothing caught it.
- `release.yml` will not publish on a tag push. Both publish jobs now require an
  explicit `workflow_dispatch`, because the package is installed from git and is
  not on PyPI.
- Governance profiles: `power_user` gains full episodic read/write;
  `end_user` may log and replay its own threads but not query by correlation id
  (trace ids come from operators, not from a user's own session).
  `set_activity_retention` stays admin-only.

### Fixed

- `voyage-4`, `voyage-4-large`, and `voyage-4-lite` are recognized in the model
  dimension table (all 1024), so `embedding_dimension` auto-aligns for them
  instead of staying at the 1536 default. An unrecognized model left the default
  in place, which builds a 1536-dim index for a 1024-dim embedder. The README now
  documents that the Atlas embeddings gateway and the public Voyage API need
  different keys, URLs, and models.
- **A dimension mismatch is now caught at startup.** It used to be caught nowhere:
  Atlas accepts a 1024-dim vector into a 1536-dim index without complaint and
  simply never returns that document from `$vectorSearch`, so the symptom was
  empty recall and every write until someone noticed had to be re-embedded.
  `create()` now compares the declared dimension against the model's documented
  output and raises `ConfigError` on disagreement. The check consults a built-in
  table first, so it works when the embedding endpoint does not — which is exactly
  when a stale `embedding_dimension` is most likely, since a model was probably
  just changed. Unknown models still fall back to probing the embedder; when
  neither can answer it logs a warning rather than passing quietly. The docstring
  previously claimed a table was consulted while the code only ever probed.
- The FastAPI `version` in the OpenAPI document was a hardcoded `4.0.0`, stale at
  4.1.0. It reads from package metadata like `app_version` — a second copy of a
  version is a copy that goes stale, and this one goes stale where a client reads
  it to decide what the API supports.
- **A partly-failed episodic batch counted as a total failure.** `insert_many`
  runs with `ordered=False`, so a `BulkWriteError` means the good documents are
  already stored and the exception reports which ones were not — the old handler
  threw that away. One malformed document in a batch of 20 added 20 to
  `write_failures` and 0 to `written`, so the counter a `/health` probe watches
  reported a total outage during what was really a 5% error rate, and the audit
  trail recorded 19 users' successfully stored turns as errors against their own
  ids. The batch is now split on `writeErrors[].index` and each half accounted and
  audited separately, with the driver's own `nInserted` as the counter authority.
- **Thread replay could silently reorder a conversation.** `get_thread` sorted on
  `step` first, and the writer deliberately stores `step: null` rather than
  dropping a turn when the durable counter round trip fails ("a logged turn beats a
  lost one"). Null sorts below every number in BSON, so an Atlas hiccup during turn
  4 of 40 did not lose that turn — it *relocated* it to the front of the replay.
  The reader saw a coherent conversation in the wrong order with nothing in the
  output saying so. The sort now leads on `ts`, which is always present and
  monotonic per thread, with `step` as the tie-break.
- **Two `episodes` indexes did not match the queries they served.**
  `ix_episodes_thread_step` was keyed `[thread_id, step]` against a read that
  filters on `(user_id, thread_id)` and sorts on `(ts, step)` — neither half. Both
  it and `ix_episodes_correlation` now mirror their query exactly: equality prefix
  first, then the sort keys in order. `user_id` leads, because every episodic read
  is tenant-scoped and a thread id is not a capability — keyed on `thread_id`
  alone, the isolation filter was a residual predicate the server applied after the
  scan. **Deployment note:** an existing deployment holds the old index under the
  same name. `ensure_indexes` handles this — MongoDB rejects a same-name /
  different-keys `create_index` with code 86, and the handler drops and recreates
  on that code — so an upgrade replaces the index on the next startup with no
  operator action. The rebuild is `background=True`; on a large `episodes`
  collection, expect thread-replay reads to fall back to a collection scan until it
  completes.
- **`max_response_bytes` was declared, documented, and read by nothing.** `limit`
  bounds the result *count*, not its size, and an episodic document carries
  projected message content plus todos plus files touched — a hundred turns is tens
  of megabytes in one MCP frame or HTTP body. All five read methods now share one
  enforcement point (a source-inspection test keeps it single, since the original
  bug's shape was exactly "declared once, honoured nowhere"). Truncation drops
  whole documents from the tail, never bytes, and reports `truncated` and
  `total_count`; at least one document always survives, because the caller has no
  way to ask for a smaller one. An untruncated response is byte-identical to
  before — no `truncated: false` added to every response for a case that did not
  happen.
- **Auto-capture truncation could invert the meaning of a stored memory.** It cut
  the joined string at `max_content_length`, so a large params dict consumed the
  whole budget and the text ended mid-key with the result — the actual outcome of
  the call, and the only part worth remembering — absent entirely. Worse, the cut
  landed inside a repr, so `Result: {'status': 'fail` read as a complete value to
  the embedder, to the enrichment LLM, and to a human reading recall output, and
  was then stored and recalled as fact. Query and result now get separate budgets
  (the result weighted, unused query budget reclaimed) and every truncation is
  marked.
- **Auto-capture tasks could be garbage-collected mid-write.**
  `asyncio.create_task` returns the only strong reference — the loop holds a weak
  one — so a discarded handle let the task be collected while awaiting: the write
  stopped half-done and nothing raised. The race gets *rarer* under light load,
  which is the worst possible profile: it will not reproduce in testing and shows
  up in production as memories that occasionally go missing. Tasks are now retained
  until completion and `drain()` waits for in-flight captures at shutdown.
- **The MCP shell's `/health` returned a bare `{"status": "ok"}`** — a probe
  reporting only that the process accepts sockets. In a dual-transport deployment
  both shells hold the *same* facade, so whichever port a monitor happened to
  target decided whether a dead worker was visible at all. Both shells now build
  the body from one shared function, and `Memory.worker_status()` gained its
  blocking twin (delegating directly rather than hopping the event loop, since a
  health probe must not queue behind the loop it is checking).

### Security

Findings from three independent reviews of 4.1.0. Every one of these was a *silent
wrong answer* rather than a crash: each succeeded, returned or wrote something
plausible, and was wrong in a way that surfaced much later — which is why a green
suite passed over all of them. Where a test asserted the old behaviour, the test
was asserting the bug.

- **Rate limits did not hold under concurrency.** The limiter counted documents in
  a window and *then* inserted its own, with no lock between the two, so every
  request in a burst read the same below-limit count and every one was admitted. A
  20-caller burst against a limit of 5 admitted all 20. The increment and the
  decision are now one atomic `find_one_and_update` on a per-window counter; the
  same burst admits exactly 5. Fixed-window rather than sliding, deliberately —
  see the module docstring for why every sliding shape reintroduces the same race
  one level up. The prior tests asserted the racy shape and all passed, because
  each made a single sequential call, the one access pattern the broken code
  handled correctly.
- **`invalidate(pattern=...)` interpolated caller input into `$regex`.** A pattern
  of `.*` cleared the entire cache for that user while asking to remove one entry,
  and a pathological pattern could pin CPU on backtracking. Patterns are now
  `re.escape`d and matched literally. `invalidate_all` remains the explicit way to
  clear everything.
- **Denied and throttled calls were not audited.** The access check ran before the
  audited block, so a refusal raised past the audit write and left no record — the
  two events an audit log exists to capture were the only two it could not show.
  Refusals are now audited with distinct statuses: `denied` (who the caller is),
  `throttled` (how often they ask), and `error` (a fault in the service), because
  collapsing them makes a policy refusal indistinguishable from a bug precisely
  where the difference matters. `log_activity` keeps its batched success auditing
  and gains the same refusal record.
- **The merge path read its target without a `user_id` filter.** A
  `merge_target_id` pointing at another tenant's memory would have that memory's
  content read into this user's document and the victim's record soft-deleted. The
  fetch and the soft-delete are both scoped to the same user and to a live
  document; merging an already-deleted target would otherwise resurrect content
  the user had deleted.
- **The merge rewrote `content` and left the old `embedding`.** The result read as
  the merged memory and *searched* as its own pre-merge half, so the information
  the merge existed to preserve became unretrievable — silently, since the
  document looks correct in Compass. Content is now re-embedded before the write,
  and an embedding failure leaves the merge unwritten at `merge_pending` for
  retry: a merge that has not happened yet beats a content/embedding pair that
  disagrees.
- **`evolve_memory` reinforced the memory it was evolving.** It runs after the
  document is stored, so its own top hit was itself at similarity ~1.0 — above
  `reinforce_threshold` by construction. Every enrichment pass inflated its own
  importance, incremented its own access count, and never reached the real
  duplicates ranked below it, so genuine near-duplicates were never merged and
  importance drifted upward on nothing. The caller now passes `exclude_id`.
- **`user_id` is bound to the verified token.** Handlers read identity only from
  an injected `Caller`, resolved in one place, so a request cannot name a tenant
  its token does not authorise. With auth off the request's own `user_id` is all
  there is, and naming nobody is a 400 rather than an unscoped query.
- **`wipe_user_data` missed collections.** It cleared three and left the rest, so
  a deletion request reported success while data remained. Every user-scoped
  collection is enumerated in one place, asserted against the collection-names
  module by test, and each is reported separately. `episodes_counters` is keyed by
  a composite `_id` with no top-level `user_id` — the one collection where the
  obvious query silently matches nothing.
- `BedrockLLMProvider.chat()` no longer discards `**kwargs`. It accepted them
  and dropped them, so a caller passing `system`, `inferenceConfig`, or a
  per-call `modelId` got none of it and no error — the parameters simply had no
  effect. Anything the Converse API accepts now passes through.
- Vector-index filter fields are now declared for `user_id`, `thread_id`, and
  `agent_name`. An undeclared field used in a `$vectorSearch` pre-filter makes
  the branch return nothing silently, with no error — the failure mode is an
  empty result set that looks like "no matches."
- Full-text index fields backing exact `equals` filters use the `token` type
  rather than `string`.
- The async worker wraps its sleep inside the `try`, so cancellation during
  sleep unwinds cleanly instead of escaping the handler.
- **Audit records no longer carry raw exception text.** Driver errors quote the
  connection string they failed on, credentials included; provider errors echo
  `Authorization` headers and request payloads; a duplicate-key error quotes the
  key's *value*, which on the episodic path is projected turn content. An audit
  record is a wider and longer-lived audience than a process log — a MongoDB
  collection with a TTL in weeks, readable by anyone holding `admin`. Every audit
  and health error string now goes through `core/redaction.py`, which keeps the
  exception type (actionable, and it can never hold a secret) and a scrubbed,
  length-capped message. Deliberately not a general secret scanner: it targets the
  three shapes that actually occur, and a test guards against *over*-redaction,
  because an audit trail of `[redacted]` trains operators to ignore the field.
- **`/health` was serving `repr(exc)` for a crashed worker, unauthenticated.**
  `/health` is the one route exempt from auth, on purpose — a probe that needs a
  token fails during exactly the incident it exists to detect. That exemption is
  fine for counters and booleans and not fine for an exception repr: a crashed
  worker's exception is usually a driver error, so an unauthenticated endpoint
  would hand the cluster's connection string to anyone who could reach the port.
  `worker_status()` now redacts.
- **API-key lookup is constant-time with respect to the submitted key.** It was a
  plain `dict.get(api_key)` on the raw string, so the work done before the lookup
  varied with the key's length and content. Keys are now hashed to a fixed-length
  digest before the lookup and compared with `hmac.compare_digest`; the raw key is
  never retained, so it cannot surface in a heap dump, a traceback repr, or a
  debugger session. SHA-256 rather than a password KDF on purpose — an API key is
  high-entropy operator-chosen material, not a memorable password, so offline
  brute force does not apply and a per-lookup KDF would add tens of milliseconds
  to every authenticated request. The exposure here was narrow, but this is the
  function that turns a bearer string into an identity.
- **JWTs without `exp` were accepted forever.** PyJWT validates `exp` when the
  claim is present and ignores its absence, so a token minted without one never
  expired. `create_token` always sets it, which is exactly why the gap survived
  review — but the verifier's job is policing tokens it did *not* mint, and anyone
  holding the shared secret could produce an immortal one. With HS256 there is no
  other revocation path: a leaked token with no `exp` is valid until the secret is
  rotated. `exp`, `iat`, and `sub` are now all required, an `exp` present but
  non-integer is refused rather than treated as "never expires", and a
  correctly-signed token missing a required claim logs at warning — that is a
  minting bug in some other service, not a forgery, and the operator needs to know
  which.

### Notes on the write path

`log_activity` deliberately does not go through the facade's audit wrapper. That
wrapper writes one audit record per call, and a turn log is high-volume by
nature — routing it there means logging the agent costs more writes than the
agent. Governance and rate limiting still apply on every call; the worker emits
one audit entry per flushed batch, grouped by `user_id`, because a batch can
span users and misattributing turns would be worse than no audit trail at all.

The worker keeps three guarantees worth stating: when the queue is full it drops
the *oldest* turn, never the newest; a failure of the durable step counter
inserts the document with a null step rather than dropping it; and the embedding
is generated before `search_text` is assigned, so an embedding failure leaves
neither field rather than a searchable document with no vector.
