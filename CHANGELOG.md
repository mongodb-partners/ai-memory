# Changelog

All notable changes to `agent-memory` are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Governance profiles: `power_user` gains full episodic read/write;
  `end_user` may log and replay its own threads but not query by correlation id
  (trace ids come from operators, not from a user's own session).
  `set_activity_retention` stays admin-only.

### Fixed

- `voyage-4`, `voyage-4-large`, and `voyage-4-lite` are recognized in the model
  dimension table (all 1024), so `embedding_dimension` auto-aligns for them
  instead of staying at the 1536 default. An unrecognized model left the default
  in place, which builds a 1536-dim index for a 1024-dim embedder — and that
  mismatch does not raise, it just returns nothing from recall. The README now
  documents that the Atlas embeddings gateway and the public Voyage API need
  different keys, URLs, and models.
- Vector-index filter fields are now declared for `user_id`, `thread_id`, and
  `agent_name`. An undeclared field used in a `$vectorSearch` pre-filter makes
  the branch return nothing silently, with no error — the failure mode is an
  empty result set that looks like "no matches."
- Full-text index fields backing exact `equals` filters use the `token` type
  rather than `string`.
- The async worker wraps its sleep inside the `try`, so cancellation during
  sleep unwinds cleanly instead of escaping the handler.

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
