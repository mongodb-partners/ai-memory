# agent-memory SP3 — Requirements (EARS)

**Date:** 2026-06-26
**Workflow:** Brownfield enhancement/refactor (tests exist) — full revamp
**Design source:** `2026-06-26-agent-memory-library-core-design.md`
**Substrate:** memory-mcp v3.2.1, restructured into `agent_memory/` (commit 82cd583)

This document derives atomic, testable requirements from the SP3 design. The
substrate's service internals are unchanged; requirements cover the **new**
surface (facade, sync wrapper, exceptions, config, providers, shells,
dual-transport) and the **invariants** that existing behavior must preserve.

---

## Codebase Context

```
Stack:          Python 3.11+, FastMCP, PyMongo (async), Pydantic v2, pytest (asyncio_mode=auto)
Patterns:       constructor-injected services; ProviderManager factory (match on name);
                ServiceRegistry singleton + per-tool orchestration (check_access → service → audit)
Affected area:  NEW agent_memory/memory.py, exceptions.py, config.py(.from_env),
                providers/{openai,anthropic}.py, providers/manager.py, shells/{mcp,rest}/;
                services gain extracted methods (hybrid_search, admin health/wipe)
Existing tests: 344 service/core/provider/tool/auth unit tests pass under agent_memory.*
                (test_server.py + test_packaging.py intentionally superseded)
Test state:     tests exist — standard brownfield
Prior specs:    design doc (this SP3)
Closest analog: tools/memory_tools.py orchestration → absorbed by AsyncMemory._run
```

---

## New Requirements

### Exceptions (`agent_memory/exceptions.py`)

- **REQ-E-001:** THE SYSTEM SHALL define `AccessError`, `RateLimitError`,
  `NotFoundError`, and `ConfigError` exception classes, all subclassing a common
  `MemoryError` base.
- **REQ-E-002:** THE SYSTEM SHALL define `RateLimitError` as a subclass of
  `AccessError`, so a caller catching `AccessError` also catches rate-limit
  rejections.

### Configuration (`agent_memory/config.py`)

- **REQ-E-010:** THE SYSTEM SHALL provide a `MemoryConfig` Pydantic settings
  object exposing all memory-mcp config fields plus `openai_api_key`,
  `openai_base_url`, `openai_model`, `openai_embedding_model`,
  `anthropic_api_key`, `anthropic_base_url`, `anthropic_model`, and
  `workers_in_process` (default `True`).
- **REQ-E-011:** THE SYSTEM SHALL provide `MemoryConfig.from_env()` returning a
  config populated from environment variables, backward-compatible with
  memory-mcp's variable names (e.g. `MONGODB_CONNECTION_STRING`).
- **REQ-E-012:** `from_env()` THE SYSTEM SHALL default `llm_provider` and
  `embedding_provider` to `bedrock`.

### AsyncMemory facade (`agent_memory/memory.py`)

- **REQ-E-020:** THE SYSTEM SHALL provide `AsyncMemory.create(config)` that
  initializes the database, ensures Stage-1 indexes, builds providers, and
  instantiates services — equivalent to today's MCP `lifespan` startup.
- **REQ-E-021:** WHEN `config.workers_in_process` is `True` THE SYSTEM SHALL
  start the enrichment, consolidation, and audit-flush workers during `create()`.
- **REQ-E-022:** WHEN `config.workers_in_process` is `False` THE SYSTEM SHALL NOT
  start in-process workers and SHALL emit a warning log that reactive work is
  disabled.
- **REQ-E-023:** `AsyncMemory.close()` THE SYSTEM SHALL cancel running workers,
  flush the audit buffer, and close the database connection, in that order — a
  worker cancelled mid-write returns its batch to the buffer, so the flush must
  follow the cancellation to write it.
- **REQ-E-023a:** `AuditService.flush()` THE SYSTEM SHALL, on return, guarantee
  that every entry buffered when it was called has reached MongoDB or the
  fallback file, including entries a concurrent flush had already removed from
  the buffer. Concurrent flushes SHALL NOT each issue their own write.
  WHERE a flush is cancelled THE SYSTEM SHALL return its batch to the buffer
  rather than discard it or write it to the fallback file.
- **REQ-E-024:** THE SYSTEM SHALL support `async with AsyncMemory.create(cfg) as m:`
  (async context manager calling `close()` on exit).
- **REQ-E-025:** Every public facade method THE SYSTEM SHALL route through one
  orchestration path that performs access-check → service call → audit log.
- **REQ-E-026:** WHEN governance denies an operation THE SYSTEM SHALL raise
  `AccessError`.
- **REQ-E-027:** WHEN the rate limit is exceeded THE SYSTEM SHALL raise
  `RateLimitError`.
- **REQ-E-028:** WHEN a wrapped operation succeeds THE SYSTEM SHALL write an audit
  log entry with status `success`; IF it raises THEN THE SYSTEM SHALL write an
  audit entry with status `error` and re-raise.
- **REQ-E-029:** THE SYSTEM SHALL expose facade methods `add`, `recall`,
  `search`, `delete`, `check_cache`, `store_cache`, `invalidate_cache`,
  `remember_decision`, `recall_decision`, `health`, `wipe_user_data`.
  (`search_web`/Tavily web search was removed post-SP3 as out of scope for a
  memory library.)
- **REQ-E-030:** `recall` THE SYSTEM SHALL return curated, importance-ranked
  memories (delegating to `MemoryService.recall`); `search` THE SYSTEM SHALL
  return raw hybrid `$rankFusion` (RRF) matches with scores (delegating to
  `MemoryService.hybrid_search`).
- **REQ-E-031:** WHEN `create()` runs THE SYSTEM SHALL validate that
  `config.embedding_dimension` matches the dimension the configured embedder
  emits; IF they differ THEN THE SYSTEM SHALL raise `ConfigError` before any
  vector write.

### Memory sync wrapper (`agent_memory/memory.py`)

- **REQ-E-040:** THE SYSTEM SHALL provide a `Memory` class that runs an
  `AsyncMemory` core on a dedicated daemon-thread event loop and exposes a
  blocking twin for every async facade method.
- **REQ-E-041:** `Memory` methods THE SYSTEM SHALL return correct results when
  called from a plain synchronous context AND when called from inside an already
  running event loop (notebook scenario), without raising "loop already running".
- **REQ-E-042:** `Memory.close()` THE SYSTEM SHALL stop the background loop and
  release its thread.

### Providers (`agent_memory/providers/`)

- **REQ-E-050:** THE SYSTEM SHALL provide `OpenAILLMProvider` implementing `chat`,
  `assess_importance`, `generate_summary` via the `openai` async SDK, honoring an
  optional `openai_base_url` (Grove gateway).
- **REQ-E-051:** THE SYSTEM SHALL provide `OpenAIEmbeddingProvider` implementing
  `generate_embedding` and `generate_embeddings_batch`, supporting
  `text-embedding-3-small` (1536) and `text-embedding-3-large` (3072), honoring
  `openai_base_url`.
- **REQ-E-052:** THE SYSTEM SHALL provide `AnthropicLLMProvider` implementing the
  three `LLMProvider` methods via the `anthropic` async SDK, honoring an optional
  `anthropic_base_url`.
- **REQ-E-053:** `ProviderManager` THE SYSTEM SHALL construct providers for
  `llm_provider ∈ {bedrock, openai, anthropic}` and `embedding_provider ∈
  {bedrock, voyage, openai}`.
- **REQ-E-054:** IF a non-default provider is selected but its SDK is not
  installed THEN THE SYSTEM SHALL raise `ConfigError` whose message includes the
  install hint (e.g. `pip install agent-memory[openai]`).

### MCP shell (`agent_memory/shells/mcp/`)

- **REQ-E-060:** THE SYSTEM SHALL expose the MCP tools by delegating each to the
  corresponding `AsyncMemory` method (no business logic in the tool).
- **REQ-E-061:** WHEN a facade call raises `AccessError` (incl. `RateLimitError`)
  THE SYSTEM SHALL return `{"error": <message>}` from the MCP tool.
- **REQ-E-062:** THE SYSTEM SHALL create the `AsyncMemory` instance in the MCP
  server lifespan and close it on shutdown.
- **REQ-E-063:** WHEN auto-capture is enabled THE SYSTEM SHALL persist captured
  turns by calling `AsyncMemory.add` (not a service directly).

### REST shell (`agent_memory/shells/rest/`)

- **REQ-E-070:** THE SYSTEM SHALL expose FastAPI routes mirroring the facade:
  `POST /memories`, `GET /memories/recall`, `GET /memories/search`,
  `DELETE /memories`, `POST /decisions`, `GET /decisions`, `GET /health`.
- **REQ-E-071:** WHEN a facade call raises `RateLimitError` THE SYSTEM SHALL
  respond `429`; WHEN it raises `AccessError` (non-rate-limit) THE SYSTEM SHALL
  respond `403`; WHEN it raises `NotFoundError` THE SYSTEM SHALL respond `404`.
- **REQ-E-072:** THE SYSTEM SHALL reuse the existing `auth/` token verifier for
  REST authentication (not a reimplementation).

### Dual-transport entrypoint & packaging

- **REQ-E-080:** WHEN `TRANSPORT=both` THE SYSTEM SHALL serve the MCP and REST
  shells from a single `AsyncMemory` instance in one process; `TRANSPORT=mcp` and
  `TRANSPORT=rest` SHALL serve only the named shell.
- **REQ-E-081:** THE SYSTEM SHALL declare optional-dependency groups `[openai]`,
  `[anthropic]`, `[rest]`, `[all]`, with the package named `agent-memory` at
  version `4.0.0` and the package id `agent_memory`.
- **REQ-E-082:** THE SYSTEM `agent_memory.__init__` SHALL export `Memory`,
  `AsyncMemory`, `MemoryConfig`, and the exception classes.
- **REQ-E-082a:** THE CI lint gate SHALL cover every tracked Python file, not
  `agent_memory/` alone, and SHALL be asserted both as "the workflow passes no
  narrowing path" and as "ruff's `--show-files` reaches `tests/`, `scripts/`, and
  `examples/`" — the second because an `extend-exclude` narrows the scope while the
  workflow still reads correctly. The tests SHALL NOT be exempted via `ignore` or
  `per-file-ignores`: a test is what certifies the library, so a weakened assertion
  is the one defect with nothing above it. `RUF002` is ignored (en dashes in prose
  docstrings); `RUF001`/`RUF003` SHALL NOT be, because they cover identifiers,
  literals, and inline comments.
- **REQ-E-082b:** THE artifact secret scan SHALL iterate over `dist/*.tar.gz`
  rather than expanding the glob into one `tar tzf` invocation, and SHALL fail WHEN
  `dist/` contains no sdist. `tar` reads only its first argument as the archive and
  searches for the remainder inside it, so two artifacts present at once make the
  scan exit non-zero for an unrelated reason — a security gate whose failure invites
  being silenced rather than fixed.

### Episodic memory — the agent activity log (`agent_memory/services/episodic.py`)

Added in 4.1.0. The fourth memory tier: an append-only, per-turn record of what
the agent *did*, as distinct from what it *knows*. Ported from
`langchain-mongodb-agent-log` as framework-neutral code.

**Projection (`core/projection.py`)**

- **REQ-E-090:** THE SYSTEM SHALL project messages to exactly seven keys in
  order — `type`, `content`, `tool_calls`, `tool_call_id`, `usage`, `model_id`,
  `finish_reason` — accepting either a `Mapping` or an attribute-bearing object
  as the message.
- **REQ-E-091:** WHEN a message's `content` is a list of blocks THE SYSTEM SHALL
  keep only `{"type": "text"}` blocks and bare strings, discarding all others.
- **REQ-E-092:** WHEN projected text exceeds the configured cap THE SYSTEM SHALL
  truncate it and append a marker reporting the original length; WHEN the cap is
  less than or equal to zero THE SYSTEM SHALL NOT truncate.
- **REQ-E-093:** THE SYSTEM SHALL project todos to `id` / `content` / `status`,
  clamping any status outside `pending|in_progress|completed` to `pending` rather
  than raising, and accepting `text` as an alias for `content`.
- **REQ-E-094:** THE SYSTEM SHALL derive `files_touched` from filesystem-write
  tool calls on assistant messages only, keeping the last write per path, sorted
  by path, labelling `op` as `write` for create-tools and `edit` otherwise.
- **REQ-E-095:** THE SYSTEM SHALL classify a turn as a final step only when its
  last assistant message carries no `tool_calls`, and SHALL build `search_text`
  from the first human message and the last assistant message, returning the
  empty string when either is absent.

**Per-user scoping (`core/context.py`, `core/correlation.py`)**

- **REQ-E-096:** THE SYSTEM SHALL expose ambient user scoping via a
  `ContextVar`, isolated per asyncio Task and per thread, restoring the previous
  value on scope exit including on exception.
- **REQ-E-097:** THE SYSTEM SHALL derive a correlation id in precedence order —
  explicit `correlation_id`, W3C `traceparent` trace id, `x_request_id`, then a
  fresh UUID4 — and SHALL never return an empty string.

**Write path (`services/episodic.py`, `services/episodic_worker.py`)**

- **REQ-E-100:** WHEN `log_activity` is called THE SYSTEM SHALL enqueue the
  projected document and return without awaiting the database or the embedding
  provider.
- **REQ-E-101:** IF `user_id` or `thread_id` is missing or empty THEN THE SYSTEM
  SHALL NOT write a document.
- **REQ-E-102:** WHEN the bounded queue is full THE SYSTEM SHALL evict the
  oldest pending turn, count the eviction, and retain the newest turn.
- **REQ-E-103:** THE SYSTEM SHALL assign a durable monotonic `step` per thread
  and set `parent_step` to `step - 1`, or to `null` at step zero.
- **REQ-E-104:** IF the durable step counter fails THEN THE SYSTEM SHALL insert
  the document with `step` and `parent_step` set to `null` rather than dropping
  the turn.
- **REQ-E-105:** WHEN a turn is a final step THE SYSTEM SHALL generate the
  embedding before assigning `search_text`, so an embedding failure leaves
  neither field present.
- **REQ-E-106:** THE SYSTEM SHALL swallow and count every write and embedding
  failure, exposing them through `stats()`, and SHALL NOT propagate an exception
  to the caller of `log_activity`.
- **REQ-E-107:** THE SYSTEM SHALL provide a bounded `flush(timeout)` returning a
  boolean and an idempotent `close()`, neither of which raises.

**Read path**

- **REQ-E-110:** THE SYSTEM SHALL provide `recall_activity` using `$rankFusion`
  over a vector branch and a full-text branch with the `user_id` filter inside
  both branches.
- **REQ-E-111:** THE SYSTEM SHALL provide `get_thread` ordered by `step` and
  `get_activity_by_correlation` filtered by `user_id` and `correlation_id`.
- **REQ-E-112:** THE SYSTEM SHALL coerce `ObjectId` and `datetime` values to
  JSON-safe strings on read, recursing into nested lists as well as dicts.

**Storage and governance**

- **REQ-E-115:** THE SYSTEM SHALL store episodic records in a dedicated
  `episodes` collection with five B-tree indexes, a TTL index on `ts`, a vector
  index on `embedding` declaring `user_id` / `thread_id` / `agent_name` as filter
  fields, and a full-text index on `search_text`.
- **REQ-E-116:** THE SYSTEM SHALL provide `set_activity_retention` implemented
  with `collMod` and a `create_index` fallback, which SHALL NOT raise. WHERE the
  change fails THE SYSTEM SHALL return `status: "error"` carrying a redacted
  reason, and SHALL record that call in the audit log with status `"error"` rather
  than `"success"`.
- **REQ-E-117:** WHEN `log_activity` is called THE SYSTEM SHALL enforce
  governance and rate limits but SHALL NOT emit one audit record per call;
  instead it SHALL emit one audit record per flushed batch.
- **REQ-E-118:** THE SYSTEM SHALL grant the episodic operations to the
  `power_user` and `end_user` governance profiles, and `seed_defaults` SHALL add
  operations missing from an already-seeded profile.
- **REQ-E-119:** WHEN parsing an importance score from an LLM reply THE SYSTEM
  SHALL accept both a 1–10 rating and a 0.0–1.0 fraction, inferring the scale from
  the value, and SHALL clamp the result to [0.1, 1.0]. WHERE the reply contains no
  parseable number THE SYSTEM SHALL return the default of 0.5. WHERE the value is
  exactly 1 — ambiguous between the two scales — THE SYSTEM SHALL resolve it to
  1.0, because the opposite reading places the memory at the forgetting threshold
  and is therefore unrecoverable.
- **REQ-E-120:** THE SYSTEM SHALL provide `LLMProvider.user_turn(text)` returning
  a one-message list in the provider's own native shape, and
  `LLMProvider.complete(text, **kwargs)` which sends it via `chat` and forwards
  every keyword argument. WHERE a provider's API requires content blocks rather
  than a string — Bedrock Converse — it SHALL override `user_turn`. Every
  library-internal caller holding only a prompt string SHALL use `complete` and
  SHALL NOT construct a `messages` list itself, because both shapes are a
  `list[dict]` and a mismatch is therefore invisible to typing, to review, and to
  any test that mocks `chat`.
- **REQ-E-121:** THE SYSTEM SHALL provide `is_usable_summary(summary, content)`
  and SHALL NOT store a summary that fails it, leaving the field absent so every
  reader falls back to `content`. WHERE `content` is shorter than
  `MIN_SUMMARIZABLE_CHARS` THE SYSTEM SHALL NOT call the model at all, because a
  single conversational turn is already shorter than any summary of it. A reply
  SHALL be rejected WHEN it is empty, WHEN it is at least as long as its source,
  or WHEN it matches a known refusal form — `generate_summary` returns whatever the
  model said, and on short input what it says is frequently "I don't see the
  original text that needs to be summarized", which is a successful call returning
  a well-formed string. Readers preferring `summary` over `content` SHALL apply the
  same check, so documents written before this guard still read correctly.

### Non-functional

- **REQ-NF-COV:** THE SYSTEM SHALL maintain at minimum 80% line coverage on
  `agent_memory/memory.py`, `exceptions.py`, `config.py`, and the new provider
  modules, with no individual new module below 60%.

---

## Unchanged Behavior (Invariants — regression protection)

The 344 ported service/core/provider/tool tests are the regression net. These
invariants name behaviors the revamp must NOT change:

- **INV-001:** THE SYSTEM SHALL CONTINUE TO store STM docs with a 24h TTL and
  queue LTM candidates with `enrichment_status=pending` on `add`.
- **INV-002:** THE SYSTEM SHALL CONTINUE TO rank `recall` results by recency,
  importance, and relevance (MemoryService.recall internals unchanged).
- **INV-003:** THE SYSTEM SHALL CONTINUE TO perform hybrid retrieval via
  MongoDB `$rankFusion` RRF combining `$vectorSearch` and `$search`.
- **INV-004:** THE SYSTEM SHALL CONTINUE TO soft-delete by default and require
  `confirm=true` for bulk deletes, with `dry_run` preview.
- **INV-005:** THE SYSTEM SHALL CONTINUE TO map search operations to
  `max_searches_per_day` and write operations to `max_memories_per_day` when a
  governance profile is present.
- **INV-006:** THE SYSTEM SHALL CONTINUE TO run enrichment (importance + summary
  + evolution: reinforce/merge/create) and consolidation (STM→LTM promotion,
  forgetting) as background workers when workers are in-process.
- **INV-007:** THE SYSTEM SHALL CONTINUE TO buffer audit entries and flush them
  on interval and on shutdown.
- **INV-008:** THE SYSTEM SHALL CONTINUE TO construct Bedrock and Voyage
  providers with their existing behavior.

---

## Premortem

| # | Failure mode | Mitigation (EARS) |
|---|---|---|
| 1 | `RateLimitError` subclasses `AccessError`; REST catches `AccessError` first, collapsing 429 into 403. Backoff never triggers. | WHEN both handlers are present THE SYSTEM SHALL order the `RateLimitError` handler before `AccessError`, verified by a test asserting a rate-limited call yields 429 (REQ-E-071). |
| 2 | `workers_in_process=False` silently drops enrichment; memories persist but never enrich. Looks healthy, degrades quietly. | WHEN workers are disabled THE SYSTEM SHALL emit a warning log, asserted by a test (REQ-E-022). |
| 3 | `embedding_dimension` mismatches the live embedder; writes silently corrupt the vector index (never match `numDimensions`). | WHEN `create()` runs THE SYSTEM SHALL raise `ConfigError` on dimension mismatch before any write (REQ-E-031). |
| 4 | Sync `Memory` wrapper called inside a running loop (Jupyter) raises "loop already running" or deadlocks. | THE SYSTEM SHALL run the core on a separate daemon-thread loop, asserted by a notebook-scenario test (REQ-E-041). |
| 5 | A non-default provider is configured but its SDK is absent; failure surfaces as an opaque `ImportError` deep in a request. | IF the SDK is missing THEN THE SYSTEM SHALL raise `ConfigError` with an install hint at construction (REQ-E-054). |

Each premortem row maps to at least one acceptance test below.

---

## Boundary Inventory

| # | Boundary | From | To | Acceptance test |
|---|---|---|---|---|
| 1 | Sync→Async | `Memory` (sync) | `AsyncMemory` coro on bg loop | TC-SYNC-001/002 |
| 2 | Library→Service | `AsyncMemory` facade | constructor-injected services | TC-FAC-001..012 |
| 3 | MCP transport | FastMCP tool | `AsyncMemory` method | TC-MCP-001..003 |
| 4 | HTTP entry | FastAPI route | `AsyncMemory` method | TC-REST-001..005 |
| 5 | Provider SDK | provider class | OpenAI/Anthropic async SDK (mocked) | TC-PROV-001..006 |
| 6 | Embedder↔index | `create()` dim guard | live embedder dimension | TC-FAC-DIM-001 |
| 7 | Persisted hybrid retrieval | facade `search` | Atlas `$rankFusion` (live) | TC-INTEG-LIVE-001 (Atlas-gated, skipped without creds) |

Boundary 7 cannot be exercised by mongomock; it is covered by the optional
Atlas-gated integration tier (design §8) and explicitly skipped in CI without
credentials. All other boundaries have in-process acceptance tests.
