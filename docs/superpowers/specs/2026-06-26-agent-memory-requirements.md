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
  flush the audit buffer, and close the database connection.
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
