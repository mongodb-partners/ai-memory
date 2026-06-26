# agent-memory — Library Core + MCP + REST Revamp (SP3)

**Date:** 2026-06-26
**Status:** Approved design, pre-implementation
**Repo:** `mongodb-partners/ai-memory` → to be renamed `mongodb-partners/agent-memory`
**Working path:** `/Users/mdf/git/partners/agent-memory`
**Branch:** `revamp/agent-memory-v4`
**Target version:** `4.0.0` (major; supersedes memory-mcp 3.x)

---

## 1. Context & Goal

`mongodb-partners/ai-memory` is the original project from which `memory-mcp` was
later derived. memory-mcp (v3.2.1) is now the mature codebase: a MongoDB
Atlas–backed MCP server for AI agent memory, with two-tier STM/LTM storage,
hybrid vector + full-text retrieval (RRF), background enrichment/consolidation
workers, governance, rate limiting, audit logging, and Bedrock/Voyage providers.

This effort revamps `ai-memory` into **`agent-memory`**: a single codebase whose
**core is a programmatic memory library**, with the MCP server and a new REST API
as thin shells over that core. memory-mcp's source is the substrate; the old
ai-memory FastAPI code is replaced entirely (fresh rewrite, no data migration).

This is **sub-project SP3** of a larger sequenced roadmap:

- **SP3 (this spec):** library facade + MCP shell + REST API + OpenAI/Anthropic
  providers. The foundation.
- **SP1 (later):** prospective memory — pending-task documents, scheduled-job
  worker (time triggers), Change Stream listener / Atlas Trigger templates
  (event triggers). Completes the blog's four cognitive-memory quadrants.
- **SP2 (later):** retrieval upgrades adapted from mem0 — entity
  extraction/linking, first-class memory typing (semantic/episodic/procedural),
  temporal query reasoning, richer filter operators.

SP3 is designed so SP1 and SP2 land as new facade methods + thin shell exposure,
inheriting the orchestration, lifecycle, and worker seams built here.

### Why this serves the "MongoDB is the ultimate memory system" story

Scoped to MongoDB Atlas and MongoDB products, this codebase showcases MongoDB as
the entire memory stack (vector search, full-text search, RRF hybrid retrieval,
TTL decay, operational reads/writes, and — via SP1 — Change Streams / Atlas
Triggers). Unlike backend-agnostic memory libraries, every cognitive memory type
is MongoDB-native here. Atlas is the system of record and the reactive-trigger
substrate; the library/service tier is the broader application backend. Atlas
Triggers/Functions are used as a complementary reactive layer (short-running,
event-driven), **not** as the primary general-purpose runtime.

---

## 2. Architecture

**Facade-as-core, three shells, one Atlas backend.**

```text
        from agent_memory import Memory, AsyncMemory, MemoryConfig
                            │
   ┌────────────────────────┼────────────────────────┐
   │ Library import         │ MCP shell               │ REST shell (FastAPI)
   │ (native embed)         │ (thin FastMCP tools)    │ (thin routes)
   └────────────────────────┴────────────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │  AsyncMemory  (facade / core)   │  owns orchestration:
            │  add · recall · search · ...    │  access-check → service → audit
            │  Memory = sync wrapper over it  │  + worker lifecycle
            └───────────────┬─────────────────┘
                            │ constructor-injected services (internals unchanged)
   MemoryService · CacheService · DecisionService · GovernanceService
   · RateLimiter · AuditService · ProviderManager
                            │
            ┌───────────────▼─────────────────┐
            │   MongoDB Atlas (single backend) │  vector + FTS + RRF + TTL
            │   + optional Atlas Triggers      │  (reactive layer, SP1)
            └──────────────────────────────────┘
```

### Package layout

```text
agent_memory/
  __init__.py          # exports Memory, AsyncMemory, MemoryConfig, exceptions
  memory.py            # AsyncMemory facade + Memory sync wrapper
  config.py            # MemoryConfig (programmatic) + .from_env()
  exceptions.py        # AccessError, RateLimitError(AccessError), NotFoundError, ConfigError, ...
  core/                # database, migrations, collections (registry deleted)
  providers/
    base.py            # EmbeddingProvider, LLMProvider ABCs (unchanged)
    manager.py         # factory: match on provider name
    bedrock.py         # existing
    voyage.py          # existing (embeddings)
    openai.py          # NEW: OpenAILLMProvider + OpenAIEmbeddingProvider
    anthropic.py       # NEW: AnthropicLLMProvider (no embeddings API)
  services/            # memory, cache, decision, governance, rate_limiter,
                       # audit, enrichment, consolidation, audit_flush,
                       # auto_capture, prompt_library (internals unchanged)
  shells/
    mcp/               # thin FastMCP tools (today's tools/, slimmed)
    rest/              # thin FastAPI routes (new)
auth/                  # existing token verifier / API-key manager (reused)
```

**Key principle:** services keep their current constructor-injected design (no
MCP/REST imports). The facade is the only component that knows about
orchestration. Shells know only the facade.

---

## 3. The `AsyncMemory` facade (core)

The facade absorbs the orchestration pattern currently duplicated in every MCP
tool (`ServiceRegistry.get()` → `check_access()` → call service →
`audit_service.log()`), so every consumer gets identical access control and
auditing through one code path.

### Lifecycle

```python
class AsyncMemory:
    @classmethod
    async def create(cls, config: MemoryConfig) -> "AsyncMemory":
        # 1. DatabaseManager.initialize(config)   → Atlas connection
        # 2. ensure_indexes(db)                   → Stage-1 indexes (blocking)
        # 3. ProviderManager(config)              → embedding + LLM providers
        #    + validate embedding_dimension matches the live embedder
        #      (known per-model dim, else probe one embedding's length);
        #      mismatch → ConfigError before any vector write
        # 4. instantiate services (memory, cache, decision, governance,
        #    rate_limiter, audit)
        # 5. seed defaults (governance, prompts, decisions) — best-effort
        # 6. start workers (enrichment, consolidation, audit-flush)
        #    IF config.workers_in_process
        # 7. schedule Stage-2 Atlas Search indexes (background)
        ...

    async def close(self) -> None:
        # cancel workers, flush audit, close db — mirrors today's lifespan teardown
        ...
```

`create()`/`close()` are today's FastMCP `lifespan` startup/shutdown, lifted out
and made callable by anyone. Also supports
`async with AsyncMemory.create(cfg) as m:`.

### Orchestration wrapper

```python
async def _run(self, user_id, operation, category, coro_factory, **audit_fields):
    # _check_access runs governance THEN rate limit, raising the matching typed
    # error so shells can distinguish 403 (denied) from 429 (throttled):
    #   - governance denial → AccessError
    #   - rate-limit exceeded → RateLimitError (subclass of AccessError)
    await self._check_access(user_id, operation)
    start = time.time()
    try:
        result = await coro_factory()
        await self._audit(user_id, category, operation, "success", start, **audit_fields)
        return result
    except Exception as e:
        await self._audit(user_id, category, operation, "error", start, error=str(e))
        raise
```

### Public method surface

| Facade method | Wraps | Replaces MCP tool |
|---|---|---|
| `add(user_id, conversation_id, messages)` | `memory_service.store` | `store_memory` |
| `recall(user_id, query, *, tier, memory_type, tags, limit)` | `memory_service.recall` | `recall_memory` |
| `search(user_id, query, *, tier, limit, memory_type, tags)` | hybrid RRF search | `hybrid_search` |
| `delete(user_id, *, memory_id, tags, time_range, confirm, dry_run)` | `memory_service.delete` | `delete_memory` |
| `check_cache(user_id, query, *, similarity_threshold)` | `cache_service.check` | `check_cache` |
| `store_cache(user_id, query, response)` | `cache_service.store` | `store_cache` |
| `invalidate_cache(user_id, *, pattern, invalidate_all)` | `cache_service.invalidate` | `cache_invalidate` |
| `remember_decision(user_id, key, value, *, ttl_days)` | `decision_service.store` | `store_decision` |
| `recall_decision(user_id, key)` | `decision_service.recall` | `recall_decision` |
| `search_web(user_id, query)` | Tavily | `search_web` |
| `health()` | admin | `memory_health` |
| `wipe_user_data(user_id)` | admin | `wipe_user_data` |

**`recall` vs `search` — the contract.** They share inputs but differ in intent
and output, and both are kept because their internals differ:

- **`recall`** is the curated memory read: it returns evolution-aware,
  importance-ranked memories (post-enrichment: reinforced/merged records,
  importance score honored), shaped for an agent asking "what do I know about
  X?" This is the default for agent loops.
- **`search`** is the raw retrieval primitive: hybrid vector + full-text
  `$rankFusion` (RRF) over the store, returning fused-relevance matches with
  scores and no curation. This is for callers that want ranked hits, debugging,
  or to build their own re-ranking on top.

In short: `recall` answers a question; `search` returns ranked documents. The
shells document both with this one-line distinction.

### Design decisions

- **Facade raises typed exceptions** (`AccessError`, `RateLimitError`,
  `NotFoundError`, `ConfigError`, …). `RateLimitError` subclasses `AccessError`
  so callers can catch the base while shells distinguish throttling (429) from
  denial (403). Each shell translates to its own error shape; error policy stays
  out of the core.
- **No global singleton.** `ServiceRegistry` is deleted; the facade is the
  container. Clean because this is a fresh rewrite (no migration shim needed).
- **Pluggable worker lifecycle.** `config.workers_in_process=True` (default) runs
  enrichment/consolidation/audit-flush in-process. `False` means an external
  runtime (Atlas Triggers/Functions, separate worker process) owns reactive
  work. This is the SP1 seam, designed now. **In SP3, `False` is purely a disable
  switch** — no external consumer ships until SP1, so setting it leaves the
  enrichment/consolidation queues undrained (memories persist but are never
  enriched, promoted, or forgotten). `create()` logs a warning when
  `workers_in_process=False` so this is never silent.

### Sync `Memory` wrapper

```python
class Memory:
    def __init__(self, config: MemoryConfig):
        # dedicated event loop on a daemon thread (safe in notebooks/Jupyter)
        ...
    def add(self, *a, **k):    return self._submit(self._async.add(*a, **k))
    def recall(self, *a, **k): return self._submit(self._async.recall(*a, **k))
    # every async method gets a blocking twin
    # (_submit schedules the coro on the background loop and blocks for the
    #  result — distinct from AsyncMemory._run, which is the orchestration wrapper)
```

`AsyncMemory` is the real implementation (used directly by the MCP/REST shells,
which are async-native). `Memory` runs the async core on a dedicated background
event loop so sync consumers (scripts, notebooks, sync agent frameworks) work out
of the box. Mirrors mem0's `Memory`/`AsyncMemory` split.

---

## 4. Providers (OpenAI + Anthropic, Grove-ready)

memory-mcp already has the right seam: `ProviderManager` factory matches on
`config.llm_provider` / `config.embedding_provider`, backed by `LLMProvider`
(`chat`, `assess_importance`, `generate_summary`) and `EmbeddingProvider`
(`generate_embedding`, `generate_embeddings_batch`) ABCs. Adding providers is new
classes + new `match` arms — no architectural change.

### New providers

- **`OpenAILLMProvider`** — official `openai` async SDK (`AsyncOpenAI`).
  Implements `chat`, `assess_importance`, `generate_summary`. Optional
  `base_url`.
- **`OpenAIEmbeddingProvider`** — `text-embedding-3-small` (1536) /
  `text-embedding-3-large` (3072). Optional `base_url`.
- **`AnthropicLLMProvider`** — official `anthropic` async SDK (`AsyncAnthropic`).
  Implements the same three methods. Optional `base_url`. **No embeddings**
  (Anthropic has no embeddings API); pairs with an OpenAI/Voyage/Bedrock embedder.

Custom, maintainable code matching memory-mcp's existing provider style — **not**
LangChain — to keep the dependency surface light.

### Grove (MongoDB Azure-based AI gateway)

Following the reference project
(`mongodb-langchain-deep-agents-retail-qs/src/deep_agent/models.py`), Grove is
**not a separate provider**: it is the same provider SDK with `base_url`/endpoint
pointed at the gateway (exactly how Voyage uses `VOYAGE_BASE_URL`). Each provider
exposes an optional `*_base_url`, so Grove access is free — Grove offers
OpenAI- and Anthropic-compatible endpoints.

### Provider matrix

- **LLM:** `bedrock` (default) | `openai` | `anthropic`
- **Embeddings:** `bedrock` (default) | `voyage` | `openai`
- Each gateway-routable via `*_base_url`.

**Defaults.** `from_env()` defaults both `llm_provider` and `embedding_provider`
to `bedrock`. Bedrock (`boto3`) therefore stays in base `dependencies`; the
`openai` and `anthropic` SDKs are opt-in extras (§9). Choosing a non-default
provider without installing its extra raises `ConfigError` with the install hint
(e.g. `pip install agent-memory[openai]`).

### Embedding dimension coupling

Embedding dimension is provider-coupled (Titan 1536, OpenAI `-3-small` 1536 /
`-3-large` 3072, Voyage configurable). The Atlas vector index `numDimensions`
must match the active embedder. `MemoryConfig.embedding_dimension` is
authoritative; the migration layer reads it. **Switching embedders requires
re-provisioning the vector index.** This is documented explicitly.

**Startup guard.** `create()` (step 3) validates that `embedding_dimension`
matches what the configured embedder actually emits — using a known per-model
dimension table where available, otherwise probing the length of a single test
embedding. A mismatch raises `ConfigError` at startup, turning a silent
vector-index corruption (writes that never match the index `numDimensions`)
into a fast, legible failure before any data is written.

---

## 5. Shells (MCP + REST)

Both shells are thin translation layers: transport, auth wiring, and error-shape
translation only. Zero business logic.

### MCP shell (`shells/mcp/`)

Today's `tools/` slimmed down:

```python
@mcp.tool(name="recall_memory", description="...")
async def recall_memory(user_id: str, query: str, tier: str | None = None,
                        limit: int = 10) -> dict:
    try:
        return await app.recall(user_id, query, tier=tier, limit=limit)
    except AccessError as e:
        return {"error": str(e)}      # MCP's existing error convention
```

The `app` (`AsyncMemory`) is created in FastMCP `lifespan` and closed on
shutdown. Auto-capture middleware still wraps tools but calls `app.add(...)`
instead of reaching into a service. Health route unchanged. (MCP's
`{"error": ...}` convention carries rate-limit rejections too; the message text
distinguishes throttling from denial.)

**Auto-capture is MCP-only.** The middleware intercepts tool calls to
opportunistically persist conversation turns. REST has no equivalent
interception layer — REST callers persist explicitly via `POST /memories`
(`app.add`). This is intentional: REST is the explicit-control surface.

### REST shell (`shells/rest/`)

New, thin FastAPI:

```python
@router.post("/memories")
async def add_memory(body: AddRequest):
    try:
        return await app.add(body.user_id, body.conversation_id, body.messages)
    except RateLimitError as e:        # checked before AccessError — it's a subclass
        raise HTTPException(429, str(e))
    except AccessError as e:
        raise HTTPException(403, str(e))
    except NotFoundError as e:
        raise HTTPException(404, str(e))
```

`RateLimitError` is caught before `AccessError` (it subclasses it) so throttling
maps to **429 Too Many Requests** — the status clients and load balancers use to
trigger backoff — while governance denials stay **403 Forbidden**.

Endpoints mirror the facade: `POST /memories`, `GET /memories/search`,
`GET /memories/recall`, `DELETE /memories`, `POST`/`GET /decisions`,
`GET /health`. Auth reuses the existing `X-API-Key`/JWT verifier from `auth/`
(not reimplemented). OpenAPI docs at `/docs`.

### One process, both shells

The Docker image can run MCP (`/mcp`) and REST (`/`) off the **same**
`AsyncMemory` instance via `TRANSPORT=mcp|rest|both`. One Atlas connection pool,
one set of workers, two protocols — the "memory platform" story in a single
deployable unit.

### Config resolution

- Library: `AsyncMemory.create(MemoryConfig(...))` — explicit object.
- MCP/REST: `AsyncMemory.create(MemoryConfig.from_env())` — env, as today.

---

## 6. Data flow — `add` end to end

```text
caller (import | MCP tool | POST /memories)
   → AsyncMemory.add(user_id, conv_id, messages)
       → _run(): check_access (governance + rate limit)
       → memory_service.store(): embed each msg → write STM docs (24h TTL)
                                → queue LTM candidates (enrichment_status=pending)
       → _run(): audit log "memory:write" success
   ← {"stm_ids": [...], "count": N}

[async, in-process worker] EnrichmentWorker polls pending
   → importance score + summary (LLM) → evolution check (vector similarity)
   → reinforce / merge / create
[async] ConsolidationWorker (24h): STM→LTM promotion, forgetting, compression
```

Identical regardless of entry shell — all three call the same facade method.

---

## 7. Configuration (`MemoryConfig`)

Programmatic Pydantic object, constructible in code, with `.from_env()` for the
deployed shells. Backward-compatible with memory-mcp's env var names.

```python
cfg = MemoryConfig(
    mongo_uri="mongodb+srv://...",
    llm_provider="anthropic",          # bedrock | openai | anthropic
    embedding_provider="openai",       # bedrock | voyage | openai
    embedding_dimension=1536,
    anthropic_base_url=None,           # set → Grove gateway
    openai_base_url=None,              # set → Grove gateway
    workers_in_process=True,           # False → Atlas Triggers / external runtime
)
m = await AsyncMemory.create(cfg)

# Deployed shells:
m = await AsyncMemory.create(MemoryConfig.from_env())
```

New fields beyond memory-mcp's config: `openai_api_key`, `openai_base_url`,
`openai_model`, `openai_embedding_model`, `anthropic_api_key`,
`anthropic_base_url`, `anthropic_model`, `workers_in_process`.

---

## 8. Testing strategy

- **Port service tests** (~26 unit tests) — service internals don't change;
  repoint to the new package path. Covers memory, cache, enrichment,
  consolidation, governance, rate_limiter, audit, database, migrations.
- **New facade tests** (`test_memory_facade.py`) — assert each method runs
  access-check → service → audit; `AccessError` raised on governance denial;
  audit emitted on both success and error paths; `create()`/`close()` start/stop
  workers. Highest-value new coverage.
- **New shell tests** — MCP tools translate `AccessError` → `{"error": ...}`;
  REST routes translate exceptions → correct HTTP status codes, including
  `RateLimitError → 429` distinct from `AccessError → 403` and
  `NotFoundError → 404`.
- **Sync wrapper test** — `Memory.add/recall` work from plain sync context and
  inside a running event loop (notebook scenario).
- **New provider tests** — OpenAI/Anthropic providers honor `base_url` (Grove);
  OpenAI embedding dimension handling; `embedding_dimension` mismatch raises
  `ConfigError` at `create()`; missing-extra for a non-default provider raises
  `ConfigError` with install hint. SDK clients mocked; no live API calls.
- **Integration tier (optional, Atlas-gated)** — unit tests mock Atlas, so
  `$rankFusion`/RRF hybrid retrieval and the vector + FTS indexes can't be
  exercised in-process. A small integration suite, guarded by an env flag and a
  live Atlas connection (skipped in CI without credentials), runs `add` →
  `search`/`recall` end to end to confirm hybrid retrieval actually fuses and
  ranks. Keeps the one MongoDB-specific behavior from being entirely unverified.
- **TDD:** each facade method gets a failing test first, then implementation.

---

## 9. Repo, version & branch strategy

- Work on branch `revamp/agent-memory-v4` in
  `/Users/mdf/git/partners/agent-memory` (cloned from
  `mongodb-partners/ai-memory`).
- **Full rewrite:** memory-mcp source restructured into `agent_memory/` +
  `shells/`. Old ai-memory FastAPI / `memory_nodes` code replaced entirely.
- **Version `4.0.0`** (supersedes memory-mcp 3.x).
- **Repo rename** `ai-memory` → `agent-memory` performed on GitHub at the end
  (auto-redirects the old URL). Package id `agent_memory`, dist name
  `agent-memory`.
- `pyproject.toml`: renamed package; optional-dependency groups `[openai]`,
  `[anthropic]`, `[rest]` (fastapi/uvicorn), `[all]` — SDKs opt-in to keep base
  install light.
- Merge to `main` as the major-version release.

---

## 10. Out of scope (deferred)

- **SP1** — prospective memory: pending-task schema, scheduled-job worker, Change
  Stream listener, Atlas Trigger templates. SP3 only leaves the
  `workers_in_process` seam.
- **SP2** — entity extraction/linking, first-class memory typing, temporal query
  reasoning, richer filter operators.
- Managed/hosted SaaS platform (rejected: weakens the "runs on YOUR Atlas"
  showcase, heavy operational commitment).
- Production data migration from old ai-memory (fresh slate).
