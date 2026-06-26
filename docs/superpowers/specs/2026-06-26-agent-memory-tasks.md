# agent-memory SP3 — Test Plan & Tasks

**Derived from:** `2026-06-26-agent-memory-requirements.md`
**Method:** TDD — each task writes failing tests first, then minimal code.

Test files live under `tests/unit/` (matching existing convention), with the
optional live tier under `tests/integration/`. Conventions: pytest,
`asyncio_mode=auto`, `snake_case` test names, mocked external SDKs/DB.

---

## Task 0 — DONE: substrate restructure (commit 82cd583)

Substrate copied into `agent_memory/`, package renamed, old ai-memory code
removed, 344 ported tests green. `test_server.py`/`test_packaging.py` fail by
design (superseded by shells in Task 5 / packaging in Task 6).

---

## Task 1 — Exceptions + MemoryConfig + from_env  → REQ-E-001,002,010,011,012

- **Files:** CREATE `agent_memory/exceptions.py`; CREATE `agent_memory/config.py`.
- **Tests (`tests/unit/test_exceptions.py`, `tests/unit/test_memory_config.py`):**
  - TC-EXC-001: `RateLimitError` is a subclass of `AccessError`; both subclass `MemoryError`.
  - TC-EXC-002: `NotFoundError`, `ConfigError` subclass `MemoryError`.
  - TC-CFG-001: `MemoryConfig` exposes new fields with documented defaults (`workers_in_process=True`).
  - TC-CFG-002: `MemoryConfig.from_env()` reads `MONGODB_CONNECTION_STRING` and legacy var names.
  - TC-CFG-003: `from_env()` defaults `llm_provider` and `embedding_provider` to `bedrock`.
- **Acceptance:** new tests pass; 344 ported tests still green.

## Task 2 — AsyncMemory facade  → REQ-E-020..031, INV-001..007

- **Files:** CREATE `agent_memory/memory.py` (`AsyncMemory`); extract
  `hybrid_search` into `MemoryService.hybrid_search` and health/wipe into a
  small `AdminService` (or `MemoryService` methods) so the facade stays thin.
- **Tests (`tests/unit/test_memory_facade.py`):**
  - TC-FAC-001..012: each public method delegates to the right service and returns its result (services mocked).
  - TC-FAC-AUDIT-001: success path writes one `success` audit entry.
  - TC-FAC-AUDIT-002: service error writes one `error` audit entry and re-raises.
  - TC-FAC-ACC-001: governance denial → `AccessError` raised (no service call).
  - TC-FAC-ACC-002: rate-limit exceeded → `RateLimitError` raised.
  - TC-FAC-LIFE-001: `create()` with `workers_in_process=True` starts 3 workers; `close()` cancels them + flushes audit + closes db.
  - TC-FAC-LIFE-002: `workers_in_process=False` starts no workers and logs a warning (premortem #2).
  - TC-FAC-LIFE-003: `async with AsyncMemory.create(cfg) as m` calls `close()` on exit.
  - TC-FAC-DIM-001: `create()` raises `ConfigError` when `embedding_dimension` ≠ embedder dimension (premortem #3, boundary #6).
  - TC-FAC-RECALL-001 / TC-FAC-SEARCH-001: `recall`→`MemoryService.recall`; `search`→`MemoryService.hybrid_search` (REQ-E-030).
- **Acceptance:** facade tests pass; ported service tests still green.

## Task 3 — Memory sync wrapper  → REQ-E-040,041,042

- **Files:** extend `agent_memory/memory.py` with `Memory`.
- **Tests (`tests/unit/test_memory_sync.py`):**
  - TC-SYNC-001: `Memory.add/recall` return correct results from a plain sync context (async core mocked).
  - TC-SYNC-002: same calls succeed when invoked from inside a running event loop (notebook scenario) — no "loop already running" (premortem #4, boundary #1).
  - TC-SYNC-003: `close()` stops the background thread.
- **Acceptance:** sync tests pass.

## Task 4 — OpenAI + Anthropic providers + manager arms + dim validation  → REQ-E-050..054, REQ-E-031

- **Files:** CREATE `providers/openai.py`, `providers/anthropic.py`; EDIT
  `providers/manager.py`; add dimension-introspection helper used by Task 2's guard.
- **Tests (`tests/unit/test_openai_provider.py`, `test_anthropic_provider.py`, extend `test_providers.py`):**
  - TC-PROV-001: `OpenAILLMProvider.chat/assess_importance/generate_summary` call the mocked `AsyncOpenAI` client; `base_url` is forwarded (Grove).
  - TC-PROV-002: `OpenAIEmbeddingProvider` returns correct-length vectors for `-3-small` (1536) and `-3-large` (3072).
  - TC-PROV-003: `AnthropicLLMProvider` three methods call the mocked `AsyncAnthropic` client; `base_url` forwarded.
  - TC-PROV-004: `ProviderManager` builds each `{bedrock,openai,anthropic}` LLM and `{bedrock,voyage,openai}` embedder.
  - TC-PROV-005: selecting `openai`/`anthropic` without the SDK → `ConfigError` with `pip install agent-memory[...]` hint (premortem #5, boundary #5).
  - TC-PROV-006: dimension helper returns known per-model dims and falls back to probing.
- **Acceptance:** provider tests pass with SDKs mocked; no live API calls.

## Task 5 — MCP shell over facade  → REQ-E-060,061,062,063

- **Files:** CREATE `agent_memory/shells/mcp/` (slim tools + lifespan); retire
  module-level `server.py` orchestration. Update/replace `test_server.py`.
- **Tests (`tests/unit/test_mcp_shell.py`):**
  - TC-MCP-001: each tool delegates to the matching `app.<method>` and returns its dict.
  - TC-MCP-002: `AccessError`/`RateLimitError` → `{"error": ...}` (REQ-E-061).
  - TC-MCP-003: auto-capture calls `app.add` (REQ-E-063).
  - TC-MCP-LIFE-001: lifespan creates and closes the `AsyncMemory` instance.
- **Acceptance:** MCP shell tests pass; old `test_server.py` removed/replaced.

## Task 6 — REST shell + dual-transport + packaging  → REQ-E-070,071,072,080,081,082, REQ-NF-COV

- **Files:** CREATE `agent_memory/shells/rest/`; CREATE dual-transport entry in
  `agent_memory/__main__.py`; UPDATE `agent_memory/__init__.py` exports; UPDATE
  `pyproject.toml` (done in Task 0) and `test_packaging.py` for the new layout.
- **Tests (`tests/unit/test_rest_shell.py`, `tests/unit/test_packaging.py`, `tests/unit/test_exports.py`):**
  - TC-REST-001..004: routes call the facade and return its payload (TestClient, facade mocked).
  - TC-REST-005: `RateLimitError`→429, `AccessError`→403, `NotFoundError`→404, ordered correctly (premortem #1, REQ-E-071).
  - TC-REST-AUTH-001: protected route uses the existing `auth/` verifier.
  - TC-TRANS-001: `TRANSPORT=both` builds one `AsyncMemory` shared by both shells; `mcp`/`rest` build only one shell (REQ-E-080).
  - TC-PKG-001: `pyproject` declares `agent-memory` 4.0.0 + `[openai]/[anthropic]/[rest]/[all]` (REQ-E-081).
  - TC-EXP-001: `from agent_memory import Memory, AsyncMemory, MemoryConfig, AccessError, RateLimitError, NotFoundError, ConfigError` (REQ-E-082).
- **Acceptance:** all new tests pass; full suite green.

## Task 7 — Verification

- Full `pytest` suite green; coverage floors met (REQ-NF-COV); phase-end demos
  (import facade, REST curl over uvicorn, MCP tool list); optional Atlas-gated
  `TC-INTEG-LIVE-001` (boundary #7) run if creds present, else skipped; change
  summary.

---

## Traceability Matrix

| Req ID | Test Case IDs | Status |
|---|---|---|
| REQ-E-001 | TC-EXC-001 | Passing |
| REQ-E-002 | TC-EXC-001 | Passing |
| REQ-E-010 | TC-CFG-001 | Passing |
| REQ-E-011 | TC-CFG-002 | Passing |
| REQ-E-012 | TC-CFG-003 | Passing |
| REQ-E-020 | TC-FAC-LIFE-001 | Passing |
| REQ-E-021 | TC-FAC-LIFE-001 | Passing |
| REQ-E-022 | TC-FAC-LIFE-002 | Passing |
| REQ-E-023 | TC-FAC-LIFE-001 | Passing |
| REQ-E-024 | TC-FAC-LIFE-003 | Passing |
| REQ-E-025 | TC-FAC-AUDIT-001 | Passing |
| REQ-E-026 | TC-FAC-ACC-001 | Passing |
| REQ-E-027 | TC-FAC-ACC-002 | Passing |
| REQ-E-028 | TC-FAC-AUDIT-001, TC-FAC-AUDIT-002 | Passing |
| REQ-E-029 | TC-FAC-001..012 | Passing |
| REQ-E-030 | TC-FAC-RECALL-001, TC-FAC-SEARCH-001 | Passing |
| REQ-E-031 | TC-FAC-DIM-001, TC-PROV-006 | Passing |
| REQ-E-040 | TC-SYNC-001 | Passing |
| REQ-E-041 | TC-SYNC-002 | Passing |
| REQ-E-042 | TC-SYNC-003 | Passing |
| REQ-E-050 | TC-PROV-001 | Passing |
| REQ-E-051 | TC-PROV-002 | Passing |
| REQ-E-052 | TC-PROV-003 | Passing |
| REQ-E-053 | TC-PROV-004 | Passing |
| REQ-E-054 | TC-PROV-005 | Passing |
| REQ-E-060 | TC-MCP-001 | Passing |
| REQ-E-061 | TC-MCP-002 | Passing |
| REQ-E-062 | TC-MCP-LIFE-001 | Passing |
| REQ-E-063 | TC-MCP-003 | Passing |
| REQ-E-070 | TC-REST-001..004 | Passing |
| REQ-E-071 | TC-REST-005 | Passing |
| REQ-E-072 | TC-REST-AUTH-001 | Passing |
| REQ-E-080 | TC-TRANS-001 | Passing |
| REQ-E-081 | TC-PKG-001 | Passing |
| REQ-E-082 | TC-EXP-001 | Passing |
| REQ-NF-COV | coverage report (Task 7) | Passing |
| INV-001..008 | 344 ported tests (baseline green) | Passing (baseline) |
