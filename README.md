# agent-memory

**Four kinds of agent memory in one MongoDB Atlas cluster.** No agent framework
required — and none imported.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Install from the repository — this is not on PyPI:

```bash
uv add git+https://github.com/mongodb-partners/ai-memory.git
# or
pip install git+https://github.com/mongodb-partners/ai-memory.git
```

---

## The problem

The model forgets. A bigger context window does not fix that — it just makes
forgetting more expensive. So teams bolt on a vector store for semantic recall, a
key-value store for session state, a separate log for what the agent actually
did, and a cache in front of the model. Four systems, four consistency stories,
four things to operate.

They are all queries over the same data. Atlas does all four.

## The four memories

Most memory libraries give you one. An agent needs four, because they answer
different questions.

| Tier | Answers | Lifetime | How it is retrieved |
|---|---|---|---|
| **Short-term** | "What are we doing right now?" | TTL'd (24h default) | Recent-first, scoped to the conversation |
| **Long-term semantic** | "What do I know about this user?" | Importance-scored; reinforced, merged, decayed | Hybrid vector + full-text, importance-ranked |
| **Episodic** | "What did we actually *do* last Tuesday?" | TTL'd (30d default), tunable in place | Hybrid over turns: messages, tools, files, todos |
| **Semantic cache** | "Have I answered this already?" | TTL'd | Vector similarity above a threshold |

Episodic is the one most libraries skip. Short-term state holds what the agent is
doing and long-term memory holds what it knows, but neither records what it
*did* — which tool it called, which file it wrote, how many steps it took, and
under which trace id. That is the tier you need when something goes wrong, and
it is the tier an agent needs to reason about its own past work.

## 60-second quickstart

```python
import asyncio
from agent_memory import AsyncMemory, MemoryConfig

async def main():
    memory = await AsyncMemory.create(MemoryConfig(
        mongodb_connection_string="mongodb+srv://...",
        embedding_provider="voyage",          # or "bedrock" (default), "openai"
        voyage_api_key="...",                 # see "Two Voyage endpoints" below
    ))

    # Write a conversation. Short-term is immediate; promotion to long-term is
    # importance-scored and happens in the background.
    await memory.add("user-1", "conv-1", [
        {"message_type": "human", "content": "I'm vegetarian, and I hate cilantro."},
        {"message_type": "ai", "content": "Noted — no meat, no cilantro."},
    ])

    # Record what the agent did. Non-blocking: this never awaits Atlas.
    await memory.log_activity("user-1", "thread-1", [
        {"type": "human", "content": "Book me somewhere for Friday"},
        {"type": "ai", "content": "Booked Nopalito, 7pm", "tool_calls": [
            {"name": "search_restaurants", "args": {"cuisine": "vegetarian"}},
        ]},
    ], correlation_id="trace-abc")

    # Recall knowledge — hybrid vector + full-text, importance-ranked.
    print(await memory.recall("user-1", "what should I cook?"))

    # Recall actions — the same hybrid search over the turn log.
    print(await memory.recall_activity("user-1", "friday dinner booking"))

    await memory.close()   # drains queued turns before closing the connection

asyncio.run(main())
```

Not in an async context? `Memory` is the blocking twin, with every method
mirrored. It runs the async core on its own event loop on a daemon thread, so it
works from plain scripts *and* from inside a notebook cell that already has a
running loop.

```python
from agent_memory import Memory, MemoryConfig

with Memory(MemoryConfig(mongodb_connection_string="mongodb+srv://...")) as memory:
    memory.add("user-1", "conv-1", [{"message_type": "human", "content": "hi"}])
    print(memory.recall("user-1", "greeting"))
```

## Library API

Convention throughout: `user_id` is positional, everything else is keyword-only.
Every method is scoped to a user — there is no unscoped read.

### Semantic memory

```python
await memory.add(user_id, conversation_id, messages)          # → {"stm_ids": [...], "count": n}
await memory.recall(user_id, query, *, tier=None, memory_type=None, tags=None, limit=10)
await memory.search(user_id, query, *, tier=None, limit=10, memory_type=None, tags=None)
await memory.delete(user_id, *, memory_id=None, tags=None, time_range=None,
                    confirm=False, dry_run=False)
```

`recall` is curated: hybrid search, then deduplication and importance-weighted
re-ranking. `search` is the raw `$rankFusion` result with scores intact. Use
`recall` to build a prompt, `search` to see what the index actually thinks.

### Episodic memory

```python
await memory.log_activity(user_id, thread_id, messages, *, todos=None,
                          agent_name=None, correlation_id=None,
                          conversation_id=None, ts=None)
await memory.recall_activity(user_id, query, *, thread_id=None, agent_name=None,
                             since=None, limit=5)
await memory.get_thread(user_id, thread_id, *, limit=None, ascending=True)
await memory.get_activity_by_correlation(user_id, correlation_id)
await memory.flush_activity(timeout=5.0)          # → bool; bounded, never raises
await memory.set_activity_retention(user_id, *, ttl_seconds=7200)   # None = forever
memory.activity_stats()                            # synchronous; safe in a probe
```

`log_activity` builds the document and enqueues it. It never awaits Atlas or the
embedder, so logging cannot slow the agent down. A single consumer task batches
inserts, which keeps per-thread step numbers monotonic.

Three behaviours worth knowing, because they are the ones that matter when
something is already going wrong:

- When the queue is full, the **oldest** pending turn is dropped and counted. The
  newest turn always survives — a stale turn is worth less than a fresh one.
- If the durable step counter fails, the document is inserted with a null step
  rather than dropped. A logged turn beats a lost one.
- The embedding is generated *before* `search_text` is assigned, so an embedding
  failure leaves neither field — never a searchable document with no vector.

`correlation_id` accepts a W3C `traceparent`, so this joins to your existing
tracing stack rather than introducing a competing id.

### Semantic cache and sticky decisions

```python
await memory.check_cache(user_id, query, *, similarity_threshold=None)
await memory.store_cache(user_id, query, response)
await memory.invalidate_cache(user_id, *, pattern=None, invalidate_all=False)

await memory.remember_decision(user_id, key, value, *, ttl_days=None)
await memory.recall_decision(user_id, key)
```

A sticky decision is a durable key/value the agent should not re-litigate — a
chosen shipping address, a confirmed plan step, a locked-in preference.

### Health and teardown

```python
await memory.health(user_id)
await memory.wipe_user_data(user_id, confirm=True)
await memory.close()
```

`close()` drains the episodic queue **first**, while its consumer task is alive
and the connection is still open, bounded by
`episodic_shutdown_timeout_seconds`. Cancelling workers first would silently
discard turns that never reached Atlas.

## Running it as a server

Both shells wrap the same facade and enforce the same access-control path. Pick a
transport with `TRANSPORT`:

```bash
export MONGODB_CONNECTION_STRING="mongodb+srv://..."
TRANSPORT=both agent-memory        # mcp | rest | both — 'both' shares one instance
```

### MCP tools

`store_memory` · `recall_memory` · `hybrid_search` · `delete_memory` ·
`check_cache` · `store_cache` · `cache_invalidate` · `store_decision` ·
`recall_decision` · `log_activity` · `search_activity` · `get_thread` ·
`get_correlation` · `set_activity_retention` · `memory_health` · `wipe_user_data`

The MCP shell can also auto-capture significant tool interactions, so the store
fills even when the agent never calls `store_memory`. Auto-capture is MCP-only by
design; REST is the explicit-control surface.

### REST routes

| Method | Path |
|---|---|
| `POST` | `/memories` |
| `GET` | `/memories/recall`, `/memories/search` |
| `DELETE` | `/memories` |
| `POST` | `/activity` |
| `GET` | `/activity/search`, `/activity/thread/{thread_id}`, `/activity/correlation/{correlation_id}` |
| `PUT` | `/activity/retention` |
| `POST` / `GET` | `/decisions` |
| `GET` | `/health` |

`/health` is open; every other route requires a Bearer token when
`AUTH_ENABLED=true`. `RateLimitError` maps to 429, `AccessError` to 403,
`NotFoundError` to 404.

`/health` also returns the episodic writer's counters — queue depth, throughput,
write and embed failures. A 200 with a saturated queue and rising failures is not
health, so the probe reports both.

## Providers

Embeddings and chat are pluggable; nothing above changes when you switch.

| Provider | Embeddings | LLM |
|---|---|---|
| Amazon Bedrock (default) | ✅ | ✅ |
| Voyage AI (direct or via the Atlas embeddings gateway) | ✅ | — |
| OpenAI (any `base_url`) | ✅ | ✅ |
| Anthropic | — | ✅ |

`create()` validates on startup that the configured embedding dimension matches
what the provider actually returns, and fails fast if not. A mismatch otherwise
surfaces much later as an empty result set with no error.

## Indexes and DDL

`create()` ensures every index the library needs, including the Atlas Search
definitions, so there is no manual DDL step. Two details are load-bearing if you
manage indexes yourself:

1. **Any field used in a `$vectorSearch` pre-filter must be declared as
   `{"type": "filter"}`** in the index definition. An undeclared filter field
   does not raise — the branch just returns nothing, which looks exactly like
   "no matches."
2. **Fields backing an exact `equals` filter in Atlas Search must use the
   `token` type, not `string`.** A `string` field is analyzed, so exact
   equality quietly stops matching.

Retention is tunable at runtime. `set_activity_retention` issues a `collMod` on
the existing TTL index rather than dropping and rebuilding it, and falls back to
`create_index` if `collMod` is unavailable.

## Configuration

Every field is settable in code via `MemoryConfig(...)` or from the environment
via `MemoryConfig.from_env()` (case-insensitive names). The frequently-used ones:

| Setting | Default | Notes |
|---|---|---|
| `mongodb_connection_string` | — | The only required field |
| `mongodb_database_name` | `agent_memory` | |
| `embedding_provider` / `llm_provider` | `bedrock` | `voyage`, `openai`, `anthropic` |
| `embedding_dimension` | `1536` | Auto-aligned to the model for Voyage |
| `stm_ttl_hours` | `24` | Short-term retention |
| `episodic_enabled` | `True` | `False` accepts and discards, so callers need no conditionals |
| `episodic_queue_size` | `1000` | Bounded; full → drop oldest |
| `episodic_batch_size` | `20` | Turns per `insert_many` |
| `episodic_flush_interval_seconds` | `1.0` | Max wait before writing a partial batch |
| `episodic_embed_final_steps_only` | `True` | A mid-turn step has no answer worth embedding |
| `workers_in_process` | `True` | `False` → an external runtime owns background work |
| `await_search_indexes` | `False` | Set `True` in short-lived scripts, or the process can exit before indexes are queryable |

### Two Voyage endpoints, and the key decides which

Voyage embeddings are reachable two ways, and a key that works with one is
rejected by the other with a `403`:

| Key from | `voyage_base_url` | Models |
|---|---|---|
| Voyage AI | `https://api.voyageai.com/v1/embeddings` (default) | the full Voyage catalogue |
| MongoDB Atlas | `https://ai.mongodb.com/v1/embeddings` | a subset — `voyage-4*`, `voyage-3*`, `voyage-code-*`, `voyage-law-2`, `voyage-finance-2` |

An Atlas key (`al-…`) against the default URL fails with *"Voyage AI API keys
work with Voyage AI endpoints, and MongoDB Atlas API keys work with MongoDB
endpoints."* Point `voyage_base_url` at the gateway and set a model it serves:

```bash
VOYAGE_BASE_URL=https://ai.mongodb.com/v1/embeddings
VOYAGE_MODEL=voyage-4
```

Every gateway model is **1024** dimensions, against the `1536` default. That is
handled — `ProviderManager` aligns `embedding_dimension` to the model, unless you
pinned a non-default value — but it is baked into the vector indexes at creation.
Switching a provider after documents exist means re-creating
`memories_vector_index`, `episodes_vector_index`, and `cache_vector_index` at the
new dimension. A dimension mismatch does not raise; recall just returns nothing.

### One caveat about `workers_in_process=False`

It disables the background workers, which means enrichment, consolidation, audit
flushing, and the episodic writer all stop. Without a consumer, `log_activity`
fills its bounded queue and then starts discarding the oldest turns. Set
`episodic_enabled=False` if that is what you intend, so the behaviour is
explicit rather than inferred from a full queue.

## Governance and audit

Access is profile-based: `admin`, `power_user`, `end_user`, each with allowed
operations and per-day quotas. Every operation goes through one path —
access check, then the service call, then an audit record — so there is no
surface that skips governance.

One exception, and it is deliberate: `log_activity` does not write a
per-call audit record. A turn log is high-volume by nature, and one audit write
per turn would mean logging the agent costs more writes than the agent. It still
enforces governance and rate limits on every call; the worker emits one audit
entry per flushed batch, grouped by `user_id`, since a batch can span users and
misattributing turns would be worse than no audit trail.

Profile seeding is additive. When a release adds an operation, existing profiles
gain it via `$addToSet` — custom quotas and any operations an operator added by
hand are preserved, and nothing is ever removed.

## Development

```bash
uv sync --all-extras
uv run pytest -q          # unit suite, fully mocked — no Atlas needed
uv run ruff check agent_memory/
```

The integration tier gates on server reachability rather than an env flag: start
the server against a real cluster, then run `uv run pytest tests/integration -q`.

## License

Apache 2.0. See [LICENSE](LICENSE).
