# agent-memory

**Four kinds of agent memory in one MongoDB Atlas cluster.** No agent framework
required, and none imported.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Install from the repository. This is not on PyPI:

```bash
uv add git+https://github.com/mongodb-partners/agent-memory.git
# or
pip install git+https://github.com/mongodb-partners/agent-memory.git
```

---

## The problem

The model forgets. A bigger context window does not fix that. It just makes
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
*did*: which tool it called, which file it wrote, how many steps it took, and
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
        {"message_type": "ai", "content": "Noted: no meat, no cilantro."},
    ])

    # Record what the agent did. Non-blocking: this never awaits Atlas.
    await memory.log_activity("user-1", "thread-1", [
        {"type": "human", "content": "Book me somewhere for Friday"},
        {"type": "ai", "content": "Booked Nopalito, 7pm", "tool_calls": [
            {"name": "search_restaurants", "args": {"cuisine": "vegetarian"}},
        ]},
    ], correlation_id="trace-abc")

    # Recall knowledge: hybrid vector + full-text, importance-ranked.
    print(await memory.recall("user-1", "what should I cook?"))

    # Recall actions: the same hybrid search over the turn log.
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

## One-command setup

Two scripts take a fresh checkout to a running demo: the memory server plus the
sample UI, with a seeded user. Both need a filled-in `.env`; run either one
without it and it writes the template and tells you what to fill in.

```bash
scripts/quick_setup.sh    # local: uv + npm, runs in the foreground
scripts/docker_setup.sh   # Docker: compose, exits and leaves the stack running
```

Both end up at http://localhost:5173 with the demo backend on 8100. The Docker
path also serves the MCP/REST shells on 8000; the local path adds them with
`--with-server`, since the UI embeds the library in-process and does not call
them. `--no-seed` skips seeding and `--user <id>` retargets it. Seeding wipes the
user it targets, and the UI still opens with `memory-demo` in the header, so you
need to type the id in to see that user's panel.

The local script runs `uv sync --extra demo`, which uninstalls dev tooling
(pytest, ruff, scikit-learn). Correct for a demo environment, but if you run it
on your working checkout, `uv sync --extra all` afterwards to restore them.

Every port is published to `127.0.0.1` only. See
[Deploy the server](docs/how-to/deployment.md) before exposing any of them.

## Library API

Convention throughout: `user_id` is positional, everything else is keyword-only.
Every method is scoped to a user, and there is no unscoped read.

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

```text
await memory.log_activity(user_id, thread_id, messages, *, todos=None,
                          agent_name=None, correlation_id=None,
                          conversation_id=None, ts=None)
await memory.recall_activity(user_id, query, *, thread_id=None, agent_name=None,
                             since=None, limit=5)
await memory.get_thread(user_id, thread_id, *, limit=None, ascending=True)
await memory.get_activity_by_correlation(user_id, correlation_id)
await memory.flush_activity(timeout=5.0)          # → bool; bounded, never raises
await memory.set_activity_retention(user_id, *, ttl_seconds=7200)   # None = forever
                                                  # → read the returned `status`
memory.activity_stats()                            # synchronous; safe in a probe
```

`log_activity` builds the document and enqueues it. It never awaits Atlas or the
embedder, so logging cannot slow the agent down. A single consumer task batches
inserts, which keeps per-thread step numbers monotonic.

Three behaviours worth knowing, because they are the ones that matter when
something is already going wrong:

- When the queue is full, the **oldest** pending turn is dropped and counted. The
  newest turn always survives, since a stale turn is worth less than a fresh one.
- If the durable step counter fails, the document is inserted with a null step
  rather than dropped. A logged turn beats a lost one.
- The embeddings are generated *before* `search_text` is assigned, so an embedding
  failure leaves neither field, never a searchable document with no vector. One
  provider call covers the whole batch, so a provider failure degrades every turn
  in that batch to text-only rather than just one. `embed_failures` counts
  documents rather than calls, so the number means the same thing either way.

`correlation_id` accepts a W3C `traceparent`, so this joins to your existing
tracing stack rather than introducing a competing id.

### Semantic cache and sticky decisions

```text
await memory.check_cache(user_id, query, *, similarity_threshold=None)
await memory.store_cache(user_id, query, response)
await memory.invalidate_cache(user_id, *, pattern=None, invalidate_all=False)

await memory.remember_decision(user_id, key, value, *, ttl_days=None)
await memory.recall_decision(user_id, key)
```

A sticky decision is a durable key/value the agent should not re-litigate: a
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

`close()` is idempotent on both facades, so calling it explicitly inside a `with`
block is safe.

A **failed** `create()` releases everything it acquired before raising (the
connection pool, the worker tasks, the episodic queue), so you do not need (and
cannot write) a cleanup path for it: `create()` did not return, so there is no
object to close. This matters because the pool is shared per process and
reference-counted: a leaked claim would not fail loudly, it would make a *later*
`close()` from a healthy facade stop closing the client, and make a retry against
a corrected config raise "already connected to a different MongoDB target". Both
symptoms point away from the startup that actually failed. Retrying `create()`
after a failure is therefore safe, including from the synchronous `Memory`
wrapper, whose background loop thread is torn down on the same path.

## Running it as a server

Both shells wrap the same facade and enforce the same access-control path. Pick a
transport with `TRANSPORT`:

```bash
export MONGODB_CONNECTION_STRING="mongodb+srv://..."
TRANSPORT=both agent-memory        # mcp | rest | both; 'both' shares one instance
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
`NotFoundError` to 404, `EmbeddingError` to 502.

502 is the embedding provider answering with something that does not describe the
request: fewer vectors than inputs, or a vector of the wrong width. Nothing was
written, and retrying is the right response: the write is refused precisely so the
caller still holds its own data. See [Embedding replies are
validated](#embedding-replies-are-validated).

`/health` also returns the episodic writer's counters: queue depth, throughput,
write and embed failures. A 200 with a saturated queue and rising failures is not
health, so the probe reports both.

## Providers

Embeddings and chat are pluggable; nothing above changes when you switch.

| Provider | Embeddings | LLM |
|---|---|---|
| Amazon Bedrock (default) | ✅ | ✅ |
| Voyage AI (direct or via the Atlas embeddings gateway) | ✅ | ❌ |
| OpenAI (any `base_url`) | ✅ | ✅ |
| Anthropic | ❌ | ✅ |

`create()` validates on startup that the configured embedding dimension matches
what the provider actually returns, and fails fast if not. A mismatch otherwise
surfaces much later as an empty result set with no error.

## Indexes and DDL

`create()` ensures every index the library needs, including the Atlas Search
definitions, so there is no manual DDL step. Two details are load-bearing if you
manage indexes yourself:

1. **Any field used in a `$vectorSearch` pre-filter must be declared as
   `{"type": "filter"}`** in the index definition. An undeclared filter field
   does not raise. The branch just returns nothing, which looks exactly like
   "no matches."
2. **Fields backing an exact `equals` filter in Atlas Search must use the
   `token` type, not `string`.** A `string` field is analyzed, so exact
   equality quietly stops matching.

Retention is configured, and reconciled to the configuration on every startup.
See [Retention](#retention) below. `set_activity_retention` is the runtime
override for episodic retention specifically: it issues a `collMod` on the
existing TTL index rather than dropping and rebuilding it (falling back to
`create_index` where `collMod` is unavailable), which makes it cheap but not
durable: the next startup reconciles the index back to
`episodic_retention_days`.

It is also the one method whose failure is a **return value** rather than an
exception, so read the `status`: `updated`/`created`/`removed` mean the index
changed, `error` means it did not. The audit entry now agrees. It used to record
every call as a success, including the ones that changed nothing, which mattered
most in the destructive direction: shortening retention collection-wide and having
it silently fail is indistinguishable, from the response alone, from having it
work.

## Configuration

Every field is settable in code via `MemoryConfig(...)` or from the environment
via `MemoryConfig.from_env()` (case-insensitive names). The frequently-used ones:

| Setting | Default | Notes |
|---|---|---|
| `mongodb_connection_string` | *no default* | The only required field |
| `mongodb_database_name` | `agent_memory` | |
| `embedding_provider` / `llm_provider` | `bedrock` | `voyage`, `openai`, `anthropic` |
| `embedding_dimension` | `1536` | Auto-aligned to the model for Voyage while left at the default; a mismatch raises at startup |
| `allow_embedding_dimension_change` | `False` | Startup refuses when the dimension changed and existing vectors would be orphaned. See below |
| `stm_ttl_hours` | `24` | Short-term retention |
| `ltm_candidate_min_chars` | `31` | Shortest human message promoted to a long-term candidate. Below it, a message is stored as STM but never enriched or recalled. See [What becomes long-term](#what-becomes-long-term) |
| `soft_delete_purge_days` | `30` | How long a soft-deleted memory survives before purge |
| `episodic_retention_days` | `30` | Turn-log retention. See [Retention](#retention) |
| `cache_ttl_seconds` | `3600` | Semantic-cache entry lifetime |
| `audit_retention_days` | `365` | Audit-log retention |
| `audit_fallback_path` | `audit_fallback.jsonl` | Where audit entries go when MongoDB refuses them; resolved to an absolute path once, at startup. Empty string discards them instead |
| `audit_fallback_max_bytes` | `52428800` | Ceiling before the fallback rotates to one `.1` sibling, so twice this on disk. `0` disables rotation |
| `episodic_enabled` | `True` | `False` accepts and discards, so callers need no conditionals |
| `episodic_queue_size` | `1000` | Bounded; full → drop oldest |
| `episodic_batch_size` | `20` | Turns per `insert_many` |
| `episodic_flush_interval_seconds` | `1.0` | Max wait before writing a partial batch |
| `episodic_embed_final_steps_only` | `True` | A mid-turn step has no answer worth embedding |
| `workers_in_process` | `True` | `False` → an external runtime owns background work |
| `await_search_indexes` | `False` | Set `True` in short-lived scripts, or the process can exit before indexes are queryable |
| `importance_scorer` | `llm` | `local` scores in-process instead of calling the LLM |
| `importance_model_path` | `None` | Explicit coefficient artifact; unset auto-selects a bundled one |

### Retention

Every retention duration is a config value that reaches the index enforcing it.
Startup reconciles each TTL index to the configuration, so changing a value and
restarting rebuilds the index at the new duration:

| Setting | Default | Expires |
|---|---|---|
| `soft_delete_purge_days` | `30` | Soft-deleted memories, permanently |
| `episodic_retention_days` | `30` | Logged turns |
| `cache_ttl_seconds` | `3600` | Semantic-cache entries |
| `audit_retention_days` | `365` | Audit records |
| `rate_limit_retention_seconds` | `86400` | Spent rate-limit window counters |

**Shortening a retention deletes data.** The rebuilt index applies to documents
already stored, so anything past the new cutoff is expired by Atlas's TTL monitor
within a minute or two of the restart, not at the next write, and with no
confirmation step. Check the value before restarting a deployment whose history
matters. Lengthening one is safe; it only stops future expiry, and nothing
already deleted comes back.

`rate_limit_retention_seconds` is raised to `rate_limit_window_seconds` when that
is longer. A counter *is* the enforcement state, so expiring it inside its own
window would let a caller who had exhausted the limit start a fresh count.

Long-term retention works differently: it is per-tier, applied as a per-document
`expires_at` at write time rather than as one collection-wide duration, via
`ltm_retention_critical_days` / `_reference_days` / `_standard_days` /
`_temporary_days`. Changing those affects memories written afterwards. The
`expires_at` already stamped on existing documents is not rewritten.

### What becomes long-term

`add()` stores every message as STM. It *additionally* queues a long-term
candidate for each **human** message of at least `ltm_candidate_min_chars`
characters (default `31`). Only candidates are enriched, promoted, and returned by
`recall`, so this one number decides what the agent is able to remember later.

It is a length, not a measure of meaning, and that is why it is a setting rather
than a cleverer inline rule. The default drops the acknowledgements that dominate
a real transcript ("ok", "thanks", "yes do that"), and it also drops *"I'm
allergic to penicillin"*, which is 26 characters. Adjust it to your traffic:

- **Lower it** where users write telegraphically, or where short facts matter more
  than enrichment cost.
- **Raise it** where every turn is a paragraph and you would rather not pay an
  enrichment per turn.
- **`0`** keeps every human message, which is the honest way to say "let
  importance scoring decide", and means one LLM enrichment per turn.

Assistant messages are never candidates at any threshold. Promoting the agent's
own output would let its summaries become the facts it later recalls.

If you retrain the local importance model, `scripts/train_importance.py` reads
this same value to filter its corpus, so a tuned threshold and its model stay in
step.

### Changing the embedding model later

Switching embedding provider or model usually changes the vector width, and a
vector index cannot have its `numDimensions` edited, so it has to be dropped and
rebuilt. The documents are untouched by that, which sounds safe and is the
problem: every vector already stored keeps the old width, and the rebuilt index
returns none of them from `$vectorSearch`.

Nothing about that failure is visible. No exception, no change in document count,
`find` still shows every memory. Recall goes empty for the whole history while
working perfectly for anything written afterwards, so it reads as "the user has no
memories about that". Undoing it means re-embedding every document with the
*previous* provider, the config you just replaced.

So startup refuses, naming the affected indexes and the counts at risk. An empty
collection is not affected and never triggers this. To proceed, pick one:

* Restore the previous `embedding_provider` / model.
* Re-embed every document at the new width, then start.
* Drop the affected collections, if the history is expendable.
* `ALLOW_EMBEDDING_DIMENSION_CHANGE=true`, accepting that the old vectors become
  unsearchable.

### Embedding replies are validated

The dimension guard above covers a *changed* configuration. A provider can also
answer a single request wrongly, and the two ways it does are both silent.

**A short batch.** `store` embeds a batch of messages in one call and pairs the
results with the inputs positionally. A provider that returns nine vectors for ten
messages used to produce nine memories: `zip` stops at the shorter sequence, the
insert succeeds, and the call returns nine ids to a caller that handed over ten.
Nothing raised, nothing logged, and no record existed of which message was lost.

**A wrong width.** Atlas accepts a 1024-wide vector into a 1536-wide index. The
document is stored, the count goes up, `find` returns it, and `$vectorSearch`
never does. The memory exists and is not recallable.

Both are refused now, before anything is written, with `EmbeddingError` (a
`MemoryError`, so an existing `except MemoryError` catches it; 502 over REST). The
width compared against is the *resolved* one (Voyage's native 1024, not the
1536 a config may still declare), so a correct Voyage deployment is unaffected.

There is no repair, only a refusal. Padding a short vector, truncating a long one,
or re-embedding the missing tail all invent data and store it as though the
provider had produced it. Failing the call leaves the messages with the caller,
which is the only state from which a retry is possible.

The background paths degrade rather than fail, because they have no caller to
return to: a wrong-width episodic vector logs the turn as text-only and counts an
`embed_failures`, and a wrong-width merge re-embed leaves the merge `merge_pending`
for the next pass rather than committing content and vector that disagree.

### Importance scoring without the LLM

Importance decides three things: whether a memory is eventually forgotten, whether
it is promoted to long-term memory, and how it ranks in recall. By default it costs
one LLM round trip per long-term memory, on the enrichment worker's path.

`IMPORTANCE_SCORER=local` replaces that call with a logistic model evaluated
in-process: microseconds, no network, no tokens. It works because the embedding
already exists by the time enrichment runs, so scoring is a dot product over a
vector the worker was already holding. No encoder, no new dependency; the scorer is
pure Python.

With `importance_model_path` unset, one artifact loads, whatever the embedder:
`lexical`, and it is trained. It scores seven bounded text features rather than the
embedding, so it is provider-independent and applies to every deployment.

Earlier versions shipped `titan-1536` and `voyage-3-1024`
alongside it; both were zero-coefficient placeholders that scored every memory
0.5 (above the forgetting threshold, below the promotion threshold), so
`IMPORTANCE_SCORER=local` on those embedders did not approximate the LLM, it
quietly turned importance-based promotion off while looking healthy. They have been
deleted rather than trained, and selection now resolves to `lexical` **by design
rather than by fallthrough**.

Skipping the embedding head is a measurement, not a shortcut. Its held-out Spearman
tops out near 0.45 and its *in-sample* ceiling (fitted and scored on the same
1,234 rows) is only 0.70, so more labels buy at most part of that gap on a
mediocre maximum. `assess_importance` returns a 1-10 integer, which is 9 distinct
label values for a 1024-coefficient fit to aim at. Regularization does not rescue
it either (alpha 1 → 100,000 moves Spearman 0.45 → 0.34, shrinking toward the
constant). A trained lexical model beats an untrained embedding head at any
embedder, which is why the table is uniform.

If you want an embedding head for your own embedder, train one with
`--space embedding`, commit the JSON under `agent_memory/data/importance/`, and add
its `(provider, model, dimension)` triple to `_BUNDLED_ARTIFACTS` in
`agent_memory/providers/manager.py`. Selection is on the whole triple, not the
provider, because coefficients are positional: `voyage-3` emits 1024 dimensions and
`voyage-3-lite` emits 512, and loading 1024 coefficients against a 512-vector is
the mismatch selection exists to refuse. Either way, the startup log line names the
artifact that actually loaded.

The lexical artifact is trained on LLM-distilled labels only. The two long-term
memory benchmarks (locomo, longmemeval) are collected by the trainer but carry zero
weight in the fit by default, and that default is a measurement: their labels mark a
turn positive when a later question cited it as evidence, and the questions are
time-anchored, so `yesterday` carries a 21.7× lift toward positives. Any nonzero
weight trains the `temporal` feature strongly positive and produces a model that
promotes *"let's pick this up after lunch, I'm busy today and tomorrow"* (0.775)
over *"our policy is that customer data never leaves the EU region"* (0.435). The
shipped artifact scores those 0.377 and 0.719 respectively. Coefficient signs are
readable in the file and match the design: `preference` and `identity` positive,
`temporal` and `interrogative` negative.

Two limits worth knowing before you tune it. The seven lexical features cannot
separate 31.5% of the label variance (measured, as rows sharing a feature vector
while carrying different labels), and the learning curve is flat past ~460 rows,
so a larger training set will not improve this artifact. And `assess_importance`
returns a 1-10 integer, so the distilled labels take only 9 distinct values; the
local scorer cannot be more granular than its teacher.

**Check calibration before switching a production deployment.** Consolidation
compares importance against *absolute* thresholds: below `forgetting_score_threshold`
(0.1) a memory is deleted, at or above `promotion_importance_threshold` (0.6) it is
promoted. So a model that ranks memories perfectly but sits systematically low will
forget more and promote less than the LLM did, and the symptom is degraded recall
weeks later, not an error. Read `forget_agreement` and `promote_agreement` from the
artifact's `training.metrics` and compare against your own thresholds first.

Two failure modes are deliberately loud rather than quiet. An embedding whose
dimension does not match the artifact raises instead of scoring the overlapping
prefix; the affected memories land in `enrichment_status: "failed"`, where they can
be counted and re-run. And an `importance_model_path` pointing at a missing file
refuses to start rather than falling back to a bundled artifact the operator did
not ask for.

Retraining is offline and optional: `pip install 'agent-memory[training]'` then
`python scripts/train_importance.py --help`. Nothing under `agent_memory/` imports
scikit-learn or numpy, and a packaging test enforces it.

The trainer refuses to write a model that ranks expiring task chatter above standing
preferences, on eight held-out cases, regardless of how good its aggregate metrics
look. That gate is not redundant with the metrics: it rejected a candidate scoring
0.85 on the composite. Calibration metrics are satisfiable by a model that predicts
the training mean for every input and discriminates nothing, which is exactly the
model a threshold-based consolidator must not be given.

`--source mongodb` is the path past the shipped artifact: it labels your own memories
from `access_count`, age, and whether consolidation soft-deleted them. No LLM calls,
and unlike the benchmarks it is drawn from your deployment's own distribution. It
needs a store with real usage history, so it is worth revisiting once memories have
been recalled a few times rather than on day one.

### Two Voyage endpoints, and the key decides which

Voyage embeddings are reachable two ways, and a key that works with one is
rejected by the other with a `403`:

| Key from | `voyage_base_url` | Models |
|---|---|---|
| Voyage AI | `https://api.voyageai.com/v1/embeddings` (default) | the full Voyage catalogue |
| MongoDB Atlas | `https://ai.mongodb.com/v1/embeddings` | a subset: `voyage-4*`, `voyage-3*`, `voyage-code-*`, `voyage-law-2`, `voyage-finance-2` |

An Atlas key (`al-…`) against the default URL fails with *"Voyage AI API keys
work with Voyage AI endpoints, and MongoDB Atlas API keys work with MongoDB
endpoints."* Point `voyage_base_url` at the gateway and set a model it serves:

```bash
VOYAGE_BASE_URL=https://ai.mongodb.com/v1/embeddings
VOYAGE_MODEL=voyage-4
```

The gateway models are **1024** dimensions (`voyage-3-lite` is 512), against the
`1536` default. That is handled: `ProviderManager` aligns `embedding_dimension` to
the model, but *only* while `embedding_dimension` is still the default. Pin it
explicitly and you own it, alignment is skipped, and a wrong value is your value.

The dimension is baked into the vector indexes at creation, so switching provider
or model after documents exist means re-creating `memories_vector_index`,
`episodes_vector_index`, and `cache_vector_index` at the new dimension.

**A mismatch is caught at startup, not at recall.** `AsyncMemory.create()` compares
the declared `embedding_dimension` against the model's documented output and raises
`ConfigError` if they disagree: from a built-in table for known models, so the
check works even when the embedding endpoint is down, and by probing the embedder
otherwise. If neither can answer, it logs a warning saying so rather than passing
quietly. This matters because the underlying failure is silent: Atlas accepts a
1024-dim vector into a 1536-dim index without complaint and simply never returns
that document from `$vectorSearch`, so recall goes empty and every write until
someone notices has to be re-embedded.

The realistic way to hit this is a model upgrade within one provider:
`amazon.titan-embed-text-v1` is 1536 and `amazon.titan-embed-text-v2:0` is 1024.
When the guard fires, set `embedding_dimension` to the value it names and
re-provision the index `numDimensions` to match.

### One caveat about `workers_in_process=False`

It disables the background workers, which means enrichment, consolidation, audit
flushing, and the episodic writer all stop. Without a consumer, `log_activity`
fills its bounded queue and then starts discarding the oldest turns. Set
`episodic_enabled=False` if that is what you intend, so the behaviour is
explicit rather than inferred from a full queue.

## Governance and audit

Access is profile-based: `admin`, `power_user`, `end_user`, each with allowed
operations and per-day quotas. Every operation goes through one path (access
check, then the service call, then an audit record), so there is no surface that
skips governance.

One exception, and it is deliberate: `log_activity` does not write a
per-call audit record. A turn log is high-volume by nature, and one audit write
per turn would mean logging the agent costs more writes than the agent. It still
enforces governance and rate limits on every call; the worker emits one audit
entry per flushed batch, grouped by `user_id`, since a batch can span users and
misattributing turns would be worse than no audit trail.

Profile seeding is additive. When a release adds an operation, existing profiles
gain it via `$addToSet`. Custom quotas and any operations an operator added by
hand are preserved, and nothing is ever removed.

## Development

```bash
uv sync --all-extras
uv run pytest -q          # unit suite, fully mocked; no Atlas needed
uv run ruff check         # whole repo, matching CI; no path argument
```

The lint gate takes no path. It used to be `ruff check agent_memory/`, which left
the tests unlinted for long enough to accumulate 134 findings, two of them real
defects in assertions, where a weakened check has nothing above it to catch it.
Adding a path back is a test failure, not just a style choice.

The integration tier gates on server reachability rather than an env flag: start
the server against a real cluster, then run `uv run pytest tests/integration -q`.

## Documentation

This README is the overview. [`docs/`](docs/README.md) goes deeper: the full
configuration table, both shells' surfaces, the document shapes, the governance
and audit contracts, and how-to guides for deploying, monitoring, and tuning
retention.

## License

Apache 2.0. See [LICENSE](LICENSE).
