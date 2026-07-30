# Architecture

`agent-memory` has one shape, repeated at every level: a thin facade over
stateless services over one database connection. Nothing in the library imports an
agent framework.

```
              MCP shell            REST shell            direct import
                  │                    │                      │
                  └────────────────────┼──────────────────────┘
                                       ▼
                         AsyncMemory  (the facade)
                         Memory       (blocking twin)
                                       │
                    ┌──────────────────┴──────────────────┐
                    │            _run / _check_access     │
                    │   access check → service → audit    │
                    └──────────────────┬──────────────────┘
                                       ▼
      MemoryService  EpisodicService  CacheService  DecisionService
      GovernanceService  AuditService  RateLimiter
                                       │
                                       ▼
                    DatabaseManager  (one client, one pool)
                                       ▼
                              MongoDB Atlas
                    memories · episodes · cache · decisions
                    governance_profiles · audit_log
```

## The facade is the only orchestrator

Services are stateless and each owns exactly one collection. They do not call each
other, they do not check access, and they do not write audit records. Everything
that spans concerns happens in the facade's `_run`:

```python
async def _run(self, user_id, operation, category, fn, **audit_fields):
    await self._check_access(user_id, operation)   # governance + rate limit
    result = await fn()                            # the service call
    await self._audit(user_id, operation, category, **audit_fields)
    return result
```

One path means there is no surface that skips governance. Adding an operation that
forgets the access check requires bypassing `_run` deliberately, which is visible
in review rather than inferable only from behaviour.

Both shells wrap the same facade. The MCP tools and the REST routes are
adapters — argument shapes and error mapping — with no logic of their own. That is
why `TRANSPORT=both` can share a single instance: there is only one place the
behaviour lives.

## The one deliberate exception

`log_activity` does not go through `_run`. It calls `_check_access` and enqueues.

`_run` writes one audit record per call. A turn log is high-volume by nature, so
routing it through `_run` means logging the agent costs more writes than the agent
does — audit amplification, where the observability system becomes the load.

Governance and rate limiting still apply to every call. What is batched is the
audit *record*: the worker emits one entry per flushed batch, grouped by
`user_id`. Grouping by user matters because a batch can span users, and
misattributing turns in an audit trail is worse than not having one.

This exception is pinned by tests from three directions — that no audit record is
written, that governance still denies, and that rate limits still throttle — so a
future refactor cannot quietly turn it back into a full `_run` call or quietly
drop the access check.

## The episodic write path

This is the most carefully built part of the library, because it has to be fast
and it has to not lose data, and those pull in opposite directions.

```
log_activity(...)                      ← caller's coroutine
  ├── _check_access                    ← governance + rate limit
  ├── projection.py builds the doc     ← pure CPU, no I/O
  └── queue.put_nowait                 ← returns
                                          │
             ┌────────────────────────────┘
             ▼
   EpisodicWorker.run()                 ← ONE consumer task
     ├── batch up to episodic_batch_size, or
     │   episodic_flush_interval_seconds elapses
     ├── durable step per thread        ← find_one_and_update $inc
     ├── embed final steps              ← providers.embedding
     └── insert_many                    ← one round trip per batch
             │
             └── one audit entry per batch, grouped by user_id
```

Six properties hold this together, each of which is a bug if reversed:

**Nothing on the caller's path awaits I/O.** The projection is pure CPU and the
enqueue is non-blocking. The caller cannot be slowed by Atlas latency or by a slow
embedding provider.

**Counters increment before the put, not after.** If the order were reversed the
consumer could dequeue and decrement before the producer incremented, orphaning
the in-flight count so `flush()` would never reach zero.

**Exactly one consumer task.** FIFO per thread only holds with a single consumer,
and `step` monotonicity depends on it. Two consumers would interleave.

**A full queue drops the oldest, and counts it.** The newest turn always survives.
A stale turn is worth less than a fresh one, and evicting the newest means the log
goes blind exactly when the agent is busiest.

**A step-counter failure inserts with a null step.** The counter is a database
round trip and can fail. A logged turn without ordering beats no logged turn, so
`step` and `parent_step` go `None` and the insert proceeds.

**Embed before assigning `search_text`.** An embedding failure then leaves
*neither* field. The alternative ordering produces a document with searchable text
and no vector to match it — indexed-looking and unfindable, the worst outcome
because it is invisible.

Every exception in the worker is swallowed and counted rather than raised. There is
no caller left to raise to, and a crashed consumer means the queue fills and
everything is lost. [Observability](../how-to/observability.md) covers reading
those counters.

## Shutdown ordering

`close()` drains episodic **first**, while the consumer task is alive and the
connection is still open, bounded by `episodic_shutdown_timeout_seconds`. Only then
are the workers cancelled, the audit buffer flushed, and the connection closed.

Cancelling workers first would leave queued turns with no consumer and a closing
connection — silently discarding writes the caller believes succeeded. The
ordering is a correctness requirement, not a tidiness preference.

The audit flush comes *after* the cancellations for the mirror-image reason.
`AuditFlushWorker` may be mid-`insert_many` when it is cancelled; `flush()` puts
that batch back in the buffer instead of dropping it, so the flush that follows
is what actually writes it. Flushing before cancelling would leave that window
open.

`AuditService.flush()` is serialised on a lock, and what it buys is a
postcondition rather than throughput: when it returns, everything buffered at
call time has reached MongoDB or the fallback file — *including* entries a
concurrent flush had already taken out of the buffer. `wipe_user_data` depends on
that. It flushes before deleting so no buffered row naming the user outlives the
wipe; a flush that returned while another's write was still in flight would let
that row land after the delete, undoing the erasure the same way an undrained
episodic queue would.

## Why `episodes` is a separate collection

The `memories` tier has deduplication, importance scoring, reinforcement, merging,
and calibrated re-ranking. None of that should touch a turn log: turns are
append-only facts about what happened, not knowledge to be consolidated. Merging
two similar turns would be data loss.

The document shape, retention policy, index set, and query patterns all differ
too. So: separate collection, same database, same cluster. The "one database
instead of four systems" claim is about operational surface, and collections are
how a single database serves different shapes — not a compromise on it.

What they *share* is the retrieval pipeline. `services/search_pipeline.py` holds
one parameterized `$rankFusion` builder used by both tiers, so a fix to the fusion
logic cannot land in only one of them.

## Hybrid retrieval

Both semantic and episodic recall use `$rankFusion` — MongoDB's native reciprocal
rank fusion — over two branches: `$vectorSearch` for meaning, `$search` for exact
terms. Fusion happens *in the database*, not in application code, so there is no
round trip to merge result sets and no re-implementation of RRF to get subtly
wrong.

Two index details are load-bearing, and both fail *silently*:

**Fields used in a `$vectorSearch` pre-filter must be declared
`{"type": "filter"}`.** An undeclared field does not raise; the branch returns
nothing. `user_id`, `thread_id`, and `agent_name` are declared for this reason.

**Fields backing an exact `equals` filter in Atlas Search must use the `token`
type, not `string`.** A `string` field is analyzed, so exact equality quietly
stops matching.

Both failures look identical to "no matching documents," which is why they are
called out here rather than left to discovery.

`user_id` goes into **both** branches — the `$vectorSearch` filter and the
`$search` compound clause. Isolation is enforced by the engine, not by callers
remembering to add it. `since` is applied *after* fusion, because `ts` is high
cardinality and declaring it as a vector-index filter field would bloat the index
for a rarely-used narrowing.

## Framework neutrality

There is no agent framework in the dependency list, and no framework-specific code
in the library. `core/projection.py` accepts both attribute-style message objects
and plain dicts, and projects them identically. That is the entire adaptation
layer — everything above it is a MongoDB library.

The practical consequence: a hand-rolled agent loop, a framework, or a background
job all feed the same API, and switching frameworks does not migrate your memory.

## Configuration

`MemoryConfig` is the library surface; `MCPConfig` is the server surface, which
extends it. `MemoryConfig.from_env()` reads the environment case-insensitively.

One asymmetry worth knowing: `ProviderManager` **mutates config in place** during
`create()`. With Voyage, `embedding_dimension` auto-aligns to the model. Both
vector indexes must then agree on `numDimensions`, so `create()` validates that the
configured dimension matches what the provider actually returns and fails fast if
not. A mismatch otherwise surfaces much later as an empty result set with no
error — the same silent-failure class as the index gotchas above.

## Background work

Four workers run in-process by default: enrichment (STM → LTM promotion),
consolidation (reinforce, merge, decay), audit flush, and the episodic writer.

`workers_in_process=False` disables all four. That is correct when an external
runtime owns background work, and a trap otherwise: without a consumer,
`log_activity` fills its bounded queue and then starts discarding the oldest turns.
Set `episodic_enabled=False` alongside it so the behaviour is explicit rather than
inferred from a full queue and a climbing `dropped` counter.

## See also

- [Why episodic memory](why-episodic-memory.md) — what the tier is for
- [The document shape](../reference/episodic-document-shape.md) — the contract
- [Observability](../how-to/observability.md) — the counters
- [Per-user scoping](../how-to/per-user-scoping.md) — the context var
