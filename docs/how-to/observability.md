# How to monitor the episodic writer

Episodic logging is deliberately fire-and-forget: `log_activity` builds the
document, enqueues it, and returns without awaiting Atlas or the embedder. That
is what keeps it off the agent's critical path — and it is also why it can
degrade without anyone noticing. Nothing raises. The counters are how you find
out.

## The counters

```python
stats = memory.activity_stats()     # synchronous — safe inside a health probe
```

```jsonc
{
  "queue_depth": 3,          // pending turns not yet written
  "queue_capacity": 1000,    // episodic_queue_size
  "worker_alive": true,      // is the consumer task running
  "enqueued": 1482,          // accepted from callers
  "written": 1479,           // reached Atlas
  "dropped": 0,              // evicted because the queue was full
  "batches": 91,             // insert_many calls
  "embed_failures": 0,       // turns stored without search_text/embedding
  "write_failures": 0,       // failed insert_many calls
  "last_write_ts": "2026-07-29T18:04:11.502841+00:00"   // ISO string, or null
}
```

`activity_stats()` is synchronous on purpose. A liveness probe that has to await
an event loop is a probe that can hang for the same reason the thing it is
checking hung.

## Reading them

The counters are cumulative since process start, so what matters is the
*relationships* between them, not their absolute values.

| Signal | What it means | What to do |
|---|---|---|
| `enqueued - written - queue_depth > 0` | Turns went in and did not come out | Check `dropped` and `write_failures` |
| `dropped` rising | The queue filled; the oldest turns were evicted | The writer cannot keep up — raise `episodic_batch_size`, or check Atlas latency |
| `queue_depth` near `queue_capacity` | Saturation is imminent | Same as above, before data loss starts |
| `worker_alive: false` with `episodic_enabled: true` | No consumer. The queue will fill, then drop | Usually `workers_in_process=False` without an external runtime |
| `write_failures` rising | `insert_many` is failing | Connection, auth, or a write concern that cannot be satisfied |
| `embed_failures` rising | Turns are stored but not recallable by search | The embedding provider is failing; `get_thread` still works |
| `last_write_ts` stale while `enqueued` climbs | The consumer is stuck, not slow | Check the process for a blocked event loop |

Two of those deserve emphasis because they are *silent partial* failures rather
than outages:

**`embed_failures` means searchable recall is broken while everything looks
fine.** The turn is in Atlas. `get_thread` returns it. `get_activity_by_correlation`
returns it. Only `recall_activity` cannot find it, because the document has
neither `search_text` nor `embedding` — the embedding is generated *before*
`search_text` is assigned, so a failure leaves both absent rather than leaving a
searchable document with no vector to match. That is the safe failure, but it
still costs you recall.

**`dropped` is the only counter that means data is gone.** When the queue is
full the *oldest* pending turn is evicted, never the newest — a stale turn is
worth less than a fresh one, and dropping the newest would mean the log goes
blind exactly when the agent is busiest. But a drop is unrecoverable.

## Wiring it to `/health`

The REST shell already does this:

```bash
curl localhost:8000/health
```

```jsonc
{
  "status": "ok",
  "episodic": { "queue_depth": 3, "written": 1479, ... }
}
```

`/health` returns the counters alongside liveness because a 200 with a saturated
queue and climbing `write_failures` is not health — it is a process that is up and
losing data. Reporting both lets the probe be honest.

If `activity_stats()` itself raises, `/health` still returns 200 with the
`episodic` key absent. A liveness probe that 500s because its *reporting* broke
takes down a healthy process.

## Suggested alerts

```
rate(dropped) > 0                        page — data is being lost
rate(write_failures) > 0 for 5m          page — nothing is reaching Atlas
rate(embed_failures) / rate(enqueued) > 0.05 for 10m   warn — recall degrading
queue_depth / queue_capacity > 0.8 for 5m              warn — saturation ahead
worker_alive == false                    page (unless episodic_enabled is false)
```

Alert on `dropped` as a *rate*, not a total. It never resets within a process
lifetime, so a threshold on the total fires forever after one bad minute and then
tells you nothing.

## Forcing a write

`flush_activity` waits, bounded, for the queue to drain:

```python
ok = await memory.flush_activity(timeout=5.0)   # → bool; never raises
```

Useful in tests and in short-lived scripts where the process would otherwise
exit before the consumer runs. `False` means the timeout elapsed with turns still
pending, not that anything failed.

You rarely need it in a long-running service: `close()` drains the queue first,
while the consumer task is still alive and the connection is still open, bounded
by `episodic_shutdown_timeout_seconds`. Cancelling the workers before draining
would silently discard turns that never reached Atlas, which is why the ordering
is fixed rather than incidental.

## Checking the data directly

The counters tell you about the writer. Compass tells you about the result:

```javascript
// Turns stored but not recallable — should track embed_failures
db.episodes.countDocuments({ search_text: { $exists: false } })

// Turns whose step counter failed. Nonzero means the counter collection is
// under pressure; the turns were kept anyway, without ordering.
db.episodes.countDocuments({ step: null })

// Throughput by hour
db.episodes.aggregate([
  { $group: { _id: { $dateTrunc: { date: "$ts", unit: "hour" } }, n: { $sum: 1 } } },
  { $sort: { _id: -1 } }, { $limit: 24 },
])
```

A large `search_text: {$exists: false}` count with `embed_failures: 0` is not a
failure — it is `episodic_embed_final_steps_only` doing its job. Mid-turn steps
are stored unembedded by design.

## Audit trail

Episodic writes appear in the audit log, but not one entry per turn. The worker
emits **one audit entry per flushed batch, grouped by `user_id`**. A turn log is
high-volume by nature and a per-call audit record would mean logging the agent
costs more writes than the agent itself. Grouping by user rather than emitting
one entry per batch matters because a batch can span users, and misattributing
turns in an audit trail is worse than having none.

Governance and rate limiting still apply on every single `log_activity` call —
it is the audit *record* that is batched, not the access check.
