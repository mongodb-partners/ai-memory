# How to configure episodic retention

Turn logs grow fast and most of their value decays in days. Episodic memory
therefore has a TTL index on `ts`, set to **30 days** by default, and you can
change it at runtime without a redeploy or an index rebuild.

## Change it

```python
await memory.set_activity_retention("admin-user", ttl_seconds=7 * 86400)  # 7 days
await memory.set_activity_retention("admin-user", ttl_seconds=7200)       # 2 hours
await memory.set_activity_retention("admin-user", ttl_seconds=None)       # forever
```

```bash
curl -X PUT localhost:8000/activity/retention \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "admin-user", "ttl_seconds": 7200}'
```

The return value tells you which path was taken:

```jsonc
{"status": "updated", "ttl_seconds": 7200}   // collMod: the index was modified in place
{"status": "created", "ttl_seconds": 7200}   // collMod unavailable; the index was rebuilt
{"status": "removed", "ttl_seconds": null}   // the TTL index was dropped
{"status": "error",   "ttl_seconds": 7200, "error": "..."}
```

`None` is meaningful, not missing: it drops the TTL index entirely and keeps the
log permanently. In the REST body, omitting `ttl_seconds` is the same as sending
`null`, so a truncated request means "keep forever" rather than "no change."

This is an **admin-only** operation. Retention is a compliance decision, not a
user preference, so it stays out of the `power_user` and `end_user` profiles even
though those profiles have full episodic read/write.

## Why `collMod`

The obvious implementation (drop the index, create a new one) leaves a window
with no TTL index at all. On a large collection index creation is not instant, and
during that window nothing expires. Worse, if the create fails, retention is now
silently unbounded.

`collMod` mutates `expireAfterSeconds` on the existing index. No window, no
rebuild, no reindex cost. The `create_index` fallback exists for deployments
where `collMod` on an index is unavailable, and it is a fallback rather than the
primary path for exactly the reason above.

`set_retention` never raises. Retention management should not be able to take down
a request; check the returned `status` instead.

**Check it.** A failure is a return value, so a caller that ignores the response
learns nothing, and until 4.2.0 neither did the audit log, which recorded every
call as a `success` with the requested `ttl_seconds` beside it. A failed
`{"status": "error"}` is now audited as an `error` carrying the reason, so an
operator reviewing the log sees the attempt that changed nothing. Both directions
of a silent failure matter: a lengthened retention that did not take effect leaves
turns expiring on the old schedule, and a shortened one deletes other tenants'
data on the TTL monitor's schedule with no signal in the response either way.

The `error` string is scrubbed of credential-shaped substrings before it is
returned or audited. It is a driver message, and a driver quotes the URI it
failed to authenticate against. A total failure names both attempts (`collMod`
*and* the `create_index` fallback), because "collMod is unavailable on this
deployment" and "this principal may not create an index" need different fixes.

## What TTL actually guarantees

Three things worth knowing before you rely on a number:

**Deletion is not immediate.** MongoDB's TTL monitor runs about once a minute and
deletes in batches. A 7200-second TTL means "expires after ~2 hours," not "gone at
exactly 7200 seconds." Under write pressure the lag is longer. Do not use TTL as
a security boundary. For that, use `wipe_user_data`, which deletes now.

**It keys off `ts`, not insertion time.** `log_activity` accepts an explicit `ts`.
Backfilling historical turns with their original timestamps means they may expire
immediately, which is usually what you want and occasionally a surprise. Pass no
`ts` to use write time.

**It applies to the whole collection.** A TTL index cannot be per-user. If one
tenant needs 7 years and another needs 7 days, the TTL has to cover the longest
requirement and the shorter one needs an explicit deletion job.

## Verifying it

```javascript
// In Compass or mongosh: the TTL index and its current setting
db.episodes.getIndexes().filter(i => i.expireAfterSeconds !== undefined)
// → [{ v: 2, key: { ts: 1 }, name: "ix_episodes_ttl", expireAfterSeconds: 7200 }]

// Oldest surviving turn: should sit within the retention window
db.episodes.find({}, { ts: 1 }).sort({ ts: 1 }).limit(1)
```

If `expireAfterSeconds` is absent from every index, there is no TTL and the log is
permanent. That is a valid state (`ttl_seconds=None` produces it), but it is worth
confirming it was intentional.

## Choosing a number

| Retention | Fits |
|---|---|
| `3600` to `7200` | Load tests and demos. Cleans up after itself. |
| `7 * 86400` | Active debugging. A week covers "what happened last Tuesday." |
| `30 * 86400` (default) | General production. Long enough for a support cycle. |
| `365 * 86400` | Regulated workflows that must retain agent actions. |
| `None` | Audited systems where deletion needs an explicit decision. |

The default is 30 days because it is the shortest window that still answers the
question episodic memory exists to answer. If you are storing turns to satisfy a
retention *requirement*, set the number explicitly rather than inheriting the
default. The default is a convenience, not a policy.

## The initial value

The 30-day default comes from `EPISODES_DEFAULT_TTL_SECONDS` in
`agent_memory/core/collections.py`, which is a static list of index definitions
and therefore not config-driven. `create()` applies it on first run.

To deploy with a different value from the start, call `set_activity_retention`
once after `create()`:

```python
memory = await AsyncMemory.create(config)
await memory.set_activity_retention("admin-user", ttl_seconds=7 * 86400)
```

It is idempotent, so leaving that call in a startup path is fine. It is a
`collMod` to the same value on every subsequent boot.

## See also

- [Observability](observability.md): the writer's counters
- [The document shape](../reference/episodic-document-shape.md): indexes,
  including the TTL index
