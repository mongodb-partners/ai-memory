# Compass tabs to pre-open

Screen 3 of the talk is Compass. Four tabs, opened and scrolled to position
*before* the audience arrives — a booth crowd will not wait while you navigate a
collection tree, and clicking around live is where the demo goes quiet.

Every filter below was run against the seeded `ai4-demo` data and returns the
documents described. Connection: the Atlas cluster in `examples/memory-ui/.env`,
database `agent_memory`.

---

## Tab order matters

The tabs are in narration order, and it is the order of the talk's own argument:
one short-term document, one long-term document, one episode, then the indexes
that make all three queryable. Do not reorder them on the day — the fourth tab
only lands after the audience has seen what the first three contain.

---

## Tab 1 — a short-term document

**Collection:** `memories` · **Filter:**

```js
{ user_id: "ai4-demo", tier: "stm" }
```

Returns 12. Any one will do; the fields to point at are the same on all of them.

What to say, and the fields to have visible:

- `expires_at` — a real date, ~24h out. This is the tier's whole argument: state
  that expires on its own, enforced by a TTL index, not by application cleanup
  code that someone has to remember to write.
- `content` — the raw turn, unsummarized. Short-term memory is not condensed;
  condensing is what promotion is for.
- `importance: 0.5` — the placeholder. Unenriched. Worth naming out loud, because
  the next tab shows what a real score looks like and the contrast is the point.
- `source_stm_id: null` — this document *is* the short-term original.

**Do not promise a `thread_id` field.** These documents carry `conversation_id`
(the `episodes` collection is the one with `thread_id`). Saying the wrong field
name while it is on screen is a small error that reads as not knowing the data.

## Tab 2 — a long-term document with a real score

**Collection:** `memories` · **Filter:**

```js
{ user_id: "ai4-demo", tier: "ltm", enrichment_status: "complete" }
```

Returns 6 — one per fact. Sort by `importance: -1` and open the shellfish
allergy, which scores **0.9**: the highest in the set, and self-evidently the one
that should be.

- `importance: 0.9` — the model's own judgement, not a number the seed script
  chose. Worth saying, since the obvious audience suspicion is that the demo data
  was rigged to look good.
- `access_count` — recall reinforces. A memory that gets used gets harder to
  forget, which is the mechanism, not a metaphor.
- `source_stm_id` — points back at the Tab 1 document. The provenance chain is on
  screen: this fact started as a turn and earned durability.
- `enrichment_status: "complete"` — a background worker did this, after the turn
  returned. The user never waited for it.
- `expires_at` — **~90 days out, where Tab 1's was ~24 hours.** Same field, same
  TTL index, different `retention_tier`. That contrast is the tier distinction in
  one glance, and it is a stronger line than "long-term has no expiry" because it
  shows retention as a *value* rather than an exception: one mechanism, tuned per
  tier, no second system for the durable case.

**Do not say long-term memory has no `expires_at`.** Every LTM document carries
one — `retention_tier: "standard"` resolves to 90 days at write time. An earlier
version of this file claimed the field was absent; it is on screen with a date in
it, and the audience can read it.

Note: `summary` is absent on this data. That is correct, not missing — these
memories are single sentences, already shorter than any summary of them, so the
worker skips them (REQ-E-121). If asked: the field populates for long memories.
Do not describe it as a bug.

## Tab 3 — an episode

**Collection:** `episodes` · **Filter:**

```js
{ user_id: "ai4-demo" }
```

Returns 3, one per seeded thread. Open the `seed-thread-hosting` one and expand
`messages`.

This is the differentiator, so it gets the most airtime of the four:

- `messages[]` — the actual turn, both sides, with `tool_calls` where there were
  any. Not a summary of what happened. What happened.
- `step` / `parent_step` — a durable monotonic counter per thread. Ordered
  history, so "what did we do, in what order" is a query and not an inference.
- `search_text` — the question and the answer, joined. This is what gets embedded,
  and it is why semantic search over *activity* works at all.
- `correlation_id` — a real UUID. Ties this turn to whatever else logged under it,
  including traces if the caller passed a W3C `traceparent`.
- `ts` + the TTL on it — activity logs are the tier that grows without bound, so
  this is the one where retention is a hard requirement rather than hygiene.

`files_touched` is empty on the seeded data — the demo agent has no filesystem
tools. Do not point at it and imply otherwise. If someone asks what it is for,
that is a good moment to mention the coding-agent case, where it is the field that
answers "what did it change?".

## Tab 4 — the indexes

**Collection:** `memories` → **Indexes** tab, and keep `episodes` → **Indexes** in
reach. Verified live:

| Collection | Index | Type | Note |
|---|---|---|---|
| `memories` | `memories_vector_index` | vectorSearch | 1024 dims, READY |
| `memories` | `memories_fts_index` | search | the lexical half of the fusion |
| `memories` | `ix_memories_expires_at` | TTL | `expireAfterSeconds: 0` — expiry is per-document, from `expires_at` |
| `memories` | `ix_memories_deleted_at_ttl` | TTL | 30 days — soft-deletes are reaped, not kept forever |
| `episodes` | `episodes_vector_index` | vectorSearch | 1024 dims, READY |
| `episodes` | `episodes_fts_index` | search | |
| `episodes` | `ix_episodes_ttl` | TTL | 30 days on `ts` |

The line this tab exists to earn: **both vector indexes are 1024 dimensions in the
same cluster, and the TTLs are indexes rather than cron jobs.** Four boxes
collapse to one because the retrieval, the expiry, and the records are all the
same engine.

`expireAfterSeconds: 0` on `ix_memories_expires_at` is worth one sentence if the
room is technical — it is not "expire immediately", it is "expire at the date in
the field", which is how per-document TTL works in MongoDB and surprises people.

---

## Setup order — this one bites

**Start the demo server first, then seed.** Not the other way round.

`ConsolidationWorker.run` calls `consolidate()` *before* its first sleep, so a
server booting up immediately promotes every eligible short-term memory. Seed
first and the server's own worker consumes all 5 promotion candidates within
seconds of startup — Compass pipeline `03-stm-to-ltm-promotion.json` then returns
zero rows, and Tab 1's STM count drops from 12 to 7 while LTM doubles. Nothing
errors. The demo just quietly stops showing the thing it was built to show.

```bash
# 1. Server up, and give it time to finish its startup consolidation pass
cd examples/memory-ui
uv run --extra demo uvicorn server.app:app --host 127.0.0.1 --port 8100

# 2. Then seed, in a second shell
uv run --extra demo python -m demo.seed --user ai4-demo

# 3. Empty the second user. Wipe, do not seed — see below.
uv run --extra demo python -m demo.seed --user alex --wipe-only

# 4. Then verify the pipelines, which also confirms 03 has rows
uv run --extra demo python -m demo.compass_pipelines \
    --query "what can't I eat?" --user ai4-demo --out /tmp/pipelines
```

Expected after step 4: `stm 12 · ltm 6 · episodes 3 · candidates 5`, and all four
pipelines non-empty. If `candidates` is 0 or `03` returns nothing, the server ate
them — re-seed, and do not restart the server afterwards.

**Step 3 is not redundant, and it is the one you will skip.** The second user proves
per-user isolation by recalling nothing, so it must be empty — but rehearsing types
that beat's question at it with memory ON, and the question is *stored*. Measured
after one dry run: `alex` held 2 STM documents, 1 episode, and 1 cache entry, one of
which was the demo's own question verbatim. The next run can then recall its own
residue and return hits where the whole point is zero. `--wipe-only` clears a user
across all three collections and plants nothing, which `--user` alone cannot express.

Verify it reports `memories=0 episodes=0 cache=0`. Anything nonzero is residue this
step just saved you from showing.

## Pre-flight, in order

1. Server up, health 200.
2. Seed the demo user. Confirm the counts above.
3. Wipe the second user. Confirm `memories=0 episodes=0 cache=0`.
4. Compass pipelines verified non-empty.
5. All four tabs open, in order, scrolled to the field being discussed.
6. One warm-up turn through the UI, **as the demo user**. Atlas Search indexes are
   eventually consistent, so the first turn after a write may miss; spend that miss
   in private rather than on stage. Do not warm up as the second user — that is how
   the residue gets planted.
7. Recorded OFF-vs-ON capture queued and ready in a window you can reach without
   alt-tabbing through anything with credentials in it.
