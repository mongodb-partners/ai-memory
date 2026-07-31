# Reference: the memory document shape

One document per memory, in the `memories` collection. Both semantic tiers live
here: short-term and long-term are a field, not a collection. This page is the
contract: if you query these documents from Compass, an aggregation, or another
service, this is what you can rely on.

For the episodic tier, see [the episodic document
shape](episodic-document-shape.md).

## Fields at insert

```jsonc
{
  "_id": ObjectId,             // coerced to a string on read
  "user_id": "u-1",
  "tier": "stm",               // "stm" | "ltm"
  "content": "I'm vegetarian", // the memory itself
  "summary": null,             // set by enrichment or STM compression
  "embedding": [ /* floats */ ],
  "memory_type": null,         // see below: filterable, never populated
  "retention_tier": "ephemeral",  // decides expires_at
  "tags": [],                  // from the caller's message
  "importance": 0.5,           // neutral prior until scored
  "access_count": 0,
  "last_accessed": null,
  "conversation_id": "c-1",
  "message_type": "human",     // "human" | "ai" | "system"
  "source_stm_id": null,       // set on an LTM candidate; links to its STM twin
  "enrichment_status": "not_applicable",
  "enrichment_retries": 0,
  "created_at": ISODate,
  "updated_at": ISODate,
  "expires_at": ISODate,       // created_at + the retention tier's TTL
  "deleted_at": null,
  "is_deleted": false
}
```

Three more fields appear later, written by the workers rather than at insert:

| Field | Written when | Meaning |
|---|---|---|
| `enrichment_claimed_at` | A worker claims the document | A lease, `$unset` on completion or failure |
| `merge_target_id` | Evolution queues a merge | The `_id` this document will be folded into |
| `duplicate_of` | Reinforcement retires a duplicate | The `_id` that already held this content |

`duplicate_of` is why a deduplicated memory is distinguishable from one that
vanished. Without it, an operator seeing the count drop cannot tell deliberate
deduplication from data loss.

**What a read returns is not quite what is stored.** Reads project `embedding` out
(a 1024- or 1536-float array per document would dominate the response for no
reader benefit) and coerce BSON to JSON-safe values: `_id` becomes a string and
every `datetime` becomes an ISO string. `recall` also strips its internal
`vs_score` after ranking, leaving `final_score`; `search` leaves the fused `score`
in place. Query the collection directly and you see the raw BSON above.

## Two fields per stored message

`add()` writes **one STM document per message**, and *additionally* an LTM
candidate for each **human** message of at least `ltm_candidate_min_chars`
characters (31 by default). The candidate carries the same `content` and
`embedding`, with:

| Field | STM document | LTM candidate |
|---|---|---|
| `tier` | `stm` | `ltm` |
| `retention_tier` | `ephemeral` | `standard` |
| `enrichment_status` | `not_applicable` | `pending` |
| `source_stm_id` | `null` | the STM document's `_id` |

Both are searchable immediately, which is why `source_stm_id` exists: `recall`
uses it to collapse the pair back to one result, keeping whichever scored higher.
`search` does not deduplicate, so a raw hybrid search legitimately returns both.

Assistant messages are never candidates, at any threshold. Only candidates are
enriched, promoted, or curated by `recall`.

## `retention_tier` → `expires_at`

`expires_at` carries a TTL index with `expireAfterSeconds: 0`, so Atlas deletes
the document when that timestamp passes. The value is stamped at write time from
the retention tier:

| `retention_tier` | Duration | Setting |
|---|---|---|
| `ephemeral` | 24 hours | `stm_ttl_hours` |
| `temporary` | 7 days | `ltm_retention_temporary_days` |
| `standard` | 90 days | `ltm_retention_standard_days` |
| `reference` | 180 days | `ltm_retention_reference_days` |
| `critical` | 365 days | `ltm_retention_critical_days` |

An unrecognized tier falls back to `standard` rather than raising.

Because the duration is applied per document at write time, changing one of these
settings affects memories written afterwards only, unlike the collection-wide TTL
indexes on the cache, audit log, and episodes.

Promotion re-stamps `expires_at` against the new tier. It has to: a promoted
document that kept its ~24-hour short-term expiry would be deleted the next day
while every other field said it was long-term.

## `tier` and `enrichment_status`: the lifecycle

`enrichment_status` is the state machine the background workers run on. Values:

| Status | Meaning |
|---|---|
| `not_applicable` | An STM document. Never enriched |
| `pending` | Queued for enrichment |
| `merge_pending` | Enrichment found a near-duplicate; queued for merge instead |
| `complete` | Enriched, or finished by deduplication |
| `failed` | `enrichment_max_retries` attempts exhausted |

The enrichment worker claims the oldest `pending` or `merge_pending` document,
scores its importance, summarizes it, and asks the evolution check what to do:

- **`created`**: nothing similar. Status → `complete`, with `importance` and
  `summary` written.
- **`reinforced`**: similarity above `reinforce_threshold` (0.85). The existing
  memory's `importance` is multiplied by 1.1 (capped at 1.0) and its
  `access_count` incremented; this document is soft-deleted with `duplicate_of`
  set. Exactly one live document per piece of content.
- **`merge_queued`**: similarity above `merge_threshold` (0.70). This document
  goes to `merge_pending` with `merge_target_id` set, and `enrichment_retries` is
  reset, because a merge is a fresh unit of work rather than a continuation.

A `merge_pending` document is then claimed again: the LLM merges the two contents,
the result is **re-embedded**, and both `content` and `embedding` are written
together. That ordering is load-bearing. A merged `content` beside a pre-merge
`embedding` reads correctly in Compass and searches as only half of itself. If the
target has since been deleted, the merge is abandoned and the status goes to
`complete` rather than resurrecting deleted content.

`failed` documents are countable and queryable rather than silent, which is the
point: `db.memories.countDocuments({enrichment_status: "failed"})` is the metric
to watch.

The claim is a **lease**, not a lock. A document claimed longer ago than
`enrichment_lease_seconds` (300) is reclaimable, because a worker crashing
mid-LLM-call would otherwise strand it forever.

The consolidation worker runs on a longer cycle and does three things:

- **Compress.** An STM document older than `stm_compression_age_hours` with no
  `summary` gets one. Very short content is skipped, since a single conversational
  turn is already as short as its summary would be.
- **Forget.** An `ltm`, `complete` document whose `importance` is below
  `forgetting_score_threshold` (0.1) is soft-deleted.
- **Promote.** An `stm` document at or above `promotion_importance_threshold`
  (0.6) *and* `promotion_access_threshold` accesses (2) *and* older than
  `promotion_age_minutes` becomes `tier: "ltm"`, `retention_tier: "standard"`,
  `enrichment_status: "pending"` (so it is enriched as a long-term memory), with
  `expires_at` re-stamped.

Forgetting and promotion compare against **absolute** thresholds, which is why
importance-scorer calibration matters and not just ranking quality. A scorer that
ranks well but sits systematically low forgets more and promotes less, and the
symptom is degraded recall weeks later rather than an error.

## Soft delete

Deletes set `deleted_at`, `is_deleted: true`, and `updated_at`. Nothing is removed
immediately.

Both fields exist because the two query paths need different types: the vector
index and every `find` filter on `deleted_at: null`, and the Atlas Search index
filters on `is_deleted: false`, because a `token` field cannot express "is null".
They must agree. A document with one set and not the other is visible to one
branch of a hybrid search and not the other.

A separate TTL index on `deleted_at` reaps soft-deleted documents after
`soft_delete_purge_days` (30). It carries
`partialFilterExpression: {"deleted_at": {"$type": "date"}}`, so live documents,
whose `deleted_at` is `null`, are not candidates for expiry.

`wipe_user_data` is the hard delete, and it is a different operation: permanent,
across every user-scoped collection, and irreversible. See
[MCP tools](mcp-tools.md).

## `memory_type`: filterable, never populated

`memory_type` is a parameter on `recall` and `search`, and a declared filter field
in **both** the vector and full-text indexes. Nothing writes a value to it: every
insert path sets it to `null`, and no worker populates it.

So a `memory_type`-scoped query is well-formed and returns nothing. The field is
reserved for a caller that classifies its own memories by writing to the
collection directly, and the filter plumbing exists and is correct, but treat it
as unused rather than as a working feature.

`tags`, by contrast, is populated from the caller's message and works end to end.

## Filtering by tags

Tags are matched all-of, and the spelling matters in both branches:

- The `$vectorSearch` pre-filter uses `$and` of single-value equalities, one per
  tag, **not** `{"tags": {"$all": [...]}}`. `$all` is not a supported pre-filter
  operator, and an unsupported operator does not raise: the branch matches nothing,
  so every tag-filtered search comes back empty and reads as "no memories carry
  those tags".
- The Atlas Search side uses one `equals` clause per tag inside `compound.filter`,
  which is itself an AND.

Both rely on the fact that a filter or `equals` against an array field matches when
any element matches, which is what makes an `$and` of equalities mean all-of.

## Indexes

| Index | Keys | Purpose |
|---|---|---|
| `ix_memories_expires_at` | `expires_at ↑`, `expireAfterSeconds: 0` | Retention, per document |
| `ix_memories_user_tier_created` | `user_id ↑, tier ↑, created_at ↓` | Tenant-scoped tier listing (partial: `deleted_at: null`) |
| `ix_memories_conversation` | `user_id ↑, conversation_id ↑` | Per-conversation reads (partial: `deleted_at: null`) |
| `ix_memories_enrichment_queue` | `enrichment_status ↑, enrichment_claimed_at ↑, created_at ↑` | The worker's FIFO claim query |
| `ix_memories_deleted_at_ttl` | `deleted_at ↑`, `expireAfterSeconds` | Purge soft-deleted documents (partial: `deleted_at` is a date) |
| `memories_vector_index` | `embedding` | Vector branch of hybrid search |
| `memories_fts_index` | `content`, `summary` | Full-text branch of hybrid search |

`ix_memories_enrichment_queue` puts `enrichment_claimed_at` between the equality
prefix and the sort deliberately. It cannot serve the claim's `$or` as an index
bound, but keeping the field in the index means the filter is applied without
fetching each document, and the claim runs on every poll of a busy queue.

The vector index declares `user_id`, `tier`, `deleted_at`, `memory_type`, and
`tags` as filter fields. This is not optional: **a field used in a
`$vectorSearch` pre-filter must be declared `{"type": "filter"}`**, and if it is
not, the branch returns nothing with no error at all. The failure is
indistinguishable from "the user has no memories of that type".

The full-text index maps `content` and `summary` as `string` (they are searched)
and `user_id`, `tier`, `is_deleted`, `memory_type`, and `tags` as `token` (they
are filtered). An analyzed `string` field cannot back an exact `equals` filter, and
gets it wrong quietly.

`memory_type` and `tags` are declared in **both** indexes on purpose. A pre-filter
that only one branch of a `$rankFusion` applies is not a filter. The unfiltered
branch contributes matches that ignore it, and fusion mixes them into the same
ranked list.

`numDimensions` on the vector index comes from `embedding_dimension`, and Atlas
cannot edit it in place. Changing the dimension therefore means dropping and
rebuilding the index, which leaves every already-stored vector at the old width
and unreturnable by `$vectorSearch`, silently. That is what
`allow_embedding_dimension_change` guards. See
[Configuration](configuration.md).

## How `recall` ranks

`recall` over-fetches (`limit * 2` from a `numCandidates` of `limit * 10`),
collapses STM/LTM pairs, then re-sorts on three components:

```
final_score = alpha * recency + beta * importance_score + gamma * relevance

recency          = exp(-age_days / 30)
importance_score = importance * min(1 + ln(access_count + 1), 3.0) / 3
relevance        = the vector score
```

Defaults are `ranking_alpha=0.2`, `ranking_beta=0.3`, `ranking_gamma=0.5`, so
relevance dominates, with recency and reinforcement as tiebreakers.

Returned documents have their `access_count` incremented and `last_accessed`
stamped. Reading a memory is what makes it eligible for promotion, so a recall is
also a write.

## See also

- [The episodic document shape](episodic-document-shape.md): the other tier
- [Configuration](configuration.md): every threshold and duration named here
- [MCP tools](mcp-tools.md) / [REST API](rest-api.md): the read and write surfaces
- [Architecture](../explanation/architecture.md): why the workers are shaped this way
