# Reference: the episodic document shape

One document per logged turn, in the `episodes` collection. This page is the
contract: if you query these documents from Compass, an aggregation, or another
service, this is what you can rely on.

## Top-level fields

```jsonc
{
  "_id": ObjectId,             // coerced to a string on read
  "user_id": "u-1",            // required, non-empty. No user_id ⇒ no document
  "thread_id": "t-1",          // required, non-empty
  "conversation_id": "c-1",    // defaults to thread_id
  "agent_name": "main",        // defaults to "main"
  "ts": ISODate,               // caller-supplied, else write time (UTC)
  "messages": [ /* below */ ],
  "todos": [ /* below */ ],
  "files_touched": [ /* below */ ],
  "correlation_id": "trace-abc",  // "" when absent — never null
  "step": 0,                   // monotonic per thread; null if the counter failed
  "parent_step": null,         // step - 1, or null at step 0
  "search_text": "…",          // OMITTED unless the turn is searchable
  "embedding": [ /* floats */ ] // OMITTED unless the turn is searchable
}
```

Four of those defaults are load-bearing:

**`user_id` and `thread_id` are required.** A turn missing either is discarded
with a warning rather than stored unscoped. This is the write-side half of tenant
isolation — there is no such thing as an episodic record you can read without a
`user_id`.

**`correlation_id` is `""`, not `null`, when absent.** A null would create a null
bucket in the `(user_id, correlation_id)` index that every unattributed turn
shares.

**`step` can be `null`.** It comes from a durable per-thread counter, which is a
database round trip that can fail. When it does, the document is inserted with a
null step rather than dropped: a logged turn beats a lost one. Treat `step` as
"usually present and monotonic," not "guaranteed."

**`search_text` and `embedding` are omitted together, or not at all.** The
embedding is generated *before* `search_text` is assigned, so a failure leaves
neither field. You will never find a document that has searchable text but no
vector to match it, which would look indexed while being unfindable.

## `messages[]` — seven keys, in order

```jsonc
{
  "type": "human" | "ai" | "tool" | "system",   // defaults to "ai"
  "content": "…",                 // stringified and capped
  "tool_calls": [ … ],            // as provided; [] when none
  "tool_call_id": "call_abc",     // null when absent
  "usage": { … },                 // token usage, when the provider reports it
  "model_id": "…",                // from the message envelope, not the body
  "finish_reason": "…"            // the provider's stop reason
}
```

Key order is part of the contract, so two logs of the same turn produce
byte-comparable documents.

`content` is capped at `episodic_content_cap` characters (4000 by default).
Truncation is **visible**: the stored text ends with
`[truncated, original_size=N bytes]`, so a reader can tell something was cut and
by how much. A silent truncation is a data-integrity bug that looks like a short
answer.

Messages may be passed as objects with attributes *or* as plain dicts. Both
project identically, which is what keeps this framework-neutral.

## `todos[]` — three keys

```jsonc
{ "id": "1", "content": "Check availability", "status": "pending" }
```

`status` is one of `pending`, `in_progress`, `completed`. An unrecognized status
clamps to `pending` rather than raising — a malformed todo should not cost you
the whole turn. `text` is accepted as an alias for `content`, since agent
frameworks disagree about which name to use.

## `files_touched[]` — four keys, sorted by path

```jsonc
{ "path": "plan.md", "size": 1284, "content_hash": null, "op": "write" }
```

Derived from tool calls on assistant messages, never from the filesystem — this
projection does no I/O.

- **Latest call per path wins.** A write followed by an edit produces one entry,
  with `op: "edit"`.
- **Sorted by path**, so two logs of the same turn compare equal.
- `op` is `write` when the tool creates a file, `edit` otherwise. If you pass
  custom `fs_write_tools`, pass a matching `fs_create_tools` too, or every custom
  tool gets labelled an edit.
- `content_hash` is reserved. Callers that hash file contents themselves can
  populate it; the projection leaves it null.

Read-only tools are ignored entirely, so a turn that only searched files reports
no files touched.

## When a turn is searchable

By default only **final steps** get embedded — a step that ends in a tool request
has a question but no answer yet, so its vector would represent half a turn. Set
`episodic_embed_final_steps_only=False` to embed every turn instead: more recall
surface, more embedding cost.

The searchable text is built from the turn's first question and last answer,
capped at `episodic_search_text_cap` (2000 characters). A turn with only one role
— all human, or all assistant — produces no text, so it is **stored but not
recallable** by `recall_activity`. The library logs a warning once per thread
when that happens, because the document silently lacks two fields and would
otherwise just seem to be missing from search results.

You can always reach such a turn by `get_thread` or `get_activity_by_correlation`;
only the hybrid search path needs the embedding.

## Indexes

| Index | Keys | Purpose |
|---|---|---|
| `ix_episodes_thread_step` | `thread_id ↑, step ↑` | `get_thread` in step order |
| `ix_episodes_user_ts` | `user_id ↑, ts ↓` | Recent activity per user |
| `ix_episodes_thread_ts` | `thread_id ↑, ts ↓` | Recent activity per thread |
| `ix_episodes_correlation` | `user_id ↑, correlation_id ↑` | Trace-id lookup |
| `ix_episodes_ttl` | `ts ↑`, `expireAfterSeconds` | Retention (30 days default) |
| `episodes_vector_index` | `embedding` | Vector branch of hybrid search |
| `episodes_fts_index` | `search_text` | Full-text branch of hybrid search |

The vector index declares `user_id`, `thread_id`, and `agent_name` as filter
fields. This is not optional: **a field used in a `$vectorSearch` pre-filter must
be declared `{"type": "filter"}`**, and if it is not, the branch returns nothing
with no error at all. The failure looks exactly like "no matching documents."

The full-text index uses the `token` type for those same fields rather than
`string`, because a `string` field is analyzed and exact `equals` filtering
against it quietly stops matching.

## A note on `episodes_counters`

Per-thread step counters live in their own collection, not in `episodes`. That
keeps `episodes` homogeneous — one document shape, one TTL policy, one index set.
A counter document is not a turn, and mixing them would mean the TTL index either
expires counters that are still in use or retains turns longer than intended.
