# Reference: the MCP tools

Sixteen tools over the same `AsyncMemory` facade the library and the REST shell
use. Each tool translates transport only: resolve the caller, call the matching
facade method, convert a refusal into an error dict. No logic of its own.

Start the shell with `TRANSPORT=mcp agent-memory`, or `TRANSPORT=both` to serve
MCP at `/mcp` and REST at `/` from one process and one facade.

The MCP shell serves **streamable HTTP only**. There is no stdio subprocess mode:
`TRANSPORT=stdio` is a legacy alias that still serves streamable-http, so a client
connects by URL rather than launching the server. See
[`mcp.json.example`](../../mcp.json.example).

## Two conventions that govern every tool

**`user_id` is a declared parameter, but it is not believed.** Every tool takes
`user_id` (an MCP client legitimately names the user in single-tenant use) and
routes it through one identity resolver first:

| `auth_enabled` | Who the call acts as |
|---|---|
| `false` | The `user_id` argument. There is no token, so that is all there is |
| `true` | The verified token's identity. Naming a **different** user returns an error dict |

With auth on, a token that cannot be read is a refusal rather than a downgrade to
the client-supplied `user_id`. Failing open there would honour the argument on
exactly the deployment that configured tenant binding.

**A refusal is a return value, not an exception.** `AccessError`, its
`RateLimitError` subclass, and `IdentityError` all come back as
`{"error": "<message>"}`. MCP has no status codes, so this is how a denial
travels. The message text distinguishes throttling from denial.

That matters to anything wrapping these tools: "returned without raising" is not
evidence the call was authorised. A wrapper that assumes it, then re-reads
`user_id` from the raw arguments, reintroduces a cross-tenant write.

Anything else (a MongoDB failure, an `EmbeddingError`) propagates as an
exception and FastMCP reports it as a tool error.

## Response envelopes

Reads that return documents share one envelope:

```jsonc
{"results": [ … ], "count": 2}
```

`count` is the length of `results`, not the number of matches found. When a
response would exceed `max_response_bytes` it is truncated from the end and three
more keys appear:

```jsonc
{"results": [ … ], "count": 40, "truncated": true, "total_count": 100,
 "max_response_bytes": 16777216}
```

At least one document is always kept, even if it alone exceeds the cap. A
too-large answer is more useful than no answer.

## Semantic memory

### `store_memory`

```
store_memory(user_id: str, conversation_id: str, messages: list[dict]) -> dict
```

Stores messages as short-term memory; significant human messages additionally
seed a long-term candidate for background scoring and de-duplication. Each message
needs `content`; `message_type` defaults from `role`, or to `"human"`.

→ `{"stm_ids": ["…"], "count": 1}`

### `recall_memory`

```
recall_memory(user_id: str, query: str, memory_type: str | None = None,
              tags: list[str] | None = None, limit: int = 10,
              tier: list[str] | None = None) -> dict
```

Curated recall: hybrid search, then deduplication and importance-weighted
re-ranking. The tool to build a prompt from.

→ the `results` envelope.

`tier` is a list drawn from `["stm", "ltm"]`; omitted, both are searched. Each
result carries a `final_score`. `memory_type` is accepted and is a declared filter
field on both indexes, but nothing populates it. See
[the memory document shape](memory-document-shape.md).

### `hybrid_search`

```
hybrid_search(user_id: str, query: str, tier: list[str] | None = None,
              limit: int = 10, memory_type: str | None = None,
              tags: list[str] | None = None) -> dict
```

The raw `$rankFusion` result: reciprocal rank fusion over a vector branch and a
full-text branch in one round trip, so exact terms (SKUs, error codes, names) and
meaning both count. No re-ranking, no duplicate collapsing. Use it to see what the
index actually thinks.

→ the `results` envelope, each document carrying its fused `score`.

`memory_type` and `tags` are applied to **both** branches. A narrowing applied to
one branch only is worse than none: fusion mixes the unfiltered branch's matches
into the same ranked list, so the filter appears to work and returns wrong
documents rather than missing ones.

### `delete_memory`

```
delete_memory(user_id: str, memory_id: str | None = None,
              tags: list[str] | None = None, time_range: dict | None = None,
              confirm: bool = False, dry_run: bool = False) -> dict
```

Soft delete: documents are marked, excluded from every read, and reaped later by a
TTL index, so a mistake is recoverable for `soft_delete_purge_days`.

→ `{"deleted_count": n}`

Anything other than a single `memory_id` is a bulk delete and requires
`confirm=True`. `dry_run=True` returns the count without writing.

## Semantic cache

### `check_cache`

```
check_cache(user_id: str, query: str,
            similarity_threshold: float | None = None) -> dict
```

→ on a hit:

```jsonc
{"query": "…", "response": "…", "score": 0.97, "cache_hit": true}
```

→ on a miss: `{"cache_hit": false}`. The facade returns `None` there; the tool
substitutes the dict, because an MCP tool must return an object.

`similarity_threshold` overrides `cache_similarity_threshold` for this lookup.

### `store_cache`

```
store_cache(user_id: str, query: str, response: str) -> dict
```

→ `{"cache_id": "…"}`

### `cache_invalidate`

```
cache_invalidate(user_id: str, pattern: str | None = None,
                 invalidate_all: bool = False) -> dict
```

→ `{"user_id": "…", "deleted_count": n}`

An `admin`-category operation.

## Sticky decisions

A durable key/value the agent should not re-litigate: a confirmed address, a
locked-in plan step, a stated preference.

### `store_decision`

```
store_decision(user_id: str, key: str, value: str,
               ttl_days: int | None = None) -> dict
```

→ `{"key": "…", "status": "stored" | "updated"}`

`ttl_days` defaults to `decision_default_ttl_days`. Writing an existing key
replaces its value and restarts the TTL.

### `recall_decision`

```
recall_decision(user_id: str, key: str) -> dict
```

→ found:

```jsonc
{"key": "shipping_address", "value": "…", "created_at": "…",
 "updated_at": "…", "expires_at": "…"}
```

→ missing or expired: `{"key": "<key>", "value": null}`. An expired decision is
indistinguishable from one that never existed, which is the point: an expiry is
the decision ceasing to apply.

## Episodic memory

### `log_activity`

```
log_activity(user_id: str, thread_id: str, messages: list[dict],
             todos: list[dict] | None = None, agent_name: str | None = None,
             correlation_id: str | None = None,
             conversation_id: str | None = None) -> dict
```

Record one agent turn. Non-blocking: it enqueues and returns, and never awaits
Atlas or the embedder.

→ `{"enqueued": true, "thread_id": "t-1"}`

`enqueued: false` means the bounded queue was full and the **oldest** pending turn
was dropped to make room. A `true` means accepted, not durable. The
[counters](../how-to/observability.md) are how you confirm turns are landing.

This is the one tool whose successes are not individually audited. A turn log is
high-volume by nature, so the writer emits one audit entry per flushed batch;
governance and rate limiting still apply per call, and refusals are still audited
individually. See [Architecture](../explanation/architecture.md).

### `search_activity`

```
search_activity(user_id: str, query: str, thread_id: str | None = None,
                agent_name: str | None = None, limit: int = 5) -> dict
```

Hybrid recall over logged turns: "what did the agent actually do?"

→ the `results` envelope, with `count` labelled over turns.

A turn with only one role produces no searchable text, so it is stored but not
reachable here. Use `get_thread` or `get_correlation` for those.

### `get_thread`

```
get_thread(user_id: str, thread_id: str, limit: int | None = None,
           ascending: bool = True) -> dict
```

Replay a thread's turns in `step` order. → the `results` envelope.

### `get_correlation`

```
get_correlation(user_id: str, correlation_id: str,
                limit: int | None = None) -> dict
```

Every logged turn sharing a trace id, the join back to your tracing stack.
Accepts a W3C `traceparent`. → the `results` envelope.

Withheld from `end_user` by default: trace ids come from operators.

### `set_activity_retention`

```
set_activity_retention(user_id: str, ttl_seconds: int | None = None) -> dict
```

→ one of:

```jsonc
{"status": "updated", "ttl_seconds": 7200, "scope": "collection"}
{"status": "created", "ttl_seconds": 7200, "scope": "collection"}
{"status": "removed", "ttl_seconds": null,  "scope": "collection"}
{"status": "error",   "ttl_seconds": 7200,  "scope": "collection", "error": "…"}
```

**Read the `status`.** This is the one facade method whose failure is a return
value rather than an exception, so a result without an `error` key is not proof
the index changed.

`ttl_seconds=None` is meaningful, not missing: it drops the TTL index and keeps
turns forever.

`user_id` here is the principal the call is authorised and audited against, not a
scope. A TTL index belongs to the collection, hence `scope: "collection"`. Admin
only, and enforced independently of whether governance is switched on, because
one tenant must not be able to shorten another's retention. See
[Configure retention](../how-to/configure-ttl.md).

## Administration

### `memory_health`

```
memory_health(user_id: str) -> dict
```

→

```jsonc
{"user_id": "u-1", "total_memories": 412,
 "tier_stats": {"stm": 88, "ltm": 324},
 "enrichment_stats": {"not_applicable": 88, "complete": 312, "pending": 12}}
```

Both stat dicts are keyed by whatever values the documents actually carry, so a
key is absent rather than zero when nothing is in that state. Soft-deleted
memories are excluded from every count.

Per-user counts, not process health. For the process, use the unauthenticated
`GET /health` route the shell registers alongside the MCP endpoint. It returns
the same body the REST shell serves, so a monitor gets one answer about one
process regardless of which port it targets. Before lifespan startup it returns
`{"status": "starting"}`.

### `wipe_user_data`

```
wipe_user_data(user_id: str, confirm: bool = False) -> dict
```

Permanently deletes every document this user owns, in every user-scoped
collection. Irreversible, and not a soft delete.

→ without `confirm`, nothing is deleted:

```jsonc
{"error": "wipe_user_data requires confirm=true. This will permanently delete ALL user data."}
```

→ with `confirm=True`:

```jsonc
{"user_id": "u-1", "memories_deleted": 412, "cache_deleted": 20,
 "audit_deleted": 1804, "episodes_deleted": 990, "decisions_deleted": 3,
 "rate_limits_deleted": 7, "episode_counters_deleted": 4, "complete": true}
```

→ when a collection fails, the counts survive the failure and `complete` says so:

```jsonc
{"error": "…", "complete": false, "memories_deleted": 412, …,
 "failed_collections": ["episodes"]}
```

**Branch on `complete`, not on the absence of `error`.** What was and was not
deleted is the only thing that makes a retry safe.

Queued episodic turns are drained first and concurrent writes for the user are
refused for the duration, so no write from this process survives the erasure.
Writes from another replica are caught by a post-delete residue check, which
reports what is still there rather than asserting completeness.

The audit record for a successful wipe is filed against a reserved erasure
principal rather than the `user_id`, so asking to be forgotten does not leave a
row naming you. That principal cannot itself be wiped.

## Auto-capture

Auto-capture is MCP-only; REST is the explicit-control surface. When
`auto_capture_enabled` is true (the default) each listed tool call is also stored
as a memory, fire-and-forget, after the tool returns.

| Setting | Default |
|---|---|
| `auto_capture_enabled` | `True` |
| `auto_capture_tools` | `["recall_memory", "hybrid_search", "store_decision", "recall_decision"]` |
| `auto_capture_min_length` | `30` |
| `auto_capture_max_content_length` | `2000` |

Six tools are excluded unconditionally, regardless of `auto_capture_tools`:
`store_memory`, `delete_memory`, `cache_invalidate`, `log_activity`,
`set_activity_retention`, `wipe_user_data`. Capturing a write means writing about
writing.

A capture is skipped when the tool returned an error dict. There is no correct
account to file a refusal under: not the named user's, and not the caller's,
since it records an operation that did not happen.

The stored text is `Tool: <name> | Query: <params> | Result: <result>`, with each
part given its own share of the character budget (two thirds to the result, one
third to the params) and each truncation marked with an ellipsis. Budgeting per
part is what keeps the result present at all; a single cut across the joined
string let a long params dict consume the whole budget and drop the outcome, and a
cut inside a repr reads as a complete sentence that happens to be false.

Captures are stored under conversation id `auto:<tool_name>`, as a `system`
message, authorised as the calling identity and role.

## See also

- [REST API](rest-api.md): the other shell, and where its surface is narrower
- [Configuration](configuration.md): every setting named above
- [Governance](governance.md): which roles may call which of these
- [The memory document shape](memory-document-shape.md): what a stored memory looks like
- [The episodic document shape](episodic-document-shape.md): what `log_activity` produces
