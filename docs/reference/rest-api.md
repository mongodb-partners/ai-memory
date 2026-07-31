# Reference: the REST API

Twelve routes over the same facade the library and the MCP shell use. The shell
is a transport: Pydantic request bodies, handlers that call the facade, and
exception handlers that map typed errors to status codes. No logic of its own.

Start it with `TRANSPORT=rest agent-memory`, or `TRANSPORT=both` to serve REST at
`/` and MCP at `/mcp` from one process and one facade.

REST is the **explicit-control** surface. There is no auto-capture here — that is
MCP-only — so nothing is persisted that a caller did not ask for.

## Identity

Every route except `/health` resolves the caller through one funnel before doing
anything. What it resolves to depends on whether auth is on:

| `auth_enabled` | Who the request acts as |
|---|---|
| `false` | The `user_id` in the query string or JSON body. That is all there is |
| `true` | The token's identity. A request naming a **different** `user_id` is refused with 403 |

With auth on, `user_id` in a body or query string is honoured only when it
matches the token, and it is never silently rewritten — a request asking for
someone else's data is a request whose author is confused about whose data it is.

The role is read from the token's `auth_role_claim`, which is what makes the
governance profiles reachable. See [Governance](governance.md).

Send the token as `Authorization: Bearer <token>`. It may be a JWT (HS256, signed
with `auth_secret`) or an API key from `MEMORY_MCP_API_KEYS`.

## Status codes

| Code | Meaning |
|---|---|
| `401` | Missing or invalid bearer token |
| `403` | Valid token, but not for the identity or operation requested (`AccessError`, cross-tenant refusal) |
| `400` | Auth is off and the request named no `user_id` at all |
| `404` | `NotFoundError` |
| `429` | `RateLimitError` |
| `502` | `EmbeddingError` |

`502` rather than `500` is the whole message: the request was fine and this
service is fine — the embedding provider returned a reply that did not describe
its input, so **nothing was written** and the same request is worth sending
again. A `500` would tell the caller they found a bug and should stop.

## The read envelope

Every route that returns documents shares one envelope, built in the facade rather
than in either shell:

```jsonc
{"results": [ … ], "count": 2}
```

`count` is the length of `results`, not the number of matches found. When the
serialised documents would exceed `max_response_bytes` (16 MiB) the list is
truncated from the end — by whole documents, never mid-document — and three more
keys appear:

```jsonc
{"results": [ … ], "count": 40, "truncated": true, "total_count": 100,
 "max_response_bytes": 16777216}
```

At least one document is always kept, even if it alone exceeds the cap. `limit`
bounds the result *count*, not its size, so this is reachable without adversarial
input: a hundred long episodic turns is tens of megabytes in one response body.

## Error bodies

Error bodies come in two shapes, because they come from two mechanisms. The typed
exception handlers — 403 `AccessError`, 404, 429, 502 — return
`{"error": "<message>"}`. The identity and token checks run as FastAPI
dependencies and raise `HTTPException`, so 401, 400, and the cross-tenant 403
return `{"detail": "<message>"}`. Branch on the status code, not on a key.

## Semantic memory

### `POST /memories`

Store conversation messages as short-term memory. Significant human messages
additionally seed a long-term candidate, scored and de-duplicated in the
background.

```jsonc
{
  "user_id": "u-1",
  "conversation_id": "c-1",
  "messages": [{"message_type": "human", "content": "I'm vegetarian"}]
}
```

→ `{"stm_ids": ["..."], "count": 1}`

Each message needs `content`. `message_type` defaults from `role`, or to
`"human"`.

### `GET /memories/recall`

Curated recall: hybrid search, then deduplication and importance-weighted
re-ranking. This is the one to build a prompt from.

| Query parameter | Type | Default |
|---|---|---|
| `query` | string | **required** |
| `limit` | int | `10` |
| `user_id` | string | required when auth is off |

→ `{"results": [...], "count": n}`

### `GET /memories/search`

The raw `$rankFusion` result with scores intact — no re-ranking, no duplicate
collapsing. Use it to see what the index actually thinks.

Same parameters and response shape as `/memories/recall`.

**Both read routes are narrower than the library.** `recall()` and `search()`
also accept `tier`, `memory_type`, and `tags`; the REST handlers expose only
`query` and `limit`. Reach for the MCP shell or a direct import when you need to
filter by tier or tag.

### `DELETE /memories`

Soft-delete: documents are marked and excluded from every read, then reaped by a
TTL index, so a mistaken delete is recoverable for a window.

| Query parameter | Type | Default |
|---|---|---|
| `memory_id` | string | `null` |
| `confirm` | bool | `false` |
| `dry_run` | bool | `false` |

→ `{"deleted_count": n}`

Anything other than a single `memory_id` is a bulk delete and requires
`confirm=true`. Pair it with `dry_run=true` first to see the count without
writing. The library's `tags` and `time_range` selectors are not exposed here.

For erasure obligations use the library's `wipe_user_data`, which deletes
permanently across every collection — it has no REST route.

## Sticky decisions

A durable key/value the agent should not re-litigate: a chosen shipping address,
a confirmed plan step, a locked-in preference.

### `POST /decisions`

```jsonc
{"user_id": "u-1", "key": "shipping_address", "value": "...", "ttl_days": 90}
```

`ttl_days` is optional and defaults to `decision_default_ttl_days`.

### `GET /decisions`

| Query parameter | Type |
|---|---|
| `key` | string, **required** |

Returns the decision, or `{"key": "...", "value": null}` when there is none.

## Episodic memory

### `POST /activity`

Record one agent turn. Non-blocking: the handler builds the document and enqueues
it, and never awaits Atlas or the embedder.

```jsonc
{
  "user_id": "u-1",
  "thread_id": "t-1",
  "messages": [{"type": "ai", "content": "Booked Nopalito", "tool_calls": [...]}],
  "todos": [{"id": "1", "content": "Check availability", "status": "completed"}],
  "agent_name": "main",
  "correlation_id": "trace-abc",
  "conversation_id": "c-1"
}
```

Only `user_id`, `thread_id`, and `messages` are required. `correlation_id`
accepts a W3C `traceparent`. See [the episodic document
shape](episodic-document-shape.md) for what gets stored.

Because the write is queued, a `200` means "accepted", not "durable". The
[counters](../how-to/observability.md) are how you confirm turns are landing.

### `GET /activity/search`

Hybrid recall over logged turns.

| Query parameter | Type | Default |
|---|---|---|
| `query` | string | **required** |
| `thread_id` | string | `null` |
| `agent_name` | string | `null` |
| `limit` | int | `5` |

The library's `since` parameter is not exposed here.

### `GET /activity/thread/{thread_id}`

Replay a thread's turns in step order.

| Query parameter | Type | Default |
|---|---|---|
| `limit` | int | `null` |
| `ascending` | bool | `true` |

Reaches turns that hybrid search cannot — a turn with only one role produces no
searchable text, so it is stored but not recallable by `/activity/search`.

### `GET /activity/correlation/{correlation_id}`

Every logged turn sharing a trace id. Also reaches unsearchable turns.

| Query parameter | Type | Default |
|---|---|---|
| `limit` | int | `null` |

### `PUT /activity/retention`

Change episodic retention in place, via `collMod` on the existing TTL index.

```jsonc
{"user_id": "admin-user", "ttl_seconds": 7200}
```

→ one of:

```jsonc
// every response also carries "scope": "collection"
{"status": "updated", "ttl_seconds": 7200}   // collMod — modified in place
{"status": "created", "ttl_seconds": 7200}   // collMod unavailable; rebuilt
{"status": "removed", "ttl_seconds": null}   // TTL index dropped
{"status": "error",   "ttl_seconds": 7200, "error": "..."}
```

**This route's failure is a return value, not a status code.** It never raises,
so a `200` with `{"status": "error"}` means nothing changed. Read the `status`.

`ttl_seconds: null` is meaningful, not missing — it drops the TTL index and keeps
the log permanently. Omitting the field is the same as sending `null`, so a
truncated request means "keep forever" rather than "no change".

Admin-only, and collection-wide: a TTL index cannot be per-user, so this affects
every tenant. See [Configure retention](../how-to/configure-ttl.md).

## `GET /health`

The one **unauthenticated** route, deliberately: a probe that needs a token fails
during exactly the incident it exists to detect.

```jsonc
{
  "status": "ok",
  "episodic": {"queue_depth": 3, "written": 1479, ...},
  "workers": {"enabled": true, "running": true, ...}
}
```

`status` degrades to `"degraded"` when a worker that should be running is not — a
crashed enrichment loop leaves reads and writes working perfectly while the
reactive half of the system stops, so a probe that only ever said `ok` would not
be a probe.

Everything in this body is a counter, a boolean, or a name — never a document, a
user id, or a raw exception. Worker error strings are redacted, because a crashed
worker's exception is usually a driver error and driver errors quote the
connection string they failed on.

If the reporting itself raises, the route still returns `200` with the affected
key absent. A liveness probe that `500`s because its reporting broke takes down a
healthy process.

## OpenAPI

The shell is FastAPI, so the generated schema is at `/openapi.json` and the
interactive docs at `/docs`. The version there comes from the installed package.

## See also

- [MCP tools](mcp-tools.md) — the other shell, with the full parameter surface
- [Configuration](configuration.md) — `transport`, `host`, `auth_*`
- [Deployment](../how-to/deployment.md) — why binding a routable address without auth is refused
- [Governance](governance.md) — which roles may call what
