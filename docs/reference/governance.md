# Reference: identity, governance, and rate limiting

Three separate mechanisms, in the order a request meets them:

1. **Authentication** — is this a valid token? (401)
2. **Identity resolution** — which user does it act as? (403)
3. **Access check** — may that user perform this operation, this often? (403 / 429)

Each is independently switchable, and the defaults are `auth_enabled=False`,
`governance_enabled=False`, `rate_limit_enabled=False` — a library embedded in one
process needs none of them.

## Authentication

Enabled by `auth_enabled=True`, which requires a non-empty `auth_secret`;
`MemoryConfig` **refuses to construct** otherwise. That combination used to log a
warning and serve every route unauthenticated.

Both shells accept the same bearer token, verified two ways:

**API keys**, from the `MEMORY_MCP_API_KEYS` environment variable — not a config
field:

```bash
export MEMORY_MCP_API_KEYS="abc123=alice@acme.com,xyz789=bob@acme.com"
```

Keys are held as SHA-256 fingerprints, never in plaintext, and looked up in
constant time with respect to the submitted key. A resolved key produces claims
`{"sub": "<user_id>", "auth_method": "api_key"}` and scope `memory-mcp`.

**HS256 JWTs**, signed with `auth_secret`. `exp`, `iat`, and `sub` are all
*required*, not merely honoured when present:

- Without `exp`, PyJWT accepts a token forever, and there is no other revocation
  path here — a leaked token would be valid until the secret is rotated.
- Without `iat`, "reject everything issued before the breach" stops being
  available as an incident response.

The expected issuer is `memory-mcp`. A correctly signed token that is missing
`exp` is logged at **warning** rather than debug, because that is a minting bug in
whatever produced it rather than a forgery attempt, and the operator needs to know
which they are looking at.

Mint one with the verifier:

```python
from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier

verifier = MemoryMCPTokenVerifier(secret=config.auth_secret)
token = verifier.create_token("alice@acme.com", expires_in=86400)
```

## Identity resolution

One function, `resolve_caller`, used by both shells. It returns a `Caller`:

| Field | Meaning |
|---|---|
| `user_id` | The identity every downstream query is scoped to |
| `role` | The role claim, or `None` meaning "use `auth_default_role`" |
| `scopes` | The token's scopes |
| `authenticated` | Where the identity came from — **not** whether it is valid |

| `auth_enabled` | Behaviour |
|---|---|
| `false` | The caller-supplied `user_id` is the identity. Absent, that is a malformed request (REST: 400) |
| `true` | The token's `auth_user_id_claim` (`sub`) decides, falling back to the token's `client_id` |

With auth on, a request naming a **different** `user_id` is refused — never
silently rewritten, because a request asking for someone else's data is a request
whose author is confused about whose data it is, and answering a different question
hides that. The refusal is logged at warning: in a multi-tenant deployment it is
either a confused client or an enumeration attempt.

A token that authenticates but identifies no one is also refused. The safe reading
of "no identity" is not "any identity".

`IdentityError` maps to **403, not 401**. The token is valid; retrying with it will
not help, and the fix is to stop naming someone else. In MCP it comes back as an
error dict — [see the convention](mcp-tools.md).

`authenticated` is a stored fact rather than something inferred from `role` and
`scopes` being empty. A legitimate JWT can carry neither, so inferring would
re-open the cross-tenant hole for exactly the tokens that look least remarkable.

The role travels from the same token, which is what makes the governance profiles
reachable at all. Every facade method takes a keyword-only `role` and forwards it
to the access check; library callers omit it and get `auth_default_role`.

## The access check

Every operation passes through the same three gates, in this order:

**1. Erasure barrier.** A set-membership test against the users currently being
wiped. It is first because it is cheapest, because a write about to be refused
should not consume the caller's rate-limit budget, and because it is the one
refusal about the state of the data rather than the identity of the caller.

Only writes are barred — `store_memory`, `store_cache`, `store_decision`,
`log_activity`, `delete_memory`, `cache_invalidate`. Reads stay available and
return progressively less as collections empty, which is honest.
`ErasureInProgressError` subclasses `AccessError`, so it travels the existing
refusal paths and is audited as `denied` rather than as a fault.

**2. Governance.** When `governance_enabled`, the caller's role must list the
operation. Otherwise `AccessError`.

**3. Rate limit.** When `rate_limit_enabled`, the window counter is incremented and
compared. Over the limit is `RateLimitError`.

All three refusals are **audited**. The check used to run ahead of the audit block
entirely, so a denied operation and a throttled one left no record — the two events
an audit log exists to capture were the only two it could not show.

The audit status distinguishes them: `denied` is a decision about who the caller
is, `throttled` about how often they ask, `error` a fault in the service.
`RateLimitError` is tested first because it subclasses `AccessError`.

## Profiles

Three profiles are seeded into `governance_profiles` at startup:

| | `admin` | `power_user` | `end_user` |
|---|---|---|---|
| `max_memories_per_day` | 10000 | 1000 | 100 |
| `max_searches_per_day` | 10000 | 5000 | 500 |
| `allowed_operations` | `["*"]` | 11 operations | 8 operations |

Per operation:

| Operation | `admin` | `power_user` | `end_user` |
|---|---|---|---|
| `store_memory` | ✅ | ✅ | ✅ |
| `recall_memory` | ✅ | ✅ | ✅ |
| `hybrid_search` | ✅ | ✅ | ✅ |
| `check_cache` | ✅ | ✅ | ✅ |
| `store_cache` | ✅ | ✅ | ✅ |
| `log_activity` | ✅ | ✅ | ✅ |
| `search_activity` | ✅ | ✅ | ✅ |
| `get_thread` | ✅ | ✅ | ✅ |
| `get_correlation` | ✅ | ✅ | — |
| `delete_memory` | ✅ | ✅ | — |
| `memory_health` | ✅ | ✅ | — |
| `set_activity_retention` | ✅ | — | — |
| `cache_invalidate` | ✅ | — | — |
| `wipe_user_data` | ✅ | — | — |
| `store_decision` | ✅ | — | — |
| `recall_decision` | ✅ | — | — |

`get_correlation` is withheld from `end_user` because trace ids come from
operators. `set_activity_retention` is withheld from `power_user` because a TTL
index belongs to the collection: one tenant must not be able to shorten another's
retention.

The two decision operations are in no default profile but `admin`'s wildcard, so
with governance on, sticky decisions are effectively admin-only unless you add them
to a profile. That is a gap rather than a policy — worth knowing before enabling
governance on a deployment that uses them.

`governance_default_profile` defaults to `"default"`, and no profile is named
`default`, so an unknown role falls through to `end_user` — the most restrictive
profile, which is the right direction for a fallback.

### Editing a profile

Profiles are documents. Edit them in place:

```javascript
db.governance_profiles.updateOne(
  {role: "end_user"},
  {$addToSet: {allowed_operations: {$each: ["store_decision", "recall_decision"]}}}
)
```

A cached copy is held for `governance_cache_ttl_seconds` (300), so an edit takes up
to five minutes to take effect.

### Seeding is additive

`seed_defaults` runs at startup. Skip-if-exists is not enough: when a release adds
an operation, every existing deployment would keep a profile that silently denies
it, and the symptom is an `AccessError` on a feature the user just upgraded to get.

So an existing profile gets `$addToSet` of any operations it is missing — additive
only. Custom limits and operations an operator added by hand are left untouched,
and **nothing is ever removed**. A backfill also evicts that role's cache entry
rather than waiting out the TTL, since a stale copy is stale in exactly the
direction that denies access.

To restrict a default operation, remove it from the document *and* accept that the
next upgrade adding operations will not put it back — only the operations in that
release's defaults are re-added, and only if absent.

## Rate limiting

A **fixed** window, counted atomically in MongoDB. The window bucket's `_id` is
the composite `(user_id, operation, window_start)`, and the decision is the
post-increment value of a single `$inc` — so N concurrent callers get N distinct
values and only those within the limit proceed.

Window boundaries are derived from the epoch rather than from first-request time,
so every process in a deployment agrees where a window starts without
coordinating and all of them increment the same document.

**A fixed window can admit up to `2 × max` across a boundary** — `max` late in one
window, `max` early in the next. That is taken deliberately. A sliding window needs
either a count over per-request documents or a read-modify-write over a sorted
structure, and both reintroduce the race this replaced: under a burst, every
request read the same below-limit count and every one was admitted, so the limit
held against a single sequential caller and was absent against exactly the traffic
a rate limiter exists to bound.

### Which limit applies

| Condition | Limit |
|---|---|
| Governance off | `rate_limit_max_requests` (100) |
| Governance on, operation is a search | the profile's `max_searches_per_day` |
| Governance on, anything else | the profile's `max_memories_per_day` |

The search operations are `recall_memory`, `hybrid_search`, `check_cache`, and
`search_activity`.

Note the mismatch worth planning around: the profile numbers are *per day*, but the
window they are applied over is `rate_limit_window_seconds` (60 by default). A
profile quota is enforced per window, not per day. Set the window to `86400` if you
want the profile numbers to mean what they say.

A limit of `0` means "no requests", not "unlimited", and is short-circuited before
the round trip.

**The limiter fails open.** If MongoDB is unavailable the request is allowed and a
warning logged: refusing every request turns a counter outage into a total outage.
Governance has already run and is the control that must not be bypassed; this one
is a throughput guard, and the safe failure for a throughput guard is to allow.

Spent counters expire after `rate_limit_retention_seconds` (86400), raised to
`rate_limit_window_seconds` when that is longer — a counter *is* the enforcement
state, so expiring it inside its own window would reset a caller who had exhausted
the limit.

## Two guards that do not depend on governance

Governance is opt-in, and an authorisation rule that only exists when an optional
subsystem is enabled is not an authorisation rule — it is a default-open one. Two
checks therefore sit underneath it:

**`set_activity_retention` requires the `admin` role**, enforced in the facade
rather than by the governance service. Without this, on a default multi-tenant
deployment any authenticated caller could shorten every other tenant's retention
through a public REST endpoint — and quietly, since Atlas expires the documents
later and the caller sees only `{"scope": "collection"}`. The guard applies only
when auth is on, where a role claim exists and "every tenant" means something.

**The runner refuses to bind a routable address with `auth_enabled=False`.** See
[Deployment](../how-to/deployment.md). This one cannot live on the config class,
because the same class configures the library used in-process, where auth-off is
correct and no socket exists.

## The audit trail

Every `_run` operation writes one record:

```jsonc
{
  "user_id": "u-1",
  "operation": "memory:read",     // the category
  "tool_name": "recall_memory",   // the operation
  "status": "success" | "denied" | "throttled" | "error",
  "duration_ms": 12,
  "timestamp": ISODate,
  "metadata": { /* per-operation fields; error messages redacted */ }
}
```

Records are buffered (`audit_buffer_size`, 10) and flushed on an interval
(`audit_flush_interval_seconds`, 60). A flush never raises — a failure here must
not fail the operation being audited, which has usually already succeeded. When
MongoDB refuses the batch it goes to `audit_fallback_path` instead, rotated at
`audit_fallback_max_bytes`. Setting that path to `""` **discards** the records, and
the library warns once at construction rather than per failure.

Two deliberate exceptions:

**`log_activity` is not audited per call.** A turn log is high-volume by nature, and
routing it through `_run` produces audit amplification — logging the agent costs
more writes than the agent. The episodic writer emits one entry per flushed batch
instead. Refusals *are* audited individually: skipping the per-call record is a
volume decision about the success path, and a denial is rare, security-relevant,
and the thing an audit log is for.

**A successful `wipe_user_data` is filed against `_erased`**, a reserved principal,
not the `user_id`. `_run` audits after the service call, and the service call
deletes every `audit_log` row for that `user_id` — so the success record was written
into the collection the wipe had just cleared, under the identifier it had just
erased. A user who asked to be forgotten was left with a row naming them, dated a
millisecond after the deletion, created *by* the erasure and therefore surviving
every subsequent wipe.

Deleting the record instead is not the fix: a total, irreversible deletion is
precisely the operation that must leave a trace. So the trace is kept and the
subject dropped. `_erased` cannot itself be wiped. A *denied* wipe deleted nothing,
so it is audited the ordinary way, against the real identity.

## See also

- [Configuration](configuration.md) — every `auth_*`, `governance_*`, and `rate_limit_*` field
- [Deployment](../how-to/deployment.md) — turning this on for a served deployment
- [REST API](rest-api.md) — status codes
- [MCP tools](mcp-tools.md) — refusal as a return value
