# How to deploy the server

The library runs in your process and needs no deployment. This page is for the
other case: running `agent-memory` as a server that other things connect to.

Three decisions, in order: which transport, how it authenticates, and where it
binds. The third one will refuse to start if you get the second one wrong, which
is deliberate.

## Pick a transport

```bash
export MONGODB_CONNECTION_STRING="mongodb+srv://..."
TRANSPORT=both agent-memory
```

| `TRANSPORT` | Serves |
|---|---|
| `mcp` | MCP over streamable HTTP, plus `/health` |
| `rest` | The REST API at `/` |
| `both` | REST at `/` and MCP at `/mcp`, from one process |

Unset, `transport` defaults to `streamable-http`, which means `mcp`.

`both` is one process, one Atlas connection pool, one set of workers, and **one
shared facade**, not two servers in a trench coat. Anything written over REST is
immediately visible over MCP.

`streamable-http` and `stdio` are accepted as legacy aliases and both mean `mcp`.
There is no stdio subprocess mode: an MCP client connects by URL rather than
launching the server. See [`mcp.json.example`](../../mcp.json.example).

Any other value raises at startup rather than falling back to a default.

## Turn on authentication

With `AUTH_ENABLED=false` every request names the `user_id` it acts as. In-process
that is correct, since the calling app has already authenticated its user. On a port
that anything can reach, it means any client can read or permanently erase any
tenant's memories, with no record of who did.

```bash
AUTH_ENABLED=true
AUTH_SECRET=<at least 32 bytes>
```

`AUTH_ENABLED=true` with an empty secret **refuses to construct**. It used to log
a warning and serve every route unauthenticated, which is the worst available
outcome: the operator asked for auth, the deployment reports healthy, and the only
evidence is one startup log line.

Then give callers a credential. Either works, on both shells, as
`Authorization: Bearer <token>`:

```bash
# Static keys, for service-to-service callers you control
MEMORY_MCP_API_KEYS="k-alice=alice@acme.com,k-bob=bob@acme.com"
```

```python
# Or mint HS256 JWTs, for callers whose identity comes from elsewhere
from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier

print(MemoryMCPTokenVerifier(secret=AUTH_SECRET).create_token("alice@acme.com"))
```

Multi-tenant deployments want `GOVERNANCE_ENABLED=true` and
`RATE_LIMIT_ENABLED=true` alongside this. See [Governance](../reference/governance.md)
for what the roles allow and how the quotas are counted.

For a deployment where unauthenticated access is never acceptable regardless of
how the rest of the environment ends up set, add
`REQUIRE_AUTH_FOR_MULTI_TENANT=true`. It is the inverse assertion (refuse to start
*without* auth), and it is worth setting even when `AUTH_ENABLED=true` is already
there, because it survives someone else's `.env` edit.

## Where it binds, and the refusal

`HOST` defaults to `127.0.0.1`. **The runner refuses to bind anything routable
while auth is off:**

```
RuntimeError: Refusing to serve 0.0.0.0:8000 with authentication disabled.
```

Loopback means anything in `127.0.0.0/8`, plus `localhost` and `::1`. Everything
else is treated as routable, including the empty string and `0.0.0.0`, because
when an address is not recognizably local, the safe reading is that it is
reachable.

Three ways out, and only the first two are good ones:

```bash
AUTH_ENABLED=true AUTH_SECRET=...    # secure it
HOST=127.0.0.1                       # keep it local (the default)
ALLOW_UNAUTHENTICATED_NETWORK_ACCESS=true   # accept the risk
```

The third is a real configuration (an internal service behind its own gateway)
and it logs a warning on **every** start rather than once, because "we set that
flag for a spike" is how it survives into production unnoticed.

The check runs before the transport is dispatched, so it holds for all three. It
is not a validator on the config class: the same class configures the library used
in-process, where auth-off is right and there is no socket at all.

## Docker

```bash
docker compose up --build
```

The image is `python:3.11-slim`, installs with `uv` from `uv.lock`, and runs as
uid 10001. Nothing needs root, since it binds 8000 and every write goes to Atlas
rather than the filesystem. Configuration comes from `.env` via `env_file`.

Four lines in `docker-compose.yml` are load-bearing:

```yaml
build:
  context: .
  target: server           # the Dockerfile is multi-stage and its last stage is
                           # `demo`, so an omitted target builds the wrong image
environment:
  HOST: 0.0.0.0            # the published port only reaches a process bound to
                           # every container interface
  ALLOW_UNAUTHENTICATED_NETWORK_ACCESS: "true"
ports:
  - "127.0.0.1:8000:8000"  # published on the *host's* loopback, not 0.0.0.0
```

`target: server` selects the right stage from the multi-stage Dockerfile. Without
it Docker builds the final stage, which is `demo`, producing an image that listens
on 8100 rather than 8000. The healthcheck then fails because it probes the wrong
port, and the documented command above silently starts the sample UI backend
instead of the memory server.

`HOST: 0.0.0.0` is routable, so the runner refuses it with auth disabled unless
the operator says otherwise — hence `ALLOW_UNAUTHENTICATED_NETWORK_ACCESS=true`
in the compose environment, which logs a warning on every start. The check reads
the address the process binds inside the container; it cannot see the host-side
port mapping, so the two are independent concerns.

`"127.0.0.1:8000:8000"` is the second, separate control: it limits who can reach
the published port. `"8000:8000"`, the spelling most examples use, publishes on
every host interface, which on shared wifi or any VM with a public IP puts the
memory store on the internet. Widen the left side only alongside
`AUTH_ENABLED=true`.

The healthcheck polls `/health` with a 60-second `start_period`, which covers
`create()` provisioning indexes on a cold Atlas cluster.

The sample UI is behind a compose profile, so it stays out of a plain `up`:

```bash
docker compose --profile demo up --build   # + demo backend (8100) and UI (5173)
```

The UI's nginx proxies `/api` to the demo backend with `proxy_buffering off`,
without which SSE tokens arrive in one batch at the end of a turn. The demo
backend embeds the library in-process rather than calling the server on 8000, so
the two are siblings sharing an Atlas database — the UI works whether or not the
server container is running.

`scripts/docker_setup.sh` wraps all of this, waits on container health rather
than on open ports, and seeds the demo user afterwards.

## Probe it

`GET /health` is the one unauthenticated route, deliberately: a probe that needs a
token fails during exactly the incident it exists to detect. Both shells serve the
same body from the same function, so a monitor gets one answer about one process
regardless of which port it targets.

```bash
curl -s localhost:8000/health
```

```jsonc
{
  "status": "ok",                    // or "degraded"
  "episodic": {"queue_depth": 3, "written": 1479, "dropped": 0, ...},
  "workers": {
    "enabled": true, "running": true,
    "workers": {"enrichment": {...}, "consolidation": {...},
                "audit-flush": {...}, "episodic-writer": {...}}
  }
}
```

**Alert on `status != "ok"`, not on the HTTP code.** A crashed enrichment or
consolidation loop leaves reads and writes working perfectly, since only the
reactive half of the system stops, so the process would otherwise report healthy
while memories are never enriched, promoted, or forgotten. `status` degrades to
`degraded` when a worker that should be running is not.

Two things this route will not do:

- **Return 500 because its own reporting broke.** If the counters raise, the
  affected key is simply absent and the code is still 200. A liveness probe that
  takes down a healthy process is worse than a gap in a payload.
- **Leak anything.** Every value is a counter, a boolean, or a name. Worker error
  strings are redacted, because a crashed worker's exception is usually a driver
  error and driver errors quote the connection string they failed on.

Before lifespan startup, and after shutdown, the MCP shell's route returns
`{"status": "starting"}`. Treat that as not-ready rather than as failed.

The episodic counters in that body are the ones worth alerting on; see
[Observability](observability.md) for what each means when it moves.

## Workers

By default the process runs four background loops: enrichment, consolidation,
audit flush, and the episodic writer.

`WORKERS_IN_PROCESS=false` disables **all four**, for a deployment where an
external runtime owns reactive work. The failure mode to plan for is episodic:
without a consumer, `log_activity` fills its bounded queue and then discards the
oldest turns. Set `EPISODIC_ENABLED=false` alongside it so the behaviour is
explicit rather than inferred from a full queue.

A worker that crashes is logged and **not restarted**. A crash-looping worker that
silently recovers forever is harder to diagnose than one that stops and says so,
and the restart policy belongs to whatever supervises the process:
`restart: unless-stopped`, a systemd unit, a Kubernetes probe. That is why
`/health` reports per-worker liveness: it is the input your supervisor needs.

## Startup

`create()` provisions every index the library needs, including the Atlas Search
definitions, so there is no DDL step. Two things about that are worth knowing
before the first deploy:

**Search indexes are provisioned in the background by default.** A long-running
server is fine with that, and will be serving for hours before anyone notices a
few seconds of empty search results. A short-lived script is not: the process can
exit before its indexes are queryable. Set `AWAIT_SEARCH_INDEXES=true` there.

**Startup reconciles every TTL index to the configuration, and shortening a
retention deletes data.** The rebuilt index applies to documents already stored,
so anything past the new cutoff is expired by Atlas's TTL monitor within a minute
or two of the restart, with no confirmation step. Check the retention values before
restarting a deployment whose history matters. See
[Configuration](../reference/configuration.md#memory-lifetimes).

A **failed** `create()` releases everything it acquired before raising, so a retry
against a corrected config is safe and there is no cleanup path to write.

## Things worth setting that default to something local

| Setting | Why the default is wrong for a server |
|---|---|
| `AUDIT_FALLBACK_PATH` | Relative, so it resolves to whatever the working directory happens to be: a container's `WORKDIR`, a unit's `WorkingDirectory`. An operator asked to produce an audit trail should not have to reconstruct that. `""` discards the records |
| `HOST` | `127.0.0.1`, which inside a container means the published port reaches nothing |
| `EPISODIC_RETENTION_DAYS` | 30 days is a convenience, not a policy. If you are storing turns to satisfy a retention *requirement*, set it explicitly |

## Then check it end to end

```bash
# Store, then recall, with auth on
TOKEN=k-alice
curl -s -X POST localhost:8000/memories -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice@acme.com","conversation_id":"c-1",
       "messages":[{"message_type":"human","content":"I am vegetarian and I hate cilantro"}]}'

curl -s -G localhost:8000/memories/recall -H "Authorization: Bearer $TOKEN" \
  --data-urlencode 'query=what should I cook' --data-urlencode 'user_id=alice@acme.com'
```

A `403` on the second call with a valid token means the `user_id` did not match the
token's identity. With auth on, the token decides and a mismatch is refused
rather than rewritten. A `502` means the embedding provider answered with something
that did not describe its input; nothing was written, and the same request is worth
sending again.

Long-term recall is not instant. `add` returns as soon as short-term memory is
written; importance scoring, deduplication, and promotion happen on the enrichment
worker within `ENRICHMENT_INTERVAL_SECONDS` (30). A fresh deployment that recalls
nothing a second after its first write is working correctly.

## See also

- [Configuration](../reference/configuration.md): every setting named here
- [Governance](../reference/governance.md): auth, roles, quotas, and the audit trail
- [Observability](observability.md): the counters `/health` reports
- [REST API](../reference/rest-api.md) / [MCP tools](../reference/mcp-tools.md): the surfaces being served
- `.env.example`: a commented starting point
