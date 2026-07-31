# Sample UI: memory ON vs memory OFF

A one-screen demo of `agent-memory`: chat on the left, the memories that produced
each answer on the right, and a switch that turns the whole memory layer off.

The switch is the point. Same model, same prompt, same question. Flip it and the
answer changes, because the second time the agent had somewhere to look.

## What's on screen

| Panel | Shows |
|---|---|
| **Semantic cache** | HIT / MISS, and which path matched. Runs first, so a hit means the tiers below were never queried. |
| **Short-term** | This thread's state, TTL-expired. |
| **Long-term** | Durable facts with importance and access counts. |
| **Episodic** | What the agent *did*: step number, tools, files touched. |

Every recalled row carries its rank and raw fused score. That is deliberate: a
memory panel that shows text without scores is indistinguishable from a
hardcoded list, and the claim being made here is that the ranking is real.

## Run it

`scripts/quick_setup.sh` from the repository root does everything below —
dependency install, both processes, seeding, and opening the browser.
`scripts/docker_setup.sh` does the same under Docker. The manual steps follow,
because knowing them is what lets you debug the scripts.

Two processes. From the repository root:

```bash
# 1. Backend. Needs Atlas, an embedding provider, and an LLM provider.
uv run --extra demo python -m uvicorn server.app:app --port 8100 --app-dir examples/memory-ui

# 2. Frontend, in a second shell.
cd examples/memory-ui/frontend && npm install && npm run dev
```

`python -m uvicorn`, not `uvicorn`, and the difference is not cosmetic. A console
script in `.venv/bin/` selects its interpreter through a hardcoded shebang, which
is absolute and is not rewritten if the virtualenv is ever copied, moved, or
recreated from another checkout. When it goes stale, `uv run` resolves the correct
*script* and that script then re-execs a **different interpreter**, whose
`site-packages` provides a different `agent_memory`. The symptom is a
`ModuleNotFoundError` for a submodule you can see on disk, which reads as a
library bug and is not one. `-m` resolves the module through the interpreter
already running, so no shebang is consulted and the failure cannot occur.

Then open http://localhost:5173.

### Optional: seed a user, and mind the order

The UI works against an empty user: the memory-ON pass fills the panel as you go.
But the OFF-vs-ON contrast is sharper when the agent already knows things, so
`demo/seed.py` plants a deterministic set spanning all four tiers:

```bash
cd examples/memory-ui && uv run --extra demo python -m demo.seed --user memory-demo
```

**Seed after the server is up, not before.** `ConsolidationWorker` runs a
consolidation pass at startup rather than after its first interval, so a server
starting up behind a fresh seed immediately promotes every eligible short-term
memory. The seeded promotion candidates vanish, the STM/LTM split shifts, and the
promotion pipeline in `demo/compass-pipelines/` returns nothing. Nothing errors.
The state just quietly stops matching what the seed reported.

Expect `stm 12 · ltm 6 · episodes 3 · candidates 5`. If `candidates` is 0, the
order was wrong; re-seed and leave the server running.

Leave a second user id unseeded. Typing it into the header is how you show that
per-user isolation is enforced inside the query rather than in the prompt, and
that only works if the second user has nothing to recall.

Unseeded is not the same as empty, though. Asking that user a question with memory
ON *stores the question*, so a second run can recall the first run's own words and
the point evaporates. Clear it without planting anything:

```bash
uv run --extra demo python -m demo.seed --user alex --wipe-only
```

Configuration comes from the repository root's `.env` (see `.env.example`), which
`server/app.py` loads explicitly. The library itself never reads a `.env`. That
is correct for a library, and it is why the server does it.

### Preflight: run this before you present

```bash
# The import that matters, from the venv that will actually serve the demo.
uv run --extra demo python -c "
import agent_memory, sys
from agent_memory.core.correlation import derive_correlation_id
print(sys.prefix); print(agent_memory.__file__)"
```

Both paths must be inside *this* checkout. If either points somewhere else, or the
import raises `ModuleNotFoundError`, the environment is stale. Recreate it with
`uv sync --extra demo` and run the check again. Worth the thirty seconds: this
failure surfaces at server startup as a missing-submodule traceback that looks
like a library bug, and it surfaces at the moment you have an audience.

To rule out the shebang problem across the whole environment at once:

```bash
grep -rl "^#!/" .venv/bin/ 2>/dev/null \
  | xargs grep -l "^#!" | xargs head -1 \
  | grep "^#!" | grep -v "^#!$PWD/.venv/bin/" | sort -u
```

Any line printed is a script bound to a foreign interpreter. `uv sync --extra demo`
regenerates them all against the current path.

Then confirm the server agrees with what you are about to claim on a slide:

```bash
curl -s localhost:8100/health | python -m json.tool
```

`llm_model`, `embedding_model`, `embedding_dimension`, and
`episodic.worker_alive` all come from the live config. The UI header renders the
same values from `/config` for the same reason: an audience notices when the
slide and the screen disagree.

## The demo script

Four prompts, in order. The two questions must match character-for-character
between the OFF and ON passes or the comparison is not a comparison; the preset
buttons in the UI carry them verbatim so you never have to retype them on stage.

1. **Memory OFF.** `I'm allergic to shellfish, and I'm cooking for six people on Friday.`
2. **Memory OFF, new thread.** `What should I make Friday?` → it has to ask again.
3. **Memory ON**, repeat 1, then **new thread**, repeat 2 → it answers. Never
   re-asks. The panel shows which documents it drew on, with scores.
4. **Memory ON**, ask 2 a third time → cache **HIT**, sub-second, no model call.
5. **New thread.** `What have we worked on together so far?` → episodic recall.
   Nothing in this thread; everything from the activity log.

Step 4 works because the cache tries an exact-match lookup on a B-tree index
before the vector search. Atlas Search indexes are eventually consistent, so a
response cached at the end of one turn is typically not `$vectorSearch`-queryable
for a few seconds, and this demo asks the repeat question immediately. The fast
path is not a demo cheat; a production semantic cache wants it for the same
reason an exact repeat should not pay for an embedding round-trip.

## Compass

`demo/compass-pipelines/` holds the saved aggregations behind the panel. Opening
one is the "why it works" screen: the hybrid recall is `$rankFusion` over a
`$vectorSearch` branch and a `$search` branch, in one pipeline, in one database.

Two filters worth having ready, both against `memories`:
`{ user_id: "memory-demo", tier: "stm" }` and
`{ user_id: "memory-demo", tier: "ltm", enrichment_status: "complete" }`. Put them
side by side and the tier distinction is one field: both documents carry
`expires_at`, roughly 24 hours out on the first and 90 days on the second, from
the same TTL index and a different `retention_tier`. The long-term document also
carries `source_stm_id` pointing back at the short-term one it was promoted from,
so the provenance chain is on screen rather than asserted.

A note on reading those scores live: `$rankFusion` returns a reciprocal-rank sum,
so with the default `k` of 60 a **first-place** document scores about 1/61 ≈
`0.016`. On a projector that reads as a failed match. The panel therefore leads
with `#1`, `#2` and keeps the raw value beside it, with no rescaled "relevance
percentage", because that number would be invented rather than measured.

## Layout

```
server/          FastAPI + SSE. No agent framework; the loop in turn.py is the agent.
  app.py         Routes: /chat (SSE), /memories, /reset, /health, /config
  turn.py        recall → prompt → stream → write back
  cache_key.py   Semantic cache keyed on user_id AND memory_enabled
  prompt.py      Provider-shaped message construction (Bedrock/Anthropic/OpenAI)
  history.py     In-process thread history; two turns, deliberately
  sse.py         Framing, timeout, graceful drain
frontend/        React 18 + Vite. Fonts self-hosted; zero remote asset URLs.
demo/            Seed script and Compass pipelines
```

`memory_enabled: false` skips recall **and** bypasses the cache entirely. Both
halves matter: recall is the visible difference, but a cache that ignored the flag
would replay a memory-informed answer during the memory-off pass, and the demo's
central claim would be false on stage. The cache also carries `memory_enabled` as
a declared filter field in its vector index, so the two modes cannot cross even
if the bypass were removed.

## Deploying behind a proxy

SSE needs response buffering off, or tokens arrive in one batch at the end and
the stream looks broken:

```nginx
location /api/ {
    proxy_pass http://backend;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;
    chunked_transfer_encoding on;
}
```

Vite's dev proxy has the same problem; `vite.config.ts` handles it by injecting
`cache-control: no-transform`.

CORS on the demo server is locked to localhost origins. It holds Atlas
credentials, and a wide-open policy on a conference network is not something to
demonstrate.
