#!/usr/bin/env bash
# Local setup: the sample UI, running against Atlas, in one command.
#
# For the Docker equivalent see scripts/docker_setup.sh. This script leaves three
# processes in the foreground and shuts them down on Ctrl-C.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEED_USER="memory-demo"
DO_SEED=true
WITH_SERVER=false

while [ $# -gt 0 ]; do
    case "$1" in
        --no-seed)      DO_SEED=false; shift ;;
        --with-server)  WITH_SERVER=true; shift ;;
        --user)         SEED_USER="${2:?--user needs a value}"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
Usage: scripts/quick_setup.sh [options]

  --no-seed        Skip seeding; use the demo user as it already is.
  --user <id>      Seed this user id instead of memory-demo.
  --with-server    Also start the MCP/REST server on 8000. The UI does not use
                   it — the demo backend embeds the library in-process — so this
                   is only for exercising the server's own surface.
USAGE
            exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nFAILED: %s\n' "$1" >&2; exit 1; }

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
step "Checking prerequisites"
command -v uv >/dev/null 2>&1 || fail \
    "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/"
command -v npm >/dev/null 2>&1 || fail \
    "npm is not installed. The UI needs Node 20+. See https://nodejs.org/"
echo "uv $(uv --version | awk '{print $2}'), node $(node --version)"

# ── 2. Configuration ─────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    step "Creating .env from .env.example"
    cp .env.example .env
    cat <<'MSG'

.env created. Fill in at least:

  MONGODB_CONNECTION_STRING   your Atlas connection string
  EMBEDDING_PROVIDER          and that provider's credentials
  LLM_PROVIDER                and that provider's credentials

Then re-run this script.
MSG
    exit 1
fi

# ── 3. Preflight ─────────────────────────────────────────────────────────────
# Validated by loading the real config rather than by parsing .env in bash.
# `export $(cat .env | xargs)` breaks on any value containing a space or a `#`
# and exports every credential to every child process. This also catches a stale
# virtualenv: a venv rebuilt from another checkout fails the import.
step "Validating configuration"
uv run --extra demo python - <<'PY' || fail \
    "configuration is not usable. Fix .env, or rebuild the environment with \`uv sync --extra demo\`."
import sys
from agent_memory.config import MemoryConfig
from agent_memory.providers.manager import resolve_embedding

config = MemoryConfig.from_env()
if not config.mongodb_connection_string:
    sys.exit("MONGODB_CONNECTION_STRING is not set in .env")

# Set is not the same as filled in. Every line of .env.example is a syntactically
# valid non-empty value, so the "we wrote the template, fill it in, re-run"
# handshake earlier in this script did not actually gate the second run: the
# placeholders reached this preflight, passed it, and the failure surfaced much
# later as an auth error from Atlas or AWS. These literals come from
# .env.example; no real credential resembles them.
placeholders = [
    ("MONGODB_CONNECTION_STRING", "your Atlas connection string",
     "user:password@cluster.mongodb.net", config.mongodb_connection_string),
    ("AWS_ACCESS_KEY_ID", "your AWS access key id",
     "your-access-key", config.aws_access_key_id),
    ("AWS_SECRET_ACCESS_KEY", "your AWS secret access key",
     "your-secret-key", config.aws_secret_access_key),
]
unedited = [(name, hint) for name, hint, literal, value in placeholders
            if literal in (value or "")]
if unedited:
    sys.exit(
        ".env still holds the placeholder values from .env.example, so this is\n"
        "the unedited template. Replace:\n"
        + "\n".join(f"    {name:<27} {hint}" for name, hint in unedited)
        + "\n  then re-run this script."
    )

needs = {
    "voyage": ("voyage_api_key", "VOYAGE_API_KEY"),
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
}
for provider_field in ("embedding_provider", "llm_provider"):
    provider = (getattr(config, provider_field) or "").lower()
    if provider in needs:
        attr, env_name = needs[provider]
        if not getattr(config, attr, None):
            sys.exit(f"{provider_field.upper()}={provider} but {env_name} is not set")
    elif provider == "bedrock" and not config.aws_access_key_id:
        print(f"  note: {provider_field}=bedrock with no AWS_ACCESS_KEY_ID; "
              f"relying on the ambient AWS credential chain")

# `resolve_embedding`, not the raw fields. On a Voyage deployment the canonical
# model and width come from `voyage_model` and are reconciled at startup, so
# `config.embedding_model` still reads `amazon.titan-embed-text-v1` at 1536d
# while what actually gets used is voyage-4 at 1024d. Printing the raw pair
# tells the operator something false about their own setup.
embedding = resolve_embedding(config)
print(f"  database    {config.mongodb_database_name}")
print(f"  embedding   {config.embedding_provider}/{embedding.model} "
      f"({embedding.dimension}d)")
print(f"  llm         {config.llm_provider}/{config.llm_model}")
PY

# ── 4. Dependencies ──────────────────────────────────────────────────────────
step "Installing dependencies"
uv sync --frozen --extra demo || uv sync --extra demo

# ── 5. Port preflight ────────────────────────────────────────────────────────
# Before anything starts. `wait_for` probes a URL, not a process it owns, so a
# leftover listener from an earlier run satisfies it instantly while the process
# this run spawned dies unobserved on EADDRINUSE. The operator then drives the
# *previous* run's stack, and a --user change appears not to take.
step "Checking that the ports are free"
port_is_busy() {  # port_is_busy <port>
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}
if command -v lsof >/dev/null 2>&1; then
    PORTS="8100 5173"
    [ "$WITH_SERVER" = true ] && PORTS="8000 $PORTS"
    for port in $PORTS; do
        if port_is_busy "$port"; then
            fail "port $port is already in use, and this script needs it. Something from
  an earlier run may still be up. See what holds it with:
  lsof -nP -iTCP:$port -sTCP:LISTEN"
        fi
    done
    echo "  ${PORTS// /, } free"
else
    # Skipped rather than fatal: lsof ships with macOS and most Linux distros,
    # but a slim container image may not have it, and that is not a reason to
    # refuse to run.
    echo "  lsof not found; skipping the port check"
fi

# ── 6. Processes ─────────────────────────────────────────────────────────────
PIDS=()
# Parallel to PIDS, so the watch loop at the end can name what died rather than
# printing a bare pid. Two indexed arrays, not one associative array: bash 3.2.
LABELS=()
CLEANED_UP=false
# On EXIT as well as the signals: `fail` ends in `exit 1`, so without EXIT every
# failure after the first `&` left that child running, reparented to init, still
# holding its port — and the next run then found a stale listener.
cleanup() {
    # First line, before anything can overwrite it: the status that triggered
    # the EXIT trap is what this script must return, so that
    # `scripts/quick_setup.sh && echo "demo is up"` cannot print on a failure.
    local status=$?
    # `exit` below re-enters this trap. The second pass must do nothing.
    if [ "$CLEANED_UP" = true ]; then return 0; fi
    CLEANED_UP=true
    printf '\n==> Shutting down\n'
    local pid
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    # Drain before exiting: uvicorn holds 8100 until its own shutdown completes,
    # so returning the prompt immediately would hand the next run a stale
    # listener. Plain TERM above, not SIGKILL, because TERM is what `uv run`
    # forwards to the interpreter underneath it — a KILL would orphan that
    # grandchild instead of stopping it. `|| true` so an interrupted `wait` does
    # not trip `set -e` and skip the exit below.
    wait 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT
# The signal traps only set the conventional status and exit; EXIT then runs
# cleanup once. Calling cleanup directly from a signal trap instead leaves the
# `wait` below hanging until killed, on both bash 3.2 and 5.3.
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for() {  # wait_for <url> <label>
    local url="$1" label="$2" attempt=0
    printf '    waiting for %s ' "$label"
    while [ "$attempt" -lt 60 ]; do
        if curl -sf "$url" >/dev/null 2>&1; then echo " ready"; return 0; fi
        printf '.'
        attempt=$((attempt + 1))
        sleep 2
    done
    echo " timed out"
    return 1
}

if [ "$WITH_SERVER" = true ]; then
    step "Starting the memory server on 8000"
    uv run agent-memory &
    PIDS+=($!); LABELS+=("the memory server on 8000")
    wait_for http://localhost:8000/health "server" || fail \
        "the server did not become healthy. Run \`uv run agent-memory\` to see why."
fi

# `python -m uvicorn`, not the console script: a stale absolute shebang re-execs
# a different interpreter and fails with ModuleNotFoundError for a submodule that
# is visibly on disk. See examples/memory-ui/README.md.
step "Starting the demo backend on 8100"
uv run --extra demo python -m uvicorn server.app:app \
    --port 8100 --app-dir examples/memory-ui &
PIDS+=($!); LABELS+=("the demo backend on 8100")
wait_for http://localhost:8100/health "demo backend" || fail \
    "the demo backend did not become healthy. Start it alone to see the error:
  uv run --extra demo python -m uvicorn server.app:app --port 8100 --app-dir examples/memory-ui"

# ── 7. Seed ──────────────────────────────────────────────────────────────────
# After the backend is up, never before: ConsolidationWorker runs a pass at
# startup, so a server coming up behind a fresh seed promotes every eligible
# short-term memory and the promotion candidates disappear.
if [ "$DO_SEED" = true ]; then
    step "Seeding ${SEED_USER} (this WIPES that user's existing memories)"
    # seed.py validates its own work and exits non-zero with a specific
    # diagnosis, so the script surfaces that rather than re-deriving thresholds.
    ( cd examples/memory-ui && uv run --extra demo python -m demo.seed --user "$SEED_USER" ) || fail \
        "seeding reported a problem above. Retry with:
  cd examples/memory-ui && uv run --extra demo python -m demo.seed --user $SEED_USER"
fi

# ── 8. Frontend ──────────────────────────────────────────────────────────────
step "Installing frontend dependencies"
( cd examples/memory-ui/frontend && npm install )

step "Starting the frontend on 5173"
# `--host 127.0.0.1` explicitly, because vite.config.ts sets `host: true`, which
# binds every interface. That config also proxies /api to the demo backend, and
# that backend has no auth — so the default would publish every route of an
# unauthenticated memory store to the LAN. README.md also promises that this
# setup binds loopback only. The flag overrides the config; the config is left
# alone because it predates this script and `npm run dev` by hand is its own case.
( cd examples/memory-ui/frontend && npm run dev -- --host 127.0.0.1 ) &
PIDS+=($!); LABELS+=("the frontend on 5173")
wait_for http://localhost:5173/ "frontend" || fail \
    "the frontend did not come up. Start it alone to see the error:
  cd examples/memory-ui/frontend && npm run dev"

# ── 9. Done ──────────────────────────────────────────────────────────────────
UI_URL="http://localhost:5173"
case "$OSTYPE" in
    darwin*) open "$UI_URL" >/dev/null 2>&1 || true ;;
    linux*)  xdg-open "$UI_URL" >/dev/null 2>&1 || true ;;
esac

cat <<MSG

==> Running
    UI              $UI_URL
    Demo backend    http://localhost:8100/health
$([ "$WITH_SERVER" = true ] && echo "    Memory server   http://localhost:8000/health")
    Demo user       ${SEED_USER}

Ctrl-C stops everything.
MSG

# Not a bare `wait`: that returns 0 unconditionally and only after *every* child
# has finished, so a backend that crashes a minute from now would leave this
# terminal reading "Running" over a UI whose every request fails, and say nothing.
# Not `wait -n` either — that is bash 4.3+ and this runs on macOS's bash 3.2.
# So poll the PIDs we recorded, and name whichever one goes away first. The
# non-zero exit fires the EXIT trap, which tears down the rest.
while :; do
    for i in $(seq 0 $((${#PIDS[@]} - 1))); do
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            fail "${LABELS[$i]} (pid ${PIDS[$i]}) exited. The rest of the stack is
being shut down. Its last output is above; start it alone to see the error."
        fi
    done
    sleep 2
done
