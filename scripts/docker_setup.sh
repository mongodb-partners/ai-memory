#!/usr/bin/env bash
# Docker setup: memory server, demo backend, and UI under one compose project.
#
# Unlike scripts/quick_setup.sh, this script exits and leaves the stack running —
# compose owns the lifecycle. Stop it with `docker compose --profile demo down`.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEED_USER="memory-demo"
DO_SEED=true

while [ $# -gt 0 ]; do
    case "$1" in
        --no-seed) DO_SEED=false; shift ;;
        --user)    SEED_USER="${2:?--user needs a value}"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
Usage: scripts/docker_setup.sh [options]

  --no-seed     Skip seeding; use the demo user as it already is.
  --user <id>   Seed this user id instead of memory-demo.
USAGE
            exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nFAILED: %s\n' "$1" >&2; exit 1; }

# ── 1. Prerequisites ─────────────────────────────────────────────────────────
step "Checking prerequisites"
command -v docker >/dev/null 2>&1 || fail \
    "docker is not installed. See https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || fail \
    "the docker compose plugin is not available. See https://docs.docker.com/compose/install/"
docker info >/dev/null 2>&1 || fail "the Docker daemon is not running. Start it and re-run."
echo "docker $(docker version --format '{{.Server.Version}}')"

# ── 2. Configuration ─────────────────────────────────────────────────────────
# Compose reads .env through `env_file`, so the same file serves both scripts.
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

# Checked with grep rather than by sourcing the file: sourcing .env executes it,
# and a value containing a backtick or $(...) would run as a command.
step "Checking .env"
grep -Eq '^[[:space:]]*MONGODB_CONNECTION_STRING=.+' .env || fail \
    "MONGODB_CONNECTION_STRING is not set in .env"
echo "  MONGODB_CONNECTION_STRING is set"
# Note: the containers validate the rest through the library's own config at
# startup. `docker compose logs demo-api` shows any refusal verbatim.

# ── 3. Build and start ───────────────────────────────────────────────────────
step "Building and starting the stack (first build takes a few minutes)"
docker compose --profile demo up --build -d || fail \
    "compose failed to start. See: docker compose --profile demo logs"

# ── 4. Wait on container health, not on ports ────────────────────────────────
# A listening port on a crash-looping container reads as up, which is exactly
# how the compose file's own startup refusal used to hide.
health_of() {  # health_of <service>
    local cid
    cid="$(docker compose ps -q "$1" 2>/dev/null)"
    [ -n "$cid" ] || { echo "missing"; return; }
    docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
        "$cid" 2>/dev/null || echo "unknown"
}

step "Waiting for services to become healthy"
attempt=0
while [ "$attempt" -lt 60 ]; do
    server="$(health_of agent-memory)"
    api="$(health_of demo-api)"
    ui="$(health_of demo-ui)"
    printf '\r    [server %-12s | demo-api %-12s | ui %-12s] (%d/60)' \
        "$server" "$api" "$ui" "$((attempt + 1))"
    if [ "$server" = healthy ] && [ "$api" = healthy ] \
       && { [ "$ui" = healthy ] || [ "$ui" = running ]; }; then
        echo; break
    fi
    for name in agent-memory demo-api demo-ui; do
        status="$(health_of "$name")"
        if [ "$status" = restarting ]; then
            echo
            fail "$name is crash-looping. Its error is in: docker compose logs $name"
        elif [ "$status" = unhealthy ]; then
            echo
            fail "$name is up but failing its healthcheck. Its error is in: docker compose logs $name"
        fi
    done
    attempt=$((attempt + 1))
    sleep 5
done

[ "$(health_of agent-memory)" = healthy ] || fail \
    "the server never became healthy. See: docker compose logs agent-memory"
[ "$(health_of demo-api)" = healthy ] || fail \
    "the demo backend never became healthy. See: docker compose logs demo-api"

# ── 5. Seed ──────────────────────────────────────────────────────────────────
# After health, never before: ConsolidationWorker runs a pass at startup and
# would promote the seeded candidates out of existence.
if [ "$DO_SEED" = true ]; then
    step "Seeding ${SEED_USER} (this WIPES that user's existing memories)"
    docker compose exec -T demo-api python -m demo.seed --user "$SEED_USER" || fail \
        "seeding reported a problem above. Retry with:
  docker compose exec -T demo-api python -m demo.seed --user $SEED_USER"
fi

# ── 6. Done ──────────────────────────────────────────────────────────────────
UI_URL="http://localhost:5173"
case "$OSTYPE" in
    darwin*) open "$UI_URL" >/dev/null 2>&1 || true ;;
    linux*)  xdg-open "$UI_URL" >/dev/null 2>&1 || true ;;
esac

cat <<MSG

==> Running in the background
    UI              $UI_URL
    Demo backend    http://localhost:8100/health
    Memory server   http://localhost:8000/health
    Demo user       ${SEED_USER}

    Logs            docker compose --profile demo logs -f
    Stop            docker compose --profile demo down
    Re-seed         docker compose exec -T demo-api python -m demo.seed --user ${SEED_USER}

Every port is published to 127.0.0.1 only. Widening that needs AUTH_ENABLED=true.
MSG
