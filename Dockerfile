FROM python:3.11-slim AS server

WORKDIR /app

# Install uv for fast, reproducible installs. Pinned by digest, not `:latest`:
# a mutable tag can be repointed by the publisher, which makes the build both
# irreproducible and a supply-chain surface. This digest is what `:latest`
# resolved to on 2026-07-29; it does not correspond to any semver tag, so the
# digest is the only stable identifier. Bump it deliberately.
COPY --from=ghcr.io/astral-sh/uv@sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105 /uv /usr/local/bin/uv

# Dependency layer (cached): manifest + lockfile. README.md and LICENSE are
# required here, not optional niceties — pyproject declares `readme` and
# `license-files`, and hatchling fails the build outright if either is missing.
COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./
COPY LICENSE ./
RUN uv sync --frozen --no-install-project --extra all || uv sync --extra all

# Application code
COPY agent_memory/ ./agent_memory/

RUN uv pip install --system -e ".[all]"

# Drop root. Nothing here needs it: the server binds 8000, not a privileged
# port, and every write the process makes goes to Atlas rather than the
# filesystem. `uv run` resolves /app/.venv, so the venv has to be writable by
# the runtime user — hence the chown before the USER switch.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# TRANSPORT=mcp|rest|both selects which shell(s) to serve.
CMD ["uv", "run", "agent-memory"]

# ── Demo stage: the sample UI's backend ──────────────────────────────────────
# A separate target, not a wider default. `demo` is kept out of the `all` extra
# on purpose (pyproject.toml:64) — the library never imports sse-starlette, and a
# demo dependency has no business in a production install. Build with:
#   docker build --target demo -t agent-memory-demo:latest .
FROM server AS demo

USER root
RUN uv pip install --system -e ".[all,demo]"

# The demo backend imports `server.*` and `demo.*` as top-level packages, so its
# own directory is the workdir rather than /app.
COPY examples/memory-ui/server/ ./examples/memory-ui/server/
COPY examples/memory-ui/demo/ ./examples/memory-ui/demo/
RUN chown -R appuser:appuser /app/examples
USER appuser

WORKDIR /app/examples/memory-ui
EXPOSE 8100

# `python -m uvicorn`, not the `uvicorn` console script. A console script's
# shebang is absolute and is not rewritten when a venv is copied, so a stale one
# re-execs a different interpreter whose site-packages provide a different
# agent_memory. The symptom is ModuleNotFoundError for a submodule visible on
# disk. `-m` resolves through the interpreter already running.
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8100"]
