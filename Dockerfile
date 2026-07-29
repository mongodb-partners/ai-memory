FROM python:3.11-slim

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
