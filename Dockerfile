FROM python:3.11-slim

WORKDIR /app

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependency layer (cached): manifest + lockfile
COPY pyproject.toml ./
COPY uv.lock ./
RUN uv sync --frozen --no-install-project --extra all || uv sync --extra all

# Application code
COPY agent_memory/ ./agent_memory/

RUN uv pip install --system -e ".[all]"

EXPOSE 8000

# TRANSPORT=mcp|rest|both selects which shell(s) to serve.
CMD ["uv", "run", "agent-memory"]
