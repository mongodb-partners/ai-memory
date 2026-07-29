"""Pytest configuration for integration tests.

The integration suite in this package exercises the MCP tools end-to-end
against a running server (default: ``localhost:8000``). When no server is
reachable, the tests are skipped rather than failed, so ``pytest`` stays
green for the unit suite. To run them, start the server (see
``docs/deployment.md``) and re-run pytest.
"""

import http.client
import os
from pathlib import Path

import pytest

MCP_HOST = os.environ.get("MCP_HOST", "localhost")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))


def _server_reachable() -> bool:
    """Return True if the MCP server answers a /health probe."""
    try:
        conn = http.client.HTTPConnection(MCP_HOST, MCP_PORT, timeout=1)
        conn.request("GET", "/health")
        conn.getresponse()
        conn.close()
        return True
    except OSError:
        return False


def _atlas_configured() -> bool:
    """True if a connection string is reachable, from the env or a local .env.

    ``MemoryConfig`` is a pydantic-settings model and reads ``.env`` itself, so
    gating on the environment alone would skip these tests on a developer machine
    that is in fact configured — the gate would be stricter than the code it
    guards. Only the key's presence is checked; the value is never read here.
    """
    if os.environ.get("MONGODB_CONNECTION_STRING"):
        return True
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return False
    try:
        for line in env_file.read_text().splitlines():
            key = line.split("=", 1)[0].strip()
            if key.upper() == "MONGODB_CONNECTION_STRING":
                return True
    except OSError:
        return False
    return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_atlas: needs a real Atlas cluster via the library, not a server",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests whose dependency is absent.

    Two gates, because the two kinds of test need different things. Most tests
    here drive the MCP server over HTTP and gate on reachability. Tests marked
    ``live_atlas`` drive the library directly against a real cluster — no server
    involved — so they gate on a connection string instead. Gating those on
    server reachability would skip them on a machine that can run them.
    """
    server_up = _server_reachable()
    has_atlas = _atlas_configured()

    skip_server = pytest.mark.skip(
        reason=f"no MCP server reachable on {MCP_HOST}:{MCP_PORT}"
    )
    skip_atlas = pytest.mark.skip(
        reason="MONGODB_CONNECTION_STRING is set neither in the environment nor .env"
    )

    for item in items:
        if "integration" not in str(item.fspath):
            continue
        if item.get_closest_marker("live_atlas"):
            if not has_atlas:
                item.add_marker(skip_atlas)
        elif not server_up:
            item.add_marker(skip_server)
