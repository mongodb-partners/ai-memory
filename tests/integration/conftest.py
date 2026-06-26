"""Pytest configuration for integration tests.

The integration suite in this package exercises the MCP tools end-to-end
against a running server (default: ``localhost:8000``). When no server is
reachable, the tests are skipped rather than failed, so ``pytest`` stays
green for the unit suite. To run them, start the server (see
``docs/deployment.md``) and re-run pytest.
"""

import http.client
import os

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


def pytest_collection_modifyitems(config, items):
    """Skip integration tests when no MCP server is reachable."""
    if _server_reachable():
        return
    skip = pytest.mark.skip(
        reason=f"no MCP server reachable on {MCP_HOST}:{MCP_PORT}"
    )
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)
