"""Dual-transport entrypoint. REQ-E-080.

TRANSPORT=both serves MCP + REST from one shared AsyncMemory; mcp/rest serve a
single shell. We assert which shells are built and that one facade is shared,
without binding real sockets.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.shells.runner import build_shells


def _config(transport: str) -> MemoryConfig:
    return MemoryConfig(
        mongodb_connection_string="mongodb://localhost:27017",
        transport=transport, _env_file=None,
    )


@pytest.fixture
def _patched(monkeypatch):
    import agent_memory.shells.runner as runner

    facade = MagicMock()
    create = AsyncMock(return_value=facade)
    monkeypatch.setattr(runner.AsyncMemory, "create", create)
    monkeypatch.setattr(runner, "create_mcp", lambda config, app: ("mcp", app))
    monkeypatch.setattr(runner, "create_app", lambda app: ("rest", app))
    return runner, facade, create


class TestBuildShells:
    async def test_both_shares_one_facade(self, _patched):
        runner, facade, create = _patched
        shells = await runner.build_shells(_config("both"))
        create.assert_awaited_once()
        assert set(shells) == {"mcp", "rest"}
        # both shells bound to the same facade instance
        assert shells["mcp"][1] is facade
        assert shells["rest"][1] is facade

    async def test_mcp_only(self, _patched):
        runner, facade, create = _patched
        shells = await runner.build_shells(_config("mcp"))
        assert set(shells) == {"mcp"}

    async def test_rest_only(self, _patched):
        runner, facade, create = _patched
        shells = await runner.build_shells(_config("rest"))
        assert set(shells) == {"rest"}
