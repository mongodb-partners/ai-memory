"""Tests for the MCP shell over the AsyncMemory facade. REQ-E-060..063."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.exceptions import AccessError, RateLimitError
from agent_memory.shells.mcp.tools import register_all_tools
from agent_memory.shells.mcp.auto_capture import AutoCaptureMiddleware


def _capture_tool(mcp_mock):
    tools = {}
    mcp_mock.tool = lambda **kwargs: lambda fn: tools.update({kwargs["name"]: fn}) or fn
    return tools


def _app():
    app = MagicMock()
    app.add = AsyncMock(return_value={"stm_ids": ["a"], "count": 1})
    app.recall = AsyncMock(return_value={"results": [], "count": 0})
    app.search = AsyncMock(return_value={"results": [], "count": 0})
    app.delete = AsyncMock(return_value={"deleted_count": 0})
    app.health = AsyncMock(return_value={"total_memories": 0})
    return app


class TestDelegation:
    """TC-MCP-001: each tool delegates to the matching facade method."""

    async def test_store_memory_delegates_to_add(self):
        app = _app()
        mcp = MagicMock()
        tools = _capture_tool(mcp)
        register_all_tools(mcp, app)
        out = await tools["store_memory"]("u1", "c1", [{"content": "hi"}])
        app.add.assert_awaited_once()
        assert out["count"] == 1

    async def test_recall_delegates(self):
        app = _app()
        mcp = MagicMock()
        tools = _capture_tool(mcp)
        register_all_tools(mcp, app)
        await tools["recall_memory"]("u1", "q")
        app.recall.assert_awaited_once()


class TestErrorTranslation:
    """TC-MCP-002: AccessError / RateLimitError → {"error": ...}."""

    async def test_access_error_becomes_error_dict(self):
        app = _app()
        app.recall = AsyncMock(side_effect=AccessError("denied"))
        mcp = MagicMock()
        tools = _capture_tool(mcp)
        register_all_tools(mcp, app)
        out = await tools["recall_memory"]("u1", "q")
        assert out == {"error": "denied"}

    async def test_rate_limit_error_becomes_error_dict(self):
        app = _app()
        app.add = AsyncMock(side_effect=RateLimitError("slow down"))
        mcp = MagicMock()
        tools = _capture_tool(mcp)
        register_all_tools(mcp, app)
        out = await tools["store_memory"]("u1", "c1", [{"content": "x"}])
        assert out == {"error": "slow down"}


class TestAutoCapture:
    """TC-MCP-003: auto-capture persists via app.add, not a service."""

    async def test_capture_calls_app_add(self):
        app = _app()
        config = MagicMock()
        config.auto_capture_enabled = True
        config.auto_capture_tools = ["recall_memory"]
        config.auto_capture_min_length = 1
        config.auto_capture_max_content_length = 2000
        mw = AutoCaptureMiddleware(app, config)
        await mw.capture("recall_memory", {"user_id": "u1", "query": "q"}, {"result": "r"})
        app.add.assert_awaited_once()
        # conversation_id marks it as auto-captured
        assert app.add.call_args.args[1].startswith("auto:")

    async def test_excluded_tool_not_captured(self):
        app = _app()
        config = MagicMock()
        config.auto_capture_enabled = True
        config.auto_capture_tools = ["store_memory"]
        config.auto_capture_min_length = 1
        config.auto_capture_max_content_length = 2000
        mw = AutoCaptureMiddleware(app, config)
        await mw.capture("store_memory", {"user_id": "u1"}, {"result": "r"})
        app.add.assert_not_awaited()


class TestLifespan:
    """TC-MCP-LIFE-001: lifespan creates and closes its own AsyncMemory."""

    async def test_lifespan_creates_and_closes_facade(self, monkeypatch):
        from agent_memory.config import MemoryConfig
        import agent_memory.shells.mcp.server as server

        instance = MagicMock()
        instance.close = AsyncMock()
        create = AsyncMock(return_value=instance)
        monkeypatch.setattr(server.AsyncMemory, "create", create)
        monkeypatch.setattr(server, "register_all_tools", lambda m, a: None)

        cfg = MemoryConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            auto_capture_enabled=False, _env_file=None,
        )
        lifespan = server.make_lifespan(cfg)
        async with lifespan(MagicMock()):
            create.assert_awaited_once()
        instance.close.assert_awaited_once()

    async def test_lifespan_reuses_shared_app_without_closing(self, monkeypatch):
        from agent_memory.config import MemoryConfig
        import agent_memory.shells.mcp.server as server

        shared = MagicMock()
        shared.close = AsyncMock()
        create = AsyncMock()
        monkeypatch.setattr(server.AsyncMemory, "create", create)
        monkeypatch.setattr(server, "register_all_tools", lambda m, a: None)

        cfg = MemoryConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            auto_capture_enabled=False, _env_file=None,
        )
        lifespan = server.make_lifespan(cfg, app=shared)
        async with lifespan(MagicMock()):
            create.assert_not_awaited()  # shared app reused
        shared.close.assert_not_awaited()  # caller owns shared app lifecycle
