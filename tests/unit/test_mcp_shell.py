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

    async def test_all_remaining_tools_delegate(self):
        app = _app()
        app.search = AsyncMock(return_value={"results": [], "count": 0})
        app.delete = AsyncMock(return_value={"deleted_count": 0})
        app.check_cache = AsyncMock(return_value={"cache_hit": True})
        app.store_cache = AsyncMock(return_value="cid")
        app.invalidate_cache = AsyncMock(return_value={"deleted_count": 1})
        app.remember_decision = AsyncMock(return_value={"key": "k", "status": "stored"})
        app.recall_decision = AsyncMock(return_value={"key": "k", "value": "v"})
        app.wipe_user_data = AsyncMock(return_value={"memories_deleted": 0})
        mcp = MagicMock()
        tools = _capture_tool(mcp)
        register_all_tools(mcp, app)

        await tools["hybrid_search"]("u1", "q")
        await tools["delete_memory"]("u1", memory_id="m1")
        assert (await tools["check_cache"]("u1", "q"))["cache_hit"] is True
        assert (await tools["store_cache"]("u1", "q", "r"))["cache_id"] == "cid"
        await tools["cache_invalidate"]("u1", invalidate_all=True)
        await tools["store_decision"]("u1", "k", "v")
        assert (await tools["recall_decision"]("u1", "k"))["value"] == "v"
        assert (await tools["memory_health"]("u1"))["total_memories"] == 0
        await tools["wipe_user_data"]("u1", confirm=True)

        app.search.assert_awaited_once()
        app.delete.assert_awaited_once()
        app.wipe_user_data.assert_awaited_once()

    async def test_check_cache_miss_returns_false(self):
        app = _app()
        app.check_cache = AsyncMock(return_value=None)
        mcp = MagicMock()
        tools = _capture_tool(mcp)
        register_all_tools(mcp, app)
        assert (await tools["check_cache"]("u1", "q"))["cache_hit"] is False

    async def test_recall_decision_miss_returns_null_value(self):
        app = _app()
        app.recall_decision = AsyncMock(return_value=None)
        mcp = MagicMock()
        tools = _capture_tool(mcp)
        register_all_tools(mcp, app)
        assert (await tools["recall_decision"]("u1", "k"))["value"] is None


class TestEpisodicTools:
    """TC-MCP-EP-001..004: the five episodic tools over the facade."""

    def _wired(self):
        app = _app()
        app.log_activity = AsyncMock(return_value={"enqueued": True, "thread_id": "t1"})
        app.recall_activity = AsyncMock(return_value={"results": [], "count": 0})
        app.get_thread = AsyncMock(return_value={"results": [], "count": 0})
        app.get_activity_by_correlation = AsyncMock(return_value={"results": [], "count": 0})
        app.set_activity_retention = AsyncMock(return_value={"ttl_seconds": 7200})
        mcp = MagicMock()
        tools = _capture_tool(mcp)
        register_all_tools(mcp, app)
        return app, tools

    async def test_all_five_episodic_tools_delegate(self):
        app, tools = self._wired()
        assert (await tools["log_activity"]("u1", "t1", [{"type": "human"}]))["enqueued"] is True
        await tools["search_activity"]("u1", "q", thread_id="t1")
        await tools["get_thread"]("u1", "t1")
        await tools["get_correlation"]("u1", "corr-1")
        assert (await tools["set_activity_retention"]("u1", 7200))["ttl_seconds"] == 7200

        app.log_activity.assert_awaited_once()
        app.recall_activity.assert_awaited_once()
        app.get_thread.assert_awaited_once()
        app.get_activity_by_correlation.assert_awaited_once()
        app.set_activity_retention.assert_awaited_once()

    async def test_log_activity_forwards_trace_and_tenant_fields(self):
        app, tools = self._wired()
        await tools["log_activity"](
            "u1", "t1", [{"type": "ai"}], todos=[{"id": "1"}], agent_name="planner",
            correlation_id="corr", conversation_id="c1",
        )
        kwargs = app.log_activity.call_args.kwargs
        assert kwargs["correlation_id"] == "corr"
        assert kwargs["conversation_id"] == "c1"
        assert kwargs["agent_name"] == "planner"

    async def test_retention_accepts_none_as_keep_forever(self):
        """None is a value here, not a missing argument."""
        app, tools = self._wired()
        await tools["set_activity_retention"]("u1", None)
        assert app.set_activity_retention.call_args.kwargs["ttl_seconds"] is None

    async def test_episodic_denial_becomes_an_error_dict(self):
        app, tools = self._wired()
        app.recall_activity = AsyncMock(side_effect=AccessError("denied"))
        assert await tools["search_activity"]("u1", "q") == {"error": "denied"}


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

    @pytest.mark.parametrize("tool", ["log_activity", "set_activity_retention"])
    async def test_episodic_write_tools_are_never_auto_captured(self, tool):
        """The episodic tier is already the record of what the agent did.

        Capturing a turn-log write would store a memory *about* the log, so the
        two tiers would feed each other — pure amplification, and it is the
        failure mode most likely to go unnoticed because everything still works.
        """
        app = _app()
        config = MagicMock()
        config.auto_capture_enabled = True
        config.auto_capture_tools = [tool]  # explicitly opted in — exclusion still wins
        config.auto_capture_min_length = 1
        config.auto_capture_max_content_length = 2000
        mw = AutoCaptureMiddleware(app, config)
        await mw.capture(tool, {"user_id": "u1", "thread_id": "t1"}, {"result": "r"})
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
