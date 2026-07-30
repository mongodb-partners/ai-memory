"""Tests for the MCP shell over the AsyncMemory facade. REQ-E-060..063."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.exceptions import AccessError, RateLimitError
from agent_memory.shells.mcp.auto_capture import AutoCaptureMiddleware
from agent_memory.shells.mcp.tools import register_all_tools


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
        # `user_id` is the resolved identity, passed in by `wrap_tools`. `capture`
        # will not read it out of the params dict — that was the cross-tenant write.
        await mw.capture("recall_memory", {"user_id": "u1", "query": "q"},
                         {"result": "r"}, user_id="u1")
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


class TestCaptureContentBudget:
    """REQ-E-087: truncation must not silently change what a memory says."""

    @staticmethod
    def _mw(max_len=200):
        config = MagicMock()
        config.auto_capture_enabled = True
        config.auto_capture_tools = ["recall_memory"]
        config.auto_capture_min_length = 1
        config.auto_capture_max_content_length = max_len
        return AutoCaptureMiddleware(_app(), config)

    def test_a_long_query_does_not_evict_the_result(self):
        """TC-MCP-CAP-010: the result is the part worth remembering.

        The single-slice version cut the joined string at the cap, so a large params
        dict consumed the whole budget and the memory ended mid-key with the result
        absent entirely. The stored memory then described a call whose outcome it did
        not contain — and got embedded and recalled as if it did.
        """
        content = self._mw().build_content(
            "recall_memory",
            {"user_id": "u1", "filters": "x" * 5000},
            {"status": "succeeded", "count": 3},
        )
        assert "Result:" in content
        assert "succeeded" in content

    def test_every_truncation_is_marked(self):
        """TC-MCP-CAP-011: an unmarked cut of a repr reads as a complete sentence.

        `Result: {'status': 'fail` is indistinguishable from a real value to the
        embedder, to the enrichment LLM, and to a human reading recall output.
        """
        content = self._mw().build_content(
            "recall_memory", {"q": "y" * 500}, {"r": "z" * 500}
        )
        assert content.count("…") == 2

    def test_the_cap_is_respected(self):
        # TC-MCP-CAP-012
        content = self._mw(max_len=150).build_content(
            "recall_memory", {"q": "y" * 500}, {"r": "z" * 500}
        )
        assert len(content) <= 150

    def test_a_short_query_lets_the_result_run_longer(self):
        """TC-MCP-CAP-013: unused query budget is reclaimed, not wasted."""
        mw = self._mw(max_len=300)
        long_result = {"r": "z" * 5000}
        generous = mw.build_content("recall_memory", {"q": "s"}, long_result)
        tight = mw.build_content("recall_memory", {"q": "y" * 5000}, long_result)
        assert len(generous.split("Result: ")[1]) > len(tight.split("Result: ")[1])

    def test_a_pathologically_small_cap_keeps_the_tool_name(self):
        """TC-MCP-CAP-014: an unidentifiable memory is worse than a short one."""
        content = self._mw(max_len=20).build_content(
            "recall_memory", {"q": "y" * 500}, {"r": "z" * 500}
        )
        assert len(content) <= 20
        assert "recall_memory" in content

    def test_short_content_is_untouched(self):
        # TC-MCP-CAP-015: no ellipsis on the common path.
        content = self._mw().build_content("recall_memory", {"q": "hi"}, {"r": "ok"})
        assert "…" not in content


class TestCaptureTaskRetention:
    """REQ-E-088: a fire-and-forget task must still hold a strong reference."""

    async def test_spawn_retains_the_task_until_it_completes(self):
        """TC-MCP-CAP-020: `create_task` returns the only strong reference.

        The loop holds a weak one, so a discarded handle lets the task be
        garbage-collected mid-await: the write stops half-done and nothing raises.
        It is a race that gets *rarer* under light load, which is the worst profile
        — it will not reproduce in testing and shows up in production as memories
        that occasionally go missing.
        """
        mw = TestCaptureContentBudget._mw()
        task = mw.spawn("recall_memory", {"user_id": "u1", "q": "q"}, {"result": "r"})
        assert task in mw._pending
        await task
        # …and the set does not grow without bound.
        assert task not in mw._pending

    async def test_drain_waits_for_in_flight_captures(self):
        """TC-MCP-CAP-021: a process shutting down mid-capture loses the write."""
        mw = TestCaptureContentBudget._mw()
        mw.spawn("recall_memory", {"user_id": "u1", "q": "q"}, {"result": "r"},
                 user_id="u1")
        assert await mw.drain(2.0) is True
        assert mw._pending == set()
        mw.app.add.assert_awaited_once()

    async def test_drain_is_true_when_nothing_is_pending(self):
        # TC-MCP-CAP-022
        assert await TestCaptureContentBudget._mw().drain(0.1) is True


class TestLifespan:
    """TC-MCP-LIFE-001: lifespan creates and closes its own AsyncMemory."""

    async def test_lifespan_creates_and_closes_facade(self, monkeypatch):
        import agent_memory.shells.mcp.server as server
        from agent_memory.config import MemoryConfig

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
        import agent_memory.shells.mcp.server as server
        from agent_memory.config import MemoryConfig

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

    async def test_lifespan_drains_auto_capture_before_closing(self, monkeypatch):
        """TC-MCP-LIFE-003: shutdown must wait for in-flight captures.

        The middleware was constructed inline and its reference discarded, so
        `drain()` — which exists precisely for this — had no caller. The facade
        then closed underneath any capture still awaiting an embedding round trip:
        the write lost, or racing a closing Mongo client on the way out.

        Ordering matters as much as the call. Draining *after* close would await
        writes against a closed facade, so this records the sequence rather than
        just the fact.
        """
        import agent_memory.shells.mcp.server as server
        from agent_memory.config import MemoryConfig

        events: list[str] = []

        instance = MagicMock()
        instance.close = AsyncMock(side_effect=lambda: events.append("close"))
        monkeypatch.setattr(server.AsyncMemory, "create", AsyncMock(return_value=instance))
        monkeypatch.setattr(server, "register_all_tools", lambda m, a: None)
        monkeypatch.setattr(server, "wrap_tools", lambda m, c: None)

        captured: dict = {}

        class _Middleware:
            def __init__(self, app, config):
                captured["mw"] = self
                self.drain = AsyncMock(
                    side_effect=lambda *a, **k: events.append("drain") or True
                )

        monkeypatch.setattr(server, "AutoCaptureMiddleware", _Middleware)

        cfg = MemoryConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            auto_capture_enabled=True, _env_file=None,
        )
        lifespan = server.make_lifespan(cfg)
        async with lifespan(MagicMock()):
            pass

        captured["mw"].drain.assert_awaited_once()
        assert events == ["drain", "close"], (
            f"expected drain before close, got {events}"
        )


class TestHealthParity:
    """REQ-E-090: both shells answer the same question about the same process."""

    def _client(self, facade):
        from starlette.testclient import TestClient

        from agent_memory.config import MemoryConfig
        from agent_memory.shells.mcp.server import create_mcp

        config = MemoryConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            auto_capture_enabled=False, _env_file=None,
        )
        mcp = create_mcp(config=config, app=facade)
        return TestClient(mcp.http_app())

    @staticmethod
    def _facade(running=True):
        facade = MagicMock()
        facade.activity_stats = MagicMock(return_value={"enqueued": 3, "queue_depth": 0})
        facade.worker_status = MagicMock(return_value={
            "enabled": True, "running": running, "workers": {},
        })
        return facade

    def test_mcp_health_reports_the_same_body_as_rest(self):
        """TC-MCP-HEALTH-001: it used to return a bare `{"status": "ok"}`.

        That is a probe reporting only "the process accepts sockets". In a
        dual-transport deployment both shells share one facade, so an operator
        watching the MCP port got a permanently cheerful answer about a process the
        REST port would have called degraded.
        """
        with self._client(self._facade()) as client:
            body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["episodic"]["enqueued"] == 3
        assert "workers" in body

    def test_mcp_health_degrades_on_a_dead_worker(self):
        # TC-MCP-HEALTH-002: whichever port the monitor targets, the outage shows.
        with self._client(self._facade(running=False)) as client:
            assert client.get("/health").json()["status"] == "degraded"

    def test_the_two_shells_share_one_definition(self):
        """TC-MCP-HEALTH-003: same facade in, same body out.

        Asserted against the shared builder rather than by comparing two live
        clients, because the point is that there is only one implementation to
        drift from.
        """
        from agent_memory.shells.rest.app import build_health_body

        facade = self._facade(running=False)
        with self._client(facade) as client:
            mcp_body = client.get("/health").json()
        assert mcp_body == build_health_body(facade)
