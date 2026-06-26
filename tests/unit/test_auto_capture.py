"""Tests for AutoCaptureMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.core.config import MCPConfig


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


class TestShouldCapture:
    """TC-E-014/015: should_capture() logic."""

    def test_returns_true_for_configured_tool(self):
        """TC-E-014: should_capture returns True for tools in auto_capture_tools."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config()
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        assert acm.should_capture("recall_memory", {"user_id": "u1"}) is True
        assert acm.should_capture("hybrid_search", {"user_id": "u1"}) is True
        assert acm.should_capture("search_web", {"user_id": "u1"}) is True
        assert acm.should_capture("store_decision", {"user_id": "u1"}) is True
        assert acm.should_capture("recall_decision", {"user_id": "u1"}) is True

    def test_returns_false_when_disabled(self):
        """TC-E-015: should_capture returns False when auto_capture_enabled=False."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config(auto_capture_enabled=False)
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        assert acm.should_capture("recall_memory", {"user_id": "u1"}) is False


class TestCaptureToolFiltering:
    """TC-E-016/017/021/022: Tool inclusion/exclusion."""

    def test_captures_configured_tools(self):
        """TC-E-016: Captures recall_memory, hybrid_search, search_web."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config()
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        for tool in ["recall_memory", "hybrid_search", "search_web"]:
            assert acm.should_capture(tool, {"user_id": "u1"}) is True

    def test_does_not_capture_unconfigured_tools(self):
        """TC-E-017: Unconfigured tools are not captured."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config()
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        assert acm.should_capture("some_random_tool", {"user_id": "u1"}) is False

    def test_store_memory_excluded(self):
        """TC-E-021: store_memory is always excluded from capture."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config(
            auto_capture_tools=["recall_memory", "store_memory"],
        )
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        assert acm.should_capture("store_memory", {"user_id": "u1"}) is False

    def test_destructive_ops_excluded(self):
        """TC-E-022: wipe_user_data and delete_memory always excluded."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config(
            auto_capture_tools=[
                "recall_memory", "wipe_user_data", "delete_memory",
            ],
        )
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        assert acm.should_capture("wipe_user_data", {"user_id": "u1"}) is False
        assert acm.should_capture("delete_memory", {"user_id": "u1"}) is False


class TestCaptureStoresSTM:
    """TC-E-018: capture() stores STM with auto: prefix."""

    async def test_stores_with_auto_prefix(self):
        """TC-E-018: Captured interaction stored with conversation_id='auto:<tool>'."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        mock_mem = MagicMock()
        mock_mem.store_stm = AsyncMock(return_value=["id1"])
        config = _make_config()
        acm = AutoCaptureMiddleware(memory_service=mock_mem, config=config)

        await acm.capture(
            tool_name="recall_memory",
            params={"user_id": "u1", "query": "test query"},
            response={"memories": [{"content": "some result"}]},
        )

        mock_mem.store_stm.assert_called_once()
        call_kwargs = mock_mem.store_stm.call_args[1]
        assert call_kwargs["user_id"] == "u1"
        assert call_kwargs["conversation_id"] == "auto:recall_memory"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["message_type"] == "system"


class TestCaptureDisabled:
    """TC-E-019: Auto-capture disabled via config means no capture."""

    async def test_disabled_config_no_capture(self):
        """TC-E-019: When auto_capture_enabled=False, capture() is a no-op."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        mock_mem = MagicMock()
        mock_mem.store_stm = AsyncMock()
        config = _make_config(auto_capture_enabled=False)
        acm = AutoCaptureMiddleware(memory_service=mock_mem, config=config)

        await acm.capture(
            tool_name="recall_memory",
            params={"user_id": "u1", "query": "test"},
            response={"result": "data"},
        )

        mock_mem.store_stm.assert_not_called()


class TestCaptureFailureNonFatal:
    """TC-E-020: Storage failure logged, original response unaffected."""

    async def test_capture_failure_does_not_raise(self):
        """TC-E-020: If store_stm raises, capture() logs and returns None."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        mock_mem = MagicMock()
        mock_mem.store_stm = AsyncMock(side_effect=Exception("DB down"))
        config = _make_config()
        acm = AutoCaptureMiddleware(memory_service=mock_mem, config=config)

        # Should NOT raise
        await acm.capture(
            tool_name="recall_memory",
            params={"user_id": "u1", "query": "test"},
            response={"result": "data"},
        )

        mock_mem.store_stm.assert_called_once()


class TestBuildContentTruncation:
    """TC-E-035: Large responses truncated to max_content_length."""

    def test_truncates_large_content(self):
        """TC-E-035: Content exceeding max_content_length is truncated."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config(auto_capture_max_content_length=50)
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        content = acm.build_content(
            tool_name="recall_memory",
            params={"user_id": "u1", "query": "test"},
            response={"data": "x" * 200},
        )

        assert len(content) == 50

    def test_does_not_truncate_short_content(self):
        """Short content is not truncated."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config(auto_capture_max_content_length=2000)
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        content = acm.build_content(
            tool_name="recall_memory",
            params={"user_id": "u1"},
            response={"ok": True},
        )

        assert len(content) < 2000


class TestMissingUserId:
    """TC-E-036: Missing user_id in params skips capture."""

    def test_should_capture_false_without_user_id(self):
        """TC-E-036: should_capture returns False when params lack user_id."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        config = _make_config()
        acm = AutoCaptureMiddleware(memory_service=MagicMock(), config=config)

        assert acm.should_capture("recall_memory", {"query": "test"}) is False

    async def test_capture_skips_without_user_id(self):
        """TC-E-036: capture() skips when params lack user_id."""
        from agent_memory.services.auto_capture import AutoCaptureMiddleware

        mock_mem = MagicMock()
        mock_mem.store_stm = AsyncMock()
        config = _make_config()
        acm = AutoCaptureMiddleware(memory_service=mock_mem, config=config)

        await acm.capture(
            tool_name="recall_memory",
            params={"query": "test"},  # no user_id
            response={"result": "data"},
        )

        mock_mem.store_stm.assert_not_called()
