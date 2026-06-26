"""Tests for decision MCP tools (store_decision, recall_decision)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.core.registry import ServiceRegistry


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_registry(config=None, decision_enabled=True):
    reg = MagicMock(spec=ServiceRegistry)
    reg.config = config or _make_config()
    reg.audit_service = AsyncMock()
    reg.audit_service.log = AsyncMock()
    reg.check_access = AsyncMock(return_value=None)
    if decision_enabled:
        reg.decision_service = AsyncMock()
    else:
        reg.decision_service = None
    return reg


def _capture_tool(mcp_mock):
    tools = {}
    mcp_mock.tool = lambda **kwargs: lambda fn: tools.update({kwargs["name"]: fn}) or fn
    return tools


class TestStoreDecision:
    """store_decision tool tests."""

    async def test_store_success(self):
        reg = _make_registry()
        reg.decision_service.store = AsyncMock(return_value="stored")

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.decision_tools import register_decision_tools
        register_decision_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["store_decision"](
                user_id="user1", key="editor", value="vim",
            )

        assert result["action"] == "stored"
        assert result["key"] == "editor"

    async def test_store_disabled(self):
        reg = _make_registry(decision_enabled=False)

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.decision_tools import register_decision_tools
        register_decision_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["store_decision"](
                user_id="user1", key="editor", value="vim",
            )

        assert "error" in result

    async def test_store_error_logs_and_raises(self):
        reg = _make_registry()
        reg.decision_service.store = AsyncMock(side_effect=RuntimeError("db down"))

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.decision_tools import register_decision_tools
        register_decision_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            with pytest.raises(RuntimeError, match="db down"):
                await tools["store_decision"](
                    user_id="user1", key="editor", value="vim",
                )

        assert reg.audit_service.log.call_args_list[-1][0][3] == "error"


class TestRecallDecision:
    """recall_decision tool tests."""

    async def test_recall_found(self):
        reg = _make_registry()
        reg.decision_service.recall = AsyncMock(return_value={
            "key": "editor",
            "value": "vim",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "expires_at": "2025-12-31T00:00:00",
        })

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.decision_tools import register_decision_tools
        register_decision_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["recall_decision"](user_id="user1", key="editor")

        assert result["found"] is True
        assert result["decision"]["value"] == "vim"

    async def test_recall_not_found(self):
        reg = _make_registry()
        reg.decision_service.recall = AsyncMock(return_value=None)

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.decision_tools import register_decision_tools
        register_decision_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["recall_decision"](user_id="user1", key="nonexistent")

        assert result["found"] is False

    async def test_recall_disabled(self):
        reg = _make_registry(decision_enabled=False)

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.decision_tools import register_decision_tools
        register_decision_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["recall_decision"](user_id="user1", key="editor")

        assert "error" in result


class TestDecisionAccessControl:
    """Decision tools respect check_access governance/rate-limit checks."""

    async def test_store_decision_blocked(self):
        reg = _make_registry()
        reg.check_access = AsyncMock(return_value="rate limited")

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)
        from agent_memory.tools.decision_tools import register_decision_tools
        register_decision_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["store_decision"](
                user_id="user1", key="editor", value="vim",
            )

        assert result == {"error": "rate limited"}
        reg.decision_service.store.assert_not_called()

    async def test_recall_decision_blocked(self):
        reg = _make_registry()
        reg.check_access = AsyncMock(return_value="not allowed")

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)
        from agent_memory.tools.decision_tools import register_decision_tools
        register_decision_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["recall_decision"](user_id="user1", key="editor")

        assert result == {"error": "not allowed"}
        reg.decision_service.recall.assert_not_called()
