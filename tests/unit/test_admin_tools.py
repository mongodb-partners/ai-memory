"""Tests for admin MCP tools (memory_health, wipe_user_data, cache_invalidate)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.core.registry import ServiceRegistry


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_registry(config=None):
    reg = MagicMock(spec=ServiceRegistry)
    reg.config = config or _make_config()
    reg.memory_service = AsyncMock()
    reg.cache_service = AsyncMock()
    reg.audit_service = AsyncMock()
    reg.audit_service.log = AsyncMock()
    reg.providers = MagicMock()
    reg.check_access = AsyncMock(return_value=None)
    return reg


def _capture_tool(mcp_mock):
    tools = {}
    mcp_mock.tool = lambda **kwargs: lambda fn: tools.update({kwargs["name"]: fn}) or fn
    return tools


class TestMemoryHealth:
    """memory_health returns tier and enrichment stats."""

    async def test_health_returns_stats(self):
        reg = _make_registry()
        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"_id": {"tier": "stm", "enrichment_status": "not_applicable"}, "count": 10},
            {"_id": {"tier": "ltm", "enrichment_status": "complete"}, "count": 5},
            {"_id": {"tier": "ltm", "enrichment_status": "pending"}, "count": 2},
        ])
        mock_col = MagicMock()
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        mock_db_manager = MagicMock()
        mock_db_manager.db = mock_db

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.core.database.DatabaseManager") as mock_dm:
            mock_dm.get_instance = AsyncMock(return_value=mock_db_manager)

            result = await tools["memory_health"](user_id="user1")

        assert result["total_memories"] == 17
        assert result["tier_stats"]["stm"] == 10
        assert result["tier_stats"]["ltm"] == 7
        assert result["enrichment_stats"]["pending"] == 2

    async def test_health_empty_user(self):
        reg = _make_registry()
        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col = MagicMock()
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        mock_db_manager = MagicMock()
        mock_db_manager.db = mock_db

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.core.database.DatabaseManager") as mock_dm:
            mock_dm.get_instance = AsyncMock(return_value=mock_db_manager)

            result = await tools["memory_health"](user_id="user1")

        assert result["total_memories"] == 0
        assert result["tier_stats"] == {}


class TestWipeUserData:
    """wipe_user_data hard-deletes from all collections."""

    async def test_wipe_requires_confirm(self):
        reg = _make_registry()
        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["wipe_user_data"](user_id="user1", confirm=False)

        assert "error" in result
        assert "confirm" in result["error"].lower()

    async def test_wipe_deletes_all_collections(self):
        reg = _make_registry()
        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        mock_memories = MagicMock()
        mock_memories.delete_many = AsyncMock(return_value=MagicMock(deleted_count=10))
        mock_cache = MagicMock()
        mock_cache.delete_many = AsyncMock(return_value=MagicMock(deleted_count=3))
        mock_audit = MagicMock()
        mock_audit.delete_many = AsyncMock(return_value=MagicMock(deleted_count=5))

        mock_db = MagicMock()
        def getitem(name):
            return {"memories": mock_memories, "semantic_cache": mock_cache, "audit_log": mock_audit}[name]
        mock_db.__getitem__ = MagicMock(side_effect=getitem)

        mock_db_manager = MagicMock()
        mock_db_manager.db = mock_db

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.core.database.DatabaseManager") as mock_dm:
            mock_dm.get_instance = AsyncMock(return_value=mock_db_manager)

            result = await tools["wipe_user_data"](user_id="user1", confirm=True)

        assert result["memories_deleted"] == 10
        assert result["cache_deleted"] == 3
        assert result["audit_deleted"] == 5

    async def test_wipe_without_confirm_default(self):
        reg = _make_registry()
        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["wipe_user_data"](user_id="user1")

        assert "error" in result

    async def test_wipe_logs_audit(self):
        reg = _make_registry()
        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        mock_col = MagicMock()
        mock_col.delete_many = AsyncMock(return_value=MagicMock(deleted_count=0))
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        mock_db_manager = MagicMock()
        mock_db_manager.db = mock_db

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.core.database.DatabaseManager") as mock_dm:
            mock_dm.get_instance = AsyncMock(return_value=mock_db_manager)

            await tools["wipe_user_data"](user_id="user1", confirm=True)

        reg.audit_service.log.assert_called_once()


class TestCacheInvalidate:
    """cache_invalidate wraps CacheService.invalidate."""

    async def test_invalidate_all(self):
        reg = _make_registry()
        reg.cache_service.invalidate = AsyncMock(return_value=5)

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["cache_invalidate"](
                user_id="user1", invalidate_all=True,
            )

        assert result["deleted_count"] == 5
        reg.cache_service.invalidate.assert_called_once_with(
            "user1", pattern=None, invalidate_all=True,
        )

    async def test_invalidate_pattern(self):
        reg = _make_registry()
        reg.cache_service.invalidate = AsyncMock(return_value=2)

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["cache_invalidate"](
                user_id="user1", pattern="weather.*",
            )

        assert result["deleted_count"] == 2

    async def test_invalidate_none_returns_zero(self):
        reg = _make_registry()
        reg.cache_service.invalidate = AsyncMock(return_value=0)

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["cache_invalidate"](user_id="user1")

        assert result["deleted_count"] == 0

    async def test_invalidate_error_logs_and_raises(self):
        reg = _make_registry()
        reg.cache_service.invalidate = AsyncMock(side_effect=RuntimeError("db down"))

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            with pytest.raises(RuntimeError, match="db down"):
                await tools["cache_invalidate"](user_id="user1", invalidate_all=True)

        audit_calls = reg.audit_service.log.call_args_list
        assert audit_calls[-1][0][3] == "error"


class TestAdminAccessControl:
    """Admin tools respect check_access governance/rate-limit checks."""

    async def test_memory_health_blocked(self):
        reg = _make_registry()
        reg.check_access = AsyncMock(return_value="not allowed")

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)
        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["memory_health"](user_id="user1")

        assert result == {"error": "not allowed"}

    async def test_wipe_user_data_blocked(self):
        reg = _make_registry()
        reg.check_access = AsyncMock(return_value="not allowed")

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)
        from agent_memory.tools.admin_tools import register_admin_tools
        register_admin_tools(mcp_mock)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["wipe_user_data"](user_id="user1", confirm=True)

        assert result == {"error": "not allowed"}
