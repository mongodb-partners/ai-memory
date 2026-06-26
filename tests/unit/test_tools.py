"""Tests for MCP tool functions (memory, cache, search)."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.core.registry import ServiceRegistry


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_registry(config=None):
    """Build a mock ServiceRegistry with all services stubbed."""
    reg = MagicMock(spec=ServiceRegistry)
    reg.config = config or _make_config()
    reg.memory_service = AsyncMock()
    reg.cache_service = AsyncMock()
    reg.audit_service = AsyncMock()
    reg.audit_service.log = AsyncMock()
    reg.providers = MagicMock()
    reg.providers.embedding = AsyncMock()
    reg.providers.embedding.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    reg.check_access = AsyncMock(return_value=None)
    return reg


def _capture_tool(mcp_mock):
    """Return a dict that captures tool functions registered via @mcp.tool()."""
    tools = {}
    mcp_mock.tool = lambda **kwargs: lambda fn: tools.update({kwargs["name"]: fn}) or fn
    return tools


# ─── Memory Tools ───────────────────────────────────────────────


class TestStoreMemory:
    """TC-045: store_memory tool delegates to memory_service.store_stm."""

    async def test_store_memory_success(self):
        reg = _make_registry()
        reg.memory_service.store_stm = AsyncMock(return_value=["id1", "id2"])

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["store_memory"](
                user_id="user1",
                conversation_id="conv1",
                messages=[{"content": "Hello", "message_type": "human"}],
            )

        assert result["stm_ids"] == ["id1", "id2"]
        assert result["count"] == 2
        reg.memory_service.store_stm.assert_called_once_with(
            "user1", "conv1", [{"content": "Hello", "message_type": "human"}],
        )
        reg.audit_service.log.assert_called_once()

    async def test_store_memory_error_logs_and_raises(self):
        reg = _make_registry()
        reg.memory_service.store_stm = AsyncMock(side_effect=RuntimeError("db down"))

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            with pytest.raises(RuntimeError, match="db down"):
                await tools["store_memory"](
                    user_id="user1",
                    conversation_id="conv1",
                    messages=[{"content": "Hello", "message_type": "human"}],
                )

        # Audit log should have been called with error status
        audit_calls = reg.audit_service.log.call_args_list
        assert len(audit_calls) == 1
        assert audit_calls[0][0][3] == "error"


class TestStoreMemoryNormalization:
    """store_memory normalizes 'role' to 'message_type' for MCP clients."""

    async def test_role_field_mapped_to_message_type(self):
        reg = _make_registry()
        reg.memory_service.store_stm = AsyncMock(return_value=["id1"])

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            await tools["store_memory"](
                user_id="user1",
                conversation_id="conv1",
                messages=[{"content": "Hello", "role": "human"}],
            )

        call_args = reg.memory_service.store_stm.call_args[0]
        assert call_args[2][0]["message_type"] == "human"

    async def test_missing_role_defaults_to_human(self):
        reg = _make_registry()
        reg.memory_service.store_stm = AsyncMock(return_value=["id1"])

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            await tools["store_memory"](
                user_id="user1",
                conversation_id="conv1",
                messages=[{"content": "Hello"}],
            )

        call_args = reg.memory_service.store_stm.call_args[0]
        assert call_args[2][0]["message_type"] == "human"

    async def test_explicit_message_type_not_overwritten(self):
        reg = _make_registry()
        reg.memory_service.store_stm = AsyncMock(return_value=["id1"])

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            await tools["store_memory"](
                user_id="user1",
                conversation_id="conv1",
                messages=[{"content": "Hello", "message_type": "ai", "role": "assistant"}],
            )

        call_args = reg.memory_service.store_stm.call_args[0]
        assert call_args[2][0]["message_type"] == "ai"


class TestRecallMemory:
    """TC-046: recall_memory tool delegates to memory_service.recall."""

    async def test_recall_memory_success(self):
        reg = _make_registry()
        reg.memory_service.recall = AsyncMock(return_value=[
            {"_id": "m1", "content": "test", "importance": 0.7}
        ])

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["recall_memory"](
                user_id="user1", query="test query",
            )

        assert result["count"] == 1
        assert result["results"][0]["content"] == "test"


class TestDeleteMemory:
    """TC-047: delete_memory tool delegates to memory_service.delete."""

    async def test_delete_memory_by_id(self):
        reg = _make_registry()
        reg.memory_service.delete = AsyncMock(return_value={"deleted_count": 1, "dry_run": False})

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["delete_memory"](
                user_id="user1", memory_id="abc123",
            )

        assert result["deleted_count"] == 1
        reg.memory_service.delete.assert_called_once()


# ─── Cache Tools ────────────────────────────────────────────────


class TestCheckCache:
    """TC-048: check_cache tool delegates to cache_service.check."""

    async def test_check_cache_hit(self):
        reg = _make_registry()
        reg.cache_service.check = AsyncMock(return_value={
            "cache_hit": True, "response": "cached answer", "score": 0.98,
        })

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.cache_tools import register_cache_tools
        register_cache_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["check_cache"](user_id="user1", query="test query")

        assert result["cache_hit"] is True
        assert result["response"] == "cached answer"

    async def test_check_cache_miss(self):
        reg = _make_registry()
        reg.cache_service.check = AsyncMock(return_value=None)

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.cache_tools import register_cache_tools
        register_cache_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["check_cache"](user_id="user1", query="test query")

        assert result == {"cache_hit": False}


class TestStoreCache:
    """TC-049: store_cache tool delegates to cache_service.store."""

    async def test_store_cache_success(self):
        reg = _make_registry()
        reg.cache_service.store = AsyncMock(return_value="cache_id_123")

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.cache_tools import register_cache_tools
        register_cache_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["store_cache"](
                user_id="user1", query="test query", response="test response",
            )

        assert result["cache_id"] == "cache_id_123"
        reg.cache_service.store.assert_called_once_with("user1", "test query", "test response")


# ─── Search Tools ───────────────────────────────────────────────


class TestRankFusionPipeline:
    """TC-050: $rankFusion pipeline builds correctly."""

    async def test_rankfusion_pipeline_structure(self):
        """$rankFusion pipeline contains vectorPipeline and fullTextPipeline."""
        reg = _make_registry()

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp_mock)

        mock_col = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.tools.search_tools._get_db", new_callable=AsyncMock, return_value=mock_db):
            await tools["hybrid_search"](user_id="user1", query="test query")

        pipeline = mock_col.aggregate.call_args[0][0]
        rank_fusion = pipeline[0]["$rankFusion"]
        assert "vectorPipeline" in rank_fusion["input"]["pipelines"]
        assert "fullTextPipeline" in rank_fusion["input"]["pipelines"]

    async def test_rankfusion_respects_config_weights(self):
        """$rankFusion combination weights match config values."""
        config = _make_config(rrf_vector_weight=0.6, rrf_text_weight=0.4)
        reg = _make_registry(config=config)

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp_mock)

        mock_col = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.tools.search_tools._get_db", new_callable=AsyncMock, return_value=mock_db):
            await tools["hybrid_search"](user_id="user1", query="test")

        pipeline = mock_col.aggregate.call_args[0][0]
        weights = pipeline[0]["$rankFusion"]["combination"]["weights"]
        assert weights["vectorPipeline"] == 0.6
        assert weights["fullTextPipeline"] == 0.4

    async def test_rankfusion_respects_limit(self):
        """$rankFusion pipeline includes $limit stage matching requested limit."""
        reg = _make_registry()

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp_mock)

        mock_col = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.tools.search_tools._get_db", new_callable=AsyncMock, return_value=mock_db):
            await tools["hybrid_search"](user_id="user1", query="test", limit=7)

        pipeline = mock_col.aggregate.call_args[0][0]
        assert pipeline[1] == {"$limit": 7}

    async def test_rankfusion_includes_project_stage(self):
        """$rankFusion pipeline includes $project to exclude embedding."""
        reg = _make_registry()

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp_mock)

        mock_col = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.tools.search_tools._get_db", new_callable=AsyncMock, return_value=mock_db):
            await tools["hybrid_search"](user_id="user1", query="test")

        pipeline = mock_col.aggregate.call_args[0][0]
        assert pipeline[2] == {"$project": {"embedding": 0}}

    async def test_rankfusion_combination_has_weights_only(self):
        """$rankFusion combination contains weights but no rankConstant."""
        config = _make_config(rrf_vector_weight=0.8, rrf_text_weight=0.4)
        reg = _make_registry(config=config)

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp_mock)

        mock_col = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.tools.search_tools._get_db", new_callable=AsyncMock, return_value=mock_db):
            await tools["hybrid_search"](user_id="user1", query="test")

        pipeline = mock_col.aggregate.call_args[0][0]
        combination = pipeline[0]["$rankFusion"]["combination"]
        assert combination["weights"]["vectorPipeline"] == 0.8
        assert combination["weights"]["fullTextPipeline"] == 0.4
        assert "rankConstant" not in combination


class TestHybridSearch:
    """TC-051: hybrid_search tool executes $rankFusion pipeline."""

    async def test_hybrid_search_success(self):
        reg = _make_registry()

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp_mock)

        mock_col = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"_id": "m1", "content": "result 1", "importance": 0.7},
            {"_id": "m2", "content": "result 2", "importance": 0.6},
        ])
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.tools.search_tools._get_db", new_callable=AsyncMock, return_value=mock_db):
            result = await tools["hybrid_search"](
                user_id="user1", query="test query",
            )

        assert "results" in result
        assert "count" in result
        assert result["count"] == 2
        # aggregate called once (single $rankFusion pipeline)
        assert mock_col.aggregate.call_count == 1


class TestSearchWeb:
    """TC-052: search_web tool delegates to Tavily."""

    async def test_search_web_no_api_key(self):
        reg = _make_registry(config=_make_config(tavily_api_key=""))

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["search_web"](user_id="user1", query="test")

        assert "error" in result
        assert "Tavily" in result["error"]

    async def test_search_web_success(self):
        reg = _make_registry(config=_make_config(tavily_api_key="test-key"))

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp)

        mock_tavily_client = MagicMock()
        mock_tavily_client.search = MagicMock(return_value={
            "results": [{"title": "Result 1", "url": "http://example.com"}]
        })

        import asyncio as real_asyncio

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch.dict("sys.modules", {"tavily": MagicMock()}), \
             patch("agent_memory.tools.search_tools.asyncio") as mock_asyncio:

            import sys
            mock_tavily_mod = sys.modules["tavily"]
            mock_tavily_mod.TavilyClient = MagicMock(return_value=mock_tavily_client)

            mock_asyncio.to_thread = AsyncMock(return_value={
                "results": [{"title": "Result 1", "url": "http://example.com"}]
            })
            # Need gather to still work for hybrid_search
            mock_asyncio.gather = real_asyncio.gather

            result = await tools["search_web"](
                user_id="user1", query="test query",
            )

        assert result["results"][0]["title"] == "Result 1"
        assert result["query"] == "test query"


# ─── Tool Error Paths ──────────────────────────────────────────


class TestRecallMemoryError:
    """recall_memory tool error path."""

    async def test_recall_memory_error_logs_and_raises(self):
        reg = _make_registry()
        reg.memory_service.recall = AsyncMock(side_effect=RuntimeError("db down"))

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            with pytest.raises(RuntimeError, match="db down"):
                await tools["recall_memory"](user_id="user1", query="test")

        audit_calls = reg.audit_service.log.call_args_list
        assert len(audit_calls) == 1
        assert audit_calls[0][0][3] == "error"


class TestDeleteMemoryError:
    """delete_memory tool error path."""

    async def test_delete_memory_error_logs_and_raises(self):
        reg = _make_registry()
        reg.memory_service.delete = AsyncMock(side_effect=RuntimeError("db down"))

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            with pytest.raises(RuntimeError, match="db down"):
                await tools["delete_memory"](user_id="user1", memory_id="abc", confirm=True)

        audit_calls = reg.audit_service.log.call_args_list
        assert len(audit_calls) == 1
        assert audit_calls[0][0][3] == "error"


class TestCheckCacheError:
    """check_cache tool error path."""

    async def test_check_cache_error_logs_and_raises(self):
        reg = _make_registry()
        reg.cache_service.check = AsyncMock(side_effect=RuntimeError("db down"))

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.cache_tools import register_cache_tools
        register_cache_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            with pytest.raises(RuntimeError, match="db down"):
                await tools["check_cache"](user_id="user1", query="test")

        audit_calls = reg.audit_service.log.call_args_list
        assert len(audit_calls) == 1
        assert audit_calls[0][0][3] == "error"


class TestStoreCacheError:
    """store_cache tool error path."""

    async def test_store_cache_error_logs_and_raises(self):
        reg = _make_registry()
        reg.cache_service.store = AsyncMock(side_effect=RuntimeError("db down"))

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.cache_tools import register_cache_tools
        register_cache_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            with pytest.raises(RuntimeError, match="db down"):
                await tools["store_cache"](
                    user_id="user1", query="q", response="r",
                )

        audit_calls = reg.audit_service.log.call_args_list
        assert len(audit_calls) == 1
        assert audit_calls[0][0][3] == "error"


class TestHybridSearchError:
    """hybrid_search tool error path."""

    async def test_hybrid_search_error_logs_and_raises(self):
        reg = _make_registry()

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.tools.search_tools._get_db", new_callable=AsyncMock,
                   side_effect=RuntimeError("db down")):
            with pytest.raises(RuntimeError, match="db down"):
                await tools["hybrid_search"](user_id="user1", query="test")

        audit_calls = reg.audit_service.log.call_args_list
        assert len(audit_calls) == 1
        assert audit_calls[0][0][3] == "error"


class TestSearchWebError:
    """search_web tool error path."""

    async def test_search_web_exception_logs_and_raises(self):
        reg = _make_registry(config=_make_config(tavily_api_key="test-key"))

        mcp = MagicMock()
        tools = _capture_tool(mcp)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch.dict("sys.modules", {"tavily": MagicMock()}), \
             patch("agent_memory.tools.search_tools.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(side_effect=RuntimeError("tavily down"))
            import sys
            sys.modules["tavily"].TavilyClient = MagicMock()
            with pytest.raises(RuntimeError, match="tavily down"):
                await tools["search_web"](user_id="user1", query="test")

        audit_calls = reg.audit_service.log.call_args_list
        assert len(audit_calls) == 1
        assert audit_calls[0][0][3] == "error"


class TestHybridSearchFilters:
    """hybrid_search applies memory_type and tags filters via $rankFusion."""

    async def test_hybrid_search_with_memory_type_and_tags(self):
        reg = _make_registry()

        mcp_mock = MagicMock()
        tools = _capture_tool(mcp_mock)

        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp_mock)

        mock_col = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.aggregate = AsyncMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_col)

        with patch.object(ServiceRegistry, "get", return_value=reg), \
             patch("agent_memory.tools.search_tools._get_db", new_callable=AsyncMock, return_value=mock_db):
            result = await tools["hybrid_search"](
                user_id="user1", query="test",
                memory_type="factual", tags=["topic:ai"],
            )

        assert result["count"] == 0
        # Verify filters were applied to vectorSearch inside $rankFusion
        pipeline = mock_col.aggregate.call_args[0][0]
        rank_fusion = pipeline[0]["$rankFusion"]
        vector_pipeline = rank_fusion["input"]["pipelines"]["vectorPipeline"]
        vs_filter = vector_pipeline[0]["$vectorSearch"]["filter"]
        assert vs_filter.get("memory_type") == "factual"
        assert vs_filter.get("tags") == {"$all": ["topic:ai"]}


class TestGetDb:
    """_get_db helper fetches db from DatabaseManager."""

    async def test_get_db_returns_db(self):
        from agent_memory.tools.search_tools import _get_db
        from agent_memory.core.database import DatabaseManager

        mock_db = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.db = mock_db

        with patch.object(DatabaseManager, "get_instance", return_value=mock_mgr):
            result = await _get_db()
        assert result is mock_db


class TestSearchToolsSanitizeDoc:
    """_sanitize_doc in search_tools handles ObjectId, datetime, nested dict."""

    def test_sanitize_objectid(self):
        from agent_memory.tools.search_tools import _sanitize_doc
        from bson import ObjectId
        oid = ObjectId()
        doc = {"_id": oid}
        _sanitize_doc(doc)
        assert doc["_id"] == str(oid)

    def test_sanitize_datetime(self):
        from agent_memory.tools.search_tools import _sanitize_doc
        from datetime import datetime, timezone
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        doc = {"ts": dt}
        _sanitize_doc(doc)
        assert doc["ts"] == dt.isoformat()

    def test_sanitize_nested_dict(self):
        from agent_memory.tools.search_tools import _sanitize_doc
        from bson import ObjectId
        oid = ObjectId()
        doc = {"nested": {"_id": oid}}
        _sanitize_doc(doc)
        assert doc["nested"]["_id"] == str(oid)


# ─── Access Control (check_access) ────────────────────────────


class TestCheckAccessBlocking:
    """Verify tools return error dict when check_access denies access."""

    async def test_store_memory_blocked_by_governance(self):
        reg = _make_registry()
        reg.check_access = AsyncMock(return_value="Operation 'store_memory' not allowed for role 'end_user'")

        mcp = MagicMock()
        tools = _capture_tool(mcp)
        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["store_memory"](
                user_id="user1", conversation_id="conv1", messages=[],
            )

        assert "error" in result
        assert "not allowed" in result["error"]
        reg.memory_service.store_stm.assert_not_called()

    async def test_recall_memory_blocked_by_rate_limit(self):
        reg = _make_registry()
        reg.check_access = AsyncMock(return_value="Rate limit exceeded for 'recall_memory'")

        mcp = MagicMock()
        tools = _capture_tool(mcp)
        from agent_memory.tools.memory_tools import register_memory_tools
        register_memory_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["recall_memory"](user_id="user1", query="test")

        assert "error" in result
        assert "Rate limit" in result["error"]

    async def test_check_cache_blocked(self):
        reg = _make_registry()
        reg.check_access = AsyncMock(return_value="blocked")

        mcp = MagicMock()
        tools = _capture_tool(mcp)
        from agent_memory.tools.cache_tools import register_cache_tools
        register_cache_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["check_cache"](user_id="user1", query="test")

        assert result == {"error": "blocked"}

    async def test_hybrid_search_blocked(self):
        reg = _make_registry()
        reg.check_access = AsyncMock(return_value="blocked")

        mcp = MagicMock()
        tools = _capture_tool(mcp)
        from agent_memory.tools.search_tools import register_search_tools
        register_search_tools(mcp)

        with patch.object(ServiceRegistry, "get", return_value=reg):
            result = await tools["hybrid_search"](user_id="user1", query="test")

        assert result == {"error": "blocked"}
