"""Tests for CacheService."""

from unittest.mock import AsyncMock, MagicMock

from bson import ObjectId

from agent_memory.core.config import MCPConfig
from agent_memory.services.cache import CacheService


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_embedding_provider():
    provider = AsyncMock()
    provider.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    return provider


class TestCacheServiceCheck:
    """TC-034, TC-035: Cache check hit/miss."""

    async def test_check_returns_hit_above_threshold(self):
        col = AsyncMock()
        config = _make_config(cache_similarity_threshold=0.95)
        embedding = _make_embedding_provider()
        service = CacheService(col, config, embedding)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"query": "test query", "response": "cached response", "score": 0.98}
        ])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        result = await service.check("user1", "test query")
        assert result is not None
        assert result["cache_hit"] is True
        assert result["response"] == "cached response"
        assert result["score"] == 0.98

    async def test_check_returns_none_below_threshold(self):
        col = AsyncMock()
        config = _make_config(cache_similarity_threshold=0.95)
        embedding = _make_embedding_provider()
        service = CacheService(col, config, embedding)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"query": "different query", "response": "resp", "score": 0.80}
        ])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        result = await service.check("user1", "test query")
        assert result is None

    async def test_check_empty_collection_returns_none(self):
        """REQ-EC-007: Empty collection returns None."""
        col = AsyncMock()
        config = _make_config()
        embedding = _make_embedding_provider()
        service = CacheService(col, config, embedding)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        result = await service.check("user1", "test query")
        assert result is None


class TestCacheServiceStore:
    """TC-036: Cache store."""

    async def test_store_inserts_document(self):
        col = AsyncMock()
        config = _make_config()
        embedding = _make_embedding_provider()
        service = CacheService(col, config, embedding)

        inserted_id = ObjectId()
        col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))

        result = await service.store("user1", "test query", "test response")
        assert result == str(inserted_id)
        col.insert_one.assert_called_once()

        doc = col.insert_one.call_args[0][0]
        assert doc["user_id"] == "user1"
        assert doc["query"] == "test query"
        assert doc["response"] == "test response"
        assert doc["embedding"] == [0.1] * 1536
        assert "created_at" in doc


class TestCacheServiceInvalidate:
    """TC-037: Cache invalidation (hard delete)."""

    async def test_invalidate_all_deletes_user_entries(self):
        col = AsyncMock()
        config = _make_config()
        embedding = _make_embedding_provider()
        service = CacheService(col, config, embedding)

        col.delete_many = AsyncMock(return_value=MagicMock(deleted_count=5))

        count = await service.invalidate("user1", invalidate_all=True)
        assert count == 5
        col.delete_many.assert_called_once_with({"user_id": "user1"})

    async def test_invalidate_by_pattern(self):
        col = AsyncMock()
        config = _make_config()
        embedding = _make_embedding_provider()
        service = CacheService(col, config, embedding)

        col.delete_many = AsyncMock(return_value=MagicMock(deleted_count=2))

        count = await service.invalidate("user1", pattern="weather.*")
        assert count == 2

    async def test_invalidate_no_args_returns_zero(self):
        col = AsyncMock()
        config = _make_config()
        embedding = _make_embedding_provider()
        service = CacheService(col, config, embedding)

        count = await service.invalidate("user1")
        assert count == 0
