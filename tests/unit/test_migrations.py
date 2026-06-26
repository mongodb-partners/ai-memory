"""Tests for migrations.py (ensure_indexes, ensure_search_indexes)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.core.collections import STANDARD_INDEXES, SEARCH_INDEXES


class TestEnsureIndexes:
    """REQ-DB-001: Standard indexes created idempotently on startup."""

    async def test_creates_all_standard_indexes(self):
        """Each STANDARD_INDEXES entry results in a create_index call."""
        from agent_memory.core.migrations import ensure_indexes

        mock_db = MagicMock()
        collections = {}

        def get_col(name):
            if name not in collections:
                col = MagicMock()
                col.create_index = AsyncMock(return_value="ok")
                collections[name] = col
            return collections[name]

        mock_db.__getitem__ = MagicMock(side_effect=get_col)

        await ensure_indexes(mock_db)

        total_calls = sum(
            col.create_index.call_count for col in collections.values()
        )
        assert total_calls == len(STANDARD_INDEXES)

    async def test_idempotent_no_error_on_existing(self):
        """Calling ensure_indexes twice should not raise."""
        from agent_memory.core.migrations import ensure_indexes

        mock_db = MagicMock()
        col = MagicMock()
        col.create_index = AsyncMock(return_value="ok")
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(mock_db)
        await ensure_indexes(mock_db)
        # Should succeed without exception


class TestEnsureIndexesConflict:
    """ensure_indexes handles OperationFailure code 86 (index conflict)."""

    async def test_conflict_drops_and_recreates(self):
        from agent_memory.core.migrations import ensure_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        # First call raises code 86, subsequent calls succeed
        col.create_index = AsyncMock(
            side_effect=[OperationFailure("conflict", code=86)] +
                        [AsyncMock(return_value="ok")] * (len(STANDARD_INDEXES) * 2)
        )
        col.drop_index = AsyncMock()
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(mock_db)
        col.drop_index.assert_called_once()

    async def test_conflict_recreate_failure_logs(self):
        from agent_memory.core.migrations import ensure_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        col.create_index = AsyncMock(
            side_effect=OperationFailure("conflict", code=86)
        )
        col.drop_index = AsyncMock(side_effect=Exception("drop failed"))
        mock_db.__getitem__ = MagicMock(return_value=col)

        # Should not raise — logs the error
        await ensure_indexes(mock_db)

    async def test_non_conflict_operation_failure_logs(self):
        from agent_memory.core.migrations import ensure_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        col.create_index = AsyncMock(
            side_effect=OperationFailure("other error", code=42)
        )
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(mock_db)


class TestEnsureSearchIndexes:
    """REQ-DB-002..004: Atlas Search indexes created in background."""

    async def test_creates_search_indexes_when_not_existing(self):
        """Each SEARCH_INDEXES entry results in create_search_index if not found."""
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        collections = {}

        async def empty_list_search(*args, **kwargs):
            # Simulate no existing indexes
            return AsyncMock(__aiter__=lambda self: self, __anext__=_stop_aiter)()

        async def _stop_aiter(self):
            raise StopAsyncIteration

        def get_col(name):
            if name not in collections:
                col = MagicMock()
                col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
                col.create_search_index = AsyncMock(return_value="idx_name")
                collections[name] = col
            return collections[name]

        mock_db.__getitem__ = MagicMock(side_effect=get_col)

        with patch("agent_memory.core.migrations._wait_for_search_index",
                    new_callable=AsyncMock, return_value=True):
            await ensure_search_indexes(mock_db)

        total_calls = sum(
            col.create_search_index.call_count for col in collections.values()
        )
        assert total_calls == len(SEARCH_INDEXES)

    async def test_skips_existing_search_index(self):
        """REQ-DB-003: If index already exists, skip creation."""
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        col = MagicMock()

        # Return an existing index whose name matches whatever is queried
        def make_existing_iter(index_name):
            return _async_iter_of([{"name": index_name, "queryable": True}])

        col.list_search_indexes = AsyncMock(side_effect=make_existing_iter)
        col.create_search_index = AsyncMock()
        mock_db.__getitem__ = MagicMock(return_value=col)

        with patch("agent_memory.core.migrations._wait_for_search_index",
                    new_callable=AsyncMock, return_value=True):
            await ensure_search_indexes(mock_db)

        col.create_search_index.assert_not_called()

    async def test_handles_non_atlas_gracefully(self):
        """REQ-DB-004: Non-Atlas deployment logs warning, doesn't raise."""
        from agent_memory.core.migrations import ensure_search_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            side_effect=OperationFailure("not supported", code=None)
        )
        mock_db.__getitem__ = MagicMock(return_value=col)

        # Should not raise
        await ensure_search_indexes(mock_db)

    async def test_non_atlas_skips_remaining_indexes(self):
        """After first index fails with OperationFailure, remaining indexes are skipped via break."""
        from agent_memory.core.migrations import ensure_search_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        cols = {}
        call_count = 0

        def get_col(name):
            nonlocal call_count
            if name not in cols:
                col = MagicMock()
                col.list_search_indexes = AsyncMock(
                    side_effect=OperationFailure("not supported", code=None)
                )
                col.create_search_index = AsyncMock()
                cols[name] = col
            call_count += 1
            return cols[name]

        mock_db.__getitem__ = MagicMock(side_effect=get_col)

        await ensure_search_indexes(mock_db)

        # Only the first collection should have list_search_indexes called;
        # the rest should be skipped by the atlas_available break
        total_list_calls = sum(
            c.list_search_indexes.call_count for c in cols.values()
        )
        assert total_list_calls == 1
        # No create calls at all
        total_create_calls = sum(
            c.create_search_index.call_count for c in cols.values()
        )
        assert total_create_calls == 0

    async def test_dimension_mismatch_drops_and_recreates(self):
        """Vector index with wrong dims is dropped and recreated."""
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        col = MagicMock()

        # Existing index has 1536 dims, but we request 1024
        def make_existing_iter(index_name):
            return _async_iter_of([{
                "name": index_name,
                "queryable": True,
                "latestDefinition": {
                    "fields": [{"type": "vector", "path": "embedding", "numDimensions": 1536}]
                },
            }])

        col.list_search_indexes = AsyncMock(side_effect=make_existing_iter)
        col.drop_search_index = AsyncMock()
        col.create_search_index = AsyncMock(return_value="idx_name")
        mock_db.__getitem__ = MagicMock(return_value=col)

        with patch("agent_memory.core.migrations._wait_for_search_index_dropped",
                    new_callable=AsyncMock), \
             patch("agent_memory.core.migrations._wait_for_search_index",
                    new_callable=AsyncMock, return_value=True):
            await ensure_search_indexes(mock_db, embedding_dimension=1024)

        # Should drop and recreate vector indexes
        assert col.drop_search_index.call_count >= 1
        assert col.create_search_index.call_count >= 1

    async def test_search_index_not_queryable_within_timeout(self):
        """Logs warning when index doesn't become queryable."""
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        collections = {}

        def get_col(name):
            if name not in collections:
                col = MagicMock()
                col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
                col.create_search_index = AsyncMock(return_value="idx_name")
                collections[name] = col
            return collections[name]

        mock_db.__getitem__ = MagicMock(side_effect=get_col)

        with patch("agent_memory.core.migrations._wait_for_search_index",
                    new_callable=AsyncMock, return_value=False):
            await ensure_search_indexes(mock_db)  # Should not raise

    async def test_create_search_index_operation_failure(self):
        """OperationFailure on create_search_index is handled."""
        from agent_memory.core.migrations import ensure_search_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
        col.create_search_index = AsyncMock(
            side_effect=OperationFailure("already exists", code=68)
        )
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_search_indexes(mock_db)  # Should not raise

    async def test_create_search_index_unexpected_exception(self):
        """Unexpected exception on create_search_index is handled."""
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        col = MagicMock()
        col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
        col.create_search_index = AsyncMock(side_effect=RuntimeError("boom"))
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_search_indexes(mock_db)  # Should not raise


class TestGetExistingDims:
    """_get_existing_dims extracts numDimensions from index info."""

    def test_extracts_from_latest_definition(self):
        from agent_memory.core.migrations import _get_existing_dims
        info = {"latestDefinition": {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1024}
        ]}}
        assert _get_existing_dims(info) == 1024

    def test_extracts_from_definition_fallback(self):
        from agent_memory.core.migrations import _get_existing_dims
        info = {"definition": {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1536}
        ]}}
        assert _get_existing_dims(info) == 1536

    def test_returns_none_for_non_vector(self):
        from agent_memory.core.migrations import _get_existing_dims
        info = {"latestDefinition": {"fields": [{"type": "filter", "path": "user_id"}]}}
        assert _get_existing_dims(info) is None

    def test_returns_none_for_empty(self):
        from agent_memory.core.migrations import _get_existing_dims
        assert _get_existing_dims({}) is None


class TestWaitForSearchIndexDropped:
    """_wait_for_search_index_dropped polls until index is gone."""

    async def test_returns_when_index_gone(self):
        from agent_memory.core.migrations import _wait_for_search_index_dropped
        col = MagicMock()
        col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            await _wait_for_search_index_dropped(col, "test_idx", timeout=5)

    async def test_returns_on_exception(self):
        from agent_memory.core.migrations import _wait_for_search_index_dropped
        col = MagicMock()
        col.list_search_indexes = AsyncMock(side_effect=Exception("fail"))
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            await _wait_for_search_index_dropped(col, "test_idx", timeout=5)

    async def test_timeout_logs_warning(self):
        from agent_memory.core.migrations import _wait_for_search_index_dropped
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            return_value=_async_iter_of([{"name": "still_here"}])
        )
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            await _wait_for_search_index_dropped(col, "test_idx", timeout=0)

    async def test_polls_then_finds_gone(self):
        """Index still present on first poll, gone on second — covers loop body."""
        from agent_memory.core.migrations import _wait_for_search_index_dropped
        col = MagicMock()
        # First call: index still present; second call: gone
        col.list_search_indexes = AsyncMock(side_effect=[
            _async_iter_of([{"name": "test_idx"}]),
            _empty_async_iter(),
        ])
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            await _wait_for_search_index_dropped(col, "test_idx", timeout=10)


class TestWaitForSearchIndex:
    """_wait_for_search_index polls until index is queryable."""

    async def test_returns_true_when_queryable(self):
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            return_value=_async_iter_of([{"queryable": True}])
        )
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=5)
        assert result is True

    async def test_returns_false_on_timeout(self):
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            return_value=_async_iter_of([{"queryable": False}])
        )
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=0)
        assert result is False

    async def test_handles_exception_during_poll(self):
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(side_effect=Exception("network"))
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=0)
        assert result is False

    async def test_exception_then_queryable(self):
        """Exception on first poll, queryable on second — covers loop body + exception pass."""
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(side_effect=[
            Exception("transient"),
            _async_iter_of([{"queryable": True}]),
        ])
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=10)
        assert result is True

    async def test_not_queryable_then_queryable(self):
        """Not queryable on first poll, queryable on second — covers normal loop body."""
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(side_effect=[
            _async_iter_of([{"queryable": False}]),
            _async_iter_of([{"queryable": True}]),
        ])
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=10)
        assert result is True


# ─── Helpers for async iteration mocking ──────────────────────

class _AsyncIter:
    """Async iterator helper for mocking."""
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _async_iter_of(items):
    return _AsyncIter(items)


def _empty_async_iter():
    return _AsyncIter([])
