"""create() can block until search indexes are queryable (live-test fix #1).

A short-lived library/script caller (`async with AsyncMemory.create(cfg) as m`)
otherwise exits before the background index task finishes, so search/recall
silently return nothing. `await_search_indexes=True` makes create() await index
creation; the default (False) keeps the background behaviour for long-running
servers.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.config import MemoryConfig


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


class TestAwaitSearchIndexesFlag:
    def test_flag_defaults_false(self):
        assert _config().await_search_indexes is False

    def test_flag_settable(self):
        assert _config(await_search_indexes=True).await_search_indexes is True

    def test_flag_from_env(self, monkeypatch):
        monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://h:27017")
        monkeypatch.setenv("AWAIT_SEARCH_INDEXES", "true")
        assert MemoryConfig.from_env(_env_file=None).await_search_indexes is True


class TestCreateAwaitsIndexes:
    """create() awaits (not backgrounds) index build when the flag is set."""

    async def test_await_true_calls_ensure_and_no_pending_task(self, monkeypatch):
        import agent_memory.memory as mem

        ensure_search = AsyncMock()
        self = mem.AsyncMemory.__new__(mem.AsyncMemory)
        await self._provision_search_indexes(MagicMock(), _config(await_search_indexes=True), ensure_search)
        ensure_search.assert_awaited_once()
        # nothing left running in the background
        assert getattr(self, "_search_index_task", None) is None

    async def test_await_false_backgrounds_the_task(self, monkeypatch):
        import agent_memory.memory as mem

        ensure_search = AsyncMock()
        self = mem.AsyncMemory.__new__(mem.AsyncMemory)
        await self._provision_search_indexes(MagicMock(), _config(await_search_indexes=False), ensure_search)
        # a background task was scheduled (not yet awaited)
        assert self._search_index_task is not None
        await self._search_index_task  # let it finish
        ensure_search.assert_awaited_once()
