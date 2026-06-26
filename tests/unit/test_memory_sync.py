"""Tests for the Memory sync wrapper. REQ-E-040,041,042.

The wrapper runs an AsyncMemory core on a dedicated daemon-thread event loop so
sync callers (scripts, notebooks) work without managing a loop. We inject a fake
async core so these tests stay free of Atlas/providers.
"""

import asyncio

import pytest

from agent_memory.sync import Memory


class _FakeAsyncCore:
    """Stands in for AsyncMemory — records calls, returns canned values."""

    def __init__(self):
        self.closed = False
        self.calls = []

    async def add(self, user_id, conversation_id, messages):
        self.calls.append(("add", user_id))
        await asyncio.sleep(0)  # force a real await hop
        return {"stm_ids": ["x"], "count": 1}

    async def recall(self, user_id, query, **kwargs):
        self.calls.append(("recall", query))
        await asyncio.sleep(0)
        return {"results": [{"id": 1}], "count": 1}

    async def close(self):
        self.closed = True


def _memory_with_fake():
    """Build Memory bound to a fake core, bypassing AsyncMemory.create()."""
    core = _FakeAsyncCore()
    m = Memory.__new__(Memory)
    m._start_loop()
    m._async = core
    return m, core


class TestSyncWrapper:
    def test_add_from_plain_sync_context(self):
        # TC-SYNC-001
        m, core = _memory_with_fake()
        try:
            out = m.add("u1", "c1", [{"content": "hi"}])
            assert out["count"] == 1
            assert ("add", "u1") in core.calls
        finally:
            m.close()

    def test_recall_returns_result(self):
        m, core = _memory_with_fake()
        try:
            out = m.recall("u1", "q")
            assert out["count"] == 1
        finally:
            m.close()

    def test_works_inside_running_event_loop(self):
        # TC-SYNC-002 (notebook scenario, premortem #4, boundary #1)
        m, core = _memory_with_fake()

        async def notebook_cell():
            # Calling the *sync* API from within a running loop must not raise
            # "loop already running" — the core runs on its own thread/loop.
            return await asyncio.to_thread(m.recall, "u1", "from-loop")

        try:
            out = asyncio.run(notebook_cell())
            assert out["count"] == 1
        finally:
            m.close()

    def test_close_stops_background_thread(self):
        # TC-SYNC-003
        m, core = _memory_with_fake()
        m.close()
        assert core.closed is True
        assert not m._thread.is_alive()

    def test_context_manager(self):
        m, core = _memory_with_fake()
        with m:
            pass
        assert core.closed is True

    def test_all_blocking_twins_present(self):
        # every async facade method has a sync twin
        m, core = _memory_with_fake()
        try:
            for name in ("add", "recall", "search", "delete", "check_cache",
                         "store_cache", "invalidate_cache", "remember_decision",
                         "recall_decision", "search_web", "health", "wipe_user_data"):
                assert callable(getattr(m, name))
        finally:
            m.close()
