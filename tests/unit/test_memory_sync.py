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

    async def log_activity(self, user_id, thread_id, messages, **kwargs):
        self.calls.append(("log_activity", thread_id))
        await asyncio.sleep(0)
        return {"enqueued": True, "thread_id": thread_id}

    async def recall_activity(self, user_id, query, **kwargs):
        self.calls.append(("recall_activity", query))
        await asyncio.sleep(0)
        return {"results": [{"step": 0}], "count": 1}

    async def get_thread(self, user_id, thread_id, **kwargs):
        self.calls.append(("get_thread", thread_id))
        await asyncio.sleep(0)
        return {"results": [{"step": 0}], "count": 1}

    async def get_activity_by_correlation(self, user_id, correlation_id, **kwargs):
        self.calls.append(("get_correlation", correlation_id))
        await asyncio.sleep(0)
        return {"results": [], "count": 0}

    async def flush_activity(self, timeout=5.0):
        self.calls.append(("flush_activity", timeout))
        await asyncio.sleep(0)
        return True

    async def set_activity_retention(self, user_id, *, ttl_seconds):
        self.calls.append(("set_activity_retention", ttl_seconds))
        await asyncio.sleep(0)
        return {"ttl_seconds": ttl_seconds}

    def activity_stats(self):
        """Synchronous on the real core too — no await, no loop hop."""
        self.calls.append(("activity_stats", None))
        return {"enqueued": 1}

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
                         "recall_decision", "health", "wipe_user_data",
                         "log_activity", "recall_activity", "get_thread",
                         "get_activity_by_correlation", "flush_activity",
                         "set_activity_retention", "activity_stats"):
                assert callable(getattr(m, name))
        finally:
            m.close()

    def test_no_facade_method_is_missing_a_twin(self):
        """Derived from AsyncMemory, so a new facade method can't be forgotten.

        ``Memory`` has no ``__getattr__`` fallthrough — every twin is written by
        hand, which makes an omission silent until a user hits ``AttributeError``
        at runtime. Enumerating the async surface turns that into a test failure.
        """
        import inspect

        from agent_memory.memory import AsyncMemory

        public = {
            name for name, fn in inspect.getmembers(AsyncMemory, inspect.isfunction)
            if not name.startswith("_") and name != "create"
        }
        missing = sorted(name for name in public if not hasattr(Memory, name))
        assert missing == [], f"Memory is missing blocking twins for: {missing}"

    def test_episodic_twins_reach_the_core(self):
        # TC-SYNC-EP-001: the twins actually delegate, not just exist.
        m, core = _memory_with_fake()
        try:
            assert m.log_activity("u1", "t1", [{"type": "human"}])["enqueued"] is True
            assert m.recall_activity("u1", "q")["count"] == 1
            assert m.get_thread("u1", "t1")["count"] == 1
            assert m.flush_activity() is True
            assert m.activity_stats() == {"enqueued": 1}
            assert ("log_activity", "t1") in core.calls
        finally:
            m.close()

    def test_activity_stats_needs_no_loop_hop(self):
        """It is already synchronous on the core, so it must not be submitted.

        Routing it through the background loop would make an /health probe wait
        on whatever else that loop is doing — the one call that must never block.
        """
        m, core = _memory_with_fake()
        try:
            m._submit = lambda coro: pytest.fail("activity_stats went through _submit")
            assert m.activity_stats() == {"enqueued": 1}
        finally:
            m._loop.call_soon_threadsafe(m._loop.stop)
            m._thread.join(timeout=5)
            m._loop.close()
