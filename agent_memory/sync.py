"""Synchronous ``Memory`` wrapper over the async ``AsyncMemory`` core.

Runs the async core on a dedicated event loop on a daemon thread, so every
async facade method gets a blocking twin that is safe to call from plain scripts
*and* from inside an already-running event loop (Jupyter/notebook) — the
"loop already running" trap is avoided because the core's loop lives on another
thread. Mirrors mem0's ``Memory`` / ``AsyncMemory`` split.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

from agent_memory.config import MemoryConfig


class Memory:
    """Blocking facade. Construct with ``Memory(config)``."""

    def __init__(self, config: MemoryConfig):
        self._start_loop()
        from agent_memory.memory import AsyncMemory

        self._async = self._submit(AsyncMemory.create(config))

    # ── Background loop plumbing ──────────────────────────────────────────────

    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="agent-memory-loop", daemon=True
        )
        self._thread.start()

    def _submit(self, coro):
        """Schedule a coroutine on the background loop and block for its result.

        Distinct from ``AsyncMemory._run`` (the orchestration wrapper); this only
        bridges sync↔async across the thread boundary.
        """
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ── Blocking twins ────────────────────────────────────────────────────────

    def add(self, *a, **k):
        return self._submit(self._async.add(*a, **k))

    def recall(self, *a, **k):
        return self._submit(self._async.recall(*a, **k))

    def search(self, *a, **k):
        return self._submit(self._async.search(*a, **k))

    def delete(self, *a, **k):
        return self._submit(self._async.delete(*a, **k))

    def check_cache(self, *a, **k):
        return self._submit(self._async.check_cache(*a, **k))

    def store_cache(self, *a, **k):
        return self._submit(self._async.store_cache(*a, **k))

    def invalidate_cache(self, *a, **k):
        return self._submit(self._async.invalidate_cache(*a, **k))

    def remember_decision(self, *a, **k):
        return self._submit(self._async.remember_decision(*a, **k))

    def recall_decision(self, *a, **k):
        return self._submit(self._async.recall_decision(*a, **k))

    def health(self, *a, **k):
        return self._submit(self._async.health(*a, **k))

    def wipe_user_data(self, *a, **k):
        return self._submit(self._async.wipe_user_data(*a, **k))

    # ── Episodic memory ───────────────────────────────────────────────────────

    def log_activity(self, *a, **k):
        return self._submit(self._async.log_activity(*a, **k))

    def recall_activity(self, *a, **k):
        return self._submit(self._async.recall_activity(*a, **k))

    def get_thread(self, *a, **k):
        return self._submit(self._async.get_thread(*a, **k))

    def get_activity_by_correlation(self, *a, **k):
        return self._submit(self._async.get_activity_by_correlation(*a, **k))

    def flush_activity(self, *a, **k):
        return self._submit(self._async.flush_activity(*a, **k))

    def set_activity_retention(self, *a, **k):
        return self._submit(self._async.set_activity_retention(*a, **k))

    def activity_stats(self):
        """Already synchronous on the core — no loop hop needed."""
        return self._async.activity_stats()

    def worker_status(self):
        """Already synchronous on the core — no loop hop needed.

        Reads task state, which is owned by the background loop's thread, but only
        via ``Task.done()``/``cancelled()``/``exception()`` — all of which are plain
        attribute reads under the GIL, not loop operations. Hopping the loop to
        fetch them would make a health probe wait on the very loop it is checking.
        """
        return self._async.worker_status()

    # ── Teardown ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the async core, stop the background loop, join its thread."""
        try:
            if getattr(self, "_async", None) is not None:
                self._submit(self._async.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
