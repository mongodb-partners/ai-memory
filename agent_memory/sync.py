"""Synchronous ``Memory`` wrapper over the async ``AsyncMemory`` core.

Runs the async core on a dedicated event loop on a daemon thread, so every
async facade method gets a blocking twin that is safe to call from plain scripts
*and* from inside an already-running event loop (Jupyter/notebook) — the
"loop already running" trap is avoided because the core's loop lives on another
thread. Mirrors mem0's ``Memory`` / ``AsyncMemory`` split.

Every twin delegates through ``*a, **k`` and then adopts its async counterpart's
signature and docstring (see ``_adopt_async_signatures``). Both halves matter:
delegating positionally means a new keyword argument on the async method works on
the sync one with no edit here, and adopting the signature means ``help(Memory.add)``
and every IDE report the real parameters instead of ``(*a, **k)``. Without the
adoption the sync class — which the README presents first, because it is the one
a script reaches for — was the half of the API with no discoverable signature and
no arity checking: ``Memory.add()`` with no arguments raised ``TypeError`` from
inside the coroutine rather than at the call.

``_adopt_async_signatures`` also asserts each twin exists on the async core, so a
method renamed there fails at import rather than surviving as a twin that
delegates to nothing.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
from concurrent.futures import Future

from agent_memory.config import MemoryConfig


class Memory:
    """Blocking facade. Construct with ``Memory(config)``."""

    def __init__(self, config: MemoryConfig):
        """Start the background loop, then build the async core on it.

        A failed ``create()`` used to leave the loop and its thread running. The
        thread is a daemon, so the process still exits — but nothing else about it
        is harmless: ``__init__`` raised, so there is no object to call ``close()``
        on, and a caller that retries construction (a script looping over configs,
        a test parametrized across them) accumulates one live event loop and one
        thread per attempt for the life of the process. The loop keeps whatever the
        failed startup scheduled on it, and Python reports "Task was destroyed but
        it is pending" from a thread the caller has no handle on.

        So the loop is torn down here on the way out. The async core's own cleanup
        is its problem — ``AsyncMemory.create`` releases the pool it acquired — and
        must happen *before* the loop stops, which it does, since ``_submit``
        returns only once the coroutine has finished unwinding.
        """
        self._start_loop()
        from agent_memory.memory import AsyncMemory

        try:
            self._async = self._submit(AsyncMemory.create(config))
        except BaseException:
            self._stop_loop()
            raise

    # ── Background loop plumbing ──────────────────────────────────────────────

    def _start_loop(self) -> None:
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="agent-memory-loop", daemon=True
        )
        self._thread.start()

    def _stop_loop(self) -> None:
        """Stop the loop and join its thread. Idempotent, and never raises.

        Shared by ``close()`` and the failed-construction path. Never raises
        because both callers have something more important to report: ``close()``
        must finish releasing the rest, and ``__init__`` must propagate the startup
        error rather than a symptom of tidying up after it.
        """
        loop = getattr(self, "_loop", None)
        if loop is None:
            return
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:  # pragma: no cover - already stopping
            pass
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            # Bounded: a wedged loop must not hang the caller's teardown. A join
            # that times out leaves a daemon thread, which the interpreter will not
            # wait for at exit.
            thread.join(timeout=5)
        try:
            if not loop.is_closed():
                loop.close()
        except RuntimeError:  # pragma: no cover - loop still running after timeout
            pass

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
        """Close the async core, stop the background loop, join its thread.

        Idempotent, and the flag is what makes it so. ``_stop_loop`` tolerating an
        already-stopped loop is not enough on its own: a second ``close()`` would
        reach ``_submit`` first and hand a fresh coroutine to a loop that is gone,
        which raises ``RuntimeError: Event loop is closed`` and leaves the
        coroutine unawaited. That is a real path — ``close()`` inside a ``with``
        block, followed by ``__exit__`` calling it again — where the caller did
        nothing wrong and there is nothing left to do.

        The core is closed *before* the loop stops, because the core's teardown is
        a coroutine that runs on this loop. Stopping the loop first would leave
        ``AsyncMemory.close()`` unable to run at all, leaking from the tidy path
        the pool that the failure path is careful to release.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            if getattr(self, "_async", None) is not None:
                self._submit(self._async.close())
        finally:
            self._stop_loop()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _adopt_async_signatures() -> None:
    """Give each blocking twin its async counterpart's signature and docstring.

    Called once at import. For every ``Memory`` method that names a coroutine on
    ``AsyncMemory``, this copies ``__doc__``, ``__annotations__``, and
    ``__signature__`` across, so introspection, ``help()``, IDE completion, and
    generated docs describe the real parameters instead of ``(*a, **k)``.

    To be precise about the limit: this fixes *description*, not enforcement.
    ``__signature__`` is metadata — the interpreter still dispatches on the real
    ``*a, **k``, so a wrong-arity call is caught by the async method, not here. The
    win is that a caller can now discover the parameters at all, which on the class
    the README presents first was previously impossible without reading the source
    of a different class.

    The bodies stay ``*a, **k``. That is deliberate: it is what keeps a new keyword
    argument on an async method working here without a matching edit, which is the
    property that stopped the two surfaces drifting in the first place. Only the
    *description* is copied, never the calling convention.

    ``close`` is skipped — the sync version does strictly more than await the async
    one (it also stops the loop and joins the thread), so its own docstring is the
    accurate one. ``activity_stats`` is skipped because it is already sync on the
    core and hops no loop.
    """
    from agent_memory.memory import AsyncMemory

    skip = {"close", "activity_stats"}
    for name, twin in list(vars(Memory).items()):
        if name.startswith("_") or name in skip or not callable(twin):
            continue
        target = getattr(AsyncMemory, name, None)
        if target is None:
            # A twin delegating to a method that no longer exists would raise
            # AttributeError on first call, in whatever code happened to use it.
            # Import time is a much better place to find out.
            raise AttributeError(
                f"Memory.{name} has no counterpart on AsyncMemory. Rename or "
                f"remove the blocking twin."
            )
        if not inspect.iscoroutinefunction(target):
            continue
        functools.update_wrapper(twin, target, assigned=("__doc__", "__annotations__"))
        # Drop the coroutine's `-> Coroutine[...]` framing: the twin returns the
        # awaited value, so the async return annotation is already correct.
        twin.__signature__ = inspect.signature(target)
        twin.__wrapped__ = target


_adopt_async_signatures()
