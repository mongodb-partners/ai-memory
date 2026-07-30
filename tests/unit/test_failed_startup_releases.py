"""A startup that fails must leave nothing running.

``AsyncMemory.create`` acquires a reference-counted claim on a process-wide
connection pool, and then — several steps later — starts background worker tasks.
When a step in between raised, the half-built facade was dropped on the floor with
both still held, and nobody could release them: ``create()`` never returned, so no
caller had an object to call ``close()`` on.

Neither symptom shows up where you would look for it:

- **The refcount.** ``DatabaseManager.close`` only closes the client when the last
  holder leaves. One leaked claim means a *legitimate* ``close()`` later — from a
  facade that started fine — decrements to a non-zero count and returns without
  closing anything. The pool outlives the process's use of it, and the code path
  that appears to be at fault did nothing wrong.
- **The workers.** Started at step 6, so a failure at step 7 leaves four tasks
  polling Atlas through a connection the caller believes never opened.

The sync ``Memory`` wrapper has the same shape one level up: it starts an event
loop on a thread before building the core, so a failed ``__init__`` leaves the
thread alive with no object to close it through.

These tests run the *real* ``DatabaseManager`` over a fake client rather than
mocking ``initialize``. Mocking it would hide the finding entirely — the leak is
the refcount, and a mocked initialize has none. Each assertion is on released
state, not on a cleanup method having been called, so an implementation that
calls it in the wrong order or swallows its effect still fails.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.core.database import DatabaseManager
from agent_memory.memory import AsyncMemory


def _config(**overrides) -> MemoryConfig:
    defaults = {
        "mongodb_connection_string": "mongodb://localhost:27017",
        "governance_enabled": False,
        "rate_limit_enabled": False,
        "workers_in_process": False,
    }
    defaults.update(overrides)
    # `_env_file=None`: a live .env in the working tree would otherwise supply a
    # real cluster and real credentials to every config built here.
    return MemoryConfig(**defaults, _env_file=None)


class _FakeClient:
    """Stands in for ``AsyncMongoClient`` so the real refcount logic runs."""

    def __init__(self):
        self.closed = False
        self.admin = MagicMock()
        self.admin.command = AsyncMock(return_value={"ok": 1})

    def __getitem__(self, name):
        return MagicMock()

    async def close(self):
        self.closed = True


@pytest.fixture
def clients(monkeypatch):
    """Every client the pool constructs, with the singleton reset around the test.

    ``_instance`` is class-level state. Leaving one behind would make the next
    test's ``initialize`` hand back this pool and increment it, which reads as a
    passing refcount assertion for the wrong reason.
    """
    import agent_memory.core.database as dbmod

    created: list[_FakeClient] = []

    def _factory(*a, **k):
        client = _FakeClient()
        created.append(client)
        return client

    monkeypatch.setattr(dbmod, "AsyncMongoClient", _factory)
    DatabaseManager._instance = None
    try:
        yield created
    finally:
        DatabaseManager._instance = None


async def _idle():
    """A worker loop that neither finishes nor touches the database."""
    await asyncio.sleep(3600)


def _wire(monkeypatch, *, fail_at=None, exc=None):
    """Wire ``create()``'s collaborators, optionally failing one named step.

    ``fail_at`` is ``"indexes"`` (step 2), ``"providers"`` (step 3) or ``"search"``
    (step 7) — one before the workers start and one after, so both halves of the
    cleanup are exercised by a real failure rather than a synthetic one.
    """
    import agent_memory.core.migrations as mig
    import agent_memory.memory as mem
    import agent_memory.providers.manager as pm
    import agent_memory.services.audit_flush_worker as afw
    import agent_memory.services.consolidation as cons
    import agent_memory.services.enrichment as enr
    from agent_memory.services.episodic_worker import EpisodicWorker

    error = exc if exc is not None else RuntimeError("step failed")

    monkeypatch.setattr(
        mig,
        "ensure_indexes",
        AsyncMock(side_effect=error if fail_at == "indexes" else None),
    )
    monkeypatch.setattr(
        mig, "find_stranding_dimension_changes", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(mig, "ensure_search_indexes", AsyncMock())

    def _providers(config):
        from agent_memory.providers.manager import resolve_embedding

        if fail_at == "providers":
            raise error
        p = MagicMock()
        p.embedding = AsyncMock()
        p.embedding_spec = resolve_embedding(config)
        p.scorer = MagicMock()
        return p

    monkeypatch.setattr(pm, "ProviderManager", _providers)

    # The worker loops are replaced with an idle sleep. The real ones would query
    # a MagicMock database and die on the first iteration, which would make
    # "were the workers cancelled?" unanswerable — a dead task is also a
    # cancelled-looking one.
    monkeypatch.setattr(enr.EnrichmentWorker, "run", lambda self: _idle())
    monkeypatch.setattr(cons.ConsolidationWorker, "run", lambda self: _idle())
    monkeypatch.setattr(afw.AuditFlushWorker, "run", lambda self: _idle())
    monkeypatch.setattr(EpisodicWorker, "run", lambda self: _idle())

    if fail_at == "search":
        async def _boom(self, db, config, ensure=None):
            raise error

        monkeypatch.setattr(AsyncMemory, "_provision_search_indexes", _boom)
    else:
        monkeypatch.setattr(mem, "_ensure_search_indexes_bg",
                            lambda *a, **k: _idle())


class TestTheDatabasePoolIsReleased:
    """The refcount must come back down, whichever step failed."""

    @pytest.mark.parametrize("fail_at", ["indexes", "providers", "search"])
    async def test_a_failure_releases_the_pool(self, monkeypatch, clients, fail_at):
        _wire(monkeypatch, fail_at=fail_at)

        with pytest.raises(RuntimeError):
            await AsyncMemory.create(_config())

        # The client is actually shut, not merely decremented: the last holder left.
        assert DatabaseManager._instance is None
        assert len(clients) == 1
        assert clients[0].closed is True

    async def test_a_later_legitimate_close_still_closes_the_client(
        self, monkeypatch, clients
    ):
        # The finding stated as the failure an operator would actually hit. Holder
        # A starts fine; a second startup fails; A's close() must still close the
        # client. With a leaked claim the refcount goes 1→2 on the failed attempt,
        # so A's close decrements 2→1 and returns, and the pool stays open for the
        # life of the process with nobody left who can release it.
        _wire(monkeypatch)
        holder = await AsyncMemory.create(_config())

        _wire(monkeypatch, fail_at="indexes")
        with pytest.raises(RuntimeError):
            await AsyncMemory.create(_config())

        assert clients[0].closed is False, "the live holder's pool was closed"
        await holder.close()
        assert clients[0].closed is True

    async def test_a_retry_after_a_failure_is_not_refused_as_a_different_target(
        self, monkeypatch, clients
    ):
        # `initialize` refuses a config pointing somewhere else, by design. A leaked
        # claim leaves a pool nobody holds to be compared against, so a process that
        # fixes its config and retries gets "already connected to a different
        # MongoDB target" — a message about a pool that has no owner.
        _wire(monkeypatch, fail_at="indexes")
        with pytest.raises(RuntimeError):
            await AsyncMemory.create(_config(mongodb_database_name="wrong_db"))

        _wire(monkeypatch)
        memory = await AsyncMemory.create(_config(mongodb_database_name="right_db"))
        try:
            assert memory._db_manager is DatabaseManager._instance
        finally:
            await memory.close()

    async def test_a_successful_startup_keeps_its_pool(self, monkeypatch, clients):
        # The paired case. Cleanup that ran unconditionally would satisfy every
        # test above and leave the facade holding a closed client.
        _wire(monkeypatch)
        memory = await AsyncMemory.create(_config())
        try:
            assert DatabaseManager._instance is not None
            assert clients[0].closed is False
            assert memory._db_manager is not None
        finally:
            await memory.close()
        assert clients[0].closed is True

    async def test_a_cancellation_also_releases_the_pool(self, monkeypatch, clients):
        # `except Exception` would miss this, and it is the case where cleanup
        # matters most: a caller whose timeout expired is by definition not around
        # to tidy up after the startup it abandoned.
        _wire(monkeypatch, fail_at="indexes", exc=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await AsyncMemory.create(_config())
        assert DatabaseManager._instance is None
        assert clients[0].closed is True

    async def test_the_startup_error_still_reaches_the_caller(
        self, monkeypatch, clients
    ):
        # Cleanup must not become the reported failure. The startup fault is the
        # only thing the caller can act on.
        _wire(monkeypatch, fail_at="indexes", exc=RuntimeError("the real reason"))
        with pytest.raises(RuntimeError, match="the real reason"):
            await AsyncMemory.create(_config())

    async def test_a_cleanup_failure_does_not_mask_the_startup_error(
        self, monkeypatch, clients
    ):
        # A driver error during teardown — the pool's own close() raising. Cleanup
        # swallowing it is the point: replacing "why startup failed" with
        # "teardown hit a reset connection" sends the operator after the wrong bug.
        _wire(monkeypatch, fail_at="search", exc=RuntimeError("the real reason"))

        async def _bad_close(self):
            raise OSError("connection reset during teardown")

        monkeypatch.setattr(DatabaseManager, "close", _bad_close)
        with pytest.raises(RuntimeError, match="the real reason"):
            await AsyncMemory.create(_config())


class TestTheWorkersAreStopped:
    """Tasks started at step 6 must not outlive a failure at step 7."""

    async def test_a_later_failure_cancels_the_workers(self, monkeypatch, clients):
        _wire(monkeypatch, fail_at="search")

        started: list[asyncio.Task] = []
        original = AsyncMemory._supervise

        def _record(self, coro, name):
            task = original(self, coro, name)
            started.append(task)
            return task

        monkeypatch.setattr(AsyncMemory, "_supervise", _record)

        with pytest.raises(RuntimeError):
            await AsyncMemory.create(_config(workers_in_process=True))

        assert started, "workers should have started before the failing step"
        await asyncio.sleep(0)  # let the cancellations be delivered
        for task in started:
            assert task.cancelled() or task.done(), (
                f"{task.get_name()} is still running after a failed startup"
            )

    async def test_a_successful_startup_leaves_its_workers_running(
        self, monkeypatch, clients
    ):
        # Paired: cancelling unconditionally would satisfy the test above while
        # silently disabling every background worker on a healthy startup.
        _wire(monkeypatch)
        memory = await AsyncMemory.create(_config(workers_in_process=True))
        try:
            await asyncio.sleep(0)
            assert len(memory._workers) == 4
            assert all(not task.done() for task in memory._workers)
        finally:
            await memory.close()

    async def test_the_episodic_queue_stops_accepting_turns(
        self, monkeypatch, clients
    ):
        # The episodic consumer is one of the four cancelled tasks, so anything
        # still holding a reference to the abandoned facade would enqueue into a
        # queue nothing will ever drain.
        _wire(monkeypatch, fail_at="search")
        captured: dict = {}

        original = AsyncMemory._maybe_start_workers

        async def _capture(self):
            await original(self)
            captured["episodic"] = self.episodic_service.worker

        monkeypatch.setattr(AsyncMemory, "_maybe_start_workers", _capture)

        with pytest.raises(RuntimeError):
            await AsyncMemory.create(_config(workers_in_process=True))

        assert captured["episodic"]._closed is True

    async def test_a_backgrounded_search_index_task_is_cancelled(
        self, monkeypatch, clients
    ):
        # With `await_search_indexes=False` (the default) step 7 schedules a task
        # instead of awaiting one, so a failure after it leaves that task building
        # indexes for a facade nobody holds.
        import agent_memory.memory as mem

        monkeypatch.setattr(mem, "_ensure_search_indexes_bg",
                            lambda *a, **k: _idle())

        facade = AsyncMemory.__new__(AsyncMemory)
        config = _config()
        facade.config = config
        facade._workers = []
        facade._db_manager = None
        await facade._provision_search_indexes(
            MagicMock(), config, ensure=AsyncMock()
        )
        task = facade._search_index_task
        assert task is not None and not task.done()

        await facade._abandon_startup()
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()


class TestTheSyncWrapperReleasesItsThread:
    """``Memory.__init__`` starts a loop thread before it can fail."""

    @staticmethod
    def _loop_threads() -> list[threading.Thread]:
        return [t for t in threading.enumerate() if t.name == "agent-memory-loop"]

    def test_a_failed_construction_stops_the_loop_thread(self, monkeypatch):
        import agent_memory.memory as mem
        from agent_memory.sync import Memory

        monkeypatch.setattr(
            mem.AsyncMemory, "create",
            AsyncMock(side_effect=RuntimeError("startup failed")),
        )
        before = len(self._loop_threads())

        with pytest.raises(RuntimeError, match="startup failed"):
            Memory(_config())

        # The thread is a daemon, so the process still exits — but `__init__`
        # raised, so there is no object to close, and a caller that retries (a
        # script looping over configs, a parametrized test) accumulates one live
        # loop and one thread per attempt.
        assert len(self._loop_threads()) == before, (
            "the background loop thread outlived a failed __init__"
        )

    def test_an_interrupted_construction_stops_the_loop_thread(self, monkeypatch):
        # `except Exception` would miss this, and here it is not a corner case:
        # `__init__` blocks the calling thread in `future.result()` while the core
        # builds, so a Ctrl-C during a slow startup — a wrong connection string
        # waiting out the 5s server-selection timeout — arrives as
        # KeyboardInterrupt at exactly that line. A script interrupted there would
        # otherwise be left with a live loop thread and no handle on it.
        from agent_memory.sync import Memory

        def _interrupted(self, coro):
            # The interrupt is raised where it really arrives — in the calling
            # thread, at the blocking wait — not inside the coroutine. Raising it
            # on the background loop instead would hang the test: asyncio lets
            # KeyboardInterrupt escape `run_forever` rather than setting it on the
            # future, so `future.result()` would never return.
            coro.close()
            raise KeyboardInterrupt

        monkeypatch.setattr(Memory, "_submit", _interrupted)
        before = len(self._loop_threads())

        with pytest.raises(KeyboardInterrupt):
            Memory(_config())

        assert len(self._loop_threads()) == before, (
            "the background loop thread outlived an interrupted __init__"
        )

    def test_a_successful_construction_keeps_its_loop(self, monkeypatch):
        import agent_memory.memory as mem
        from agent_memory.sync import Memory

        core = AsyncMock()
        core.close = AsyncMock()
        monkeypatch.setattr(mem.AsyncMemory, "create", AsyncMock(return_value=core))

        memory = Memory(_config())
        try:
            assert memory._thread.is_alive()
            assert not memory._loop.is_closed()
        finally:
            memory.close()
        assert memory._loop.is_closed()

    def test_close_is_idempotent(self, monkeypatch):
        # `_stop_loop` is shared with the failure path, so it has to tolerate an
        # already-stopped loop. Without that, a second close() — or a `close()`
        # inside `__exit__` after an explicit one — raises RuntimeError from
        # `call_soon_threadsafe` on a closed loop, turning clean teardown into an
        # error the caller has no way to avoid.
        import agent_memory.memory as mem
        from agent_memory.sync import Memory

        core = AsyncMock()
        core.close = AsyncMock()
        monkeypatch.setattr(mem.AsyncMemory, "create", AsyncMock(return_value=core))

        memory = Memory(_config())
        memory.close()
        memory.close()
        assert memory._loop.is_closed()

    def test_the_core_is_closed_before_the_loop_stops(self, monkeypatch):
        # Ordering, not just presence: the core's own teardown is async and runs
        # *on* the background loop, so stopping the loop first would leave
        # `AsyncMemory.close()` unable to run at all — the pool would leak from
        # the tidy path this fix is meant to complete.
        import agent_memory.memory as mem
        from agent_memory.sync import Memory

        order: list[str] = []
        core = AsyncMock()

        async def _core_close():
            order.append("core")

        core.close = _core_close
        monkeypatch.setattr(mem.AsyncMemory, "create", AsyncMock(return_value=core))

        memory = Memory(_config())
        original = memory._stop_loop

        def _tracked():
            order.append("loop")
            original()

        memory._stop_loop = _tracked
        memory.close()
        assert order == ["core", "loop"]
