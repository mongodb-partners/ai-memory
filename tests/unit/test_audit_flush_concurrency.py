"""Concurrent flushes must not each start their own write.

``AuditService.flush`` is reachable from five places at once: every audited
operation (``log`` flushes when the buffer fills or the interval elapses),
``AuditFlushWorker`` on its timer, ``wipe_user_data`` before the delete,
``close()``, and any caller who flushes explicitly. It had no mutual exclusion.

The buffer *swap* was already safe — there is no ``await`` between the copy and
the reset, so no entry was ever written twice or lost to an interleaving. The
damage was elsewhere:

* **``flush()`` returned while entries were still in flight.** It swapped the
  buffer, saw nothing left, and returned — even though another flush's
  ``insert_many`` had not completed. ``wipe_user_data`` flushes *before* deleting
  for exactly one reason: so that no buffered row naming the user survives the
  wipe. A flush that returns early lets that row land after the delete, which
  undoes the erasure through a different door than the episodic queue did.
* **One write per caller.** N concurrent flushes produced N concurrent
  ``insert_many`` calls, each carrying a fraction of the batch. Under load — where
  the interval has always elapsed — that is one round trip per audited request,
  which is the cost the buffer exists to avoid.
* **A cancelled flush discarded its batch.** ``close()`` cancels the worker
  tasks; a cancellation mid-``insert_many`` left the entries in neither the
  buffer, MongoDB, nor the fallback file.

Each is pinned in both directions: a lock that serialised the *empty* case, or
one that made a flush swallow cancellation, would satisfy half of this and cost
more than it fixed.
"""

import asyncio
import logging
from unittest.mock import AsyncMock

from agent_memory.core.config import MCPConfig
from agent_memory.services.audit import AuditService


def _config(**overrides) -> MCPConfig:
    # `_env_file=None`: a live .env in the working tree would otherwise decide
    # the buffer size and interval these tests depend on.
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


class _Collection:
    """A collection whose ``insert_many`` can be held open on demand.

    ``AsyncMock`` cannot express "started but not finished", which is the only
    state any of this is about.
    """

    def __init__(self, *, hold: bool = True) -> None:
        self.gate = asyncio.Event()
        self.started = asyncio.Event()
        self.concurrent = 0
        self.max_concurrent = 0
        self.batches: list[list[dict]] = []
        self.landed: list[str] = []
        self._hold = hold
        if not hold:
            self.gate.set()

    async def insert_many(self, batch) -> None:
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.started.set()
        self.batches.append(list(batch))
        try:
            # An unconditional yield, even when the gate is already open. A real
            # `insert_many` suspends on the network; without this the whole flush
            # runs without ever handing control back, and every test here would
            # be measuring a scheduler artefact rather than the lock.
            await asyncio.sleep(0)
            await self.gate.wait()
            self.landed.extend(e["user_id"] for e in batch)
        finally:
            self.concurrent -= 1

    def release(self) -> None:
        self.gate.set()


def _service(collection, **overrides) -> AuditService:
    overrides.setdefault("audit_buffer_size", 1000)
    overrides.setdefault("audit_flush_on_write", False)
    return AuditService(collection, _config(**overrides))


class TestOnlyOneWriteIsInFlight:
    """The buffer exists to turn many records into few round trips. Concurrent
    flushes were undoing that."""

    async def test_concurrent_flushes_do_not_overlap(self):
        collection = _Collection()
        service = _service(collection)
        for i in range(6):
            await service.log(f"u{i}", "memory:write", "store_memory", "success", 1)

        flushes = [asyncio.create_task(service.flush()) for _ in range(6)]
        await collection.started.wait()
        await asyncio.sleep(0)  # let every other flush reach its await
        assert collection.max_concurrent == 1

        collection.release()
        await asyncio.gather(*flushes)
        # One write carrying all six, not six carrying one each.
        assert len(collection.batches) == 1
        assert len(collection.batches[0]) == 6

    async def test_a_log_storm_does_not_become_one_write_per_call(self):
        """`audit_flush_interval_seconds=0` is the steady state under load: the
        timer has always elapsed, so every `log()` flushes."""
        collection = _Collection(hold=False)
        service = _service(collection, audit_flush_interval_seconds=0)

        await asyncio.gather(*[
            service.log(f"u{i}", "memory:write", "store_memory", "success", 1)
            for i in range(20)
        ])
        assert collection.max_concurrent == 1
        # Not 20. The exact count depends on scheduling; what matters is that
        # concurrent callers coalesce rather than each paying a round trip.
        assert len(collection.batches) < 20
        assert sorted(collection.landed) == sorted(f"u{i}" for i in range(20))

    async def test_a_waiting_flush_does_not_write_an_empty_batch(self):
        """The re-check under the lock. Without it, the flush that was queued
        behind another wakes to an empty buffer and inserts nothing — a round trip
        for no records."""
        collection = _Collection()
        service = _service(collection)
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        first = asyncio.create_task(service.flush())
        await collection.started.wait()
        second = asyncio.create_task(service.flush())
        await asyncio.sleep(0)
        collection.release()
        await asyncio.gather(first, second)

        assert len(collection.batches) == 1

    async def test_an_idle_flush_costs_nothing(self):
        """The paired direction. `AuditFlushWorker` calls this on every interval
        for the whole life of a quiet process, so an empty buffer must not produce
        a write — and, since an uncontended `asyncio.Lock` needs no suspension, it
        does not need a pre-lock shortcut to be cheap."""
        collection = _Collection(hold=False)
        service = _service(collection)

        await asyncio.wait_for(service.flush(), timeout=0.5)
        assert collection.batches == []
        assert not service._flush_lock.locked()

    async def test_an_empty_buffer_still_waits_for_an_outstanding_write(self):
        """"Nothing buffered" is not "nothing outstanding", which is why there is
        no early return before the lock: a concurrent flush may be holding this
        batch in an in-flight insert, and that is exactly the case
        `wipe_user_data` must not race."""
        collection = _Collection()
        service = _service(collection)
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        holder = asyncio.create_task(service.flush())
        await collection.started.wait()
        assert service._buffer == []

        waiter = asyncio.create_task(service.flush())
        await asyncio.sleep(0)
        assert not waiter.done()

        collection.release()
        await asyncio.gather(holder, waiter)
        assert collection.landed == ["alice"]


class TestFlushMeansTheEntriesHaveLanded:
    """`wipe_user_data` flushes before deleting so that no buffered row naming the
    user outlives the wipe. That only works if `flush()` waits."""

    async def test_flush_waits_for_an_in_flight_write(self):
        collection = _Collection()
        service = _service(collection)
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        first = asyncio.create_task(service.flush())
        await collection.started.wait()

        # The erasure's pre-delete flush. Before the fix this returned
        # immediately, with alice's row still in flight — so the delete ran, and
        # *then* the row landed.
        second = asyncio.create_task(service.flush())
        await asyncio.sleep(0)
        assert not second.done(), "flush() returned while a write was in flight"

        collection.release()
        await asyncio.gather(first, second)
        # By the time the erasure's flush returns, the row it was waiting for is
        # in the collection, where the delete that follows will remove it.
        assert collection.landed == ["alice"]

    async def test_an_erasure_flush_sees_a_concurrently_started_batch(self):
        """The end-to-end shape of the defect, without the facade: the entry is
        taken out of the buffer by the worker's flush, so the erasure's flush finds
        nothing to write — and must still not return until that write is done."""
        collection = _Collection()
        service = _service(collection)
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        worker_flush = asyncio.create_task(service.flush())
        await collection.started.wait()
        assert service._buffer == []  # nothing left to write

        erasure_flush = asyncio.create_task(service.flush())
        await asyncio.sleep(0)
        assert not erasure_flush.done()
        assert collection.landed == []

        collection.release()
        await asyncio.gather(worker_flush, erasure_flush)
        assert collection.landed == ["alice"]


class TestCancellationDoesNotDiscardTheBatch:
    """`close()` cancels the worker tasks and then flushes. A batch cancelled
    mid-write was in neither the buffer, MongoDB, nor the fallback file."""

    async def test_a_cancelled_flush_returns_its_batch_to_the_buffer(self):
        collection = _Collection()
        service = _service(collection)
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        flush = asyncio.create_task(service.flush())
        await collection.started.wait()
        flush.cancel()
        with_cancel = False
        try:
            await flush
        except asyncio.CancelledError:
            with_cancel = True

        assert with_cancel, "cancellation must propagate, not be swallowed"
        assert [e["user_id"] for e in service._buffer] == ["alice"]

    async def test_the_next_flush_writes_the_recovered_batch(self):
        """This is what makes `close()` correct: cancel the workers, then flush."""
        collection = _Collection()
        service = _service(collection)
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        flush = asyncio.create_task(service.flush())
        await collection.started.wait()
        flush.cancel()
        try:
            await flush
        except asyncio.CancelledError:
            pass

        collection.release()
        await service.flush()
        assert collection.landed == ["alice"]

    async def test_the_recovered_batch_keeps_its_place_in_the_chronology(self):
        """Prepended, not appended. The fallback file and the collection are both
        read as a sequence, and these entries predate anything logged while the
        cancelled insert was in flight."""
        collection = _Collection()
        service = _service(collection)
        await service.log("first", "memory:write", "store_memory", "success", 1)

        flush = asyncio.create_task(service.flush())
        await collection.started.wait()
        await service.log("second", "memory:write", "store_memory", "success", 1)
        flush.cancel()
        try:
            await flush
        except asyncio.CancelledError:
            pass

        assert [e["user_id"] for e in service._buffer] == ["first", "second"]

    async def test_a_cancelled_batch_is_not_written_to_the_fallback_file(
        self, tmp_path
    ):
        """Cancellation is not a write failure. The batch's fate is unknown, so
        writing it to disk risks a record that exists twice — and a duplicated
        audit entry is a quieter lie than a missing one, because nothing about the
        collection says which of the two actually happened."""
        collection = _Collection()
        service = _service(
            collection, audit_fallback_path=str(tmp_path / "audit.jsonl")
        )
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        flush = asyncio.create_task(service.flush())
        await collection.started.wait()
        flush.cancel()
        try:
            await flush
        except asyncio.CancelledError:
            pass

        assert not (tmp_path / "audit.jsonl").exists()

    async def test_a_real_write_failure_still_reaches_the_fallback_file(
        self, tmp_path, caplog
    ):
        """The paired direction: only *cancellation* is treated as unknown. An
        ordinary failure must still put the batch on disk."""
        collection = AsyncMock()
        collection.insert_many.side_effect = RuntimeError("DB down")
        service = _service(
            collection, audit_fallback_path=str(tmp_path / "audit.jsonl")
        )
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        with caplog.at_level(logging.ERROR, logger="agent_memory.services.audit"):
            await service.flush()

        assert "alice" in (tmp_path / "audit.jsonl").read_text()
        assert service._buffer == []


class TestTheLockDoesNotChangeSingleCallerBehaviour:
    """Everything above is about contention. The uncontended path — which is every
    existing caller and every existing test — must be untouched."""

    async def test_a_lone_flush_writes_once_and_empties_the_buffer(self):
        collection = _Collection(hold=False)
        service = _service(collection)
        await service.log("alice", "memory:write", "store_memory", "success", 1)
        await service.flush()

        assert len(collection.batches) == 1
        assert service._buffer == []

    async def test_the_timer_is_reset_by_a_successful_flush(self):
        collection = _Collection(hold=False)
        service = _service(collection, audit_flush_interval_seconds=3600)
        service._last_flush = 0.0
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        # The elapsed interval triggered the flush, and the flush reset it — so
        # the next log() buffers rather than writing again.
        assert len(collection.batches) == 1
        await service.log("bob", "memory:write", "store_memory", "success", 1)
        assert len(collection.batches) == 1
        assert len(service._buffer) == 1

    async def test_flush_never_raises_on_a_write_failure(self):
        """Unchanged contract. The operation being audited has usually already
        succeeded; failing it because its record could not be written would make
        the audit log a source of outages."""
        collection = AsyncMock()
        collection.insert_many.side_effect = RuntimeError("DB down")
        service = _service(collection, audit_fallback_path="")
        await service.log("alice", "memory:write", "store_memory", "success", 1)

        await service.flush()  # must not raise
        assert service._buffer == []
