"""Tests for the episodic background writer. REQ-E-100..107."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import PyMongoError

from agent_memory.config import MemoryConfig
from agent_memory.services.episodic_worker import EpisodicWorker


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


def _counters(seq: int = 1):
    counters = MagicMock()
    counters.find_one_and_update = AsyncMock(return_value={"_id": "t1", "seq": seq})
    return counters


def _providers(embedding=None):
    providers = MagicMock()
    providers.embedding = MagicMock()
    providers.embedding.generate_embedding = AsyncMock(
        return_value=embedding if embedding is not None else [0.1, 0.2]
    )
    return providers


def _worker(collection=None, *, counters=None, providers=None, **config_overrides):
    collection = collection if collection is not None else _collection()
    return EpisodicWorker(
        collection,
        counters if counters is not None else _counters(),
        providers if providers is not None else _providers(),
        _config(**config_overrides),
    )


def _collection():
    col = MagicMock()
    col.insert_one = AsyncMock()
    col.insert_many = AsyncMock()
    return col


async def _run_until_drained(worker, timeout=2.0):
    """Start the consumer, flush, then stop it — the standard test cycle."""
    task = asyncio.create_task(worker.run())
    try:
        drained = await worker.flush(timeout)
    finally:
        worker.stop()
        await asyncio.wait_for(task, timeout=timeout)
    return drained


class TestEnqueue:
    """REQ-E-100: the caller never waits."""

    async def test_enqueue_does_not_await_the_database(self):
        # TC-EP-WRK-001: a slow insert must not be visible to the caller.
        col = _collection()

        async def slow_insert(*args, **kwargs):
            await asyncio.sleep(0.5)

        col.insert_one = AsyncMock(side_effect=slow_insert)
        worker = _worker(col)

        loop = asyncio.get_running_loop()
        started = loop.time()
        worker.enqueue({"user_id": "u1"})
        assert loop.time() - started < 0.2

        task = asyncio.create_task(worker.run())
        await worker.flush(2.0)
        worker.stop()
        await asyncio.wait_for(task, timeout=2.0)

    async def test_enqueued_documents_are_written(self):
        # TC-EP-WRK-002
        col = _collection()
        worker = _worker(col)
        worker.enqueue({"user_id": "u1", "thread_id": "t1"})

        assert await _run_until_drained(worker) is True
        assert col.insert_one.await_count == 1
        assert worker.stats()["written"] == 1

    async def test_enqueue_after_close_is_discarded(self):
        # TC-EP-WRK-003
        worker = _worker()
        await worker.close(0.1)
        worker.enqueue({"user_id": "u1"})
        assert worker.stats()["enqueued"] == 0

    async def test_inflight_is_incremented_before_the_put(self):
        # TC-EP-WRK-004: reversed, the consumer's decrement is swallowed by the
        # ``> 0`` guard and flush() can never observe zero again.
        worker = _worker()
        worker.enqueue({"user_id": "u1"})
        assert worker._inflight == 1
        assert await _run_until_drained(worker) is True
        assert worker._inflight == 0


class TestBackpressure:
    """REQ-E-102: drop the oldest, keep the newest."""

    async def test_a_full_queue_drops_the_oldest(self):
        # TC-EP-WRK-010
        col = _collection()
        worker = _worker(col, episodic_queue_size=2)
        for i in range(4):
            worker.enqueue({"user_id": "u1", "n": i})

        assert worker.stats()["dropped"] == 2
        await _run_until_drained(worker)

        written = [
            doc["n"]
            for call in col.insert_many.await_args_list
            for doc in call.args[0]
        ] + [call.args[0]["n"] for call in col.insert_one.await_args_list]
        # The newest turn always survives; the oldest are what got dropped.
        assert 3 in written
        assert 0 not in written

    async def test_queue_capacity_is_reported(self):
        # TC-EP-WRK-011
        worker = _worker(episodic_queue_size=7)
        assert worker.stats()["queue_capacity"] == 7

    async def test_queue_size_of_zero_is_clamped(self):
        # TC-EP-WRK-012: a zero-maxsize asyncio.Queue is unbounded, which would
        # silently remove all backpressure.
        worker = _worker(episodic_queue_size=0)
        assert worker.stats()["queue_capacity"] == 1


class TestBatching:
    async def test_multiple_docs_use_insert_many(self):
        # TC-EP-WRK-020
        col = _collection()
        worker = _worker(col, episodic_batch_size=10)
        for i in range(3):
            worker.enqueue({"user_id": "u1", "n": i})

        await _run_until_drained(worker)
        assert col.insert_many.await_count == 1
        assert len(col.insert_many.await_args.args[0]) == 3
        assert col.insert_many.await_args.kwargs["ordered"] is False

    async def test_batch_size_bounds_a_single_insert(self):
        # TC-EP-WRK-021
        col = _collection()
        worker = _worker(col, episodic_batch_size=2)
        for i in range(4):
            worker.enqueue({"user_id": "u1", "n": i})

        await _run_until_drained(worker)
        assert all(
            len(call.args[0]) <= 2 for call in col.insert_many.await_args_list
        )
        assert worker.stats()["written"] == 4

    async def test_fifo_order_is_preserved(self):
        # TC-EP-WRK-022: one consumer, so step stays monotonic per thread.
        col = _collection()
        worker = _worker(col, episodic_batch_size=10)
        for i in range(5):
            worker.enqueue({"user_id": "u1", "n": i})

        await _run_until_drained(worker)
        assert [d["n"] for d in col.insert_many.await_args.args[0]] == [0, 1, 2, 3, 4]


class TestDurableStep:
    """REQ-E-103/104: monotonic step, and never lose a turn over it."""

    async def test_first_step_is_zero_with_no_parent(self):
        # TC-EP-WRK-030
        col = _collection()
        worker = _worker(col, counters=_counters(seq=1))
        worker.enqueue({"user_id": "u1", "__assign_step": "t1"})

        await _run_until_drained(worker)
        doc = col.insert_one.await_args.args[0]
        assert doc["step"] == 0
        assert doc["parent_step"] is None

    async def test_later_steps_reference_their_parent(self):
        # TC-EP-WRK-031
        col = _collection()
        worker = _worker(col, counters=_counters(seq=8))
        worker.enqueue({"user_id": "u1", "__assign_step": "t1"})

        await _run_until_drained(worker)
        doc = col.insert_one.await_args.args[0]
        assert doc["step"] == 7
        assert doc["parent_step"] == 6

    async def test_the_internal_key_is_stripped(self):
        # TC-EP-WRK-032
        col = _collection()
        worker = _worker(col)
        worker.enqueue({"user_id": "u1", "__assign_step": "t1"})

        await _run_until_drained(worker)
        assert "__assign_step" not in col.insert_one.await_args.args[0]

    async def test_a_counter_failure_still_inserts(self):
        # TC-EP-WRK-033: a logged turn beats a lost one.
        col = _collection()
        counters = MagicMock()
        counters.find_one_and_update = AsyncMock(side_effect=PyMongoError("down"))
        worker = _worker(col, counters=counters)
        worker.enqueue({"user_id": "u1", "__assign_step": "t1"})

        await _run_until_drained(worker)
        doc = col.insert_one.await_args.args[0]
        assert doc["step"] is None
        assert doc["parent_step"] is None
        assert col.insert_one.await_count == 1

    async def test_a_malformed_counter_response_still_inserts(self):
        # TC-EP-WRK-034
        col = _collection()
        counters = MagicMock()
        counters.find_one_and_update = AsyncMock(return_value={"_id": "t1"})
        worker = _worker(col, counters=counters)
        worker.enqueue({"user_id": "u1", "__assign_step": "t1"})

        await _run_until_drained(worker)
        assert col.insert_one.await_args.args[0]["step"] is None

    async def test_documents_without_the_key_are_untouched(self):
        # TC-EP-WRK-035
        col = _collection()
        counters = _counters()
        worker = _worker(col, counters=counters)
        worker.enqueue({"user_id": "u1", "step": 42})

        await _run_until_drained(worker)
        counters.find_one_and_update.assert_not_awaited()
        assert col.insert_one.await_args.args[0]["step"] == 42


class TestEmbedding:
    """REQ-E-105: embed first, then assign both fields."""

    async def test_search_text_and_embedding_are_both_written(self):
        # TC-EP-WRK-040
        col = _collection()
        worker = _worker(col, providers=_providers([0.5] * 4))
        worker.enqueue({"user_id": "u1", "__search_text": "q\n\na"})

        await _run_until_drained(worker)
        doc = col.insert_one.await_args.args[0]
        assert doc["embedding"] == [0.5] * 4
        assert doc["search_text"] == "q\n\na"
        assert "__search_text" not in doc

    async def test_an_embedding_failure_leaves_neither_field(self):
        # TC-EP-WRK-041: text without a vector would rank inconsistently
        # between the two branches of hybrid recall.
        col = _collection()
        providers = MagicMock()
        providers.embedding = MagicMock()
        providers.embedding.generate_embedding = AsyncMock(
            side_effect=RuntimeError("provider down")
        )
        worker = _worker(col, providers=providers)
        worker.enqueue({"user_id": "u1", "__search_text": "q\n\na"})

        await _run_until_drained(worker)
        doc = col.insert_one.await_args.args[0]
        assert "embedding" not in doc
        assert "search_text" not in doc
        assert worker.stats()["embed_failures"] == 1
        # The turn itself is still stored.
        assert col.insert_one.await_count == 1

    async def test_no_search_text_means_no_embedding_call(self):
        # TC-EP-WRK-042
        providers = _providers()
        worker = _worker(providers=providers)
        worker.enqueue({"user_id": "u1"})

        await _run_until_drained(worker)
        providers.embedding.generate_embedding.assert_not_awaited()


class TestFailureCounting:
    """REQ-E-106: failures are counters, not exceptions."""

    async def test_an_insert_failure_is_counted_not_raised(self):
        # TC-EP-WRK-050
        col = _collection()
        col.insert_one = AsyncMock(side_effect=PyMongoError("write failed"))
        worker = _worker(col)
        worker.enqueue({"user_id": "u1"})

        assert await _run_until_drained(worker) is True
        assert worker.stats()["write_failures"] == 1
        assert worker.stats()["written"] == 0

    async def test_the_consumer_survives_a_failure(self):
        # TC-EP-WRK-051: one bad batch must not kill logging for the process.
        col = _collection()
        col.insert_one = AsyncMock(side_effect=[PyMongoError("boom"), None])
        worker = _worker(col)
        task = asyncio.create_task(worker.run())
        try:
            worker.enqueue({"user_id": "u1", "n": 0})
            await worker.flush(2.0)
            worker.enqueue({"user_id": "u1", "n": 1})
            await worker.flush(2.0)
        finally:
            worker.stop()
            await asyncio.wait_for(task, timeout=2.0)

        stats = worker.stats()
        assert stats["write_failures"] == 1
        assert stats["written"] == 1


class TestLifecycle:
    """REQ-E-107: bounded flush, idempotent close, neither raises."""

    async def test_flush_returns_true_when_idle(self):
        # TC-EP-WRK-060
        assert await _worker().flush(0.1) is True

    async def test_flush_returns_false_when_nothing_consumes(self):
        # TC-EP-WRK-061: bounded — it must not hang when the worker is stalled.
        worker = _worker()
        worker.enqueue({"user_id": "u1"})
        assert await worker.flush(0.1) is False

    async def test_flush_returns_true_once_drained(self):
        # TC-EP-WRK-062: the False-then-True sequence.
        worker = _worker()
        worker.enqueue({"user_id": "u1"})
        assert await worker.flush(0.05) is False
        assert await _run_until_drained(worker) is True

    async def test_close_is_idempotent(self):
        # TC-EP-WRK-063
        worker = _worker()
        assert await worker.close(0.5) is True
        assert await worker.close(0.5) is True

    async def test_close_stops_the_consumer(self):
        # TC-EP-WRK-064
        worker = _worker()
        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0)
        await worker.close(1.0)
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()

    async def test_close_drains_pending_turns(self):
        # TC-EP-WRK-065
        col = _collection()
        worker = _worker(col)
        task = asyncio.create_task(worker.run())
        worker.enqueue({"user_id": "u1"})
        assert await worker.close(2.0) is True
        await asyncio.wait_for(task, timeout=2.0)
        assert col.insert_one.await_count == 1

    async def test_a_shutdown_sentinel_mid_batch_writes_first(self):
        # TC-EP-WRK-066: pending turns are not sacrificed to shutdown.
        col = _collection()
        worker = _worker(col, episodic_batch_size=10)
        worker.enqueue({"user_id": "u1", "n": 0})
        worker.stop()
        worker._closed = False
        worker.enqueue({"user_id": "u1", "n": 1})
        await asyncio.wait_for(asyncio.create_task(worker.run()), timeout=2.0)
        assert worker.stats()["written"] >= 1

    async def test_neither_flush_nor_close_raises_on_a_broken_collection(self):
        # TC-EP-WRK-067
        col = _collection()
        col.insert_one = AsyncMock(side_effect=RuntimeError("gone"))
        worker = _worker(col)
        task = asyncio.create_task(worker.run())
        worker.enqueue({"user_id": "u1"})
        assert await worker.flush(2.0) is True
        assert await worker.close(2.0) is True
        await asyncio.wait_for(task, timeout=2.0)


class TestStats:
    """REQ-E-106: what a /health probe reads."""

    def test_all_expected_keys_are_present(self):
        # TC-EP-WRK-070
        assert set(_worker().stats()) == {
            "queue_depth",
            "queue_capacity",
            "worker_alive",
            "enqueued",
            "written",
            "dropped",
            "batches",
            "embed_failures",
            "write_failures",
            "last_write_ts",
        }

    def test_initial_counters_are_zero(self):
        # TC-EP-WRK-071
        stats = _worker().stats()
        assert stats["enqueued"] == 0
        assert stats["written"] == 0
        assert stats["dropped"] == 0
        assert stats["last_write_ts"] is None
        assert stats["worker_alive"] is False

    async def test_last_write_ts_is_an_isoformat_string(self):
        # TC-EP-WRK-072: a probe serializes this to JSON.
        worker = _worker()
        worker.enqueue({"user_id": "u1"})
        await _run_until_drained(worker)

        last = worker.stats()["last_write_ts"]
        assert isinstance(last, str)
        from datetime import datetime

        assert datetime.fromisoformat(last)

    async def test_worker_alive_reflects_the_consumer(self):
        # TC-EP-WRK-073
        worker = _worker()
        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0)
        assert worker.stats()["worker_alive"] is True
        worker.stop()
        await asyncio.wait_for(task, timeout=1.0)
        assert worker.stats()["worker_alive"] is False

    async def test_batches_counts_flushes_not_documents(self):
        # TC-EP-WRK-074
        worker = _worker(episodic_batch_size=10)
        for i in range(3):
            worker.enqueue({"user_id": "u1", "n": i})
        await _run_until_drained(worker)

        stats = worker.stats()
        assert stats["written"] == 3
        assert stats["batches"] == 1
