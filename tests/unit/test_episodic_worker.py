"""Tests for the episodic background writer. REQ-E-100..107."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

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
    """A provider stack whose batch call returns one copy of ``embedding`` per text.

    Only ``generate_embeddings_batch`` is stubbed as an ``AsyncMock``. The
    single-text ``generate_embedding`` is left as a plain ``MagicMock`` attribute
    on purpose: the worker must not call it, and if it ever does, awaiting a
    ``MagicMock`` result fails loudly rather than quietly returning a vector.
    """
    vector = embedding if embedding is not None else [0.1, 0.2]
    providers = MagicMock()
    providers.embedding = MagicMock()
    providers.embedding.generate_embeddings_batch = AsyncMock(
        side_effect=lambda texts: [list(vector) for _ in texts]
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
        providers.embedding.generate_embeddings_batch = AsyncMock(
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
        providers.embedding.generate_embeddings_batch.assert_not_awaited()


class TestAWrongWidthVectorIsAnEmbeddingFailure:
    """A vector Atlas would accept and never return counts as a failure.

    The width has to be checked here, because nothing downstream will object:
    Atlas stores a 1024-wide vector in a 1536-wide index, ``find`` returns the
    document, and ``$vectorSearch`` never does. The turn would read as logged and
    be unfindable — the one outcome ``_attach_embeddings``'s fail-closed ordering
    exists to prevent.
    """

    @staticmethod
    def _spec_providers(width):
        from agent_memory.providers.manager import ResolvedEmbedding

        providers = _providers([0.1] * width)
        providers.embedding_spec = ResolvedEmbedding(model="m", dimension=4)
        return providers

    async def test_a_wrong_width_leaves_neither_field(self):
        col = _collection()
        worker = _worker(col, providers=self._spec_providers(3))
        worker.enqueue({"user_id": "u1", "__search_text": "q\n\na"})

        await _run_until_drained(worker)
        doc = col.insert_one.await_args.args[0]
        assert "embedding" not in doc
        assert "search_text" not in doc
        assert worker.stats()["embed_failures"] == 1
        # Degraded to text-only, not dropped: a logged turn beats a lost one.
        assert col.insert_one.await_count == 1

    async def test_the_right_width_is_stored(self):
        # The paired case. Without it, a check that rejected every vector would
        # satisfy the test above while silently disabling episodic search.
        col = _collection()
        worker = _worker(col, providers=self._spec_providers(4))
        worker.enqueue({"user_id": "u1", "__search_text": "q\n\na"})

        await _run_until_drained(worker)
        doc = col.insert_one.await_args.args[0]
        assert doc["embedding"] == [0.1] * 4
        assert doc["search_text"] == "q\n\na"
        assert worker.stats()["embed_failures"] == 0

    async def test_a_stack_without_a_spec_stores_whatever_it_gets(self):
        # No declared width means nothing to compare against — the existing
        # `_providers()` helper, and every provider stack that is not the real
        # manager. Refusing here would break logging on those.
        col = _collection()
        worker = _worker(col, providers=_providers([0.1] * 7))
        worker.enqueue({"user_id": "u1", "__search_text": "q\n\na"})

        await _run_until_drained(worker)
        doc = col.insert_one.await_args.args[0]
        assert doc["embedding"] == [0.1] * 7
        assert worker.stats()["embed_failures"] == 0


class TestOneEmbeddingCallPerBatch:
    """REQ-E-108: a batch costs one provider round trip, not one per document.

    It used to be one per document — twenty round trips for a twenty-turn batch,
    which measured as two thirds of the batch's wall clock.
    """

    async def test_twenty_turns_are_one_provider_call(self):
        # TC-EP-WRK-050. The count is the assertion: `gather` over the
        # single-text call would collapse the same wall clock while still
        # spending twenty times the rate-limit budget.
        col = _collection()
        providers = _providers([0.1] * 4)
        worker = _worker(col, providers=providers, episodic_batch_size=20)
        for i in range(20):
            worker.enqueue({"user_id": "u1", "__search_text": f"turn {i}"})

        await _run_until_drained(worker)
        assert providers.embedding.generate_embeddings_batch.await_count == 1
        assert col.insert_many.await_count == 1
        assert len(col.insert_many.await_args.args[0]) == 20

    async def test_every_document_keeps_its_own_text_and_vector(self):
        # The paired case, and the one that catches the real hazard in batching:
        # texts and documents are two lists that must stay aligned. Distinct
        # vectors per text, so a transposition cannot pass.
        col = _collection()
        providers = MagicMock()
        providers.embedding = MagicMock()
        providers.embedding.generate_embeddings_batch = AsyncMock(
            side_effect=lambda texts: [[float(len(t))] * 2 for t in texts]
        )
        worker = _worker(col, providers=providers, episodic_batch_size=10)
        for text in ("a", "bb", "cccc"):
            worker.enqueue({"user_id": "u1", "__search_text": text})

        await _run_until_drained(worker)
        docs = col.insert_many.await_args.args[0]
        assert [d["search_text"] for d in docs] == ["a", "bb", "cccc"]
        assert [d["embedding"] for d in docs] == [[1.0] * 2, [2.0] * 2, [4.0] * 2]

    async def test_documents_without_text_do_not_shift_the_alignment(self):
        # A doc with no `__search_text` is skipped, so `texts` is shorter than
        # the batch. Zipping the vectors against the *batch* rather than against
        # the documents that contributed a text would attach turn 2's vector to
        # turn 1 — and every assertion above would still pass.
        col = _collection()
        providers = MagicMock()
        providers.embedding = MagicMock()
        providers.embedding.generate_embeddings_batch = AsyncMock(
            side_effect=lambda texts: [[float(len(t))] * 2 for t in texts]
        )
        worker = _worker(col, providers=providers, episodic_batch_size=10)
        worker.enqueue({"user_id": "u1", "n": 0})  # no text
        worker.enqueue({"user_id": "u1", "n": 1, "__search_text": "bb"})
        worker.enqueue({"user_id": "u1", "n": 2})  # no text
        worker.enqueue({"user_id": "u1", "n": 3, "__search_text": "cccc"})

        await _run_until_drained(worker)
        docs = {d["n"]: d for d in col.insert_many.await_args.args[0]}
        assert "embedding" not in docs[0] and "search_text" not in docs[0]
        assert "embedding" not in docs[2] and "search_text" not in docs[2]
        assert docs[1]["search_text"] == "bb"
        assert docs[1]["embedding"] == [2.0] * 2
        assert docs[3]["search_text"] == "cccc"
        assert docs[3]["embedding"] == [4.0] * 2

    async def test_a_short_reply_degrades_every_turn_rather_than_dropping_one(self):
        # The provider returns two vectors for three texts. `zip` would attach
        # both and silently leave the third turn unsearchable; `check_batch`
        # refuses the whole reply, so all three degrade to text-only and the
        # counter says three.
        col = _collection()
        providers = MagicMock()
        providers.embedding = MagicMock()
        providers.embedding.generate_embeddings_batch = AsyncMock(
            return_value=[[0.1] * 2, [0.2] * 2]
        )
        worker = _worker(col, providers=providers, episodic_batch_size=10)
        for text in ("a", "bb", "ccc"):
            worker.enqueue({"user_id": "u1", "__search_text": text})

        await _run_until_drained(worker)
        docs = col.insert_many.await_args.args[0]
        assert len(docs) == 3
        assert all("embedding" not in d and "search_text" not in d for d in docs)
        # Documents, not calls: the number reads the same as it did when each
        # document had its own request.
        assert worker.stats()["embed_failures"] == 3

    async def test_the_single_text_call_is_not_used(self):
        # `generate_embeddings_batch` is abstract on `EmbeddingProvider`, so
        # there is no provider that needs the one-at-a-time path, and a fallback
        # to it would let this whole class pass while nothing was batched.
        col = _collection()
        providers = _providers([0.1] * 4)
        providers.embedding.generate_embedding = AsyncMock(return_value=[0.9] * 4)
        worker = _worker(col, providers=providers, episodic_batch_size=5)
        for i in range(3):
            worker.enqueue({"user_id": "u1", "__search_text": f"turn {i}"})

        await _run_until_drained(worker)
        providers.embedding.generate_embedding.assert_not_awaited()


class TestStepAssignmentStaysSerial:
    """REQ-E-103: ``step`` follows enqueue order, so the replay is the conversation.

    The counter round trips are the other half of a batch's cost and look like
    the obvious thing to parallelise. They are not. ``$inc`` is atomic, so
    concurrent calls do get *distinct* sequence numbers — but in arrival order,
    and a connection-pool wait reorders arrivals. Gathering five turns behind one
    slow acquisition assigns ``[4, 0, 1, 2, 3]``, and ``get_thread`` sorts by
    ``step``: turn 0 replays last.
    """

    @staticmethod
    def _ordered_counters():
        """An atomic ``$inc`` behind a connection-pool wait on the first call.

        Two details make this the model that catches concurrent assignment, and
        an earlier version of this helper had both wrong:

        - The sequence is claimed **after** the wait and **before** any further
          suspension, because that is where MongoDB claims it: server-side, when
          the request arrives. A fake that increments before sleeping is not
          modelling ``$inc``, it is modelling a client-side read-modify-write,
          and it reports duplicate steps that the real server would never emit.
        - The wait is on **acquiring the connection**, not on the reply. Reply
          latency does not reorder anything under ``gather``, since ``gather``
          starts its coroutines in order and each claims on arrival — so a fake
          that delays the reply lets concurrent assignment pass. A pool wait
          happens before the request goes out, which is exactly what lets the
          four later turns overtake the first.
        """
        seq = {"n": 0}
        waits = iter([0.02, 0.0, 0.0, 0.0, 0.0])

        async def find_one_and_update(flt, update, **kw):
            await asyncio.sleep(next(waits, 0.0))  # waiting for a pool slot
            seq["n"] += 1  # the server's atomic $inc, on arrival
            return {"_id": flt["_id"], "seq": seq["n"]}

        counters = MagicMock()
        counters.find_one_and_update = find_one_and_update
        return counters

    async def test_steps_follow_enqueue_order_under_a_pool_wait(self):
        # TC-EP-WRK-051
        col = _collection()
        worker = _worker(
            col, counters=self._ordered_counters(), episodic_batch_size=5
        )
        for i in range(5):
            worker.enqueue(
                {"user_id": "u1", "n": i, "__assign_step": {"user_id": "u1", "thread_id": "t1"}}
            )

        await _run_until_drained(worker)
        docs = col.insert_many.await_args.args[0]
        steps = [d["step"] for d in sorted(docs, key=lambda d: d["n"])]
        assert steps == [0, 1, 2, 3, 4], (
            f"steps {steps} do not follow enqueue order; `get_thread` sorts by "
            "step, so the replayed conversation would be reordered"
        )

    async def test_parent_step_chains_the_turns_in_that_order(self):
        # The pairing: monotonic steps are only useful if `parent_step` agrees
        # with them, since that is what makes the thread a chain rather than a
        # set of numbered rows.
        col = _collection()
        worker = _worker(
            col, counters=self._ordered_counters(), episodic_batch_size=5
        )
        for i in range(3):
            worker.enqueue(
                {"user_id": "u1", "n": i, "__assign_step": {"user_id": "u1", "thread_id": "t1"}}
            )

        await _run_until_drained(worker)
        docs = sorted(col.insert_many.await_args.args[0], key=lambda d: d["n"])
        assert [d["parent_step"] for d in docs] == [None, 0, 1]


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


class TestPartialBatchFailure:
    """REQ-E-106: `ordered=False` means partial, and the accounting must say so."""

    @staticmethod
    def _bulk_error(batch_size, failed_indexes):
        from pymongo.errors import BulkWriteError

        inserted = batch_size - len(failed_indexes)
        return BulkWriteError({
            "writeErrors": [
                {"index": i, "code": 11000, "errmsg": "duplicate key"}
                for i in failed_indexes
            ],
            "nInserted": inserted,
        })

    async def _drain(self, col, n, audit=None):
        worker = EpisodicWorker(
            col, _counters(), _providers(), _config(episodic_batch_size=64),
            audit_service=audit,
        )
        for i in range(n):
            worker.enqueue({"user_id": f"u{i}", "n": i})
        await _run_until_drained(worker)
        return worker

    async def test_only_the_rejected_documents_count_as_failures(self):
        """TC-EP-WRK-055: one bad doc in twenty is a 5% error rate, not an outage.

        `insert_many(ordered=False)` inserts every valid document and *then* raises
        `BulkWriteError` describing the ones it rejected. The previous handler
        caught it as a total failure: `write_failures += len(batch)` and `written`
        untouched. So a single malformed document made `/health` report twenty
        failed writes and zero successes — the counter an operator alerts on said
        the episodic tier was completely down while nineteen turns were sitting in
        the collection.
        """
        col = _collection()
        col.insert_many = AsyncMock(side_effect=self._bulk_error(20, [7]))
        stats = (await self._drain(col, 20)).stats()

        assert stats["write_failures"] == 1
        assert stats["written"] == 19
        # The batch did land, so the throughput counters must reflect it.
        assert stats["batches"] == 1
        assert stats["last_write_ts"] is not None

    async def test_the_audit_trail_splits_successes_from_failures(self):
        """TC-EP-WRK-056: a stored turn must not be audited as an error.

        The old handler audited the whole batch as `"error"` against each user's own
        id. A batch spans users, so nineteen tenants got an audit record saying
        their turn failed to store when it had stored fine — and the audit log is
        the artifact you consult precisely when you no longer trust the data.
        """
        audit = MagicMock()
        audit.log = AsyncMock()
        col = _collection()
        col.insert_many = AsyncMock(side_effect=self._bulk_error(4, [2]))
        await self._drain(col, 4, audit=audit)

        by_status: dict[str, set[str]] = {}
        for call in audit.log.await_args_list:
            user_id, _category, _op, status = call.args[:4]
            by_status.setdefault(status, set()).add(user_id)

        assert by_status["error"] == {"u2"}
        assert by_status["success"] == {"u0", "u1", "u3"}

    async def test_a_total_failure_is_still_counted_in_full(self):
        """TC-EP-WRK-057: the fix must not under-count a genuine total failure."""
        col = _collection()
        col.insert_many = AsyncMock(side_effect=self._bulk_error(5, [0, 1, 2, 3, 4]))
        stats = (await self._drain(col, 5)).stats()

        assert stats["write_failures"] == 5
        assert stats["written"] == 0
        # Nothing landed, so this was not a batch.
        assert stats["batches"] == 0
        assert stats["last_write_ts"] is None

    async def test_a_malformed_details_payload_falls_back_to_the_partition(self):
        """TC-EP-WRK-058: never trust the driver's payload shape blindly.

        A `BulkWriteError` with no usable `details` must not crash the consumer or
        silently record every document as written.
        """
        from pymongo.errors import BulkWriteError

        col = _collection()
        col.insert_many = AsyncMock(side_effect=BulkWriteError(None))
        stats = (await self._drain(col, 3)).stats()

        assert stats["written"] == 3
        assert stats["write_failures"] == 0

    async def test_the_audit_error_string_is_redacted(self):
        """TC-EP-WRK-059: a duplicate-key error quotes the key's value.

        On the episodic path that value is projected turn content, so an
        unredacted message copies user text out of the tenant-scoped tier into an
        admin-readable audit collection.
        """
        audit = MagicMock()
        audit.log = AsyncMock()
        col = _collection()
        col.insert_many = AsyncMock(side_effect=PyMongoError(
            "connection to mongodb+srv://svc:s3cr3t-pw@cluster0.abc.mongodb.net failed"
        ))
        worker = EpisodicWorker(
            col, _counters(), _providers(), _config(episodic_batch_size=64),
            audit_service=audit,
        )
        worker.enqueue({"user_id": "u1"})
        worker.enqueue({"user_id": "u2"})
        await _run_until_drained(worker)

        errors = [c.kwargs.get("error", "") for c in audit.log.await_args_list]
        assert errors and all("s3cr3t-pw" not in e for e in errors)
        # The type survives, because that is what makes the entry actionable.
        assert all("PyMongoError" in e for e in errors)


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
