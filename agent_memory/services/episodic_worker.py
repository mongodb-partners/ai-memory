"""Background writer for the episodic activity log.

The agent's turn must never wait on its own diary. ``EpisodicService.enqueue``
builds a document and hands it to this worker, which batches inserts to Atlas
from a single consumer task. Everything expensive — the durable step counter,
the embedding round trip, the insert — happens here, off the caller's path.

Design decisions worth keeping if this is ever rewritten:

- **A single consumer.** FIFO per thread holds trivially with one consumer, so
  ``step`` stays monotonic. Multiple consumers would need per-thread
  partitioning for no throughput win at these queue depths.
- **Drop the oldest, never the newest.** A full queue means the writer is behind;
  the freshest turn is the one worth keeping.
- **Every failure is a counter, not an exception.** Logging is not the product.
  A dead Atlas connection must degrade telemetry, not break the agent.
- **A logged turn beats a lost one.** If the step counter fails, the document is
  inserted with ``step=None`` rather than dropped.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import BulkWriteError, PyMongoError

from agent_memory.core.redaction import redact_error

logger = logging.getLogger(__name__)

# Internal keys stripped from the document before insert. They carry work the
# worker must do rather than data to store.
_KEY_ASSIGN_STEP = "__assign_step"
_KEY_SEARCH_TEXT = "__search_text"

# Sentinel that tells the consumer to stop. Deliberately excluded from the
# in-flight count, or flush() would wait for a document that never lands.
_SHUTDOWN = object()


class EpisodicWorker:
    """Batch episodic documents to MongoDB from a bounded queue."""

    def __init__(
        self,
        collection,
        counter_collection,
        providers,
        config,
        *,
        audit_service=None,
    ) -> None:
        self.collection = collection
        self.counters = counter_collection
        self.providers = providers
        self.config = config
        # One audit entry per flushed batch, not per turn. Routing a turn log
        # through per-call auditing costs more writes than the agent it records.
        self.audit_service = audit_service

        self._queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=max(1, config.episodic_queue_size)
        )
        # Documents enqueued but not yet processed. flush() waits on this
        # reaching zero, so it must be incremented before the put becomes
        # visible to the consumer — see enqueue().
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False
        self._running = False

        self._enqueued = 0
        self._written = 0
        self._dropped = 0
        self._embed_failures = 0
        self._write_failures = 0
        self._batches = 0
        self._last_write_ts: datetime | None = None

    # ─── Enqueue ─────────────────────────────────────────────────

    def enqueue(self, doc: dict[str, Any]) -> None:
        """Hand a document to the worker. Never blocks, never raises."""
        if self._closed:
            return

        # Count the document in flight BEFORE it becomes visible to the
        # consumer. Reversed, the consumer can dequeue and decrement first; the
        # ``> 0`` guard then swallows that decrement and _inflight is orphaned
        # above zero, so flush() never returns True again.
        self._inflight += 1
        self._enqueued += 1
        self._idle.clear()

        try:
            self._queue.put_nowait(doc)
        except asyncio.QueueFull:
            self._evict_oldest()
            try:
                self._queue.put_nowait(doc)
            except asyncio.QueueFull:  # pragma: no cover - lost the race twice
                self._inflight -= 1
                self._enqueued -= 1
                self._dropped += 1
                self._settle()
                logger.warning("Episodic queue full; dropped incoming turn.")

    def _evict_oldest(self) -> None:
        """Discard the head so the newest turn can be enqueued."""
        try:
            self._queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - drained under us
            return
        if self._inflight > 0:
            self._inflight -= 1
        self._dropped += 1
        self._settle()
        logger.warning("Episodic queue full; dropped the oldest turn.")

    def _settle(self) -> None:
        if self._inflight <= 0:
            self._inflight = 0
            self._idle.set()

    # ─── Consumer loop ───────────────────────────────────────────

    async def run(self) -> None:
        """Drain the queue in batches until shutdown. The single consumer."""
        self._running = True
        try:
            while True:
                batch = await self._collect_batch()
                if batch is None:
                    return
                if batch:
                    await self._write_batch(batch)
        finally:
            self._running = False

    async def _collect_batch(self) -> list[dict[str, Any]] | None:
        """Gather up to ``episodic_batch_size`` docs, or return None to stop.

        Blocks on the first document so an idle worker costs nothing, then takes
        whatever else is already queued without waiting — a partial batch beats
        a stale one.
        """
        first = await self._queue.get()
        if first is _SHUTDOWN:
            return None

        batch = [first]
        while len(batch) < max(1, self.config.episodic_batch_size):
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is _SHUTDOWN:
                # Write what we have, then stop on the next iteration.
                self._queue.put_nowait(_SHUTDOWN)
                break
            batch.append(item)
        return batch

    async def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        """Resolve steps and embeddings, then insert. Never raises."""
        try:
            for doc in batch:
                await self._assign_durable_step(doc)
                await self._attach_embedding(doc)
            await self._insert(batch)
        except Exception:  # pragma: no cover - last-resort guard
            logger.warning("Episodic batch write failed unexpectedly.", exc_info=True)
        finally:
            self._inflight -= len(batch)
            self._settle()

    async def _assign_durable_step(self, doc: dict[str, Any]) -> None:
        """Assign a monotonic per-thread ``step`` from the persisted counter.

        Durable rather than in-memory so ``step`` keeps counting across process
        restarts — a turn log whose numbering resets is misleading about order.
        """
        key = doc.pop(_KEY_ASSIGN_STEP, None)
        if key is None:
            return
        # `_id` is the composite {user_id, thread_id} rather than the bare
        # thread_id. Two things depend on it: sequences cannot collide across
        # tenants that pick the same thread name, and `wipe_user_data` can
        # actually find a user's counters to delete. A str is tolerated for
        # documents enqueued by an older build mid-upgrade.
        if isinstance(key, str):
            key = {"user_id": doc.get("user_id") or "", "thread_id": key}
        try:
            result = await self.counters.find_one_and_update(
                {"_id": key},
                {"$inc": {"seq": 1}},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            step = int(result["seq"]) - 1
            doc["step"] = step
            doc["parent_step"] = step - 1 if step > 0 else None
        except (PyMongoError, KeyError, TypeError, ValueError) as exc:
            # Insert anyway: a turn with an unknown step is still evidence.
            doc["step"] = None
            doc["parent_step"] = None
            logger.warning("Episodic step assignment failed: %s", exc)

    async def _attach_embedding(self, doc: dict[str, Any]) -> None:
        """Embed the search text, then attach both fields.

        Order matters: the embedding is generated first, so a provider failure
        leaves *neither* ``embedding`` nor ``search_text`` on the document. A
        document with searchable text but no vector would rank inconsistently
        between the two branches of hybrid recall.
        """
        search_text = doc.pop(_KEY_SEARCH_TEXT, None)
        if not search_text:
            return
        try:
            doc["embedding"] = await self.providers.embedding.generate_embedding(
                search_text
            )
            doc["search_text"] = search_text
        except Exception as exc:
            self._embed_failures += 1
            logger.warning("Episodic embedding failed: %s", exc)

    async def _insert(self, batch: list[dict[str, Any]]) -> None:
        started = datetime.now(UTC)
        try:
            if len(batch) == 1:
                await self.collection.insert_one(batch[0])
            else:
                # ordered=False so one bad document does not abort the rest.
                await self.collection.insert_many(batch, ordered=False)
        except BulkWriteError as exc:
            # The whole point of ordered=False is that this is a *partial*
            # failure: the good documents are already in the collection and the
            # exception reports which ones were not. Treating it as a total
            # failure — which the previous handler did — got both halves of the
            # telemetry wrong at once. One malformed document in a batch of 20
            # added 20 to `write_failures` and 0 to `written`, so the counter a
            # `/health` probe watches reported a total outage during what was
            # really a 5% error rate, and the audit trail recorded 19 users'
            # successfully stored turns as errors against their own ids.
            await self._record_partial(batch, exc, started)
            return
        except Exception as exc:
            self._write_failures += len(batch)
            logger.warning("Episodic insert failed (%d docs): %s", len(batch), exc)
            await self._audit(batch, "error", started, error=redact_error(exc))
            return
        self._written += len(batch)
        self._batches += 1
        self._last_write_ts = datetime.now(UTC)
        await self._audit(batch, "success", started)

    async def _record_partial(
        self, batch: list[dict[str, Any]], exc: BulkWriteError, started: datetime
    ) -> None:
        """Split a partially-failed batch and account for each half separately.

        ``writeErrors[].index`` positions each failure within the batch we
        submitted, so the successes are exactly the complement — this is
        recoverable precisely, not estimated. ``nInserted`` is used as the
        authority for the counters because a driver that reports a failure
        without an index (a duplicate-key report on a retried write, for
        instance) would otherwise be silently counted as a success.
        """
        details = getattr(exc, "details", None) or {}
        write_errors = details.get("writeErrors") or []
        failed_idx = {
            e["index"] for e in write_errors
            if isinstance(e, dict) and isinstance(e.get("index"), int)
        }
        failed = [d for i, d in enumerate(batch) if i in failed_idx]
        ok = [d for i, d in enumerate(batch) if i not in failed_idx]

        # Prefer the driver's own count; fall back to the partition when the
        # details payload is absent or malformed.
        inserted = details.get("nInserted")
        inserted = len(ok) if not isinstance(inserted, int) else inserted
        n_failed = max(0, len(batch) - inserted)

        self._written += inserted
        self._write_failures += n_failed
        if inserted:
            self._batches += 1
            self._last_write_ts = datetime.now(UTC)

        logger.warning(
            "Episodic insert partially failed: %d of %d documents rejected (%s).",
            n_failed, len(batch), redact_error(exc),
        )
        if ok:
            await self._audit(ok, "success", started)
        if failed:
            await self._audit(failed, "error", started, error=redact_error(exc))

    async def _audit(
        self,
        batch: list[dict[str, Any]],
        status: str,
        started: datetime,
        **metadata: Any,
    ) -> None:
        """Emit one audit entry per batch, attributed per user. Never raises.

        A batch can span users, so it is grouped by ``user_id`` — an audit trail
        that attributed one user's turns to another would be worse than none.
        """
        if self.audit_service is None:
            return
        duration_ms = int(
            (datetime.now(UTC) - started).total_seconds() * 1000
        )
        counts: dict[str, int] = {}
        for doc in batch:
            user_id = doc.get("user_id") or ""
            counts[user_id] = counts.get(user_id, 0) + 1
        try:
            for user_id, count in counts.items():
                await self.audit_service.log(
                    user_id, "episodic:write", "log_activity", status,
                    duration_ms, turns=count, **metadata,
                )
        except Exception:
            logger.debug("Episodic batch audit failed.", exc_info=True)

    # ─── Lifecycle ───────────────────────────────────────────────

    async def flush(self, timeout: float = 5.0) -> bool:
        """Wait for the queue to drain, bounded by ``timeout``. Never raises.

        Returns True if everything enqueued has been processed. Does not stop
        the worker, so it is safe to call between turns.
        """
        if self._inflight <= 0:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    async def close(self, timeout: float = 5.0) -> bool:
        """Flush, signal shutdown, and refuse further writes. Idempotent."""
        already_closed = self._closed
        self._closed = True
        if already_closed:
            return True
        drained = await self.flush(timeout)
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except asyncio.QueueFull:  # pragma: no cover
            self._evict_oldest()
            try:
                self._queue.put_nowait(_SHUTDOWN)
            except asyncio.QueueFull:
                pass
        return drained

    def stop(self) -> None:
        """Signal shutdown without waiting. Matches the other workers' API."""
        self._closed = True
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except asyncio.QueueFull:  # pragma: no cover
            pass

    # ─── Introspection ───────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Queue and throughput counters, with no database round trip.

        This is what a ``/health`` probe reads: a climbing ``dropped`` or
        ``write_failures`` is the signal that logging is degrading silently.
        """
        return {
            "queue_depth": self._queue.qsize(),
            "queue_capacity": self._queue.maxsize,
            "worker_alive": self._running,
            "enqueued": self._enqueued,
            "written": self._written,
            "dropped": self._dropped,
            "batches": self._batches,
            "embed_failures": self._embed_failures,
            "write_failures": self._write_failures,
            "last_write_ts": (
                self._last_write_ts.isoformat()
                if self._last_write_ts is not None
                else None
            ),
        }


__all__ = ["EpisodicWorker"]
