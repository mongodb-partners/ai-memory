"""Buffered audit log service."""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)

# The principal an erasure is recorded against, in place of the erased user.
#
# A wipe deletes every ``audit_log`` document matching ``{"user_id": <them>}``,
# and the operation then has to be audited — a total, irreversible deletion is
# the last thing that should happen without a record. Auditing it the ordinary
# way writes that identifier straight back into the collection the wipe just
# cleared, so the answer to "delete everything you hold about me" ended with a
# fresh row naming them, timestamped a millisecond later.
#
# Recording nothing is not the alternative; accountability for a destructive
# operation is the other half of the same obligation. So the record is kept and
# the subject is dropped: what happened, when, how long it took, and how many
# documents went from each collection, filed against this reserved id. What it
# deliberately cannot answer is "was user X erased?" — that question cannot be
# answered by anyone who has genuinely stopped holding X.
#
# The leading underscore keeps it out of the space of real identifiers, which
# come from a token claim. ``wipe_user_data`` additionally refuses it as a
# target, so an erasure trail cannot be deleted by asking to be forgotten under
# this name.
ERASURE_PRINCIPAL = "_erased"


class AuditService:
    """Buffered audit log writes with configurable flush strategy."""

    def __init__(self, audit_collection, config: MCPConfig) -> None:
        self.audit_log = audit_collection
        self.config = config
        self._buffer: list[dict] = []
        self._last_flush = time.time()
        # Serialises flushes. Five callers can reach `flush()` concurrently — every
        # audited operation via `log()`, `AuditFlushWorker` on its interval,
        # `wipe_user_data` before the delete, `close()`, and a caller's own
        # `flush()` — and without this each one starts its own `insert_many`.
        #
        # The buffer swap in `flush()` is already atomic (no `await` between the
        # copy and the reset), so this is not about losing entries. It is about
        # what "flush returned" is allowed to mean. `wipe_user_data` flushes
        # *before* deleting precisely so no buffered row naming the user survives
        # the wipe; if that call can return while another flush's `insert_many` is
        # still in flight, the row lands after the delete and the erasure is undone
        # — the same defect as the episodic queue, through a different door. The
        # lock makes `flush()` mean "everything buffered when I was called has
        # reached MongoDB, or the fallback file", which is the postcondition that
        # path depends on.
        self._flush_lock = asyncio.Lock()
        # Resolved once, here, rather than on every write. The path was built as a
        # bare relative `Path("audit_fallback.jsonl")` inside `_write_to_file`, so
        # it meant "the process's current directory *at the moment of an outage*" —
        # a process that chdirs between two failed flushes splits one incident's
        # records across two files, and neither is where anyone looks.
        raw = (getattr(config, "audit_fallback_path", "") or "").strip()
        self._fallback_path: Path | None = Path(raw).expanduser().resolve() if raw else None
        if self._fallback_path is None:
            # Said once, at construction, not per failure. Disabling the fallback
            # means a MongoDB outage discards audit records outright, which is a
            # legitimate choice on a read-only filesystem and not one to make
            # silently.
            logger.warning(
                "AUDIT_FALLBACK_PATH is empty: audit entries will be DISCARDED, not "
                "written to disk, whenever a flush to MongoDB fails."
            )

    async def log(
        self,
        user_id: str,
        operation: str,
        tool_name: str,
        status: str,
        duration_ms: int,
        **metadata,
    ) -> None:
        entry = {
            "user_id": user_id,
            "operation": operation,
            "tool_name": tool_name,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(UTC),
            "metadata": metadata if metadata else {},
        }
        self._buffer.append(entry)

        should_flush = (
            self.config.audit_flush_on_write
            or len(self._buffer) >= self.config.audit_buffer_size
            or time.time() - self._last_flush >= self.config.audit_flush_interval_seconds
        )
        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        """Write everything buffered at call time to MongoDB, or to the fallback file.

        Serialised on ``_flush_lock``, so this returns only once the entries that
        existed when it was called have actually landed somewhere — including any
        that a *concurrent* flush had already taken out of the buffer. Callers rely
        on that: ``wipe_user_data`` flushes before deleting so that no buffered row
        naming the user outlives the wipe, and ``close()`` flushes so the last
        records are not lost with the process.

        Never raises. A failure here must not fail the operation being audited,
        which has usually already succeeded by the time this runs.
        """
        # There is deliberately no "buffer is empty, return early" check before
        # the lock. "Nothing buffered" is not "nothing outstanding": a concurrent
        # flush may be holding this batch in an in-flight `insert_many`, and
        # returning here is precisely the early return that breaks the
        # postcondition `wipe_user_data` depends on. Acquiring an uncontended
        # `asyncio.Lock` costs no suspension, so the idle case — the interval
        # worker on a quiet process — is already free.
        async with self._flush_lock:
            # Checked under the lock, which is the only place the answer is
            # stable. It covers both the genuinely-empty case and the flush that
            # was queued behind another and woke to find its entries already
            # taken; inserting an empty batch would mean N concurrent callers
            # produce N round trips for one batch's worth of records.
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer = []
            try:
                await self.audit_log.insert_many(batch)
            except asyncio.CancelledError:
                # Cancellation is not a write failure and the batch's fate is
                # unknown, so it goes back in the buffer for the next flush rather
                # than to the fallback file: `close()` cancels the worker tasks, and
                # a batch written to disk *and* to MongoDB is a duplicated audit
                # record, which is a quieter lie than a missing one.
                #
                # Prepended, because these entries are older than anything logged
                # while the insert was in flight, and the file is read as a
                # chronology.
                self._buffer[:0] = batch
                raise
            except Exception:
                logger.exception("Failed to flush audit entries to MongoDB")
                self._write_to_file(batch)
            self._last_flush = time.time()

    def _write_to_file(self, entries: list[dict]) -> None:
        """Append entries to the fallback file, rotating it if it has grown too big.

        This runs only while MongoDB is refusing writes, which means it runs for as
        long as the incident lasts. Unbounded, that filled the disk of a host that
        was already having an outage — and a full disk stops the process for a
        reason unrelated to the original fault, which is a worse failure than the
        one being recovered from.

        The size check is *before* the append, so the bound is "roughly this, plus
        one batch" rather than exact. Trimming mid-file to hit a byte count exactly
        would mean rewriting the file on the disk that may be the thing under
        pressure, and would leave a truncated JSON line where a reader expects one
        record per line.
        """
        if self._fallback_path is None:
            return
        try:
            self._rotate_if_needed()
            # Parents created rather than assumed: an operator setting
            # AUDIT_FALLBACK_PATH to `/var/log/agent-memory/audit.jsonl` has named a
            # location, and failing every write because one directory is missing
            # loses the records this file exists to keep.
            self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with self._fallback_path.open("a") as f:
                for entry in entries:
                    # Convert datetime to ISO string for JSON serialization
                    serializable = {
                        k: v.isoformat() if isinstance(v, datetime) else v
                        for k, v in entry.items()
                    }
                    f.write(json.dumps(serializable, default=str) + "\n")
        except Exception:
            logger.exception("Failed to write audit entries to fallback file")

    def _rotate_if_needed(self) -> None:
        """Move the fallback file aside once it exceeds the configured ceiling.

        One generation, so the cost on disk is bounded at twice the ceiling. Both
        ends of the window are worth keeping: ``.1`` holds where the outage began
        and the live file holds what is happening now. Keeping only the newest
        records loses the first failure, which is usually the informative one.

        A rotation that fails is logged and then ignored, and the append proceeds:
        exceeding the ceiling is better than dropping the record, and the next
        write will try again.
        """
        max_bytes = getattr(self.config, "audit_fallback_max_bytes", 0) or 0
        if max_bytes <= 0 or self._fallback_path is None:
            return
        try:
            if self._fallback_path.stat().st_size < max_bytes:
                return
        except OSError:
            # Not there yet, or unreadable. Either way there is nothing to rotate,
            # and the append below will report the real problem if there is one.
            return
        rotated = self._fallback_path.with_suffix(self._fallback_path.suffix + ".1")
        try:
            # `replace`, not `rename`: on Windows a rename onto an existing path
            # raises, so the second rotation of any deployment's lifetime would
            # fail and the file would then grow without bound — the exact condition
            # this method exists to prevent, reappearing only after the first fill.
            self._fallback_path.replace(rotated)
            logger.warning(
                "Audit fallback file reached %d bytes; rotated to %s. Earlier "
                "entries in that file are the only copy — MongoDB never took them.",
                max_bytes,
                rotated,
            )
        except OSError:
            logger.exception("Failed to rotate the audit fallback file")
