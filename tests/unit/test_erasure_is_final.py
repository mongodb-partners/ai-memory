"""A wipe must not leave — or create — a record naming the user it erased.

``wipe_user_data`` deletes every ``audit_log`` document matching the user's id.
It then had to be audited, because a total irreversible deletion is exactly the
operation that must leave a trace, and it was audited the ordinary way: through
``_run``, which writes its success entry *after* the service call returns.

So the last thing the erasure did was write the erased identifier back into the
collection it had just cleared. Not a leftover the wipe failed to catch — a row
the wipe *created*, dated a millisecond after the deletion, and one that survived
every later wipe too, because each one recreated it.

Two further paths lead to the same place:

* **The audit buffer.** ``audit_flush_on_write`` defaults to False, so up to
  ``audit_buffer_size`` entries for this user sit in memory when the wipe runs.
  ``delete_many`` cannot see a pending write; the buffer flushed afterwards and
  restored the very rows the wipe removed.
* **A partial failure.** ``wipe_user_data`` collected per-collection errors and
  *returned* them. ``_run`` derives its status from whether the coroutine raised,
  so a wipe that cleared three collections and failed on four was audited
  ``"success"`` — and an operator reading a success has no reason to retry.

The fix keeps the accountability and drops the subject: the record is filed
against ``ERASURE_PRINCIPAL``, carries what happened and how many documents left
each collection, and names nobody. What it deliberately cannot answer is "was
user X erased?", which is not a question anyone who has genuinely stopped holding
X can answer.

These tests run the real ``AuditService`` against a collection that really stores
and really deletes, because the defect lived in the *ordering* of two writes and
a delete. A mocked audit service cannot show it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.services.admin import AdminService, PartialWipeError
from agent_memory.services.audit import ERASURE_PRINCIPAL, AuditService


def _config(**overrides) -> MemoryConfig:
    defaults = {
        "mongodb_connection_string": "mongodb://localhost:27017",
        "_env_file": None,
    }
    defaults.update(overrides)
    return MemoryConfig(**defaults)


class _Collection:
    """A collection that stores documents and honours ``delete_many``.

    Only the two operators this path uses: a plain field match and the
    ``_id.user_id`` dotted form ``episodes_counters`` needs.
    """

    def __init__(
        self, name: str, *, failing: bool = False, count_failing: bool = False
    ) -> None:
        self.name = name
        self.docs: list[dict] = []
        self.failing = failing
        self.count_failing = count_failing

    async def insert_many(self, batch: list[dict]) -> None:
        if self.failing:
            raise RuntimeError(f"{self.name} unavailable")
        self.docs.extend(batch)

    async def delete_many(self, query: dict):
        if self.failing:
            raise RuntimeError(f"{self.name} unavailable")
        before = len(self.docs)
        self.docs = [d for d in self.docs if not self._matches(d, query)]
        return MagicMock(deleted_count=before - len(self.docs))

    async def count_documents(self, query: dict) -> int:
        """Used by the facade's post-wipe residue check.

        Real rather than stubbed to 0: without it the check cannot distinguish
        "verified empty" from "could not look", and its whole job is to catch data
        the deletion's own counts do not know about. `count_failing` is separate
        from `failing` so a collection that deletes fine but cannot be read back
        is expressible — that is the "unverified" case, which must not be reported
        as residue.
        """
        if self.failing or self.count_failing:
            raise RuntimeError(f"{self.name} unavailable")
        return sum(1 for d in self.docs if self._matches(d, query))

    @staticmethod
    def _matches(doc: dict, query: dict) -> bool:
        for path, value in query.items():
            target = doc
            for part in path.split("."):
                if not isinstance(target, dict) or part not in target:
                    return False
                target = target[part]
            if target != value:
                return False
        return True


class _DB:
    def __init__(
        self,
        *,
        failing: set[str] | None = None,
        count_failing: set[str] | None = None,
    ) -> None:
        self._failing = failing or set()
        self._count_failing = count_failing or set()
        self.cols: dict[str, _Collection] = {}

    def __getitem__(self, name: str) -> _Collection:
        if name not in self.cols:
            self.cols[name] = _Collection(
                name,
                failing=name in self._failing,
                count_failing=name in self._count_failing,
            )
        return self.cols[name]


class _Episodic:
    """Stands in for ``EpisodicService`` — only what the erasure path touches.

    ``flush`` returning False is the case that matters: queued turns that did not
    land before the deletion are pending writes, and the wipe must not claim
    completeness over them.
    """

    def __init__(self, *, drains: bool = True, raises: bool = False) -> None:
        self.drains = drains
        self.raises = raises
        self.flush_calls: list[float] = []

    async def flush(self, timeout: float = 5.0) -> bool:
        self.flush_calls.append(timeout)
        if self.raises:
            raise RuntimeError("episodic worker is gone")
        return self.drains


def _facade(db: _DB, *, episodic: _Episodic | None = None, **config_overrides):
    """A facade wired to the real ``AdminService`` and real ``AuditService``."""
    from agent_memory.memory import AsyncMemory

    config = _config(**config_overrides)
    m = AsyncMemory.__new__(AsyncMemory)
    m.config = config
    m.admin_service = AdminService(db)
    m.audit_service = AuditService(db["audit_log"], config)
    m.governance_service = None
    m.rate_limiter = None
    # `create()` sets this; a facade built by hand gets it here so the erasure
    # barrier is exercised rather than falling back to the class default.
    m._erasing = set()
    m.episodic_service = episodic if episodic is not None else _Episodic()
    return m


def _entries(db: _DB, *, operation: str | None = None) -> list[dict]:
    docs = db["audit_log"].docs
    if operation is None:
        return docs
    return [d for d in docs if d["tool_name"] == operation]


class TestTheWipeDoesNotRecreateTheUser:
    """The reported defect, at the level it was reported."""

    async def test_no_audit_row_names_the_erased_user_afterwards(self) -> None:
        db = _DB()
        app = _facade(db)

        # Ordinary activity first, so there is a real audit history to erase.
        for _ in range(3):
            await app.audit_service.log("alice", "memory:write", "store_memory",
                                        "success", 1)
        await app.audit_service.flush()
        assert len(_entries(db)) == 3, "fixture: no history to erase"

        await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        survivors = [d for d in _entries(db) if d["user_id"] == "alice"]
        assert survivors == [], (
            f"{len(survivors)} audit row(s) still name the erased user: {survivors}"
        )

    async def test_the_erasure_is_still_audited(self) -> None:
        """Dropping the record is not the fix — accountability is the other half.

        A wipe that leaves no trace is a wipe nobody can be held to.
        """
        db = _DB()
        app = _facade(db)

        await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        records = _entries(db, operation="wipe_user_data")
        assert len(records) == 1, "the erasure left no audit record"
        assert records[0]["user_id"] == ERASURE_PRINCIPAL
        assert records[0]["status"] == "success"
        assert records[0]["operation"] == "admin"

    async def test_the_record_carries_the_counts_but_not_the_subject(self) -> None:
        """What was deleted is per-collection integers, which identify no one."""
        db = _DB()
        db["memories"].docs.extend([{"user_id": "alice"} for _ in range(4)])
        db["episodes"].docs.extend([{"user_id": "alice"} for _ in range(9)])
        app = _facade(db)

        await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        record = _entries(db, operation="wipe_user_data")[0]
        deleted = record["metadata"]["deleted"]
        assert deleted["memories_deleted"] == 4
        assert deleted["episodes_deleted"] == 9
        assert "user_id" not in deleted, "the subject came back in via the counts"
        assert "alice" not in repr(record), f"the erased id survives in {record!r}"

    async def test_the_erasure_record_is_flushed_immediately(self) -> None:
        """It is the only evidence the operation happened.

        ``audit_flush_on_write`` is False by default and the buffer holds ten
        entries, so left buffered the record would be lost to any process that
        exits before the next flush — which is a plausible thing for a process to
        do right after an admin wipe.
        """
        db = _DB()
        app = _facade(db)

        await app.wipe_user_data("alice", confirm=True)
        # No explicit flush.
        assert _entries(db, operation="wipe_user_data"), (
            "the erasure record was still buffered when the call returned"
        )

    async def test_a_second_wipe_finds_nothing_left(self) -> None:
        """The recreated row survived every later wipe, because each recreated it.

        A wipe that is not idempotent in this sense never converges: the user is
        never actually gone.
        """
        db = _DB()
        app = _facade(db)

        await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()
        second = await app.wipe_user_data("alice", confirm=True)

        assert second["audit_deleted"] == 0, (
            "the first wipe left audit rows for the second to delete"
        )


class TestBufferedEntriesAreAccountedFor:
    """A pending write is a write. ``delete_many`` cannot see the buffer."""

    async def test_unflushed_entries_do_not_survive_the_wipe(self) -> None:
        db = _DB()
        app = _facade(db)

        # Buffered, not yet in Atlas: the default config flushes at ten entries
        # or after sixty seconds, so this is the normal state, not a contrived one.
        for _ in range(3):
            await app.audit_service.log("alice", "memory:write", "store_memory",
                                        "success", 1)
        assert _entries(db) == [], "fixture: entries already flushed"

        await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        survivors = [d for d in _entries(db) if d["user_id"] == "alice"]
        assert survivors == [], (
            f"{len(survivors)} buffered row(s) were flushed after the wipe swept "
            f"the collection: {survivors}"
        )

    async def test_another_users_buffered_entries_are_untouched(self) -> None:
        """The pre-wipe flush must not be a chance to lose someone else's records."""
        db = _DB()
        app = _facade(db)

        await app.audit_service.log("bob", "memory:write", "store_memory",
                                    "success", 1)
        await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        assert [d["user_id"] for d in _entries(db, operation="store_memory")] == ["bob"]


class TestAPartialWipeIsNotASuccess:
    """``_run`` reads the status from whether the call raised. So it must raise."""

    async def test_a_failed_collection_makes_the_wipe_raise(self) -> None:
        db = _DB(failing={"episodes"})
        app = _facade(db)

        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)

    async def test_it_is_audited_as_an_error_not_a_success(self) -> None:
        """The defect: seven collections attempted, four failed, audited "success".

        An operator reading that has been told the data is gone.
        """
        db = _DB(failing={"episodes", "decisions"})
        app = _facade(db)

        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        record = _entries(db, operation="wipe_user_data")[0]
        assert record["status"] == "error", (
            f"an incomplete erasure audited as {record['status']!r}"
        )

    async def test_the_error_record_says_what_is_left(self) -> None:
        """What survived is the only thing that makes the retry safe."""
        db = _DB(failing={"episodes"})
        app = _facade(db)

        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        meta = _entries(db, operation="wipe_user_data")[0]["metadata"]
        assert meta["failed_collections"] == ["episodes"]
        assert "memories_deleted" in meta["deleted"]

    async def test_the_error_record_still_names_nobody(self) -> None:
        db = _DB(failing={"episodes"})
        app = _facade(db)

        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        record = _entries(db, operation="wipe_user_data")[0]
        assert record["user_id"] == ERASURE_PRINCIPAL
        assert "alice" not in repr(record)

    async def test_the_other_collections_are_still_cleared(self) -> None:
        """Raising must not mean abandoning the rest — that leaves the most data."""
        db = _DB(failing={"memories"})
        db["episodes"].docs.append({"user_id": "alice"})
        app = _facade(db)

        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)

        assert db["episodes"].docs == [], (
            "a failure on the first collection stopped the wipe"
        )

    async def test_the_exception_carries_the_counts(self) -> None:
        db = _DB(failing={"episodes"})
        db["memories"].docs.extend([{"user_id": "alice"} for _ in range(2)])
        app = _facade(db)

        with pytest.raises(PartialWipeError) as exc_info:
            await app.wipe_user_data("alice", confirm=True)

        assert exc_info.value.counts["memories_deleted"] == 2
        assert exc_info.value.counts["episodes_deleted"] == 0
        assert "episodes" in exc_info.value.errors

    async def test_a_clean_wipe_still_reports_complete(self) -> None:
        db = _DB()
        result = await _facade(db).wipe_user_data("alice", confirm=True)
        assert result["complete"] is True
        assert "errors" not in result


class TestTheErasurePrincipalIsNotATenant:
    """Otherwise the erasure trail is deletable by asking to be forgotten."""

    async def test_it_cannot_be_wiped(self) -> None:
        db = _DB()
        with pytest.raises(ValueError, match="reserved"):
            await AdminService(db).wipe_user_data(ERASURE_PRINCIPAL)

    async def test_nothing_is_deleted_when_it_is_refused(self) -> None:
        """The refusal has to come before the first ``delete_many``."""
        db = _DB()
        db["memories"].docs.append({"user_id": ERASURE_PRINCIPAL})
        with pytest.raises(ValueError):
            await AdminService(db).wipe_user_data(ERASURE_PRINCIPAL)
        assert db["memories"].docs, "documents were deleted before the refusal"

    async def test_erasure_records_survive_a_users_wipe(self) -> None:
        """The trail of *other* erasures is not part of anyone's own data."""
        db = _DB()
        app = _facade(db)

        await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()
        await app.wipe_user_data("bob", confirm=True)
        await app.audit_service.flush()

        assert len(_entries(db, operation="wipe_user_data")) == 2, (
            "one erasure record deleted the other"
        )

    def test_it_is_not_a_plausible_real_identifier(self) -> None:
        """Identifiers come from a token claim; a leading underscore keeps this
        one out of that space. A collision would silently merge a real tenant's
        records into the erasure trail — and make them undeletable."""
        assert ERASURE_PRINCIPAL.startswith("_")


class TestTheAccessDecisionStillUsesTheRealIdentity:
    """Authorisation is about who is asking. That is decided before anything goes."""

    @staticmethod
    def _denying_facade(db: _DB):
        app = _facade(db, auth_enabled=True, auth_secret="s" * 32)
        app.governance_service = MagicMock()
        app.governance_service.check_allowed = AsyncMock(return_value=False)
        app.governance_service.get_profile = AsyncMock(return_value=None)
        return app

    async def test_a_denied_wipe_deletes_nothing(self) -> None:
        from agent_memory.exceptions import AccessError

        db = _DB()
        db["memories"].docs.append({"user_id": "alice"})
        app = self._denying_facade(db)

        with pytest.raises(AccessError):
            await app.wipe_user_data("alice", confirm=True)

        assert db["memories"].docs, "the wipe ran despite being denied"

    async def test_a_denied_wipe_is_audited_against_the_real_user(self) -> None:
        """A refused wipe erased nothing, so there is no erasure to respect —
        and an attempt to wipe a tenant is precisely what an auditor needs
        attributed. Only a wipe that actually ran withholds the subject."""
        from agent_memory.exceptions import AccessError

        db = _DB()
        app = self._denying_facade(db)

        with pytest.raises(AccessError):
            await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        record = _entries(db, operation="wipe_user_data")[0]
        assert record["status"] == "denied"
        assert record["user_id"] == "alice"

    async def test_a_throttled_wipe_is_labelled_throttled(self) -> None:
        """``RateLimitError`` subclasses ``AccessError``, so testing the base
        first would label every throttle "denied" — the more alarming label."""
        from agent_memory.exceptions import RateLimitError

        db = _DB()
        app = _facade(db)
        app.rate_limiter = MagicMock()
        app.rate_limiter.check_rate_limit = AsyncMock(return_value=False)

        with pytest.raises(RateLimitError):
            await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        assert _entries(db, operation="wipe_user_data")[0]["status"] == "throttled"

    async def test_the_confirm_gate_still_short_circuits(self) -> None:
        """Without ``confirm`` nothing runs and nothing is audited."""
        db = _DB()
        app = _facade(db)

        result = await app.wipe_user_data("alice")
        await app.audit_service.flush()

        assert "error" in result
        assert db["memories"].docs == []
        assert _entries(db, operation="wipe_user_data") == []


class TestTheMCPToolReportsThePartialState:
    """The counts must survive translation to the MCP error convention."""

    @staticmethod
    def _register(app):
        """Collect the tool functions FastMCP would register."""
        from agent_memory.shells.mcp.tools import register_all_tools

        tools: dict = {}
        mcp = MagicMock()

        def tool(name: str, description: str = ""):
            def decorator(fn):
                tools[name] = fn
                return fn
            return decorator

        mcp.tool = tool
        register_all_tools(mcp, app)
        return tools

    async def test_a_partial_wipe_returns_the_counts_not_just_a_message(self) -> None:
        db = _DB(failing={"episodes"})
        db["memories"].docs.extend([{"user_id": "alice"} for _ in range(3)])
        wipe = self._register(_facade(db))["wipe_user_data"]

        out = await wipe(user_id="alice", confirm=True)

        assert out["complete"] is False
        assert out["memories_deleted"] == 3
        assert out["failed_collections"] == ["episodes"]
        assert "error" in out

    async def test_a_clean_wipe_is_unchanged(self) -> None:
        db = _DB()
        wipe = self._register(_facade(db))["wipe_user_data"]

        out = await wipe(user_id="alice", confirm=True)

        assert out["complete"] is True
        assert "failed_collections" not in out


class TestQueuedTurnsCannotOutliveTheErasure:
    """`log_activity` returns once the turn is *queued*; the insert happens later.

    So a turn accepted a moment before the wipe was still in memory when the
    collections were swept, and landed afterwards — recreating episodic history
    for a user who had asked to be forgotten. The audit buffer had exactly this
    shape and was already flushed for exactly this reason; the episodic queue is
    the same argument applied to the other pending write.
    """

    async def test_the_queue_is_drained_before_the_delete(self) -> None:
        order: list[str] = []
        db = _DB()

        class _Recording(_Episodic):
            async def flush(self, timeout: float = 5.0) -> bool:
                order.append("drain")
                return await super().flush(timeout)

        episodic = _Recording()
        app = _facade(db, episodic=episodic)
        original = app.admin_service.wipe_user_data

        async def _traced(user_id: str):
            order.append("delete")
            return await original(user_id)

        app.admin_service.wipe_user_data = _traced
        await app.wipe_user_data("alice", confirm=True)

        assert order == ["drain", "delete"], (
            "draining after the delete lets a queued turn land in a swept "
            f"collection; order was {order}"
        )

    async def test_the_drain_uses_the_configured_timeout(self) -> None:
        episodic = _Episodic()
        app = _facade(
            _DB(), episodic=episodic, episodic_shutdown_timeout_seconds=2.5
        )
        await app.wipe_user_data("alice", confirm=True)
        assert episodic.flush_calls == [2.5]

    async def test_an_undrained_queue_is_not_reported_complete(self) -> None:
        """The turns have not landed *yet*, so the collections read empty. The
        deletion's own counts cannot see them, which is the whole problem."""
        app = _facade(_DB(), episodic=_Episodic(drains=False))

        with pytest.raises(PartialWipeError) as exc:
            await app.wipe_user_data("alice", confirm=True)
        assert "episodes" in exc.value.errors

    async def test_a_drain_that_raises_is_not_reported_complete(self) -> None:
        app = _facade(_DB(), episodic=_Episodic(raises=True))

        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)

    async def test_a_broken_drain_still_deletes_everything_it_can(self) -> None:
        """Most of a user's data is in collections the queue never touches.
        Refusing to delete them because the drain failed would leave *more*
        behind, not less."""
        db = _DB()
        db["memories"].docs.extend([{"user_id": "alice"} for _ in range(4)])
        app = _facade(db, episodic=_Episodic(drains=False))

        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)
        assert db["memories"].docs == [], "gave up on the deletion it could do"

    async def test_no_episodic_service_is_not_a_failure(self) -> None:
        """`episodic_enabled=False` builds a facade without one."""
        app = _facade(_DB())
        app.episodic_service = None
        result = await app.wipe_user_data("alice", confirm=True)
        assert result["complete"] is True


class TestWritesAreRefusedWhileAUserIsBeingErased:
    """Nothing stopped a write that arrived mid-wipe from inserting into a
    collection the deletion had already swept.

    The barrier is per-process and does not pretend otherwise — it is not a
    distributed lock. What it guarantees is that no write from *this* process
    survives the erasure; the residue check below covers the rest.
    """

    @staticmethod
    def _app():
        app = _facade(_DB())
        app._erasing.add("alice")
        return app

    @pytest.mark.parametrize(
        "operation",
        ["store_memory", "store_cache", "store_decision", "log_activity",
         "delete_memory", "cache_invalidate"],
    )
    async def test_every_write_operation_is_refused(self, operation) -> None:
        from agent_memory.exceptions import ErasureInProgressError

        with pytest.raises(ErasureInProgressError):
            await self._app()._check_access("alice", operation)

    @pytest.mark.parametrize(
        "operation",
        ["recall_memory", "hybrid_search", "check_cache", "search_activity",
         "get_thread", "recall_decision", "memory_health"],
    )
    async def test_reads_stay_available(self, operation) -> None:
        """A read during a wipe returns progressively less, which is honest.
        Refusing them would make the erasure look like an outage."""
        await self._app()._check_access("alice", operation)

    async def test_other_users_are_unaffected(self) -> None:
        await self._app()._check_access("bob", "store_memory")

    async def test_the_refusal_is_an_access_error(self) -> None:
        """So the shells' existing 403 mapping and `_run`'s "denied" audit status
        both apply without either learning a new exception type."""
        from agent_memory.exceptions import AccessError, ErasureInProgressError

        assert issubclass(ErasureInProgressError, AccessError)

    async def test_the_barrier_is_lifted_afterwards(self) -> None:
        app = _facade(_DB())
        await app.wipe_user_data("alice", confirm=True)
        assert "alice" not in app._erasing
        await app._check_access("alice", "store_memory")

    async def test_the_barrier_is_lifted_even_when_the_wipe_fails(self) -> None:
        """A user permanently unable to write because an erasure errored once is
        worse than the incomplete deletion, and the caller was told to retry."""
        app = _facade(_DB(failing={"episodes"}))

        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)
        assert "alice" not in app._erasing
        await app._check_access("alice", "store_memory")

    async def test_a_write_during_the_wipe_is_refused_for_real(self) -> None:
        """Driven through the facade rather than `_check_access` directly: the
        barrier is only worth anything if it sits on the path `add()` takes."""
        from agent_memory.exceptions import ErasureInProgressError

        db = _DB()
        app = _facade(db)
        app.memory_service = MagicMock()
        # `store_stm` is what `add()` actually calls; stubbed so that if the
        # barrier ever stops working, this test fails on the assertion below
        # rather than on an incomplete fake.
        app.memory_service.store_stm = AsyncMock(return_value=["m1"])
        refused: list = []
        original = app.admin_service.wipe_user_data

        async def _wipe_then_race(user_id: str):
            # Mid-erasure, exactly where the interleaving write used to survive.
            try:
                await app.add("alice", "c1", [{"role": "user", "content": "hi"}])
            except ErasureInProgressError as e:
                refused.append(e)
            return await original(user_id)

        app.admin_service.wipe_user_data = _wipe_then_race
        await app.wipe_user_data("alice", confirm=True)

        assert refused, "a concurrent add() was accepted during the erasure"
        assert not app.memory_service.store_stm.await_count, (
            "the write reached the service layer during the erasure"
        )

    def test_the_write_list_covers_every_write_the_facade_audits(self) -> None:
        """A new write path gets no barrier unless it is listed, and nothing about
        adding one would make anybody look. So the list is checked against the
        facade's own audit categories: any operation the facade records under a
        `:write`/`:delete`/admin-mutation category must be in it.
        """
        import re
        from pathlib import Path

        from agent_memory.memory import _WRITE_OPERATIONS

        source = Path("agent_memory/memory.py").read_text()
        # `_run(user_id, "<operation>", "<category>", ...)` and the two
        # hand-rolled audit sites (`log_activity`, `wipe_user_data`).
        pairs = set(re.findall(r'"([a-z_]+)",\s*"([a-z_:]+)"', source))
        writes = {
            op for op, cat in pairs
            if cat.endswith((":write", ":delete"))
        }
        # `log_activity` is audited as `episodic:write` with the arguments in the
        # other order, so it is named rather than pattern-matched.
        writes.add("log_activity")
        missing = writes - _WRITE_OPERATIONS
        assert not missing, (
            f"write operations with no erasure barrier: {sorted(missing)} — add "
            "them to _WRITE_OPERATIONS in agent_memory/memory.py"
        )


class TestCompletenessIsVerifiedRatherThanAsserted:
    """`complete: true` used to mean "no `delete_many` raised".

    That is a claim about this process's own calls, and says nothing about a
    write from another replica that landed between the delete and the reply. For
    the one operation whose entire value is being final, the answer is checked.
    """

    async def test_residue_left_by_another_writer_is_caught(self) -> None:
        db = _DB()
        app = _facade(db)
        original = app.admin_service.wipe_user_data

        async def _wipe_then_another_replica_writes(user_id: str):
            out = await original(user_id)
            # A second process, unaffected by this one's in-memory barrier.
            db["memories"].docs.append({"user_id": "alice", "content": "late"})
            return out

        app.admin_service.wipe_user_data = _wipe_then_another_replica_writes

        with pytest.raises(PartialWipeError) as exc:
            await app.wipe_user_data("alice", confirm=True)
        assert "memories" in exc.value.errors

    async def test_the_residue_is_audited_as_an_error(self) -> None:
        db = _DB()
        app = _facade(db)
        original = app.admin_service.wipe_user_data

        async def _leaves_residue(user_id: str):
            out = await original(user_id)
            db["episodes"].docs.append({"user_id": "alice"})
            return out

        app.admin_service.wipe_user_data = _leaves_residue
        with pytest.raises(PartialWipeError):
            await app.wipe_user_data("alice", confirm=True)
        await app.audit_service.flush()

        record = _entries(db, operation="wipe_user_data")[0]
        assert record["status"] == "error"
        assert record["user_id"] == ERASURE_PRINCIPAL, "residue re-identified them"

    async def test_the_counters_composite_id_is_checked_correctly(self) -> None:
        """`episodes_counters` is keyed by a composite `_id`. A residue check
        using the obvious `{"user_id": ...}` filter would match nothing there and
        report clean over surviving counters."""
        db = _DB()
        app = _facade(db)
        original = app.admin_service.wipe_user_data

        async def _leaves_a_counter(user_id: str):
            out = await original(user_id)
            db["episodes_counters"].docs.append(
                {"_id": {"user_id": "alice", "thread_id": "t1"}, "seq": 3}
            )
            return out

        app.admin_service.wipe_user_data = _leaves_a_counter
        with pytest.raises(PartialWipeError) as exc:
            await app.wipe_user_data("alice", confirm=True)
        assert "episodes_counters" in exc.value.errors

    async def test_an_unreadable_collection_is_unverified_not_residue(self) -> None:
        """The delete succeeded; a follow-up read failing is a reason to say
        "could not confirm", not to tell the operator their data is still there."""
        db = _DB(count_failing={"decisions"})
        app = _facade(db)

        result = await app.wipe_user_data("alice", confirm=True)

        assert result["complete"] is True
        assert result["unverified_collections"] == ["decisions"]

    async def test_a_verified_wipe_says_nothing_about_unverified(self) -> None:
        result = await _facade(_DB()).wipe_user_data("alice", confirm=True)
        assert "unverified_collections" not in result

    async def test_the_check_looks_in_every_collection_the_wipe_touches(self) -> None:
        """Two copies of the collection list would drift, and a residue check
        missing a collection produces the exact wrong answer this path exists to
        prevent."""
        from agent_memory.services.admin import erasure_targets

        db = _DB()
        app = _facade(db)
        counted: list[str] = []
        for _key, name, _q in erasure_targets("alice"):
            col = db[name]
            original_count = col.count_documents

            async def _record(query, _name=name, _orig=original_count):
                counted.append(_name)
                return await _orig(query)

            col.count_documents = _record

        await app.wipe_user_data("alice", confirm=True)

        expected = {name for _k, name, _q in erasure_targets("alice")}
        assert set(counted) == expected, (
            f"unchecked collections: {sorted(expected - set(counted))}"
        )
