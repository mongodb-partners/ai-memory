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

    def __init__(self, name: str, *, failing: bool = False) -> None:
        self.name = name
        self.docs: list[dict] = []
        self.failing = failing

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
    def __init__(self, *, failing: set[str] | None = None) -> None:
        self._failing = failing or set()
        self.cols: dict[str, _Collection] = {}

    def __getitem__(self, name: str) -> _Collection:
        if name not in self.cols:
            self.cols[name] = _Collection(name, failing=name in self._failing)
        return self.cols[name]


def _facade(db: _DB, **config_overrides):
    """A facade wired to the real ``AdminService`` and real ``AuditService``."""
    from agent_memory.memory import AsyncMemory

    config = _config(**config_overrides)
    m = AsyncMemory.__new__(AsyncMemory)
    m.config = config
    m.admin_service = AdminService(db)
    m.audit_service = AuditService(db["audit_log"], config)
    m.governance_service = None
    m.rate_limiter = None
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
