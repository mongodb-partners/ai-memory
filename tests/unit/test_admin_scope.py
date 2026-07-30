"""Deletion must be complete, retention must be honest, pools must not be shared blind.

Three findings that share a shape: an operation whose real scope differs from the
scope its signature and docstring advertise.

1. **C3 — ``set_activity_retention`` takes a ``user_id`` and ignores it.** A TTL
   index belongs to a collection, so ``collMod`` retunes retention for every
   tenant. The parameter is the principal being authorised, not a scope. Left
   unstated, an operator reads the signature and believes one user's retention
   changed.

2. **``wipe_user_data`` promised "ALL user data" and cleared three collections of
   seven.** Episodic turns, sticky decisions, step counters, and rate-limit
   records survived — so the answer to a deletion request was wrong, not merely
   partial. ``episodes_counters`` is the subtle one: keyed by a composite ``_id``,
   so the obvious ``{"user_id": ...}`` filter matches nothing at all.

3. **``DatabaseManager`` was first-config-wins and closed unconditionally.** A
   second facade built against another cluster silently used the first one's, and
   whichever shell shut down first severed the connection for the other.

REQ-E-144 (an operation's scope is the scope it states).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.core.database import DatabaseManager
from agent_memory.services.admin import AdminService


def _config(**overrides) -> MemoryConfig:
    """`MemoryConfig`, not `MCPConfig`: the `episodic_*` knobs live on the subclass."""
    defaults = {
        "mongodb_connection_string": "mongodb://localhost:27017",
        "_env_file": None,
    }
    defaults.update(overrides)
    return MemoryConfig(**defaults)


class _RecordingDB:
    """A db whose ``delete_many`` calls are all recorded, keyed by collection."""

    def __init__(self, *, failing: set[str] | None = None, counts: dict | None = None):
        self.calls: dict[str, dict] = {}
        self._failing = failing or set()
        self._counts = counts or {}
        self._cols: dict[str, MagicMock] = {}

    def __getitem__(self, name: str):
        if name not in self._cols:
            col = MagicMock()

            async def _delete_many(query, _name=name):
                self.calls[_name] = query
                if _name in self._failing:
                    raise RuntimeError(f"{_name} unavailable")
                return MagicMock(deleted_count=self._counts.get(_name, 1))

            col.delete_many = _delete_many
            self._cols[name] = col
        return self._cols[name]


class TestWipeIsComplete:
    """"Delete all my data" has to mean all of it."""

    async def test_every_user_scoped_collection_is_cleared(self) -> None:
        """The finding: four of seven collections were left behind.

        Asserted against the collection-names module rather than a copied list,
        so adding a user-scoped collection without adding it to the wipe fails
        here instead of shipping.
        """
        from agent_memory.core import collections as names

        db = _RecordingDB()
        await AdminService(db).wipe_user_data("alice")

        expected = {
            names.MEMORIES,
            names.SEMANTIC_CACHE,
            names.AUDIT_LOG,
            names.EPISODES,
            names.DECISIONS,
            names.RATE_LIMITS,
            names.EPISODES_COUNTERS,
        }
        assert set(db.calls) == expected, (
            f"collections never cleared: {expected - set(db.calls)}"
        )

    async def test_episodes_and_decisions_are_included(self) -> None:
        """Named individually because these two are the user-visible ones.

        A user who asked to be forgotten kept their entire activity log and every
        sticky decision made about them.
        """
        db = _RecordingDB()
        result = await AdminService(db).wipe_user_data("alice")
        assert db.calls["episodes"] == {"user_id": "alice"}
        assert db.calls["decisions"] == {"user_id": "alice"}
        assert "episodes_deleted" in result and "decisions_deleted" in result

    async def test_the_counter_filter_matches_the_composite_id(self) -> None:
        """`episodes_counters` has no top-level `user_id`.

        Its `_id` is `{user_id, thread_id}`, so `{"user_id": ...}` would delete
        nothing while reporting a successful wipe — the failure mode this asserts
        against is a query that is silently always-empty.
        """
        db = _RecordingDB()
        await AdminService(db).wipe_user_data("alice")
        assert db.calls["episodes_counters"] == {"_id.user_id": "alice"}

    async def test_counts_are_reported_per_collection(self) -> None:
        db = _RecordingDB(counts={"memories": 5, "episodes": 12, "decisions": 2})
        result = await AdminService(db).wipe_user_data("alice")
        assert result["memories_deleted"] == 5
        assert result["episodes_deleted"] == 12
        assert result["decisions_deleted"] == 2
        assert result["complete"] is True

    async def test_one_failing_collection_does_not_abandon_the_rest(self) -> None:
        """Stopping at the first error leaves the most data behind.

        The old shape had no error handling, so a failure on `memories` — the
        first call — meant nothing at all was deleted and the caller saw an
        exception with no record of what had happened.

        It is raised rather than returned — see
        `TestAPartialWipeIsNotASuccess` — but the point asserted here is
        unchanged: every remaining collection is still attempted, and what did
        and did not get deleted is reported.
        """
        from agent_memory.services.admin import PartialWipeError

        db = _RecordingDB(failing={"memories"})
        with pytest.raises(PartialWipeError) as exc_info:
            await AdminService(db).wipe_user_data("alice")

        assert "episodes" in db.calls, "a failure on the first collection stopped the wipe"
        assert "memories" in exc_info.value.errors
        assert exc_info.value.counts["memories_deleted"] == 0

    async def test_a_clean_wipe_reports_complete(self) -> None:
        result = await AdminService(_RecordingDB()).wipe_user_data("alice")
        assert result["complete"] is True
        assert "errors" not in result


class TestRetentionStatesItsScope:
    """C3: the operation is collection-wide and now says so."""

    def _service(self, *, command=None, create=None, drop=None):
        from agent_memory.services.episodic import EpisodicService

        episodes = MagicMock()
        episodes.name = "episodes"
        episodes.database = MagicMock()
        episodes.database.command = command or AsyncMock(return_value={"ok": 1})
        episodes.create_index = create or AsyncMock()
        episodes.drop_index = drop or AsyncMock()
        return EpisodicService(episodes, _config(), MagicMock(), worker=MagicMock())

    async def test_updating_declares_collection_scope(self) -> None:
        """Without this, the `user_id` in the signature reads as the scope."""
        result = await self._service().set_retention(7200)
        assert result["scope"] == "collection"
        assert result["status"] == "updated"

    async def test_removal_declares_collection_scope(self) -> None:
        result = await self._service().set_retention(None)
        assert result["scope"] == "collection"

    async def test_the_create_index_fallback_declares_it_too(self) -> None:
        svc = self._service(command=AsyncMock(side_effect=RuntimeError("no collMod")))
        result = await svc.set_retention(3600)
        assert result["status"] == "created"
        assert result["scope"] == "collection"

    async def test_errors_declare_it_too(self) -> None:
        """A caller handling the error path still needs to know what it affected."""
        svc = self._service(
            command=AsyncMock(side_effect=RuntimeError("no collMod")),
            create=AsyncMock(side_effect=RuntimeError("denied")),
        )
        result = await svc.set_retention(3600)
        assert result["status"] == "error"
        assert result["scope"] == "collection"

    async def test_retention_is_withheld_from_power_user(self) -> None:
        """The governance half of the honesty fix.

        If a tenant role could call this, one tenant could shorten every other
        tenant's retention. Stating the scope in the docstring is not a control;
        withholding the operation is.
        """
        from agent_memory.services.governance import _DEFAULT_PROFILES

        for role in ("power_user", "end_user"):
            allowed = _DEFAULT_PROFILES[role]["allowed_operations"]
            assert "set_activity_retention" not in allowed, (
                f"{role} can retune every tenant's episodic retention"
            )
        assert _DEFAULT_PROFILES["admin"]["allowed_operations"] == ["*"]


class TestRetentionIsNotOpenByDefault:
    """The governance profile above is a control only when governance is *on*.

    ``governance_enabled`` defaults to False, so on a stock multi-tenant
    deployment ``_check_access`` skipped the profile check entirely and the
    ``admin`` category bought nothing. Any authenticated ``end_user`` could
    retune — or delete, via a short TTL — every other tenant's episodic turns
    through the public REST endpoint, and quietly: Atlas expires the documents
    on the TTL monitor's own schedule, so the caller sees only
    ``{"scope": "collection"}`` and the data goes away later.

    An authorisation rule that exists only when an optional subsystem is
    enabled is a default-open rule. These tests pin the floor underneath it.
    """

    @staticmethod
    def _facade(**overrides):
        from agent_memory.memory import AsyncMemory

        m = AsyncMemory.__new__(AsyncMemory)
        m.config = _config(**overrides)
        m.audit_service = AsyncMock()
        m.episodic_service = AsyncMock()
        m.episodic_service.set_retention = AsyncMock(
            return_value={"status": "updated", "scope": "collection"}
        )
        # Governance *off* — the whole point. `rate_limiter` off too, so nothing
        # but the guard under test can refuse.
        m.governance_service = None
        m.rate_limiter = None
        return m

    @staticmethod
    def _authed(**overrides):
        return TestRetentionIsNotOpenByDefault._facade(
            auth_enabled=True, auth_secret="s" * 32, **overrides
        )

    @pytest.mark.parametrize("role", ["end_user", "power_user", "viewer", None])
    async def test_a_non_admin_role_cannot_retune_every_tenant(self, role) -> None:
        """TC-ADMIN-RET-001: the finding, with governance off as shipped.

        ``role=None`` is the case that made this reachable in practice: no role
        claim on the token falls back to ``auth_default_role``, which is
        ``end_user``.
        """
        from agent_memory.exceptions import AccessError

        m = self._authed()
        with pytest.raises(AccessError, match="requires the 'admin' role"):
            await m.set_activity_retention("alice", ttl_seconds=1, role=role)
        m.episodic_service.set_retention.assert_not_awaited()

    async def test_an_admin_can_still_retune(self) -> None:
        """TC-ADMIN-RET-002: the guard must not break the operation's owner."""
        m = self._authed()
        result = await m.set_activity_retention("root", ttl_seconds=7200, role="admin")
        assert result["scope"] == "collection"
        m.episodic_service.set_retention.assert_awaited_once_with(7200)

    async def test_the_configured_default_role_is_honoured(self) -> None:
        """TC-ADMIN-RET-003: a deployment whose default role *is* admin.

        The guard reads the same ``auth_default_role`` fallback ``_check_access``
        uses, rather than hard-refusing a missing claim — otherwise this
        deployment could not call the operation at all.
        """
        m = self._authed(auth_default_role="admin")
        await m.set_activity_retention("root", ttl_seconds=60, role=None)
        m.episodic_service.set_retention.assert_awaited_once()

    async def test_single_tenant_deployments_are_unaffected(self) -> None:
        """TC-ADMIN-RET-004: with auth off there is one tenant and no role claim.

        "Collection-wide" means "the only tenant's own data", so refusing here
        would remove a working operation from every library and stdio user in
        exchange for closing nothing. ``require_auth_for_multi_tenant`` is the
        control for deployments where that posture is unacceptable.
        """
        m = self._facade()
        assert m.config.auth_enabled is False
        await m.set_activity_retention("u1", ttl_seconds=30, role="end_user")
        m.episodic_service.set_retention.assert_awaited_once()

    async def test_the_refusal_is_audited_as_denied(self) -> None:
        """TC-ADMIN-RET-005: the one cross-tenant attempt worth seeing.

        The guard runs *inside* the ``_run`` coroutine factory for this reason.
        Raising before ``_run`` would refuse correctly and leave no record, so a
        credential probing for reachable operations would still be invisible —
        the exact gap the audited-access-check change closed.
        """
        from agent_memory.exceptions import AccessError

        m = self._authed()
        with pytest.raises(AccessError):
            await m.set_activity_retention("alice", ttl_seconds=1, role="end_user")

        m.audit_service.log.assert_awaited_once()
        args = m.audit_service.log.call_args.args
        assert args[0] == "alice"
        assert args[2] == "set_activity_retention"
        assert args[3] == "denied", f"a cross-tenant attempt audited as {args[3]!r}"


class TestStepCountersAreUserScoped:
    """A caller-supplied `thread_id` is not a namespace."""

    def _worker(self, counters):
        from agent_memory.services.episodic_worker import EpisodicWorker

        # (collection, counter_collection, providers, config) — config is fourth.
        return EpisodicWorker(MagicMock(), counters, MagicMock(), _config())

    async def test_the_counter_id_includes_the_user(self) -> None:
        """Two tenants both naming a thread "main" shared one sequence.

        Each then saw its own step numbers skip, which is exactly the property
        the durable counter exists to provide.
        """
        counters = MagicMock()
        counters.find_one_and_update = AsyncMock(return_value={"seq": 1})
        doc = {
            "user_id": "alice",
            "__assign_step": {"user_id": "alice", "thread_id": "main"},
        }
        await self._worker(counters)._assign_durable_step(doc)

        query = counters.find_one_and_update.await_args.args[0]
        assert query == {"_id": {"user_id": "alice", "thread_id": "main"}}
        assert doc["step"] == 0 and doc["parent_step"] is None

    async def test_a_bare_thread_id_is_still_tolerated(self) -> None:
        """Documents enqueued by an older build mid-upgrade must not be dropped."""
        counters = MagicMock()
        counters.find_one_and_update = AsyncMock(return_value={"seq": 3})
        doc = {"user_id": "alice", "__assign_step": "main"}
        await self._worker(counters)._assign_durable_step(doc)

        query = counters.find_one_and_update.await_args.args[0]
        assert query == {"_id": {"user_id": "alice", "thread_id": "main"}}
        assert doc["step"] == 2 and doc["parent_step"] == 1

    async def test_log_activity_builds_the_composite_key(self) -> None:
        """The write side of the same fact, asserted at the enqueue boundary."""
        from agent_memory.services.episodic import EpisodicService

        worker = MagicMock()
        svc = EpisodicService(MagicMock(), _config(), MagicMock(), worker=worker)
        svc.log_activity("alice", "main", [{"type": "human", "content": "hello there"}])

        doc = worker.enqueue.call_args.args[0]
        assert doc["__assign_step"] == {"user_id": "alice", "thread_id": "main"}


class TestPoolSharingIsChecked:
    """One process, one pool — but not silently across different targets."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        DatabaseManager._instance = None
        yield
        DatabaseManager._instance = None

    def _client(self):
        client = AsyncMock()
        client.__getitem__ = MagicMock(return_value=MagicMock())
        client.admin = MagicMock()
        client.admin.command = AsyncMock(return_value={"ok": 1})
        client.close = AsyncMock()
        return client

    async def test_an_equivalent_config_shares_the_pool(self) -> None:
        """The behaviour worth keeping: TRANSPORT=both runs on one connection."""
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = self._client()
            a = await DatabaseManager.initialize(_config())
            b = await DatabaseManager.initialize(_config())
            assert a is b
            assert cls.call_count == 1, "a second Atlas connection pool was opened"

    async def test_a_different_database_is_refused(self) -> None:
        """The finding: this used to return the first pool and ignore the config.

        Data written by the second facade landed in the first facade's database,
        with nothing logged.
        """
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = self._client()
            await DatabaseManager.initialize(_config(mongodb_database_name="one"))
            with pytest.raises(ValueError, match="different MongoDB target"):
                await DatabaseManager.initialize(_config(mongodb_database_name="two"))

    async def test_a_different_cluster_is_refused(self) -> None:
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = self._client()
            await DatabaseManager.initialize(_config())
            with pytest.raises(ValueError, match="different MongoDB target"):
                await DatabaseManager.initialize(
                    _config(mongodb_connection_string="mongodb://elsewhere:27017")
                )

    async def test_the_error_does_not_leak_the_connection_string(self) -> None:
        """The fingerprint is hashed: this message reaches logs and stderr."""
        secret = "mongodb://user:sup3rsecret@host/db"
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = self._client()
            await DatabaseManager.initialize(_config(mongodb_connection_string=secret))
            with pytest.raises(ValueError) as exc:
                await DatabaseManager.initialize(_config())
        assert "sup3rsecret" not in str(exc.value)

    async def test_pool_sizes_do_not_count_as_a_different_target(self) -> None:
        """Tuning differences are not worth refusing to start over."""
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = self._client()
            a = await DatabaseManager.initialize(_config(mongodb_max_pool_size=20))
            b = await DatabaseManager.initialize(_config(mongodb_max_pool_size=50))
            assert a is b

    async def test_the_first_holder_closing_does_not_break_the_second(self) -> None:
        """The other half of the sharing bug.

        `close()` reset the class-level `_instance` unconditionally, so in
        TRANSPORT=both, shutting down one shell left the other holding a closed
        client — every later query failing on a connection someone else ended.
        """
        client = self._client()
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = client
            first = await DatabaseManager.initialize(_config())
            second = await DatabaseManager.initialize(_config())

            await first.close()
            client.close.assert_not_awaited()
            assert second.db is not None
            assert await DatabaseManager.get_instance() is second

            await second.close()
            client.close.assert_awaited_once()

    async def test_the_last_holder_closing_releases_the_pool(self) -> None:
        client = self._client()
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = client
            instance = await DatabaseManager.initialize(_config())
            await instance.close()
            client.close.assert_awaited_once()
            with pytest.raises(RuntimeError):
                await DatabaseManager.get_instance()

    async def test_closing_twice_is_harmless(self) -> None:
        """Teardown paths run twice; that must not raise."""
        client = self._client()
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = client
            instance = await DatabaseManager.initialize(_config())
            await instance.close()
            await instance.close()
            client.close.assert_awaited_once()

    async def test_a_new_config_works_after_a_full_close(self) -> None:
        """The refusal is about concurrent use, not a permanent lock."""
        with patch("agent_memory.core.database.AsyncMongoClient") as cls:
            cls.return_value = self._client()
            first = await DatabaseManager.initialize(_config(mongodb_database_name="one"))
            await first.close()
            second = await DatabaseManager.initialize(
                _config(mongodb_database_name="two")
            )
            assert second is not first
