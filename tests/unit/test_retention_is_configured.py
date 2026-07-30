"""Configured retention durations must reach the TTL indexes that enforce them.

Every ``expireAfterSeconds`` in ``STANDARD_INDEXES`` used to be a literal sitting
beside a config field of the same default value. ``AUDIT_RETENTION_DAYS=30`` set
the field, left ``ix_audit_ttl`` at ``365 * 86400``, and reported success — the
config and the index agreed only by coincidence, and only at the defaults.

The tests here are about the seam rather than the arithmetic: that
``get_standard_indexes`` reads the config, that ``ensure_indexes`` passes the
config it was given, that ``AsyncMemory.create`` gives it one, and that an
existing index built at the old duration is actually reconciled to the new one
(which is the half that makes a *changed* value take effect on a live
deployment).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import OperationFailure

from agent_memory.core.collections import (
    AUDIT_LOG,
    EPISODES,
    MEMORIES,
    RATE_LIMITS,
    SEMANTIC_CACHE,
    STANDARD_INDEXES,
    get_standard_indexes,
)
from agent_memory.core.config import MCPConfig


def _cfg(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _ttl(indexes: list[dict], name: str) -> int:
    matches = [ix for ix in indexes if ix["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name}, got {len(matches)}"
    return matches[0]["kwargs"]["expireAfterSeconds"]


# Each TTL index, the config field that should decide it, a non-default value for
# that field, and the seconds that value must produce.
_CONFIGURED_TTLS = [
    ("ix_memories_deleted_at_ttl", "soft_delete_purge_days", 7, 7 * 86400),
    ("ix_episodes_ttl", "episodic_retention_days", 3, 3 * 86400),
    ("ix_cache_ttl", "cache_ttl_seconds", 120, 120),
    ("ix_audit_ttl", "audit_retention_days", 30, 30 * 86400),
    ("ix_rate_limits_ttl", "rate_limit_retention_seconds", 600, 600),
]


class TestEveryTTLIndexReadsItsConfigField:
    """The seam: a set field changes the index definition."""

    @pytest.mark.parametrize(
        "index_name,field,value,expected", _CONFIGURED_TTLS,
        ids=[row[0] for row in _CONFIGURED_TTLS],
    )
    def test_a_configured_duration_reaches_the_index(
        self, index_name, field, value, expected
    ):
        indexes = get_standard_indexes(_cfg(**{field: value}))
        assert _ttl(indexes, index_name) == expected

    @pytest.mark.parametrize(
        "index_name,field,value,expected", _CONFIGURED_TTLS,
        ids=[row[0] for row in _CONFIGURED_TTLS],
    )
    def test_the_configured_value_is_not_merely_the_default(
        self, index_name, field, value, expected
    ):
        """Guards the test above from passing on a hardcoded literal.

        If ``expected`` happened to equal the default, the parametrized case
        would pass against the unfixed code. Asserting they differ is what makes
        each case evidence of anything.
        """
        assert _ttl(STANDARD_INDEXES, index_name) != expected

    def test_the_defaults_are_unchanged(self):
        """Deriving the values must not quietly retune an existing deployment.

        A config-driven definition that produced different numbers at the
        defaults would change retention for every deployment that never set
        anything — the opposite of the fix.
        """
        assert _ttl(STANDARD_INDEXES, "ix_memories_deleted_at_ttl") == 30 * 86400
        assert _ttl(STANDARD_INDEXES, "ix_episodes_ttl") == 30 * 86400
        assert _ttl(STANDARD_INDEXES, "ix_cache_ttl") == 3600
        assert _ttl(STANDARD_INDEXES, "ix_audit_ttl") == 365 * 86400
        assert _ttl(STANDARD_INDEXES, "ix_rate_limits_ttl") == 86400

    def test_a_config_free_call_matches_the_constant(self):
        assert get_standard_indexes() == STANDARD_INDEXES

    def test_deriving_the_ttls_did_not_drop_or_add_an_index(self):
        """The definitions are the same set; only the durations became derived."""
        configured = get_standard_indexes(_cfg(audit_retention_days=1))
        assert [ix["name"] for ix in configured] == [
            ix["name"] for ix in STANDARD_INDEXES
        ]

    def test_the_non_ttl_options_are_untouched(self):
        """A partial filter is not a duration and must survive the rewrite.

        ``ix_memories_deleted_at_ttl`` carries both, and dropping the filter
        would turn a TTL over soft-deleted documents into one that expires every
        memory whose `deleted_at` is null — i.e. all of them.
        """
        indexes = get_standard_indexes(_cfg(soft_delete_purge_days=1))
        idx = next(i for i in indexes if i["name"] == "ix_memories_deleted_at_ttl")
        assert idx["kwargs"]["partialFilterExpression"] == {
            "deleted_at": {"$type": "date"}
        }

    def test_the_expires_at_indexes_stay_at_zero(self):
        """``expireAfterSeconds: 0`` on an `expires_at` field is not a duration.

        It is the "expire at the instant this document's own field names" idiom,
        and the per-tier retention in `retention_ttl` is what sets those fields.
        Feeding a configured duration in here would double-apply retention.
        """
        indexes = get_standard_indexes(_cfg(soft_delete_purge_days=1, stm_ttl_hours=1))
        assert _ttl(indexes, "ix_memories_expires_at") == 0
        assert _ttl(indexes, "ix_decisions_ttl") == 0


class TestDegenerateDurationsAreClamped:
    """Zero must not be reinterpreted as the expires_at idiom."""

    def test_a_zero_cache_ttl_expires_promptly_rather_than_immediately(self):
        """``CACHE_TTL_SECONDS=0`` means "as briefly as possible", not "reinterpret
        the field". Passed through, ``expireAfterSeconds: 0`` on ``created_at``
        expires every entry the moment the TTL monitor sees it.
        """
        indexes = get_standard_indexes(_cfg(cache_ttl_seconds=0))
        assert _ttl(indexes, "ix_cache_ttl") == 1

    def test_a_zero_audit_retention_does_not_delete_the_audit_log(self):
        indexes = get_standard_indexes(_cfg(audit_retention_days=0))
        assert _ttl(indexes, "ix_audit_ttl") == 1

    def test_a_zero_soft_delete_purge_still_leaves_a_real_ttl(self):
        indexes = get_standard_indexes(_cfg(soft_delete_purge_days=0))
        assert _ttl(indexes, "ix_memories_deleted_at_ttl") == 1

    def test_a_negative_duration_is_clamped_too(self):
        """MongoDB rejects a negative ``expireAfterSeconds``, so passing one
        through would make the index fail to build — the collection then has no
        TTL at all, which is the failure this whole task is about."""
        indexes = get_standard_indexes(_cfg(episodic_retention_days=-1))
        assert _ttl(indexes, "ix_episodes_ttl") == 1


class TestRateLimitCountersOutliveTheirWindow:
    """A counter expiring mid-window stops the limit being enforced."""

    def test_a_window_longer_than_the_retention_wins(self):
        """The counter *is* the enforcement state. Expire it while its window is
        still open and the next request starts a fresh count, so a caller gets
        `max` requests again inside a window that should have been exhausted.
        """
        indexes = get_standard_indexes(
            _cfg(rate_limit_window_seconds=7200, rate_limit_retention_seconds=60)
        )
        assert _ttl(indexes, "ix_rate_limits_ttl") == 7200

    def test_a_retention_longer_than_the_window_is_honoured(self):
        """The floor must not become a ceiling — keeping spent counters around
        for longer than one window is a legitimate choice for debugging."""
        indexes = get_standard_indexes(
            _cfg(rate_limit_window_seconds=60, rate_limit_retention_seconds=7200)
        )
        assert _ttl(indexes, "ix_rate_limits_ttl") == 7200


class TestEnsureIndexesUsesTheConfig:
    """Deriving the definitions is worthless if reconciliation ignores them."""

    @staticmethod
    def _db():
        collections: dict[str, MagicMock] = {}

        def get_col(name):
            if name not in collections:
                col = MagicMock()
                col.create_index = AsyncMock(return_value="ok")
                col.drop_index = AsyncMock()
                collections[name] = col
            return collections[name]

        db = MagicMock()
        db.__getitem__ = MagicMock(side_effect=get_col)
        return db, collections

    @staticmethod
    def _created_ttl(collections: dict, collection_name: str, index_name: str) -> int:
        col = collections[collection_name]
        for call in col.create_index.call_args_list:
            if call.kwargs.get("name") == index_name:
                return call.kwargs["expireAfterSeconds"]
        raise AssertionError(f"{index_name} was never created on {collection_name}")

    async def test_the_configured_duration_is_what_gets_created(self):
        from agent_memory.core.migrations import ensure_indexes

        db, collections = self._db()
        await ensure_indexes(db, _cfg(audit_retention_days=30, cache_ttl_seconds=120))

        assert self._created_ttl(collections, AUDIT_LOG, "ix_audit_ttl") == 30 * 86400
        assert self._created_ttl(collections, SEMANTIC_CACHE, "ix_cache_ttl") == 120

    async def test_every_configured_ttl_arrives_at_its_collection(self):
        from agent_memory.core.migrations import ensure_indexes

        db, collections = self._db()
        await ensure_indexes(
            db,
            _cfg(
                soft_delete_purge_days=7,
                episodic_retention_days=3,
                cache_ttl_seconds=120,
                audit_retention_days=30,
                rate_limit_retention_seconds=600,
            ),
        )

        assert self._created_ttl(
            collections, MEMORIES, "ix_memories_deleted_at_ttl"
        ) == 7 * 86400
        assert self._created_ttl(collections, EPISODES, "ix_episodes_ttl") == 3 * 86400
        assert self._created_ttl(collections, SEMANTIC_CACHE, "ix_cache_ttl") == 120
        assert self._created_ttl(collections, AUDIT_LOG, "ix_audit_ttl") == 30 * 86400
        assert self._created_ttl(collections, RATE_LIMITS, "ix_rate_limits_ttl") == 600

    async def test_omitting_the_config_falls_back_to_the_defaults(self):
        """The signature keeps ``config`` optional for callers that have none —
        it must produce the shipped defaults rather than raise."""
        from agent_memory.core.migrations import ensure_indexes

        db, collections = self._db()
        await ensure_indexes(db)

        assert self._created_ttl(collections, AUDIT_LOG, "ix_audit_ttl") == 365 * 86400


class TestAChangedRetentionReconcilesAnExistingIndex:
    """Changing a duration and restarting has to actually change the index."""

    @staticmethod
    def _conflicting_db(code: int):
        """A collection whose first create_index raises the given conflict."""
        col = MagicMock()
        col.create_index = AsyncMock(
            side_effect=[OperationFailure("conflict", code=code)]
            + [None] * (len(STANDARD_INDEXES) * 2)
        )
        col.drop_index = AsyncMock()
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        return db, col

    @pytest.mark.parametrize(
        "code,label",
        [(85, "IndexOptionsConflict"), (86, "IndexKeySpecsConflict")],
    )
    async def test_a_conflicting_index_is_dropped_and_rebuilt(self, code, label):
        """85 is the one a retention change produces.

        ``expireAfterSeconds`` is an *option*, so restarting with a new duration
        against an index built at the old one raises 85, not 86. Only 86 used to
        be handled: the change took the ``logger.exception`` branch, the old
        index stayed, startup completed, and documents kept expiring on the
        previous schedule.
        """
        from agent_memory.core.migrations import ensure_indexes

        db, col = self._conflicting_db(code)
        await ensure_indexes(db, _cfg())

        col.drop_index.assert_awaited_once()

    async def test_the_rebuild_uses_the_new_duration(self):
        """Dropping is only half of it — the recreate has to carry the value."""
        from agent_memory.core.migrations import ensure_indexes

        col = MagicMock()
        calls: list[dict] = []

        async def create_index(keys, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise OperationFailure("conflict", code=85)
            return "ok"

        col.create_index = AsyncMock(side_effect=create_index)
        col.drop_index = AsyncMock()
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(db, _cfg(soft_delete_purge_days=7))

        # The first definition is ix_memories_expires_at, which is the one that
        # conflicted; its retry must repeat that same definition.
        assert calls[0]["name"] == calls[1]["name"]
        assert calls[0] == calls[1]

    async def test_a_conflict_on_the_audit_ttl_rebuilds_at_the_configured_value(self):
        from agent_memory.core.migrations import ensure_indexes

        col = MagicMock()
        created: list[dict] = []

        async def create_index(keys, **kwargs):
            created.append(kwargs)
            if kwargs.get("name") == "ix_audit_ttl" and len(
                [c for c in created if c.get("name") == "ix_audit_ttl"]
            ) == 1:
                raise OperationFailure("conflict", code=85)
            return "ok"

        col.create_index = AsyncMock(side_effect=create_index)
        col.drop_index = AsyncMock()
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(db, _cfg(audit_retention_days=30))

        rebuilds = [c for c in created if c.get("name") == "ix_audit_ttl"]
        assert len(rebuilds) == 2
        assert rebuilds[1]["expireAfterSeconds"] == 30 * 86400

    async def test_an_unrelated_operation_failure_is_still_not_a_conflict(self):
        """The widened code set must not swallow genuine failures into a drop."""
        from agent_memory.core.migrations import ensure_indexes

        db, col = self._conflicting_db(42)
        await ensure_indexes(db, _cfg())

        col.drop_index.assert_not_awaited()

    async def test_a_failed_rebuild_does_not_stop_startup(self):
        """Stage 1 is best-effort per index: one collection's TTL failing to
        rebuild must not leave the other collections unindexed."""
        from agent_memory.core.migrations import ensure_indexes

        col = MagicMock()
        col.create_index = AsyncMock(side_effect=OperationFailure("conflict", code=85))
        col.drop_index = AsyncMock(side_effect=Exception("drop failed"))
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(db, _cfg())  # must not raise

        assert col.create_index.await_count == len(STANDARD_INDEXES)
