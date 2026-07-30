"""Tests for GovernanceService."""

import time
from unittest.mock import AsyncMock, MagicMock

from agent_memory.core.config import MCPConfig
from agent_memory.services.governance import _DEFAULT_PROFILES, GovernanceService


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_collection():
    col = MagicMock()
    col.find_one = AsyncMock(return_value=None)
    col.insert_one = AsyncMock()
    col.update_one = AsyncMock()
    return col


def _complete_profile(role: str) -> dict:
    """An existing profile that already has every default operation."""
    return {
        "_id": "existing",
        "role": role,
        "allowed_operations": list(_DEFAULT_PROFILES[role]["allowed_operations"]),
    }


class TestGetProfile:
    """get_profile fetches from DB and caches."""

    async def test_profile_from_db(self):
        col = _make_collection()
        config = _make_config()
        svc = GovernanceService(col, config)

        db_profile = {"role": "admin", "max_memories_per_day": 10000, "allowed_operations": ["*"]}
        col.find_one = AsyncMock(return_value={"_id": "123", **db_profile})

        result = await svc.get_profile("admin")
        assert result["role"] == "admin"
        assert "_id" not in result

    async def test_profile_cached(self):
        col = _make_collection()
        config = _make_config(governance_cache_ttl_seconds=300)
        svc = GovernanceService(col, config)

        db_profile = {"role": "admin", "allowed_operations": ["*"]}
        col.find_one = AsyncMock(return_value={"_id": "123", **db_profile})

        await svc.get_profile("admin")
        await svc.get_profile("admin")

        # Only one DB call due to caching
        assert col.find_one.call_count == 1

    async def test_profile_expired_cache(self):
        col = _make_collection()
        config = _make_config(governance_cache_ttl_seconds=0)
        svc = GovernanceService(col, config)

        db_profile = {"role": "admin", "allowed_operations": ["*"]}
        col.find_one = AsyncMock(return_value={"_id": "123", **db_profile})

        await svc.get_profile("admin")
        # With TTL=0, next call should go to DB again
        await svc.get_profile("admin")

        assert col.find_one.call_count == 2

    async def test_profile_fallback_default(self):
        col = _make_collection()
        config = _make_config()
        svc = GovernanceService(col, config)

        col.find_one = AsyncMock(return_value=None)

        result = await svc.get_profile("unknown_role")
        # Falls back to governance_default_profile
        assert "allowed_operations" in result


class TestCheckAllowed:
    """check_allowed validates operation against profile."""

    async def test_admin_allowed_all(self):
        col = _make_collection()
        config = _make_config()
        svc = GovernanceService(col, config)

        col.find_one = AsyncMock(return_value={
            "_id": "123",
            "role": "admin",
            "allowed_operations": ["*"],
        })

        assert await svc.check_allowed("user1", "admin", "wipe_user_data") is True

    async def test_end_user_denied_admin_op(self):
        col = _make_collection()
        config = _make_config()
        svc = GovernanceService(col, config)

        col.find_one = AsyncMock(return_value={
            "_id": "123",
            "role": "end_user",
            "allowed_operations": ["store_memory", "recall_memory"],
        })

        assert await svc.check_allowed("user1", "end_user", "wipe_user_data") is False

    async def test_allowed_specific_op(self):
        col = _make_collection()
        config = _make_config()
        svc = GovernanceService(col, config)

        col.find_one = AsyncMock(return_value={
            "_id": "123",
            "role": "end_user",
            "allowed_operations": ["store_memory", "recall_memory"],
        })

        assert await svc.check_allowed("user1", "end_user", "store_memory") is True


class TestSeedDefaults:
    """seed_defaults inserts default profiles."""

    async def test_seed_inserts_all(self):
        col = _make_collection()
        config = _make_config()
        svc = GovernanceService(col, config)

        col.find_one = AsyncMock(return_value=None)

        count = await svc.seed_defaults()

        assert count == len(_DEFAULT_PROFILES)
        assert col.insert_one.call_count == len(_DEFAULT_PROFILES)

    async def test_seed_skips_existing(self):
        col = _make_collection()
        config = _make_config()
        svc = GovernanceService(col, config)

        # Every profile already exists AND is already complete.
        col.find_one = AsyncMock(side_effect=lambda q: _complete_profile(q["role"]))

        count = await svc.seed_defaults()

        assert count == 0
        col.insert_one.assert_not_called()
        col.update_one.assert_not_called()

    async def test_seed_backfills_operations_added_by_an_upgrade(self):
        """An existing profile must gain operations a release added.

        Skip-if-exists would leave every upgraded deployment denying the new
        operation, and the symptom is an AccessError on the feature the user
        upgraded to get.
        """
        col = _make_collection()
        svc = GovernanceService(col, _make_config())

        # A profile frozen at an older release: it predates episodic memory.
        stale = {"_id": "x", "role": "end_user", "allowed_operations": ["store_memory"]}
        col.find_one = AsyncMock(
            side_effect=lambda q: stale if q["role"] == "end_user"
            else _complete_profile(q["role"])
        )

        count = await svc.seed_defaults()

        assert count == 1
        col.insert_one.assert_not_called()
        update = col.update_one.await_args
        assert update.args[0] == {"role": "end_user"}
        added = update.args[1]["$addToSet"]["allowed_operations"]["$each"]
        # Additive only: what it already had is not re-added.
        assert "store_memory" not in added
        assert "log_activity" in added
        assert "search_activity" in added

    async def test_backfill_does_not_remove_operator_additions(self):
        """An operation an operator added by hand must survive seeding."""
        col = _make_collection()
        svc = GovernanceService(col, _make_config())
        custom = {
            "_id": "x",
            "role": "end_user",
            "allowed_operations": ["custom_operation"],
        }
        col.find_one = AsyncMock(
            side_effect=lambda q: custom if q["role"] == "end_user"
            else _complete_profile(q["role"])
        )

        await svc.seed_defaults()

        update = col.update_one.await_args.args[1]
        # $addToSet only ever adds, and nothing $unsets or overwrites the list.
        assert set(update) == {"$addToSet", "$set"}
        assert set(update["$set"]) == {"updated_at"}

    async def test_backfill_evicts_the_stale_cache_entry(self):
        """The cached copy is stale in the direction that denies access."""
        col = _make_collection()
        svc = GovernanceService(col, _make_config())
        svc._cache["end_user"] = {"role": "end_user", "allowed_operations": []}
        svc._cache_time["end_user"] = time.time()
        col.find_one = AsyncMock(
            side_effect=lambda q: {"_id": "x", "role": "end_user", "allowed_operations": []}
            if q["role"] == "end_user"
            else _complete_profile(q["role"])
        )

        await svc.seed_defaults()

        assert "end_user" not in svc._cache
        assert "end_user" not in svc._cache_time


class TestEpisodicOperations:
    """The episodic operations must be reachable, and scoped by role."""

    def _svc(self):
        """A service whose DB serves the seeded profile for each role."""
        col = _make_collection()
        col.find_one = AsyncMock(side_effect=lambda q: _complete_profile(q["role"]))
        return GovernanceService(col, _make_config())

    async def test_power_user_gets_full_episodic_access(self):
        svc = self._svc()
        for op in ("log_activity", "search_activity", "get_thread", "get_correlation"):
            assert await svc.check_allowed("u1", "power_user", op) is True

    async def test_end_user_may_log_and_replay_its_own_threads(self):
        svc = self._svc()
        for op in ("log_activity", "search_activity", "get_thread"):
            assert await svc.check_allowed("u1", "end_user", op) is True

    async def test_end_user_cannot_query_by_correlation_id(self):
        """Trace ids come from operators, not from the user's own session."""
        svc = self._svc()
        assert await svc.check_allowed("u1", "end_user", "get_correlation") is False

    async def test_retention_stays_admin_only(self):
        svc = self._svc()
        assert await svc.check_allowed("u1", "power_user", "set_activity_retention") is False
        assert await svc.check_allowed("u1", "admin", "set_activity_retention") is True
