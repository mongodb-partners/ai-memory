"""Tests for GovernanceService."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.services.governance import GovernanceService, _DEFAULT_PROFILES


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_collection():
    col = MagicMock()
    col.find_one = AsyncMock(return_value=None)
    col.insert_one = AsyncMock()
    return col


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

        # All profiles already exist
        col.find_one = AsyncMock(return_value={"_id": "existing", "role": "admin"})

        count = await svc.seed_defaults()

        assert count == 0
        col.insert_one.assert_not_called()
