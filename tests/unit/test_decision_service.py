"""Tests for DecisionService."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.services.decision import DecisionService


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_collection():
    col = MagicMock()
    col.update_one = AsyncMock(return_value=MagicMock(upserted_id="new_id"))
    col.find_one = AsyncMock(return_value=None)
    return col


class TestDecisionStore:
    """store() creates or updates decisions."""

    async def test_store_new_decision(self):
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        col.update_one = AsyncMock(return_value=MagicMock(upserted_id="new_id"))

        result = await svc.store("user1", "editor", "vim")
        assert result == "stored"
        col.update_one.assert_called_once()

        call_args = col.update_one.call_args
        assert call_args[0][0] == {"user_id": "user1", "key": "editor"}
        assert call_args[1]["upsert"] is True

    async def test_store_upsert_updates(self):
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        col.update_one = AsyncMock(return_value=MagicMock(upserted_id=None))

        result = await svc.store("user1", "editor", "emacs")
        assert result == "updated"

    async def test_store_custom_ttl(self):
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        col.update_one = AsyncMock(return_value=MagicMock(upserted_id="new"))

        await svc.store("user1", "theme", "dark", ttl_days=7)

        update_doc = col.update_one.call_args[0][1]
        expires_at = update_doc["$set"]["expires_at"]
        now = datetime.now(timezone.utc)
        # TTL should be roughly 7 days from now
        delta = expires_at - now
        assert 6 <= delta.days <= 7

    async def test_store_default_ttl(self):
        col = _make_collection()
        config = _make_config(decision_default_ttl_days=30)
        svc = DecisionService(col, config)

        col.update_one = AsyncMock(return_value=MagicMock(upserted_id="new"))

        await svc.store("user1", "theme", "dark")

        update_doc = col.update_one.call_args[0][1]
        expires_at = update_doc["$set"]["expires_at"]
        now = datetime.now(timezone.utc)
        delta = expires_at - now
        assert 29 <= delta.days <= 30


class TestDecisionRecall:
    """recall() retrieves decisions by key."""

    async def test_recall_found(self):
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        now = datetime.now(timezone.utc)
        col.find_one = AsyncMock(return_value={
            "key": "editor",
            "value": "vim",
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(days=30),
        })

        result = await svc.recall("user1", "editor")
        assert result is not None
        assert result["key"] == "editor"
        assert result["value"] == "vim"

    async def test_recall_not_found(self):
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        col.find_one = AsyncMock(return_value=None)

        result = await svc.recall("user1", "nonexistent")
        assert result is None

    async def test_recall_filters_expired(self):
        """recall query filters for expires_at > now."""
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        col.find_one = AsyncMock(return_value=None)

        await svc.recall("user1", "expired_key")

        query = col.find_one.call_args[0][0]
        assert query["user_id"] == "user1"
        assert query["key"] == "expired_key"
        assert "$gt" in query["expires_at"]


class TestDecisionSeedDefaults:
    """seed_defaults inserts system-default decisions."""

    async def test_seed_inserts_all(self):
        """TC-E-008: All system defaults inserted on empty collection."""
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        col.find_one = AsyncMock(return_value=None)
        col.update_one = AsyncMock(return_value=MagicMock(upserted_id="new"))

        count = await svc.seed_defaults()

        from agent_memory.services.decision import _SYSTEM_DEFAULTS
        assert count == len(_SYSTEM_DEFAULTS)

    async def test_seed_skips_existing(self):
        """TC-E-009: Existing system decisions are not overwritten."""
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        now = datetime.now(timezone.utc)
        col.find_one = AsyncMock(return_value={
            "key": "system:governance_profile",
            "value": "admin",
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(days=365),
        })

        count = await svc.seed_defaults()

        assert count == 0

    async def test_seed_returns_count(self):
        """TC-E-010: Returns count of inserted decisions."""
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        call_count = 0
        async def find_one_side_effect(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First recall returns existing
                now = datetime.now(timezone.utc)
                return {
                    "key": "exists",
                    "value": "val",
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": now + timedelta(days=365),
                }
            return None

        col.find_one = AsyncMock(side_effect=find_one_side_effect)
        col.update_one = AsyncMock(return_value=MagicMock(upserted_id="new"))

        count = await svc.seed_defaults()

        from agent_memory.services.decision import _SYSTEM_DEFAULTS
        assert count == len(_SYSTEM_DEFAULTS) - 1

    async def test_seed_uses_system_user_id(self):
        """System decisions use user_id='system'."""
        col = _make_collection()
        config = _make_config()
        svc = DecisionService(col, config)

        col.find_one = AsyncMock(return_value=None)
        col.update_one = AsyncMock(return_value=MagicMock(upserted_id="new"))

        await svc.seed_defaults()

        # Check that recall was called with user_id="system"
        recall_query = col.find_one.call_args[0][0]
        assert recall_query["user_id"] == "system"
