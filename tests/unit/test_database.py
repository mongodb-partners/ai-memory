"""Tests for DatabaseManager (PyMongo Async singleton)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.core.database import DatabaseManager


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset DatabaseManager singleton between tests."""
    DatabaseManager._instance = None
    yield
    DatabaseManager._instance = None


class TestDatabaseManagerInitialize:
    """TC-004: DatabaseManager initializes correctly."""

    async def test_initialize_creates_singleton(self):
        config = _make_config()
        with patch("agent_memory.core.database.AsyncMongoClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client.admin = MagicMock()
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_client_cls.return_value = mock_client

            instance = await DatabaseManager.initialize(config)
            assert instance is not None
            assert instance.db is not None

    async def test_initialize_twice_returns_same_instance(self):
        config = _make_config()
        with patch("agent_memory.core.database.AsyncMongoClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client.admin = MagicMock()
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_client_cls.return_value = mock_client

            instance1 = await DatabaseManager.initialize(config)
            instance2 = await DatabaseManager.initialize(config)
            assert instance1 is instance2

    async def test_initialize_pings_server(self):
        config = _make_config()
        with patch("agent_memory.core.database.AsyncMongoClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client.admin = MagicMock()
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_client_cls.return_value = mock_client

            await DatabaseManager.initialize(config)
            mock_client.admin.command.assert_called_once_with("ping")

    async def test_initialize_raises_on_connection_failure(self):
        config = _make_config()
        with patch("agent_memory.core.database.AsyncMongoClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client.admin = MagicMock()
            mock_client.admin.command = AsyncMock(side_effect=ConnectionError("fail"))
            mock_client_cls.return_value = mock_client

            with pytest.raises(ConnectionError):
                await DatabaseManager.initialize(config)


class TestDatabaseManagerGetInstance:
    """TC-005: get_instance returns singleton or raises."""

    async def test_get_instance_before_init_raises(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            await DatabaseManager.get_instance()

    async def test_get_instance_after_init_returns_instance(self):
        config = _make_config()
        with patch("agent_memory.core.database.AsyncMongoClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client.admin = MagicMock()
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_client_cls.return_value = mock_client

            await DatabaseManager.initialize(config)
            instance = await DatabaseManager.get_instance()
            assert instance is not None


class TestDatabaseManagerClose:
    """TC-006: close() cleans up properly."""

    async def test_close_resets_singleton(self):
        config = _make_config()
        with patch("agent_memory.core.database.AsyncMongoClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client.admin = MagicMock()
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            instance = await DatabaseManager.initialize(config)
            await instance.close()

            with pytest.raises(RuntimeError):
                await DatabaseManager.get_instance()

    async def test_db_property_raises_after_close(self):
        config = _make_config()
        with patch("agent_memory.core.database.AsyncMongoClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__getitem__ = MagicMock(return_value=MagicMock())
            mock_client.admin = MagicMock()
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            instance = await DatabaseManager.initialize(config)
            await instance.close()

            with pytest.raises(RuntimeError, match="not connected"):
                _ = instance.db
