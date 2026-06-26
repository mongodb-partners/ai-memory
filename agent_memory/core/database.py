"""Async-safe MongoDB connection pool singleton using PyMongo Async API."""

import asyncio
from typing import ClassVar

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from agent_memory.core.config import MCPConfig


class DatabaseManager:
    """Async-safe MongoDB connection pool.

    Call ``await DatabaseManager.initialize(config)`` once during FastMCP
    lifespan startup.  After that, tools call
    ``await DatabaseManager.get_instance()`` (no parameters) to obtain the
    shared singleton.

    ``initialize()`` must be called exactly once during lifespan startup
    (single coroutine context).  The class-level ``_lock`` is allocated
    eagerly at class definition time to avoid a TOCTOU race.
    """

    _instance: ClassVar["DatabaseManager | None"] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self) -> None:
        self._client: AsyncMongoClient | None = None
        self._db: AsyncDatabase | None = None

    @classmethod
    async def initialize(cls, config: MCPConfig) -> "DatabaseManager":
        """Create AsyncMongoClient, verify connectivity, cache singleton."""
        async with cls._lock:
            if cls._instance is not None:
                return cls._instance

            instance = cls()
            instance._client = AsyncMongoClient(
                config.mongodb_connection_string,
                maxPoolSize=config.mongodb_max_pool_size,
                minPoolSize=config.mongodb_min_pool_size,
                serverSelectionTimeoutMS=5000,
            )
            instance._db = instance._client[config.mongodb_database_name]

            # Connectivity probe
            try:
                await instance._client.admin.command("ping")
            except Exception:
                instance._client = None
                instance._db = None
                raise

            cls._instance = instance
            return instance

    @classmethod
    async def get_instance(cls) -> "DatabaseManager":
        """Return the cached singleton.  Raises if initialize() not called."""
        if cls._instance is not None:
            return cls._instance
        raise RuntimeError(
            "DatabaseManager not initialized. "
            "Call `await DatabaseManager.initialize(config)` during lifespan startup."
        )

    @property
    def db(self) -> AsyncDatabase:
        if self._db is None:
            raise RuntimeError("DatabaseManager not connected.")
        return self._db

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        self._client = None
        self._db = None
        type(self)._instance = None
