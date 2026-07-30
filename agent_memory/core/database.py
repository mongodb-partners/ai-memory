"""Async-safe MongoDB connection pool singleton using PyMongo Async API."""

import asyncio
import hashlib
from typing import ClassVar

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from agent_memory.core.config import MCPConfig


def _fingerprint(config: MCPConfig) -> str:
    """Identify the pool a config asks for, without retaining the credential.

    Hashed rather than stored: the connection string carries a password, and this
    value ends up in an error message. Only the fields that determine *which*
    server and database are reached are included — pool sizes are tuning, and
    disagreeing about them is not worth refusing to start over.
    """
    raw = f"{config.mongodb_connection_string}\x00{config.mongodb_database_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DatabaseManager:
    """Async-safe MongoDB connection pool, shared per process.

    Call ``await DatabaseManager.initialize(config)`` during startup. After that,
    ``await DatabaseManager.get_instance()`` (no parameters) returns the shared
    singleton.

    The class-level ``_lock`` is allocated eagerly at class definition time to
    avoid a TOCTOU race.

    **Sharing is deliberate, and now checked.** One process, one pool is the point
    — ``TRANSPORT=both`` runs two shells off one Atlas connection. But
    ``initialize()`` used to return the existing instance whatever config it was
    handed, so a second facade built against a *different* cluster or database
    silently read and wrote the first one's data. Nothing logged, and the symptom
    was data appearing in the wrong database. A differing config is now a startup
    error: it is always a mistake, and the only question is whether it is found
    now or in production.

    ``close()`` is reference-counted for the same reason. It used to reset the
    class-level ``_instance`` unconditionally, so the first facade to close
    severed the connection out from under every other holder — in
    ``TRANSPORT=both``, shutting down one shell broke the other. The pool is now
    closed when the last holder releases it.
    """

    _instance: ClassVar["DatabaseManager | None"] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self) -> None:
        self._client: AsyncMongoClient | None = None
        self._db: AsyncDatabase | None = None
        self._fingerprint: str | None = None
        # Number of holders that called initialize() and have not closed yet.
        self._refcount: int = 0

    @classmethod
    async def initialize(cls, config: MCPConfig) -> "DatabaseManager":
        """Create AsyncMongoClient, verify connectivity, cache singleton.

        Returns the existing pool when called again with an equivalent config,
        incrementing its reference count. Raises ``ValueError`` when called with a
        config pointing somewhere else.
        """
        fingerprint = _fingerprint(config)
        async with cls._lock:
            if cls._instance is not None:
                existing = cls._instance._fingerprint
                if existing is not None and existing != fingerprint:
                    raise ValueError(
                        "DatabaseManager is already connected to a different "
                        f"MongoDB target (existing fingerprint {existing}, "
                        f"requested {fingerprint}). One process shares one pool, "
                        "so the second config would have been silently ignored "
                        "and its data written to the first target's database. "
                        "Close the existing instance first, or use one config."
                    )
                cls._instance._refcount += 1
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

            instance._fingerprint = fingerprint
            instance._refcount = 1
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
        """Release one holder's claim; close the pool when the last one leaves.

        Reference-counted because the instance is shared. ``TRANSPORT=both`` has
        two shells holding the same pool, and an unconditional close meant the
        first shutdown left the second shell with a closed client and a
        class-level ``_instance`` of None — every subsequent query failing on a
        connection someone else had ended.

        Idempotent: closing more times than initialize() was called is a no-op
        rather than an error, so teardown paths that run twice stay harmless.
        """
        if self._refcount > 1:
            self._refcount -= 1
            return

        self._refcount = 0
        if self._client is not None:
            await self._client.close()
        self._client = None
        self._db = None
        self._fingerprint = None
        if type(self)._instance is self:
            type(self)._instance = None
