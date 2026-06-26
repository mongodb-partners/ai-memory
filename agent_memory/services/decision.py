"""Decision service — store and recall keyed decisions with TTL.

Stores decisions in MongoDB ``decisions`` collection with upsert by
``(user_id, key)`` and TTL index on ``expires_at``.
"""

import logging
from datetime import datetime, timedelta, timezone

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)

_SYSTEM_DEFAULTS = {
    "system:governance_profile": "end_user",
    "system:prompt_experiment": "true",
}


class DecisionService:
    """Store and recall sticky decisions with configurable TTL."""

    def __init__(self, decisions_collection, config: MCPConfig) -> None:
        self.collection = decisions_collection
        self.config = config

    async def store(
        self,
        user_id: str,
        key: str,
        value: str,
        ttl_days: int | None = None,
    ) -> str:
        """Store or update a decision. Returns 'stored' or 'updated'."""
        ttl = ttl_days if ttl_days is not None else self.config.decision_default_ttl_days
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=ttl)

        result = await self.collection.update_one(
            {"user_id": user_id, "key": key},
            {
                "$set": {
                    "value": value,
                    "updated_at": now,
                    "expires_at": expires_at,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "key": key,
                    "created_at": now,
                },
            },
            upsert=True,
        )

        if result.upserted_id:
            return "stored"
        return "updated"

    async def recall(self, user_id: str, key: str) -> dict | None:
        """Recall a decision by key. Returns None if not found or expired."""
        now = datetime.now(timezone.utc)
        doc = await self.collection.find_one(
            {
                "user_id": user_id,
                "key": key,
                "expires_at": {"$gt": now},
            }
        )
        if not doc:
            return None

        return {
            "key": doc["key"],
            "value": doc["value"],
            "created_at": doc.get("created_at", "").isoformat() if isinstance(doc.get("created_at"), datetime) else str(doc.get("created_at", "")),
            "updated_at": doc.get("updated_at", "").isoformat() if isinstance(doc.get("updated_at"), datetime) else str(doc.get("updated_at", "")),
            "expires_at": doc.get("expires_at", "").isoformat() if isinstance(doc.get("expires_at"), datetime) else str(doc.get("expires_at", "")),
        }

    async def seed_defaults(self, system_user_id: str = "system") -> int:
        """Seed system-default decisions. Returns count inserted.

        Idempotent: skips decisions that already exist for the system user.
        """
        count = 0
        for key, value in _SYSTEM_DEFAULTS.items():
            existing = await self.recall(system_user_id, key)
            if existing is None:
                await self.store(system_user_id, key, value)
                count += 1
        return count
