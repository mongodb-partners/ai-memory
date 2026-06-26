"""Governance service — profile-based access control policies.

Stores governance profiles in MongoDB ``governance_profiles`` collection
with in-memory caching.
"""

import logging
import time
from datetime import datetime, timezone

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)

_DEFAULT_PROFILES = {
    "admin": {
        "role": "admin",
        "max_memories_per_day": 10000,
        "max_searches_per_day": 10000,
        "allowed_operations": ["*"],
    },
    "power_user": {
        "role": "power_user",
        "max_memories_per_day": 1000,
        "max_searches_per_day": 5000,
        "allowed_operations": [
            "store_memory", "recall_memory", "delete_memory",
            "hybrid_search", "check_cache", "store_cache", "search_web",
            "memory_health",
        ],
    },
    "end_user": {
        "role": "end_user",
        "max_memories_per_day": 100,
        "max_searches_per_day": 500,
        "allowed_operations": [
            "store_memory", "recall_memory", "hybrid_search",
            "check_cache", "store_cache",
        ],
    },
}


class GovernanceService:
    """Profile-based governance with MongoDB persistence and in-memory cache."""

    def __init__(self, governance_collection, config: MCPConfig) -> None:
        self.collection = governance_collection
        self.config = config
        self._cache: dict[str, dict] = {}
        self._cache_time: dict[str, float] = {}

    async def get_profile(self, role: str) -> dict:
        """Get governance profile by role. Uses cache with TTL."""
        now = time.time()
        cached_at = self._cache_time.get(role, 0)

        if role in self._cache and (now - cached_at) < self.config.governance_cache_ttl_seconds:
            return self._cache[role]

        doc = await self.collection.find_one({"role": role})
        if doc:
            doc.pop("_id", None)
            self._cache[role] = doc
            self._cache_time[role] = now
            return doc

        # Fallback to default
        default = _DEFAULT_PROFILES.get(self.config.governance_default_profile, _DEFAULT_PROFILES["end_user"])
        return default

    async def check_allowed(self, user_id: str, role: str, operation: str) -> bool:
        """Check if the given role is allowed to perform the operation."""
        profile = await self.get_profile(role)
        allowed = profile.get("allowed_operations", [])
        return "*" in allowed or operation in allowed

    async def seed_defaults(self) -> int:
        """Seed default governance profiles into MongoDB. Returns count inserted."""
        count = 0
        for role, profile in _DEFAULT_PROFILES.items():
            existing = await self.collection.find_one({"role": role})
            if not existing:
                doc = {**profile, "created_at": datetime.now(timezone.utc)}
                await self.collection.insert_one(doc)
                count += 1
        return count
