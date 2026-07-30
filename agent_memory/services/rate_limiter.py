"""Rate limiter — sliding-window per user in MongoDB.

Uses a ``rate_limits`` collection with TTL index on ``timestamp``.
"""

import logging
from datetime import UTC, datetime, timedelta

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window rate limiter backed by MongoDB."""

    def __init__(self, rate_limits_collection, config: MCPConfig) -> None:
        self.collection = rate_limits_collection
        self.config = config

    async def check_rate_limit(
        self, user_id: str, operation: str, max_requests: int | None = None,
    ) -> bool:
        """Return True if within rate limit, False if exceeded.

        Records the operation attempt and checks against the rate limit window.
        When ``max_requests`` is provided (e.g. from a governance profile),
        it overrides the global ``config.rate_limit_max_requests``.
        """
        if not self.config.rate_limit_enabled:
            return True

        effective_max = max_requests if max_requests is not None else self.config.rate_limit_max_requests

        now = datetime.now(UTC)
        window_start = now - timedelta(seconds=self.config.rate_limit_window_seconds)

        # Count operations in the window
        count = await self.collection.count_documents({
            "user_id": user_id,
            "operation": operation,
            "timestamp": {"$gte": window_start},
        })

        if count >= effective_max:
            logger.warning(
                "Rate limit exceeded for user %s on %s: %d/%d",
                user_id, operation, count, self.config.rate_limit_max_requests,
            )
            return False

        # Record this operation
        await self.collection.insert_one({
            "user_id": user_id,
            "operation": operation,
            "timestamp": now,
        })

        return True
