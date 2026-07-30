"""Rate limiter — fixed-window per user/operation, counted atomically in MongoDB.

Uses a ``rate_limits`` collection with a TTL index on ``timestamp``.

The previous implementation counted the documents in a sliding window and *then*
inserted its own:

    count = await collection.count_documents({...})   # every caller sees N
    if count >= limit: return False
    await collection.insert_one({...})                # every caller then inserts

Between the count and the insert there is no lock, so under concurrency every
request in a burst reads the same below-limit count and every one of them is
admitted. The limit held against a single sequential caller and was absent against
exactly the traffic a rate limiter exists to bound.

The fix makes the increment and the decision one operation:
``find_one_and_update`` with ``$inc`` on a per-window counter document, reading the
post-increment value. MongoDB serialises updates to a single document, so N
concurrent callers get N distinct values and only those within the limit proceed.

**Fixed window, not sliding.** This tradeoff is taken deliberately. A sliding
window needs either a count over per-request documents — the racy shape above — or
a read-modify-write over a sorted structure, which has the same problem one level
up. A fixed window can admit up to ``2 * max`` across a boundary (``max`` late in
one window, ``max`` early in the next). That is the standard fixed-window caveat
and a far smaller error than the unbounded over-admission it replaces.
"""

import logging
from datetime import UTC, datetime

from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)


class RateLimiter:
    """Fixed-window rate limiter backed by a single atomically-incremented doc."""

    def __init__(self, rate_limits_collection, config: MCPConfig) -> None:
        self.collection = rate_limits_collection
        self.config = config

    def _window_start(self, now: datetime) -> datetime:
        """Floor ``now`` to the current window boundary.

        Derived from the epoch rather than from first-request time, so every
        process in a deployment agrees where the window starts without
        coordinating. The bucket ``_id`` is then the same document for all of
        them, which is what makes the ``$inc`` one shared count rather than one
        count per process.
        """
        window = max(1, self.config.rate_limit_window_seconds)
        epoch_seconds = int(now.timestamp())
        return datetime.fromtimestamp(
            epoch_seconds - (epoch_seconds % window), tz=UTC
        )

    async def check_rate_limit(
        self, user_id: str, operation: str, max_requests: int | None = None,
    ) -> bool:
        """Return True if within the limit, False if exceeded.

        Increments the window counter and decides from the post-increment value in
        one atomic document update. When ``max_requests`` is provided (e.g. from a
        governance profile) it overrides ``config.rate_limit_max_requests``.
        """
        if not self.config.rate_limit_enabled:
            return True

        effective_max = (
            max_requests if max_requests is not None
            else self.config.rate_limit_max_requests
        )
        # Zero means "no requests", not "unlimited". Short-circuited before the
        # round trip, since $inc would exceed it on the first call anyway.
        if effective_max is not None and effective_max <= 0:
            return False

        now = datetime.now(UTC)
        window_start = self._window_start(now)

        # The composite `_id` *is* the window bucket: deterministic, so an upsert
        # from any process targets the same document — no extra index, and no
        # chance of two buckets for one window.
        bucket_id = {
            "user_id": user_id,
            "operation": operation,
            "window_start": window_start,
        }

        try:
            doc = await self.collection.find_one_and_update(
                {"_id": bucket_id},
                {
                    "$inc": {"count": 1},
                    # `user_id` and `operation` are duplicated out of the `_id` so
                    # `wipe_user_data` finds these records by a plain
                    # `{"user_id": ...}` query. `timestamp` feeds the existing TTL
                    # index, which is what expires the bucket.
                    "$setOnInsert": {
                        "user_id": user_id,
                        "operation": operation,
                        "window_start": window_start,
                        "timestamp": now,
                    },
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError:
            # Fail open on an unavailable limiter: refusing every request turns a
            # counter outage into a total outage. Governance has already run and is
            # the control that must not be bypassed; this one is a throughput
            # guard, and the safe failure for a throughput guard is to allow.
            logger.warning(
                "Rate limit check failed for user %s on %s; allowing.",
                user_id, operation, exc_info=True,
            )
            return True

        count = (doc or {}).get("count", 1)
        if count > effective_max:
            logger.warning(
                "Rate limit exceeded for user %s on %s: %d/%d",
                user_id, operation, count, effective_max,
            )
            return False

        return True
