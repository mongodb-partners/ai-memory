"""Admin service — health statistics and full user-data wipe.

Extracted from the former ``memory_health`` / ``wipe_user_data`` MCP tools so
the facade orchestration layer (access-check → service → audit) stays thin and
identical across shells.
"""

from __future__ import annotations


class AdminService:
    """Cross-collection administrative operations for a single user."""

    def __init__(self, db) -> None:
        self.db = db

    async def health(self, user_id: str) -> dict:
        """Return tier counts, enrichment-status counts, and total memories."""
        memories_col = self.db["memories"]
        pipeline = [
            {"$match": {"user_id": user_id, "deleted_at": None}},
            {
                "$group": {
                    "_id": {
                        "tier": "$tier",
                        "enrichment_status": "$enrichment_status",
                    },
                    "count": {"$sum": 1},
                }
            },
        ]
        cursor = await memories_col.aggregate(pipeline)
        results = await cursor.to_list(None)

        tier_stats: dict = {}
        enrichment_stats: dict = {}
        total = 0
        for r in results:
            tier = r["_id"]["tier"]
            status = r["_id"]["enrichment_status"]
            count = r["count"]
            total += count
            tier_stats[tier] = tier_stats.get(tier, 0) + count
            enrichment_stats[status] = enrichment_stats.get(status, 0) + count

        return {
            "user_id": user_id,
            "total_memories": total,
            "tier_stats": tier_stats,
            "enrichment_stats": enrichment_stats,
        }

    async def wipe_user_data(self, user_id: str) -> dict:
        """Permanently delete all memories, cache, and audit entries for a user.

        The caller (facade) is responsible for the ``confirm`` gate; this method
        performs the irreversible deletion unconditionally.
        """
        memories_result = await self.db["memories"].delete_many({"user_id": user_id})
        cache_result = await self.db["semantic_cache"].delete_many({"user_id": user_id})
        audit_result = await self.db["audit_log"].delete_many({"user_id": user_id})
        return {
            "user_id": user_id,
            "memories_deleted": memories_result.deleted_count,
            "cache_deleted": cache_result.deleted_count,
            "audit_deleted": audit_result.deleted_count,
        }
