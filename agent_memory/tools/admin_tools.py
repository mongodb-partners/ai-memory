"""MCP Admin Tools — memory_health, wipe_user_data, cache_invalidate."""

import time

from agent_memory.core.registry import ServiceRegistry


def register_admin_tools(mcp):
    """Register admin MCP tools on the FastMCP server."""

    @mcp.tool(
        name="memory_health",
        description=(
            "Get health statistics for a user's memory store. "
            "Returns tier counts, pending enrichments, and total memories."
        ),
    )
    async def memory_health(user_id: str) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "memory_health")
        if access_err:
            return {"error": access_err}
        start = time.time()

        try:
            from agent_memory.core.database import DatabaseManager

            db = (await DatabaseManager.get_instance()).db
            memories_col = db["memories"]

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

            tier_stats = {}
            enrichment_stats = {}
            total = 0

            for r in results:
                tier = r["_id"]["tier"]
                status = r["_id"]["enrichment_status"]
                count = r["count"]
                total += count
                tier_stats[tier] = tier_stats.get(tier, 0) + count
                enrichment_stats[status] = enrichment_stats.get(status, 0) + count

            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "admin", "memory_health", "success", duration_ms,
            )
            return {
                "user_id": user_id,
                "total_memories": total,
                "tier_stats": tier_stats,
                "enrichment_stats": enrichment_stats,
            }
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "admin", "memory_health", "error", duration_ms,
                error=str(e),
            )
            raise

    @mcp.tool(
        name="wipe_user_data",
        description=(
            "Permanently delete ALL data for a user (memories, cache, audit log). "
            "Requires confirm=true. This action is irreversible."
        ),
    )
    async def wipe_user_data(user_id: str, confirm: bool = False) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "wipe_user_data")
        if access_err:
            return {"error": access_err}
        start = time.time()

        if not confirm:
            return {
                "error": "wipe_user_data requires confirm=true. "
                "This will permanently delete ALL user data."
            }

        try:
            from agent_memory.core.database import DatabaseManager

            db = (await DatabaseManager.get_instance()).db

            memories_result = await db["memories"].delete_many({"user_id": user_id})
            cache_result = await db["semantic_cache"].delete_many({"user_id": user_id})
            audit_result = await db["audit_log"].delete_many({"user_id": user_id})

            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "admin", "wipe_user_data", "success", duration_ms,
                memories_deleted=memories_result.deleted_count,
                cache_deleted=cache_result.deleted_count,
                audit_deleted=audit_result.deleted_count,
            )
            return {
                "user_id": user_id,
                "memories_deleted": memories_result.deleted_count,
                "cache_deleted": cache_result.deleted_count,
                "audit_deleted": audit_result.deleted_count,
            }
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "admin", "wipe_user_data", "error", duration_ms,
                error=str(e),
            )
            raise

    @mcp.tool(
        name="cache_invalidate",
        description=(
            "Invalidate cached entries for a user. "
            "Use invalidate_all=true to clear all, or pattern to match queries."
        ),
    )
    async def cache_invalidate(
        user_id: str,
        pattern: str | None = None,
        invalidate_all: bool = False,
    ) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "cache_invalidate")
        if access_err:
            return {"error": access_err}
        start = time.time()

        try:
            deleted = await svc.cache_service.invalidate(
                user_id, pattern=pattern, invalidate_all=invalidate_all,
            )

            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "admin", "cache_invalidate", "success", duration_ms,
                deleted_count=deleted,
            )
            return {"user_id": user_id, "deleted_count": deleted}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "admin", "cache_invalidate", "error", duration_ms,
                error=str(e),
            )
            raise
