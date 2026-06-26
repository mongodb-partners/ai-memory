"""MCP Memory Tools — store, recall, delete."""

import time

from agent_memory.core.registry import ServiceRegistry


def register_memory_tools(mcp):
    """Register memory MCP tools on the FastMCP server."""

    @mcp.tool(
        name="store_memory",
        description=(
            "Store conversation messages as short-term memories. "
            "For human messages >30 chars, also creates long-term memory candidates."
        ),
    )
    async def store_memory(
        user_id: str,
        conversation_id: str,
        messages: list[dict],
    ) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "store_memory")
        if access_err:
            return {"error": access_err}
        start = time.time()
        try:
            # Normalize messages: accept "role" or "message_type"
            for msg in messages:
                if "message_type" not in msg:
                    msg["message_type"] = msg.get("role", "human")
            stm_ids = await svc.memory_service.store_stm(user_id, conversation_id, messages)
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "memory:write", "store_memory", "success", duration_ms,
                conversation_id=conversation_id, count=len(messages),
            )
            return {"stm_ids": stm_ids, "count": len(stm_ids)}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "memory:write", "store_memory", "error", duration_ms,
                error=str(e),
            )
            raise

    @mcp.tool(
        name="recall_memory",
        description=(
            "Semantically search stored memories. Returns results ranked by "
            "recency, importance, and relevance."
        ),
    )
    async def recall_memory(
        user_id: str,
        query: str,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        tier: list[str] | None = None,
    ) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "recall_memory")
        if access_err:
            return {"error": access_err}
        start = time.time()
        try:
            results = await svc.memory_service.recall(
                user_id, query, tier=tier, memory_type=memory_type,
                tags=tags, limit=limit,
            )
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "memory:read", "recall_memory", "success", duration_ms,
                query=query, result_count=len(results),
            )
            return {"results": results, "count": len(results)}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "memory:read", "recall_memory", "error", duration_ms,
                error=str(e),
            )
            raise

    @mcp.tool(
        name="delete_memory",
        description=(
            "Soft-delete memories by ID, tags, or time range. "
            "Bulk deletes require confirm=true. Use dry_run to preview."
        ),
    )
    async def delete_memory(
        user_id: str,
        memory_id: str | None = None,
        tags: list[str] | None = None,
        time_range: dict | None = None,
        confirm: bool = False,
        dry_run: bool = False,
    ) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "delete_memory")
        if access_err:
            return {"error": access_err}
        start = time.time()
        try:
            result = await svc.memory_service.delete(
                user_id, memory_id=memory_id, tags=tags,
                time_range=time_range, confirm=confirm, dry_run=dry_run,
            )
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "memory:delete", "delete_memory", "success", duration_ms,
                deleted_count=result["deleted_count"], dry_run=dry_run,
            )
            return result
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "memory:delete", "delete_memory", "error", duration_ms,
                error=str(e),
            )
            raise
