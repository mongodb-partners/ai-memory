"""MCP Cache Tools — check and store semantic cache."""

import time

from agent_memory.core.registry import ServiceRegistry


def register_cache_tools(mcp):
    """Register cache MCP tools on the FastMCP server."""

    @mcp.tool(
        name="check_cache",
        description=(
            "Check semantic cache for a similar previous query. "
            "Returns cached response if similarity >= threshold."
        ),
    )
    async def check_cache(
        user_id: str,
        query: str,
        similarity_threshold: float | None = None,
    ) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "check_cache")
        if access_err:
            return {"error": access_err}
        start = time.time()
        try:
            result = await svc.cache_service.check(
                user_id, query, similarity_threshold=similarity_threshold,
            )
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "cache:read", "check_cache", "success", duration_ms,
                cache_hit=result is not None,
            )
            return result or {"cache_hit": False}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "cache:read", "check_cache", "error", duration_ms,
                error=str(e),
            )
            raise

    @mcp.tool(
        name="store_cache",
        description="Cache a query-response pair for future similarity lookups.",
    )
    async def store_cache(user_id: str, query: str, response: str) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "store_cache")
        if access_err:
            return {"error": access_err}
        start = time.time()
        try:
            cache_id = await svc.cache_service.store(user_id, query, response)
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "cache:write", "store_cache", "success", duration_ms,
            )
            return {"cache_id": cache_id}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "cache:write", "store_cache", "error", duration_ms,
                error=str(e),
            )
            raise
