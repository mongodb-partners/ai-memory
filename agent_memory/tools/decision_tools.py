"""MCP Decision Tools — store_decision, recall_decision."""

import time

from agent_memory.core.registry import ServiceRegistry


def register_decision_tools(mcp):
    """Register decision MCP tools on the FastMCP server."""

    @mcp.tool(
        name="store_decision",
        description=(
            "Store a keyed decision for a user. Decisions persist across conversations "
            "with configurable TTL. Use for preferences, choices, and sticky settings."
        ),
    )
    async def store_decision(
        user_id: str,
        key: str,
        value: str,
        ttl_days: int | None = None,
    ) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "store_decision")
        if access_err:
            return {"error": access_err}
        start = time.time()

        if svc.decision_service is None:
            return {"error": "Decision service is not enabled"}

        try:
            action = await svc.decision_service.store(
                user_id, key, value, ttl_days=ttl_days,
            )
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "decision:write", "store_decision", "success", duration_ms,
                key=key, action=action,
            )
            return {"key": key, "action": action, "user_id": user_id}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "decision:write", "store_decision", "error", duration_ms,
                error=str(e),
            )
            raise

    @mcp.tool(
        name="recall_decision",
        description=(
            "Recall a previously stored decision by key for a user. "
            "Returns the decision value and metadata, or not_found."
        ),
    )
    async def recall_decision(user_id: str, key: str) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "recall_decision")
        if access_err:
            return {"error": access_err}
        start = time.time()

        if svc.decision_service is None:
            return {"error": "Decision service is not enabled"}

        try:
            result = await svc.decision_service.recall(user_id, key)
            duration_ms = int((time.time() - start) * 1000)

            if result is None:
                await svc.audit_service.log(
                    user_id, "decision:read", "recall_decision", "success", duration_ms,
                    key=key, found=False,
                )
                return {"key": key, "found": False}

            await svc.audit_service.log(
                user_id, "decision:read", "recall_decision", "success", duration_ms,
                key=key, found=True,
            )
            return {"key": key, "found": True, "decision": result}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "decision:read", "recall_decision", "error", duration_ms,
                error=str(e),
            )
            raise
