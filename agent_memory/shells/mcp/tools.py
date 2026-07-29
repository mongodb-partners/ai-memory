"""Thin FastMCP tools over the AsyncMemory facade.

Each tool only translates transport: it calls the matching ``app.<method>`` and
maps facade exceptions to MCP's ``{"error": ...}`` convention. No business logic
lives here — that is the facade's job. ``AccessError`` (and its ``RateLimitError``
subclass) become an error dict; the message text distinguishes throttling from
denial.
"""

from __future__ import annotations

from agent_memory.exceptions import AccessError


def register_all_tools(mcp, app) -> None:
    """Register every memory tool on ``mcp``, bound to facade ``app``."""

    def _err(exc: Exception) -> dict:
        return {"error": str(exc)}

    @mcp.tool(name="store_memory", description="Store conversation messages as memories.")
    async def store_memory(user_id: str, conversation_id: str, messages: list[dict]) -> dict:
        try:
            return await app.add(user_id, conversation_id, messages)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="recall_memory", description="Curated, importance-ranked memory recall.")
    async def recall_memory(user_id: str, query: str, memory_type: str | None = None,
                            tags: list[str] | None = None, limit: int = 10,
                            tier: list[str] | None = None) -> dict:
        try:
            return await app.recall(user_id, query, tier=tier, memory_type=memory_type,
                                    tags=tags, limit=limit)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="hybrid_search", description="Raw hybrid vector+FTS search ($rankFusion RRF).")
    async def hybrid_search(user_id: str, query: str, tier: list[str] | None = None,
                            limit: int = 10, memory_type: str | None = None,
                            tags: list[str] | None = None) -> dict:
        try:
            return await app.search(user_id, query, tier=tier, limit=limit,
                                    memory_type=memory_type, tags=tags)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="delete_memory", description="Soft-delete memories by id/tags/time range.")
    async def delete_memory(user_id: str, memory_id: str | None = None,
                            tags: list[str] | None = None, time_range: dict | None = None,
                            confirm: bool = False, dry_run: bool = False) -> dict:
        try:
            return await app.delete(user_id, memory_id=memory_id, tags=tags,
                                    time_range=time_range, confirm=confirm, dry_run=dry_run)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="check_cache", description="Semantic cache lookup.")
    async def check_cache(user_id: str, query: str, similarity_threshold: float | None = None) -> dict:
        try:
            result = await app.check_cache(user_id, query, similarity_threshold=similarity_threshold)
            return result if result is not None else {"cache_hit": False}
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="store_cache", description="Store a query/response pair in the cache.")
    async def store_cache(user_id: str, query: str, response: str) -> dict:
        try:
            cache_id = await app.store_cache(user_id, query, response)
            return {"cache_id": cache_id}
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="cache_invalidate", description="Invalidate cached entries.")
    async def cache_invalidate(user_id: str, pattern: str | None = None,
                               invalidate_all: bool = False) -> dict:
        try:
            return await app.invalidate_cache(user_id, pattern=pattern, invalidate_all=invalidate_all)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="store_decision", description="Store or update a sticky decision.")
    async def store_decision(user_id: str, key: str, value: str, ttl_days: int | None = None) -> dict:
        try:
            return await app.remember_decision(user_id, key, value, ttl_days=ttl_days)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="recall_decision", description="Recall a sticky decision by key.")
    async def recall_decision(user_id: str, key: str) -> dict:
        try:
            result = await app.recall_decision(user_id, key)
            return result if result is not None else {"key": key, "value": None}
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="log_activity",
              description="Record one agent turn in the episodic activity log.")
    async def log_activity(user_id: str, thread_id: str, messages: list[dict],
                           todos: list[dict] | None = None, agent_name: str | None = None,
                           correlation_id: str | None = None,
                           conversation_id: str | None = None) -> dict:
        try:
            return await app.log_activity(
                user_id, thread_id, messages, todos=todos, agent_name=agent_name,
                correlation_id=correlation_id, conversation_id=conversation_id,
            )
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="search_activity",
              description="Hybrid recall over logged agent turns — what did the agent do?")
    async def search_activity(user_id: str, query: str, thread_id: str | None = None,
                              agent_name: str | None = None, limit: int = 5) -> dict:
        try:
            return await app.recall_activity(user_id, query, thread_id=thread_id,
                                             agent_name=agent_name, limit=limit)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="get_thread", description="Replay a thread's logged turns in step order.")
    async def get_thread(user_id: str, thread_id: str, limit: int | None = None,
                         ascending: bool = True) -> dict:
        try:
            return await app.get_thread(user_id, thread_id, limit=limit, ascending=ascending)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="get_correlation",
              description="Every logged turn sharing a trace/correlation id.")
    async def get_correlation(user_id: str, correlation_id: str,
                              limit: int | None = None) -> dict:
        try:
            return await app.get_activity_by_correlation(user_id, correlation_id, limit=limit)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="set_activity_retention",
              description="Change episodic retention in place (null = keep forever).")
    async def set_activity_retention(user_id: str, ttl_seconds: int | None = None) -> dict:
        try:
            return await app.set_activity_retention(user_id, ttl_seconds=ttl_seconds)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="memory_health", description="Health statistics for a user's memory store.")
    async def memory_health(user_id: str) -> dict:
        try:
            return await app.health(user_id)
        except AccessError as e:
            return _err(e)

    @mcp.tool(name="wipe_user_data", description="Permanently delete ALL data for a user.")
    async def wipe_user_data(user_id: str, confirm: bool = False) -> dict:
        try:
            return await app.wipe_user_data(user_id, confirm=confirm)
        except AccessError as e:
            return _err(e)
