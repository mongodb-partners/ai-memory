"""Thin FastMCP tools over the AsyncMemory facade.

Each tool only translates transport: it calls the matching ``app.<method>`` and
maps facade exceptions to MCP's ``{"error": ...}`` convention. No business logic
lives here — that is the facade's job. ``AccessError`` (and its ``RateLimitError``
subclass) become an error dict; the message text distinguishes throttling from
denial.

``user_id`` remains a declared tool parameter — an MCP client legitimately names
the user in single-tenant use, and removing it would break every existing caller
— but it is no longer *believed*. Each tool routes it through :func:`_who`, which
resolves the identity from the request's verified token when auth is on. With auth
on, a client that names another user gets an error dict rather than that user's
memory.

Note for anything wrapping these tools: a refusal is a *return value*, not an
exception — the ``except`` clauses convert ``IdentityError`` into ``{"error": ...}``
so the MCP client sees a normal result. A wrapper that treats "returned without
raising" as "the call was authorised", and then re-reads ``user_id`` from the raw
arguments, reintroduces exactly the cross-tenant write this module removes. See
``auto_capture.resolve_capture_identity``.
"""

from __future__ import annotations

import logging

from agent_memory.auth.identity import IdentityError, resolve_caller
from agent_memory.exceptions import AccessError
from agent_memory.services.admin import PartialWipeError

logger = logging.getLogger(__name__)


def register_all_tools(mcp, app) -> None:
    """Register every memory tool on ``mcp``, bound to facade ``app``."""

    def _err(exc: Exception) -> dict:
        return {"error": str(exc)}

    def _who(requested_user_id: str):
        """Resolve the calling identity for one tool invocation.

        Reads the ambient access token via FastMCP's ``get_access_token()``,
        which is per-request — there is no token argument on a tool, so this is
        the only way a tool can learn who is calling. Returns ``None`` for the
        token when auth is off or no request context exists (stdio transport,
        direct calls in tests), which ``resolve_caller`` treats as the
        single-tenant case and honours ``requested_user_id``.

        Whether auth is *configured* is read from the config, not inferred from
        whether a token turned up. ``get_access_token`` already returns None for
        the legitimately-absent cases, so it raises only when something is wrong
        — an incompatible ``AccessToken`` type, for instance, which it converts to
        a ``TypeError``. Treating that as "auth is off" would fail open: the
        request's own ``user_id`` would be honoured on a deployment that
        configured tenant binding, which is the single failure this module exists
        to prevent. With auth on, a token we cannot read is a refusal.
        """
        config = getattr(app, "config", None)
        auth_on = bool(getattr(config, "auth_enabled", False))

        access = None
        try:
            from fastmcp.server.dependencies import get_access_token

            access = get_access_token()
        except Exception as exc:
            if auth_on:
                # Deliberately not `logger.debug`: on an auth-enabled deployment
                # this is either a misconfiguration or an attempt, and it now
                # produces a refusal rather than a silent downgrade.
                logger.warning(
                    "Could not read the MCP access token on an auth-enabled "
                    "deployment; refusing rather than trusting the request's "
                    "user_id: %s",
                    exc,
                )
                raise IdentityError(
                    "could not determine the authenticated identity for this "
                    "request; refusing rather than acting as the requested user_id"
                ) from exc
            logger.debug("No MCP access token in context; using caller-supplied user_id.")

        # Deliberately *not* refusing when `access` is None with auth on. Over HTTP
        # that state is unreachable: FastMCP's auth layer rejects an
        # unauthenticated request before any tool runs, so a None token here means
        # there is no request at all — stdio, or an in-process call. Refusing it
        # would break the stdio transport, which is a documented single-tenant
        # posture, without closing anything a remote caller can reach.
        return resolve_caller(access, requested_user_id, config)

    @mcp.tool(name="store_memory", description="Store conversation messages as memories.")
    async def store_memory(user_id: str, conversation_id: str, messages: list[dict]) -> dict:
        try:
            who = _who(user_id)
            return await app.add(who.user_id, conversation_id, messages, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="recall_memory", description="Curated, importance-ranked memory recall.")
    async def recall_memory(user_id: str, query: str, memory_type: str | None = None,
                            tags: list[str] | None = None, limit: int = 10,
                            tier: list[str] | None = None) -> dict:
        try:
            who = _who(user_id)
            return await app.recall(who.user_id, query, tier=tier, memory_type=memory_type,
                                    tags=tags, limit=limit, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="hybrid_search", description="Raw hybrid vector+FTS search ($rankFusion RRF).")
    async def hybrid_search(user_id: str, query: str, tier: list[str] | None = None,
                            limit: int = 10, memory_type: str | None = None,
                            tags: list[str] | None = None) -> dict:
        try:
            who = _who(user_id)
            return await app.search(who.user_id, query, tier=tier, limit=limit,
                                    memory_type=memory_type, tags=tags, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="delete_memory", description="Soft-delete memories by id/tags/time range.")
    async def delete_memory(user_id: str, memory_id: str | None = None,
                            tags: list[str] | None = None, time_range: dict | None = None,
                            confirm: bool = False, dry_run: bool = False) -> dict:
        try:
            who = _who(user_id)
            return await app.delete(who.user_id, memory_id=memory_id, tags=tags,
                                    time_range=time_range, confirm=confirm,
                                    dry_run=dry_run, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="check_cache", description="Semantic cache lookup.")
    async def check_cache(user_id: str, query: str, similarity_threshold: float | None = None) -> dict:
        try:
            who = _who(user_id)
            result = await app.check_cache(who.user_id, query, role=who.role,
                                           similarity_threshold=similarity_threshold)
            return result if result is not None else {"cache_hit": False}
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="store_cache", description="Store a query/response pair in the cache.")
    async def store_cache(user_id: str, query: str, response: str) -> dict:
        try:
            who = _who(user_id)
            cache_id = await app.store_cache(who.user_id, query, response, role=who.role)
            return {"cache_id": cache_id}
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="cache_invalidate", description="Invalidate cached entries.")
    async def cache_invalidate(user_id: str, pattern: str | None = None,
                               invalidate_all: bool = False) -> dict:
        try:
            who = _who(user_id)
            return await app.invalidate_cache(who.user_id, pattern=pattern,
                                              invalidate_all=invalidate_all, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="store_decision", description="Store or update a sticky decision.")
    async def store_decision(user_id: str, key: str, value: str, ttl_days: int | None = None) -> dict:
        try:
            who = _who(user_id)
            return await app.remember_decision(who.user_id, key, value,
                                               ttl_days=ttl_days, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="recall_decision", description="Recall a sticky decision by key.")
    async def recall_decision(user_id: str, key: str) -> dict:
        try:
            who = _who(user_id)
            result = await app.recall_decision(who.user_id, key, role=who.role)
            return result if result is not None else {"key": key, "value": None}
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="log_activity",
              description="Record one agent turn in the episodic activity log.")
    async def log_activity(user_id: str, thread_id: str, messages: list[dict],
                           todos: list[dict] | None = None, agent_name: str | None = None,
                           correlation_id: str | None = None,
                           conversation_id: str | None = None) -> dict:
        try:
            who = _who(user_id)
            return await app.log_activity(
                who.user_id, thread_id, messages, todos=todos, agent_name=agent_name,
                correlation_id=correlation_id, conversation_id=conversation_id,
                role=who.role,
            )
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="search_activity",
              description="Hybrid recall over logged agent turns — what did the agent do?")
    async def search_activity(user_id: str, query: str, thread_id: str | None = None,
                              agent_name: str | None = None, limit: int = 5) -> dict:
        try:
            who = _who(user_id)
            return await app.recall_activity(who.user_id, query, thread_id=thread_id,
                                             agent_name=agent_name, limit=limit,
                                             role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="get_thread", description="Replay a thread's logged turns in step order.")
    async def get_thread(user_id: str, thread_id: str, limit: int | None = None,
                         ascending: bool = True) -> dict:
        try:
            who = _who(user_id)
            return await app.get_thread(who.user_id, thread_id, limit=limit,
                                        ascending=ascending, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="get_correlation",
              description="Every logged turn sharing a trace/correlation id.")
    async def get_correlation(user_id: str, correlation_id: str,
                              limit: int | None = None) -> dict:
        try:
            who = _who(user_id)
            return await app.get_activity_by_correlation(who.user_id, correlation_id,
                                                         limit=limit, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="set_activity_retention",
              description="Change episodic retention in place (null = keep forever).")
    async def set_activity_retention(user_id: str, ttl_seconds: int | None = None) -> dict:
        try:
            who = _who(user_id)
            return await app.set_activity_retention(who.user_id, ttl_seconds=ttl_seconds,
                                                    role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="memory_health", description="Health statistics for a user's memory store.")
    async def memory_health(user_id: str) -> dict:
        try:
            who = _who(user_id)
            return await app.health(who.user_id, role=who.role)
        except (AccessError, IdentityError) as e:
            return _err(e)

    @mcp.tool(name="wipe_user_data", description="Permanently delete ALL data for a user.")
    async def wipe_user_data(user_id: str, confirm: bool = False) -> dict:
        try:
            who = _who(user_id)
            return await app.wipe_user_data(who.user_id, confirm=confirm, role=who.role)
        except PartialWipeError as e:
            # Caught explicitly so the per-collection counts survive the
            # translation. The generic `_err` would reduce a half-finished
            # irreversible deletion to a message, and what was and was not deleted
            # is the only thing that makes the retry safe. `complete: False` is
            # the field a client should branch on.
            return {
                "error": str(e),
                "complete": False,
                **{k: v for k, v in e.counts.items() if k != "user_id"},
                "failed_collections": sorted(e.errors),
            }
        except (AccessError, IdentityError) as e:
            return _err(e)
