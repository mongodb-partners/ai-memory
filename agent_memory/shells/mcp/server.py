"""FastMCP server shell.

Thin transport: builds one ``AsyncMemory`` in ``lifespan``, registers the slim
tools bound to it, wires auto-capture, and closes the facade on shutdown. All
business logic and orchestration live in the facade.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from agent_memory.config import MemoryConfig
from agent_memory.memory import AsyncMemory
from agent_memory.shells.mcp.auto_capture import AutoCaptureMiddleware, wrap_tools
from agent_memory.shells.mcp.tools import register_all_tools

logger = logging.getLogger(__name__)


def build_auth(config: MemoryConfig):
    """Return a token verifier when auth is enabled, else None.

    The "enabled but no secret" case used to log a warning and return None,
    serving every tool unauthenticated. ``MemoryConfig`` now refuses to construct
    in that state, so this only has to handle auth being genuinely off.
    """
    if not config.auth_enabled or not config.auth_secret:
        return None
    from agent_memory.auth.api_keys import APIKeyManager
    from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier

    return MemoryMCPTokenVerifier(
        secret=config.auth_secret, api_key_manager=APIKeyManager()
    )


def make_lifespan(
    config: MemoryConfig,
    app: AsyncMemory | None = None,
    holder: dict | None = None,
):
    """Build the FastMCP lifespan context manager.

    If ``app`` is provided (dual-transport: one facade shared across shells),
    it is reused and its lifecycle is owned by the caller. Otherwise the
    lifespan creates and closes its own ``AsyncMemory``.

    ``holder`` is a dict the lifespan publishes the live facade into. Routes
    registered outside the lifespan — ``/health`` is the only one — have no other
    way to reach a facade the lifespan created, and a route that closes over the
    possibly-``None`` ``app`` argument would report on nothing forever in the
    self-owned case. Mirrors ``create_managed_app``'s ``_Proxy`` in the REST shell.
    """

    @asynccontextmanager
    async def lifespan(_mcp: FastMCP):
        owns_app = app is None
        instance = app or await AsyncMemory.create(config)
        if holder is not None:
            holder["app"] = instance
        register_all_tools(_mcp, instance)
        # Kept in a local rather than discarded: shutdown has to be able to wait
        # for in-flight captures. Without the reference there was nothing to call
        # `drain()` on, so the facade closed underneath any capture still awaiting
        # an embedding call — the write lost, or racing a closing client.
        capture = None
        if config.auto_capture_enabled:
            capture = AutoCaptureMiddleware(instance, config)
            wrap_tools(_mcp, capture)
        logger.info("agent-memory MCP shell started.")
        try:
            yield
        finally:
            # Drain before close, and bounded: a capture that cannot finish must
            # delay shutdown by seconds, not hold the process open.
            if capture is not None and not await capture.drain():
                logger.warning(
                    "Auto-capture did not drain before shutdown; "
                    "in-flight captures were dropped."
                )
            if owns_app:
                await instance.close()
            if holder is not None:
                holder["app"] = None
            logger.info("agent-memory MCP shell stopped.")

    return lifespan


def create_mcp(config: MemoryConfig | None = None, app: AsyncMemory | None = None) -> FastMCP:
    """Build a FastMCP server bound to a (possibly shared) facade."""
    config = config or MemoryConfig.from_env()
    # The lifespan publishes the live facade here so /health — registered now,
    # before any facade exists — can reach it later.
    facade_holder: dict = {"app": app}
    mcp = FastMCP(
        "agent-memory",
        lifespan=make_lifespan(config, app, facade_holder),
        auth=build_auth(config),
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request):
        """Liveness, with the same body the REST shell serves.

        This used to return a bare `{"status": "ok"}` — a probe that reports
        nothing but "the process accepts sockets". In a dual-transport deployment
        the two shells share one facade, so an operator pointing a monitor at the
        MCP port and one at the REST port got two different answers about the same
        process: one that degrades on a dead enrichment worker and one that never
        does. Whichever port the monitor happened to target decided whether the
        outage was visible.

        `build_health_body` is shared with the REST shell, so there is one
        definition of what health means and neither can drift.
        """
        from starlette.responses import JSONResponse

        from agent_memory.shells.rest.app import build_health_body

        facade = facade_holder.get("app")
        if facade is None:
            # Before lifespan startup, or after shutdown. The process is up; there
            # is nothing to ask about the workers yet.
            return JSONResponse({"status": "starting"})
        return JSONResponse(build_health_body(facade))

    return mcp
