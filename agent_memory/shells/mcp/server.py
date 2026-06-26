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
    """Return a token verifier when auth is enabled, else None."""
    if not config.auth_enabled or not config.auth_secret:
        if config.auth_enabled and not config.auth_secret:
            logger.warning("AUTH_ENABLED=true but AUTH_SECRET is empty — auth disabled.")
        return None
    from agent_memory.auth.api_keys import APIKeyManager
    from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier

    return MemoryMCPTokenVerifier(
        secret=config.auth_secret, api_key_manager=APIKeyManager()
    )


def make_lifespan(config: MemoryConfig, app: AsyncMemory | None = None):
    """Build the FastMCP lifespan context manager.

    If ``app`` is provided (dual-transport: one facade shared across shells),
    it is reused and its lifecycle is owned by the caller. Otherwise the
    lifespan creates and closes its own ``AsyncMemory``.
    """

    @asynccontextmanager
    async def lifespan(_mcp: FastMCP):
        owns_app = app is None
        instance = app or await AsyncMemory.create(config)
        register_all_tools(_mcp, instance)
        if config.auto_capture_enabled:
            wrap_tools(_mcp, AutoCaptureMiddleware(instance, config))
        logger.info("agent-memory MCP shell started.")
        try:
            yield
        finally:
            if owns_app:
                await instance.close()
            logger.info("agent-memory MCP shell stopped.")

    return lifespan


def create_mcp(config: MemoryConfig | None = None, app: AsyncMemory | None = None) -> FastMCP:
    """Build a FastMCP server bound to a (possibly shared) facade."""
    config = config or MemoryConfig.from_env()
    mcp = FastMCP("agent-memory", lifespan=make_lifespan(config, app), auth=build_auth(config))

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request):  # noqa: ANN001
        from starlette.responses import JSONResponse

        return JSONResponse({"status": "ok"})

    return mcp
