"""Transport runner: serve MCP, REST, or both off one AsyncMemory instance.

``TRANSPORT=mcp|rest|both`` selects which shells run. ``both`` shares a single
facade (one Atlas connection pool, one set of workers, two protocols) — the
"memory platform in one deployable unit" story.
"""

from __future__ import annotations

import logging

from agent_memory.config import MemoryConfig
from agent_memory.memory import AsyncMemory
from agent_memory.shells.mcp.server import create_mcp
from agent_memory.shells.rest.app import create_app

logger = logging.getLogger(__name__)


async def build_shells(config: MemoryConfig) -> dict:
    """Create the requested shells, all bound to one shared facade.

    Returns a dict keyed by shell name (``"mcp"`` / ``"rest"``). The shared
    ``AsyncMemory`` is created here so a single instance backs every shell.
    """
    transport = (config.transport or "mcp").lower()
    if transport in ("streamable-http", "stdio"):
        transport = "mcp"  # legacy memory-mcp transport values → MCP shell

    app = await AsyncMemory.create(config)
    shells: dict = {}
    if transport in ("mcp", "both"):
        shells["mcp"] = create_mcp(config, app=app)
    if transport in ("rest", "both"):
        shells["rest"] = create_app(app)
    if not shells:
        raise ValueError(f"Unknown TRANSPORT: {config.transport!r}")
    return shells


def run(config: MemoryConfig | None = None) -> None:
    """Blocking entrypoint used by ``python -m agent_memory``.

    For a single shell, runs it directly. For ``both``, mounts the REST app and
    the MCP ASGI app under one uvicorn server so both share the process and the
    one facade created in the MCP lifespan.
    """
    import uvicorn

    config = config or MemoryConfig.from_env()
    transport = (config.transport or "mcp").lower()
    if transport in ("streamable-http", "stdio", "mcp"):
        create_mcp(config).run(
            transport="streamable-http", host="0.0.0.0", port=config.port
        )
        return
    if transport == "rest":
        # REST shell owns the facade lifecycle via FastAPI lifespan.
        from agent_memory.shells.rest.app import create_managed_app

        uvicorn.run(create_managed_app(config), host="0.0.0.0", port=config.port)
        return
    if transport == "both":
        from agent_memory.shells.rest.app import create_managed_app

        mcp = create_mcp(config)
        api = create_managed_app(config)
        api.mount("/mcp", mcp.http_app())
        uvicorn.run(api, host="0.0.0.0", port=config.port)
        return
    raise ValueError(f"Unknown TRANSPORT: {config.transport!r}")
