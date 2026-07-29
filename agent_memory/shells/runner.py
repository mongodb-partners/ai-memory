"""Transport runner: serve MCP, REST, or both off one AsyncMemory instance.

``TRANSPORT=mcp|rest|both`` selects which shells run. ``both`` shares a single
facade (one Atlas connection pool, one set of workers, two protocols) — the
"memory platform in one deployable unit" story.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

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


def build_combined_app(config: MemoryConfig):
    """Build one FastAPI app serving REST at ``/`` and MCP at ``/mcp``, both
    bound to a SINGLE shared ``AsyncMemory`` created once in the app lifespan.

    The facade is created on the serving event loop (not at import/build time),
    which is required because ``AsyncMongoClient`` binds to the loop it is created
    on. The mounted MCP ASGI app's own lifespan is chained in so its tools
    register against the same facade.
    """
    shared: dict = {"app": None}

    class _Proxy:
        def __getattr__(self, name):
            return getattr(shared["app"], name)

    proxy = _Proxy()
    mcp = create_mcp(config, app=proxy)
    mcp_asgi = mcp.http_app()

    @asynccontextmanager
    async def lifespan(_api):
        # 1. create the single shared facade on the serving loop
        shared["app"] = await AsyncMemory.create(config)
        try:
            # 2. run the mounted MCP sub-app's lifespan (registers tools, etc.)
            async with AsyncExitStack() as stack:
                mcp_lifespan = getattr(mcp_asgi, "lifespan", None) or getattr(
                    mcp_asgi.router, "lifespan_context", None
                )
                if mcp_lifespan is not None:
                    await stack.enter_async_context(mcp_lifespan(mcp_asgi))
                yield
        finally:
            await shared["app"].close()

    api = create_app(proxy, config=config)
    api.router.lifespan_context = lifespan
    api.mount("/mcp", mcp_asgi)
    return api


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
        uvicorn.run(build_combined_app(config), host="0.0.0.0", port=config.port)
        return
    raise ValueError(f"Unknown TRANSPORT: {config.transport!r}")
