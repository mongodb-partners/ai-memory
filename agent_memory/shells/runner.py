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
        # `config` is not optional here. Omitting it built the REST app with
        # `config=None`, which makes `_build_auth_dependency` return the no-op —
        # so `TRANSPORT=rest` served every route unauthenticated no matter what
        # AUTH_ENABLED said, while the MCP shell built from the same config
        # enforced it.
        shells["rest"] = create_app(app, config=config)
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


#: Addresses that reach only this host. A shell bound to one of these is
#: reachable by processes on the same machine and nothing else, which is the
#: local-development posture where unauthenticated access is reasonable.
#: ``::1`` and its IPv4-mapped form are listed because uvicorn accepts both.
_LOOPBACK_HOSTS = frozenset(
    {"127.0.0.1", "localhost", "::1", "[::1]", "::ffff:127.0.0.1"}
)


def _is_loopback(host: str) -> bool:
    """Whether ``host`` binds loopback only.

    Anything in ``127.0.0.0/8`` counts, not just ``127.0.0.1`` — ``127.0.0.2`` is
    as unroutable as its more famous sibling, and treating it as public would
    refuse a working local setup for no gain. Everything else is treated as
    routable, including the empty string and ``0.0.0.0``: when the address is not
    recognizably local, the safe reading is that it is reachable.
    """
    normalized = (host or "").strip().strip("[]").lower()
    if normalized in {h.strip("[]") for h in _LOOPBACK_HOSTS}:
        return True
    return normalized.startswith("127.")


def _refuse_to_serve_open(config: MemoryConfig, host: str) -> None:
    """Refuse to bind a routable address with authentication disabled.

    With ``auth_enabled`` off, ``resolve_caller`` takes the identity from the
    request — every caller names the ``user_id`` it acts as. In-process that is
    the documented single-tenant posture and it is correct: the calling app has
    already authenticated its user. On a published port it means any client that
    can route to the process may read any tenant's memories or invoke the
    permanent-erasure path against them, and there is no record of who did.

    The default was to serve exactly that: ``AUTH_ENABLED`` off, ``0.0.0.0``
    hardcoded at every bind, and ``docker-compose.yml`` publishing 8000.

    This cannot be a validator on ``MCPConfig``. The same class configures the
    library used in-process, where auth-off is right and no socket is involved;
    a model-level refusal would reject the majority of correct uses. ``run()`` is
    the narrowest place that knows a listening socket is about to exist.

    ``require_auth_for_multi_tenant`` does not cover this. It is the inverse
    assertion — "refuse to start without auth" — which an operator sets *having
    already thought about it*. This is the case where they have not.
    """
    if config.auth_enabled or _is_loopback(host):
        return
    if config.allow_unauthenticated_network_access:
        # Deliberate: an internal-only deployment behind its own gateway is a
        # real configuration. Logged at warning on every start, because "we set
        # that flag for a spike" is how it survives into production unnoticed.
        logger.warning(
            "Serving UNAUTHENTICATED on %s:%s — every request names the user_id "
            "it acts as, so any client that can reach this port can read or "
            "erase any tenant's memories. Permitted by "
            "ALLOW_UNAUTHENTICATED_NETWORK_ACCESS=true.",
            host,
            config.port,
        )
        return
    raise RuntimeError(
        f"Refusing to serve {host}:{config.port} with authentication disabled. "
        "Any client that can reach this port could read or permanently erase "
        "any user's memories, because with AUTH_ENABLED=false the caller "
        "supplies its own user_id.\n"
        "  • To secure it: set AUTH_ENABLED=true and AUTH_SECRET.\n"
        f"  • For local development: HOST=127.0.0.1 (the default; {host} was "
        "requested).\n"
        "  • To accept the risk on a trusted network: "
        "ALLOW_UNAUTHENTICATED_NETWORK_ACCESS=true."
    )


def run(config: MemoryConfig | None = None) -> None:
    """Blocking entrypoint used by ``python -m agent_memory``.

    For a single shell, runs it directly. For ``both``, mounts the REST app and
    the MCP ASGI app under one uvicorn server so both share the process and the
    one facade created in the MCP lifespan.

    Binds ``config.host`` — loopback unless asked otherwise — and refuses to
    bind a routable address without authentication. See
    ``_refuse_to_serve_open``.
    """
    import uvicorn

    config = config or MemoryConfig.from_env()
    transport = (config.transport or "mcp").lower()
    host = config.host
    # Checked before dispatch so it holds for every transport. Previously each
    # branch passed its own hardcoded "0.0.0.0", so a guard added to one of them
    # would have left the other two open.
    _refuse_to_serve_open(config, host)
    if transport in ("streamable-http", "stdio", "mcp"):
        create_mcp(config).run(
            transport="streamable-http", host=host, port=config.port
        )
        return
    if transport == "rest":
        # REST shell owns the facade lifecycle via FastAPI lifespan.
        from agent_memory.shells.rest.app import create_managed_app

        uvicorn.run(create_managed_app(config), host=host, port=config.port)
        return
    if transport == "both":
        uvicorn.run(build_combined_app(config), host=host, port=config.port)
        return
    raise ValueError(f"Unknown TRANSPORT: {config.transport!r}")
