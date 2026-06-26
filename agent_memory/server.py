"""FastMCP server with lifespan integration for Memory-MCP v3.2."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from agent_memory.auth.api_keys import APIKeyManager
from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier
from agent_memory.core.config import MCPConfig
from agent_memory.core.database import DatabaseManager
from agent_memory.core.migrations import ensure_indexes, ensure_search_indexes
from agent_memory.core.registry import ServiceRegistry
from agent_memory.providers.manager import ProviderManager
from agent_memory.services.audit import AuditService
from agent_memory.services.audit_flush_worker import AuditFlushWorker
from agent_memory.services.auto_capture import AutoCaptureMiddleware, wrap_tools
from agent_memory.services.cache import CacheService
from agent_memory.services.consolidation import ConsolidationWorker
from agent_memory.services.decision import DecisionService
from agent_memory.services.enrichment import EnrichmentWorker
from agent_memory.services.governance import GovernanceService
from agent_memory.services.memory import MemoryService
from agent_memory.services.prompt_library import PromptLibrary
from agent_memory.services.rate_limiter import RateLimiter
from agent_memory.tools.admin_tools import register_admin_tools
from agent_memory.tools.cache_tools import register_cache_tools
from agent_memory.tools.decision_tools import register_decision_tools
from agent_memory.tools.memory_tools import register_memory_tools
from agent_memory.tools.search_tools import register_search_tools

logger = logging.getLogger(__name__)


def _build_auth(config: MCPConfig) -> MemoryMCPTokenVerifier | None:
    """Return a token verifier when auth is enabled, else None."""
    if not config.auth_enabled:
        return None
    if not config.auth_secret:
        logger.warning("AUTH_ENABLED=true but AUTH_SECRET is empty — auth disabled.")
        return None
    api_key_manager = APIKeyManager()
    return MemoryMCPTokenVerifier(
        secret=config.auth_secret,
        api_key_manager=api_key_manager,
    )


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Initialize all services at startup, tear down on shutdown."""
    # Startup
    config = MCPConfig()
    db_manager = await DatabaseManager.initialize(config)

    # Stage 1: Standard indexes (fast, blocking)
    await ensure_indexes(db_manager.db)
    logger.info("Standard indexes ensured.")

    providers = ProviderManager(config)

    memory_service = MemoryService(
        db_manager.db["memories"], config, providers,
    )
    cache_service = CacheService(
        db_manager.db["semantic_cache"], config, providers.embedding,
    )
    audit_service = AuditService(
        db_manager.db["audit_log"], config,
    )

    registry = ServiceRegistry.initialize(
        config=config,
        memory_service=memory_service,
        cache_service=cache_service,
        audit_service=audit_service,
        providers=providers,
    )

    # Conditionally create Phase 2 services
    if config.governance_enabled:
        registry.governance_service = GovernanceService(
            db_manager.db["governance_profiles"], config,
        )
    if config.rate_limit_enabled:
        registry.rate_limiter = RateLimiter(
            db_manager.db["rate_limits"], config,
        )

    prompt_library = PromptLibrary(
        db_manager.db["prompts"], config,
    )
    registry.prompt_library = prompt_library

    decision_service = DecisionService(
        db_manager.db["decisions"], config,
    )
    registry.decision_service = decision_service

    # Stage 1b: Seed essential data (best-effort, non-fatal)
    if config.governance_enabled and registry.governance_service is not None:
        try:
            count = await registry.governance_service.seed_defaults()
            logger.info("Governance profiles seeded: %d new.", count)
        except Exception:
            logger.warning("Governance seed failed (non-fatal).", exc_info=True)

    try:
        count = await prompt_library.seed_defaults()
        logger.info("Prompt templates seeded: %d new.", count)
    except Exception:
        logger.warning("Prompt seed failed (non-fatal).", exc_info=True)

    try:
        count = await decision_service.seed_defaults()
        logger.info("System decisions seeded: %d new.", count)
    except Exception:
        logger.warning("Decision seed failed (non-fatal).", exc_info=True)

    # Start enrichment background task
    enrichment_worker = EnrichmentWorker(
        db_manager.db["memories"], config, providers, memory_service,
        prompt_library=registry.prompt_library,
    )
    enrichment_task = asyncio.create_task(enrichment_worker.run())

    # Start consolidation background task
    consolidation_worker = ConsolidationWorker(
        db_manager.db["memories"], config, providers,
    )
    consolidation_task = asyncio.create_task(consolidation_worker.run())

    # Start audit flush background task
    audit_flush_worker = AuditFlushWorker(audit_service, config)
    audit_flush_task = asyncio.create_task(audit_flush_worker.run())

    # Stage 2: Atlas Search indexes (background, non-blocking)
    search_index_task = asyncio.create_task(
        _ensure_search_indexes_bg(db_manager.db, config.embedding_dimension)
    )

    # Auto-capture: wrap registered tools with memory capture
    if config.auto_capture_enabled:
        auto_capture = AutoCaptureMiddleware(memory_service, config)
        wrap_tools(mcp, auto_capture)
        logger.info("Auto-capture enabled for %d tool(s).", len(config.auto_capture_tools))

    logger.info("Memory-MCP v%s started", config.app_version)

    yield

    # Shutdown
    enrichment_task.cancel()
    consolidation_task.cancel()
    audit_flush_task.cancel()
    if not search_index_task.done():
        search_index_task.cancel()
    await audit_service.flush()
    await db_manager.close()
    logger.info("Memory-MCP shut down")


async def _ensure_search_indexes_bg(db, embedding_dimension: int = 1536) -> None:
    """Background wrapper for Atlas Search index creation.

    Exceptions are logged but never propagated — search index creation
    is non-fatal.
    """
    try:
        await ensure_search_indexes(db, embedding_dimension=embedding_dimension)
        logger.info("Atlas Search indexes ready.")
    except asyncio.CancelledError:
        logger.debug("Atlas Search index creation cancelled (server shutting down).")
    except Exception:
        logger.warning(
            "Atlas Search index creation failed (non-fatal).",
            exc_info=True,
        )


_config = MCPConfig()
_auth = _build_auth(_config)

mcp = FastMCP("MongoDB Memory MCP Server", lifespan=lifespan, auth=_auth)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Unauthenticated health check endpoint for Docker/load balancer probes."""
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok"})


# Register all tools
register_memory_tools(mcp)
register_cache_tools(mcp)
register_search_tools(mcp)
register_admin_tools(mcp)
register_decision_tools(mcp)
