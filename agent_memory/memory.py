"""AsyncMemory facade (core) + Memory sync wrapper.

``AsyncMemory`` is the real implementation used directly by the async-native
MCP/REST shells. It absorbs the orchestration pattern that every MCP tool used
to duplicate (``check_access`` → call service → ``audit_service.log``) into one
``_run`` path, so every consumer gets identical access control and auditing.

``Memory`` is a thin synchronous wrapper that runs the async core on a dedicated
background event loop (notebook/Jupyter-safe). It is defined in
``agent_memory/sync.py`` and re-exported here.
"""

from __future__ import annotations

import asyncio
import logging
import time

from agent_memory.config import MemoryConfig
from agent_memory.exceptions import AccessError, ConfigError, RateLimitError

logger = logging.getLogger(__name__)

# Operations classified as "search" for governance limit mapping (INV-005).
_SEARCH_OPERATIONS = frozenset(
    {"recall_memory", "hybrid_search", "search_web", "check_cache"}
)


class AsyncMemory:
    """Programmatic async memory core. Build via ``await AsyncMemory.create(cfg)``."""

    # ── Lifecycle ──────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, config: MemoryConfig) -> "AsyncMemory":
        """Initialize Atlas, providers, services, and (optionally) workers.

        Equivalent to the former FastMCP ``lifespan`` startup, lifted out and
        made callable by any consumer.
        """
        from agent_memory.core.database import DatabaseManager
        from agent_memory.core.migrations import ensure_indexes, ensure_search_indexes
        from agent_memory.providers.manager import ProviderManager
        from agent_memory.services.admin import AdminService
        from agent_memory.services.audit import AuditService
        from agent_memory.services.cache import CacheService
        from agent_memory.services.decision import DecisionService
        from agent_memory.services.governance import GovernanceService
        from agent_memory.services.memory import MemoryService
        from agent_memory.services.prompt_library import PromptLibrary
        from agent_memory.services.rate_limiter import RateLimiter

        self = cls.__new__(cls)
        self.config = config
        self._workers = []

        # 1. Database + 2. Stage-1 indexes (blocking)
        db_manager = await DatabaseManager.initialize(config)
        self._db_manager = db_manager
        db = db_manager.db
        await ensure_indexes(db)

        # 3. Providers + embedding-dimension guard (before any vector write)
        self.providers = ProviderManager(config)
        await self._validate_embedding_dimension(
            self.providers, expected=config.embedding_dimension
        )

        # 4. Services
        self.memory_service = MemoryService(db["memories"], config, self.providers)
        self.cache_service = CacheService(
            db["semantic_cache"], config, self.providers.embedding
        )
        self.audit_service = AuditService(db["audit_log"], config)
        self.decision_service = DecisionService(db["decisions"], config)
        self.prompt_library = PromptLibrary(db["prompts"], config)
        self.admin_service = AdminService(db)

        self.governance_service = (
            GovernanceService(db["governance_profiles"], config)
            if config.governance_enabled
            else None
        )
        self.rate_limiter = (
            RateLimiter(db["rate_limits"], config)
            if config.rate_limit_enabled
            else None
        )

        # 5. Seed defaults (best-effort, non-fatal)
        await self._seed_defaults()

        # 6. Workers (conditional on the SP1 seam)
        await self._maybe_start_workers()

        # 7. Stage-2 Atlas Search indexes (awaited or backgrounded per config)
        await self._provision_search_indexes(db, config)

        logger.info("AsyncMemory started (workers_in_process=%s)", config.workers_in_process)
        return self

    async def _provision_search_indexes(self, db, config, ensure=None) -> None:
        """Create Atlas Search indexes — awaited or backgrounded per config.

        ``await_search_indexes=True`` blocks until indexes are queryable (right
        for short-lived library/script callers, which would otherwise exit
        before background creation finishes and see empty search/recall).
        ``False`` (default) schedules a non-blocking background task for
        long-running servers. ``ensure`` is injectable for testing.
        """
        from agent_memory.core.migrations import ensure_search_indexes

        ensure = ensure or ensure_search_indexes
        self._search_index_task = None
        if config.await_search_indexes:
            await ensure(db, embedding_dimension=config.embedding_dimension)
        else:
            self._search_index_task = asyncio.create_task(
                _ensure_search_indexes_bg(
                    db, config.embedding_dimension, ensure=ensure
                )
            )

    async def _seed_defaults(self) -> None:
        if self.governance_service is not None:
            try:
                await self.governance_service.seed_defaults()
            except Exception:
                logger.warning("Governance seed failed (non-fatal).", exc_info=True)
        for svc in (self.prompt_library, self.decision_service):
            try:
                await svc.seed_defaults()
            except Exception:
                logger.warning("Seed failed (non-fatal).", exc_info=True)

    async def _maybe_start_workers(self) -> None:
        """Start in-process workers, or warn that reactive work is disabled."""
        if not self.config.workers_in_process:
            logger.warning(
                "workers_in_process=False: in-process enrichment/consolidation/"
                "audit-flush are disabled. Memories persist but are never enriched, "
                "promoted, or forgotten until an external runtime (SP1) drains the "
                "queues."
            )
            self._workers = []
            return

        from agent_memory.services.audit_flush_worker import AuditFlushWorker
        from agent_memory.services.consolidation import ConsolidationWorker
        from agent_memory.services.enrichment import EnrichmentWorker

        memories = self.memory_service.memories
        enrichment = EnrichmentWorker(
            memories, self.config, self.providers, self.memory_service,
            prompt_library=self.prompt_library,
        )
        consolidation = ConsolidationWorker(memories, self.config, self.providers)
        audit_flush = AuditFlushWorker(self.audit_service, self.config)
        self._workers = [
            asyncio.create_task(enrichment.run()),
            asyncio.create_task(consolidation.run()),
            asyncio.create_task(audit_flush.run()),
        ]

    @staticmethod
    async def _validate_embedding_dimension(providers, expected: int) -> None:
        """Raise ConfigError if the live embedder's dimension != expected.

        Uses a known per-model table where possible; otherwise probes one
        embedding's length. Turns silent vector-index corruption into a fast,
        legible startup failure (REQ-E-031).
        """
        # Probe is authoritative and cheap; do it directly.
        try:
            vec = await providers.embedding.generate_embedding("dimension probe")
        except Exception:
            # Cannot probe (e.g. offline) — skip rather than block startup.
            logger.debug("Embedding dimension probe skipped (embedder unavailable).")
            return
        actual = len(vec)
        if actual != expected:
            raise ConfigError(
                f"embedding_dimension mismatch: config declares {expected} but the "
                f"configured embedder emits {actual}. Set embedding_dimension={actual} "
                f"and re-provision the Atlas vector index numDimensions to match."
            )

    async def close(self) -> None:
        """Cancel workers, flush audit, close the database connection."""
        for task in getattr(self, "_workers", []):
            task.cancel()
        search_task = getattr(self, "_search_index_task", None)
        if search_task is not None and not search_task.done():
            search_task.cancel()
        if getattr(self, "audit_service", None) is not None:
            await self.audit_service.flush()
        if getattr(self, "_db_manager", None) is not None:
            await self._db_manager.close()
        logger.info("AsyncMemory closed.")

    async def __aenter__(self) -> "AsyncMemory":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ── Orchestration ──────────────────────────────────────────────────────

    async def _check_access(self, user_id: str, operation: str, role: str | None = None) -> None:
        """Governance THEN rate limit. Raises AccessError / RateLimitError."""
        effective_role = role or self.config.auth_default_role

        profile = None
        if self.governance_service is not None:
            allowed = await self.governance_service.check_allowed(
                user_id, effective_role, operation
            )
            if not allowed:
                raise AccessError(
                    f"Operation '{operation}' not allowed for role '{effective_role}'"
                )
            profile = await self.governance_service.get_profile(effective_role)

        if self.rate_limiter is not None:
            if profile is not None:
                if operation in _SEARCH_OPERATIONS:
                    max_requests = profile.get("max_searches_per_day")
                else:
                    max_requests = profile.get("max_memories_per_day")
                within = await self.rate_limiter.check_rate_limit(
                    user_id, operation, max_requests=max_requests
                )
            else:
                within = await self.rate_limiter.check_rate_limit(user_id, operation)
            if not within:
                raise RateLimitError(f"Rate limit exceeded for '{operation}'")

    async def _run(self, user_id, operation, category, coro_factory, **audit_fields):
        """access-check → service call → audit. The single consumer path."""
        await self._check_access(user_id, operation)
        start = time.time()
        try:
            result = await coro_factory()
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await self.audit_service.log(
                user_id, category, operation, "error", duration_ms, error=str(e)
            )
            raise
        duration_ms = int((time.time() - start) * 1000)
        await self.audit_service.log(
            user_id, category, operation, "success", duration_ms, **audit_fields
        )
        return result

    # ── Public method surface ────────────────────────────────────────────────

    async def add(self, user_id: str, conversation_id: str, messages: list[dict]) -> dict:
        for msg in messages:
            if "message_type" not in msg:
                msg["message_type"] = msg.get("role", "human")

        async def _do():
            stm_ids = await self.memory_service.store_stm(user_id, conversation_id, messages)
            return {"stm_ids": stm_ids, "count": len(stm_ids)}

        return await self._run(
            user_id, "store_memory", "memory:write", _do,
            conversation_id=conversation_id, count=len(messages),
        )

    async def recall(self, user_id: str, query: str, *, tier=None, memory_type=None,
                     tags=None, limit: int = 10) -> dict:
        async def _do():
            results = await self.memory_service.recall(
                user_id, query, tier=tier, memory_type=memory_type, tags=tags, limit=limit
            )
            return {"results": results, "count": len(results)}

        return await self._run(
            user_id, "recall_memory", "memory:read", _do, query=query,
        )

    async def search(self, user_id: str, query: str, *, tier=None, limit: int = 10,
                     memory_type=None, tags=None) -> dict:
        async def _do():
            results = await self.memory_service.hybrid_search(
                user_id, query, tier=tier, limit=limit, memory_type=memory_type, tags=tags
            )
            return {"results": results, "count": len(results)}

        return await self._run(
            user_id, "hybrid_search", "search", _do, query=query,
        )

    async def delete(self, user_id: str, *, memory_id=None, tags=None, time_range=None,
                     confirm: bool = False, dry_run: bool = False) -> dict:
        async def _do():
            return await self.memory_service.delete(
                user_id, memory_id=memory_id, tags=tags, time_range=time_range,
                confirm=confirm, dry_run=dry_run,
            )

        return await self._run(user_id, "delete_memory", "memory:delete", _do, dry_run=dry_run)

    async def check_cache(self, user_id: str, query: str, *, similarity_threshold=None) -> dict | None:
        async def _do():
            return await self.cache_service.check(
                user_id, query, similarity_threshold=similarity_threshold
            )

        return await self._run(user_id, "check_cache", "cache:read", _do)

    async def store_cache(self, user_id: str, query: str, response: str) -> str:
        async def _do():
            return await self.cache_service.store(user_id, query, response)

        return await self._run(user_id, "store_cache", "cache:write", _do)

    async def invalidate_cache(self, user_id: str, *, pattern=None, invalidate_all: bool = False) -> dict:
        async def _do():
            deleted = await self.cache_service.invalidate(
                user_id, pattern=pattern, invalidate_all=invalidate_all
            )
            return {"user_id": user_id, "deleted_count": deleted}

        return await self._run(user_id, "cache_invalidate", "admin", _do)

    async def remember_decision(self, user_id: str, key: str, value: str, *, ttl_days=None) -> dict:
        async def _do():
            status = await self.decision_service.store(user_id, key, value, ttl_days=ttl_days)
            return {"key": key, "status": status}

        return await self._run(user_id, "store_decision", "decision:write", _do, key=key)

    async def recall_decision(self, user_id: str, key: str) -> dict | None:
        async def _do():
            return await self.decision_service.recall(user_id, key)

        return await self._run(user_id, "recall_decision", "decision:read", _do, key=key)

    async def search_web(self, user_id: str, query: str) -> dict:
        async def _do():
            if not self.config.tavily_api_key:
                return {"error": "Web search service unavailable: Tavily API key not configured"}
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.config.tavily_api_key)
            response = await asyncio.to_thread(client.search, query)
            return {"results": response.get("results", []), "query": query}

        return await self._run(user_id, "search_web", "search", _do, query=query)

    async def health(self, user_id: str) -> dict:
        async def _do():
            return await self.admin_service.health(user_id)

        return await self._run(user_id, "memory_health", "admin", _do)

    async def wipe_user_data(self, user_id: str, confirm: bool = False) -> dict:
        if not confirm:
            return {
                "error": "wipe_user_data requires confirm=true. "
                "This will permanently delete ALL user data."
            }

        async def _do():
            return await self.admin_service.wipe_user_data(user_id)

        return await self._run(user_id, "wipe_user_data", "admin", _do)


async def _ensure_search_indexes_bg(db, embedding_dimension: int = 1536, ensure=None) -> None:
    """Background Atlas Search index creation — failures are non-fatal."""
    from agent_memory.core.migrations import ensure_search_indexes

    ensure = ensure or ensure_search_indexes
    try:
        await ensure(db, embedding_dimension=embedding_dimension)
        logger.info("Atlas Search indexes ready.")
    except asyncio.CancelledError:
        logger.debug("Atlas Search index creation cancelled (shutting down).")
    except Exception:
        logger.warning("Atlas Search index creation failed (non-fatal).", exc_info=True)


# Re-export the sync wrapper (Task 3) so `from agent_memory.memory import Memory` works.
try:  # pragma: no cover - import wiring
    from agent_memory.sync import Memory  # noqa: F401
except ImportError:  # sync wrapper not yet present during early TDD
    pass
