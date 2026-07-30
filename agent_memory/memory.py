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
from agent_memory.core.redaction import redact_error
from agent_memory.core.response_limit import cap_results
from agent_memory.exceptions import AccessError, ConfigError, RateLimitError

logger = logging.getLogger(__name__)

# Operations classified as "search" for governance limit mapping (INV-005).
_SEARCH_OPERATIONS = frozenset(
    {"recall_memory", "hybrid_search", "check_cache", "search_activity"}
)


class AsyncMemory:
    """Programmatic async memory core. Build via ``await AsyncMemory.create(cfg)``."""

    # ── Lifecycle ──────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, config: MemoryConfig) -> AsyncMemory:
        """Initialize Atlas, providers, services, and (optionally) workers.

        Equivalent to the former FastMCP ``lifespan`` startup, lifted out and
        made callable by any consumer.
        """
        from agent_memory.core.collections import EPISODES, EPISODES_COUNTERS
        from agent_memory.core.database import DatabaseManager
        from agent_memory.core.migrations import ensure_indexes
        from agent_memory.providers.manager import ProviderManager
        from agent_memory.services.admin import AdminService
        from agent_memory.services.audit import AuditService
        from agent_memory.services.cache import CacheService
        from agent_memory.services.decision import DecisionService
        from agent_memory.services.episodic import EpisodicService
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
        # `config` is passed so the guard can use the model table, which works when
        # the embedding endpoint does not. ProviderManager has already aligned
        # `embedding_dimension` to the Voyage model above, so this compares the
        # post-alignment value.
        await self._validate_embedding_dimension(
            self.providers, expected=config.embedding_dimension, config=config
        )

        # 4. Services
        self.memory_service = MemoryService(db["memories"], config, self.providers)
        self.cache_service = CacheService(
            db["semantic_cache"], config, self.providers.embedding
        )
        self.audit_service = AuditService(db["audit_log"], config)
        self.decision_service = DecisionService(db["decisions"], config)
        # Episodic owns its own writer task; _maybe_start_workers schedules it.
        self.episodic_service = EpisodicService(
            db[EPISODES], config, self.providers,
            counter_collection=db[EPISODES_COUNTERS],
            audit_service=self.audit_service,
        )
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
                "queues. Episodic logging also has no consumer, so log_activity() "
                "fills its bounded queue and then discards the oldest turns; set "
                "episodic_enabled=False to make that explicit."
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
            # Built by ProviderManager from config. Omitting this is a silent no-op
            # for IMPORTANCE_SCORER=local: the worker would construct its own
            # LLMScorer, startup would still log the artifact it loaded, and every
            # enrichment would still bill a token.
            scorer=self.providers.scorer,
        )
        consolidation = ConsolidationWorker(memories, self.config, self.providers)
        audit_flush = AuditFlushWorker(self.audit_service, self.config)
        self._workers = [
            self._supervise(enrichment.run(), "enrichment"),
            self._supervise(consolidation.run(), "consolidation"),
            self._supervise(audit_flush.run(), "audit-flush"),
            # The episodic writer's consumer loop. Unlike the other three it owns
            # queued data, so close() flushes it explicitly rather than relying on
            # task.cancel() — see close().
            self._supervise(self.episodic_service.worker.run(), "episodic-writer"),
        ]

    def _supervise(self, coro, name: str) -> asyncio.Task:
        """Schedule a worker loop and make its death audible.

        A bare ``create_task`` has two failure modes that both end in silence. If
        the coroutine raises, asyncio reports "Task exception was never retrieved"
        — but only when the task object is garbage-collected, which for a task held
        in ``self._workers`` may be at interpreter shutdown or never. And nothing
        else notices at all: the facade keeps answering, ``/health`` keeps
        returning 200, and the only symptom of a dead enrichment loop is that
        memories stop being enriched, which looks like an empty backlog rather than
        a crash.

        Each ``run()`` already wraps its body in try/except, so reaching this
        callback means the loop itself broke rather than one iteration — precisely
        the case worth shouting about. Cancellation during shutdown is expected and
        stays at debug.

        This logs rather than restarts. A crash-looping worker that silently
        restarts forever is harder to diagnose than one that stops and says so, and
        the restart policy belongs to whatever supervises the process — which for a
        library embedded in someone else's app is not ours to choose.
        """
        task = asyncio.create_task(coro, name=f"agent-memory:{name}")

        def _report(finished: asyncio.Task) -> None:
            if finished.cancelled():
                logger.debug("Worker %s cancelled (shutting down).", name)
                return
            exc = finished.exception()
            if exc is not None:
                logger.error(
                    "Worker %s died and will not restart; the work it performs has "
                    "stopped for the lifetime of this process.",
                    name, exc_info=exc,
                )
            else:
                # run() loops until stopped, so returning is itself unexpected
                # unless stop() was called — which close() does before cancelling.
                logger.debug("Worker %s exited cleanly.", name)

        task.add_done_callback(_report)
        return task

    @staticmethod
    async def _validate_embedding_dimension(
        providers, expected: int, config=None
    ) -> None:
        """Raise ConfigError if the embedder's dimension != expected.

        Turns silent vector-index corruption into a fast, legible startup failure
        (REQ-E-031). The failure this prevents does not raise on its own: Atlas
        accepts a 1024-dim vector into a 1536-dim index and simply never returns
        the document from ``$vectorSearch``, so recall goes quietly empty and every
        write until someone notices has to be re-embedded.

        Two sources, checked in that order:

        1. **The model table** (``known_embedding_dimension``), when ``config`` is
           supplied. Free, deterministic, and — importantly — available when the
           embedder is not. The docstring here used to claim a table was consulted
           while the code only ever probed; the table is real now.
        2. **A live probe**, for models the table does not know.

        The probe used to swallow every exception and return, so an unreachable
        embedder meant startup proceeded with the declared dimension unverified —
        which is exactly the situation where a stale ``embedding_dimension`` is
        most likely. Now a probe failure is only tolerated when the table has
        already answered, or when nothing else can answer either; in the latter
        case it is a warning, not a debug line, because the guard did not run.
        """
        if config is not None:
            from agent_memory.providers.manager import known_embedding_dimension

            known = known_embedding_dimension(config)
            if known is not None:
                if known != expected:
                    raise ConfigError(
                        f"embedding_dimension mismatch: config declares {expected} "
                        f"but {config.embedding_model!r} emits {known}. Set "
                        f"embedding_dimension={known} and re-provision the Atlas "
                        f"vector index numDimensions to match."
                    )
                # The table is authoritative for a known model; a probe would only
                # add a network round trip to startup to confirm it.
                return

        try:
            vec = await providers.embedding.generate_embedding("dimension probe")
        except Exception:
            # Nothing could verify the dimension — neither the table nor the
            # embedder. Startup continues, because refusing to boot when the
            # embedding endpoint is briefly down is worse than the risk, but this
            # is said out loud rather than logged at debug and forgotten.
            logger.warning(
                "Embedding dimension could not be verified: the model is not in "
                "the known-dimension table and the embedder is unreachable. If "
                "embedding_dimension (%d) is wrong, vectors will be written that "
                "$vectorSearch silently never returns.",
                expected,
                exc_info=True,
            )
            return
        actual = len(vec)
        if actual != expected:
            raise ConfigError(
                f"embedding_dimension mismatch: config declares {expected} but the "
                f"configured embedder emits {actual}. Set embedding_dimension={actual} "
                f"and re-provision the Atlas vector index numDimensions to match."
            )

    async def close(self) -> None:
        """Drain episodic, cancel workers, flush audit, close the connection.

        Order matters. Episodic is drained *first*, while its consumer task is
        still alive and the database is still open — the queue holds turns that
        have not reached Atlas, and cancelling the task would discard them.
        """
        episodic = getattr(self, "episodic_service", None)
        if episodic is not None:
            try:
                await episodic.close(self.config.episodic_shutdown_timeout_seconds)
            except Exception:
                logger.warning("Episodic drain failed on close.", exc_info=True)

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

    async def __aenter__(self) -> AsyncMemory:
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

    async def _run(self, user_id, operation, category, coro_factory, *, role=None,
                   **audit_fields):
        """access-check → service call → audit. The single consumer path.

        ``role`` is keyword-only and separate from ``audit_fields`` because it is
        an *input* to the access decision, not a field to record. It arrives from
        the shells as ``Caller.role`` — the value of the token's role claim. Every
        facade method accepts and forwards it; before that, ``_check_access`` took
        a ``role`` parameter that nothing ever passed, so every caller was
        evaluated as ``auth_default_role`` and the ``admin`` profile was
        unreachable through any transport.

        Library callers pass nothing and keep the old behaviour: no token, no
        role claim, the configured default.
        """
        start = time.time()
        # The access check is audited, not silent. It used to run ahead of the
        # audit block entirely, so a denied operation and a throttled one left no
        # record at all — the two events an audit log exists to capture were the
        # only two it could not show. An auditor reading it saw a quiet system
        # rather than one refusing requests, and a credential probing for reachable
        # operations left no trace.
        try:
            await self._check_access(user_id, operation, role=role)
            result = await coro_factory()
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            # Three distinct statuses, because these are three different events.
            # "denied" is a decision about who the caller is, "throttled" about how
            # often they ask, "error" a fault in the service. Collapsing them makes
            # a policy refusal indistinguishable from a bug in exactly the records
            # where the difference matters most.
            #
            # RateLimitError is checked FIRST because it subclasses AccessError
            # (see exceptions.py — deliberate, so shells can catch the base). Test
            # the base first and every throttle records as "denied", which is the
            # more alarming of the two labels and would send an operator looking
            # for an authorisation problem that does not exist.
            if isinstance(e, RateLimitError):
                status = "throttled"
            elif isinstance(e, AccessError):
                status = "denied"
            else:
                status = "error"
            # `redact_error` rather than `str(e)`: this string lands in the audit
            # collection, which is retained for weeks and readable by anyone with
            # the admin role. Driver and provider exceptions quote the thing they
            # failed on — a `mongodb+srv://user:password@` URI, an `Authorization`
            # header, the rejected payload — so the unfiltered message writes
            # cluster credentials into the database they authenticate to. The
            # exception *type* is what makes the entry actionable and is kept.
            await self.audit_service.log(
                user_id, category, operation, status, duration_ms,
                error=redact_error(e),
            )
            raise
        duration_ms = int((time.time() - start) * 1000)
        await self.audit_service.log(
            user_id, category, operation, "success", duration_ms, **audit_fields
        )
        return result

    def _results(self, results: list[dict], label: str) -> dict:
        """Wrap a read's documents in the standard envelope, honouring the size cap.

        The single place `max_response_bytes` is applied, which is the point — it
        was declared in the config, documented in the README table, asserted by a
        unit test, and read by absolutely nothing. A limit that exists only as a
        setting is worse than no limit: it is what an operator lowers when a client
        is choking on a response, and lowering it did nothing at all.

        Overflow needs no adversarial input to reach. `limit` bounds the result
        *count*, not its size, and an episodic document carries projected message
        content plus a todo list plus a files-touched array — a hundred turns of a
        long agent conversation is tens of megabytes in one MCP frame or one HTTP
        body. What breaks is the client buffering it, or the model whose context it
        lands in.

        `count` deliberately reports what is *in* the list rather than what was
        found, so it stays honest about the payload; `total_count` appears
        alongside `truncated` when the two differ. Nothing is added on the common
        path, so an untruncated response is byte-identical to before.
        """
        kept, meta = cap_results(
            results, self.config.max_response_bytes, label=label
        )
        return {"results": kept, "count": len(kept), **meta}

    # ── Public method surface ────────────────────────────────────────────────

    # Every method takes a keyword-only `role`: the role claim from the caller's
    # verified token, forwarded to the access check. Omitted by library callers,
    # who have no token and get `auth_default_role` as before.

    async def add(self, user_id: str, conversation_id: str, messages: list[dict],
                  *, role=None) -> dict:
        """Store conversation messages as short-term memory.

        Each message needs ``content``; ``message_type`` defaults from ``role``, or
        to ``"human"``. Returns ``{"stm_ids": [...], "count": n}``.

        Significant human messages additionally seed a long-term candidate, which
        the enrichment worker scores, summarises, and de-duplicates in the
        background — so this call stays one round trip and no LLM work happens on
        the caller's path.
        """
        # Copy before defaulting `message_type`: `add()` used to mutate the
        # caller's own dicts, so a caller that reused its message list saw fields
        # appear in it. A transport is not entitled to edit its caller's data.
        messages = [
            msg if "message_type" in msg
            else {**msg, "message_type": msg.get("role", "human")}
            for msg in messages
        ]

        async def _do():
            stm_ids = await self.memory_service.store_stm(user_id, conversation_id, messages)
            return {"stm_ids": stm_ids, "count": len(stm_ids)}

        return await self._run(
            user_id, "store_memory", "memory:write", _do, role=role,
            conversation_id=conversation_id, count=len(messages),
        )

    async def recall(self, user_id: str, query: str, *, tier=None, memory_type=None,
                     tags=None, limit: int = 10, role=None) -> dict:
        """Retrieve memories relevant to ``query``, curated for use as context.

        Vector search followed by calibrated re-ranking over recency, importance,
        and relevance, with short-term/long-term duplicates collapsed. Returns
        ``{"results": [...], "count": n}``.

        This is the tier-aware read: prefer it when assembling a prompt. Use
        ``search`` instead when you want raw fused relevance with no curation.
        """
        async def _do():
            results = await self.memory_service.recall(
                user_id, query, tier=tier, memory_type=memory_type, tags=tags, limit=limit
            )
            return self._results(results, "memories")

        return await self._run(
            user_id, "recall_memory", "memory:read", _do, role=role, query=query,
        )

    async def search(self, user_id: str, query: str, *, tier=None, limit: int = 10,
                     memory_type=None, tags=None, role=None) -> dict:
        """Hybrid search over memories via ``$rankFusion`` — raw relevance, no curation.

        Reciprocal rank fusion over a vector branch and a full-text branch in one
        round trip, so exact terms (SKUs, error codes, names) and meaning both
        count. Results carry their fused ``score``.

        The counterpart to ``recall``: no re-ranking, no duplicate collapsing.
        Reach for this when you want to see what matched and why.
        """
        async def _do():
            results = await self.memory_service.hybrid_search(
                user_id, query, tier=tier, limit=limit, memory_type=memory_type, tags=tags
            )
            return self._results(results, "memories")

        return await self._run(
            user_id, "hybrid_search", "search", _do, role=role, query=query,
        )

    async def delete(self, user_id: str, *, memory_id=None, tags=None, time_range=None,
                     confirm: bool = False, dry_run: bool = False, role=None) -> dict:
        """Soft-delete memories by id, tags, or time range.

        Returns ``{"deleted_count": n}``. Soft, not hard: documents are marked
        ``deleted_at`` and excluded from every read, then reaped by a TTL index —
        so a mistaken bulk delete is recoverable for a window.

        Anything other than a single ``memory_id`` is a bulk delete and requires
        ``confirm=True``. Pair it with ``dry_run=True`` first to see the count
        without writing. For erasure obligations use ``wipe_user_data``, which
        deletes permanently and across every collection.
        """
        async def _do():
            return await self.memory_service.delete(
                user_id, memory_id=memory_id, tags=tags, time_range=time_range,
                confirm=confirm, dry_run=dry_run,
            )

        return await self._run(user_id, "delete_memory", "memory:delete", _do,
                               role=role, dry_run=dry_run)

    async def check_cache(self, user_id: str, query: str, *, similarity_threshold=None,
                          role=None) -> dict | None:
        async def _do():
            return await self.cache_service.check(
                user_id, query, similarity_threshold=similarity_threshold
            )

        return await self._run(user_id, "check_cache", "cache:read", _do, role=role)

    async def store_cache(self, user_id: str, query: str, response: str, *, role=None) -> str:
        async def _do():
            return await self.cache_service.store(user_id, query, response)

        return await self._run(user_id, "store_cache", "cache:write", _do, role=role)

    async def invalidate_cache(self, user_id: str, *, pattern=None,
                               invalidate_all: bool = False, role=None) -> dict:
        async def _do():
            deleted = await self.cache_service.invalidate(
                user_id, pattern=pattern, invalidate_all=invalidate_all
            )
            return {"user_id": user_id, "deleted_count": deleted}

        return await self._run(user_id, "cache_invalidate", "admin", _do, role=role)

    async def remember_decision(self, user_id: str, key: str, value: str, *,
                                ttl_days=None, role=None) -> dict:
        async def _do():
            status = await self.decision_service.store(user_id, key, value, ttl_days=ttl_days)
            return {"key": key, "status": status}

        return await self._run(user_id, "store_decision", "decision:write", _do,
                               role=role, key=key)

    async def recall_decision(self, user_id: str, key: str, *, role=None) -> dict | None:
        async def _do():
            return await self.decision_service.recall(user_id, key)

        return await self._run(user_id, "recall_decision", "decision:read", _do,
                               role=role, key=key)

    # ── Episodic memory (the agent activity log) ─────────────────────────────

    async def log_activity(self, user_id: str, thread_id: str, messages: list,
                           *, todos=None, agent_name=None, correlation_id=None,
                           conversation_id=None, ts=None, role=None) -> dict:
        """Record one agent turn. Non-blocking: enqueues and returns.

        Deliberately does **not** go through ``_run``. ``_run`` writes one audit
        record per call, and a turn log is high-volume by nature — routing it
        there produces audit amplification, where logging the agent costs more
        writes than the agent. Governance and rate limiting still apply via
        ``_check_access``; the worker emits one audit entry per flushed batch.

        Refusals *are* audited even though successes are batched. Skipping the
        per-call audit is a volume decision about the success path; a denial is
        rare, security-relevant, and the thing an audit log is for.
        """
        try:
            await self._check_access(user_id, "log_activity", role=role)
        except AccessError as e:
            # RateLimitError first: it subclasses AccessError, so the base test
            # matches both and would label every throttle "denied". Catching the
            # base is still right — it is the only thing `_check_access` raises.
            status = "throttled" if isinstance(e, RateLimitError) else "denied"
            await self.audit_service.log(
                user_id, "episodic:write", "log_activity", status, 0,
                error=redact_error(e),
            )
            raise
        enqueued = self.episodic_service.log_activity(
            user_id, thread_id, messages, todos=todos, agent_name=agent_name,
            correlation_id=correlation_id, conversation_id=conversation_id, ts=ts,
        )
        return {"enqueued": enqueued, "thread_id": thread_id}

    async def recall_activity(self, user_id: str, query: str, *, thread_id=None,
                              agent_name=None, since=None, limit: int = 5,
                              role=None) -> dict:
        """Hybrid recall over logged turns — "what did I actually do?"."""
        async def _do():
            results = await self.episodic_service.search(
                user_id, query, thread_id=thread_id, agent_name=agent_name,
                since=since, limit=limit,
            )
            return self._results(results, "turns")

        return await self._run(
            user_id, "search_activity", "episodic:read", _do, role=role, query=query,
        )

    async def get_thread(self, user_id: str, thread_id: str, *, limit=None,
                        ascending: bool = True, role=None) -> dict:
        """Replay a thread's turns in step order."""
        async def _do():
            results = await self.episodic_service.get_thread(
                user_id, thread_id, limit=limit, ascending=ascending
            )
            return self._results(results, "turns")

        return await self._run(
            user_id, "get_thread", "episodic:read", _do, role=role,
            thread_id=thread_id,
        )

    async def get_activity_by_correlation(self, user_id: str, correlation_id: str,
                                          *, limit=None, role=None) -> dict:
        """Every turn sharing a trace id — the join back to your tracing stack."""
        async def _do():
            results = await self.episodic_service.get_by_correlation_id(
                user_id, correlation_id, limit=limit
            )
            return self._results(results, "turns")

        return await self._run(
            user_id, "get_correlation", "episodic:read", _do, role=role,
            correlation_id=correlation_id,
        )

    async def flush_activity(self, timeout: float = 5.0) -> bool:
        """Wait for queued turns to reach Atlas. No user scope; no audit entry.

        Not an access-controlled operation: it is a lifecycle call about this
        process's own buffer, not a read or write of anyone's data.
        """
        return await self.episodic_service.flush(timeout)

    async def set_activity_retention(self, user_id: str, *, ttl_seconds, role=None) -> dict:
        """Change episodic retention **collection-wide**. ``None`` = keep forever.

        ``user_id`` is the principal this operation is authorised and audited
        against, *not* a scope: a TTL index belongs to the collection, so the new
        retention applies to every tenant's turns. The result carries
        ``scope: "collection"`` to say so at the call site. This is an ``admin``
        operation and is withheld from ``power_user`` precisely because one tenant
        must not be able to shorten another's retention.
        """
        async def _do():
            return await self.episodic_service.set_retention(ttl_seconds)

        return await self._run(
            user_id, "set_activity_retention", "admin", _do, role=role,
            ttl_seconds=ttl_seconds,
        )

    def activity_stats(self) -> dict:
        """Episodic queue and throughput counters. Synchronous, no round trip."""
        return self.episodic_service.stats()

    def worker_status(self) -> dict:
        """Liveness of each in-process worker. Synchronous, no round trip.

        A dead worker is otherwise invisible to a health probe: the facade keeps
        answering reads and writes normally, because the workers do the *reactive*
        half of the system. Enrichment stopping means memories are stored but never
        scored or de-duplicated; consolidation stopping means nothing is promoted or
        forgotten. Both look like a quiet system rather than a broken one.

        ``running`` is the aggregate an alert should watch: with
        ``workers_in_process=True`` it is false whenever any worker has exited,
        which — since each ``run()`` loops until told to stop — means it crashed.
        When ``workers_in_process=False`` there are no tasks and ``running`` is
        false by definition; ``enabled`` distinguishes the two cases.
        """
        tasks = getattr(self, "_workers", [])
        detail = {}
        for task in tasks:
            # Names are set by _supervise as "agent-memory:<name>".
            label = (task.get_name() or "worker").removeprefix("agent-memory:")
            entry: dict = {"done": task.done(), "cancelled": task.cancelled()}
            if task.done() and not task.cancelled():
                exc = task.exception()
                # `redact_error`, not `repr(exc)`. This dict is served by
                # `/health`, which is deliberately the one route exempt from auth —
                # a probe that needs a token is a probe that fails during exactly
                # the incident it exists to detect. That exemption is fine for
                # counters and booleans and not fine for an exception repr: a
                # crashed worker's exception is most often a driver or provider
                # error, and those quote what they failed on. An unauthenticated
                # endpoint would then serve the cluster's connection string,
                # credentials included, to anyone who can reach the port.
                entry["error"] = redact_error(exc) if exc is not None else None
            detail[label] = entry
        return {
            "enabled": bool(self.config.workers_in_process),
            "running": bool(tasks) and all(not t.done() for t in tasks),
            "workers": detail,
        }

    async def health(self, user_id: str, *, role=None) -> dict:
        async def _do():
            return await self.admin_service.health(user_id)

        return await self._run(user_id, "memory_health", "admin", _do, role=role)

    async def wipe_user_data(self, user_id: str, confirm: bool = False, *, role=None) -> dict:
        if not confirm:
            return {
                "error": "wipe_user_data requires confirm=true. "
                "This will permanently delete ALL user data."
            }

        async def _do():
            return await self.admin_service.wipe_user_data(user_id)

        return await self._run(user_id, "wipe_user_data", "admin", _do, role=role)


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
