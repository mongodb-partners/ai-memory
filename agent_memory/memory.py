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
from agent_memory.exceptions import (
    AccessError,
    ConfigError,
    ErasureInProgressError,
    RateLimitError,
)
from agent_memory.services.audit import ERASURE_PRINCIPAL

logger = logging.getLogger(__name__)

# Operations classified as "search" for governance limit mapping (INV-005).
_SEARCH_OPERATIONS = frozenset(
    {"recall_memory", "hybrid_search", "check_cache", "search_activity"}
)

# Operations that persist something attributable to a user. A wipe in progress
# refuses exactly these, because a write that lands mid-deletion survives it and
# leaves the user with data they asked to have destroyed.
#
# Listed explicitly rather than derived from the audit category, for two reasons:
# reads stay available during a wipe (they return progressively less, which is
# honest), and a future operation gets no write barrier by accident. A new write
# path must be added here — the test below enumerates the facade's `_run` call
# sites and fails when one is missing, so "someone forgets" is a test failure
# rather than a silent hole.
_WRITE_OPERATIONS = frozenset(
    {
        "store_memory",
        "store_cache",
        "store_decision",
        "log_activity",
        "delete_memory",
        "cache_invalidate",
    }
)


def _retention_outcome(result) -> tuple[str, dict] | None:
    """Read ``set_retention``'s reported status for the audit log.

    ``EpisodicService.set_retention`` never raises — retention management must not
    be able to fail a request — so it reports a failure as
    ``{"status": "error", "error": ...}``. ``_run`` had no way to see that: the
    coroutine returned, so the entry read ``success``, with the caller's requested
    ``ttl_seconds`` recorded beside it as though the index now carried it.

    That is the worst possible record of this particular operation, in both
    directions. Lengthening retention that silently failed leaves data expiring on
    the old schedule while the log says otherwise. *Shortening* it is destructive
    and equally invisible from the response — Atlas deletes on the TTL monitor's
    own schedule, so a caller sees only ``{"scope": "collection"}`` either way, and
    the audit log was the one place the difference could have shown up. It said
    success too.

    Returns ``None`` for every non-failure status (``updated``, ``created``,
    ``removed``), which keeps the default. A missing or unrecognised ``status`` is
    also left alone: this reads a contract, and inventing a failure from a shape it
    does not recognise would make a service change look like an outage.
    """
    if isinstance(result, dict) and result.get("status") == "error":
        # The service already scrubbed this string (`redact_error`), so it is not
        # re-redacted here — doing so would only re-scan a scrubbed message, and
        # skipping it in the service instead would leave the REST response body
        # unscrubbed.
        return "error", {"error": result.get("error", "retention change failed")}
    return None


class AsyncMemory:
    """Programmatic async memory core. Build via ``await AsyncMemory.create(cfg)``."""

    #: Users with a permanent erasure in flight — writes for them are refused
    #: while their data is being deleted. `create()` replaces this with a
    #: per-instance set; the class-level empty frozenset is the safe default for
    #: an instance built by other means (tests construct facades directly), where
    #: an absent attribute would turn the barrier into an AttributeError on the
    #: ordinary write path. Immutable so a stray `add` here fails loudly instead
    #: of quietly blocking one user across every instance in the process.
    _erasing: frozenset | set = frozenset()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    @classmethod
    async def create(cls, config: MemoryConfig) -> AsyncMemory:
        """Initialize Atlas, providers, services, and (optionally) workers.

        Equivalent to the former FastMCP ``lifespan`` startup, lifted out and
        made callable by any consumer.

        **A failed startup leaves nothing running.** Everything after the database
        connection is wrapped, because a partially-built facade is never returned
        to anyone and so has no owner to close it. What leaks without that is not
        obvious from the failure: `DatabaseManager` is a reference-counted
        singleton, so an abandoned `create()` leaves the pool's refcount one too
        high and the *next* `close()` — a legitimate one, from a facade that
        started fine — decrements to a non-zero count and never closes the client.
        Worse, in a process that retries startup, the raised `ValueError` for a
        mismatched target now compares against a pool nobody holds. Worker tasks
        started at step 6 keep polling Atlas through a connection the caller
        believes failed to open.
        """
        from agent_memory.core.database import DatabaseManager

        self = cls.__new__(cls)
        self.config = config
        self._workers = []
        # Users with a wipe in flight. Read by `_check_access` to refuse writes
        # for the duration; see `wipe_user_data`. A class attribute backs this so
        # a facade built by a test without going through `create()` still has the
        # barrier rather than an AttributeError — see `_erasing` on the class.
        self._erasing = set()

        # 1. Database. Taken first because everything below needs it, and held
        # under the try/except that follows: from here on a failure owns a
        # reference-counted claim on the shared pool, and possibly running worker
        # tasks, that nothing else will release. See `_abandon_startup`.
        db_manager = await DatabaseManager.initialize(config)
        self._db_manager = db_manager
        try:
            return await self._build(config, db_manager)
        except BaseException:
            # `BaseException`, not `Exception`: a `CancelledError` — the caller's
            # timeout expiring, or the task group around it being torn down — is
            # exactly the case where nobody is left to clean up, and it is the one
            # `Exception` would miss.
            await self._abandon_startup()
            raise

    async def _build(self, config: MemoryConfig, db_manager) -> AsyncMemory:
        """Steps 2–7 of ``create()``, with the database already acquired.

        Split out so ``create()`` can wrap every step that runs *after* the pool
        is claimed in one handler. Inlining the same try/except would have worked
        and read far worse: the interesting part of startup would sit one indent
        deeper than the reason for it.
        """
        from agent_memory.core.collections import EPISODES, EPISODES_COUNTERS
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

        # 2. Stage-1 indexes (blocking)
        db = db_manager.db
        # `config` is what carries the retention durations into the TTL indexes.
        # Called without it, `ensure_indexes` builds them from the defaults —
        # which is how every configured retention value used to be discarded.
        await ensure_indexes(db, config)

        # 3. Providers + embedding-dimension guard (before any vector write)
        self.providers = ProviderManager(config)
        # `expected` is the *resolved* dimension, not `config.embedding_dimension`
        # — on a Voyage deployment the declared value is Titan's inherited 1536
        # while the embedder emits 1024. This used to read the config field and
        # work only because the factory had just overwritten it in place; the
        # resolution is now a value the manager publishes.
        #
        # `config` is passed as well so the guard can consult the model table,
        # which answers when the embedding endpoint does not.
        self._embedding_dimension = self.providers.embedding_spec.dimension
        await self._validate_embedding_dimension(
            self.providers, expected=self._embedding_dimension, config=config
        )
        # Awaited here, before any service exists, because the check is about what
        # is *already* in the database and the answer decides whether to boot at
        # all. Stage 2 (step 7) is where the destructive rebuild would happen, and
        # by default that runs as a background task whose exceptions are logged
        # and dropped — so a refusal raised there would not stop anything.
        await self._refuse_to_strand_existing_vectors(db, config)

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

    async def _refuse_to_strand_existing_vectors(self, db, config) -> None:
        """Fail startup when the embedding dimension changed under stored vectors.

        The dimension guard immediately above this checks the config against the
        *embedder*. This checks it against the *database*, which is a different
        question with a different failure: a config that is internally consistent
        and correct for every future write can still be wrong for everything
        already written.

        Uses the resolved dimension, so a Voyage deployment is compared on the 1024
        its embedder actually emits rather than the 1536 its config declares —
        otherwise this would report a stranding on every correctly-configured
        Voyage startup and refuse to boot.
        """
        if config.allow_embedding_dimension_change:
            return
        from agent_memory.core.migrations import (
            find_stranding_dimension_changes,
            stranding_error,
        )

        findings = await find_stranding_dimension_changes(
            db, self._embedding_dimension
        )
        if findings:
            raise stranding_error(findings)

    async def _provision_search_indexes(self, db, config, ensure=None) -> None:
        """Create Atlas Search indexes — awaited or backgrounded per config.

        ``await_search_indexes=True`` blocks until indexes are queryable (right
        for short-lived library/script callers, which would otherwise exit
        before background creation finishes and see empty search/recall).
        ``False`` (default) schedules a non-blocking background task for
        long-running servers. ``ensure`` is injectable for testing.

        The dimension comes from the resolved embedding spec, which is what makes
        the ``numDimensions`` in the index agree with the vectors that will be
        written into it. Reading ``config.embedding_dimension`` here would build a
        1536-dim index for a 1024-dim Voyage embedder — a mismatch Atlas accepts
        without complaint and then answers every ``$vectorSearch`` with nothing.
        """
        from agent_memory.core.migrations import ensure_search_indexes

        ensure = ensure or ensure_search_indexes
        # `create()` sets `_embedding_dimension` before calling this. The fallback
        # is for a facade built by hand (tests call this method directly), where
        # the declared value is the best available answer and an AttributeError
        # would be a worse one.
        dimension = getattr(
            self, "_embedding_dimension", config.embedding_dimension
        )
        # Forwarded because reconciliation refuses a stranding rebuild on its own,
        # independently of the startup preflight — see `ensure_search_indexes`.
        # Startup has already refused by this point unless the operator opted in,
        # so in practice this says "the operator opted in" rather than granting
        # anything new; the flag has to travel for the two checks to agree.
        allow_change = getattr(config, "allow_embedding_dimension_change", False)
        self._search_index_task = None
        if config.await_search_indexes:
            await ensure(
                db,
                embedding_dimension=dimension,
                allow_dimension_change=allow_change,
            )
        else:
            self._search_index_task = asyncio.create_task(
                _ensure_search_indexes_bg(
                    db, dimension, ensure=ensure,
                    allow_dimension_change=allow_change,
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
            from agent_memory.providers.manager import (
                known_embedding_dimension,
                resolve_embedding,
            )

            known = known_embedding_dimension(config)
            if known is not None:
                if known != expected:
                    # The resolved model name, not `config.embedding_model` — that
                    # field is Titan's default on a Voyage deployment, so the
                    # message used to name a model the operator is not using and
                    # send them to check the wrong setting.
                    model = resolve_embedding(config).model
                    raise ConfigError(
                        f"embedding_dimension mismatch: config declares {expected} "
                        f"but {model!r} emits {known}. Set "
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

    async def _abandon_startup(self) -> None:
        """Release everything a failed ``create()`` acquired. Never raises.

        Called from ``create()``'s handler on the way out, and only there. The
        object it cleans up is half-built by definition — which step failed decides
        which attributes exist — so every read goes through ``getattr`` and
        ``close()``'s ordering is reused rather than reimplemented.

        Never raises, because the exception that brought us here is the one the
        caller needs. A cleanup failure that replaced it would report a symptom of
        the shutdown instead of the reason for it; anything that goes wrong in here
        is logged and the original propagates.

        Cancels before closing the pool, unlike ``close()``, and does not drain:
        a worker still polling while its client shuts down logs a connection error
        that reads like the startup fault but is not, and there is nothing worth
        draining — no caller ever held this facade, so no queued turn came from one.
        """
        for task in getattr(self, "_workers", []):
            task.cancel()
        self._workers = []
        search_task = getattr(self, "_search_index_task", None)
        if search_task is not None and not search_task.done():
            search_task.cancel()

        episodic = getattr(self, "episodic_service", None)
        if episodic is not None:
            # The consumer task is already cancelled above, so this only flips the
            # worker's `_closed` flag and refuses further writes — it is not a
            # drain. Called anyway so a facade that somehow escapes into a
            # `log_activity` cannot enqueue into a queue with no consumer.
            try:
                episodic.worker.stop()
            except Exception:
                logger.debug("Episodic stop failed while abandoning startup.",
                             exc_info=True)

        db_manager = getattr(self, "_db_manager", None)
        if db_manager is not None:
            try:
                await db_manager.close()
            except Exception:
                logger.warning(
                    "Could not release the database pool after a failed startup. "
                    "The pool's reference count is now too high, so a later "
                    "close() will not actually close the client.",
                    exc_info=True,
                )
            self._db_manager = None
        logger.debug("Released partially-initialized state after a failed startup.")

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
        """Erasure barrier, THEN governance, THEN rate limit.

        Raises ``ErasureInProgressError`` / ``AccessError`` / ``RateLimitError``.

        The erasure check is first and cheapest: a set membership test against
        the users currently being wiped. It runs before governance because a
        write that is about to be refused should not also consume the caller's
        rate-limit budget, and because this is the one refusal that is about the
        state of the data rather than the identity of the caller.
        """
        if operation in _WRITE_OPERATIONS and user_id in self._erasing:
            raise ErasureInProgressError(
                f"'{operation}' refused: user data is being permanently erased. "
                "Retry once the erasure completes."
            )

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
                   outcome=None, **audit_fields):
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

        ``outcome`` is for the services that report failure in their return value
        instead of raising. Without it, "the coroutine returned" is taken to mean
        "the operation succeeded", which is not the same statement: a service that
        catches its own exceptions and hands back ``{"status": "error", ...}``
        lands in the audit log as a ``success``, with the caller's *requested*
        parameters recorded as though they had taken effect. It is a callable
        ``result -> (status, extra_audit_fields) | None``; ``None`` means the
        default. Only ``set_activity_retention`` needs it today — see
        :func:`_retention_outcome` — and the hook is deliberately narrow rather
        than a general "inspect every result", because the honest fix for a new
        service is to raise.
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
        status = "success"
        if outcome is not None:
            # The service returned without raising, which is not by itself a
            # success — see the `outcome` note above.
            verdict = outcome(result)
            if verdict is not None:
                status, extra = verdict
                audit_fields = {**audit_fields, **extra}
        await self.audit_service.log(
            user_id, category, operation, status, duration_ms, **audit_fields
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

        That withholding is enforced by the governance service, which is
        **optional and off by default** — so on a default multi-tenant deployment
        the `"admin"` category above bought nothing, and any authenticated caller
        could shorten every other tenant's retention (or set it to keep-forever)
        through a public REST endpoint. Deleting other tenants' turns is the
        destructive direction, and it happens quietly: Atlas expires the documents
        later, so the caller sees only ``{"scope": "collection"}`` and the data
        goes away on the TTL monitor's schedule.

        Hence :meth:`_require_admin_for_global_mutation` below, which does not
        depend on governance being switched on. Every other ``admin``-category
        operation is scoped to one ``user_id``; this is the only one that reaches
        across tenants, which is why the guard lives here rather than in
        ``_check_access``.

        **Read the returned ``status``.** This is the one facade method whose
        failure is a return value rather than an exception, inherited from
        :meth:`EpisodicService.set_retention`: ``updated`` / ``created`` /
        ``removed`` mean the index changed, ``error`` means it did not and carries
        an ``error`` string. The audit entry now says the same thing — it recorded
        every call as a success, including the ones that changed nothing.
        """
        async def _do():
            # Inside the factory, so the refusal runs within `_run` and is audited
            # as `denied` like every other access failure. Raising before `_run`
            # would leave the one cross-tenant attempt worth seeing unlogged.
            self._require_admin_for_global_mutation("set_activity_retention", role)
            return await self.episodic_service.set_retention(ttl_seconds)

        # `outcome` because the service reports failure rather than raising it, so
        # `_run`'s default reading — returned, therefore succeeded — is wrong for
        # exactly this call. See `_retention_outcome`.
        return await self._run(
            user_id, "set_activity_retention", "admin", _do, role=role,
            outcome=_retention_outcome, ttl_seconds=ttl_seconds,
        )

    def _require_admin_for_global_mutation(self, operation: str, role: str | None) -> None:
        """Refuse a cross-tenant mutation to anyone who is not an admin.

        Deliberately independent of ``governance_service``. Governance is opt-in,
        and an authorisation rule that only exists when an optional subsystem is
        enabled is not an authorisation rule — it is a default-open one. The
        ``admin`` category on the ``_run`` call is still correct and still applies
        when governance *is* on; this is the floor underneath it.

        With auth off there is no role claim and no way to tell callers apart, but
        there is also only one tenant: that is the documented single-tenant
        posture, ``require_auth_for_multi_tenant`` exists to forbid it where it is
        unacceptable, and "collection-wide" means "the only tenant's own data". So
        the guard applies only when auth is on, where a role claim is available and
        `"every tenant"` means something.
        """
        if not getattr(self.config, "auth_enabled", False):
            return
        effective_role = role or self.config.auth_default_role
        if effective_role != "admin":
            raise AccessError(
                f"Operation '{operation}' changes retention for every tenant and "
                f"requires the 'admin' role; this token's role is "
                f"'{effective_role}'"
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
        """Permanently delete every document this user owns. Irreversible.

        Audited against :data:`ERASURE_PRINCIPAL` rather than ``user_id``, which is
        the whole reason this method does not simply call ``_run``.

        ``_run`` audits *after* the service call, and the service call deletes
        every ``audit_log`` row matching this ``user_id``. So the success record
        was written into the collection the wipe had just cleared, under the
        identifier it had just erased — a user who asked to be forgotten was left
        with a row naming them, dated a millisecond after the deletion, and that
        row was the only thing standing between the operation and doing what it
        said. Worse than a leftover: it was created *by* the erasure, so it
        survived every subsequent wipe too, each one recreating it.

        Deleting the record instead is not the fix. A total, irreversible deletion
        is precisely the operation that must leave a trace. So the trace is kept
        and the subject dropped: what ran, when, how long it took, and how many
        documents left each collection, filed against a reserved principal that is
        not a tenant. See :data:`ERASURE_PRINCIPAL`.

        The access check still runs against the real ``user_id`` — authorisation is
        about who is asking, and that decision has to be made before anything is
        deleted. Only the audit subject changes.

        **Pending and concurrent writes.** A deletion is not final if something can
        write the same data back a moment later, and two things could:

        - *Queued episodic turns.* ``log_activity`` returns as soon as the turn is
          on the worker's queue; the insert happens later. Turns queued before the
          wipe would land after it. Drained here, the same way the audit buffer is
          flushed — a pending write is a write.
        - *Concurrent calls.* Nothing stopped an ``add()`` that arrived mid-wipe
          from inserting into a collection the deletion had already swept. The
          user is added to ``_erasing`` for the duration, and ``_check_access``
          refuses writes for them while they are in it. Reads are left alone:
          they return progressively less as collections empty, which is honest.

        Neither is a lock, and this is deliberate: a distributed lock across
        replicas is a much larger change, and the honest statement of what this
        gives is "no write from *this process* survives the erasure". The residue
        check below is what catches the multi-process case — it looks at the
        collections after deleting and reports what is still there rather than
        asserting completeness, so a write from another replica surfaces as a
        ``PartialWipeError`` telling the operator to retry instead of a
        ``complete: true`` that is wrong.
        """
        if not confirm:
            return {
                "error": "wipe_user_data requires confirm=true. "
                "This will permanently delete ALL user data."
            }

        start = time.time()
        # `_check_access` against the real identity, and its refusals audited the
        # ordinary way: a *denied* wipe deleted nothing, so there is no erasure to
        # respect and the attempt is worth attributing. Only a wipe that actually
        # ran needs the subject withheld.
        try:
            await self._check_access(user_id, "wipe_user_data", role=role)
        except AccessError as e:
            status = "throttled" if isinstance(e, RateLimitError) else "denied"
            await self.audit_service.log(
                user_id, "admin", "wipe_user_data", status,
                int((time.time() - start) * 1000), error=redact_error(e),
            )
            raise

        # The barrier goes up before anything is drained or deleted, and comes
        # down only in the `finally` below. Ordering is the whole point: drain
        # first and a turn enqueued during the drain lands after it.
        #
        # `_erasing` is per-instance and replaced by `create()`; guarded here so a
        # facade built without it (tests construct them directly) still runs the
        # erasure rather than failing on the class-level frozenset.
        if not isinstance(self._erasing, set):
            self._erasing = set()
        self._erasing.add(user_id)
        try:
            # Any buffered entry naming this user is flushed *before* the delete,
            # so the delete sees it and removes it. `audit_flush_on_write`
            # defaults to False, so up to `audit_buffer_size` of this user's
            # records normally sit in memory; without this they would be written
            # to Atlas after the wipe had already swept the collection, restoring
            # exactly the rows it removed. An audit buffer is a pending write, and
            # a wipe has to account for it.
            await self.audit_service.flush()
            # Same argument, different queue. `log_activity` returns once the turn
            # is queued, so turns accepted before the barrier went up are still in
            # memory with their inserts pending. Drained rather than discarded:
            # they are then deleted by the wipe below, which is what makes the
            # erasure cover them. A drain that times out is reported, not ignored
            # — see `_drain_episodic_for_erasure`.
            drained = await self._drain_episodic_for_erasure()

            result = await self.admin_service.wipe_user_data(user_id)
            # Verified, not asserted. `admin_service` reports what its own
            # `delete_many` calls removed, which says nothing about a write from
            # another replica that landed in between. This re-reads the
            # collections and raises if anything is left, so `complete: true`
            # means "checked and empty" rather than "no error was raised".
            await self._verify_erased(user_id, result, drained)
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            fields = {"error": redact_error(e)}
            # A partial wipe deleted real data; what survived is the only thing
            # that makes a retry possible, so it goes in the record. The counts
            # are per-collection integers and name no one.
            counts = getattr(e, "counts", None)
            if isinstance(counts, dict):
                fields["deleted"] = {k: v for k, v in counts.items()
                                     if k != "user_id"}
                fields["failed_collections"] = sorted(getattr(e, "errors", {}))
            await self.audit_service.log(
                ERASURE_PRINCIPAL, "admin", "wipe_user_data", "error",
                duration_ms, **fields,
            )
            raise
        finally:
            # Released even when the wipe failed. A user permanently unable to
            # write because an erasure errored once is a worse outcome than the
            # incomplete deletion itself, and the caller has been told to retry.
            self._erasing.discard(user_id)

        duration_ms = int((time.time() - start) * 1000)
        # `result` minus `user_id`: the counts describe the erasure without
        # re-identifying its subject, which is the point of the whole method.
        await self.audit_service.log(
            ERASURE_PRINCIPAL, "admin", "wipe_user_data", "success", duration_ms,
            deleted={k: v for k, v in result.items() if k != "user_id"},
        )
        # Flushed immediately rather than left buffered: this record is the only
        # evidence the operation happened, and the process could exit before the
        # buffer fills or its interval elapses.
        await self.audit_service.flush()
        return result

    async def _drain_episodic_for_erasure(self) -> bool:
        """Land every queued turn before the deletion runs. Returns success.

        Called with the erasure barrier already up, so nothing new joins the queue
        while this waits. The queue is not per-user and cannot be — one worker
        writes turns for everybody — so this drains all of it. That is a short
        wait shared with other tenants' turns, on an operation that runs rarely
        and must be right.

        A timeout is *not* swallowed. Undrained turns are pending writes for
        someone, possibly this user, and they will land after the collections are
        swept. The boolean reaches ``_verify_erased``, which turns it into a
        refusal to claim completeness.
        """
        episodic = getattr(self, "episodic_service", None)
        if episodic is None:
            return True
        try:
            return bool(
                await episodic.flush(self.config.episodic_shutdown_timeout_seconds)
            )
        except Exception as exc:
            # The drain failing is not a reason to abandon the deletion — most of
            # the user's data is in collections the queue never touches. It is a
            # reason not to call the result complete.
            logger.warning("Episodic drain before erasure failed: %s", exc)
            return False

    async def _verify_erased(self, user_id: str, result: dict, drained: bool) -> None:
        """Re-read the collections and refuse to report an incomplete erasure.

        ``admin_service.wipe_user_data`` reports what its own ``delete_many``
        calls removed. That is not the same question as "is this user's data
        gone": a turn that landed from another replica between the delete and now
        is not in those counts, and neither is anything the drain above failed to
        flush. So the claim is checked rather than inferred.

        Raises :class:`PartialWipeError` on residue, which is the same channel a
        failed delete already uses — callers, the audit record, and the demo's
        ``/reset`` all handle it, and the operator instruction is identical: retry.

        A ``count_documents`` failure is deliberately *not* treated as residue.
        The wipe itself succeeded; a follow-up read erroring is a reason to say
        "unverified", not to tell the operator their data is still there. It is
        logged and recorded on the result instead.
        """
        from agent_memory.services.admin import PartialWipeError, erasure_targets

        # `admin_service.db`, not a handle of our own: the check has to read the
        # same database the deletion wrote to, and the facade holds only the
        # manager.
        db = self.admin_service.db
        residue: dict = {}
        unverified: list[str] = []
        for _key, collection, query in erasure_targets(user_id):
            try:
                remaining = await db[collection].count_documents(query)
            except Exception as exc:
                unverified.append(collection)
                logger.warning(
                    "Could not verify erasure of %s: %s", collection,
                    redact_error(exc),
                )
                continue
            if remaining:
                residue[collection] = f"{remaining} document(s) still present"

        if not drained and not residue:
            # The drain timed out but nothing is in the collections *yet* — the
            # pending turns land next. Reported as residue because the caller's
            # only useful action is the same one: retry, which will both drain
            # again and delete whatever arrived.
            residue["episodes"] = (
                "queued turns did not drain before the deletion; they may still "
                "be written"
            )

        if residue:
            raise PartialWipeError(dict(result), residue)
        if unverified:
            result["unverified_collections"] = sorted(unverified)


async def _ensure_search_indexes_bg(
    db,
    embedding_dimension: int = 1536,
    ensure=None,
    allow_dimension_change: bool = False,
) -> None:
    """Background Atlas Search index creation — failures are non-fatal.

    Non-fatal is why ``allow_dimension_change`` has to be threaded down rather
    than checked here: a refusal on this path would be logged and dropped, so the
    decision belongs to reconciliation itself, which can decline the destructive
    step and still complete everything else.
    """
    from agent_memory.core.migrations import ensure_search_indexes

    ensure = ensure or ensure_search_indexes
    try:
        await ensure(
            db,
            embedding_dimension=embedding_dimension,
            allow_dimension_change=allow_dimension_change,
        )
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
