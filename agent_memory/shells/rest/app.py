"""FastAPI REST shell over the AsyncMemory facade.

Thin transport: Pydantic request bodies, route handlers that call the facade,
and exception handlers that translate typed errors to HTTP status codes. No
business logic. REST is the explicit-control surface — there is no auto-capture
(that is MCP-only); callers persist via ``POST /memories``.

Error mapping (REQ-E-071): ``RateLimitError`` → 429 (registered as its own
handler so it wins over its ``AccessError`` base), ``AccessError`` → 403,
``NotFoundError`` → 404.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_memory.auth.identity import Caller, IdentityError, resolve_caller
from agent_memory.config import MemoryConfig
from agent_memory.exceptions import AccessError, NotFoundError, RateLimitError
from agent_memory.memory import AsyncMemory
from agent_memory.version import __version__


def _build_auth_dependency(config: MemoryConfig | None):
    """Return a FastAPI dependency enforcing Bearer auth, reusing auth/.

    When auth is disabled (or no config), the dependency is a no-op so the
    routes stay open — same posture as the MCP shell. ``MemoryConfig`` refuses to
    construct with ``auth_enabled`` and no secret, so "enabled but silently open"
    is no longer reachable from here.
    """
    if config is None or not config.auth_enabled or not config.auth_secret:
        async def _noop():
            return None

        return _noop

    from agent_memory.auth.api_keys import APIKeyManager
    from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier

    verifier = MemoryMCPTokenVerifier(
        secret=config.auth_secret, api_key_manager=APIKeyManager()
    )

    async def _require_token(authorization: str | None = Header(default=None)):
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        access = await verifier.verify_token(token)
        if access is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return access

    return _require_token


def _build_caller_dependency(config: MemoryConfig | None):
    """Return the dependency every handler uses to learn who is calling.

    This replaces ``dependencies=protected``. That form is side-effect-only: it
    runs the verifier and throws the result away, so the token authenticated the
    request and then had no bearing on it. Every handler read ``user_id`` from the
    query string or the JSON body, which means any valid token could read, edit,
    or wipe any other tenant's memory by naming them.

    Injecting a ``Caller`` instead makes the identity an argument the handler
    cannot avoid receiving, and ``resolve_caller`` — shared with the MCP shell —
    is the only thing that decides it.
    """
    token_dep = _build_auth_dependency(config)

    async def _caller(
        user_id: str | None = None,
        access=Depends(token_dep),
    ) -> _PendingCaller:
        """Verify the token and hold onto the request's ``?user_id=``.

        Deliberately does *not* finish resolving. Identity resolution needs the
        ``user_id`` the request named, and for POST/PUT routes that value is in the
        JSON body — which FastAPI cannot inject into a dependency. So the
        dependency verifies the token (the part that must happen before the
        handler runs) and :func:`_identify` completes the resolution once the body
        is available.
        """
        return _PendingCaller(access=access, query_user_id=user_id, config=config)

    return _caller


@dataclass(frozen=True)
class _PendingCaller:
    """A verified token plus the request's claimed ``user_id``, not yet resolved."""

    access: object | None
    query_user_id: str | None
    config: MemoryConfig | None


def _identify(pending: _PendingCaller, body_user_id: str | None = None) -> Caller:
    """Resolve the caller. **Every handler must call this.**

    One funnel, so the rule is stated once: with auth on the token decides and a
    request naming anyone else is refused; with auth off the request's own
    ``user_id`` is all there is. Handlers with a body pass ``body.user_id``;
    handlers without pass nothing and the query parameter is used.

    A handler that reads ``body.user_id`` or the raw query parameter instead of
    calling this is the vulnerability this function exists to remove, so the
    ``Caller`` it returns is the only thing the routes are allowed to read a
    ``user_id`` from.
    """
    claimed = body_user_id or pending.query_user_id
    try:
        return resolve_caller(pending.access, claimed, pending.config)
    except IdentityError as exc:
        # 403 rather than 401: the token is valid, the identity it asked for is
        # not one it may act as, and retrying the same token will not change that.
        # The one exception is auth-off with no user_id at all, which is a
        # malformed request — 400.
        status = 400 if pending.access is None else 403
        raise HTTPException(status_code=status, detail=str(exc)) from exc


class AddRequest(BaseModel):
    user_id: str
    conversation_id: str
    messages: list[dict]


class DecisionRequest(BaseModel):
    user_id: str
    key: str
    value: str
    ttl_days: int | None = None


class ActivityRequest(BaseModel):
    user_id: str
    thread_id: str
    messages: list[dict]
    todos: list[dict] | None = None
    agent_name: str | None = None
    correlation_id: str | None = None
    conversation_id: str | None = None


class RetentionRequest(BaseModel):
    user_id: str
    # None is meaningful, not missing: it drops the TTL and keeps the log forever.
    ttl_seconds: int | None = None


def build_health_body(app) -> dict:
    """Assemble the /health payload for facade ``app``. Never raises.

    Shared with the MCP shell, which used to serve a bare `{"status": "ok"}`. In a
    dual-transport deployment the two shells hold the *same* facade, so an operator
    watching the MCP port and one watching the REST port got different answers
    about one process — whichever port the monitor happened to target decided
    whether a dead worker was visible at all. One function, one definition of
    health, and no drift.

    The queue depth and failure counts are what an operator actually needs to see:
    a 200 with a full queue and rising write failures is not health.

    `workers` is here for the same reason. A crashed enrichment or consolidation
    loop leaves reads and writes working perfectly — only the reactive half of the
    system stops — so without this a probe reports a healthy service whose memories
    are never enriched, promoted, or forgotten. `status` degrades to `"degraded"`
    when a worker that should be running is not, because a probe that only ever
    says `ok` is not a probe.

    **This body is served unauthenticated.** `/health` is the one route exempt from
    auth, deliberately: a probe that needs a token fails during exactly the
    incident it exists to detect. So everything here is a counter, a boolean, or a
    name — never a document, a user id, or a raw exception. `worker_status` runs its
    error strings through `redact_error` for that reason: a crashed worker's
    exception is usually a driver error, and driver errors quote the connection
    string they failed on.
    """
    body: dict = {"status": "ok"}
    try:
        body["episodic"] = app.activity_stats()
    except Exception:  # pragma: no cover - a probe must never 500
        pass
    try:
        workers = app.worker_status()
        body["workers"] = workers
        if workers.get("enabled") and not workers.get("running"):
            body["status"] = "degraded"
    except Exception:  # pragma: no cover - a probe must never 500
        pass
    return body


def create_app(app, config: MemoryConfig | None = None) -> FastAPI:
    """Build a FastAPI app bound to facade ``app``.

    When ``config`` enables auth, every route except ``/health`` requires a valid
    Bearer token, verified by the existing ``auth/`` verifier (REQ-E-072).
    """
    # Version from the installed package, not a literal. This one is served in the
    # OpenAPI document, so a hardcoded copy goes stale exactly where a client reads
    # it to decide what the API supports — the literal here still said 4.0.0 at
    # 4.1.0. `app_version` exists to be the single source; use it.
    api = FastAPI(title="agent-memory", version=__version__)
    # One dependency, injected by value into every handler. `Caller` carries the
    # user_id and the role, so a handler physically cannot serve a tenant the
    # token does not name, and the governance role reaches `_check_access`.
    caller_dep = Depends(_build_caller_dependency(config))

    # ── Exception handlers (most specific first) ──────────────────────────
    @api.exception_handler(RateLimitError)
    async def _rate_limited(_req: Request, exc: RateLimitError):
        return JSONResponse(status_code=429, content={"error": str(exc)})

    @api.exception_handler(AccessError)
    async def _denied(_req: Request, exc: AccessError):
        return JSONResponse(status_code=403, content={"error": str(exc)})

    @api.exception_handler(NotFoundError)
    async def _not_found(_req: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"error": str(exc)})

    # ── Routes ────────────────────────────────────────────────────────────
    #
    # Every handler resolves through `_identify` and then reads `who.user_id` /
    # `who.role`. A route that uses `body.user_id` or the raw `user_id` query
    # parameter directly is the bug this shape exists to prevent.
    @api.post("/memories")
    async def add_memory(body: AddRequest, pending=caller_dep):
        who = _identify(pending, body.user_id)
        return await app.add(who.user_id, body.conversation_id, body.messages,
                             role=who.role)

    @api.get("/memories/recall")
    async def recall(query: str, limit: int = 10, pending=caller_dep):
        who = _identify(pending)
        return await app.recall(who.user_id, query, limit=limit, role=who.role)

    @api.get("/memories/search")
    async def search(query: str, limit: int = 10, pending=caller_dep):
        who = _identify(pending)
        return await app.search(who.user_id, query, limit=limit, role=who.role)

    @api.delete("/memories")
    async def delete(memory_id: str | None = None, confirm: bool = False,
                     dry_run: bool = False, pending=caller_dep):
        who = _identify(pending)
        return await app.delete(who.user_id, memory_id=memory_id, confirm=confirm,
                                dry_run=dry_run, role=who.role)

    @api.post("/decisions")
    async def remember_decision(body: DecisionRequest, pending=caller_dep):
        who = _identify(pending, body.user_id)
        return await app.remember_decision(who.user_id, body.key, body.value,
                                           ttl_days=body.ttl_days, role=who.role)

    @api.get("/decisions")
    async def recall_decision(key: str, pending=caller_dep):
        who = _identify(pending)
        return await app.recall_decision(who.user_id, key, role=who.role)

    # ── Episodic memory (the agent activity log) ──────────────────────────
    @api.post("/activity")
    async def log_activity(body: ActivityRequest, pending=caller_dep):
        who = _identify(pending, body.user_id)
        return await app.log_activity(
            who.user_id, body.thread_id, body.messages, todos=body.todos,
            agent_name=body.agent_name, correlation_id=body.correlation_id,
            conversation_id=body.conversation_id, role=who.role,
        )

    @api.get("/activity/search")
    async def search_activity(query: str, thread_id: str | None = None,
                              agent_name: str | None = None, limit: int = 5,
                              pending=caller_dep):
        who = _identify(pending)
        return await app.recall_activity(who.user_id, query, thread_id=thread_id,
                                         agent_name=agent_name, limit=limit,
                                         role=who.role)

    @api.get("/activity/thread/{thread_id}")
    async def get_thread(thread_id: str, limit: int | None = None,
                         ascending: bool = True, pending=caller_dep):
        who = _identify(pending)
        return await app.get_thread(who.user_id, thread_id, limit=limit,
                                    ascending=ascending, role=who.role)

    @api.get("/activity/correlation/{correlation_id}")
    async def get_correlation(correlation_id: str, limit: int | None = None,
                              pending=caller_dep):
        who = _identify(pending)
        return await app.get_activity_by_correlation(who.user_id, correlation_id,
                                                     limit=limit, role=who.role)

    @api.put("/activity/retention")
    async def set_activity_retention(body: RetentionRequest, pending=caller_dep):
        who = _identify(pending, body.user_id)
        return await app.set_activity_retention(who.user_id,
                                                ttl_seconds=body.ttl_seconds,
                                                role=who.role)

    @api.get("/health")
    async def health():
        """Liveness, the episodic writer's counters, and worker status."""
        return build_health_body(app)

    return api


def create_managed_app(config: MemoryConfig, app: AsyncMemory | None = None) -> FastAPI:
    """REST app that creates and closes its own ``AsyncMemory`` via lifespan.

    If ``app`` is provided it is reused and its lifecycle is owned by the caller
    (dual-transport). The mounted routes are bound lazily to the live facade.
    """
    holder: dict = {"app": app}

    @asynccontextmanager
    async def lifespan(_api: FastAPI):
        owns = app is None
        instance = app or await AsyncMemory.create(config)
        holder["app"] = instance
        try:
            yield
        finally:
            if owns:
                await instance.close()

    # Build the routed app against a proxy that resolves to the live facade.
    class _Proxy:
        def __getattr__(self, name):
            return getattr(holder["app"], name)

    api = create_app(_Proxy(), config=config)
    api.router.lifespan_context = lifespan
    return api
