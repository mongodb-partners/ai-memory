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

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_memory.config import MemoryConfig
from agent_memory.exceptions import AccessError, NotFoundError, RateLimitError
from agent_memory.memory import AsyncMemory


def _build_auth_dependency(config: MemoryConfig | None):
    """Return a FastAPI dependency enforcing Bearer auth, reusing auth/.

    When auth is disabled (or no config), the dependency is a no-op so the
    routes stay open — same posture as the MCP shell.
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


def create_app(app, config: MemoryConfig | None = None) -> FastAPI:
    """Build a FastAPI app bound to facade ``app``.

    When ``config`` enables auth, every route except ``/health`` requires a valid
    Bearer token, verified by the existing ``auth/`` verifier (REQ-E-072).
    """
    api = FastAPI(title="agent-memory", version="4.0.0")
    auth = Depends(_build_auth_dependency(config))
    protected = [auth]

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
    @api.post("/memories", dependencies=protected)
    async def add_memory(body: AddRequest):
        return await app.add(body.user_id, body.conversation_id, body.messages)

    @api.get("/memories/recall", dependencies=protected)
    async def recall(user_id: str, query: str, limit: int = 10):
        return await app.recall(user_id, query, limit=limit)

    @api.get("/memories/search", dependencies=protected)
    async def search(user_id: str, query: str, limit: int = 10):
        return await app.search(user_id, query, limit=limit)

    @api.delete("/memories", dependencies=protected)
    async def delete(user_id: str, memory_id: str | None = None, confirm: bool = False,
                     dry_run: bool = False):
        return await app.delete(user_id, memory_id=memory_id, confirm=confirm, dry_run=dry_run)

    @api.post("/decisions", dependencies=protected)
    async def remember_decision(body: DecisionRequest):
        return await app.remember_decision(body.user_id, body.key, body.value, ttl_days=body.ttl_days)

    @api.get("/decisions", dependencies=protected)
    async def recall_decision(user_id: str, key: str):
        return await app.recall_decision(user_id, key)

    # ── Episodic memory (the agent activity log) ──────────────────────────
    @api.post("/activity", dependencies=protected)
    async def log_activity(body: ActivityRequest):
        return await app.log_activity(
            body.user_id, body.thread_id, body.messages, todos=body.todos,
            agent_name=body.agent_name, correlation_id=body.correlation_id,
            conversation_id=body.conversation_id,
        )

    @api.get("/activity/search", dependencies=protected)
    async def search_activity(user_id: str, query: str, thread_id: str | None = None,
                              agent_name: str | None = None, limit: int = 5):
        return await app.recall_activity(user_id, query, thread_id=thread_id,
                                         agent_name=agent_name, limit=limit)

    @api.get("/activity/thread/{thread_id}", dependencies=protected)
    async def get_thread(thread_id: str, user_id: str, limit: int | None = None,
                         ascending: bool = True):
        return await app.get_thread(user_id, thread_id, limit=limit, ascending=ascending)

    @api.get("/activity/correlation/{correlation_id}", dependencies=protected)
    async def get_correlation(correlation_id: str, user_id: str, limit: int | None = None):
        return await app.get_activity_by_correlation(user_id, correlation_id, limit=limit)

    @api.put("/activity/retention", dependencies=protected)
    async def set_activity_retention(body: RetentionRequest):
        return await app.set_activity_retention(body.user_id, ttl_seconds=body.ttl_seconds)

    @api.get("/health")
    async def health():
        """Liveness plus the episodic writer's counters.

        The queue depth and failure counts are what an operator actually needs
        to see: a 200 with a full queue and rising write failures is not health.
        """
        body = {"status": "ok"}
        try:
            body["episodic"] = app.activity_stats()
        except Exception:  # pragma: no cover - a probe must never 500
            pass
        return body

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
