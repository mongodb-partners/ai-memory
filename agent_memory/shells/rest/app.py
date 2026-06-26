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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_memory.config import MemoryConfig
from agent_memory.exceptions import AccessError, NotFoundError, RateLimitError
from agent_memory.memory import AsyncMemory


class AddRequest(BaseModel):
    user_id: str
    conversation_id: str
    messages: list[dict]


class DecisionRequest(BaseModel):
    user_id: str
    key: str
    value: str
    ttl_days: int | None = None


def create_app(app) -> FastAPI:
    """Build a FastAPI app bound to facade ``app``."""
    api = FastAPI(title="agent-memory", version="4.0.0")

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
    @api.post("/memories")
    async def add_memory(body: AddRequest):
        return await app.add(body.user_id, body.conversation_id, body.messages)

    @api.get("/memories/recall")
    async def recall(user_id: str, query: str, limit: int = 10):
        return await app.recall(user_id, query, limit=limit)

    @api.get("/memories/search")
    async def search(user_id: str, query: str, limit: int = 10):
        return await app.search(user_id, query, limit=limit)

    @api.delete("/memories")
    async def delete(user_id: str, memory_id: str | None = None, confirm: bool = False,
                     dry_run: bool = False):
        return await app.delete(user_id, memory_id=memory_id, confirm=confirm, dry_run=dry_run)

    @api.post("/decisions")
    async def remember_decision(body: DecisionRequest):
        return await app.remember_decision(body.user_id, body.key, body.value, ttl_days=body.ttl_days)

    @api.get("/decisions")
    async def recall_decision(user_id: str, key: str):
        return await app.recall_decision(user_id, key)

    @api.get("/health")
    async def health():
        return {"status": "ok"}

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

    api = create_app(_Proxy())
    api.router.lifespan_context = lifespan
    return api
