"""Demo server for the memory UI — FastAPI + SSE, no agent framework.

The point of this file is what is absent from it. There is no orchestration
library, no graph, no callback plumbing: ``agent-memory`` already abstracts the
LLM behind ``LLMProvider`` and the memory behind ``AsyncMemory``, so the loop in
``turn.py`` is the entire agent. That is the talk's thesis stated as code — memory
is a data-layer concern, and once it is solved there, the agent is small.

Routes:

* ``POST /chat``        — SSE stream of one turn
* ``GET  /memories``    — browse what the agent knows, per tier, without a turn
* ``POST /reset``       — wipe a demo user's data (rehearsal between passes)
* ``GET  /health``      — readiness, including the episodic writer's counters
* ``GET  /config``      — the model and provider actually in use, for the UI header
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_memory.config import MemoryConfig
from agent_memory.core.correlation import derive_correlation_id
from agent_memory.memory import AsyncMemory
from agent_memory.services.admin import PartialWipeError

from .cache_key import DemoResponseCache
from .history import ConversationHistory
from .sse import SHUTDOWN, drain_in_flight, reset_shutdown_state, sse_response
from .turn import TurnRunner, project_episode_hit, project_memory_hit

log = logging.getLogger(__name__)

# The library reads config from the real environment and never loads a .env of
# its own — correct for a library, but it means `uvicorn server.app:app` in a
# fresh shell fails on a missing connection string. Loading the repo root's .env
# here keeps the documented one-line run command working. `override=False` so an
# explicitly exported value still wins, which is how the booth machine can point
# at a different cluster without editing the file.
# Repository root, three parents up. Inside the demo container this path does not
# exist; `load_dotenv` on a missing file is a silent no-op and compose supplies
# the same values through `env_file`, so both deployments read one config source.
# `override=False` so a real environment variable always wins over the file.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

# Per-turn ceiling. Generous enough for a cold Bedrock call on booth wifi, short
# enough that a hung turn surfaces as an error frame while the presenter is still
# mid-sentence rather than after the audience has left.
TURN_TIMEOUT = float(os.environ.get("DEMO_TURN_TIMEOUT", "45"))


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    # The toggle. Defaults to True so a client that omits it gets the interesting
    # behaviour rather than silently demoing the broken case.
    memory_enabled: bool = True


class ResetRequest(BaseModel):
    user_id: str = Field(min_length=1)
    confirm: bool = False


def create_app() -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # `SHUTDOWN` is a module global, so a previous lifespan's shutdown is still
        # visible here — and a set flag makes every chat request in this lifecycle
        # answer with a `shutdown` error frame while the server otherwise reports
        # healthy. Cleared first, before anything can serve. See
        # `reset_shutdown_state`.
        reset_shutdown_state()
        config = MemoryConfig.from_env(
            # Short-lived clients would otherwise race index creation, but this
            # is a long-running server, so the background path is right — the
            # first turn may miss until the indexes are queryable, which is
            # exactly why the pre-flight checklist sends a warm-up turn.
            await_search_indexes=False,
        )
        memory = await AsyncMemory.create(config)
        cache = DemoResponseCache(
            memory._db_manager.db, memory.providers.embedding
        )
        await cache.ensure_indexes()
        history = ConversationHistory()

        state["memory"] = memory
        state["cache"] = cache
        state["history"] = history
        state["runner"] = TurnRunner(
            memory,
            provider_name=config.llm_provider,
            cache=cache,
            history=history,
        )
        state["config"] = config
        # Resolved embedding, not the declared config fields — same reason as in
        # `/config` below: the declared pair is Titan's default on a Voyage
        # deployment, and a startup line naming the wrong embedder is the first
        # thing anyone greps when retrieval looks wrong.
        spec = memory.providers.embedding_spec
        log.info(
            "demo server ready (llm=%s/%s embeddings=%s/%s dims=%s)",
            config.llm_provider, config.llm_model,
            config.embedding_provider, spec.model,
            spec.dimension,
        )
        try:
            yield
        finally:
            # Order matters: stop accepting, let live streams finish, flush the
            # episodic queue, then close the client. Closing first would strand
            # queued turns and log a wall of connection errors on Ctrl-C.
            SHUTDOWN.set()
            await drain_in_flight()
            await memory.close()

    api = FastAPI(title="agent-memory demo", lifespan=lifespan)

    # The Vite dev server is a different origin. Locked to localhost rather than
    # "*" — this server holds Atlas credentials and a wide-open CORS policy on a
    # conference network is not a thing to demo.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def _require(name: str):
        value = state.get(name)
        if value is None:
            raise HTTPException(status_code=503, detail="server still starting")
        return value

    @api.post("/chat")
    async def chat(body: ChatRequest, request: Request):
        runner = _require("runner")
        # Reuse an inbound trace id when there is one, so an episode joins the
        # same trace as the HTTP request that produced it.
        correlation_id = derive_correlation_id(
            {
                "traceparent": request.headers.get("traceparent"),
                "x_request_id": request.headers.get("x-request-id"),
            }
        )

        def drive():
            return runner.run(
                user_id=body.user_id,
                thread_id=body.thread_id,
                message=body.message,
                memory_enabled=body.memory_enabled,
                correlation_id=correlation_id,
            )

        return sse_response(drive, correlation_id, TURN_TIMEOUT)

    @api.get("/memories")
    async def memories(user_id: str, query: str = "", limit: int = 20):
        """Browse what the agent knows, grouped by tier.

        ``query`` empty lists the most recent documents; non-empty runs the same
        hybrid search a turn would. Both are useful on stage: the list proves the
        documents exist, the search proves ranking is real.
        """
        memory = _require("memory")
        db = memory._db_manager.db

        if query:
            semantic = (await memory.search(user_id, query, limit=limit)).get(
                "results", []
            )
            episodic = (
                await memory.recall_activity(user_id, query, limit=limit)
            ).get("results", [])
        else:
            cursor = (
                db["memories"]
                .find(
                    {"user_id": user_id, "deleted_at": None},
                    {"embedding": 0},
                )
                .sort([("created_at", -1)])
                .limit(limit)
            )
            semantic = await cursor.to_list(None)
            cursor = (
                db["episodes"]
                .find({"user_id": user_id}, {"embedding": 0})
                .sort([("ts", -1)])
                .limit(limit)
            )
            episodic = await cursor.to_list(None)

        groups: dict[str, list[dict]] = {"stm": [], "ltm": [], "episodic": []}
        for index, doc in enumerate(semantic):
            doc.pop("_id", None)
            hit = project_memory_hit(doc, index)
            groups["stm" if doc.get("tier") == "stm" else "ltm"].append(hit)
        for index, doc in enumerate(episodic):
            doc.pop("_id", None)
            groups["episodic"].append(project_episode_hit(doc, index))

        return {"user_id": user_id, "query": query, "groups": groups}

    @api.post("/reset")
    async def reset(body: ResetRequest):
        """Wipe a demo user so a rehearsal starts from the same state twice.

        Gated on ``confirm`` because this is destructive and a stray click
        between the two booth mornings would cost the seeded data.
        """
        if not body.confirm:
            raise HTTPException(status_code=400, detail="confirm=true required")
        memory = _require("memory")
        cache = _require("cache")
        history = _require("history")

        # `wipe_user_data` covers every user-scoped collection the library owns,
        # `episodes` among them — this route used to delete `episodes` itself and
        # then overwrite the library's real count with its own zero, so a reset that
        # cleared nine episodes reported none. The demo's own response cache and the
        # in-process thread history are not the library's, so they are still cleared
        # here.
        try:
            result = await memory.wipe_user_data(body.user_id, confirm=True)
        except PartialWipeError as exc:
            # A 500 would say "the server is broken" when what happened is that
            # some data is still there. 409 with the counts, so a rehearsal knows
            # to retry rather than to start seeding on top of a half-cleared user.
            raise HTTPException(
                status_code=409,
                detail={
                    "error": str(exc),
                    "complete": False,
                    **{k: v for k, v in exc.counts.items() if k != "user_id"},
                    "failed_collections": sorted(exc.errors),
                },
            ) from exc
        cached = await cache.clear(body.user_id)
        history.clear(body.user_id)
        return {**result, "demo_cache_deleted": cached}

    @api.get("/health")
    async def health():
        memory = state.get("memory")
        if memory is None:
            raise HTTPException(status_code=503, detail="starting")
        config = state["config"]
        spec = memory.providers.embedding_spec
        return {
            "status": "ok",
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "embedding_model": spec.model,
            "embedding_dimension": spec.dimension,
            # A 200 with a full queue and rising write failures is not health.
            "episodic": memory.activity_stats(),
            "threads_in_memory": state["history"].thread_count(),
        }

    @api.get("/config")
    async def config_route():
        """What the UI header displays. Read from the live config, never
        hardcoded — a slide claiming one model while the server runs another is
        the kind of error an audience catches.

        The embedding pair comes from ``providers.embedding_spec``, not from
        ``config.embedding_model``/``embedding_dimension``. Those two are the
        *declared* values and keep Titan's defaults on a Voyage deployment, so
        this route used to put `amazon.titan-embed-text-v1 (1536d)` in the header
        while every vector on screen came from voyage-4 at 1024 — the exact
        mislabelling the docstring above warns about.
        """
        config = _require("config")
        spec = _require("memory").providers.embedding_spec
        return {
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "embedding_provider": config.embedding_provider,
            "embedding_model": spec.model,
            "embedding_dimension": spec.dimension,
            "database": config.mongodb_database_name,
        }

    return api


app = create_app()
