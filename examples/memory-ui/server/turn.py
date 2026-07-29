"""One agent turn: recall → prompt → stream → write back.

This is the whole "agent" in this demo, and it is deliberately small with no
framework in it. That is the talk's argument made executable: once memory is a
database concern, the agent loop is short enough to read in one sitting.

The loop yields SSE frames as it goes, so the UI shows *when* each memory
operation happens relative to the tokens — recall before the first token, the
write after the last. A panel that only showed final state would lose that.

Frame sequence for one turn::

    memory{phase:cache}                    (skipped entirely when memory is off)
    memory{phase:recall, tier:ltm|stm|episodic}
    token*                                 (or none, on a cache hit)
    memory{phase:write, tier:ltm}
    memory{phase:write, tier:episodic}

``memory_enabled=False`` skips recall *and* the cache. Both halves matter: recall
is the visible difference, but a cache that ignored the flag would replay a
memory-informed answer during the memory-off pass, and the demo's central claim
would be false on stage. See ``cache_key.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .prompt import build_call, build_system_text, format_context

log = logging.getLogger(__name__)

# Recall depth per tier. Small on purpose: a booth screen shows four or five hits
# legibly, and a longer context makes it harder to argue the answer came from the
# memory you can see rather than from bulk.
RECALL_LIMIT = 4
EPISODIC_LIMIT = 3

# Wall-clock ceiling per recall. A slow recall on booth wifi degrades to "no
# context" rather than holding the first token — an audience reads a spinner as a
# broken demo regardless of the cause.
RECALL_TIMEOUT = 6.0

# Turns of this thread's own history replayed into the prompt. Chat history is
# not memory, and conflating the two is the misconception the talk exists to
# correct: two turns is enough for pronouns to resolve, and short enough that
# nobody can claim the recalled facts came from the transcript.
HISTORY_TURNS = 2


def frame(event: str, data: Any) -> dict[str, Any]:
    """Build an SSE frame. ``token`` carries raw text; everything else is JSON."""
    if event == "token":
        return {"event": "token", "data": data}
    return {"event": event, "data": json.dumps(data, default=str)}


def _first_text(doc: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def rank_label(index: int, score: float | None) -> str:
    """A rank the audience can read, since the raw RRF score is unreadable.

    ``$rankFusion`` returns a reciprocal-rank sum: with the default ``k`` of 60,
    a document ranked first in one branch scores about ``1/61`` — so a *perfect*
    top hit displays as **0.016**. On a booth screen that reads as a failed
    match, and it takes a paragraph of explanation to undo.

    So the panel leads with ``#1``, ``#2`` and keeps the raw value alongside it
    for anyone who asks. The score is still reported verbatim in the frame — this
    is a label, not a replacement, and inventing a rescaled "relevance
    percentage" would be presenting a made-up number as a measurement.
    """
    if score is None:
        return f"#{index + 1}"
    return f"#{index + 1} · rrf {score:.4f}"


def project_memory_hit(doc: dict, index: int = 0) -> dict:
    """Project a ``memories`` document to the panel's hit shape.

    Keeps ``score`` and ``importance``. The retail UI this borrows from joined
    recalled memories into one string and discarded the scores — precisely the
    evidence a memory demo needs on screen.
    """
    # `final_score` is the calibrated blend (recency + importance + relevance)
    # that ranking applies on top of retrieval; `score` is the raw fused rank.
    # Prefer the calibrated one when present — it is what actually ordered the
    # list — but `or` would swallow a legitimate 0.0, so test for None.
    score = doc.get("final_score")
    if score is None:
        score = doc.get("score")
    return {
        "text": _first_text(doc, ("summary", "content")),
        "score": score,
        "rank": rank_label(index, score),
        "importance": doc.get("importance"),
        "access_count": doc.get("access_count"),
        "ts": doc.get("created_at"),
        "tier": doc.get("tier"),
    }


def project_episode_hit(doc: dict, index: int = 0) -> dict:
    """Project an ``episodes`` document. Tool names and files are this tier's point.

    ``search_text`` (the embedded question-plus-answer) reads better on screen
    than the raw message array.
    """
    tools: list[str] = []
    for message in doc.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("name"):
                tools.append(str(call["name"]))
    return {
        "text": _first_text(doc, ("search_text",)),
        "score": doc.get("score"),
        "rank": rank_label(index, doc.get("score")),
        "ts": doc.get("ts"),
        "step": doc.get("step"),
        "tools": tools,
        "files": [
            f.get("path")
            for f in (doc.get("files_touched") or [])
            if isinstance(f, dict) and f.get("path")
        ],
    }


class TurnRunner:
    """Runs turns against one ``AsyncMemory`` instance and one LLM provider.

    Holds no per-request state — a single instance serves every connection.
    """

    def __init__(self, memory, *, provider_name: str, cache, history) -> None:
        self._memory = memory
        self._provider_name = provider_name
        self._cache = cache
        self._history = history

    # ── Recall ───────────────────────────────────────────────────────────

    async def _recall_all(self, user_id: str, query: str) -> dict[str, list[dict]]:
        """Recall from every tier concurrently, tolerating individual failures.

        The tiers are independent, so they run in parallel and a failure in one
        costs only that tier. ``return_exceptions=True`` matters here: without it
        one slow episodic query would sink the semantic recall that was already
        finished, and the demo would show an empty panel for a reason the
        audience cannot see.
        """

        async def _semantic() -> list[dict]:
            result = await self._memory.search(user_id, query, limit=RECALL_LIMIT)
            return result.get("results", [])

        async def _episodic() -> list[dict]:
            result = await self._memory.recall_activity(
                user_id, query, limit=EPISODIC_LIMIT
            )
            return result.get("results", [])

        async def _guarded(name: str, coro):
            try:
                return await asyncio.wait_for(coro, timeout=RECALL_TIMEOUT)
            except TimeoutError:
                log.warning("recall tier %s timed out after %ss", name, RECALL_TIMEOUT)
                return []
            except Exception:
                # A missing Atlas Search index surfaces here. Degrading to "no
                # context" is right: the turn still answers, just without memory.
                log.exception("recall tier %s failed", name)
                return []

        semantic, episodic = await asyncio.gather(
            _guarded("semantic", _semantic()),
            _guarded("episodic", _episodic()),
        )

        # One hybrid query covers both semantic tiers; split by the document's own
        # tier field so the panel can show which tier each hit came from.
        groups: dict[str, list[dict]] = {"stm": [], "ltm": [], "episodic": []}
        for index, doc in enumerate(semantic):
            # The rank is the document's position in the *combined* result set,
            # not within its tier — that is the order retrieval actually
            # returned, and renumbering per tier would misreport it.
            hit = project_memory_hit(doc, index)
            groups["stm" if doc.get("tier") == "stm" else "ltm"].append(hit)
        groups["episodic"] = [
            project_episode_hit(doc, index) for index, doc in enumerate(episodic)
        ]
        return groups

    # ── The turn ─────────────────────────────────────────────────────────

    async def run(
        self,
        *,
        user_id: str,
        thread_id: str,
        message: str,
        memory_enabled: bool,
        correlation_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE frames for one turn. Never raises for expected failures."""
        started = time.monotonic()
        groups: dict[str, list[dict]] = {}
        context_block = ""

        # ── Cache ────────────────────────────────────────────────────────
        if memory_enabled:
            cached = await self._cache.lookup(
                user_id, message, memory_enabled=memory_enabled
            )
            if cached is not None:
                yield frame(
                    "memory",
                    {
                        "phase": "cache",
                        "tier": "cache",
                        "query": message,
                        "cache_hit": True,
                        "score": cached.get("score"),
                        # "exact" or "semantic". Reported because they are
                        # different claims: one skipped an embedding call as well
                        # as the model, the other proves the cache matches meaning
                        # rather than characters.
                        "match": cached.get("match"),
                        "hits": [],
                    },
                )
                # Replayed a token at a time so a cache hit looks like an answer
                # rather than a paste. It is also honest about what the cache
                # saves: the tokens arrive, the inference did not happen.
                for chunk in _chunk_text(cached["response"]):
                    yield frame("token", chunk)
                self._history.append(user_id, thread_id, "user", message)
                self._history.append(
                    user_id, thread_id, "assistant", cached["response"]
                )
                yield frame(
                    "memory",
                    {
                        "phase": "cache",
                        "tier": "cache",
                        "cache_hit": True,
                        "replayed": True,
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                        "hits": [],
                    },
                )
                return
            yield frame(
                "memory",
                {
                    "phase": "cache",
                    "tier": "cache",
                    "query": message,
                    "cache_hit": False,
                    "hits": [],
                },
            )

            # ── Recall ───────────────────────────────────────────────────
            groups = await self._recall_all(user_id, message)
            context_block = format_context(groups)
            for tier in ("ltm", "stm", "episodic"):
                yield frame(
                    "memory",
                    {
                        "phase": "recall",
                        "tier": tier,
                        "query": message,
                        "hits": groups.get(tier, []),
                    },
                )

        # ── Generate ─────────────────────────────────────────────────────
        turns = self._history.turns(user_id, thread_id, limit=HISTORY_TURNS)
        turns.append(("user", message))
        messages, kwargs = build_call(
            self._provider_name, build_system_text(context_block), turns
        )

        answer_parts: list[str] = []
        async for chunk in self._memory.providers.llm.chat_stream(messages, **kwargs):
            answer_parts.append(chunk)
            yield frame("token", chunk)
        answer = "".join(answer_parts)

        self._history.append(user_id, thread_id, "user", message)
        self._history.append(user_id, thread_id, "assistant", answer)

        # ── Write back ───────────────────────────────────────────────────
        if memory_enabled and answer:
            async for write_frame in self._persist(
                user_id=user_id,
                thread_id=thread_id,
                message=message,
                answer=answer,
                correlation_id=correlation_id,
            ):
                yield write_frame

    async def _persist(
        self,
        *,
        user_id: str,
        thread_id: str,
        message: str,
        answer: str,
        correlation_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Write the turn to every tier, reporting each as its own frame.

        Failures are logged and reported, not raised: the user already has their
        answer, and losing it to a write error would be the worse outcome. Each
        tier is attempted independently so one failure does not skip the others.
        """
        # Semantic: STM now, and an LTM candidate the enrichment worker scores.
        try:
            await self._memory.add(
                user_id,
                thread_id,
                [
                    {"message_type": "human", "content": message},
                    {"message_type": "ai", "content": answer},
                ],
            )
            yield frame(
                "memory",
                {
                    "phase": "write",
                    "tier": "ltm",
                    "hits": [{"text": message}],
                    "note": "stored as short-term; queued for long-term scoring",
                },
            )
        except Exception as exc:
            log.exception("semantic write failed")
            yield frame(
                "memory",
                {"phase": "write", "tier": "ltm", "error": str(exc), "hits": []},
            )

        # Episodic: the turn itself. The message shape is the projection layer's
        # neutral dict form — `type` rather than `role`, which is what
        # `project_messages` reads.
        try:
            await self._memory.log_activity(
                user_id,
                thread_id,
                [
                    {"type": "human", "content": message},
                    {"type": "ai", "content": answer, "tool_calls": []},
                ],
                correlation_id=correlation_id,
                conversation_id=thread_id,
            )
            yield frame(
                "memory",
                {
                    "phase": "write",
                    "tier": "episodic",
                    "correlation_id": correlation_id,
                    "hits": [],
                    "note": "turn logged",
                },
            )
        except Exception as exc:
            log.exception("episodic write failed")
            yield frame(
                "memory",
                {"phase": "write", "tier": "episodic", "error": str(exc), "hits": []},
            )

        # Cache last: only cache an answer that was actually produced.
        try:
            await self._cache.store(user_id, message, answer, memory_enabled=True)
        except Exception:
            log.exception("cache store failed")


def _chunk_text(text: str, size: int = 24) -> list[str]:
    """Split a cached answer into token-sized pieces for replay."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]
