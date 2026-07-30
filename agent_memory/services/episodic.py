"""Episodic memory — the agent activity log.

The fourth memory tier, and the one most systems skip. Semantic memory knows
*facts* ("this user avoids shellfish"); episodic memory knows *what happened*
("at step 7 I called the menu tool, wrote plan.md, and told them Friday works").

Ask an agent "what did we decide about the Q3 forecast, and what did you actually
change?" — distilled facts cannot answer that. A per-turn record can.

One document per turn: what was said, which tools ran, what was planned, which
files were touched, in what order, under which trace id. Append-only, TTL'd, and
hybrid-searchable per user, exactly like the semantic tier — the same
``$rankFusion`` builder serves both.

The write path is deliberately asymmetric with the read path: writes are
fire-and-forget onto a bounded queue (see ``episodic_worker``), reads are
ordinary awaited queries.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from agent_memory.core.projection import (
    DEFAULT_FS_CREATE_TOOLS,
    DEFAULT_FS_WRITE_TOOLS,
    build_search_text,
    is_final_step,
    project_files,
    project_messages,
    project_todos,
)
from agent_memory.services.memory import _sanitize_doc
from agent_memory.services.search_pipeline import rank_fusion_pipeline

logger = logging.getLogger(__name__)

VECTOR_INDEX = "episodes_vector_index"
FTS_INDEX = "episodes_fts_index"
TTL_INDEX = "ix_episodes_ttl"

# Excluded from read results: it is large, unreadable, and the caller already
# has the query that matched it.
_READ_PROJECTION = {"embedding": 0}


class EpisodicService:
    """Append-only per-turn activity log over the ``episodes`` collection."""

    def __init__(
        self,
        episodes_collection,
        config,
        providers,
        *,
        counter_collection=None,
        worker=None,
        audit_service=None,
    ) -> None:
        self.episodes = episodes_collection
        self.config = config
        self.providers = providers
        # Per-thread step counters. A separate collection so the episodes
        # collection stays homogeneous (one shape, one TTL policy).
        self.counters = (
            counter_collection
            if counter_collection is not None
            else episodes_collection.database["episodes_counters"]
        )

        if worker is not None:
            self.worker = worker
        else:
            from agent_memory.services.episodic_worker import EpisodicWorker

            self.worker = EpisodicWorker(
                self.episodes, self.counters, providers, config,
                audit_service=audit_service,
            )

        # Threads already warned about an unsearchable final step; warn once
        # each rather than on every turn.
        self._search_warned: set[str] = set()

    # ─── Write path ──────────────────────────────────────────────

    def log_activity(
        self,
        user_id: str,
        thread_id: str,
        messages: list[Any],
        *,
        todos: list[dict] | None = None,
        agent_name: str | None = None,
        correlation_id: str | None = None,
        conversation_id: str | None = None,
        ts: datetime | None = None,
        fs_write_tools: frozenset[str] = DEFAULT_FS_WRITE_TOOLS,
        fs_create_tools: frozenset[str] = DEFAULT_FS_CREATE_TOOLS,
    ) -> bool:
        """Project one turn and enqueue it. Synchronous, non-blocking, never raises.

        Not a coroutine on purpose: there is nothing to await. Everything that
        touches the network happens on the worker, so a caller cannot
        accidentally add latency to a turn by logging it.

        Returns True if the turn was enqueued. False means it was discarded —
        episodic memory is disabled, or the identifiers were missing.
        """
        if not self.config.episodic_enabled:
            return False
        # No tenant, no document. This is the write-side half of isolation;
        # there is no "unscoped" episodic record.
        if not user_id or not thread_id:
            logger.warning("Episodic log skipped: user_id and thread_id are required.")
            return False

        cap = self.config.episodic_content_cap
        messages_proj = project_messages(messages, cap=cap)
        doc: dict[str, Any] = {
            "user_id": user_id,
            "thread_id": thread_id,
            "conversation_id": conversation_id or thread_id,
            "agent_name": agent_name or "main",
            "ts": ts if ts is not None else datetime.now(UTC),
            "messages": messages_proj,
            "todos": project_todos(todos if todos is not None else []),
            "files_touched": project_files(
                messages,
                fs_write_tools=fs_write_tools,
                fs_create_tools=fs_create_tools,
            ),
            # "" rather than None so an absent id does not create a null bucket
            # in the (user_id, correlation_id) index.
            "correlation_id": correlation_id or "",
            # The worker resolves this into step/parent_step from the durable
            # counter, keeping that round trip off this call.
            "__assign_step": thread_id,
        }

        self._maybe_attach_search_text(doc, messages_proj, thread_id)
        self.worker.enqueue(doc)
        return True

    def _maybe_attach_search_text(
        self, doc: dict[str, Any], messages_proj: list[dict], thread_id: str
    ) -> None:
        """Mark the turn for embedding when it has answer text worth searching.

        Only final steps qualify: a step that ends in a tool request has a
        question but no answer, so its embedding would represent half a turn.
        """
        if self.config.episodic_embed_final_steps_only and not is_final_step(
            messages_proj
        ):
            return
        text = build_search_text(
            messages_proj, cap=self.config.episodic_search_text_cap
        )
        if text:
            doc["__search_text"] = text
        elif thread_id not in self._search_warned:
            # A single-role turn is stored but invisible to recall. Worth saying
            # once, because the document silently lacks two fields.
            self._search_warned.add(thread_id)
            logger.warning(
                "Episodic turn for thread=%s has no searchable text (needs both a "
                "human and an ai message); stored but not recallable.",
                thread_id,
            )

    async def flush(self, timeout: float = 5.0) -> bool:
        """Wait for pending turns to reach Atlas. Bounded; never raises."""
        return await self.worker.flush(timeout)

    async def close(self, timeout: float = 5.0) -> bool:
        """Flush and stop accepting writes. Idempotent."""
        return await self.worker.close(timeout)

    def stats(self) -> dict[str, Any]:
        """Worker queue and throughput counters. No database round trip."""
        return self.worker.stats()

    # ─── Read path ───────────────────────────────────────────────

    async def search(
        self,
        user_id: str,
        query: str,
        *,
        thread_id: str | None = None,
        agent_name: str | None = None,
        since: datetime | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Hybrid recall over logged turns via ``$rankFusion``.

        Meaning and exact terms, fused in the database. The ``user_id`` filter
        goes into both branches, so isolation is enforced by the engine rather
        than by the caller remembering to add it.

        ``since`` is applied after fusion: ``ts`` is not a vector-index filter
        field, and declaring a high-cardinality date as one would bloat the index
        for a rarely-used narrowing.
        """
        limit = min(limit, self.config.max_results_per_query)
        query_embedding = await self.providers.embedding.generate_embedding(query)

        vs_filter: dict[str, Any] = {"user_id": user_id}
        fts_filter_clauses: list[dict[str, Any]] = [
            {"equals": {"path": "user_id", "value": user_id}}
        ]
        if thread_id:
            vs_filter["thread_id"] = thread_id
            fts_filter_clauses.append(
                {"equals": {"path": "thread_id", "value": thread_id}}
            )
        if agent_name:
            vs_filter["agent_name"] = agent_name
            fts_filter_clauses.append(
                {"equals": {"path": "agent_name", "value": agent_name}}
            )

        pipeline = rank_fusion_pipeline(
            query=query,
            query_embedding=query_embedding,
            vector_index=VECTOR_INDEX,
            fts_index=FTS_INDEX,
            fts_paths=["search_text"],
            vs_filter=vs_filter,
            fts_filter_clauses=fts_filter_clauses,
            limit=limit,
            vector_weight=self.config.rrf_vector_weight,
            text_weight=self.config.rrf_text_weight,
        )
        if since is not None:
            pipeline.append({"$match": {"ts": {"$gte": since}}})

        cursor = await self.episodes.aggregate(pipeline)
        results = await cursor.to_list(None)
        for doc in results:
            _sanitize_doc(doc)
        return results

    async def get_thread(
        self,
        user_id: str,
        thread_id: str,
        *,
        limit: int | None = None,
        ascending: bool = True,
    ) -> list[dict]:
        """Return a thread's turns in step order — the replay read.

        ``user_id`` is required, not optional: a thread id is not a capability.
        """
        direction = 1 if ascending else -1
        return await self._read(
            {"user_id": user_id, "thread_id": thread_id},
            sort=[("step", direction), ("ts", direction)],
            limit=limit,
        )

    async def get_by_correlation_id(
        self, user_id: str, correlation_id: str, *, limit: int | None = None
    ) -> list[dict]:
        """Return every turn sharing a trace id, oldest first.

        This is the join back to your tracing stack: one trace id, every turn
        the agent logged while serving that request.
        """
        return await self._read(
            {"user_id": user_id, "correlation_id": correlation_id},
            sort=[("ts", 1), ("step", 1)],
            limit=limit,
        )

    async def _read(
        self,
        query: dict[str, Any],
        *,
        sort: list[tuple[str, int]],
        limit: int | None,
    ) -> list[dict]:
        cursor = self.episodes.find(query, _READ_PROJECTION).sort(sort)
        if limit is not None:
            cursor = cursor.limit(limit)
        docs = await cursor.to_list(None)
        out = []
        for doc in docs:
            doc = dict(doc)
            _sanitize_doc(doc)
            out.append(doc)
        return out

    # ─── Retention ───────────────────────────────────────────────

    async def set_retention(self, ttl_seconds: int | None) -> dict[str, Any]:
        """Change episodic retention in place. Never raises.

        ``collMod`` mutates ``expireAfterSeconds`` without dropping the index,
        so retention is a runtime knob rather than a redeploy. ``None`` removes
        the TTL index, making the log permanent.

        Falls back to ``create_index`` on deployments without ``collMod``.
        """
        if ttl_seconds is None:
            try:
                await self.episodes.drop_index(TTL_INDEX)
                return {"status": "removed", "ttl_seconds": None}
            except Exception as exc:
                logger.warning("Episodic TTL removal failed: %s", exc)
                return {"status": "error", "ttl_seconds": None, "error": str(exc)}

        try:
            await self.episodes.database.command(
                {
                    "collMod": self.episodes.name,
                    "index": {
                        "name": TTL_INDEX,
                        "expireAfterSeconds": ttl_seconds,
                    },
                }
            )
            return {"status": "updated", "ttl_seconds": ttl_seconds}
        except Exception as exc:
            logger.warning(
                "Episodic set_retention: collMod unavailable (%s); creating the "
                "index instead.",
                exc,
            )

        try:
            await self.episodes.create_index(
                [("ts", 1)], name=TTL_INDEX, expireAfterSeconds=ttl_seconds
            )
            return {"status": "created", "ttl_seconds": ttl_seconds}
        except Exception as exc:
            logger.warning("Episodic TTL creation failed: %s", exc)
            return {"status": "error", "ttl_seconds": ttl_seconds, "error": str(exc)}


__all__ = ["EpisodicService"]
