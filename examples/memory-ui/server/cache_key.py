"""Semantic cache with ``memory_enabled`` as part of the key.

This exists for one reason, and it is a demo-correctness reason rather than a
performance one.

The library's ``check_cache`` filters on ``user_id`` only. That is right for
production — the same user asking the same question should get the cached answer.
It is wrong for a demo whose entire claim is *"same model, same prompt, one
toggle, different answer"*: the memory-ON pass caches a memory-informed answer,
and the memory-OFF pass then finds it by similarity and replays it. The screen
would show identical answers with the toggle flipped, on stage, in front of the
people you are trying to convince.

So this wrapper keeps its own collection with ``memory_enabled`` as a second
filter axis. Two consequences worth being explicit about:

* A memory-OFF turn is cached separately, never crossed with a memory-ON one.
* Turning the toggle off does not merely skip recall; it also bypasses this cache
  entirely (see ``turn.py``), so a memory-OFF answer is always freshly generated.
  That is deliberately stricter than necessary — belt and braces on the one
  failure that cannot be recovered from mid-talk.

The collection is separate from the library's ``semantic_cache`` rather than an
extra field on it, because a filter field must be declared in the vector index
definition, and adding one to the library's index for a demo's benefit would be
the wrong trade. An undeclared filter field does not error — it silently returns
nothing, which is the worst of both outcomes.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

COLLECTION = "demo_response_cache"
VECTOR_INDEX = "demo_cache_vector_index"

# Higher than the library's 0.95 default. A booth demo asks near-identical
# questions by design ("what should I make Friday?" twice), and a loose threshold
# turns a deliberate second ask into a cache hit when the point was to show a
# fresh recall.
SIMILARITY_THRESHOLD = 0.97

# Cache entries live for the length of a rehearsal plus a talk. Long enough that
# the second morning's warm-up is fast, short enough that Tuesday's answers do
# not surface on Wednesday when the seed data has been reset.
TTL_SECONDS = 6 * 60 * 60


def normalize(query: str) -> str:
    """Collapse a query to a comparable key: casefolded, punctuation-free.

    ``"What should I make Friday?"`` and ``"what should i make friday"`` are the
    same question and must be the same key.
    """
    return re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()


class DemoResponseCache:
    """Vector-similarity response cache, scoped by user and memory mode."""

    def __init__(self, db, embedding_provider) -> None:
        self._collection = db[COLLECTION]
        self._embedding = embedding_provider

    async def ensure_indexes(self) -> None:
        """Create the TTL and vector indexes. Idempotent; never raises.

        A cache is an optimization, so a deployment that cannot create the index
        should still serve turns — ``lookup`` will simply always miss.
        """
        try:
            await self._collection.create_index(
                [("created_at", 1)], name="ix_demo_cache_ttl",
                expireAfterSeconds=TTL_SECONDS,
            )
        except Exception:
            log.warning("demo cache TTL index not created", exc_info=True)

        # Backs the exact-match fast path in `lookup`. A plain B-tree index, so it
        # is queryable the instant the write commits — which is the entire reason
        # that path exists.
        try:
            await self._collection.create_index(
                [("user_id", 1), ("memory_enabled", 1), ("query_key", 1)],
                name="ix_demo_cache_exact",
            )
        except Exception:
            log.warning("demo cache exact index not created", exc_info=True)

        try:
            existing = await (
                await self._collection.list_search_indexes()
            ).to_list(None)
            if any(idx.get("name") == VECTOR_INDEX for idx in existing):
                return
            dimension = len(await self._embedding.generate_embedding("dimension probe"))
            await self._collection.create_search_index(
                {
                    "name": VECTOR_INDEX,
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                # Probed from the live embedder rather than read
                                # from config: a mismatch here does not raise, it
                                # makes every lookup silently miss.
                                "numDimensions": dimension,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "user_id"},
                            # The axis this whole module exists for. Undeclared,
                            # the filter would match nothing and every lookup
                            # would miss — which fails safe, but silently.
                            {"type": "filter", "path": "memory_enabled"},
                        ]
                    },
                }
            )
            log.info("created %s (%d dims)", VECTOR_INDEX, dimension)
        except Exception:
            log.warning("demo cache vector index not created", exc_info=True)

    async def lookup(
        self, user_id: str, query: str, *, memory_enabled: bool
    ) -> dict[str, Any] | None:
        """Return a cached response for this user *and* this memory mode.

        Two lookups, in order, and the order is the point.

        **Exact match first.** Atlas Search indexes are eventually consistent: a
        document inserted at the end of one turn is typically not queryable by
        ``$vectorSearch`` for a few seconds. The demo asks the identical question
        immediately after the answer that populated the cache, so the vector path
        alone reliably *misses* on the one repeat the audience is watching for —
        and the fix cannot be "the presenter pauses long enough", because on stage
        that pause is dead air. A normalized-string lookup on a B-tree index is
        queryable the moment the write commits, so the repeat hits every time.

        This is not a demo cheat. A production semantic cache wants the same fast
        path for the same reason: an exact repeat should never pay for an
        embedding round-trip plus a vector search.

        **Then similarity.** The vector path is what earns the word *semantic* —
        it catches a reworded question that the string key cannot. Both are real;
        the frame reports which one answered so the panel can say so.
        """
        try:
            exact = await self._collection.find_one(
                {
                    "user_id": user_id,
                    "memory_enabled": memory_enabled,
                    "query_key": normalize(query),
                },
                {"embedding": 0},
                sort=[("created_at", -1)],
            )
            if exact is not None:
                return {
                    "query": exact.get("query"),
                    "response": exact.get("response"),
                    # An exact string match is a similarity of 1.0 by definition,
                    # not a measurement being passed off as one.
                    "score": 1.0,
                    "match": "exact",
                }
        except Exception:
            log.warning("demo cache exact lookup failed; trying vector", exc_info=True)

        try:
            embedding = await self._embedding.generate_embedding(query)
            cursor = await self._collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": VECTOR_INDEX,
                            "path": "embedding",
                            "queryVector": embedding,
                            "numCandidates": 20,
                            "limit": 1,
                            "filter": {
                                "user_id": {"$eq": user_id},
                                "memory_enabled": {"$eq": memory_enabled},
                            },
                        }
                    },
                    {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                    {"$project": {"embedding": 0}},
                ]
            )
            results = await cursor.to_list(None)
        except Exception:
            # Every cache failure is a bypass, never an error to the caller.
            log.warning("demo cache lookup failed; bypassing", exc_info=True)
            return None

        if not results:
            return None
        top = results[0]
        if top.get("score", 0) < SIMILARITY_THRESHOLD:
            return None
        return {
            "query": top.get("query"),
            "response": top.get("response"),
            "score": top.get("score"),
            "match": "semantic",
        }

    async def store(
        self, user_id: str, query: str, response: str, *, memory_enabled: bool
    ) -> None:
        """Cache a response under this user and memory mode. Never raises."""
        if not response.strip():
            return
        try:
            embedding = await self._embedding.generate_embedding(query)
            await self._collection.insert_one(
                {
                    "user_id": user_id,
                    "memory_enabled": memory_enabled,
                    "query": query,
                    # The exact-match key. Stored alongside the raw query rather
                    # than replacing it, because the panel displays what was
                    # actually asked.
                    "query_key": normalize(query),
                    "response": response,
                    "embedding": embedding,
                    "created_at": datetime.now(timezone.utc),
                }
            )
        except Exception:
            log.warning("demo cache store failed", exc_info=True)

    async def clear(self, user_id: str) -> int:
        """Drop this user's cached responses. Backs the UI's reset button."""
        result = await self._collection.delete_many({"user_id": user_id})
        return result.deleted_count
