"""Core memory service — store, recall, delete, evolve."""

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from bson import ObjectId

from agent_memory.core.config import MCPConfig
from agent_memory.services.search_pipeline import rank_fusion_pipeline

logger = logging.getLogger(__name__)

# The retention tier a promoted memory lands in. Named rather than inlined because
# promotion sets `retention_tier` and `expires_at` and the two must agree — a
# document claiming `standard` while carrying an STM expiry is how long-term
# memories came to be deleted on a 24-hour schedule.
PROMOTED_RETENTION_TIER = "standard"


def retention_ttl(config: MCPConfig, retention_tier: str) -> timedelta:
    """TTL for a retention tier. Unknown tiers fall back to standard.

    Module-level rather than a `MemoryService` method because the consolidation
    worker needs exactly this mapping when it promotes STM→LTM, and it holds no
    `MemoryService`. Duplicating the table there is what let promotion change
    `tier` and `retention_tier` while leaving `expires_at` on its short-term
    schedule: the promoted memory kept the 24-hour TTL it was born with, so the
    TTL index deleted it a day later despite it being long-term by every field
    that describes it. One table, one meaning.
    """
    tier_map = {
        "critical": timedelta(days=config.ltm_retention_critical_days),
        "reference": timedelta(days=config.ltm_retention_reference_days),
        "standard": timedelta(days=config.ltm_retention_standard_days),
        "temporary": timedelta(days=config.ltm_retention_temporary_days),
        "ephemeral": timedelta(hours=config.stm_ttl_hours),
    }
    return tier_map.get(
        retention_tier, timedelta(days=config.ltm_retention_standard_days)
    )


def _sanitize_value(val):
    """Return ``val`` with BSON types replaced by JSON-safe equivalents."""
    if isinstance(val, ObjectId):
        return str(val)
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, dict):
        _sanitize_doc(val)
        return val
    if isinstance(val, list):
        return [_sanitize_value(item) for item in val]
    return val


def tag_filter(tags: list[str]) -> dict:
    """An all-of tag restriction expressed in operators ``$vectorSearch`` supports.

    ``{"tags": {"$all": [...]}}`` is the obvious spelling and the wrong one here.
    A ``$vectorSearch`` pre-filter accepts only ``$eq``/``$ne``, the range
    operators, ``$in``/``$nin``, ``$exists``, and ``$and``/``$or``/``$not``/``$nor``
    — ``$all`` is not among them, and an unsupported operator in a pre-filter does
    not raise. The branch matches nothing, so every tag-filtered search came back
    empty and read as "no memories carry those tags".

    ``filter`` fields may be arrays, and an array field matches when *any* element
    matches. So all-of is an ``$and`` of single-value equalities: each clause
    requires that one tag be present, and together they require all of them. This
    is the same semantics ``$all`` has, in operators the engine will actually
    evaluate.

    A single tag needs no wrapper, which keeps the common filter simple enough to
    read in a log.
    """
    if len(tags) == 1:
        return {"tags": tags[0]}
    return {"$and": [{"tags": t} for t in tags]}


def tag_fts_clauses(tags: list[str]) -> list[dict]:
    """The same all-of tag restriction as Atlas Search ``equals`` clauses.

    ``compound.filter`` is itself an AND, so one ``equals`` per tag gives all-of
    without further nesting. ``equals`` matches an array field when any element
    matches, and requires the field to be indexed as ``token`` — see
    ``memories_fts_index``.
    """
    return [{"equals": {"path": "tags", "value": t}} for t in tags]


def _sanitize_doc(doc: dict) -> None:
    """Convert BSON types (ObjectId, datetime) to JSON-safe strings in place.

    Recurses into lists as well as dicts: episodic documents carry
    ``messages[]``, ``todos[]``, and ``files_touched[]``, which would otherwise
    leak raw BSON through the JSON boundary.
    """
    for key, val in list(doc.items()):
        doc[key] = _sanitize_value(val)


class MemoryService:
    """Encapsulates memory CRUD operations.

    All query methods inject ``user_id`` and ``deleted_at: null`` automatically
    via ``_base_filter()``.
    """

    def __init__(self, memories_collection, config: MCPConfig, providers) -> None:
        self.memories = memories_collection
        self.config = config
        self.providers = providers

    def _retention_ttl(self, retention_tier: str) -> timedelta:
        """Return TTL for a given retention tier."""
        return retention_ttl(self.config, retention_tier)

    def _base_filter(self, user_id: str, **extra) -> dict:
        """Base filter injecting user isolation and soft-delete exclusion."""
        f: dict = {"user_id": user_id, "deleted_at": None}
        f.update(extra)
        return f

    async def store_stm(
        self,
        user_id: str,
        conversation_id: str,
        messages: list[dict],
    ) -> list[str]:
        """Store STM messages.  For significant human messages, also queue LTM creation."""
        if not messages:
            return []

        texts = [m["content"] for m in messages]
        embeddings = await self.providers.embedding.generate_embeddings_batch(texts)

        docs = []
        for msg, emb in zip(messages, embeddings):
            now = datetime.now(UTC)
            stm_doc = {
                "user_id": user_id,
                "tier": "stm",
                "content": msg["content"],
                "summary": None,
                "embedding": emb,
                "memory_type": None,
                "retention_tier": "ephemeral",
                "tags": msg.get("tags", []),
                "importance": 0.5,
                "access_count": 0,
                "last_accessed": None,
                "conversation_id": conversation_id,
                "message_type": msg["message_type"],
                "source_stm_id": None,
                "enrichment_status": "not_applicable",
                "enrichment_retries": 0,
                "created_at": now,
                "updated_at": now,
                "expires_at": now + self._retention_ttl("ephemeral"),
                "deleted_at": None,
                "is_deleted": False,
            }
            docs.append(stm_doc)

        result = await self.memories.insert_many(docs)
        stm_ids = result.inserted_ids

        # Create LTM candidates for significant human messages
        ltm_docs = []
        for i, msg in enumerate(messages):
            if msg["message_type"] == "human" and len(msg["content"]) > 30:
                ltm_now = datetime.now(UTC)
                ltm_doc = {
                    "user_id": user_id,
                    "tier": "ltm",
                    "content": msg["content"],
                    "summary": None,
                    "embedding": embeddings[i],
                    "memory_type": None,
                    "retention_tier": "standard",
                    "tags": msg.get("tags", []),
                    "importance": 0.5,
                    "access_count": 0,
                    "last_accessed": None,
                    "conversation_id": conversation_id,
                    "message_type": msg["message_type"],
                    "source_stm_id": stm_ids[i],
                    "enrichment_status": "pending",
                    "enrichment_retries": 0,
                    "created_at": ltm_now,
                    "updated_at": ltm_now,
                    "expires_at": ltm_now + self._retention_ttl("standard"),
                    "deleted_at": None,
                    "is_deleted": False,
                }
                ltm_docs.append(ltm_doc)

        if ltm_docs:
            try:
                await self.memories.insert_many(ltm_docs)
            except Exception:
                # Partial failure acceptable — STM persisted, LTM creation retryable
                logger.exception("Failed to insert LTM candidates")

        # Returns only STM document IDs.  LTM candidates are internal
        # implementation details not exposed to MCP clients.
        return [str(id_) for id_ in stm_ids]

    async def recall(
        self,
        user_id: str,
        query: str,
        tier: list[str] | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Semantic search with calibrated ranking and STM/LTM dedup.

        ``memory_type`` and ``tags`` are ``$vectorSearch`` *pre-filters*, so both
        paths have to be declared as ``filter`` fields in
        ``memories_vector_index`` — see ``core.collections``. An undeclared path
        silently matches nothing rather than erroring, so the two narrowing
        arguments this method advertises used to guarantee an empty result.
        """
        limit = min(limit or 10, self.config.max_results_per_query)
        query_embedding = await self.providers.embedding.generate_embedding(query)

        # Build vector search filter
        vs_filter: dict = {"user_id": user_id, "deleted_at": None}
        if tier:
            vs_filter["tier"] = {"$in": tier}
        if memory_type:
            vs_filter["memory_type"] = memory_type
        if tags:
            # `tag_filter`, not `{"$all": tags}`: `$all` is not a supported
            # pre-filter operator and fails silently. See `tag_filter`.
            vs_filter.update(tag_filter(tags))

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "memories_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit * 2,  # Over-fetch for dedup
                    "filter": vs_filter,
                }
            },
            {"$addFields": {"vs_score": {"$meta": "vectorSearchScore"}}},
        ]

        cursor = await self.memories.aggregate(pipeline)
        results = await cursor.to_list(None)

        if not results:
            return []

        # Deduplicate STM/LTM pairs by source_stm_id
        results = self._deduplicate(results)

        # Apply calibrated 3-component ranking (Section 4.2 of design spec)
        now = datetime.now(UTC)
        results = self._calibrated_rank(results, now)

        # Trim to limit
        results = results[:limit]

        # Increment access_count on returned results
        if results:
            result_ids = [r["_id"] for r in results]
            await self.memories.update_many(
                {"_id": {"$in": result_ids}},
                {
                    "$inc": {"access_count": 1},
                    "$set": {"last_accessed": datetime.now(UTC)},
                },
            )

        # Strip internal scores, sanitize BSON types for JSON serialization
        for r in results:
            r.pop("embedding", None)
            r.pop("vs_score", None)
            _sanitize_doc(r)

        return results

    def _calibrated_rank(self, results: list[dict], now: datetime) -> list[dict]:
        """Apply calibrated 3-component scoring and re-sort.

        final_score = alpha * recency + beta * importance_score + gamma * relevance

        Where:
          recency    = exp(-age_days / 30)                                  [0, 1]
          importance = importance * min(1 + ln(access_count + 1), 3.0) / 3  [0, 1]
          relevance  = vs_score (cosine similarity)                         [0, 1]
        """
        alpha = self.config.ranking_alpha
        beta = self.config.ranking_beta
        gamma = self.config.ranking_gamma

        for r in results:
            created_at = r.get("created_at", now)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            age_days = max((now - created_at).total_seconds() / 86400, 0)
            recency = math.exp(-age_days / 30)

            importance = r.get("importance", 0.5)
            access_count = r.get("access_count", 0)
            importance_score = importance * min(1 + math.log(access_count + 1), 3.0) / 3.0

            relevance = r.get("vs_score", 0)

            r["final_score"] = alpha * recency + beta * importance_score + gamma * relevance

        results.sort(key=lambda r: r["final_score"], reverse=True)
        return results

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """Deduplicate STM/LTM pairs linked by source_stm_id.

        When both an STM document and its LTM candidate appear, keep the
        higher-scoring one and suppress the other.
        """
        seen_stm_ids: dict[ObjectId, dict] = {}
        deduped = []

        for r in results:
            source_stm_id = r.get("source_stm_id")
            if source_stm_id:
                # This is an LTM candidate — check if we already have its STM
                if source_stm_id in seen_stm_ids:
                    existing = seen_stm_ids[source_stm_id]
                    if r.get("vs_score", 0) > existing.get("vs_score", 0):
                        # Replace STM with this LTM
                        deduped.remove(existing)
                        deduped.append(r)
                        seen_stm_ids[source_stm_id] = r
                    # else: keep existing, skip this one
                else:
                    seen_stm_ids[source_stm_id] = r
                    deduped.append(r)
            else:
                stm_id = r.get("_id")
                if stm_id in seen_stm_ids:
                    existing = seen_stm_ids[stm_id]
                    if r.get("vs_score", 0) > existing.get("vs_score", 0):
                        deduped.remove(existing)
                        deduped.append(r)
                        seen_stm_ids[stm_id] = r
                else:
                    seen_stm_ids[stm_id] = r
                    deduped.append(r)

        return deduped

    async def delete(
        self,
        user_id: str,
        memory_id: str | None = None,
        tags: list[str] | None = None,
        time_range: dict | None = None,
        confirm: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Soft-delete memories matching criteria."""
        is_bulk = memory_id is None
        if is_bulk and not confirm:
            raise ValueError(
                "Bulk delete requires confirm=True. "
                "Pass confirm=True to proceed or use memory_id for single delete."
            )

        # Build filter
        query_filter = self._base_filter(user_id)
        if memory_id:
            query_filter["_id"] = ObjectId(memory_id)
        if tags:
            query_filter["tags"] = {"$all": tags}
        if time_range:
            time_filter = {}
            if "start" in time_range:
                time_filter["$gte"] = time_range["start"]
            if "end" in time_range:
                time_filter["$lte"] = time_range["end"]
            if time_filter:
                query_filter["created_at"] = time_filter

        if dry_run:
            count = await self.memories.count_documents(query_filter)
            return {"deleted_count": count, "dry_run": True}

        now = datetime.now(UTC)
        result = await self.memories.update_many(
            query_filter,
            {"$set": {"deleted_at": now, "is_deleted": True, "updated_at": now}},
        )
        return {"deleted_count": result.modified_count}

    async def hybrid_search(
        self,
        user_id: str,
        query: str,
        tier: list[str] | None = None,
        limit: int = 10,
        memory_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """Hybrid vector + full-text search via MongoDB ``$rankFusion`` (RRF).

        Returns raw fused-relevance matches (with scores), no curation — the
        ``search`` primitive. Extracted from the former ``hybrid_search`` MCP
        tool so the facade stays a thin orchestration layer.

        ``memory_type`` and ``tags`` are applied to **both** branches. Applying a
        narrowing to only one is worse than not applying it: the vector branch
        honoured it, the full-text branch did not, and ``$rankFusion`` fused the two
        ranked lists — so a search scoped to one ``memory_type`` returned documents
        of other types, mixed in by relevance and indistinguishable from correct
        hits. ``user_id`` was always in both, so this was never an isolation bug,
        but the caller's filter meant whatever the fusion happened to produce.
        """
        limit = min(limit, self.config.max_results_per_query)
        tiers = tier or ["stm", "ltm"]
        query_embedding = await self.providers.embedding.generate_embedding(query)

        vs_filter: dict = {"user_id": user_id, "deleted_at": None, "tier": {"$in": tiers}}
        if memory_type:
            vs_filter["memory_type"] = memory_type
        if tags:
            vs_filter.update(tag_filter(tags))

        fts_filter_clauses = [
            {"equals": {"path": "user_id", "value": user_id}},
            {"equals": {"path": "is_deleted", "value": False}},
        ]
        if tiers:
            fts_filter_clauses.append({"in": {"path": "tier", "value": tiers}})
        if memory_type:
            fts_filter_clauses.append(
                {"equals": {"path": "memory_type", "value": memory_type}}
            )
        if tags:
            fts_filter_clauses.extend(tag_fts_clauses(tags))

        pipeline = rank_fusion_pipeline(
            query=query,
            query_embedding=query_embedding,
            vector_index="memories_vector_index",
            fts_index="memories_fts_index",
            fts_paths=["content", "summary"],
            vs_filter=vs_filter,
            fts_filter_clauses=fts_filter_clauses,
            limit=limit,
            vector_weight=self.config.rrf_vector_weight,
            text_weight=self.config.rrf_text_weight,
        )

        cursor = await self.memories.aggregate(pipeline)
        results = await cursor.to_list(None)
        for r in results:
            _sanitize_doc(r)
        return results

    async def evolve_memory(
        self,
        user_id: str,
        content: str,
        embedding: list[float],
        *,
        exclude_id: Any = None,
    ) -> str:
        """Check for similar memories and reinforce/merge/create.

        ``exclude_id`` is the memory this call is evolving, and passing it matters.
        The only caller is the enrichment worker, which runs *after* the document
        is already stored — so a search for "memories similar to this content"
        found the document itself, at a similarity of essentially 1.0. That is
        above ``reinforce_threshold`` by construction, so every enrichment pass
        took the reinforce branch against its own ``_id``: it multiplied its own
        importance by 1.1, incremented its own ``access_count``, and returned
        "reinforced" without ever looking at the real duplicates ranked below it.

        Two things were wrong at once. Genuine near-duplicates were never merged,
        because the self-match consumed the decision; and importance drifted upward
        on nothing — a memory that no one had read became "important" purely by
        being enriched, which then biased ranking and promotion in its favour.
        """
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": "memories_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 50,
                    # One extra candidate, so excluding the self-match below still
                    # leaves five real ones to consider.
                    "limit": 6 if exclude_id is not None else 5,
                    "filter": {
                        "user_id": user_id,
                        "tier": "ltm",
                        "deleted_at": None,
                    },
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        ]
        if exclude_id is not None:
            # A `$match` after the search rather than a `$vectorSearch` filter:
            # filter fields have to be declared in the index definition and `_id`
            # is not one of them, so putting it there would be silently ignored.
            pipeline.append({"$match": {"_id": {"$ne": exclude_id}}})

        cursor = await self.memories.aggregate(pipeline)
        similar = await cursor.to_list(None)

        if not similar:
            return "created"

        top = similar[0]
        similarity = top.get("score", 0)

        if similarity > self.config.reinforce_threshold:
            await self.memories.update_one(
                {"_id": top["_id"]},
                {
                    "$set": {
                        "updated_at": datetime.now(UTC),
                        "importance": min(top.get("importance", 0.5) * 1.1, 1.0),
                    },
                    "$inc": {"access_count": 1},
                },
            )
            return "reinforced"

        if similarity > self.config.merge_threshold:
            # Create new memory immediately for searchability,
            # queue async merge via enrichment worker
            now = datetime.now(UTC)
            merge_doc = {
                "user_id": user_id,
                "tier": "ltm",
                "content": content,
                "summary": None,
                "embedding": embedding,
                "memory_type": None,
                "retention_tier": "standard",
                "tags": [],
                "importance": top.get("importance", 0.5),
                "access_count": 0,
                "last_accessed": None,
                "conversation_id": None,
                "message_type": None,
                "source_stm_id": None,
                "enrichment_status": "merge_pending",
                "enrichment_retries": 0,
                "merge_target_id": top["_id"],
                "created_at": now,
                "updated_at": now,
                "expires_at": now + self._retention_ttl("standard"),
                "deleted_at": None,
                "is_deleted": False,
            }
            await self.memories.insert_one(merge_doc)
            return "merge_queued"

        return "created"
