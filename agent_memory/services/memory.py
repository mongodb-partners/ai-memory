"""Core memory service — store, recall, delete, evolve."""

import logging
import math
from datetime import datetime, timedelta, timezone

from bson import ObjectId

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)


def _sanitize_doc(doc: dict) -> None:
    """Convert BSON types (ObjectId, datetime) to JSON-safe strings in place."""
    for key, val in list(doc.items()):
        if isinstance(val, ObjectId):
            doc[key] = str(val)
        elif isinstance(val, datetime):
            doc[key] = val.isoformat()
        elif isinstance(val, dict):
            _sanitize_doc(val)


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
        tier_map = {
            "critical": timedelta(days=self.config.ltm_retention_critical_days),
            "reference": timedelta(days=self.config.ltm_retention_reference_days),
            "standard": timedelta(days=self.config.ltm_retention_standard_days),
            "temporary": timedelta(days=self.config.ltm_retention_temporary_days),
            "ephemeral": timedelta(hours=self.config.stm_ttl_hours),
        }
        return tier_map.get(retention_tier, timedelta(days=self.config.ltm_retention_standard_days))

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
            now = datetime.now(timezone.utc)
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
                ltm_now = datetime.now(timezone.utc)
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
        """Semantic search with calibrated ranking and STM/LTM dedup."""
        limit = min(limit or 10, self.config.max_results_per_query)
        query_embedding = await self.providers.embedding.generate_embedding(query)

        # Build vector search filter
        vs_filter: dict = {"user_id": user_id, "deleted_at": None}
        if tier:
            vs_filter["tier"] = {"$in": tier}
        if memory_type:
            vs_filter["memory_type"] = memory_type
        if tags:
            vs_filter["tags"] = {"$all": tags}

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
        now = datetime.now(timezone.utc)
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
                    "$set": {"last_accessed": datetime.now(timezone.utc)},
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
                created_at = created_at.replace(tzinfo=timezone.utc)
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

        now = datetime.now(timezone.utc)
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
        """
        limit = min(limit, self.config.max_results_per_query)
        tiers = tier or ["stm", "ltm"]
        query_embedding = await self.providers.embedding.generate_embedding(query)

        vs_filter: dict = {"user_id": user_id, "deleted_at": None, "tier": {"$in": tiers}}
        if memory_type:
            vs_filter["memory_type"] = memory_type
        if tags:
            vs_filter["tags"] = {"$all": tags}

        fts_filter_clauses = [
            {"equals": {"path": "user_id", "value": user_id}},
            {"equals": {"path": "is_deleted", "value": False}},
        ]
        if tiers:
            fts_filter_clauses.append({"in": {"path": "tier", "value": tiers}})

        pipeline = [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vectorPipeline": [
                                {
                                    "$vectorSearch": {
                                        "index": "memories_vector_index",
                                        "path": "embedding",
                                        "queryVector": query_embedding,
                                        "numCandidates": 100,
                                        "limit": 20,
                                        "filter": vs_filter,
                                    }
                                },
                            ],
                            "fullTextPipeline": [
                                {
                                    "$search": {
                                        "index": "memories_fts_index",
                                        "compound": {
                                            "must": [
                                                {"text": {"query": query, "path": ["content", "summary"]}}
                                            ],
                                            "filter": fts_filter_clauses,
                                        },
                                    }
                                },
                                {"$limit": 20},
                            ],
                        }
                    },
                    "combination": {
                        "weights": {
                            "vectorPipeline": self.config.rrf_vector_weight,
                            "fullTextPipeline": self.config.rrf_text_weight,
                        },
                    },
                }
            },
            {"$limit": limit},
            {"$project": {"embedding": 0}},
        ]

        cursor = await self.memories.aggregate(pipeline)
        results = await cursor.to_list(None)
        for r in results:
            _sanitize_doc(r)
        return results

    async def evolve_memory(
        self, user_id: str, content: str, embedding: list[float]
    ) -> str:
        """Check for similar memories and reinforce/merge/create."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "memories_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 50,
                    "limit": 5,
                    "filter": {
                        "user_id": user_id,
                        "tier": "ltm",
                        "deleted_at": None,
                    },
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        ]

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
                        "updated_at": datetime.now(timezone.utc),
                        "importance": min(top.get("importance", 0.5) * 1.1, 1.0),
                    },
                    "$inc": {"access_count": 1},
                },
            )
            return "reinforced"

        if similarity > self.config.merge_threshold:
            # Create new memory immediately for searchability,
            # queue async merge via enrichment worker
            now = datetime.now(timezone.utc)
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
