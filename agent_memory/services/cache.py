"""Semantic cache service — check, store, invalidate (hard delete)."""

from datetime import datetime, timezone

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import EmbeddingProvider


class CacheService:
    """Replaces the HTTP proxy to the semantic-cache microservice."""

    def __init__(self, cache_collection, config: MCPConfig, embedding_provider: EmbeddingProvider) -> None:
        self.cache = cache_collection
        self.config = config
        self.embedding = embedding_provider

    async def check(
        self,
        user_id: str,
        query: str,
        similarity_threshold: float | None = None,
    ) -> dict | None:
        """Vector search for a semantically similar cached query."""
        threshold = similarity_threshold or self.config.cache_similarity_threshold
        query_embedding = await self.embedding.generate_embedding(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "cache_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 10,
                    "limit": 1,
                    "filter": {"user_id": user_id},
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        cursor = await self.cache.aggregate(pipeline)
        results = await cursor.to_list(None)

        if results and results[0]["score"] >= threshold:
            return {
                "query": results[0]["query"],
                "response": results[0]["response"],
                "score": results[0]["score"],
                "cache_hit": True,
            }
        return None

    async def store(self, user_id: str, query: str, response: str) -> str:
        """Cache a query-response pair with embedding for future similarity lookup."""
        embedding = await self.embedding.generate_embedding(query)
        doc = {
            "user_id": user_id,
            "query": query,
            "response": response,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc),
        }
        result = await self.cache.insert_one(doc)
        return str(result.inserted_id)

    async def invalidate(
        self,
        user_id: str,
        pattern: str | None = None,
        invalidate_all: bool = False,
    ) -> int:
        """Hard-delete cached entries. No soft-delete for cache."""
        if invalidate_all:
            result = await self.cache.delete_many({"user_id": user_id})
        elif pattern:
            result = await self.cache.delete_many(
                {"user_id": user_id, "query": {"$regex": pattern}}
            )
        else:
            return 0
        return result.deleted_count
