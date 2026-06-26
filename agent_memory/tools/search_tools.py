"""MCP Search Tools — hybrid search and web search."""

import asyncio
import time

from agent_memory.core.registry import ServiceRegistry


def register_search_tools(mcp):
    """Register search MCP tools on the FastMCP server."""

    @mcp.tool(
        name="hybrid_search",
        description=(
            "Combined vector + full-text search over memories using "
            "MongoDB $rankFusion for Reciprocal Rank Fusion (RRF)."
        ),
    )
    async def hybrid_search(
        user_id: str,
        query: str,
        tier: list[str] | None = None,
        limit: int = 10,
        memory_type: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "hybrid_search")
        if access_err:
            return {"error": access_err}
        config = svc.config
        start = time.time()

        try:
            limit = min(limit, config.max_results_per_query)
            tiers = tier or ["stm", "ltm"]
            query_embedding = await svc.providers.embedding.generate_embedding(query)

            # Vector search filter
            vs_filter = {"user_id": user_id, "deleted_at": None, "tier": {"$in": tiers}}
            if memory_type:
                vs_filter["memory_type"] = memory_type
            if tags:
                vs_filter["tags"] = {"$all": tags}

            # Full-text search filter clauses
            fts_filter_clauses = [
                {"equals": {"path": "user_id", "value": user_id}},
                {"equals": {"path": "is_deleted", "value": False}},
            ]
            if tiers:
                fts_filter_clauses.append(
                    {"in": {"path": "tier", "value": tiers}}
                )

            # Build $rankFusion pipeline
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
                                "vectorPipeline": config.rrf_vector_weight,
                                "fullTextPipeline": config.rrf_text_weight,
                            },
                        },
                    }
                },
                {"$limit": limit},
                {
                    "$project": {
                        "embedding": 0,
                    }
                },
            ]

            memories_col = (await _get_db())["memories"]
            cursor = await memories_col.aggregate(pipeline)
            results = await cursor.to_list(None)

            # Sanitize BSON types for JSON serialization
            for r in results:
                _sanitize_doc(r)

            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "search", "hybrid_search", "success", duration_ms,
                query=query, result_count=len(results),
            )
            return {"results": results, "count": len(results)}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "search", "hybrid_search", "error", duration_ms,
                error=str(e),
            )
            raise

    @mcp.tool(
        name="search_web",
        description="Web search via Tavily API. Requires user_id for audit logging.",
    )
    async def search_web(user_id: str, query: str) -> dict:
        svc = ServiceRegistry.get()
        access_err = await svc.check_access(user_id, "search_web")
        if access_err:
            return {"error": access_err}
        start = time.time()

        if not svc.config.tavily_api_key:
            await svc.audit_service.log(
                user_id, "search", "search_web", "error", 0,
                error="Tavily API key not configured",
            )
            return {"error": "Web search service unavailable: Tavily API key not configured"}

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=svc.config.tavily_api_key)
            response = await asyncio.to_thread(client.search, query)

            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "search", "search_web", "success", duration_ms,
                query=query,
            )
            return {"results": response.get("results", []), "query": query}
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            await svc.audit_service.log(
                user_id, "search", "search_web", "error", duration_ms,
                error=str(e),
            )
            raise


async def _get_db():
    """Get the DatabaseManager instance."""
    from agent_memory.core.database import DatabaseManager

    return (await DatabaseManager.get_instance()).db


def _sanitize_doc(doc: dict) -> None:
    """Convert BSON types (ObjectId, datetime) to JSON-safe strings in place."""
    from bson import ObjectId
    from datetime import datetime

    for key, val in list(doc.items()):
        if isinstance(val, ObjectId):
            doc[key] = str(val)
        elif isinstance(val, datetime):
            doc[key] = val.isoformat()
        elif isinstance(val, dict):
            _sanitize_doc(val)
