"""Tests for logic extracted from the former MCP tools into services.

Preserves the coverage that lived in test_tools.py / test_admin_tools.py:
- MemoryService.hybrid_search (the $rankFusion RRF pipeline)
- AdminService.health / wipe_user_data
"""

from unittest.mock import AsyncMock, MagicMock

from agent_memory.config import MemoryConfig
from agent_memory.services.admin import AdminService
from agent_memory.services.memory import MemoryService


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


def _cursor(docs):
    cur = MagicMock()
    cur.to_list = AsyncMock(return_value=docs)
    return cur


class TestHybridSearch:
    async def test_builds_rankfusion_pipeline_and_sanitizes(self):
        col = MagicMock()
        col.aggregate = AsyncMock(return_value=_cursor([{"content": "hi", "score": 0.9}]))
        providers = MagicMock()
        providers.embedding = AsyncMock()
        providers.embedding.generate_embedding = AsyncMock(return_value=[0.1] * 4)
        svc = MemoryService(col, _config(), providers)

        results = await svc.hybrid_search("u1", "q", limit=5)

        assert results == [{"content": "hi", "score": 0.9}]
        pipeline = col.aggregate.call_args.args[0]
        assert "$rankFusion" in pipeline[0]
        # vector + full-text pipelines both present
        pipes = pipeline[0]["$rankFusion"]["input"]["pipelines"]
        assert "vectorPipeline" in pipes and "fullTextPipeline" in pipes

    async def test_filters_by_memory_type_and_tags(self):
        col = MagicMock()
        col.aggregate = AsyncMock(return_value=_cursor([]))
        providers = MagicMock()
        providers.embedding = AsyncMock()
        providers.embedding.generate_embedding = AsyncMock(return_value=[0.1] * 4)
        svc = MemoryService(col, _config(), providers)

        await svc.hybrid_search("u1", "q", memory_type="episodic", tags=["work"])

        vs_filter = (
            col.aggregate.call_args.args[0][0]["$rankFusion"]["input"]
            ["pipelines"]["vectorPipeline"][0]["$vectorSearch"]["filter"]
        )
        assert vs_filter["memory_type"] == "episodic"
        # Equality, not `$all` — an unsupported pre-filter operator matches nothing
        # rather than erroring. See `tag_filter`.
        assert vs_filter["tags"] == "work"


class TestAdminService:
    async def test_health_aggregates_tier_and_enrichment_stats(self):
        db = {"memories": MagicMock()}
        db["memories"].aggregate = AsyncMock(return_value=_cursor([
            {"_id": {"tier": "stm", "enrichment_status": "pending"}, "count": 2},
            {"_id": {"tier": "ltm", "enrichment_status": "done"}, "count": 3},
        ]))
        svc = AdminService(db)
        out = await svc.health("u1")
        assert out["total_memories"] == 5
        assert out["tier_stats"] == {"stm": 2, "ltm": 3}
        assert out["enrichment_stats"] == {"pending": 2, "done": 3}

    async def test_wipe_deletes_across_collections(self):
        """Every user-scoped collection, not the three it used to clear.

        Episodic turns, decisions, rate-limit records, and step counters used to
        survive a wipe that reported success — see
        ``test_admin_scope.py::TestWipeIsComplete`` for the per-collection cases.
        """
        counts = {
            "memories": 5, "semantic_cache": 2, "audit_log": 1,
            "episodes": 7, "decisions": 3, "rate_limits": 4,
            "episodes_counters": 2,
        }
        db = {}
        for name, count in counts.items():
            db[name] = MagicMock()
            db[name].delete_many = AsyncMock(
                return_value=MagicMock(deleted_count=count)
            )
        svc = AdminService(db)
        out = await svc.wipe_user_data("u1")
        assert out == {
            "user_id": "u1", "memories_deleted": 5,
            "cache_deleted": 2, "audit_deleted": 1,
            "episodes_deleted": 7, "decisions_deleted": 3,
            "rate_limits_deleted": 4, "episode_counters_deleted": 2,
            "complete": True,
        }
