"""Tests for MemoryService (store_stm, recall, delete, evolve)."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.providers.manager import ResolvedEmbedding
from agent_memory.services.memory import MemoryService


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_providers():
    providers = MagicMock()
    providers.embedding = AsyncMock()
    providers.embedding.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    providers.embedding.generate_embeddings_batch = AsyncMock(
        side_effect=lambda texts: [[0.1] * 1536 for _ in texts]
    )
    # The real `ProviderManager` publishes this and `MemoryService` reads it to
    # know how wide a vector should be. A bare `MagicMock` attribute would answer
    # with another mock, which reads as "no declared width" and would leave the
    # dimension check untested for every case below.
    providers.embedding_spec = ResolvedEmbedding(
        model="amazon.titan-embed-text-v2:0", dimension=1536
    )
    return providers


def _make_collection():
    col = AsyncMock()
    return col


class TestMemoryServiceStoreStm:
    """TC-020..TC-023: store_stm behavior."""

    async def test_store_single_message(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        col.insert_many = AsyncMock(return_value=MagicMock(inserted_ids=[stm_id]))

        messages = [{"content": "Hello", "message_type": "human"}]
        result = await service.store_stm("user1", "conv1", messages)

        assert result == [str(stm_id)]
        providers.embedding.generate_embeddings_batch.assert_called_once()

    async def test_store_creates_ltm_for_long_human_messages(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        col.insert_many = AsyncMock(return_value=MagicMock(inserted_ids=[stm_id]))

        # Comfortably over `ltm_candidate_min_chars`; the boundary itself is
        # pinned in test_ltm_candidate_threshold.py.
        messages = [{"content": "A" * 50, "message_type": "human"}]
        result = await service.store_stm("user1", "conv1", messages)

        # Two insert_many calls: one for STM, one for LTM
        assert col.insert_many.call_count == 2
        ltm_docs = col.insert_many.call_args_list[1][0][0]
        assert ltm_docs[0]["tier"] == "ltm"
        assert ltm_docs[0]["enrichment_status"] == "pending"
        assert ltm_docs[0]["source_stm_id"] == stm_id

    async def test_store_no_ltm_for_short_messages(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        col.insert_many = AsyncMock(return_value=MagicMock(inserted_ids=[stm_id]))

        messages = [{"content": "Hi", "message_type": "human"}]  # under the threshold
        await service.store_stm("user1", "conv1", messages)

        # Only one insert_many call (STM only)
        assert col.insert_many.call_count == 1

    async def test_store_no_ltm_for_ai_messages(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        col.insert_many = AsyncMock(return_value=MagicMock(inserted_ids=[stm_id]))

        messages = [{"content": "A" * 50, "message_type": "ai"}]
        await service.store_stm("user1", "conv1", messages)

        assert col.insert_many.call_count == 1

    async def test_store_empty_messages_returns_empty(self):
        """REQ-EC-001: Empty messages returns empty list."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        result = await service.store_stm("user1", "conv1", [])
        assert result == []
        col.insert_many.assert_not_called()

    async def test_store_sets_is_deleted_false(self):
        """REQ-004: is_deleted and deleted_at set on new documents."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        col.insert_many = AsyncMock(return_value=MagicMock(inserted_ids=[stm_id]))

        messages = [{"content": "test content", "message_type": "human"}]
        await service.store_stm("user1", "conv1", messages)

        inserted_docs = col.insert_many.call_args_list[0][0][0]
        assert inserted_docs[0]["deleted_at"] is None
        assert inserted_docs[0]["is_deleted"] is False


class TestMemoryServiceBaseFilter:
    """TC-024: _base_filter injects user_id and deleted_at."""

    def test_base_filter_includes_user_and_deleted(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        f = service._base_filter("user123")
        assert f["user_id"] == "user123"
        assert f["deleted_at"] is None

    def test_base_filter_includes_extra_filters(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        f = service._base_filter("user123", tier="stm")
        assert f["tier"] == "stm"


class TestMemoryServiceDelete:
    """TC-028..TC-031: delete behavior."""

    async def test_delete_by_id(self):
        """REQ-010: Soft-delete by memory_id."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        memory_id = str(ObjectId())
        col.update_many = AsyncMock(return_value=MagicMock(modified_count=1))

        result = await service.delete("user1", memory_id=memory_id)
        assert result["deleted_count"] == 1
        col.update_many.assert_called_once()

    async def test_delete_dry_run(self):
        """REQ-012: dry_run returns count without modifying."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        col.count_documents = AsyncMock(return_value=5)

        result = await service.delete(
            "user1", tags=["topic:test"], confirm=True, dry_run=True
        )
        assert result["deleted_count"] == 5
        assert result["dry_run"] is True
        col.update_many.assert_not_called()

    async def test_delete_bulk_requires_confirm(self):
        """REQ-013: Bulk delete requires confirm=true."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        with pytest.raises(ValueError, match="confirm"):
            await service.delete("user1", tags=["topic:test"], confirm=False)

    async def test_delete_nonexistent_returns_zero(self):
        """REQ-EC-005: Non-existent memory_id returns deleted_count: 0."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        col.update_many = AsyncMock(return_value=MagicMock(modified_count=0))

        result = await service.delete("user1", memory_id=str(ObjectId()))
        assert result["deleted_count"] == 0

    async def test_delete_sets_is_deleted_true(self):
        """Soft-delete sets both deleted_at and is_deleted."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        col.update_many = AsyncMock(return_value=MagicMock(modified_count=1))

        await service.delete("user1", memory_id=str(ObjectId()))
        update_arg = col.update_many.call_args[0][1]
        assert "is_deleted" in update_arg["$set"]
        assert update_arg["$set"]["is_deleted"] is True
        assert "deleted_at" in update_arg["$set"]


class TestMemoryServiceRecall:
    """TC-025..TC-027: recall behavior."""

    async def test_recall_returns_results(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_id = ObjectId()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {
                "_id": mock_id,
                "user_id": "user1",
                "content": "test memory",
                "importance": 0.7,
                "access_count": 2,
                "created_at": datetime.now(timezone.utc) - timedelta(days=1),
                "tier": "ltm",
                "vs_score": 0.9,
            }
        ])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.update_many = AsyncMock()

        result = await service.recall("user1", "test query")
        assert len(result) == 1
        assert result[0]["content"] == "test memory"

    async def test_recall_empty_results(self):
        """REQ-EC-003: No results returns empty list."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        result = await service.recall("user1", "no matches here")
        assert result == []

    async def test_recall_increments_access_count(self):
        """REQ-007: Increment access_count on returned results."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_id = ObjectId()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {
                "_id": mock_id,
                "user_id": "user1",
                "content": "test memory",
                "importance": 0.7,
                "access_count": 2,
                "created_at": datetime.now(timezone.utc),
                "tier": "ltm",
                "vs_score": 0.9,
            }
        ])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.update_many = AsyncMock()

        await service.recall("user1", "test query")
        col.update_many.assert_called_once()


class TestRecallCalibratedRanking:
    """REQ-E-001..REQ-E-003: Calibrated 3-component ranking formula."""

    async def test_recall_reranks_by_calibrated_score(self):
        """REQ-E-001: Recent + important memory should outrank old + unimportant
        even if vector similarity is slightly lower."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        now = datetime.now(timezone.utc)
        # Doc A: old (30 days), low importance, high vs_score
        doc_a = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "old memory",
            "importance": 0.2,
            "access_count": 0,
            "created_at": now - timedelta(days=30),
            "tier": "ltm",
            "vs_score": 0.95,
        }
        # Doc B: recent (1 day), high importance, slightly lower vs_score
        doc_b = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "recent important memory",
            "importance": 0.9,
            "access_count": 5,
            "created_at": now - timedelta(days=1),
            "tier": "ltm",
            "vs_score": 0.85,
        }

        # $vectorSearch returns A first (higher vs_score)
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[doc_a, doc_b])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.update_many = AsyncMock()

        results = await service.recall("user1", "test query")

        # After calibrated ranking, B should outrank A:
        #   B: recency=exp(-1/30)≈0.967, importance boosted by ln(6)≈1.79, relevance=0.85
        #   A: recency=exp(-30/30)≈0.368, importance low+no access, relevance=0.95
        assert len(results) == 2
        assert results[0]["content"] == "recent important memory"
        assert results[1]["content"] == "old memory"

    async def test_recall_uses_config_weights(self):
        """REQ-E-002: Ranking uses alpha/beta/gamma from MCPConfig."""
        col = _make_collection()
        # Override: set gamma=1.0, alpha=0.0, beta=0.0 => pure relevance ranking
        config = _make_config(ranking_alpha=0.0, ranking_beta=0.0, ranking_gamma=1.0)
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        now = datetime.now(timezone.utc)
        # Doc A: high vs_score, old, unimportant
        doc_a = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "high relevance",
            "importance": 0.1,
            "access_count": 0,
            "created_at": now - timedelta(days=60),
            "tier": "ltm",
            "vs_score": 0.99,
        }
        # Doc B: low vs_score, recent, important
        doc_b = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "low relevance",
            "importance": 1.0,
            "access_count": 100,
            "created_at": now - timedelta(minutes=5),
            "tier": "ltm",
            "vs_score": 0.60,
        }

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[doc_a, doc_b])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.update_many = AsyncMock()

        results = await service.recall("user1", "test query")

        # With gamma=1.0 only, pure relevance: A (0.99) beats B (0.60)
        assert results[0]["content"] == "high relevance"

    async def test_recall_score_normalization(self):
        """REQ-E-003: All score components normalized to [0, 1]."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        now = datetime.now(timezone.utc)
        doc = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "test",
            "importance": 1.0,
            "access_count": 1000,  # Very high access count
            "created_at": now,     # Very recent
            "tier": "ltm",
            "vs_score": 1.0,      # Perfect similarity
        }

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[doc])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.update_many = AsyncMock()

        results = await service.recall("user1", "test query")
        assert len(results) == 1
        # Verify that final_score is included and bounded
        # final_score should be <= 1.0 (all components are [0,1], weights sum to 1.0)
        assert "final_score" in results[0]
        assert 0 <= results[0]["final_score"] <= 1.0

    async def test_recall_strips_internal_scores(self):
        """Recall should strip vs_score but expose final_score."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        now = datetime.now(timezone.utc)
        doc = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "test",
            "importance": 0.5,
            "access_count": 0,
            "created_at": now,
            "tier": "ltm",
            "vs_score": 0.8,
        }

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[doc])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.update_many = AsyncMock()

        results = await service.recall("user1", "test query")

        assert "vs_score" not in results[0]
        assert "embedding" not in results[0]
        assert "final_score" in results[0]


class TestEvolveMemoryMerge:
    """REQ-E-004: Evolve memory merge branch creates new memory and queues merge."""

    async def test_merge_creates_memory_and_returns_merge_queued(self):
        """REQ-E-004: When merge_threshold < similarity <= reinforce_threshold,
        create new memory with merge_pending status."""
        col = _make_collection()
        config = _make_config(reinforce_threshold=0.85, merge_threshold=0.70)
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        existing_id = ObjectId()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {
                "_id": existing_id,
                "user_id": "user1",
                "content": "existing LTM memory",
                "importance": 0.6,
                "score": 0.78,  # Between 0.70 and 0.85 = merge range
                "tier": "ltm",
            }
        ])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.insert_one = AsyncMock(return_value=MagicMock(inserted_id=ObjectId()))

        result = await service.evolve_memory(
            "user1", "new content to merge", [0.1] * 1536,
        )

        assert result == "merge_queued"
        # Should have inserted a new memory with merge_pending status
        col.insert_one.assert_called_once()
        inserted_doc = col.insert_one.call_args[0][0]
        assert inserted_doc["enrichment_status"] == "merge_pending"
        assert inserted_doc["merge_target_id"] == existing_id
        assert inserted_doc["content"] == "new content to merge"
        assert inserted_doc["user_id"] == "user1"
        assert inserted_doc["tier"] == "ltm"

    async def test_merge_does_not_create_memory_for_reinforce(self):
        """INV: Reinforce branch should NOT create a new memory."""
        col = _make_collection()
        config = _make_config(reinforce_threshold=0.85)
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {
                "_id": ObjectId(),
                "user_id": "user1",
                "content": "existing",
                "importance": 0.6,
                "score": 0.90,  # Above reinforce threshold
                "tier": "ltm",
            }
        ])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.update_one = AsyncMock()

        result = await service.evolve_memory("user1", "similar content", [0.1] * 1536)

        assert result == "reinforced"
        col.insert_one.assert_not_called()


class TestSanitizeDoc:
    """_sanitize_doc converts BSON types to JSON-safe strings."""

    def test_converts_objectid(self):
        from agent_memory.services.memory import _sanitize_doc
        oid = ObjectId()
        doc = {"_id": oid}
        _sanitize_doc(doc)
        assert doc["_id"] == str(oid)

    def test_converts_datetime(self):
        from agent_memory.services.memory import _sanitize_doc
        dt = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        doc = {"ts": dt}
        _sanitize_doc(doc)
        assert doc["ts"] == dt.isoformat()

    def test_converts_nested_dict(self):
        from agent_memory.services.memory import _sanitize_doc
        oid = ObjectId()
        doc = {"nested": {"_id": oid}}
        _sanitize_doc(doc)
        assert doc["nested"]["_id"] == str(oid)


class TestStoreStmLtmFailure:
    """LTM insert failure is caught and logged."""

    async def test_ltm_insert_failure_still_returns_stm_ids(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        call_count = [0]
        async def mock_insert_many(docs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(inserted_ids=[stm_id])
            raise Exception("LTM insert failed")

        col.insert_many = mock_insert_many

        messages = [{"content": "A" * 50, "message_type": "human"}]
        result = await service.store_stm("user1", "conv1", messages)
        assert result == [str(stm_id)]


class TestRecallFilters:
    """recall() applies tier, memory_type, tags filters."""

    async def test_recall_with_tier_filter(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        await service.recall("user1", "query", tier=["stm"])
        pipeline = col.aggregate.call_args[0][0]
        vs_filter = pipeline[0]["$vectorSearch"]["filter"]
        assert vs_filter["tier"] == {"$in": ["stm"]}

    async def test_recall_with_memory_type_filter(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        await service.recall("user1", "query", memory_type="factual")
        pipeline = col.aggregate.call_args[0][0]
        vs_filter = pipeline[0]["$vectorSearch"]["filter"]
        assert vs_filter["memory_type"] == "factual"

    async def test_recall_with_tags_filter(self):
        """A single tag is a plain equality, not `$all`.

        This assertion used to require `{"$all": [...]}`, which is the defect: a
        `$vectorSearch` pre-filter does not support `$all` and does not reject it
        either, so the branch matched nothing and every tag-filtered recall came
        back empty. An array `filter` field matches when any element matches, so
        equality is both correct and supported.
        """
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        await service.recall("user1", "query", tags=["topic:test"])
        pipeline = col.aggregate.call_args[0][0]
        vs_filter = pipeline[0]["$vectorSearch"]["filter"]
        assert vs_filter["tags"] == "topic:test"
        assert "$all" not in str(vs_filter)


class TestDeduplication:
    """_deduplicate handles LTM/STM pair suppression."""

    def test_ltm_replaces_lower_score_stm(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        results = [
            {"_id": stm_id, "content": "stm", "vs_score": 0.7, "source_stm_id": None},
            {"_id": ObjectId(), "content": "ltm", "vs_score": 0.9, "source_stm_id": stm_id},
        ]
        deduped = service._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["content"] == "ltm"

    def test_stm_kept_when_higher_score(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        results = [
            {"_id": stm_id, "content": "stm", "vs_score": 0.95, "source_stm_id": None},
            {"_id": ObjectId(), "content": "ltm", "vs_score": 0.7, "source_stm_id": stm_id},
        ]
        deduped = service._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["content"] == "stm"

    def test_stm_stm_duplicate_keeps_higher_score(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        same_id = ObjectId()
        results = [
            {"_id": same_id, "content": "first", "vs_score": 0.6, "source_stm_id": None},
            {"_id": same_id, "content": "second", "vs_score": 0.9, "source_stm_id": None},
        ]
        deduped = service._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["vs_score"] == 0.9

    def test_ltm_first_then_stm_keeps_higher_ltm(self):
        """LTM appears before its STM counterpart — LTM-first dedup path."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        results = [
            {"_id": ObjectId(), "content": "ltm", "vs_score": 0.9, "source_stm_id": stm_id},
            {"_id": stm_id, "content": "stm", "vs_score": 0.7, "source_stm_id": None},
        ]
        deduped = service._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["content"] == "ltm"

    def test_ltm_first_then_stm_keeps_higher_stm(self):
        """LTM appears first but STM has higher score — STM replaces LTM."""
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        stm_id = ObjectId()
        results = [
            {"_id": ObjectId(), "content": "ltm", "vs_score": 0.5, "source_stm_id": stm_id},
            {"_id": stm_id, "content": "stm", "vs_score": 0.9, "source_stm_id": None},
        ]
        deduped = service._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0]["content"] == "stm"


class TestDeleteTimeRange:
    """delete() with time_range filter."""

    async def test_delete_with_time_range(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        col.update_many = AsyncMock(return_value=MagicMock(modified_count=2))

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 1, tzinfo=timezone.utc)
        result = await service.delete(
            "user1", time_range={"start": start, "end": end}, confirm=True,
        )
        assert result["deleted_count"] == 2
        query_filter = col.update_many.call_args[0][0]
        assert query_filter["created_at"]["$gte"] == start
        assert query_filter["created_at"]["$lte"] == end


class TestEvolveCreated:
    """evolve_memory returns 'created' when no similar memories or below threshold."""

    async def test_evolve_no_similar_returns_created(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        result = await service.evolve_memory("user1", "content", [0.1] * 1536)
        assert result == "created"

    async def test_evolve_below_merge_threshold_returns_created(self):
        col = _make_collection()
        config = _make_config(merge_threshold=0.70)
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"_id": ObjectId(), "score": 0.50, "importance": 0.5}
        ])
        col.aggregate = AsyncMock(return_value=mock_cursor)

        result = await service.evolve_memory("user1", "content", [0.1] * 1536)
        assert result == "created"


class TestCalibratedRankNaiveDatetime:
    """_calibrated_rank handles created_at without tzinfo."""

    async def test_naive_datetime_gets_utc_attached(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        service = MemoryService(col, config, providers)

        now = datetime.now(timezone.utc)
        # Naive datetime (no tzinfo)
        doc = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "test",
            "importance": 0.5,
            "access_count": 0,
            "created_at": datetime(2025, 1, 1, 12, 0, 0),  # no tzinfo
            "tier": "ltm",
            "vs_score": 0.8,
        }

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[doc])
        col.aggregate = AsyncMock(return_value=mock_cursor)
        col.update_many = AsyncMock()

        results = await service.recall("user1", "test query")
        assert len(results) == 1
        assert "final_score" in results[0]


class TestRetentionTTL:
    """_retention_ttl returns correct timedelta per tier."""

    def _make_service(self, **config_overrides):
        col = _make_collection()
        config = _make_config(**config_overrides)
        providers = _make_providers()
        return MemoryService(col, config, providers)

    def test_critical_tier(self):
        service = self._make_service(ltm_retention_critical_days=365)
        assert service._retention_ttl("critical") == timedelta(days=365)

    def test_reference_tier(self):
        service = self._make_service(ltm_retention_reference_days=180)
        assert service._retention_ttl("reference") == timedelta(days=180)

    def test_standard_tier(self):
        service = self._make_service(ltm_retention_standard_days=90)
        assert service._retention_ttl("standard") == timedelta(days=90)

    def test_temporary_tier(self):
        service = self._make_service(ltm_retention_temporary_days=7)
        assert service._retention_ttl("temporary") == timedelta(days=7)

    def test_ephemeral_tier(self):
        service = self._make_service(stm_ttl_hours=24)
        assert service._retention_ttl("ephemeral") == timedelta(hours=24)

    def test_unknown_tier_falls_back_to_standard(self):
        service = self._make_service(ltm_retention_standard_days=90)
        assert service._retention_ttl("unknown_tier") == timedelta(days=90)
