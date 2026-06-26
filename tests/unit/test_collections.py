"""Tests for collections.py index definitions."""

import pytest

from agent_memory.core.collections import (
    MEMORIES,
    SEMANTIC_CACHE,
    AUDIT_LOG,
    STANDARD_INDEXES,
    SEARCH_INDEXES,
)


class TestCollectionNames:
    """REQ-DB-005: Collection name constants exist."""

    def test_memories_constant(self):
        assert MEMORIES == "memories"

    def test_semantic_cache_constant(self):
        assert SEMANTIC_CACHE == "semantic_cache"

    def test_audit_log_constant(self):
        assert AUDIT_LOG == "audit_log"


class TestStandardIndexes:
    """REQ-DB-001: Standard index definitions for Phase 0."""

    def test_standard_indexes_is_list(self):
        assert isinstance(STANDARD_INDEXES, list)

    def test_each_index_has_required_keys(self):
        for idx in STANDARD_INDEXES:
            assert "collection" in idx, f"Missing 'collection': {idx}"
            assert "keys" in idx, f"Missing 'keys': {idx}"
            assert "name" in idx, f"Missing 'name': {idx}"

    def test_memories_has_ttl_expires_at(self):
        """memories.expires_at TTL index."""
        ttl_idx = [i for i in STANDARD_INDEXES
                   if i["collection"] == MEMORIES
                   and i["name"] == "ix_memories_expires_at"]
        assert len(ttl_idx) == 1
        assert "expireAfterSeconds" in ttl_idx[0].get("kwargs", {})

    def test_memories_has_enrichment_queue_index(self):
        """memories enrichment_status + created_at compound index."""
        idx = [i for i in STANDARD_INDEXES
               if i["collection"] == MEMORIES
               and i["name"] == "ix_memories_enrichment_queue"]
        assert len(idx) == 1

    def test_memories_has_user_tier_created_index(self):
        """memories user_id + tier + created_at compound with partial filter."""
        idx = [i for i in STANDARD_INDEXES
               if i["collection"] == MEMORIES
               and i["name"] == "ix_memories_user_tier_created"]
        assert len(idx) == 1
        kwargs = idx[0].get("kwargs", {})
        assert "partialFilterExpression" in kwargs

    def test_memories_has_conversation_index(self):
        """memories user_id + conversation_id index."""
        idx = [i for i in STANDARD_INDEXES
               if i["collection"] == MEMORIES
               and i["name"] == "ix_memories_conversation"]
        assert len(idx) == 1

    def test_memories_has_deleted_at_ttl(self):
        """memories.deleted_at TTL index for soft-delete purge."""
        idx = [i for i in STANDARD_INDEXES
               if i["collection"] == MEMORIES
               and i["name"] == "ix_memories_deleted_at_ttl"]
        assert len(idx) == 1
        kwargs = idx[0].get("kwargs", {})
        assert "expireAfterSeconds" in kwargs

    def test_audit_log_has_user_timestamp_index(self):
        idx = [i for i in STANDARD_INDEXES
               if i["collection"] == AUDIT_LOG
               and i["name"] == "ix_audit_user_timestamp"]
        assert len(idx) == 1

    def test_audit_log_has_ttl_index(self):
        idx = [i for i in STANDARD_INDEXES
               if i["collection"] == AUDIT_LOG
               and i["name"] == "ix_audit_ttl"]
        assert len(idx) == 1
        kwargs = idx[0].get("kwargs", {})
        assert "expireAfterSeconds" in kwargs

    def test_cache_has_ttl_index(self):
        idx = [i for i in STANDARD_INDEXES
               if i["collection"] == SEMANTIC_CACHE
               and i["name"] == "ix_cache_ttl"]
        assert len(idx) == 1


class TestSearchIndexes:
    """REQ-DB-002: Atlas Search index definitions."""

    def test_search_indexes_is_list(self):
        assert isinstance(SEARCH_INDEXES, list)

    def test_each_search_index_has_required_keys(self):
        for idx in SEARCH_INDEXES:
            assert "collection" in idx
            assert "name" in idx
            assert "type" in idx
            assert "definition" in idx

    def test_memories_vector_index_exists(self):
        idx = [i for i in SEARCH_INDEXES if i["name"] == "memories_vector_index"]
        assert len(idx) == 1
        assert idx[0]["type"] == "vectorSearch"
        assert idx[0]["collection"] == MEMORIES

    def test_memories_fts_index_exists(self):
        idx = [i for i in SEARCH_INDEXES if i["name"] == "memories_fts_index"]
        assert len(idx) == 1
        assert idx[0]["type"] == "search"
        assert idx[0]["collection"] == MEMORIES

    def test_cache_vector_index_exists(self):
        idx = [i for i in SEARCH_INDEXES if i["name"] == "cache_vector_index"]
        assert len(idx) == 1
        assert idx[0]["type"] == "vectorSearch"
        assert idx[0]["collection"] == SEMANTIC_CACHE

    def test_vector_index_has_embedding_field(self):
        for idx in SEARCH_INDEXES:
            if idx["type"] == "vectorSearch":
                fields = idx["definition"]["fields"]
                vector_fields = [f for f in fields if f["type"] == "vector"]
                assert len(vector_fields) == 1
                assert vector_fields[0]["path"] == "embedding"

    def test_memories_vector_has_filter_fields(self):
        idx = [i for i in SEARCH_INDEXES if i["name"] == "memories_vector_index"][0]
        fields = idx["definition"]["fields"]
        filter_paths = {f["path"] for f in fields if f["type"] == "filter"}
        assert "user_id" in filter_paths
        assert "tier" in filter_paths
        assert "deleted_at" in filter_paths

    def test_fts_index_has_content_and_summary(self):
        idx = [i for i in SEARCH_INDEXES if i["name"] == "memories_fts_index"][0]
        field_names = set(idx["definition"]["mappings"]["fields"].keys())
        assert "content" in field_names
        assert "summary" in field_names

    def test_fts_index_has_token_filters(self):
        idx = [i for i in SEARCH_INDEXES if i["name"] == "memories_fts_index"][0]
        fields = idx["definition"]["mappings"]["fields"]
        assert fields["user_id"]["type"] == "token"
        assert fields["tier"]["type"] == "token"
        assert fields["is_deleted"]["type"] == "token"
