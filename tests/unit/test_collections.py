"""Tests for collections.py index definitions."""

import pytest

from agent_memory.core.collections import (
    AUDIT_LOG,
    MEMORIES,
    SEARCH_INDEXES,
    SEMANTIC_CACHE,
    STANDARD_INDEXES,
)


def _search_index(name: str) -> dict:
    """The one search index called ``name``, or a failure that says which is missing.

    This was `[i for i in SEARCH_INDEXES if i["name"] == name][0]`, which raises a
    bare `IndexError` on an empty list — so renaming an index failed several tests
    at once with nothing in the output naming the index that had gone.
    """
    matches = [i for i in SEARCH_INDEXES if i["name"] == name]
    assert matches, (
        f"no search index named {name!r}; SEARCH_INDEXES has "
        f"{sorted(i['name'] for i in SEARCH_INDEXES)}"
    )
    return matches[0]


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


class TestTheSearchIndexHelper:
    """The helper's whole purpose is its failure message, so test that.

    Without this, deleting the `assert matches` line leaves every test still
    passing — the diagnostic would silently revert to the bare `IndexError` it
    exists to replace, and nothing would say so.
    """

    def test_a_known_index_is_returned(self):
        assert _search_index("memories_fts_index")["name"] == "memories_fts_index"

    def test_a_missing_index_names_itself_and_the_alternatives(self):
        with pytest.raises(AssertionError) as exc:
            _search_index("memories_no_such_index")
        message = str(exc.value)
        assert "memories_no_such_index" in message, (
            "the failure must name the index that is missing, which is the only "
            "reason this helper exists rather than a bare [0]"
        )
        assert "memories_fts_index" in message, (
            "the failure must list what *is* defined, so a rename shows as a "
            "rename rather than as an absence"
        )


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
        idx = _search_index("memories_vector_index")
        fields = idx["definition"]["fields"]
        filter_paths = {f["path"] for f in fields if f["type"] == "filter"}
        assert "user_id" in filter_paths
        assert "tier" in filter_paths
        assert "deleted_at" in filter_paths
        # `recall` and `hybrid_search` both pre-filter on these. The stronger
        # assertion — every path the built pipelines actually use is declared —
        # lives in test_search_filter_contract.py.
        assert "memory_type" in filter_paths
        assert "tags" in filter_paths

    def test_fts_index_has_content_and_summary(self):
        idx = _search_index("memories_fts_index")
        field_names = set(idx["definition"]["mappings"]["fields"].keys())
        assert "content" in field_names
        assert "summary" in field_names

    def test_fts_index_has_token_filters(self):
        idx = _search_index("memories_fts_index")
        fields = idx["definition"]["mappings"]["fields"]
        assert fields["user_id"]["type"] == "token"
        assert fields["tier"]["type"] == "token"
        assert fields["is_deleted"]["type"] == "token"
