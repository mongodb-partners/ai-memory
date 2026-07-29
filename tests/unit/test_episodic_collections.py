"""Tests for the episodes collection schema and config knobs. REQ-E-115."""

from agent_memory.config import MemoryConfig
from agent_memory.core.collections import (
    EPISODES,
    EPISODES_DEFAULT_TTL_SECONDS,
    STANDARD_INDEXES,
    get_search_indexes,
)


def _episode_indexes():
    return [ix for ix in STANDARD_INDEXES if ix["collection"] == EPISODES]


def _by_name(indexes, name):
    return next(ix for ix in indexes if ix["name"] == name)


class TestStandardIndexes:
    def test_all_five_btree_indexes_are_declared(self):
        # TC-EP-COLL-001
        names = {ix["name"] for ix in _episode_indexes()}
        assert names == {
            "ix_episodes_thread_step",
            "ix_episodes_user_ts",
            "ix_episodes_thread_ts",
            "ix_episodes_correlation",
            "ix_episodes_ttl",
        }

    def test_thread_step_index_supports_ordered_replay(self):
        # TC-EP-COLL-002
        ix = _by_name(_episode_indexes(), "ix_episodes_thread_step")
        assert ix["keys"] == [("thread_id", 1), ("step", 1)]

    def test_user_ts_index_is_descending_for_recency(self):
        # TC-EP-COLL-003
        ix = _by_name(_episode_indexes(), "ix_episodes_user_ts")
        assert ix["keys"] == [("user_id", 1), ("ts", -1)]

    def test_correlation_index_is_user_scoped(self):
        # TC-EP-COLL-004: a correlation id alone must not cross tenants.
        ix = _by_name(_episode_indexes(), "ix_episodes_correlation")
        assert ix["keys"] == [("user_id", 1), ("correlation_id", 1)]

    def test_ttl_index_expires_on_ts(self):
        # TC-EP-COLL-005
        ix = _by_name(_episode_indexes(), "ix_episodes_ttl")
        assert ix["keys"] == [("ts", 1)]
        assert ix["kwargs"]["expireAfterSeconds"] == EPISODES_DEFAULT_TTL_SECONDS
        assert EPISODES_DEFAULT_TTL_SECONDS == 30 * 86400


class TestSearchIndexes:
    def test_vector_index_uses_the_shared_embedding_path(self):
        # TC-EP-COLL-010: test_collections.py asserts every vector index uses
        # exactly one vector field at path "embedding".
        idx = next(
            i for i in get_search_indexes(1024) if i["name"] == "episodes_vector_index"
        )
        vectors = [f for f in idx["definition"]["fields"] if f["type"] == "vector"]
        assert len(vectors) == 1
        assert vectors[0]["path"] == "embedding"
        assert vectors[0]["numDimensions"] == 1024

    def test_vector_index_declares_every_prefilter_field(self):
        # TC-EP-COLL-011: an undeclared filter field makes the branch return
        # nothing, silently.
        idx = next(
            i for i in get_search_indexes() if i["name"] == "episodes_vector_index"
        )
        filters = {f["path"] for f in idx["definition"]["fields"] if f["type"] == "filter"}
        assert filters == {"user_id", "thread_id", "agent_name"}

    def test_fts_index_analyzes_search_text_and_tokenizes_scopes(self):
        # TC-EP-COLL-012: an analyzed field cannot back an exact equals filter.
        idx = next(
            i for i in get_search_indexes() if i["name"] == "episodes_fts_index"
        )
        fields = idx["definition"]["mappings"]["fields"]
        assert idx["definition"]["mappings"]["dynamic"] is False
        assert fields["search_text"] == {"type": "string"}
        assert fields["user_id"] == {"type": "token"}
        assert fields["thread_id"] == {"type": "token"}
        assert fields["agent_name"] == {"type": "token"}

    def test_both_vector_indexes_share_one_dimension(self):
        # TC-EP-COLL-013: ProviderManager mutates embedding_dimension in place,
        # so the two indexes must be generated from the same value.
        indexes = get_search_indexes(1024)
        dims = {
            i["name"]: f["numDimensions"]
            for i in indexes
            if i["type"] == "vectorSearch"
            for f in i["definition"]["fields"]
            if f["type"] == "vector"
        }
        assert set(dims.values()) == {1024}
        assert "episodes_vector_index" in dims
        assert "memories_vector_index" in dims


class TestEpisodicConfig:
    def _config(self, **overrides) -> MemoryConfig:
        defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
        defaults.update(overrides)
        return MemoryConfig(**defaults, _env_file=None)

    def test_defaults(self):
        # TC-EP-COLL-020
        cfg = self._config()
        assert cfg.episodic_enabled is True
        assert cfg.episodic_queue_size == 1000
        assert cfg.episodic_batch_size == 20
        assert cfg.episodic_flush_interval_seconds == 1.0
        assert cfg.episodic_content_cap == 4000
        assert cfg.episodic_search_text_cap == 2000
        assert cfg.episodic_embed_final_steps_only is True

    def test_knobs_are_overridable(self):
        # TC-EP-COLL-021
        cfg = self._config(episodic_enabled=False, episodic_queue_size=10)
        assert cfg.episodic_enabled is False
        assert cfg.episodic_queue_size == 10

    def test_search_text_cap_is_below_the_content_cap(self):
        # TC-EP-COLL-022: embedding cost is per token; the embedded slice should
        # be the smaller of the two by default.
        cfg = self._config()
        assert cfg.episodic_search_text_cap < cfg.episodic_content_cap
