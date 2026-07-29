"""Tests for the shared $rankFusion builder and BSON sanitization. REQ-E-110/112."""

from datetime import datetime, timezone

from bson import ObjectId

from agent_memory.services.memory import _sanitize_doc
from agent_memory.services.search_pipeline import (
    TEXT_BRANCH,
    VECTOR_BRANCH,
    rank_fusion_pipeline,
)


def _pipeline(**overrides):
    kwargs = {
        "query": "shellfish",
        "query_embedding": [0.1, 0.2],
        "vector_index": "episodes_vector_index",
        "fts_index": "episodes_fts_index",
        "fts_paths": ["search_text"],
        "vs_filter": {"user_id": "u1"},
        "fts_filter_clauses": [{"equals": {"path": "user_id", "value": "u1"}}],
        "limit": 5,
        "vector_weight": 1.0,
        "text_weight": 0.7,
    }
    kwargs.update(overrides)
    return rank_fusion_pipeline(**kwargs)


class TestRankFusionPipeline:
    def test_both_branches_are_present(self):
        # TC-EP-SP-001
        pipes = _pipeline()[0]["$rankFusion"]["input"]["pipelines"]
        assert set(pipes) == {VECTOR_BRANCH, TEXT_BRANCH}

    def test_vector_branch_carries_index_path_and_filter(self):
        # TC-EP-SP-002
        stage = _pipeline()[0]["$rankFusion"]["input"]["pipelines"][VECTOR_BRANCH][0]
        vs = stage["$vectorSearch"]
        assert vs["index"] == "episodes_vector_index"
        assert vs["path"] == "embedding"
        assert vs["queryVector"] == [0.1, 0.2]
        assert vs["filter"] == {"user_id": "u1"}

    def test_text_branch_carries_query_paths_and_a_limit_stage(self):
        # TC-EP-SP-003: $search has no limit option, so depth is a stage.
        branch = _pipeline()[0]["$rankFusion"]["input"]["pipelines"][TEXT_BRANCH]
        compound = branch[0]["$search"]["compound"]
        assert compound["must"][0]["text"]["path"] == ["search_text"]
        assert compound["filter"] == [{"equals": {"path": "user_id", "value": "u1"}}]
        assert branch[1] == {"$limit": 20}

    def test_the_tenant_filter_is_in_both_branches(self):
        # TC-EP-SP-004: isolation is enforced by the engine, in both branches.
        pipes = _pipeline()[0]["$rankFusion"]["input"]["pipelines"]
        assert pipes[VECTOR_BRANCH][0]["$vectorSearch"]["filter"]["user_id"] == "u1"
        assert pipes[TEXT_BRANCH][0]["$search"]["compound"]["filter"][0]["equals"][
            "value"
        ] == "u1"

    def test_weights_are_passed_through(self):
        # TC-EP-SP-005
        weights = _pipeline()[0]["$rankFusion"]["combination"]["weights"]
        assert weights == {VECTOR_BRANCH: 1.0, TEXT_BRANCH: 0.7}

    def test_limit_and_default_projection(self):
        # TC-EP-SP-006: embedding is excluded by default — it dominates the payload.
        pipeline = _pipeline()
        assert pipeline[1] == {"$limit": 5}
        assert pipeline[2] == {"$addFields": {"score": {"$meta": "score"}}}
        assert pipeline[3] == {"$project": {"embedding": 0}}

    def test_the_fused_score_is_projected(self):
        # TC-EP-SP-009: $rankFusion does not surface its own rank, so a consumer
        # that wants to explain *why* a document ranked has nothing to show.
        pipeline = _pipeline()
        assert {"$addFields": {"score": {"$meta": "score"}}} in pipeline

    def test_the_score_is_its_own_stage_not_a_projection_field(self):
        # TC-EP-SP-010: `{"score": {"$meta": "score"}}` inside an exclusion
        # $project does not error on Atlas — it silently returns null. Keeping it
        # as a separate $addFields stage is what makes the score real.
        pipeline = _pipeline()
        project = next(s for s in pipeline if "$project" in s)["$project"]
        assert "score" not in project
        assert pipeline.index({"$addFields": {"score": {"$meta": "score"}}}) < (
            pipeline.index({"$project": project})
        )

    def test_projection_is_overridable(self):
        # TC-EP-SP-007
        pipeline = _pipeline(project={"embedding": 0, "search_text": 0})
        assert pipeline[3] == {"$project": {"embedding": 0, "search_text": 0}}

    def test_branch_depth_is_tunable(self):
        # TC-EP-SP-008
        pipeline = _pipeline(branch_limit=50, num_candidates=500)
        pipes = pipeline[0]["$rankFusion"]["input"]["pipelines"]
        assert pipes[VECTOR_BRANCH][0]["$vectorSearch"]["limit"] == 50
        assert pipes[VECTOR_BRANCH][0]["$vectorSearch"]["numCandidates"] == 500
        assert pipes[TEXT_BRANCH][1] == {"$limit": 50}


class TestSanitizeDoc:
    """REQ-E-112: recurse into lists, not just dicts."""

    def test_top_level_bson_is_coerced(self):
        # TC-EP-SP-010
        oid = ObjectId()
        ts = datetime(2026, 8, 4, 11, tzinfo=timezone.utc)
        doc = {"_id": oid, "ts": ts, "user_id": "u1"}
        _sanitize_doc(doc)
        assert doc == {"_id": str(oid), "ts": ts.isoformat(), "user_id": "u1"}

    def test_nested_dicts_are_coerced(self):
        # TC-EP-SP-011
        oid = ObjectId()
        doc = {"meta": {"ref": oid}}
        _sanitize_doc(doc)
        assert doc["meta"]["ref"] == str(oid)

    def test_lists_of_dicts_are_coerced(self):
        # TC-EP-SP-012: episodic docs carry messages[]/todos[]/files_touched[].
        ts = datetime(2026, 8, 4, tzinfo=timezone.utc)
        doc = {"messages": [{"type": "ai", "at": ts}, {"type": "human", "at": ts}]}
        _sanitize_doc(doc)
        assert [m["at"] for m in doc["messages"]] == [ts.isoformat()] * 2

    def test_lists_of_scalars_survive(self):
        # TC-EP-SP-013
        doc = {"embedding": [0.1, 0.2], "tags": ["a", "b"]}
        _sanitize_doc(doc)
        assert doc == {"embedding": [0.1, 0.2], "tags": ["a", "b"]}

    def test_deeply_nested_lists_are_coerced(self):
        # TC-EP-SP-014: tool_calls sit inside messages[].
        oid = ObjectId()
        doc = {"messages": [{"tool_calls": [{"args": {"ref": oid}}]}]}
        _sanitize_doc(doc)
        assert doc["messages"][0]["tool_calls"][0]["args"]["ref"] == str(oid)
