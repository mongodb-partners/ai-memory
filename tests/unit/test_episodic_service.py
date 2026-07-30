"""Tests for EpisodicService. REQ-E-100..116."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from bson import ObjectId

from agent_memory.config import MemoryConfig
from agent_memory.services.episodic import EpisodicService


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


def _cursor(docs):
    cur = MagicMock()
    cur.sort = MagicMock(return_value=cur)
    cur.limit = MagicMock(return_value=cur)
    cur.to_list = AsyncMock(return_value=docs)
    return cur


def _collection(docs=None):
    col = MagicMock()
    col.aggregate = AsyncMock(return_value=_cursor(docs or []))
    col.find = MagicMock(return_value=_cursor(docs or []))
    col.name = "episodes"
    col.database = MagicMock()
    col.database.command = AsyncMock()
    col.create_index = AsyncMock()
    col.drop_index = AsyncMock()
    return col


def _providers(embedding=None):
    providers = MagicMock()
    providers.embedding = MagicMock()
    providers.embedding.generate_embedding = AsyncMock(
        return_value=embedding if embedding is not None else [0.1, 0.2]
    )
    return providers


def _service(collection=None, *, providers=None, **config_overrides):
    worker = MagicMock()
    worker.enqueue = MagicMock()
    worker.flush = AsyncMock(return_value=True)
    worker.close = AsyncMock(return_value=True)
    worker.stats = MagicMock(return_value={"written": 0})
    svc = EpisodicService(
        collection if collection is not None else _collection(),
        _config(**config_overrides),
        providers if providers is not None else _providers(),
        counter_collection=MagicMock(),
        worker=worker,
    )
    return svc


def _turn(question="What should I make Friday?", answer="A no-shellfish menu."):
    """A complete two-message final-step turn."""
    return [
        SimpleNamespace(type="human", content=question, tool_calls=[]),
        SimpleNamespace(type="ai", content=answer, tool_calls=[]),
    ]


class TestLogActivityDocument:
    """REQ-E-100: the projected document shape."""

    def test_all_top_level_fields_are_present(self):
        # TC-EP-SVC-001
        svc = _service()
        assert svc.log_activity("u1", "t1", _turn()) is True

        doc = svc.worker.enqueue.call_args.args[0]
        assert set(doc) == {
            "user_id",
            "thread_id",
            "conversation_id",
            "agent_name",
            "ts",
            "messages",
            "todos",
            "files_touched",
            "correlation_id",
            "__assign_step",
            "__search_text",
        }

    def test_agent_name_defaults_to_main(self):
        # TC-EP-SVC-002
        svc = _service()
        svc.log_activity("u1", "t1", _turn())
        assert svc.worker.enqueue.call_args.args[0]["agent_name"] == "main"

    def test_conversation_id_defaults_to_the_thread_id(self):
        # TC-EP-SVC-003: lets the UI join episodic to the memories tier.
        svc = _service()
        svc.log_activity("u1", "t1", _turn())
        assert svc.worker.enqueue.call_args.args[0]["conversation_id"] == "t1"

    def test_correlation_id_is_empty_string_not_none(self):
        # TC-EP-SVC-004: null would create a null bucket in the index.
        svc = _service()
        svc.log_activity("u1", "t1", _turn())
        assert svc.worker.enqueue.call_args.args[0]["correlation_id"] == ""

    def test_explicit_metadata_is_carried_through(self):
        # TC-EP-SVC-005
        svc = _service()
        svc.log_activity(
            "u1",
            "t1",
            _turn(),
            agent_name="researcher",
            correlation_id="trace-1",
            conversation_id="c1",
        )
        doc = svc.worker.enqueue.call_args.args[0]
        assert doc["agent_name"] == "researcher"
        assert doc["correlation_id"] == "trace-1"
        assert doc["conversation_id"] == "c1"

    def test_ts_defaults_to_now_utc(self):
        # TC-EP-SVC-006
        svc = _service()
        before = datetime.now(timezone.utc)
        svc.log_activity("u1", "t1", _turn())
        ts = svc.worker.enqueue.call_args.args[0]["ts"]
        assert before <= ts <= datetime.now(timezone.utc)
        assert ts.tzinfo is not None

    def test_explicit_ts_wins(self):
        # TC-EP-SVC-007: deterministic demo seeding depends on this.
        svc = _service()
        planted = datetime(2026, 8, 4, 11, tzinfo=timezone.utc)
        svc.log_activity("u1", "t1", _turn(), ts=planted)
        assert svc.worker.enqueue.call_args.args[0]["ts"] == planted

    def test_step_assignment_is_deferred_to_the_worker(self):
        # TC-EP-SVC-008: the counter round trip stays off this call.
        svc = _service()
        svc.log_activity("u1", "t1", _turn())
        doc = svc.worker.enqueue.call_args.args[0]
        # Composite, not the bare thread_id: `thread_id` is caller-supplied and
        # not namespaced, so two tenants both calling a thread "main" shared one
        # step sequence and each saw its own numbering skip.
        assert doc["__assign_step"] == {"user_id": "u1", "thread_id": "t1"}
        assert "step" not in doc

    def test_messages_are_projected_and_capped(self):
        # TC-EP-SVC-009
        svc = _service(episodic_content_cap=10)
        svc.log_activity("u1", "t1", _turn(question="x" * 50))
        messages = svc.worker.enqueue.call_args.args[0]["messages"]
        assert len(messages) == 2
        assert "original_size=50" in messages[0]["content"]

    def test_todos_are_projected(self):
        # TC-EP-SVC-010
        svc = _service()
        svc.log_activity(
            "u1", "t1", _turn(), todos=[{"id": "1", "content": "plan", "status": "bad"}]
        )
        todos = svc.worker.enqueue.call_args.args[0]["todos"]
        assert todos == [{"id": "1", "content": "plan", "status": "pending"}]

    def test_files_touched_is_derived_from_tool_calls(self):
        # TC-EP-SVC-011
        svc = _service()
        messages = [
            SimpleNamespace(type="human", content="write it", tool_calls=[]),
            SimpleNamespace(
                type="ai",
                content="",
                tool_calls=[
                    {"name": "write_file", "args": {"file_path": "menu.md", "content": "x"}}
                ],
            ),
            SimpleNamespace(type="ai", content="done", tool_calls=[]),
        ]
        svc.log_activity("u1", "t1", messages)
        files = svc.worker.enqueue.call_args.args[0]["files_touched"]
        assert files == [
            {"path": "menu.md", "size": 1, "content_hash": None, "op": "write"}
        ]

    def test_custom_write_tools_are_honored(self):
        # TC-EP-SVC-012
        svc = _service()
        messages = [
            SimpleNamespace(type="human", content="q", tool_calls=[]),
            SimpleNamespace(
                type="ai",
                content="a",
                tool_calls=[{"name": "save_doc", "args": {"path": "a.md"}}],
            ),
        ]
        svc.log_activity(
            "u1",
            "t1",
            messages,
            fs_write_tools=frozenset({"save_doc"}),
            fs_create_tools=frozenset({"save_doc"}),
        )
        files = svc.worker.enqueue.call_args.args[0]["files_touched"]
        assert files[0] == {
            "path": "a.md",
            "size": 0,
            "content_hash": None,
            "op": "write",
        }

    def test_dict_messages_project_fully(self):
        # TC-EP-SVC-013: the shape messages arrive in over the REST shell.
        svc = _service()
        svc.log_activity(
            "u1",
            "t1",
            [
                {"type": "human", "content": "Q"},
                {"type": "ai", "content": "A", "tool_calls": []},
            ],
        )
        doc = svc.worker.enqueue.call_args.args[0]
        assert [m["content"] for m in doc["messages"]] == ["Q", "A"]
        assert doc["__search_text"] == "Q\n\nA"


class TestLogActivityGuards:
    """REQ-E-101: no tenant, no document."""

    def test_missing_user_id_writes_nothing(self):
        # TC-EP-SVC-020
        svc = _service()
        assert svc.log_activity("", "t1", _turn()) is False
        svc.worker.enqueue.assert_not_called()

    def test_missing_thread_id_writes_nothing(self):
        # TC-EP-SVC-021
        svc = _service()
        assert svc.log_activity("u1", "", _turn()) is False
        svc.worker.enqueue.assert_not_called()

    def test_disabled_writes_nothing(self):
        # TC-EP-SVC-022: callers need no conditional around log_activity.
        svc = _service(episodic_enabled=False)
        assert svc.log_activity("u1", "t1", _turn()) is False
        svc.worker.enqueue.assert_not_called()

    def test_log_activity_is_not_a_coroutine(self):
        # TC-EP-SVC-023: there is nothing to await, so awaiting cannot be
        # required — that is what keeps it off the hot path.
        import inspect

        assert not inspect.iscoroutinefunction(EpisodicService.log_activity)


class TestSearchTextGating:
    """REQ-E-105: only embed a turn that has an answer."""

    def test_a_final_step_is_marked_for_embedding(self):
        # TC-EP-SVC-030
        svc = _service()
        svc.log_activity("u1", "t1", _turn("Q", "A"))
        assert svc.worker.enqueue.call_args.args[0]["__search_text"] == "Q\n\nA"

    def test_a_mid_turn_step_is_not_marked(self):
        # TC-EP-SVC-031: it has a question but no answer.
        svc = _service()
        messages = [
            SimpleNamespace(type="human", content="Q", tool_calls=[]),
            SimpleNamespace(type="ai", content="", tool_calls=[{"name": "search"}]),
        ]
        svc.log_activity("u1", "t1", messages)
        assert "__search_text" not in svc.worker.enqueue.call_args.args[0]

    def test_a_single_role_turn_is_stored_but_not_marked(self):
        # TC-EP-SVC-032
        svc = _service()
        svc.log_activity(
            "u1", "t1", [SimpleNamespace(type="ai", content="A", tool_calls=[])]
        )
        doc = svc.worker.enqueue.call_args.args[0]
        assert "__search_text" not in doc
        assert len(doc["messages"]) == 1

    def test_the_unsearchable_warning_fires_once_per_thread(self):
        # TC-EP-SVC-033
        svc = _service()
        only_ai = [SimpleNamespace(type="ai", content="A", tool_calls=[])]
        svc.log_activity("u1", "t1", only_ai)
        svc.log_activity("u1", "t1", only_ai)
        assert svc._search_warned == {"t1"}

    def test_disabling_the_final_step_gate_embeds_every_turn(self):
        # TC-EP-SVC-034
        svc = _service(episodic_embed_final_steps_only=False)
        messages = [
            SimpleNamespace(type="human", content="Q", tool_calls=[]),
            SimpleNamespace(type="ai", content="A", tool_calls=[{"name": "search"}]),
        ]
        svc.log_activity("u1", "t1", messages)
        assert svc.worker.enqueue.call_args.args[0]["__search_text"] == "Q\n\nA"

    def test_search_text_uses_its_own_cap(self):
        # TC-EP-SVC-035: embedding cost is per token.
        svc = _service(episodic_content_cap=1000, episodic_search_text_cap=5)
        svc.log_activity("u1", "t1", _turn("x" * 50, "y"))
        assert svc.worker.enqueue.call_args.args[0]["__search_text"] == "x" * 5


class TestSearch:
    """REQ-E-110: hybrid recall with the tenant filter in both branches."""

    async def test_rankfusion_pipeline_over_episodes(self):
        # TC-EP-SVC-040
        col = _collection([{"user_id": "u1", "search_text": "Q\n\nA"}])
        svc = _service(col)

        results = await svc.search("u1", "friday dinner")

        assert results == [{"user_id": "u1", "search_text": "Q\n\nA"}]
        pipeline = col.aggregate.await_args.args[0]
        pipes = pipeline[0]["$rankFusion"]["input"]["pipelines"]
        assert pipes["vectorPipeline"][0]["$vectorSearch"]["index"] == (
            "episodes_vector_index"
        )
        assert pipes["fullTextPipeline"][0]["$search"]["index"] == "episodes_fts_index"

    async def test_the_user_filter_is_in_both_branches(self):
        # TC-EP-SVC-041: isolation is enforced by the engine.
        col = _collection()
        await _service(col).search("u1", "q")

        pipes = col.aggregate.await_args.args[0][0]["$rankFusion"]["input"]["pipelines"]
        assert pipes["vectorPipeline"][0]["$vectorSearch"]["filter"]["user_id"] == "u1"
        clauses = pipes["fullTextPipeline"][0]["$search"]["compound"]["filter"]
        assert {"equals": {"path": "user_id", "value": "u1"}} in clauses

    async def test_another_users_query_cannot_reach_this_users_turns(self):
        # TC-EP-SVC-042: the pre-filter is the only thing that matters here, so
        # assert it is scoped to the caller and never to a document's owner.
        col = _collection()
        await _service(col).search("u2", "friday dinner")

        pipes = col.aggregate.await_args.args[0][0]["$rankFusion"]["input"]["pipelines"]
        assert pipes["vectorPipeline"][0]["$vectorSearch"]["filter"] == {"user_id": "u2"}
        clauses = pipes["fullTextPipeline"][0]["$search"]["compound"]["filter"]
        assert clauses == [{"equals": {"path": "user_id", "value": "u2"}}]

    async def test_thread_and_agent_narrow_both_branches(self):
        # TC-EP-SVC-043
        col = _collection()
        await _service(col).search("u1", "q", thread_id="t1", agent_name="researcher")

        pipes = col.aggregate.await_args.args[0][0]["$rankFusion"]["input"]["pipelines"]
        assert pipes["vectorPipeline"][0]["$vectorSearch"]["filter"] == {
            "user_id": "u1",
            "thread_id": "t1",
            "agent_name": "researcher",
        }
        clauses = pipes["fullTextPipeline"][0]["$search"]["compound"]["filter"]
        assert len(clauses) == 3

    async def test_since_is_applied_after_fusion(self):
        # TC-EP-SVC-044: ts is not a declared vector filter field, so the
        # narrowing has to happen in the pipeline rather than a branch pre-filter.
        col = _collection()
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        await _service(col).search("u1", "q", since=cutoff)

        stages = col.aggregate.await_args.args[0]
        assert {"$match": {"ts": {"$gte": cutoff}}} in stages
        assert "$rankFusion" in stages[0]

    async def test_since_narrows_before_the_limit_truncates(self):
        """TC-EP-SVC-044b: the `$match` must precede `$limit`, not follow it.

        This assertion replaces one that required the opposite. The old test
        asserted `args[0][-1] == {"$match": ...}` — the `$match` as the *final*
        stage — which is precisely the bug: fusion ranked across all time, `$limit`
        kept the top N, and only then did the date filter run. Asking for the 5
        most relevant turns since yesterday returned however many of the 5
        best-all-time happened to be recent, frequently zero, while the collection
        held plenty of matching recent turns.

        The old test passed. It was pinning the defect in place, which is why the
        ordering is now asserted as an index comparison rather than a position.
        """
        col = _collection()
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        await _service(col).search("u1", "q", since=cutoff)

        stages = col.aggregate.await_args.args[0]
        match_at = next(i for i, s in enumerate(stages) if "$match" in s)
        limit_at = next(i for i, s in enumerate(stages) if "$limit" in s)
        assert match_at < limit_at, (
            "since filtered an already-truncated result set; recent matches "
            "ranked below the cutoff are silently unreachable"
        )

    async def test_no_match_stage_when_since_is_absent(self):
        """TC-EP-SVC-044c: the common path stays exactly as it was."""
        col = _collection()
        await _service(col).search("u1", "q")

        stages = col.aggregate.await_args.args[0]
        assert not any("$match" in s for s in stages)

    async def test_limit_is_clamped_to_the_configured_maximum(self):
        # TC-EP-SVC-045
        col = _collection()
        svc = _service(col, max_results_per_query=3)
        await svc.search("u1", "q", limit=100)

        limits = [s["$limit"] for s in col.aggregate.await_args.args[0] if "$limit" in s]
        assert 3 in limits

    async def test_results_are_sanitized(self):
        # TC-EP-SVC-046: BSON inside messages[] would break the JSON boundary.
        oid = ObjectId()
        ts = datetime(2026, 8, 4, tzinfo=timezone.utc)
        col = _collection([{"_id": oid, "ts": ts, "messages": [{"at": ts}]}])
        results = await _service(col).search("u1", "q")

        assert results[0]["_id"] == str(oid)
        assert results[0]["ts"] == ts.isoformat()
        assert results[0]["messages"][0]["at"] == ts.isoformat()


class TestGetThread:
    """REQ-E-111: replay in step order."""

    async def test_query_is_user_scoped_and_step_ordered(self):
        # TC-EP-SVC-050: a thread id is not a capability.
        col = _collection([{"step": 0}])
        await _service(col).get_thread("u1", "t1")

        assert col.find.call_args.args[0] == {"user_id": "u1", "thread_id": "t1"}
        assert col.find.return_value.sort.call_args.args[0] == [("ts", 1), ("step", 1)]

    async def test_ts_leads_the_sort_so_a_null_step_stays_in_place(self):
        """TC-EP-SVC-050b: `step` must not be the primary sort key.

        The worker writes `step: null` rather than dropping a turn when the durable
        counter round trip fails — "a logged turn beats a lost one". Null sorts
        below every number in BSON, so with `step` leading, an Atlas hiccup during
        turn 4 of 40 moved that turn to the *front* of the replay. The turn was not
        lost; it was relocated, and nothing in the output says so. A reader gets a
        coherent-looking conversation in the wrong order.

        This assertion replaces one that required `[("step", 1), ("ts", 1)]`.
        """
        col = _collection([{"step": None}])
        await _service(col).get_thread("u1", "t1")

        keys = [k for k, _ in col.find.return_value.sort.call_args.args[0]]
        assert keys.index("ts") < keys.index("step"), (
            "a turn whose durable step counter failed sorts to the front of the "
            "thread instead of staying where it happened"
        )

    async def test_descending_reverses_both_sort_keys(self):
        # TC-EP-SVC-051
        col = _collection()
        await _service(col).get_thread("u1", "t1", ascending=False)
        assert col.find.return_value.sort.call_args.args[0] == [("ts", -1), ("step", -1)]

    async def test_limit_is_applied_when_given(self):
        # TC-EP-SVC-052
        col = _collection()
        await _service(col).get_thread("u1", "t1", limit=5)
        col.find.return_value.limit.assert_called_once_with(5)

    async def test_no_limit_means_no_limit_stage(self):
        # TC-EP-SVC-053
        col = _collection()
        await _service(col).get_thread("u1", "t1")
        col.find.return_value.limit.assert_not_called()

    async def test_embedding_is_projected_out(self):
        # TC-EP-SVC-054
        col = _collection()
        await _service(col).get_thread("u1", "t1")
        assert col.find.call_args.args[1] == {"embedding": 0}

    async def test_results_are_sanitized(self):
        # TC-EP-SVC-055
        oid = ObjectId()
        col = _collection([{"_id": oid, "step": 0}])
        results = await _service(col).get_thread("u1", "t1")
        assert results[0]["_id"] == str(oid)

    async def test_no_match_returns_an_empty_list(self):
        # TC-EP-SVC-056
        assert await _service(_collection([])).get_thread("u1", "t1") == []


class TestGetByCorrelationId:
    """REQ-E-111: the join back to a tracing stack."""

    async def test_query_is_user_scoped_and_time_ordered(self):
        # TC-EP-SVC-060
        col = _collection([{"correlation_id": "trace-1"}])
        await _service(col).get_by_correlation_id("u1", "trace-1")

        assert col.find.call_args.args[0] == {
            "user_id": "u1",
            "correlation_id": "trace-1",
        }
        assert col.find.return_value.sort.call_args.args[0] == [("ts", 1), ("step", 1)]

    async def test_limit_is_applied_when_given(self):
        # TC-EP-SVC-061
        col = _collection()
        await _service(col).get_by_correlation_id("u1", "trace-1", limit=2)
        col.find.return_value.limit.assert_called_once_with(2)


class TestRetention:
    """REQ-E-116: collMod, with a fallback, never raising."""

    async def test_collmod_updates_the_ttl_in_place(self):
        # TC-EP-SVC-070: retention is a runtime knob, not a redeploy.
        col = _collection()
        result = await _service(col).set_retention(7200)

        # `scope` is part of the contract: a TTL index belongs to the collection,
        # so this retunes retention for every tenant. The facade takes a user_id
        # to authorise against, not to scope by, and saying so at the call site is
        # the fix for that mismatch.
        assert result == {
            "status": "updated", "ttl_seconds": 7200, "scope": "collection",
        }
        command = col.database.command.await_args.args[0]
        assert command["collMod"] == "episodes"
        assert command["index"] == {
            "name": "ix_episodes_ttl",
            "expireAfterSeconds": 7200,
        }
        col.create_index.assert_not_awaited()

    async def test_none_drops_the_ttl_index(self):
        # TC-EP-SVC-071: makes the log permanent.
        col = _collection()
        result = await _service(col).set_retention(None)

        assert result == {
            "status": "removed", "ttl_seconds": None, "scope": "collection",
        }
        col.drop_index.assert_awaited_once_with("ix_episodes_ttl")

    async def test_it_falls_back_to_create_index(self):
        # TC-EP-SVC-072: deployments without collMod.
        col = _collection()
        col.database.command = AsyncMock(side_effect=RuntimeError("no collMod"))
        result = await _service(col).set_retention(3600)

        assert result == {
            "status": "created", "ttl_seconds": 3600, "scope": "collection",
        }
        assert col.create_index.await_args.kwargs["expireAfterSeconds"] == 3600
        assert col.create_index.await_args.kwargs["name"] == "ix_episodes_ttl"

    async def test_a_total_failure_reports_instead_of_raising(self):
        # TC-EP-SVC-073
        col = _collection()
        col.database.command = AsyncMock(side_effect=RuntimeError("no collMod"))
        col.create_index = AsyncMock(side_effect=RuntimeError("no permission"))
        result = await _service(col).set_retention(3600)

        assert result["status"] == "error"
        assert "no permission" in result["error"]

    async def test_a_drop_failure_reports_instead_of_raising(self):
        # TC-EP-SVC-074
        col = _collection()
        col.drop_index = AsyncMock(side_effect=RuntimeError("missing"))
        result = await _service(col).set_retention(None)
        assert result["status"] == "error"


class TestLifecycleDelegation:
    async def test_flush_delegates_to_the_worker(self):
        # TC-EP-SVC-080
        svc = _service()
        assert await svc.flush(1.0) is True
        svc.worker.flush.assert_awaited_once_with(1.0)

    async def test_close_delegates_to_the_worker(self):
        # TC-EP-SVC-081
        svc = _service()
        assert await svc.close(1.0) is True
        svc.worker.close.assert_awaited_once_with(1.0)

    def test_stats_delegates_to_the_worker(self):
        # TC-EP-SVC-082
        svc = _service()
        assert svc.stats() == {"written": 0}

    def test_the_counter_collection_defaults_alongside_episodes(self):
        # TC-EP-SVC-083
        col = _collection()
        col.database.__getitem__ = MagicMock(return_value="counters-col")
        svc = EpisodicService(col, _config(), _providers(), worker=MagicMock())
        col.database.__getitem__.assert_called_once_with("episodes_counters")
        assert svc.counters == "counters-col"
