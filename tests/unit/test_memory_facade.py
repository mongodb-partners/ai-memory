"""Tests for the AsyncMemory facade. REQ-E-020..031, INV-001..007.

Services are mocked: the facade's job is orchestration (access-check → service →
audit), not service internals (those are covered by the ported service tests).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.exceptions import AccessError, ConfigError, RateLimitError
from agent_memory.memory import AsyncMemory


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


def _facade(config=None):
    """Construct an AsyncMemory with all collaborators mocked (no create())."""
    m = AsyncMemory.__new__(AsyncMemory)
    m.config = config or _config()
    m.memory_service = AsyncMock()
    m.cache_service = AsyncMock()
    m.decision_service = AsyncMock()
    m.admin_service = AsyncMock()
    m.audit_service = AsyncMock()
    m.episodic_service = AsyncMock()
    # log_activity and stats() are synchronous on the real service.
    m.episodic_service.log_activity = MagicMock(return_value=True)
    m.episodic_service.stats = MagicMock(return_value={"written": 0})
    m.governance_service = None
    m.rate_limiter = None
    m.providers = MagicMock()
    m._workers = []
    return m


class TestOrchestration:
    """TC-FAC-AUDIT / TC-FAC-ACC: the single orchestration path."""

    async def test_success_writes_success_audit(self):
        m = _facade()
        m.memory_service.store_stm = AsyncMock(return_value=["id1"])
        await m.add("u1", "c1", [{"content": "hi", "message_type": "human"}])
        m.audit_service.log.assert_called_once()
        assert m.audit_service.log.call_args.args[3] == "success"

    async def test_service_error_writes_error_audit_and_reraises(self):
        m = _facade()
        m.memory_service.store_stm = AsyncMock(side_effect=RuntimeError("db down"))
        with pytest.raises(RuntimeError, match="db down"):
            await m.add("u1", "c1", [{"content": "hi"}])
        assert m.audit_service.log.call_args.args[3] == "error"

    async def test_governance_denial_raises_access_error(self):
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=False)
        m.memory_service.store_stm = AsyncMock()
        with pytest.raises(AccessError):
            await m.add("u1", "c1", [{"content": "hi"}])
        m.memory_service.store_stm.assert_not_called()

    async def test_rate_limit_raises_rate_limit_error(self):
        m = _facade()
        m.rate_limiter = AsyncMock()
        m.rate_limiter.check_rate_limit = AsyncMock(return_value=False)
        with pytest.raises(RateLimitError):
            await m.add("u1", "c1", [{"content": "hi"}])

    async def test_rate_limit_error_is_catchable_as_access_error(self):
        m = _facade()
        m.rate_limiter = AsyncMock()
        m.rate_limiter.check_rate_limit = AsyncMock(return_value=False)
        with pytest.raises(AccessError):
            await m.add("u1", "c1", [{"content": "hi"}])


class TestMethodSurface:
    """TC-FAC-001..012: each public method delegates to the right service."""

    async def test_add_delegates(self):
        m = _facade()
        m.memory_service.store_stm = AsyncMock(return_value=["a", "b"])
        out = await m.add("u1", "c1", [{"content": "x"}])
        assert out["count"] == 2 and out["stm_ids"] == ["a", "b"]

    async def test_recall_delegates_to_recall(self):
        # TC-FAC-RECALL-001 (REQ-E-030)
        m = _facade()
        m.memory_service.recall = AsyncMock(return_value=[{"id": 1}])
        out = await m.recall("u1", "q")
        m.memory_service.recall.assert_called_once()
        assert out["count"] == 1

    async def test_search_delegates_to_hybrid_search(self):
        # TC-FAC-SEARCH-001 (REQ-E-030): search != recall
        m = _facade()
        m.memory_service.hybrid_search = AsyncMock(return_value=[{"id": 1, "score": 0.9}])
        out = await m.search("u1", "q")
        m.memory_service.hybrid_search.assert_called_once()
        m.memory_service.recall.assert_not_called()
        assert out["count"] == 1

    async def test_delete_delegates(self):
        m = _facade()
        m.memory_service.delete = AsyncMock(return_value={"deleted_count": 3})
        out = await m.delete("u1", memory_id="m1")
        assert out["deleted_count"] == 3

    async def test_cache_methods_delegate(self):
        m = _facade()
        m.cache_service.check = AsyncMock(return_value={"cache_hit": True})
        m.cache_service.store = AsyncMock(return_value="cid")
        m.cache_service.invalidate = AsyncMock(return_value=2)
        assert (await m.check_cache("u1", "q"))["cache_hit"] is True
        assert await m.store_cache("u1", "q", "r") == "cid"
        assert (await m.invalidate_cache("u1", invalidate_all=True))["deleted_count"] == 2

    async def test_decision_methods_delegate(self):
        m = _facade()
        m.decision_service.store = AsyncMock(return_value="stored")
        m.decision_service.recall = AsyncMock(return_value={"key": "k", "value": "v"})
        assert (await m.remember_decision("u1", "k", "v"))["status"] == "stored"
        assert (await m.recall_decision("u1", "k"))["value"] == "v"

    async def test_health_and_wipe_delegate(self):
        m = _facade()
        m.admin_service.health = AsyncMock(return_value={"total_memories": 5})
        m.admin_service.wipe_user_data = AsyncMock(return_value={"memories_deleted": 5})
        assert (await m.health("u1"))["total_memories"] == 5
        assert (await m.wipe_user_data("u1", confirm=True))["memories_deleted"] == 5

    async def test_wipe_requires_confirm(self):
        m = _facade()
        m.admin_service.wipe_user_data = AsyncMock()
        out = await m.wipe_user_data("u1")  # confirm defaults False
        assert "error" in out
        m.admin_service.wipe_user_data.assert_not_called()

    async def test_recall_decision_none_passthrough(self):
        m = _facade()
        m.decision_service.recall = AsyncMock(return_value=None)
        assert await m.recall_decision("u1", "missing") is None


class TestEpisodicSurface:
    """TC-FAC-EP-001..012 (REQ-E-100..116): the seven episodic facade methods."""

    async def test_log_activity_enqueues_and_returns_without_awaiting_the_db(self):
        # TC-FAC-EP-001: non-blocking by contract — the service call is sync.
        m = _facade()
        out = await m.log_activity("u1", "t1", [{"type": "human", "content": "hi"}])
        assert out == {"enqueued": True, "thread_id": "t1"}
        m.episodic_service.log_activity.assert_called_once()

    async def test_log_activity_forwards_every_keyword(self):
        # TC-FAC-EP-002: a dropped kwarg here silently loses trace/tenant data.
        m = _facade()
        await m.log_activity(
            "u1", "t1", [{"type": "ai"}], todos=[{"id": "1"}], agent_name="planner",
            correlation_id="corr", conversation_id="c1", ts="2026-08-04T11:00:00Z",
        )
        kwargs = m.episodic_service.log_activity.call_args.kwargs
        assert kwargs == {
            "todos": [{"id": "1"}], "agent_name": "planner",
            "correlation_id": "corr", "conversation_id": "c1",
            "ts": "2026-08-04T11:00:00Z",
        }

    async def test_log_activity_writes_no_audit_record(self):
        """TC-FAC-EP-003: audit amplification guard.

        A turn log is high-volume by nature. One audit write per turn would make
        logging the agent cost more than the agent, so the worker audits per
        flushed batch instead — the facade must stay out of ``_run``.
        """
        m = _facade()
        await m.log_activity("u1", "t1", [{"type": "human"}])
        m.audit_service.log.assert_not_called()

    async def test_log_activity_still_enforces_governance(self):
        # TC-FAC-EP-004: bypassing _run must not bypass access control.
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=False)
        with pytest.raises(AccessError):
            await m.log_activity("u1", "t1", [{"type": "human"}])
        m.episodic_service.log_activity.assert_not_called()

    async def test_log_activity_still_enforces_rate_limits(self):
        # TC-FAC-EP-005
        m = _facade()
        m.rate_limiter = AsyncMock()
        m.rate_limiter.check_rate_limit = AsyncMock(return_value=False)
        with pytest.raises(RateLimitError):
            await m.log_activity("u1", "t1", [{"type": "human"}])
        m.episodic_service.log_activity.assert_not_called()

    async def test_recall_activity_delegates_and_counts(self):
        # TC-FAC-EP-006
        m = _facade()
        m.episodic_service.search = AsyncMock(return_value=[{"step": 0}, {"step": 1}])
        out = await m.recall_activity("u1", "friday dinner", thread_id="t1", limit=3)
        assert out["count"] == 2 and len(out["results"]) == 2
        kwargs = m.episodic_service.search.call_args.kwargs
        assert kwargs["thread_id"] == "t1" and kwargs["limit"] == 3

    async def test_recall_activity_draws_from_the_search_budget(self):
        """TC-FAC-EP-007: search_activity is a search, not a write.

        It runs $rankFusion, so it must be metered against
        max_searches_per_day rather than the memory-write budget — which is why
        it belongs in _SEARCH_OPERATIONS.
        """
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=True)
        m.governance_service.get_profile = AsyncMock(return_value={
            "max_searches_per_day": 5000, "max_memories_per_day": 100,
        })
        m.rate_limiter = AsyncMock()
        m.rate_limiter.check_rate_limit = AsyncMock(return_value=True)
        m.episodic_service.search = AsyncMock(return_value=[])
        await m.recall_activity("u1", "q")
        assert m.rate_limiter.check_rate_limit.await_args.kwargs["max_requests"] == 5000

    async def test_log_activity_draws_from_the_write_budget(self):
        # TC-FAC-EP-007b: the contrast case — a write, so the write budget.
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=True)
        m.governance_service.get_profile = AsyncMock(return_value={
            "max_searches_per_day": 5000, "max_memories_per_day": 100,
        })
        m.rate_limiter = AsyncMock()
        m.rate_limiter.check_rate_limit = AsyncMock(return_value=True)
        await m.log_activity("u1", "t1", [{"type": "human"}])
        assert m.rate_limiter.check_rate_limit.await_args.kwargs["max_requests"] == 100

    async def test_get_thread_delegates(self):
        # TC-FAC-EP-008
        m = _facade()
        m.episodic_service.get_thread = AsyncMock(return_value=[{"step": 0}])
        out = await m.get_thread("u1", "t1", ascending=False)
        assert out["count"] == 1
        assert m.episodic_service.get_thread.call_args.kwargs["ascending"] is False

    async def test_get_activity_by_correlation_delegates(self):
        # TC-FAC-EP-009
        m = _facade()
        m.episodic_service.get_by_correlation_id = AsyncMock(return_value=[{"step": 2}])
        out = await m.get_activity_by_correlation("u1", "corr-1")
        assert out["count"] == 1
        assert m.episodic_service.get_by_correlation_id.call_args.args[1] == "corr-1"

    async def test_read_methods_are_audited(self):
        # TC-FAC-EP-010: reads go through _run, so each leaves a trail.
        m = _facade()
        m.episodic_service.search = AsyncMock(return_value=[])
        m.episodic_service.get_thread = AsyncMock(return_value=[])
        m.episodic_service.get_by_correlation_id = AsyncMock(return_value=[])
        await m.recall_activity("u1", "q")
        await m.get_thread("u1", "t1")
        await m.get_activity_by_correlation("u1", "corr")
        categories = [c.args[1] for c in m.audit_service.log.call_args_list]
        assert categories == ["episodic:read"] * 3

    async def test_flush_activity_needs_no_access_check(self):
        """TC-FAC-EP-011: a lifecycle call about this process's own buffer.

        It takes no user_id, so there is no scope to check — and a governance
        denial on shutdown would strand queued turns.
        """
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=False)
        m.episodic_service.flush = AsyncMock(return_value=True)
        assert await m.flush_activity(timeout=0.1) is True
        m.episodic_service.flush.assert_awaited_once_with(0.1)

    async def test_set_activity_retention_is_an_admin_operation(self):
        # TC-FAC-EP-012
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(return_value={"ttl_seconds": 7200})
        out = await m.set_activity_retention("u1", ttl_seconds=7200)
        assert out["ttl_seconds"] == 7200
        assert m.audit_service.log.call_args.args[1] == "admin"

    def test_activity_stats_is_synchronous(self):
        # TC-FAC-EP-013: an /health probe reads this; it must not need a loop.
        m = _facade()
        m.episodic_service.stats = MagicMock(return_value={"enqueued": 7})
        assert m.activity_stats() == {"enqueued": 7}


class TestLifecycle:
    """TC-FAC-LIFE / TC-FAC-DIM: create()/close() and the startup dim guard."""

    async def test_workers_disabled_logs_warning_and_starts_none(self, caplog):
        # TC-FAC-LIFE-002 (REQ-E-022, premortem #2)
        m = _build_for_lifecycle(workers_in_process=False)
        with caplog.at_level("WARNING"):
            await m._maybe_start_workers()
        assert m._workers == []
        assert any("workers_in_process" in r.message or "reactive" in r.message.lower()
                   for r in caplog.records)

    async def test_workers_enabled_starts_four(self):
        # TC-FAC-LIFE-001 (REQ-E-021): enrichment, consolidation, audit flush,
        # and the episodic writer.
        m = _build_for_lifecycle(workers_in_process=True)
        await m._maybe_start_workers()
        assert len(m._workers) == 4
        for t in m._workers:
            t.cancel()

    async def test_dimension_mismatch_raises_config_error(self):
        # TC-FAC-DIM-001 (REQ-E-031, premortem #3, boundary #6)
        providers = MagicMock()
        providers.embedding = AsyncMock()
        providers.embedding.generate_embedding = AsyncMock(return_value=[0.0] * 768)
        with pytest.raises(ConfigError, match="dimension"):
            await AsyncMemory._validate_embedding_dimension(
                providers, expected=1536,
            )

    async def test_dimension_match_passes(self):
        providers = MagicMock()
        providers.embedding = AsyncMock()
        providers.embedding.generate_embedding = AsyncMock(return_value=[0.0] * 1536)
        # should not raise
        await AsyncMemory._validate_embedding_dimension(providers, expected=1536)


class TestCreateAndClose:
    """Exercise the create() wiring and close() teardown with everything mocked."""

    async def test_create_wires_services_and_starts_workers(self, monkeypatch):
        import agent_memory.memory as mem

        # Stub DatabaseManager
        db = {f"c{i}": MagicMock() for i in range(10)}

        class FakeDB:
            def __getitem__(self, k):
                return MagicMock()

        db_manager = MagicMock()
        db_manager.db = FakeDB()
        db_manager.close = AsyncMock()
        monkeypatch.setattr(mem, "_ensure_search_indexes_bg", AsyncMock())

        import agent_memory.core.database as dbmod
        monkeypatch.setattr(dbmod.DatabaseManager, "initialize", AsyncMock(return_value=db_manager))

        import agent_memory.core.migrations as mig
        monkeypatch.setattr(mig, "ensure_indexes", AsyncMock())

        # Stub ProviderManager with a dimension-matching embedder
        import agent_memory.providers.manager as pm

        def _fake_provider_manager(config):
            p = MagicMock()
            p.embedding = AsyncMock()
            p.embedding.generate_embedding = AsyncMock(return_value=[0.0] * config.embedding_dimension)
            return p

        monkeypatch.setattr(pm, "ProviderManager", _fake_provider_manager)

        # Stub all services to no-op constructors with async seed_defaults
        for name in ("memory", "cache", "audit", "decision", "prompt_library",
                     "governance", "rate_limiter"):
            pass

        cfg = _config(governance_enabled=False, rate_limit_enabled=False, workers_in_process=True)
        m = await AsyncMemory.create(cfg)
        try:
            assert len(m._workers) == 4
            assert m.memory_service is not None
            assert m.episodic_service is not None
            assert m.admin_service is not None
        finally:
            await m.close()
        db_manager.close.assert_awaited_once()

    async def test_close_cancels_workers_and_flushes(self):
        m = AsyncMemory.__new__(AsyncMemory)
        task = MagicMock()
        m._workers = [task]
        m._search_index_task = MagicMock()
        m._search_index_task.done.return_value = False
        m.audit_service = AsyncMock()
        m._db_manager = AsyncMock()
        await m.close()
        task.cancel.assert_called_once()
        m.audit_service.flush.assert_awaited_once()
        m._db_manager.close.assert_awaited_once()


def _build_for_lifecycle(**cfg_overrides):
    m = AsyncMemory.__new__(AsyncMemory)
    m.config = _config(**cfg_overrides)
    m.memory_service = MagicMock()
    m.audit_service = MagicMock()
    m.providers = MagicMock()
    m.prompt_library = MagicMock()
    # _maybe_start_workers schedules the episodic writer's consumer loop, so
    # run() has to be awaitable for asyncio.create_task to accept it.
    m.episodic_service = MagicMock()
    m.episodic_service.worker.run = AsyncMock()
    m._workers = []
    return m
