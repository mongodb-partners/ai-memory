"""Tests for the REST shell. REQ-E-070, REQ-E-071, REQ-E-072.

The facade is mocked; FastAPI TestClient drives real HTTP through the routes.
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from agent_memory.exceptions import (
    AccessError,
    EmbeddingError,
    NotFoundError,
    RateLimitError,
)
from agent_memory.shells.rest.app import create_app


def _app():
    app = MagicMock()
    app.add = AsyncMock(return_value={"stm_ids": ["a"], "count": 1})
    app.recall = AsyncMock(return_value={"results": [], "count": 0})
    app.search = AsyncMock(return_value={"results": [], "count": 0})
    app.delete = AsyncMock(return_value={"deleted_count": 0})
    app.remember_decision = AsyncMock(return_value={"key": "k", "status": "stored"})
    app.recall_decision = AsyncMock(return_value={"key": "k", "value": "v"})
    app.log_activity = AsyncMock(return_value={"enqueued": True, "thread_id": "t1"})
    app.recall_activity = AsyncMock(return_value={"results": [], "count": 0})
    app.get_thread = AsyncMock(return_value={"results": [], "count": 0})
    app.get_activity_by_correlation = AsyncMock(return_value={"results": [], "count": 0})
    app.set_activity_retention = AsyncMock(return_value={"ttl_seconds": 7200})
    # activity_stats is synchronous on the facade — /health must not await it.
    app.activity_stats = MagicMock(return_value={"enqueued": 3, "queue_depth": 0})
    # So is worker_status, and for the same reason: a health probe must not have to
    # queue behind the very event loop whose workers it is asking about.
    app.worker_status = MagicMock(return_value={
        "enabled": True, "running": True,
        "workers": {"enrichment": {"done": False, "cancelled": False}},
    })
    return app


def _client(facade):
    return TestClient(create_app(facade))


class TestRoutes:
    """TC-REST-001..004: routes call the facade and return its payload."""

    def test_post_memories(self):
        facade = _app()
        r = _client(facade).post("/memories", json={
            "user_id": "u1", "conversation_id": "c1",
            "messages": [{"content": "hi"}],
        })
        assert r.status_code == 200
        assert r.json()["count"] == 1
        facade.add.assert_awaited_once()

    def test_get_recall(self):
        facade = _app()
        r = _client(facade).get("/memories/recall", params={"user_id": "u1", "query": "q"})
        assert r.status_code == 200
        facade.recall.assert_awaited_once()

    def test_get_search(self):
        facade = _app()
        r = _client(facade).get("/memories/search", params={"user_id": "u1", "query": "q"})
        assert r.status_code == 200
        facade.search.assert_awaited_once()

    def test_health(self):
        facade = _app()
        r = _client(facade).get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestErrorMapping:
    """TC-REST-005: exception → HTTP status, with 429 ordered before 403."""

    def test_rate_limit_maps_to_429(self):
        facade = _app()
        facade.add = AsyncMock(side_effect=RateLimitError("slow down"))
        r = _client(facade).post("/memories", json={
            "user_id": "u1", "conversation_id": "c1", "messages": [{"content": "x"}],
        })
        assert r.status_code == 429

    def test_access_denied_maps_to_403(self):
        facade = _app()
        facade.add = AsyncMock(side_effect=AccessError("denied"))
        r = _client(facade).post("/memories", json={
            "user_id": "u1", "conversation_id": "c1", "messages": [{"content": "x"}],
        })
        assert r.status_code == 403

    def test_not_found_maps_to_404(self):
        facade = _app()
        facade.recall = AsyncMock(side_effect=NotFoundError("missing"))
        r = _client(facade).get("/memories/recall", params={"user_id": "u1", "query": "q"})
        assert r.status_code == 404

    def test_an_embedding_refusal_maps_to_502(self):
        # Not 500. The request was well-formed and this service is healthy — the
        # upstream embedder answered with a reply that did not describe the input,
        # so nothing was written and retrying is the right response. A 500 would
        # tell the caller to stop and report a bug.
        facade = _app()
        facade.add = AsyncMock(side_effect=EmbeddingError("short batch"))
        r = _client(facade).post("/memories", json={
            "user_id": "u1", "conversation_id": "c1", "messages": [{"content": "x"}],
        })
        assert r.status_code == 502
        assert "short batch" in r.json()["error"]

    def test_an_embedding_refusal_is_not_swallowed_as_a_denial(self):
        # `EmbeddingError` and `AccessError` share `MemoryError` as a base but not
        # each other. Were the handler registered on the base — or the class made a
        # subclass of `AccessError` for convenience — an upstream fault would read
        # as "you are not allowed to do this", which sends the operator to the
        # wrong system entirely.
        facade = _app()
        facade.add = AsyncMock(side_effect=EmbeddingError("wrong width"))
        r = _client(facade).post("/memories", json={
            "user_id": "u1", "conversation_id": "c1", "messages": [{"content": "x"}],
        })
        assert r.status_code not in (403, 429)


class TestRemainingRoutes:
    """Cover delete + decisions routes."""

    def test_delete(self):
        facade = _app()
        r = _client(facade).request(
            "DELETE", "/memories", params={"user_id": "u1", "memory_id": "m1"}
        )
        assert r.status_code == 200
        facade.delete.assert_awaited_once()

    def test_post_decision(self):
        facade = _app()
        r = _client(facade).post("/decisions", json={"user_id": "u1", "key": "k", "value": "v"})
        assert r.status_code == 200
        facade.remember_decision.assert_awaited_once()

    def test_get_decision(self):
        facade = _app()
        r = _client(facade).get("/decisions", params={"user_id": "u1", "key": "k"})
        assert r.status_code == 200
        facade.recall_decision.assert_awaited_once()


class TestActivityRoutes:
    """TC-REST-EP-001..006: the five episodic routes plus /health counters."""

    def test_post_activity(self):
        facade = _app()
        r = _client(facade).post("/activity", json={
            "user_id": "u1", "thread_id": "t1",
            "messages": [{"type": "human", "content": "hi"}],
            "correlation_id": "corr", "agent_name": "planner",
        })
        assert r.status_code == 200
        assert r.json()["enqueued"] is True
        kwargs = facade.log_activity.call_args.kwargs
        assert kwargs["correlation_id"] == "corr" and kwargs["agent_name"] == "planner"

    def test_get_activity_search(self):
        facade = _app()
        r = _client(facade).get("/activity/search",
                                params={"user_id": "u1", "query": "q", "thread_id": "t1"})
        assert r.status_code == 200
        assert facade.recall_activity.call_args.kwargs["thread_id"] == "t1"

    def test_get_thread(self):
        facade = _app()
        r = _client(facade).get("/activity/thread/t1", params={"user_id": "u1"})
        assert r.status_code == 200
        assert facade.get_thread.await_args.args == ("u1", "t1")

    def test_get_correlation(self):
        facade = _app()
        r = _client(facade).get("/activity/correlation/corr-1", params={"user_id": "u1"})
        assert r.status_code == 200
        assert facade.get_activity_by_correlation.await_args.args == ("u1", "corr-1")

    def test_put_retention_accepts_null_as_keep_forever(self):
        """An omitted ttl_seconds means "drop the TTL", not "no change"."""
        facade = _app()
        r = _client(facade).put("/activity/retention",
                                json={"user_id": "u1", "ttl_seconds": None})
        assert r.status_code == 200
        assert facade.set_activity_retention.call_args.kwargs["ttl_seconds"] is None

    def test_health_reports_episodic_counters(self):
        facade = _app()
        body = _client(facade).get("/health").json()
        assert body["status"] == "ok"
        assert body["episodic"]["enqueued"] == 3

    def test_health_still_200_when_stats_are_unavailable(self):
        """A probe that 500s because a counter is missing is worse than useless.

        The assertion is that `episodic` is *absent* and the status is still `ok`,
        rather than that the body equals one exact dict — the latter made adding
        any new probe field a test failure, which is a test asserting that /health
        never learns anything new.
        """
        facade = _app()
        facade.activity_stats = MagicMock(side_effect=RuntimeError("not wired"))
        r = _client(facade).get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "episodic" not in r.json()

    def test_health_still_200_when_worker_status_is_unavailable(self):
        """Same guarantee for the other optional section."""
        facade = _app()
        facade.worker_status = MagicMock(side_effect=RuntimeError("not wired"))
        r = _client(facade).get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert "workers" not in r.json()

    def test_health_degrades_when_a_worker_has_died(self):
        """TC-REST-HEALTH-004: a dead worker must be visible to a probe.

        This is the whole reason `workers` is in the body. The workers do the
        *reactive* half of the system, so when enrichment or consolidation crashes,
        reads and writes keep working perfectly and `/health` kept returning a
        cheerful 200 — the only symptom is that memories are never scored,
        promoted, or forgotten, which looks like a quiet system rather than a
        broken one.
        """
        facade = _app()
        facade.worker_status = MagicMock(return_value={
            "enabled": True, "running": False,
            "workers": {"enrichment": {"done": True, "cancelled": False,
                                       "error": "RuntimeError('boom')"}},
        })
        body = _client(facade).get("/health").json()
        assert body["status"] == "degraded"
        assert body["workers"]["workers"]["enrichment"]["done"] is True

    def test_health_stays_ok_when_workers_are_intentionally_out_of_process(self):
        """`workers_in_process=False` is a deployment choice, not a fault.

        Without the `enabled` check this would report every REST-only process as
        permanently degraded, and an alert that is always firing is an alert nobody
        reads.
        """
        facade = _app()
        facade.worker_status = MagicMock(return_value={
            "enabled": False, "running": False, "workers": {},
        })
        assert _client(facade).get("/health").json()["status"] == "ok"

    def test_activity_denial_maps_to_403(self):
        facade = _app()
        facade.log_activity = AsyncMock(side_effect=AccessError("denied"))
        r = _client(facade).post("/activity", json={
            "user_id": "u1", "thread_id": "t1", "messages": [{"type": "human"}],
        })
        assert r.status_code == 403


class TestAuth:
    """TC-REST-AUTH-001: protected routes use the existing auth/ verifier."""

    def _client_with_auth(self, facade):
        from agent_memory.config import MemoryConfig
        from agent_memory.shells.rest.app import create_app

        cfg = MemoryConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            auth_enabled=True, auth_secret="x" * 32, _env_file=None,
        )
        return TestClient(create_app(facade, config=cfg)), cfg

    def test_missing_token_rejected(self):
        client, _ = self._client_with_auth(_app())
        r = client.get("/memories/recall", params={"user_id": "u1", "query": "q"})
        assert r.status_code == 401

    def test_valid_jwt_accepted(self):
        from agent_memory.auth.api_keys import APIKeyManager
        from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier

        client, cfg = self._client_with_auth(_app())
        verifier = MemoryMCPTokenVerifier(secret=cfg.auth_secret, api_key_manager=APIKeyManager())
        token = verifier.create_token(user_id="u1")
        r = client.get("/memories/recall", params={"user_id": "u1", "query": "q"},
                       headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_health_open_without_token(self):
        client, _ = self._client_with_auth(_app())
        assert client.get("/health").status_code == 200


class TestManagedApp:
    """create_managed_app builds + tears down its own facade via lifespan."""

    def test_lifespan_creates_and_closes(self, monkeypatch):
        import agent_memory.shells.rest.app as rest
        from agent_memory.config import MemoryConfig

        instance = MagicMock()
        instance.health = AsyncMock(return_value={})
        instance.close = AsyncMock()
        monkeypatch.setattr(rest.AsyncMemory, "create", AsyncMock(return_value=instance))

        cfg = MemoryConfig(mongodb_connection_string="mongodb://localhost:27017", _env_file=None)
        api = rest.create_managed_app(cfg)
        with TestClient(api) as client:  # triggers lifespan
            assert client.get("/health").status_code == 200
        instance.close.assert_awaited_once()
