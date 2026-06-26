"""Tests for the REST shell. REQ-E-070, REQ-E-071, REQ-E-072.

The facade is mocked; FastAPI TestClient drives real HTTP through the routes.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from agent_memory.exceptions import AccessError, NotFoundError, RateLimitError
from agent_memory.shells.rest.app import create_app


def _app():
    app = MagicMock()
    app.add = AsyncMock(return_value={"stm_ids": ["a"], "count": 1})
    app.recall = AsyncMock(return_value={"results": [], "count": 0})
    app.search = AsyncMock(return_value={"results": [], "count": 0})
    app.delete = AsyncMock(return_value={"deleted_count": 0})
    app.remember_decision = AsyncMock(return_value={"key": "k", "status": "stored"})
    app.recall_decision = AsyncMock(return_value={"key": "k", "value": "v"})
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
