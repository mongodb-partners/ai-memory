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
