"""Tests for MemoryMCPTokenVerifier."""

import os
import time
from unittest.mock import patch

import jwt
import pytest

from agent_memory.auth.api_keys import APIKeyManager
from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier


_TEST_SECRET = "test-secret-for-unit-tests"


def _make_verifier(api_keys: str = "") -> MemoryMCPTokenVerifier:
    """Create a verifier with optional API keys."""
    with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": api_keys}):
        mgr = APIKeyManager()
    return MemoryMCPTokenVerifier(secret=_TEST_SECRET, api_key_manager=mgr)


class TestCreateToken:
    """Token creation tests."""

    def test_create_jwt_token(self):
        verifier = _make_verifier()
        token = verifier.create_token("user1")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_jwt_contains_sub(self):
        verifier = _make_verifier()
        token = verifier.create_token("user1")
        payload = jwt.decode(token, _TEST_SECRET, algorithms=["HS256"], issuer="memory-mcp")
        assert payload["sub"] == "user1"

    def test_create_jwt_with_scopes(self):
        verifier = _make_verifier()
        token = verifier.create_token("user1", scopes=["read", "write"])
        payload = jwt.decode(token, _TEST_SECRET, algorithms=["HS256"], issuer="memory-mcp")
        assert payload["scope"] == "read write"

    def test_create_jwt_with_expiry(self):
        verifier = _make_verifier()
        token = verifier.create_token("user1", expires_in=3600)
        payload = jwt.decode(token, _TEST_SECRET, algorithms=["HS256"], issuer="memory-mcp")
        assert payload["exp"] - payload["iat"] == 3600


class TestVerifyAPIKey:
    """API key verification tests."""

    async def test_verify_valid_api_key(self):
        verifier = _make_verifier(api_keys="testkey=user@test.com")
        result = await verifier.verify_token("testkey")
        assert result is not None
        assert result.client_id == "user@test.com"
        assert result.claims["auth_method"] == "api_key"

    async def test_verify_invalid_api_key(self):
        verifier = _make_verifier(api_keys="testkey=user@test.com")
        result = await verifier.verify_token("wrongkey")
        # Falls through to JWT which also fails
        assert result is None


class TestVerifyJWT:
    """JWT verification tests."""

    async def test_verify_valid_jwt(self):
        verifier = _make_verifier()
        token = verifier.create_token("user1")
        result = await verifier.verify_token(token)
        assert result is not None
        assert result.client_id == "user1"

    async def test_verify_expired_jwt(self):
        verifier = _make_verifier()
        # Create a token that's already expired
        now = int(time.time())
        payload = {
            "sub": "user1",
            "iss": "memory-mcp",
            "iat": now - 7200,
            "exp": now - 3600,
        }
        token = jwt.encode(payload, _TEST_SECRET, algorithm="HS256")
        result = await verifier.verify_token(token)
        assert result is None

    async def test_verify_wrong_secret(self):
        verifier = _make_verifier()
        # Create token with different secret
        payload = {
            "sub": "user1",
            "iss": "memory-mcp",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        result = await verifier.verify_token(token)
        assert result is None

    async def test_verify_missing_sub_claim(self):
        verifier = _make_verifier()
        now = int(time.time())
        payload = {
            "iss": "memory-mcp",
            "iat": now,
            "exp": now + 3600,
        }
        token = jwt.encode(payload, _TEST_SECRET, algorithm="HS256")
        result = await verifier.verify_token(token)
        assert result is None

    async def test_verify_wrong_issuer(self):
        verifier = _make_verifier()
        now = int(time.time())
        payload = {
            "sub": "user1",
            "iss": "wrong-issuer",
            "iat": now,
            "exp": now + 3600,
        }
        token = jwt.encode(payload, _TEST_SECRET, algorithm="HS256")
        result = await verifier.verify_token(token)
        assert result is None


class TestRoundTrip:
    """Create + verify round-trip tests."""

    async def test_round_trip(self):
        verifier = _make_verifier()
        token = verifier.create_token("roundtrip-user")
        result = await verifier.verify_token(token)
        assert result is not None
        assert result.client_id == "roundtrip-user"

    async def test_round_trip_with_scopes(self):
        verifier = _make_verifier()
        token = verifier.create_token("user1", scopes=["admin", "write"])
        result = await verifier.verify_token(token)
        assert result is not None
        assert "admin" in result.scopes
        assert "write" in result.scopes

    async def test_api_key_takes_precedence_over_jwt(self):
        """If a token matches an API key, it's used even if it's a valid JWT."""
        verifier = _make_verifier(api_keys="some-token=api-user")
        result = await verifier.verify_token("some-token")
        assert result is not None
        assert result.client_id == "api-user"
        assert result.claims["auth_method"] == "api_key"

    async def test_jwt_expires_at_set(self):
        verifier = _make_verifier()
        token = verifier.create_token("user1", expires_in=3600)
        result = await verifier.verify_token(token)
        assert result is not None
        assert result.expires_at is not None
        assert result.expires_at > int(time.time())
