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


class TestExpiryIsMandatory:
    """REQ-E-085: a token with no expiry must be refused, not accepted forever."""

    @staticmethod
    def _mint(**payload):
        """Sign a payload directly, bypassing `create_token`.

        Necessary because `create_token` always sets `exp` — which is exactly why
        this gap survived review. The verifier's job is to police tokens it did not
        mint, and anyone holding the shared secret can produce these.
        """
        base = {"sub": "u1", "iss": "memory-mcp", "iat": int(time.time())}
        base.update(payload)
        return jwt.encode(base, _TEST_SECRET, algorithm="HS256")

    async def test_a_token_without_exp_is_refused(self):
        """TC-AUTH-JWT-020: PyJWT ignores a missing `exp` rather than failing.

        Before `options={"require": [...]}`, this token verified successfully and
        produced `expires_at=None`, which downstream code reads as "no expiry"
        rather than "unknown". There is no revocation path for an HS256 token here,
        so a leaked one stayed valid until the operator rotated the secret — an
        action nobody takes until they know there is a reason to.
        """
        verifier = _make_verifier()
        assert await verifier.verify_token(self._mint()) is None

    async def test_a_token_without_iat_is_refused(self):
        """TC-AUTH-JWT-021: no issue time means no "revoke everything before X".

        Without `iat` a token cannot be placed in time, so the cheapest incident
        response available for a shared-secret scheme — reject anything minted
        before the breach — is not available at all.
        """
        verifier = _make_verifier()
        token = jwt.encode(
            {"sub": "u1", "iss": "memory-mcp", "exp": int(time.time()) + 3600},
            _TEST_SECRET, algorithm="HS256",
        )
        assert await verifier.verify_token(token) is None

    async def test_a_non_integer_exp_is_refused_rather_than_nulled(self):
        """TC-AUTH-JWT-022: the fallback must not recreate the hole it closed.

        `int(exp) if exp else None` turned an unparseable expiry into "never
        expires" — the failure mode is the same one, reached by a different route.
        """
        verifier = _make_verifier()
        token = self._mint(exp="not-a-timestamp")
        assert await verifier.verify_token(token) is None

    async def test_a_well_formed_token_still_verifies(self):
        """TC-AUTH-JWT-023: `create_token` output must remain acceptable."""
        verifier = _make_verifier()
        result = await verifier.verify_token(verifier.create_token("u1"))
        assert result is not None and result.expires_at > int(time.time())
