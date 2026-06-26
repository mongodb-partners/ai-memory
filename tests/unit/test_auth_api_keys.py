"""Tests for APIKeyManager."""

import os
from unittest.mock import patch

import pytest

from agent_memory.auth.api_keys import APIKeyManager


class TestAPIKeyManagerLoad:
    """APIKeyManager loads keys from environment."""

    def test_load_valid_keys(self):
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": "key1=alice,key2=bob"}):
            mgr = APIKeyManager()
        assert mgr.resolve_user("key1") == "alice"
        assert mgr.resolve_user("key2") == "bob"

    def test_valid_key_returns_user(self):
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": "mykey=user@test.com"}):
            mgr = APIKeyManager()
        assert mgr.is_valid("mykey")
        assert mgr.resolve_user("mykey") == "user@test.com"

    def test_invalid_key_returns_none(self):
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": "mykey=user1"}):
            mgr = APIKeyManager()
        assert mgr.resolve_user("wrongkey") is None
        assert not mgr.is_valid("wrongkey")

    def test_list_users(self):
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": "k1=alice,k2=bob,k3=alice"}):
            mgr = APIKeyManager()
        users = mgr.list_users()
        assert users == ["alice", "bob"]

    def test_empty_env_var(self):
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": ""}):
            mgr = APIKeyManager()
        assert mgr.list_users() == []
        assert mgr.resolve_user("anything") is None

    def test_missing_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if present
            os.environ.pop("MEMORY_MCP_API_KEYS", None)
            mgr = APIKeyManager()
        assert mgr.list_users() == []

    def test_malformed_entry_skipped(self):
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": "valid=user,malformed_no_equals"}):
            mgr = APIKeyManager()
        assert mgr.resolve_user("valid") == "user"
        assert len(mgr.list_users()) == 1

    def test_whitespace_tolerance(self):
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": " key1 = alice , key2 = bob "}):
            mgr = APIKeyManager()
        assert mgr.resolve_user("key1") == "alice"
        assert mgr.resolve_user("key2") == "bob"
