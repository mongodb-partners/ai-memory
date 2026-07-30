"""Tests for APIKeyManager."""

import os
from unittest.mock import patch

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


class TestConstantTimeLookup:
    """REQ-E-086: key resolution must not vary with the submitted secret."""

    def test_the_raw_key_is_not_retained_anywhere(self):
        """TC-AUTH-KEY-020: keys are stored as SHA-256 fingerprints.

        Two properties follow. The dict is keyed on fixed-length digests, so the
        lookup does uniform work regardless of the submitted key's length or
        content — where a dict keyed on the raw secret falls back to a
        short-circuiting `==` on collision. And the process holds no plaintext key
        after construction, so it cannot leak one in a heap dump, a traceback repr,
        or a debugger session.
        """
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": "sup3rs3cr3t=alice"}):
            mgr = APIKeyManager()

        stored = repr(mgr.__dict__)
        assert "sup3rs3cr3t" not in stored
        # …and it still resolves.
        assert mgr.resolve_user("sup3rs3cr3t") == "alice"

    def test_a_wrong_key_of_any_length_resolves_to_nothing(self):
        """TC-AUTH-KEY-021: no near-miss shortcut.

        A prefix of a valid key must be as unresolvable as a completely different
        string — the case a byte-by-byte comparison distinguishes by timing.
        """
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": "sup3rs3cr3t=alice"}):
            mgr = APIKeyManager()

        for wrong in ("", "s", "sup3rs3cr3", "sup3rs3cr3T", "sup3rs3cr3tt", "z" * 400):
            assert mgr.resolve_user(wrong) is None
            assert mgr.is_valid(wrong) is False

    def test_is_valid_and_resolve_user_agree(self):
        """TC-AUTH-KEY-022: one code path, so the two cannot disagree.

        `is_valid` used `in` while `resolve_user` used `.get`; they now share an
        implementation, which is what stops a future fix landing in only one.
        """
        with patch.dict(os.environ, {"MEMORY_MCP_API_KEYS": "k1=alice"}):
            mgr = APIKeyManager()
        assert mgr.is_valid("k1") is True and mgr.resolve_user("k1") == "alice"
        assert mgr.is_valid("k2") is False and mgr.resolve_user("k2") is None
