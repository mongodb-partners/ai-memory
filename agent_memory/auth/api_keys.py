"""API key management for multi-user authentication.

Loads API key -> user_id mappings from the MEMORY_MCP_API_KEYS
environment variable and provides lookup / validation helpers.

Environment variable format:
    MEMORY_MCP_API_KEYS="key1=user1@company.com,key2=user2@company.com"
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ENV_VAR = "MEMORY_MCP_API_KEYS"


class APIKeyManager:
    """Manages the mapping between API keys and user identities.

    Keys are loaded once from the ``MEMORY_MCP_API_KEYS`` environment
    variable at construction time.  The expected format is a
    comma-separated list of ``key=user_id`` pairs::

        export MEMORY_MCP_API_KEYS="abc123=alice@acme.com,xyz789=bob@acme.com"

    Leading/trailing whitespace on both keys and user IDs is stripped.
    """

    def __init__(self) -> None:
        self._key_to_user: dict[str, str] = {}
        self._load_from_env()

    def resolve_user(self, api_key: str) -> str | None:
        """Return the user ID associated with *api_key*, or ``None``."""
        return self._key_to_user.get(api_key)

    def is_valid(self, api_key: str) -> bool:
        """Return ``True`` if *api_key* is registered."""
        return api_key in self._key_to_user

    def list_users(self) -> list[str]:
        """Return a sorted list of all registered user IDs."""
        return sorted(set(self._key_to_user.values()))

    def _load_from_env(self) -> None:
        """Parse ``MEMORY_MCP_API_KEYS`` and populate the internal map."""
        raw = os.environ.get(_ENV_VAR, "")
        if not raw.strip():
            logger.debug(
                "%s is not set or empty — API-key authentication disabled",
                _ENV_VAR,
            )
            return

        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                logger.warning(
                    "Skipping malformed entry in %s (no '=' found): %r",
                    _ENV_VAR,
                    entry,
                )
                continue

            key, _, user_id = entry.partition("=")
            key = key.strip()
            user_id = user_id.strip()

            if not key or not user_id:
                logger.warning(
                    "Skipping entry with empty key or user_id in %s: %r",
                    _ENV_VAR,
                    entry,
                )
                continue

            if key in self._key_to_user:
                logger.warning(
                    "Duplicate API key in %s — overwriting previous mapping for key %r",
                    _ENV_VAR,
                    key[:4] + "****",
                )

            self._key_to_user[key] = user_id

        logger.info(
            "Loaded %d API key(s) from %s for %d user(s)",
            len(self._key_to_user),
            _ENV_VAR,
            len(set(self._key_to_user.values())),
        )
