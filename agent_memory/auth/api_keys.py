"""API key management for multi-user authentication.

Loads API key -> user_id mappings from the MEMORY_MCP_API_KEYS
environment variable and provides lookup / validation helpers.

Environment variable format:
    MEMORY_MCP_API_KEYS="key1=user1@company.com,key2=user2@company.com"
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

logger = logging.getLogger(__name__)

_ENV_VAR = "MEMORY_MCP_API_KEYS"


def _fingerprint(api_key: str) -> str:
    """A fixed-length digest of a key, used as the dictionary key.

    Hashing before the lookup is what makes the lookup constant-time with respect
    to the *secret*. A dict keyed on the raw API key compares strings on a hash
    collision and — more importantly — the sequence of work done before that point
    varies with the key's length and content. Every digest here is 32 bytes, so
    the dict sees uniform input regardless of what was submitted.

    SHA-256 rather than a password KDF on purpose. An API key is high-entropy
    random material chosen by the operator, not a human-memorable password, so the
    attack a KDF defends against (offline brute force over a small space) does not
    apply, and a per-lookup KDF would put tens of milliseconds on every
    authenticated request. This is a lookup key, not a stored credential.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class APIKeyManager:
    """Manages the mapping between API keys and user identities.

    Keys are loaded once from the ``MEMORY_MCP_API_KEYS`` environment
    variable at construction time.  The expected format is a
    comma-separated list of ``key=user_id`` pairs::

        export MEMORY_MCP_API_KEYS="abc123=alice@acme.com,xyz789=bob@acme.com"

    Leading/trailing whitespace on both keys and user IDs is stripped.

    Keys are stored as SHA-256 fingerprints rather than in the clear, and every
    lookup goes through :func:`_fingerprint`. See :meth:`resolve_user` for why.
    """

    def __init__(self) -> None:
        # Fingerprint -> (user_id, fingerprint). The redundant fingerprint is what
        # `resolve_user` feeds to `hmac.compare_digest`; see there.
        self._key_to_user: dict[str, tuple[str, str]] = {}
        self._load_from_env()

    def resolve_user(self, api_key: str) -> str | None:
        """Return the user ID associated with *api_key*, or ``None``.

        Constant-time with respect to the submitted key, which the plain
        ``dict.get(api_key)`` this replaces was not.

        Two things leaked. The dict was keyed on the raw secret, so the work done
        before a decision varied with the key's length and its content — and on a
        hash collision Python falls back to ``==`` on the raw strings, which
        short-circuits at the first differing byte. And the *hit* path did strictly
        more work than the miss path in a way an attacker can time.

        The exposure here is narrower than it is for a password, because these keys
        are high-entropy operator-chosen strings rather than guessable secrets, and
        remote timing over a network is noisy. But the fix costs one hash per
        request and removes the question entirely, and this is the function that
        turns a bearer string into an identity — the single place in the library
        where getting it wrong means impersonation. Cheap and total beats narrow and
        argued.

        Hashing alone would not be enough: the dict lookup is uniform, but the
        returned value must still be checked against the submitted fingerprint
        without a short-circuiting comparison, which is what
        ``hmac.compare_digest`` is for.
        """
        entry = self._key_to_user.get(_fingerprint(api_key))
        if entry is None:
            return None
        user_id, stored = entry
        # Belt and braces: confirm the match without a short-circuiting compare, so
        # a digest collision cannot resolve to someone else's identity.
        if not hmac.compare_digest(stored, _fingerprint(api_key)):
            return None  # pragma: no cover - requires a SHA-256 collision
        return user_id

    def is_valid(self, api_key: str) -> bool:
        """Return ``True`` if *api_key* is registered. Constant-time; see
        :meth:`resolve_user`."""
        return self.resolve_user(api_key) is not None

    def list_users(self) -> list[str]:
        """Return a sorted list of all registered user IDs."""
        return sorted({user_id for user_id, _ in self._key_to_user.values()})

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

            digest = _fingerprint(key)
            if digest in self._key_to_user:
                logger.warning(
                    "Duplicate API key in %s — overwriting previous mapping for key %r",
                    _ENV_VAR,
                    key[:4] + "****",
                )

            # The raw key is deliberately not retained. Nothing needs it after
            # this point, and a process that holds no plaintext keys cannot leak
            # them in a heap dump, a traceback repr, or a debugger session.
            self._key_to_user[digest] = (user_id, digest)

        logger.info(
            "Loaded %d API key(s) from %s for %d user(s)",
            len(self._key_to_user),
            _ENV_VAR,
            len({user_id for user_id, _ in self._key_to_user.values()}),
        )
