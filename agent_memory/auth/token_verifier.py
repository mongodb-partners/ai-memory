"""Unified token verifier supporting API keys and HS256 JWT tokens.

Designed for use with FastMCP's built-in ``auth`` parameter.  When attached
to the MCP server, every HTTP request must carry a valid
``Authorization: Bearer <token>`` header.  The token can be either:

1. A **registered API key** from ``MEMORY_MCP_API_KEYS``.
2. A **JWT** signed with the ``auth_secret`` shared secret.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import jwt
from fastmcp.server.auth import AccessToken, TokenVerifier

from agent_memory.auth.api_keys import APIKeyManager

logger = logging.getLogger(__name__)

_JWT_ALGORITHM = "HS256"


class MemoryMCPTokenVerifier(TokenVerifier):
    """Verify Bearer tokens as API keys or HS256 JWTs.

    Parameters
    ----------
    secret:
        Shared secret used to sign / verify JWT tokens.
    api_key_manager:
        Pre-initialised :class:`APIKeyManager` for static API key lookups.
    issuer:
        Expected ``iss`` claim in JWTs.  Defaults to ``"memory-mcp"``.
    """

    def __init__(
        self,
        secret: str,
        api_key_manager: APIKeyManager | None = None,
        issuer: str = "memory-mcp",
    ) -> None:
        super().__init__()
        self._secret = secret
        self._api_key_manager = api_key_manager or APIKeyManager()
        self._issuer = issuer

    def create_token(
        self,
        user_id: str,
        expires_in: int = 86400,
        scopes: list[str] | None = None,
    ) -> str:
        """Create a signed JWT for *user_id*.

        Parameters
        ----------
        user_id:
            Identity to embed as the ``sub`` claim.
        expires_in:
            Token lifetime in seconds (default 24 h).
        scopes:
            Optional OAuth-style scopes stored in the ``scope`` claim.

        Returns
        -------
        str
            Encoded JWT string.
        """
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": user_id,
            "iss": self._issuer,
            "iat": now,
            "exp": now + expires_in,
        }
        if scopes:
            payload["scope"] = " ".join(scopes)
        return jwt.encode(payload, self._secret, algorithm=_JWT_ALGORITHM)

    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify a Bearer token (API key or JWT).

        Resolution order:

        1. Check if *token* is a registered API key → return identity.
        2. Attempt to decode *token* as a JWT signed with the shared secret.
        3. Return ``None`` if neither strategy succeeds.
        """
        # Strategy 1: static API key
        user_id = self._api_key_manager.resolve_user(token)
        if user_id is not None:
            logger.debug("Authenticated via API key: %s", user_id)
            return AccessToken(
                token=token,
                client_id=user_id,
                scopes=["memory-mcp"],
                claims={"sub": user_id, "auth_method": "api_key"},
            )

        # Strategy 2: JWT
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[_JWT_ALGORITHM],
                issuer=self._issuer,
                # `exp` is required, not merely honoured when present.
                #
                # PyJWT validates `exp` if the claim exists and ignores its absence
                # otherwise, so a token minted without one was accepted forever.
                # `create_token` always sets it, which is what made this easy to
                # miss — but the verifier's job is to police tokens it did not
                # mint, and anyone holding the shared secret (a second service, a
                # test fixture, an old script) could produce an immortal one. The
                # downstream code then treated `expires_at=None` as "no expiry"
                # rather than "unknown", so the token never came back for
                # revocation and there is no other revocation path here: a leaked
                # HS256 token with no `exp` is valid until the secret is rotated.
                #
                # `iat` is required too. Without it a token carries no issue time,
                # so nothing can distinguish one minted a minute ago from one
                # minted last year, and "reject everything issued before the
                # breach" stops being available as an incident response.
                options={"require": ["exp", "iat", "sub"]},
            )
        except jwt.ExpiredSignatureError:
            logger.debug("JWT expired")
            return None
        except jwt.MissingRequiredClaimError as exc:
            # Its own branch because this one is worth distinguishing in a log: a
            # token that is correctly signed but missing `exp` is a minting bug in
            # whatever produced it, not a forgery attempt, and the operator needs
            # to know which of the two they are looking at.
            logger.warning(
                "JWT rejected: %s. A token signed with this secret but lacking an "
                "expiry would never expire; whatever minted it must set exp/iat.",
                exc,
            )
            return None
        except jwt.InvalidTokenError as exc:
            logger.debug("JWT validation failed: %s", exc)
            return None

        sub = claims.get("sub")
        if not sub:
            logger.debug("JWT missing 'sub' claim")
            return None

        scopes_str = claims.get("scope", "")
        scopes = scopes_str.split() if scopes_str else []

        try:
            expires_at = int(claims["exp"])
        except (KeyError, TypeError, ValueError):
            # `require` above means this is unreachable for a well-formed claim, so
            # reaching it means `exp` is present but not an integer. Refuse rather
            # than fall back to None, which is the very "never expires" state the
            # requirement exists to prevent.
            logger.warning("JWT rejected: 'exp' is present but not an integer.")
            return None

        return AccessToken(
            token=token,
            client_id=str(sub),
            scopes=scopes,
            expires_at=expires_at,
            claims=claims,
        )
