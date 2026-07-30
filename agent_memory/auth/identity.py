"""Resolve *who the caller is* from a verified token — one implementation.

The shells used to take `user_id` from the request: a field in the JSON body for
REST, a tool argument for MCP. Authentication proved only that the caller held
*some* valid token; the identity it acted as was whatever it typed. Any token
holder could read, modify, or wipe any other tenant's memory by naming them, and
the query layer would scope the search faithfully to the named victim.

`auth_user_id_claim` and `auth_role_claim` existed in the config for this and were
read nowhere.

So identity resolution lives here, in one function, used by both shells:

- **Auth on** — the identity is the token's. A caller-supplied `user_id` is
  honoured only when it matches, and otherwise refused; it is never silently
  rewritten, because a request asking for someone else's data is a request whose
  author is confused about whose data it is, and answering a different question
  than the one asked hides that.
- **Auth off** — the caller-supplied value is all there is. That is the documented
  single-tenant posture, and it is why `require_auth_for_multi_tenant` exists to
  make choosing it deliberate.

Roles resolve the same way and from the same token, which is what makes the
governance profiles reachable at all: `_check_access` defaulted every caller to
`auth_default_role`, so the `admin` profile could not be selected by any request
and admin-only operations were either unreachable or open to everyone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class IdentityError(Exception):
    """The caller asked to act as an identity its token does not grant.

    Distinct from an authentication failure — the token is valid. The shells map
    this to 403, not 401: retrying with the same token will not help, and the
    fix is to stop naming someone else.
    """


@dataclass(frozen=True)
class Caller:
    """The authenticated principal, as the facade needs it.

    `role` is None when no role claim is present, which means "use the configured
    default" rather than "no permissions" — an unclaimed role is the common case
    for API-key callers.

    `authenticated` records *where the identity came from*, not whether it is
    valid. It is the field that lets a caller-supplied `user_id` be honoured with
    auth off and refused with auth on. It must be a stored fact rather than
    inferred from `role`/`scopes` being empty: a legitimate JWT can carry neither,
    so inferring would silently re-open the cross-tenant hole for exactly the
    tokens that look least remarkable.
    """

    user_id: str
    role: str | None = None
    scopes: tuple[str, ...] = ()
    authenticated: bool = False


def _claim(access, name: str):
    """Read a claim from an `AccessToken`, tolerating either shape.

    The API-key path builds `claims` itself; the JWT path passes the decoded
    payload through. Both are dicts, but `AccessToken` is a third-party model and
    `claims` has been optional in some versions, so this does not assume it exists.
    """
    claims = getattr(access, "claims", None) or {}
    value = claims.get(name)
    return value if value not in ("", None) else None


def resolve_caller(
    access,
    requested_user_id: str | None,
    config,
) -> Caller:
    """Return the identity this request may act as.

    `access` is the verified token, or None when auth is disabled.

    Raises
    ------
    IdentityError
        When a request names a `user_id` other than the token's own.
    """
    # ── Auth disabled: single-tenant, caller-supplied identity ──────────────
    if access is None:
        if not requested_user_id:
            raise IdentityError("user_id is required when auth is disabled")
        return Caller(user_id=requested_user_id)

    # ── Auth enabled: the token decides ─────────────────────────────────────
    claim_name = getattr(config, "auth_user_id_claim", "sub") or "sub"
    token_user = _claim(access, claim_name) or getattr(access, "client_id", None)
    if not token_user:
        # A token that authenticates but identifies no one cannot be scoped to a
        # tenant, and the safe reading of "no identity" is not "any identity".
        raise IdentityError(
            f"token carries no identity: claim {claim_name!r} is absent and "
            "the token has no client_id"
        )
    token_user = str(token_user)

    if requested_user_id and requested_user_id != token_user:
        # Logged at warning: in a multi-tenant deployment this is either a
        # confused client or an enumeration attempt, and both are worth seeing.
        logger.warning(
            "Refused cross-tenant request: token identifies %r, request named %r",
            token_user,
            requested_user_id,
        )
        raise IdentityError(
            "user_id does not match the authenticated identity; a token may only "
            "act as itself"
        )

    role_claim = getattr(config, "auth_role_claim", "role") or "role"
    role = _claim(access, role_claim)
    scopes = tuple(getattr(access, "scopes", ()) or ())
    return Caller(
        user_id=token_user,
        role=str(role) if role else None,
        scopes=scopes,
        authenticated=True,
    )


__all__ = ["Caller", "IdentityError", "resolve_caller"]
