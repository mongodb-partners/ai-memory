"""Make an exception safe to store in an audit record.

Audit entries are written to a MongoDB collection, retained on a TTL measured in
weeks, and read by whoever holds the ``admin`` role — a wider and longer-lived
audience than the process log. Passing ``str(exc)`` into one publishes whatever
the raising library chose to put in its message, and two families of library are
routinely careless about exactly that:

- **Driver errors quote the connection string.** A ``ServerSelectionTimeoutError``
  or an auth failure from PyMongo names the topology it failed to reach, and for a
  ``mongodb+srv://`` URI that string carries ``user:password@`` inline. The audit
  collection then holds cluster credentials, in the same database it authenticates
  to, on the tenant-readable side of the system.
- **Provider errors echo the request.** An HTTP 401 from an embedding or LLM
  endpoint frequently includes the offending ``Authorization`` header, and a 400
  often includes the payload — which for this library is the user's memory text.

There is a third leak with no credential in it at all: a duplicate-key error
quotes the key's *value*. On the episodic path that value is projected turn
content, so the failure that gets audited is also the failure that copies user
content out of the tier it was scoped to and into an admin-readable log.

So the rule here is allow-list-shaped rather than deny-list-shaped: keep the
exception's type and a scrubbed, length-capped message. The type is what makes an
audit entry actionable — ``ServerSelectionTimeoutError`` versus
``DuplicateKeyError`` versus ``ValueError`` is the whole diagnosis — and it can
never itself contain a secret. The message is best-effort.

This is not a general-purpose secret scanner and should not be sold as one. It
removes the shapes that this library's own dependencies actually emit. The
process log still gets the unredacted exception via ``exc_info``, because a
developer reading stderr on their own machine is a different threat model from a
row in a shared collection.
"""

from __future__ import annotations

import re

# Anything that looks like a URI with inline credentials. Both the `mongodb://`
# and `mongodb+srv://` forms, plus http(s) since provider SDKs echo endpoints.
_URI_CREDENTIALS = re.compile(
    r"\b([a-z][a-z0-9+.\-]*://)([^\s/@:]+)(:[^\s/@]*)?@",
    re.IGNORECASE,
)

# Bearer tokens and the common `key=value` secret spellings. The value pattern
# stops at whitespace, quotes, commas, and closing brackets so only the secret is
# taken and the surrounding message survives.
_BEARER = re.compile(r"\b(bearer\s+)[A-Za-z0-9._\-+/=]{8,}", re.IGNORECASE)
_ASSIGNED_SECRET = re.compile(
    r"""(?ix)
    \b(
        api[_\-]?key | secret | password | passwd | pwd | token |
        access[_\-]?key | auth
    )
    (\s*[=:]\s*|["']\s*:\s*["']?)
    [^\s,;'"})\]]+
    """
)

# Long opaque strings that are secrets by shape rather than by label: AWS access
# key ids, Voyage/OpenAI/Anthropic-style prefixed keys.
_PREFIXED_KEY = re.compile(
    r"\b(?:AKIA|ASIA|sk-|pa-|xoxb-|ghp_|gho_|github_pat_)[A-Za-z0-9_\-]{8,}"
)

_REDACTED = "[redacted]"

# Long enough to identify the fault, short enough that a driver dumping a full
# topology description does not become the largest field in the audit collection.
_MAX_LEN = 300


def redact_message(text: str) -> str:
    """Scrub credential-shaped substrings from a message and cap its length."""
    if not text:
        return ""
    # Credentials in a URI: keep the scheme and the user so the message still
    # identifies *which* principal failed, drop the password.
    out = _URI_CREDENTIALS.sub(rf"\1\2:{_REDACTED}@", text)
    out = _BEARER.sub(rf"\1{_REDACTED}", out)
    out = _ASSIGNED_SECRET.sub(rf"\1\g<2>{_REDACTED}", out)
    out = _PREFIXED_KEY.sub(_REDACTED, out)
    if len(out) > _MAX_LEN:
        out = out[:_MAX_LEN] + "…"
    return out


def redact_error(exc: BaseException) -> str:
    """Render an exception for an audit record: type name plus scrubbed message.

    Always returns a string, never raises. An exception whose ``__str__`` is
    itself broken still yields its class name, because losing the type would turn
    an actionable audit entry into an empty one.
    """
    kind = type(exc).__name__
    try:
        message = redact_message(str(exc))
    except Exception:  # pragma: no cover - a __str__ that raises
        return kind
    return f"{kind}: {message}" if message else kind


__all__ = ["redact_error", "redact_message"]
