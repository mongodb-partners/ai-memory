"""Typed exception hierarchy for agent-memory.

The facade raises these; each shell translates them to its own error shape
(MCP ``{"error": ...}``; REST HTTP status codes). Keeping error *policy* out of
the core is what lets one facade serve every transport.

``RateLimitError`` subclasses ``AccessError`` so a caller can catch the base
while shells distinguish throttling (429) from denial (403).
"""


class MemoryError(Exception):
    """Base class for all agent-memory errors."""


class AccessError(MemoryError):
    """Operation denied by governance / access control."""


class RateLimitError(AccessError):
    """Rate limit exceeded. Subclass of AccessError; shells map it to 429."""


class NotFoundError(MemoryError):
    """A requested resource (memory, decision, …) does not exist."""


class ConfigError(MemoryError):
    """Invalid or inconsistent configuration (e.g. embedding-dimension
    mismatch, or a selected provider whose SDK is not installed)."""
