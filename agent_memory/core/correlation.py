"""Correlation-id derivation — tie a logged turn to your tracing stack.

An episodic record is much more useful when it can be joined to the trace, log
line, or support ticket that produced it. Rather than inventing a private id
scheme, this reuses whatever the caller already has, in priority order.

W3C ``traceparent`` support is the reason this is worth a module: any
OpenTelemetry-instrumented service already propagates it, so episodic records
land with the same trace id the rest of the stack uses — no extra plumbing.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

__all__ = ["derive_correlation_id"]


def derive_correlation_id(config: Mapping[str, Any] | None) -> str:
    """Pick a correlation id from ``config``, falling back to a fresh UUID4.

    Precedence, first non-empty wins:

    1. ``correlation_id`` — an explicit value always wins.
    2. ``traceparent`` — W3C trace context; the trace id is field 2 of
       ``version-traceid-spanid-flags``. The *trace* id is used rather than the
       span id so every turn in one request shares an id.
    3. ``x_request_id`` — the common reverse-proxy header.
    4. A new UUID4, so the return value is never empty.
    """
    if not config:
        return str(uuid.uuid4())

    explicit = config.get("correlation_id")
    if explicit:
        return str(explicit)

    traceparent = config.get("traceparent")
    if traceparent:
        parts = str(traceparent).split("-")
        if len(parts) >= 2 and parts[1]:
            return parts[1]

    request_id = config.get("x_request_id")
    if request_id:
        return str(request_id)

    return str(uuid.uuid4())
