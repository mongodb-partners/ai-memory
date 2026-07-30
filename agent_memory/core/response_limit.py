"""Enforce ``max_response_bytes`` on a list of result documents.

``max_response_bytes`` was declared in the config, documented in the config table,
and asserted by a unit test — and read by nothing. A limit that exists only as a
setting is worse than no limit: it is the setting an operator lowers when they
have a problem, and lowering it changed nothing at all.

The overflow is real and does not need adversarial input to reach. ``limit``
bounds the result *count*, not its size, and an episodic document carries
projected message content, a todo list, and a files-touched array. A hundred turns
of a long agent conversation is tens of megabytes, which lands as one MCP frame or
one JSON response body. What breaks downstream is not this process — it is the
client that has to buffer it, or the model whose context it is pasted into.

Truncation is by whole documents, never by bytes. A response cut mid-document is
invalid JSON at best and, once re-serialised, a document with silently missing
fields at worst — the same class of failure as the auto-capture mid-value cut.
Dropping the tail keeps every document that survives completely intact.

The response says when this happened. A silently shortened list is
indistinguishable from a short one, and "recall returned 12 results" versus
"recall returned 12 of 40, capped at 16 MiB" is the difference between an operator
raising the cap and an operator concluding the memory store is empty.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Rough per-document overhead of the surrounding JSON (comma, whitespace) plus
# the enclosing envelope. Small and deliberately approximate: this guards a
# multi-megabyte ceiling, so being a few bytes out is irrelevant, and paying for
# an exact serialisation of the whole list to find out would defeat the purpose.
_ENVELOPE_BYTES = 64


def _size_of(doc: Any) -> int:
    """Serialised size of one document in bytes.

    ``default=str`` so a BSON type that survived sanitisation (an ObjectId, a
    datetime) is measured rather than raising. Measuring is best-effort: a
    document that cannot be sized at all is charged a nominal cost rather than
    blocking the response.
    """
    try:
        return len(json.dumps(doc, default=str).encode("utf-8"))
    except (TypeError, ValueError):  # pragma: no cover - default=str covers most
        return _ENVELOPE_BYTES


def cap_results(
    results: list[dict], max_bytes: int, *, label: str = "results"
) -> tuple[list[dict], dict[str, Any]]:
    """Return ``(kept, meta)`` — the prefix that fits, plus what was dropped.

    ``meta`` is empty when nothing was dropped, so a caller can splat it into the
    response and the common path stays byte-identical to before this existed.
    When truncation happens it carries ``truncated``, ``total_count``, and
    ``max_response_bytes`` — enough for the caller to know it happened, how much
    it lost, and which knob to turn.

    At least one document is always kept, even if it alone exceeds the cap.
    Returning an empty list for an oversized single document would turn a
    too-large answer into no answer, and the caller has no way to ask for less.
    """
    if max_bytes <= 0 or not results:
        return results, {}

    budget = max_bytes - _ENVELOPE_BYTES
    kept: list[dict] = []
    used = 0
    for doc in results:
        size = _size_of(doc) + 1  # +1 for the separating comma
        if kept and used + size > budget:
            break
        kept.append(doc)
        used += size

    if len(kept) == len(results):
        return results, {}

    logger.warning(
        "Response capped: returning %d of %d %s (~%d bytes, max_response_bytes=%d). "
        "Lower the request limit or raise max_response_bytes.",
        len(kept), len(results), label, used, max_bytes,
    )
    return kept, {
        "truncated": True,
        "total_count": len(results),
        "max_response_bytes": max_bytes,
    }


__all__ = ["cap_results"]
