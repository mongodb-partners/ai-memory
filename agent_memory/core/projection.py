"""Pure projection helpers — turn agent messages and state into log documents.

These functions are framework-agnostic and do no I/O. They accept either plain
dicts or objects with attributes (``message.content``, ``message.tool_calls``),
so they work with any agent framework's message type without importing it.

The projection is deliberately lossy. An episodic log stores what a *reader*
needs — what was said, which tools ran, which files changed — not a faithful
serialization of framework internals. Everything not in the seven-key message
shape is dropped on purpose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_ALLOWED_TODO_STATUS = ("pending", "in_progress", "completed")

# Tool names that mean "this call wrote to the filesystem". Used to derive
# ``files_touched`` from tool calls. Callers with different tool names pass
# their own set; the ``op`` label is derived by membership, not by name.
DEFAULT_FS_WRITE_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file"})

# Tool names that create a file rather than modify one. Anything in
# ``fs_write_tools`` but not here is labelled ``edit``.
DEFAULT_FS_CREATE_TOOLS: frozenset[str] = frozenset({"write_file", "create_file"})


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a Mapping or an attribute from an object.

    Message objects across frameworks are either dicts or attribute-bearing
    classes. Supporting both here is what keeps this module neutral — an
    ``getattr``-only version silently projects empty documents for dict input,
    which is the most common shape when messages arrive over HTTP.
    """
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def coerce_content(message: Any) -> str:
    """Project a message's ``content`` to a plain string.

    Several providers return content as a list of structured blocks rather than
    a string. The list form collapses to text: ``{"type": "text"}`` blocks and
    bare strings are kept, everything else (tool-use blocks, images) is dropped.
    """
    content = _get(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def truncate(text: str, cap: int) -> str:
    """Cap ``text`` at ``cap`` characters, appending a visible marker.

    ``cap <= 0`` disables truncation. The marker reports the original length so
    a reader can tell how much was lost.
    """
    if cap <= 0 or len(text) <= cap:
        return text
    return text[:cap] + f"\n[truncated, original_size={len(text)} bytes]"


def project_messages(raw: Iterable[Any], *, cap: int) -> list[dict[str, Any]]:
    """Project messages to the canonical seven-key shape, in order.

    Key order is part of the stored document contract — see
    ``docs/reference/episodic-document-shape.md``.
    """
    out: list[dict[str, Any]] = []
    for message in raw:
        extra = _get(message, "additional_kwargs", {}) or {}
        if not isinstance(extra, Mapping):
            extra = {}
        out.append(
            {
                "type": _get(message, "type", "ai"),
                "content": truncate(coerce_content(message), cap),
                "tool_calls": list(_get(message, "tool_calls", []) or []),
                "tool_call_id": _get(message, "tool_call_id"),
                "usage": _get(message, "usage_metadata"),
                # Providers put these on the message envelope, not the body.
                "model_id": extra.get("model_id"),
                "finish_reason": extra.get("stop_reason"),
            }
        )
    return out


def project_todos(raw: Any) -> list[dict[str, Any]]:
    """Project a todo list to ``id`` / ``content`` / ``status`` triples.

    Unknown statuses clamp to ``pending`` rather than raising — a malformed
    todo should not cost you the whole logged turn. ``text`` is accepted as an
    alias for ``content`` because several agent frameworks use it.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for todo in raw:
        if not isinstance(todo, Mapping):
            continue
        status = todo.get("status")
        if status not in _ALLOWED_TODO_STATUS:
            status = "pending"
        out.append(
            {
                "id": str(todo.get("id", "")),
                "content": str(todo.get("content") or todo.get("text") or ""),
                "status": status,
            }
        )
    return out


def project_files(
    messages: Iterable[Any],
    *,
    fs_write_tools: frozenset[str] = DEFAULT_FS_WRITE_TOOLS,
    fs_create_tools: frozenset[str] = DEFAULT_FS_CREATE_TOOLS,
) -> list[dict[str, Any]]:
    """Derive ``files_touched`` from tool calls on assistant messages.

    Latest call per path wins, so a write followed by an edit reports a single
    entry whose ``op`` is ``edit``. Read-only tools are ignored entirely.
    Results are sorted by path so two logs of the same turn compare equal.

    ``op`` is ``write`` when the tool is in ``fs_create_tools``, else ``edit``.
    Passing a custom ``fs_write_tools`` without a matching ``fs_create_tools``
    would otherwise label every custom tool as an edit.
    """
    seen: dict[str, dict[str, Any]] = {}
    for message in messages:
        if _get(message, "type") != "ai":
            continue
        for call in _get(message, "tool_calls", None) or []:
            if not isinstance(call, Mapping):
                continue
            name = call.get("name") or ""
            if name not in fs_write_tools:
                continue
            args = call.get("args") or {}
            if not isinstance(args, Mapping):
                continue
            path = args.get("file_path") or args.get("path")
            if not isinstance(path, str) or not path:
                continue
            size = 0
            content = args.get("content")
            if isinstance(content, (str, bytes)):
                size = len(content)
            else:
                new_string = args.get("new_string")
                if isinstance(new_string, (str, bytes)):
                    size = len(new_string)
            seen[path] = {
                "path": path,
                "size": size,
                # Reserved: callers that hash file contents themselves can
                # populate this. The projection never reads the filesystem.
                "content_hash": None,
                "op": "write" if name in fs_create_tools else "edit",
            }
    out = list(seen.values())
    out.sort(key=lambda entry: entry.get("path", ""))
    return out


def is_final_step(messages_proj: Sequence[Mapping[str, Any]]) -> bool:
    """True when the turn ended in an answer rather than another tool call.

    Only final steps are worth embedding: a mid-turn step whose last message is
    a tool request has no answer text to search on.
    """
    last_ai = next(
        (m for m in reversed(messages_proj) if m.get("type") == "ai"), None
    )
    if last_ai is None:
        return False
    return not (last_ai.get("tool_calls") or [])


def build_search_text(
    messages_proj: Sequence[Mapping[str, Any]], *, cap: int
) -> str:
    """Join the first user message and the last assistant message for search.

    Question plus answer is what a later reader searches for; the intermediate
    tool chatter mostly adds noise to the embedding. Returns ``""`` when either
    half is missing, which suppresses both the embedding and the search field.
    """
    human = next((m for m in messages_proj if m.get("type") == "human"), None)
    ai = next((m for m in reversed(messages_proj) if m.get("type") == "ai"), None)
    if human is None or ai is None:
        return ""
    text = (human.get("content") or "") + "\n\n" + (ai.get("content") or "")
    if 0 < cap < len(text):
        text = text[:cap]
    return text


__all__ = [
    "DEFAULT_FS_CREATE_TOOLS",
    "DEFAULT_FS_WRITE_TOOLS",
    "build_search_text",
    "coerce_content",
    "is_final_step",
    "project_files",
    "project_messages",
    "project_todos",
    "truncate",
]
