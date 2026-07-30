"""Auto-capture middleware for the MCP shell.

Intercepts MCP tool call/response pairs and opportunistically persists
significant interactions, so the store is populated even when the agent never
calls ``store_memory``. Unlike the substrate version, capture goes through the
facade (``app.add``) so it inherits the same access-check + audit path as every
other write.

Auto-capture is MCP-only by design; REST is the explicit-control surface.
"""

from __future__ import annotations

import asyncio
import functools
import logging

from agent_memory.auth.identity import IdentityError, resolve_caller
from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)

# Tools that must never be auto-captured regardless of config.
#
# ``log_activity`` is here for a sharper reason than the others: capturing it
# would store a memory *about* the turn log, and if that memory write were ever
# itself logged the two would feed each other. The episodic tier is already a
# complete record of what the agent did — capturing it again is pure
# amplification. ``set_activity_retention`` is an admin knob, not content.
_EXCLUDED_TOOLS = frozenset(
    {
        "store_memory", "wipe_user_data", "delete_memory", "cache_invalidate",
        "log_activity", "set_activity_retention",
    }
)

_ELLIPSIS = "…"


def _clip(text: str, budget: int) -> str:
    """Truncate to ``budget`` characters, marking the cut.

    The marker is the point: an unmarked truncation of a repr produces text that
    reads as complete and says something different from what happened.
    """
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    if budget <= len(_ELLIPSIS):
        return _ELLIPSIS[:budget]
    return text[: budget - len(_ELLIPSIS)] + _ELLIPSIS


class AutoCaptureMiddleware:
    """Transport-layer auto-capture, persisting via the facade ``app``."""

    def __init__(self, app, config: MCPConfig) -> None:
        self.app = app
        self.config = config
        # Strong references to in-flight capture tasks.
        #
        # `asyncio.create_task` returns the only strong reference to its task. The
        # loop holds a *weak* one, so a task whose handle is discarded — which the
        # bare `create_task(...)` in `wrap_tools` did — can be garbage-collected
        # mid-await and simply stop, with the write half-done and nothing raised.
        # It is a rare race that gets rarer under light load, which is the worst
        # possible profile: it will not reproduce in testing and will show up as
        # occasional missing memories in production.
        #
        # Discarding on completion keeps this bounded; it is not a queue.
        self._pending: set[asyncio.Task] = set()

    def spawn(
        self,
        tool_name: str,
        params: dict,
        response: dict,
        user_id: str | None = None,
    ) -> asyncio.Task | None:
        """Start a capture in the background, retaining a reference to it.

        Returns the task so a caller (or a test) can await it. Returns None when
        there is no running loop, which is the case in synchronous embedding
        contexts — capture is best-effort and must never be the reason a tool call
        fails.

        ``user_id`` is the resolved identity; see :meth:`capture`.
        """
        try:
            task = asyncio.create_task(
                self.capture(tool_name, params, response, user_id=user_id),
                name=f"agent-memory:auto-capture:{tool_name}",
            )
        except RuntimeError:  # pragma: no cover - no running loop
            logger.debug("Auto-capture skipped for %s: no running loop.", tool_name)
            return None
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def drain(self, timeout: float = 5.0) -> bool:
        """Wait for in-flight captures. Bounded; never raises.

        Auto-capture is fire-and-forget during a session, but a process shutting
        down mid-capture loses the write silently. This gives a shell somewhere to
        wait.
        """
        if not self._pending:
            return True
        try:
            await asyncio.wait_for(
                asyncio.gather(*list(self._pending), return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            return False
        return True

    def should_capture(self, tool_name: str, params: dict) -> bool:
        if not self.config.auto_capture_enabled:
            return False
        if tool_name in _EXCLUDED_TOOLS:
            return False
        if tool_name not in self.config.auto_capture_tools:
            return False
        if "user_id" not in params:
            return False
        return True

    def build_content(self, tool_name: str, params: dict, response: dict) -> str:
        """Render a captured call as memory text, truncating the *parts*.

        The single-slice version this replaces cut the joined string at
        ``max_content_length``, and where that cut landed decided what the memory
        said. A long ``params`` dict consumed the whole budget, so the text ended
        mid-key with the result — the actual outcome of the call, and the only part
        worth remembering — absent entirely. Worse, the cut lands inside a repr, so
        the stored memory reads as a complete sentence that happens to be false:
        ``Result: {'status': 'fail`` is indistinguishable to the embedder, to the
        enrichment LLM, and to a human reading recall output from a call that
        succeeded and got clipped. It then gets embedded and recalled as fact.

        Budgeting per part keeps all three fields present, and each truncation is
        marked with an ellipsis so a reader can see that something was dropped. The
        result gets the larger share because it is the part that says what happened.
        """
        max_len = max(0, self.config.auto_capture_max_content_length)
        frame = f"Tool: {tool_name} | Query:  | Result: "
        budget = max_len - len(frame)
        if budget <= 0:
            # Pathologically small cap: keep the tool name, which is the one field
            # that makes the record identifiable at all.
            return f"Tool: {tool_name}"[:max_len]

        # Two thirds to the result, one third to the query. The query is context;
        # the result is the outcome.
        result_budget = (budget * 2) // 3
        query_budget = budget - result_budget

        query_text = _clip(str(params), query_budget)
        result_text = _clip(str(response), result_budget)
        # Reclaim whatever the query did not use, rather than padding the cap out
        # with nothing — a short query should let a long result run further.
        slack = query_budget - len(query_text)
        if slack > 0:
            result_text = _clip(str(response), result_budget + slack)

        return f"Tool: {tool_name} | Query: {query_text} | Result: {result_text}"

    async def capture(
        self,
        tool_name: str,
        params: dict,
        response: dict,
        user_id: str | None = None,
    ) -> None:
        """Fire-and-forget memory storage via ``app.add``. Failures are logged.

        ``user_id`` is the *resolved* identity from ``wrap_tools``. It is a
        separate argument rather than a read of ``params["user_id"]`` because
        ``params`` is untrusted: it is the raw tool arguments, and on an
        auth-enabled deployment the ``user_id`` in there is whatever the client
        typed. When it is None the capture is dropped — see
        :func:`resolve_capture_identity` for why a refusal must not be stored.
        """
        if not self.should_capture(tool_name, params):
            return
        if user_id is None:
            # No resolved identity means the call was refused or the identity
            # could not be established. Falling back to `params["user_id"]` here
            # is the cross-tenant write: it would store the refusal text under
            # the very account the caller was just denied access to.
            logger.debug(
                "Auto-capture skipped for %s: no authorised identity.", tool_name
            )
            return
        content = self.build_content(tool_name, params, response)
        if len(content) < self.config.auto_capture_min_length:
            return
        try:
            await self.app.add(
                user_id,
                f"auto:{tool_name}",
                [{"role": "system", "message_type": "system", "content": content}],
            )
        except Exception:
            logger.warning("Auto-capture failed for %s", tool_name, exc_info=True)


def _was_refused(result) -> bool:
    """True when a tool returned an error dict rather than doing the work.

    The tools convert ``AccessError``/``IdentityError`` into ``{"error": ...}`` —
    a refusal is a return value, so a wrapper cannot use "did not raise" as
    evidence that the call was allowed.
    """
    return isinstance(result, dict) and "error" in result


def resolve_capture_identity(app, params: dict, result) -> str | None:
    """The identity a capture may be written under, or None to skip.

    This exists because auto-capture used to write under
    ``params["user_id"]`` — the raw, client-supplied tool argument — *after* the
    tool itself had already refused it. On an auth-enabled deployment that meant
    Alice could call any wrapped tool with ``user_id="bob"``, receive the refusal,
    and still have the refusal text embedded and stored in Bob's memory: a
    cross-tenant write through the one path that skipped identity resolution,
    and an injection vector, since the attacker controls the stored text.

    Two rules, both necessary:

    - A refused call is never captured. There is no correct account to store it
      under: not the victim's, and storing a denial under the caller's own
      account records a memory about an operation that did not happen.
    - Otherwise the identity is resolved the same way the tool resolved it, from
      the verified token. With auth off that is the caller-supplied value, which
      is the documented single-tenant posture.
    """
    if _was_refused(result):
        return None

    requested = params.get("user_id")
    config = getattr(app, "config", None)
    auth_on = bool(getattr(config, "auth_enabled", False))
    if not auth_on:
        # Single-tenant: the request's own value is the only identity there is.
        return requested if requested else None

    try:
        from fastmcp.server.dependencies import get_access_token

        access = get_access_token()
    except Exception as exc:
        # Matches `tools._who`: with auth on, a token we cannot read is a refusal,
        # not a licence to fall back to the request's own `user_id`.
        logger.warning("Auto-capture: could not read the access token: %s", exc)
        return None

    try:
        # `access` may still be None here — stdio and in-process calls have no
        # request. `resolve_caller` treats that as the single-tenant case, which is
        # the same answer the tool itself just used.
        return resolve_caller(access, requested, config).user_id
    except IdentityError:
        # Should be unreachable: the tool would have refused first and we would
        # have returned above. Kept because "unreachable" here means "silently
        # writes to the wrong tenant" if it ever stops being true.
        logger.warning("Auto-capture: refusing a capture whose identity resolution failed.")
        return None


def wrap_tools(mcp, auto_capture: AutoCaptureMiddleware) -> None:
    """Wrap registered MCP tools so each fires auto-capture after execution."""
    for key, component in list(mcp.local_provider._components.items()):
        if not key.startswith("tool:"):
            continue
        original_fn = component.fn
        tool_name = component.name

        @functools.wraps(original_fn)
        async def wrapped(*args, _original=original_fn, _name=tool_name, **kwargs):
            result = await _original(*args, **kwargs)
            # The identity is resolved here, not inside `capture` from `kwargs`:
            # `kwargs` is what the client sent, and a refused cross-tenant call
            # still returns normally.
            #
            # `spawn` rather than a bare `create_task`: it keeps a strong reference
            # so the capture cannot be garbage-collected mid-write.
            who = resolve_capture_identity(auto_capture.app, kwargs, result)
            auto_capture.spawn(_name, kwargs, {"result": str(result)}, user_id=who)
            return result

        component.fn = wrapped
