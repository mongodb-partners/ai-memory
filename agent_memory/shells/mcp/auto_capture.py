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


class AutoCaptureMiddleware:
    """Transport-layer auto-capture, persisting via the facade ``app``."""

    def __init__(self, app, config: MCPConfig) -> None:
        self.app = app
        self.config = config

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
        content = f"Tool: {tool_name} | Query: {params} | Result: {response}"
        max_len = self.config.auto_capture_max_content_length
        return content[:max_len] if len(content) > max_len else content

    async def capture(self, tool_name: str, params: dict, response: dict) -> None:
        """Fire-and-forget memory storage via ``app.add``. Failures are logged."""
        if not self.should_capture(tool_name, params):
            return
        content = self.build_content(tool_name, params, response)
        if len(content) < self.config.auto_capture_min_length:
            return
        try:
            await self.app.add(
                params["user_id"],
                f"auto:{tool_name}",
                [{"role": "system", "message_type": "system", "content": content}],
            )
        except Exception:
            logger.warning("Auto-capture failed for %s", tool_name, exc_info=True)


def wrap_tools(mcp, auto_capture: "AutoCaptureMiddleware") -> None:
    """Wrap registered MCP tools so each fires auto-capture after execution."""
    for key, component in list(mcp.local_provider._components.items()):
        if not key.startswith("tool:"):
            continue
        original_fn = component.fn
        tool_name = component.name

        @functools.wraps(original_fn)
        async def wrapped(*args, _original=original_fn, _name=tool_name, **kwargs):
            result = await _original(*args, **kwargs)
            asyncio.create_task(auto_capture.capture(_name, kwargs, {"result": str(result)}))
            return result

        component.fn = wrapped
