"""Auto-capture middleware for transparent memory storage.

Intercepts MCP tool call/response pairs and automatically stores
significant interactions as STM memories, ensuring the memory store
is populated even when the LLM does not call store_memory.
"""

import asyncio
import functools
import logging

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)

# Tools that must never be auto-captured regardless of config
_EXCLUDED_TOOLS = frozenset({
    "store_memory",
    "wipe_user_data",
    "delete_memory",
    "cache_invalidate",
})


class AutoCaptureMiddleware:
    """Transport-layer auto-capture of tool interactions as memories."""

    def __init__(self, memory_service, config: MCPConfig) -> None:
        self.memory_service = memory_service
        self.config = config

    def should_capture(self, tool_name: str, params: dict) -> bool:
        """Determine if this tool call should be captured."""
        if not self.config.auto_capture_enabled:
            return False
        if tool_name in _EXCLUDED_TOOLS:
            return False
        if tool_name not in self.config.auto_capture_tools:
            return False
        if "user_id" not in params:
            return False
        return True

    def build_content(
        self, tool_name: str, params: dict, response: dict,
    ) -> str:
        """Build memory content from tool interaction."""
        content = f"Tool: {tool_name} | Query: {params} | Result: {response}"
        max_len = self.config.auto_capture_max_content_length
        if len(content) > max_len:
            return content[:max_len]
        return content

    async def capture(
        self, tool_name: str, params: dict, response: dict,
    ) -> None:
        """Fire-and-forget memory storage.

        Evaluates the interaction for capture eligibility, builds memory
        content, and stores as STM.  Failures are logged but never propagated.
        """
        if not self.should_capture(tool_name, params):
            return

        content = self.build_content(tool_name, params, response)
        if len(content) < self.config.auto_capture_min_length:
            return

        try:
            user_id = params["user_id"]
            conv_id = f"auto:{tool_name}"
            await self.memory_service.store_stm(
                user_id=user_id,
                conversation_id=conv_id,
                messages=[{"role": "system", "message_type": "system", "content": content}],
            )
        except Exception:
            logger.warning(
                "Auto-capture failed for %s", tool_name, exc_info=True,
            )


def wrap_tools(mcp, auto_capture: "AutoCaptureMiddleware") -> None:
    """Wrap all registered MCP tools with auto-capture.

    Iterates over registered tool components and replaces each tool's
    function with a wrapper that fires auto-capture after execution.
    """
    for key, component in list(mcp.local_provider._components.items()):
        if not key.startswith("tool:"):
            continue

        original_fn = component.fn
        tool_name = component.name

        @functools.wraps(original_fn)
        async def wrapped(
            *args,
            _original=original_fn,
            _name=tool_name,
            **kwargs,
        ):
            result = await _original(*args, **kwargs)
            asyncio.create_task(
                auto_capture.capture(_name, kwargs, {"result": str(result)}),
            )
            return result

        component.fn = wrapped
