"""agent-memory: MongoDB Atlas-backed agent memory with MCP + REST shells.

Four memory tiers in one Atlas cluster — short-term state, long-term semantic
memory, episodic memory (what the agent actually did), and a semantic response
cache — with no agent-framework dependency.

Public API:
    from agent_memory import Memory, AsyncMemory, MemoryConfig
    from agent_memory import AccessError, RateLimitError, NotFoundError, ConfigError
"""

from agent_memory.config import MemoryConfig
from agent_memory.exceptions import (
    AccessError,
    ConfigError,
    MemoryError,
    NotFoundError,
    RateLimitError,
)
from agent_memory.memory import AsyncMemory
from agent_memory.sync import Memory
from agent_memory.version import __version__

__all__ = [
    "Memory",
    "AsyncMemory",
    "MemoryConfig",
    "MemoryError",
    "AccessError",
    "RateLimitError",
    "NotFoundError",
    "ConfigError",
    "__version__",
]
