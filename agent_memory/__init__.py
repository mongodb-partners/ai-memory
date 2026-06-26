"""agent-memory v4: MongoDB Atlas-backed memory library with MCP + REST shells.

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

__version__ = "4.0.0"

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
