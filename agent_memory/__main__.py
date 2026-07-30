"""Entry point: ``python -m agent_memory`` / the ``agent-memory`` script.

Dispatches to the transport runner, which serves MCP, REST, or both off one
shared ``AsyncMemory`` instance based on ``TRANSPORT``.
"""

from agent_memory.config import MemoryConfig
from agent_memory.shells.runner import run


def main():
    run(MemoryConfig.from_env())


if __name__ == "__main__":
    main()
