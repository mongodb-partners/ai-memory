"""Entry point: python -m agent_memory."""

from agent_memory.core.config import MCPConfig
from agent_memory.server import mcp


def main():
    """CLI entry point for ``memory-mcp`` script."""
    config = MCPConfig()
    if config.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=config.port)


if __name__ == "__main__":
    main()
