"""Buffered audit log service."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)


class AuditService:
    """Buffered audit log writes with configurable flush strategy."""

    def __init__(self, audit_collection, config: MCPConfig) -> None:
        self.audit_log = audit_collection
        self.config = config
        self._buffer: list[dict] = []
        self._last_flush = time.time()

    async def log(
        self,
        user_id: str,
        operation: str,
        tool_name: str,
        status: str,
        duration_ms: int,
        **metadata,
    ) -> None:
        entry = {
            "user_id": user_id,
            "operation": operation,
            "tool_name": tool_name,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata if metadata else {},
        }
        self._buffer.append(entry)

        should_flush = (
            self.config.audit_flush_on_write
            or len(self._buffer) >= self.config.audit_buffer_size
            or time.time() - self._last_flush >= self.config.audit_flush_interval_seconds
        )
        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer = []
        try:
            await self.audit_log.insert_many(batch)
        except Exception:
            logger.exception("Failed to flush audit entries to MongoDB")
            self._write_to_file(batch)
        self._last_flush = time.time()

    def _write_to_file(self, entries: list[dict]) -> None:
        """Fallback: append entries to local audit file."""
        fallback_path = Path("audit_fallback.jsonl")
        try:
            with fallback_path.open("a") as f:
                for entry in entries:
                    # Convert datetime to ISO string for JSON serialization
                    serializable = {
                        k: v.isoformat() if isinstance(v, datetime) else v
                        for k, v in entry.items()
                    }
                    f.write(json.dumps(serializable) + "\n")
        except Exception:
            logger.exception("Failed to write audit entries to fallback file")
