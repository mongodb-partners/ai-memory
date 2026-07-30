"""Periodic audit buffer flush background task.

Ensures audit entries are flushed to MongoDB at regular intervals,
preventing data loss on server crash.
"""

import asyncio
import logging

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)


class AuditFlushWorker:
    """Periodic audit buffer flush."""

    def __init__(self, audit_service, config: MCPConfig) -> None:
        self.audit_service = audit_service
        self.config = config
        self._running = True

    async def run(self) -> None:
        """Infinite loop flushing every audit_flush_interval_seconds."""
        while self._running:
            await asyncio.sleep(self.config.audit_flush_interval_seconds)
            try:
                await self.audit_service.flush()
            except Exception:
                logger.warning("Periodic audit flush failed.", exc_info=True)

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._running = False
