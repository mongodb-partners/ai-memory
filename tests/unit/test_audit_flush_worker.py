"""Tests for AuditFlushWorker."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.services.audit_flush_worker import AuditFlushWorker


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


class TestAuditFlushWorkerRun:
    """TC-E-023/025: Worker calls flush periodically."""

    async def test_worker_calls_flush(self):
        """TC-E-023: Worker calls flush after interval."""
        mock_audit = MagicMock()
        mock_audit.flush = AsyncMock()
        config = _make_config(audit_flush_interval_seconds=0)
        worker = AuditFlushWorker(mock_audit, config)

        async def stop_after_flush():
            # Wait for at least one flush
            while mock_audit.flush.call_count < 1:
                await asyncio.sleep(0.01)
            worker.stop()

        task = asyncio.create_task(worker.run())
        stopper = asyncio.create_task(stop_after_flush())

        await asyncio.wait([task, stopper], timeout=2.0)

        assert mock_audit.flush.call_count >= 1

    async def test_worker_respects_interval(self):
        """TC-E-025: Worker flushes at configured interval."""
        mock_audit = MagicMock()
        mock_audit.flush = AsyncMock()
        # Use a very short interval for testing
        config = _make_config(audit_flush_interval_seconds=0)
        worker = AuditFlushWorker(mock_audit, config)

        async def stop_after_flushes():
            while mock_audit.flush.call_count < 3:
                await asyncio.sleep(0.01)
            worker.stop()

        task = asyncio.create_task(worker.run())
        stopper = asyncio.create_task(stop_after_flushes())

        await asyncio.wait([task, stopper], timeout=2.0)

        assert mock_audit.flush.call_count >= 3


class TestAuditFlushWorkerStop:
    """TC-E-024: Worker can be stopped."""

    async def test_stop_ends_loop(self):
        """TC-E-024: stop() causes the run loop to exit."""
        mock_audit = MagicMock()
        mock_audit.flush = AsyncMock()
        config = _make_config(audit_flush_interval_seconds=0)
        worker = AuditFlushWorker(mock_audit, config)

        worker.stop()

        # run() should exit quickly since _running is False
        task = asyncio.create_task(worker.run())
        # It will still do one sleep+flush cycle due to while check order
        await asyncio.wait([task], timeout=1.0)
        assert task.done()


class TestAuditFlushWorkerErrorHandling:
    """TC-E-027: Worker continues after flush failure."""

    async def test_continues_after_flush_error(self):
        """TC-E-027: Flush failure does not stop the worker."""
        mock_audit = MagicMock()
        call_count = 0

        async def flush_with_error():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("DB down")
            # Second call succeeds

        mock_audit.flush = AsyncMock(side_effect=flush_with_error)
        config = _make_config(audit_flush_interval_seconds=0)
        worker = AuditFlushWorker(mock_audit, config)

        async def stop_after_recovery():
            while call_count < 2:
                await asyncio.sleep(0.01)
            worker.stop()

        task = asyncio.create_task(worker.run())
        stopper = asyncio.create_task(stop_after_recovery())

        await asyncio.wait([task, stopper], timeout=2.0)

        # Worker called flush at least twice (survived the error)
        assert call_count >= 2
