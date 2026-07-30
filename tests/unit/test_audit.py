"""Tests for AuditService (buffered writes)."""

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_memory.core.config import MCPConfig
from agent_memory.services.audit import AuditService


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


class TestAuditServiceBuffering:
    """TC-015: Entries are buffered in memory."""

    async def test_log_buffers_entry(self):
        mock_collection = AsyncMock()
        config = _make_config(audit_buffer_size=10, audit_flush_on_write=False)
        service = AuditService(mock_collection, config)

        await service.log("user1", "memory:write", "store_memory", "success", 42)

        # Should NOT have flushed yet (buffer size not reached)
        mock_collection.insert_many.assert_not_called()
        assert len(service._buffer) == 1


class TestAuditServiceSizeFlush:
    """TC-016: Flush triggered when buffer reaches audit_buffer_size."""

    async def test_flush_on_buffer_full(self):
        mock_collection = AsyncMock()
        config = _make_config(audit_buffer_size=3, audit_flush_on_write=False)
        service = AuditService(mock_collection, config)

        await service.log("user1", "op1", "tool1", "success", 10)
        await service.log("user1", "op2", "tool2", "success", 20)
        mock_collection.insert_many.assert_not_called()

        await service.log("user1", "op3", "tool3", "success", 30)
        mock_collection.insert_many.assert_called_once()
        assert len(service._buffer) == 0


class TestAuditServiceComplianceMode:
    """TC-017: audit_flush_on_write causes immediate flush."""

    async def test_flush_on_every_write(self):
        mock_collection = AsyncMock()
        config = _make_config(audit_flush_on_write=True)
        service = AuditService(mock_collection, config)

        await service.log("user1", "op1", "tool1", "success", 10)
        assert mock_collection.insert_many.call_count == 1

        await service.log("user1", "op2", "tool2", "success", 20)
        assert mock_collection.insert_many.call_count == 2


class TestAuditServiceFlushExplicit:
    """TC-018: Explicit flush sends all buffered entries."""

    async def test_flush_sends_buffer(self):
        mock_collection = AsyncMock()
        config = _make_config(audit_buffer_size=100, audit_flush_on_write=False)
        service = AuditService(mock_collection, config)

        await service.log("user1", "op1", "tool1", "success", 10)
        await service.log("user1", "op2", "tool2", "success", 20)
        assert mock_collection.insert_many.call_count == 0

        await service.flush()
        mock_collection.insert_many.assert_called_once()
        inserted = mock_collection.insert_many.call_args[0][0]
        assert len(inserted) == 2
        assert all(e["user_id"] == "user1" for e in inserted)

    async def test_flush_empty_buffer_is_noop(self):
        mock_collection = AsyncMock()
        config = _make_config()
        service = AuditService(mock_collection, config)

        await service.flush()
        mock_collection.insert_many.assert_not_called()


class TestAuditServiceFallback:
    """TC-019: Fallback to file on flush failure."""

    async def test_fallback_on_insert_failure(self):
        mock_collection = AsyncMock()
        mock_collection.insert_many.side_effect = Exception("DB down")
        config = _make_config(audit_flush_on_write=True)
        service = AuditService(mock_collection, config)

        with patch.object(service, "_write_to_file") as mock_file:
            await service.log("user1", "op1", "tool1", "error", 10)
            mock_file.assert_called_once()
            entries = mock_file.call_args[0][0]
            assert len(entries) == 1

    async def test_entry_has_correct_fields(self):
        mock_collection = AsyncMock()
        config = _make_config(audit_flush_on_write=True)
        service = AuditService(mock_collection, config)

        await service.log("user1", "memory:read", "recall_memory", "success", 50, query="test")
        inserted = mock_collection.insert_many.call_args[0][0]
        entry = inserted[0]
        assert entry["user_id"] == "user1"
        assert entry["operation"] == "memory:read"
        assert entry["tool_name"] == "recall_memory"
        assert entry["status"] == "success"
        assert entry["duration_ms"] == 50
        assert entry["metadata"] == {"query": "test"}
        assert "timestamp" in entry


class TestAuditServiceTimerFlush:
    """Flush triggered when flush_interval_seconds elapsed."""

    async def test_flush_on_timer(self):
        mock_collection = AsyncMock()
        config = _make_config(
            audit_buffer_size=100,
            audit_flush_on_write=False,
            audit_flush_interval_seconds=1,
        )
        service = AuditService(mock_collection, config)
        # Fake last flush was 2 seconds ago
        service._last_flush = time.time() - 2.0

        await service.log("user1", "op1", "tool1", "success", 10)
        # Should have flushed due to time elapsed
        mock_collection.insert_many.assert_called_once()


class TestAuditServiceWriteToFile:
    """_write_to_file fallback writes JSONL to disk.

    The destination is configured (``audit_fallback_path``) and resolved once at
    construction, so these tests set it rather than patching ``Path`` at write
    time — see ``test_audit_fallback_file.py`` for why that changed.
    """

    def test_write_to_file_creates_jsonl(self, tmp_path):
        from datetime import UTC, datetime
        mock_collection = AsyncMock()
        config = _make_config(audit_fallback_path=str(tmp_path / "audit.jsonl"))
        service = AuditService(mock_collection, config)

        entries = [{
            "user_id": "u1",
            "operation": "test",
            "tool_name": "t",
            "status": "ok",
            "duration_ms": 0,
            "timestamp": datetime(2025, 1, 1, tzinfo=UTC),
            "metadata": {},
        }]

        service._write_to_file(entries)

        content = (tmp_path / "audit.jsonl").read_text()
        assert "u1" in content
        assert "2025-01-01" in content

    def test_write_to_file_handles_io_error(self, tmp_path):
        mock_collection = AsyncMock()
        config = _make_config(audit_fallback_path=str(tmp_path / "audit.jsonl"))
        service = AuditService(mock_collection, config)

        with patch.object(Path, "open", side_effect=OSError("disk full")):
            # Should not raise
            service._write_to_file([{"user_id": "u1", "timestamp": "now"}])
