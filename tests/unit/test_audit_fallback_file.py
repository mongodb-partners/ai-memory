"""The audit fallback file must be findable and bounded.

``AuditService._write_to_file`` is the last copy of a record MongoDB refused. It
used to build a bare relative ``Path("audit_fallback.jsonl")`` inside the write,
with no size limit, which produced three separate problems:

* **Nobody can name the location.** Relative means "wherever this process was
  started" — a systemd unit's ``WorkingDirectory``, a container's ``WORKDIR``, a
  developer's shell. An operator asked to produce the audit trail for an outage
  has to first work out what the process's cwd was.
* **One outage's records scatter.** The path was resolved on *every* write, so a
  process that chdirs between two failed flushes leaves half the trail in one
  file and half in another, and finds only a fraction of it in either.
* **The file grows without bound, during an outage.** This code path runs only
  while MongoDB is refusing writes, so it runs for as long as the incident does.
  Filling the disk of a host that is already having an outage turns a recoverable
  fault into a stopped process, for a reason unrelated to the original one.

So: an absolute path resolved once at construction, configurable via
``AUDIT_FALLBACK_PATH``, and a ceiling with one rotation
(``AUDIT_FALLBACK_MAX_BYTES``).

Each property is pinned in both directions. A bound that discarded everything, or
a rotation that lost the live file, would satisfy half of this and be worse than
the unbounded version.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.services.audit import AuditService


def _config(**overrides) -> MCPConfig:
    # `_env_file=None`: a live .env in the working tree would otherwise decide
    # where these tests write.
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _service(tmp_path, **overrides) -> AuditService:
    overrides.setdefault("audit_fallback_path", str(tmp_path / "audit.jsonl"))
    return AuditService(AsyncMock(), _config(**overrides))


def _entry(user_id: str = "u1", filler: str = "") -> dict:
    return {
        "user_id": user_id,
        "operation": "memory:write",
        "tool_name": "store_memory",
        "status": "error",
        "duration_ms": 1,
        "timestamp": datetime(2026, 7, 30, tzinfo=UTC),
        "metadata": {"filler": filler} if filler else {},
    }


class TestThePathIsAbsoluteAndFixed:
    """Where the trail is must not depend on where the process happened to start,
    or on where it happens to be at the moment of a failure."""

    def test_a_relative_configured_path_is_resolved_to_an_absolute_one(self):
        service = AuditService(AsyncMock(), _config())
        assert service._fallback_path is not None
        assert service._fallback_path.is_absolute()

    def test_the_default_keeps_the_historical_filename(self):
        """Pinned rather than relocated: a deployment already collecting
        `audit_fallback.jsonl` from its working directory keeps working."""
        service = AuditService(AsyncMock(), _config())
        assert service._fallback_path.name == "audit_fallback.jsonl"

    def test_a_chdir_after_construction_does_not_move_the_file(self, tmp_path,
                                                               monkeypatch):
        """The defect: the path was rebuilt on every write, so one outage's
        records split across directories."""
        start = tmp_path / "start"
        start.mkdir()
        monkeypatch.chdir(start)
        service = AuditService(AsyncMock(), _config())

        service._write_to_file([_entry("before")])
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        service._write_to_file([_entry("after")])

        # Both entries in the one file the service was constructed with.
        content = (start / "audit_fallback.jsonl").read_text()
        assert "before" in content and "after" in content
        assert not (elsewhere / "audit_fallback.jsonl").exists()

    def test_an_explicit_path_is_honoured(self, tmp_path):
        service = _service(tmp_path)
        service._write_to_file([_entry()])
        assert (tmp_path / "audit.jsonl").read_text().count("\n") == 1

    def test_a_user_home_path_is_expanded(self):
        """`~/audit.jsonl` from a hand-edited .env is a real location, not a
        directory literally named `~`."""
        service = AuditService(
            AsyncMock(), _config(audit_fallback_path="~/audit-test.jsonl")
        )
        assert "~" not in str(service._fallback_path)
        assert service._fallback_path == (Path.home() / "audit-test.jsonl").resolve()

    def test_missing_parent_directories_are_created(self, tmp_path):
        """An operator naming `/var/log/agent-memory/audit.jsonl` has named a
        location; failing every write over one absent directory loses exactly the
        records this file exists to keep."""
        target = tmp_path / "deep" / "deeper" / "audit.jsonl"
        service = AuditService(
            AsyncMock(), _config(audit_fallback_path=str(target))
        )
        service._write_to_file([_entry()])
        assert target.exists()


class TestTheFileIsBounded:
    """It is written only during an outage, so unbounded means "until the disk
    is full"."""

    def test_it_rotates_once_the_ceiling_is_passed(self, tmp_path):
        service = _service(tmp_path, audit_fallback_max_bytes=200)
        live = tmp_path / "audit.jsonl"
        rotated = tmp_path / "audit.jsonl.1"

        # Each entry is comfortably over 200 bytes of filler, so the second write
        # sees an oversized file.
        service._write_to_file([_entry("first", filler="x" * 300)])
        assert not rotated.exists()
        service._write_to_file([_entry("second", filler="y" * 300)])

        assert rotated.exists()
        assert "first" in rotated.read_text()
        assert "second" in live.read_text()

    def test_the_oldest_generation_is_the_one_dropped(self, tmp_path):
        """One rotation, so disk cost is bounded at twice the ceiling. The third
        fill discards the first — the alternative is unbounded again."""
        service = _service(tmp_path, audit_fallback_max_bytes=200)
        for label in ("first", "second", "third"):
            service._write_to_file([_entry(label, filler="x" * 300)])

        rotated = (tmp_path / "audit.jsonl.1").read_text()
        live = (tmp_path / "audit.jsonl").read_text()
        assert "second" in rotated
        assert "third" in live
        assert "first" not in rotated and "first" not in live
        # And nothing accumulated a `.2`.
        assert not (tmp_path / "audit.jsonl.1.1").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "audit.jsonl", "audit.jsonl.1",
        ]

    def test_rotation_says_so_at_warning_level(self, tmp_path, caplog):
        """Silent rotation means the records that were dropped were the only copy
        and nothing said they existed."""
        service = _service(tmp_path, audit_fallback_max_bytes=200)
        service._write_to_file([_entry("first", filler="x" * 300)])
        with caplog.at_level(logging.WARNING, logger="agent_memory.services.audit"):
            service._write_to_file([_entry("second", filler="y" * 300)])
        assert any("rotated" in r.message.lower() for r in caplog.records)

    def test_a_quiet_file_is_not_rotated(self, tmp_path, caplog):
        """The paired direction. A rotation on every write would keep only the
        newest batch, which is the same data loss by a different route."""
        service = _service(tmp_path, audit_fallback_max_bytes=10_000)
        with caplog.at_level(logging.WARNING, logger="agent_memory.services.audit"):
            service._write_to_file([_entry("first")])
            service._write_to_file([_entry("second")])

        content = (tmp_path / "audit.jsonl").read_text()
        assert "first" in content and "second" in content
        assert not (tmp_path / "audit.jsonl.1").exists()
        assert not caplog.records

    def test_zero_disables_rotation(self, tmp_path):
        """For a deployment that has put this on a volume it is happy to fill and
        would rather lose nothing."""
        service = _service(tmp_path, audit_fallback_max_bytes=0)
        for label in ("first", "second", "third"):
            service._write_to_file([_entry(label, filler="x" * 300)])

        content = (tmp_path / "audit.jsonl").read_text()
        assert all(label in content for label in ("first", "second", "third"))
        assert not (tmp_path / "audit.jsonl.1").exists()

    def test_a_failed_rotation_still_writes_the_entry(self, tmp_path, caplog):
        """Exceeding the ceiling beats dropping a record. The next write retries
        the rotation."""
        service = _service(tmp_path, audit_fallback_max_bytes=200)
        service._write_to_file([_entry("first", filler="x" * 300)])

        with (
            caplog.at_level(logging.ERROR, logger="agent_memory.services.audit"),
            patch.object(Path, "replace", side_effect=OSError("read-only")),
        ):
            service._write_to_file([_entry("second", filler="y" * 300)])

        content = (tmp_path / "audit.jsonl").read_text()
        assert "first" in content and "second" in content
        assert any("rotate" in r.message.lower() for r in caplog.records)


class TestTheFallbackCanBeDisabled:
    """For a read-only filesystem, where every flush failure otherwise logs a
    fresh stack trace for a write that can never succeed."""

    def test_an_empty_path_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        service = AuditService(AsyncMock(), _config(audit_fallback_path=""))
        assert service._fallback_path is None
        service._write_to_file([_entry()])
        assert list(tmp_path.iterdir()) == []

    def test_disabling_it_warns_once_at_construction(self, caplog):
        """Discarding audit records is a legitimate choice and not a silent one.
        Said at construction rather than per failure, so an outage does not
        produce one line per flush."""
        with caplog.at_level(logging.WARNING, logger="agent_memory.services.audit"):
            AuditService(AsyncMock(), _config(audit_fallback_path=""))
        assert len(caplog.records) == 1
        assert "DISCARDED" in caplog.records[0].message

    def test_an_enabled_fallback_says_nothing(self, caplog):
        """The paired direction: the warning has to mean something when it
        appears."""
        with caplog.at_level(logging.WARNING, logger="agent_memory.services.audit"):
            AuditService(AsyncMock(), _config())
        assert not caplog.records

    def test_whitespace_is_not_a_path(self, tmp_path, monkeypatch):
        """`AUDIT_FALLBACK_PATH=" "` in a hand-edited .env is an attempt to
        disable it, not a request for a file named " "."""
        monkeypatch.chdir(tmp_path)
        service = AuditService(AsyncMock(), _config(audit_fallback_path="   "))
        assert service._fallback_path is None


class TestTheRecordsStaySerialisable:
    """A fallback that raises on the record it was given has lost it."""

    def test_datetimes_become_iso_strings(self, tmp_path):
        """ISO-8601 with the ``T``, not ``str(datetime)``.

        ``default=str`` would also render the timestamp without raising, so this
        pins the *format* rather than merely the absence of an exception:
        ``str()`` separates date and time with a space, and the documented
        recovery path — ``mongoimport`` — reads ISO-8601 as a date and anything
        else as a string. Replaying the trail would then give a collection whose
        `timestamp` cannot be range-queried against the entries MongoDB did
        take."""
        service = _service(tmp_path)
        service._write_to_file([_entry()])
        line = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
        assert line["timestamp"] == datetime(2026, 7, 30, tzinfo=UTC).isoformat()
        assert line["timestamp"] == "2026-07-30T00:00:00+00:00"

    def test_an_unserialisable_value_does_not_lose_the_record(self, tmp_path):
        """`default=str` rather than a bare `json.dumps`: a driver object or an
        ObjectId nested in metadata would otherwise raise `TypeError` and take
        the whole batch — including the entries beside it — with it."""
        entry = _entry()
        entry["metadata"] = {"exc": ValueError("not json")}
        service = _service(tmp_path)
        service._write_to_file([entry, _entry("survivor")])

        lines = (tmp_path / "audit.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert "not json" in lines[0]
        assert "survivor" in lines[1]

    async def test_a_failed_mongo_flush_reaches_the_file(self, tmp_path):
        """The end-to-end path, so the wiring is pinned and not just the helper."""
        collection = AsyncMock()
        collection.insert_many.side_effect = RuntimeError("DB down")
        service = AuditService(
            collection,
            _config(
                audit_flush_on_write=True,
                audit_fallback_path=str(tmp_path / "audit.jsonl"),
            ),
        )
        await service.log("u1", "memory:write", "store_memory", "error", 5)

        assert "u1" in (tmp_path / "audit.jsonl").read_text()

    async def test_a_successful_flush_writes_no_file(self, tmp_path):
        """The paired direction: the fallback is for an outage, not for every
        write."""
        service = AuditService(
            AsyncMock(),
            _config(
                audit_flush_on_write=True,
                audit_fallback_path=str(tmp_path / "audit.jsonl"),
            ),
        )
        await service.log("u1", "memory:write", "store_memory", "success", 5)
        assert not (tmp_path / "audit.jsonl").exists()


class TestTheConfigSurface:
    """Both knobs are settable from the environment, like everything else."""

    @pytest.mark.parametrize(
        ("env_name", "value", "field", "expected"),
        [
            ("AUDIT_FALLBACK_PATH", "/tmp/x.jsonl", "audit_fallback_path",
             "/tmp/x.jsonl"),
            ("AUDIT_FALLBACK_MAX_BYTES", "1024", "audit_fallback_max_bytes", 1024),
        ],
    )
    def test_each_knob_reads_from_the_environment(self, monkeypatch, env_name,
                                                  value, field, expected):
        monkeypatch.setenv(env_name, value)
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017", _env_file=None
        )
        assert getattr(config, field) == expected

    def test_the_default_ceiling_is_finite(self):
        """The whole point. A default of 0 would ship the unbounded behaviour to
        everyone who does not set the variable — which is everyone, until they
        have had the outage."""
        assert _config().audit_fallback_max_bytes > 0
