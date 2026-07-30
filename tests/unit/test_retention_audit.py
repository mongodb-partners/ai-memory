"""A retention change that failed must not be audited as a success.

``EpisodicService.set_retention`` deliberately never raises — retention
management should not be able to fail a request — so it reports failure as
``{"status": "error", ...}``. ``AsyncMemory._run`` treated "the coroutine
returned" as "the operation succeeded", so the audit log recorded a failed
retention change as ``success``, with the caller's *requested* ``ttl_seconds``
beside it as though the index now carried it.

Both directions matter, and neither is visible from the response:

* a **lengthened** retention that silently failed leaves turns expiring on the
  old, shorter schedule while the log says they are being kept
* a **shortened** one is destructive, and the caller cannot tell either way —
  Atlas deletes on the TTL monitor's own schedule, so the response is
  ``{"scope": "collection"}`` in both cases. The audit log was the one place the
  difference could have surfaced, and it said success too

So the tests come in pairs: a failure is audited as ``error`` *and* a success is
still audited as ``success``. A fix that labelled everything ``error`` would
satisfy half of this file and be just as useless.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.memory import AsyncMemory, _retention_outcome


def _config(**overrides) -> MemoryConfig:
    # `_env_file=None`: a live .env in the working tree would otherwise decide
    # what these tests are testing.
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


def _facade(**config_overrides):
    """An AsyncMemory with mocked collaborators — orchestration only, no create()."""
    m = AsyncMemory.__new__(AsyncMemory)
    m.config = _config(**config_overrides)
    m.episodic_service = AsyncMock()
    m.audit_service = AsyncMock()
    m.governance_service = None
    m.rate_limiter = None
    m.providers = MagicMock()
    m._workers = []
    return m


def _entry(m) -> tuple[str, dict]:
    """The status and metadata of the audit entry the facade wrote.

    ``audit_service.log(user_id, category, operation, status, duration_ms, **meta)``
    — positional, so the status is ``args[3]``.
    """
    call = m.audit_service.log.call_args
    return call.args[3], call.kwargs


class TestAFailedRetentionChangeIsAuditedAsAnError:
    """The defect: `{"status": "error"}` was recorded as `success`."""

    async def test_a_reported_failure_is_audited_as_an_error(self):
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={
                "status": "error",
                "ttl_seconds": 7200,
                "scope": "collection",
                "error": "OperationFailure: not authorized",
            }
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        status, _ = _entry(m)
        assert status == "error"

    async def test_the_reason_is_recorded_not_just_the_verdict(self):
        """An `error` entry with no reason sends an operator to the logs of a
        process that may no longer be running."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={
                "status": "error",
                "ttl_seconds": 7200,
                "scope": "collection",
                "error": "OperationFailure: not authorized",
            }
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        _, meta = _entry(m)
        assert "not authorized" in meta["error"]

    async def test_a_failure_with_no_reason_still_says_it_failed(self):
        """The status is the part that must not be lost. A service returning
        `error` without an `error` key is a contract slip, not a success."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={"status": "error", "scope": "collection"}
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        status, meta = _entry(m)
        assert status == "error"
        assert meta["error"]

    async def test_the_removal_direction_fails_the_same_way(self):
        """`ttl_seconds=None` takes a different branch in the service (a dropped
        index, not a modified one) and reports failure through the same key."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={
                "status": "error",
                "ttl_seconds": None,
                "scope": "collection",
                "error": "OperationFailure: index not found",
            }
        )
        await m.set_activity_retention("u1", ttl_seconds=None)
        status, _ = _entry(m)
        assert status == "error"

    async def test_the_caller_still_gets_the_service_result(self):
        """Auditing honestly must not change the contract. The failure is still a
        return value — callers read `status` — because raising here would let a
        retention change take down a request."""
        m = _facade()
        reported = {
            "status": "error",
            "ttl_seconds": 7200,
            "scope": "collection",
            "error": "OperationFailure: not authorized",
        }
        m.episodic_service.set_retention = AsyncMock(return_value=reported)
        out = await m.set_activity_retention("u1", ttl_seconds=7200)
        assert out == reported

    async def test_the_requested_ttl_is_still_recorded(self):
        """What was asked for is worth keeping even when it did not take effect —
        it is the difference between "someone tried to shorten retention to two
        hours and could not" and "something went wrong"."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={"status": "error", "ttl_seconds": 7200, "error": "no"}
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        _, meta = _entry(m)
        assert meta["ttl_seconds"] == 7200


class TestASuccessfulRetentionChangeIsStillASuccess:
    """The paired direction. Labelling every call `error` would be no better."""

    @pytest.mark.parametrize("reported", ["updated", "created", "removed"])
    async def test_every_success_status_audits_as_a_success(self, reported):
        """Three of them, because the service reports which path it took:
        `collMod` in place, the `create_index` fallback, or a dropped index."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={
                "status": reported,
                "ttl_seconds": 7200,
                "scope": "collection",
            }
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        status, meta = _entry(m)
        assert status == "success"
        assert meta == {"ttl_seconds": 7200}

    async def test_a_success_carries_no_error_field(self):
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={"status": "updated", "ttl_seconds": 7200}
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        _, meta = _entry(m)
        assert "error" not in meta

    async def test_an_unrecognised_status_is_left_alone(self):
        """This reads a contract. Inventing a failure from a shape it does not
        recognise would make a service change look like an outage."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={"status": "queued", "ttl_seconds": 7200}
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        status, _ = _entry(m)
        assert status == "success"

    async def test_exactly_one_entry_is_written_either_way(self):
        """A second entry would double-count the operation in any audit report."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={"status": "error", "ttl_seconds": 7200, "error": "no"}
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        assert m.audit_service.log.await_count == 1

    async def test_it_is_still_an_admin_category_operation(self):
        """The category is what makes this reachable only by an operator; the
        status change must not have disturbed it."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            return_value={"status": "error", "ttl_seconds": 7200, "error": "no"}
        )
        await m.set_activity_retention("u1", ttl_seconds=7200)
        assert m.audit_service.log.call_args.args[1] == "admin"


class TestOtherOperationsAreUnaffected:
    """The hook is opt-in per call site. A `{"status": ...}` from anything else
    means what it always meant."""

    async def test_a_decision_write_reporting_a_status_still_audits_success(self):
        """`remember_decision` returns `{"key": ..., "status": "stored"|"updated"}`
        — the same key, a different vocabulary. Reading it as a verdict on the
        call would be a bug in the opposite direction."""
        m = _facade()
        m.decision_service = AsyncMock()
        m.decision_service.store = AsyncMock(return_value="stored")
        await m.remember_decision("u1", "k", "v")
        status, _ = _entry(m)
        assert status == "success"

    async def test_a_raised_error_is_still_audited_as_an_error(self):
        """The exception path is untouched: the `outcome` hook runs only after a
        coroutine returns."""
        m = _facade()
        m.episodic_service.set_retention = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )
        with pytest.raises(RuntimeError, match="connection lost"):
            await m.set_activity_retention("u1", ttl_seconds=7200)
        status, meta = _entry(m)
        assert status == "error"
        assert "connection lost" in meta["error"]

    async def test_a_denial_is_still_audited_as_denied_not_as_an_error(self):
        """`_require_admin_for_global_mutation` runs inside the factory so its
        refusal is audited. It must keep the `denied` label — an authorisation
        refusal recorded as a service fault sends an operator hunting a bug.

        `auth_enabled=True` because the guard applies only where a role claim
        exists; with auth off there is one tenant and nothing to protect from
        whom. See `_require_admin_for_global_mutation`.
        """
        m = _facade(auth_enabled=True, auth_secret="test-secret-not-a-real-one")
        await_count_before = m.episodic_service.set_retention.await_count
        from agent_memory.exceptions import AccessError

        with pytest.raises(AccessError):
            await m.set_activity_retention("u1", ttl_seconds=7200, role="end_user")
        status, _ = _entry(m)
        assert status == "denied"
        # And the service was never reached, so nothing was changed to be audited.
        assert m.episodic_service.set_retention.await_count == await_count_before


class TestTheOutcomeReaderInIsolation:
    """`_retention_outcome` is small enough to pin directly, which keeps the
    facade tests above about auditing rather than about dict shapes."""

    def test_an_error_status_maps_to_an_error_verdict(self):
        assert _retention_outcome({"status": "error", "error": "boom"}) == (
            "error",
            {"error": "boom"},
        )

    @pytest.mark.parametrize("status", ["updated", "created", "removed", "queued"])
    def test_every_other_status_defers_to_the_default(self, status):
        assert _retention_outcome({"status": status}) is None

    def test_a_missing_status_defers_to_the_default(self):
        assert _retention_outcome({"ttl_seconds": 7200}) is None

    def test_a_non_dict_result_defers_to_the_default(self):
        """Nothing returns a non-dict here today. If something does, the default
        is the safe reading: the caller's own contract decides, not this."""
        assert _retention_outcome(None) is None
        assert _retention_outcome("updated") is None
