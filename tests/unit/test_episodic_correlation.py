"""Tests for correlation-id derivation. REQ-E-097."""

import uuid

from agent_memory.core.correlation import derive_correlation_id

_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


class TestPrecedence:
    def test_explicit_wins(self):
        # TC-EP-CORR-001
        config = {"correlation_id": "explicit", "traceparent": _TRACEPARENT}
        assert derive_correlation_id(config) == "explicit"

    def test_traceparent_yields_the_trace_id(self):
        # TC-EP-CORR-002: the trace id, not the span id, so every turn in one
        # request shares an id.
        config = {"traceparent": _TRACEPARENT, "x_request_id": "req-1"}
        assert derive_correlation_id(config) == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_request_id_is_the_third_choice(self):
        # TC-EP-CORR-003
        assert derive_correlation_id({"x_request_id": "req-1"}) == "req-1"

    def test_empty_config_yields_a_uuid(self):
        # TC-EP-CORR-004: the return value is never empty.
        assert _is_uuid(derive_correlation_id({}))

    def test_none_config_yields_a_uuid(self):
        # TC-EP-CORR-005
        assert _is_uuid(derive_correlation_id(None))


class TestMalformedInput:
    def test_malformed_traceparent_falls_through(self):
        # TC-EP-CORR-006
        config = {"traceparent": "garbage", "x_request_id": "req-1"}
        assert derive_correlation_id(config) == "req-1"

    def test_traceparent_with_an_empty_trace_id_falls_through(self):
        # TC-EP-CORR-007
        config = {"traceparent": "00--span-01", "x_request_id": "req-1"}
        assert derive_correlation_id(config) == "req-1"

    def test_empty_values_are_skipped(self):
        # TC-EP-CORR-008
        config = {"correlation_id": "", "traceparent": "", "x_request_id": ""}
        assert _is_uuid(derive_correlation_id(config))

    def test_non_string_values_are_coerced(self):
        # TC-EP-CORR-009
        assert derive_correlation_id({"correlation_id": 12345}) == "12345"
