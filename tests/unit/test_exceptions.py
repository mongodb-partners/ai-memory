"""Tests for the agent_memory typed exception hierarchy (REQ-E-001, REQ-E-002)."""

from agent_memory.exceptions import (
    AccessError,
    ConfigError,
    MemoryError,
    NotFoundError,
    RateLimitError,
)


class TestExceptionHierarchy:
    """TC-EXC-001 / TC-EXC-002: typed exceptions and their relationships."""

    def test_rate_limit_error_is_access_error(self):
        # TC-EXC-001: catching AccessError also catches throttling
        assert issubclass(RateLimitError, AccessError)

    def test_all_subclass_memory_error(self):
        # TC-EXC-001 / TC-EXC-002
        for exc in (AccessError, RateLimitError, NotFoundError, ConfigError):
            assert issubclass(exc, MemoryError)

    def test_access_error_caught_as_memory_error(self):
        try:
            raise RateLimitError("slow down")
        except AccessError as e:
            assert str(e) == "slow down"
        else:  # pragma: no cover
            raise AssertionError("RateLimitError not caught as AccessError")

    def test_config_error_is_not_access_error(self):
        assert not issubclass(ConfigError, AccessError)
