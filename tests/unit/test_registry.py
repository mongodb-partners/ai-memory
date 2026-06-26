"""Tests for ServiceRegistry singleton."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_memory.core.config import MCPConfig
from agent_memory.core.registry import ServiceRegistry


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset ServiceRegistry singleton between tests."""
    ServiceRegistry._instance = None
    yield
    ServiceRegistry._instance = None


class TestServiceRegistryInitialize:
    """TC-038: ServiceRegistry initializes and holds services."""

    def test_initialize_stores_services(self):
        config = MagicMock()
        memory_svc = MagicMock()
        cache_svc = MagicMock()
        audit_svc = MagicMock()
        providers = MagicMock()

        registry = ServiceRegistry.initialize(
            config=config,
            memory_service=memory_svc,
            cache_service=cache_svc,
            audit_service=audit_svc,
            providers=providers,
        )

        assert registry.config is config
        assert registry.memory_service is memory_svc
        assert registry.cache_service is cache_svc
        assert registry.audit_service is audit_svc
        assert registry.providers is providers


class TestServiceRegistryGet:
    """TC-039: get() returns singleton or raises."""

    def test_get_before_init_raises(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            ServiceRegistry.get()

    def test_get_after_init_returns_same(self):
        ServiceRegistry.initialize(
            config=MagicMock(),
            memory_service=MagicMock(),
            cache_service=MagicMock(),
            audit_service=MagicMock(),
            providers=MagicMock(),
        )

        reg1 = ServiceRegistry.get()
        reg2 = ServiceRegistry.get()
        assert reg1 is reg2


class TestCheckAccess:
    """check_access integrates governance and rate limiting."""

    async def test_no_governance_no_rate_limit_returns_none(self):
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            _env_file=None,
        )
        reg = ServiceRegistry.initialize(
            config=config,
            memory_service=MagicMock(),
            cache_service=MagicMock(),
            audit_service=MagicMock(),
            providers=MagicMock(),
        )
        # governance_service and rate_limiter default to None
        result = await reg.check_access("user1", "store_memory")
        assert result is None

    async def test_governance_blocks_returns_error(self):
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            _env_file=None,
        )
        reg = ServiceRegistry.initialize(
            config=config,
            memory_service=MagicMock(),
            cache_service=MagicMock(),
            audit_service=MagicMock(),
            providers=MagicMock(),
        )
        reg.governance_service = AsyncMock()
        reg.governance_service.check_allowed = AsyncMock(return_value=False)

        result = await reg.check_access("user1", "store_memory")
        assert result is not None
        assert "not allowed" in result

    async def test_rate_limiter_blocks_returns_error(self):
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            _env_file=None,
        )
        reg = ServiceRegistry.initialize(
            config=config,
            memory_service=MagicMock(),
            cache_service=MagicMock(),
            audit_service=MagicMock(),
            providers=MagicMock(),
        )
        reg.rate_limiter = AsyncMock()
        reg.rate_limiter.check_rate_limit = AsyncMock(return_value=False)

        result = await reg.check_access("user1", "store_memory")
        assert result is not None
        assert "Rate limit" in result

    async def test_governance_passes_rate_limit_blocks(self):
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            _env_file=None,
        )
        reg = ServiceRegistry.initialize(
            config=config,
            memory_service=MagicMock(),
            cache_service=MagicMock(),
            audit_service=MagicMock(),
            providers=MagicMock(),
        )
        reg.governance_service = AsyncMock()
        reg.governance_service.check_allowed = AsyncMock(return_value=True)
        reg.rate_limiter = AsyncMock()
        reg.rate_limiter.check_rate_limit = AsyncMock(return_value=False)

        result = await reg.check_access("user1", "store_memory")
        assert result is not None
        assert "Rate limit" in result


class TestCheckAccessGovernanceAware:
    """TC-E-029/030: check_access passes governance limits to rate limiter."""

    async def test_check_access_passes_governance_limits(self):
        """TC-E-029: check_access fetches governance profile and passes limits."""
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            _env_file=None,
        )
        reg = ServiceRegistry.initialize(
            config=config,
            memory_service=MagicMock(),
            cache_service=MagicMock(),
            audit_service=MagicMock(),
            providers=MagicMock(),
        )
        reg.governance_service = MagicMock()
        reg.governance_service.check_allowed = AsyncMock(return_value=True)
        reg.governance_service.get_profile = AsyncMock(return_value={
            "role": "power_user",
            "max_memories_per_day": 1000,
            "max_searches_per_day": 5000,
            "allowed_operations": ["*"],
        })
        reg.rate_limiter = MagicMock()
        reg.rate_limiter.check_rate_limit = AsyncMock(return_value=True)

        result = await reg.check_access("user1", "store_memory", role="power_user")
        assert result is None

        # Rate limiter should have been called with max_requests from profile
        reg.rate_limiter.check_rate_limit.assert_called_once_with(
            "user1", "store_memory", max_requests=1000,
        )

    async def test_check_access_search_uses_search_limit(self):
        """TC-E-029: Search operations use max_searches_per_day."""
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            _env_file=None,
        )
        reg = ServiceRegistry.initialize(
            config=config,
            memory_service=MagicMock(),
            cache_service=MagicMock(),
            audit_service=MagicMock(),
            providers=MagicMock(),
        )
        reg.governance_service = MagicMock()
        reg.governance_service.check_allowed = AsyncMock(return_value=True)
        reg.governance_service.get_profile = AsyncMock(return_value={
            "role": "end_user",
            "max_memories_per_day": 100,
            "max_searches_per_day": 500,
            "allowed_operations": ["*"],
        })
        reg.rate_limiter = MagicMock()
        reg.rate_limiter.check_rate_limit = AsyncMock(return_value=True)

        result = await reg.check_access("user1", "hybrid_search", role="end_user")
        assert result is None

        reg.rate_limiter.check_rate_limit.assert_called_once_with(
            "user1", "hybrid_search", max_requests=500,
        )

    async def test_check_access_fallback_when_no_governance(self):
        """TC-E-030: Without governance service, rate limiter uses config default."""
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017",
            _env_file=None,
        )
        reg = ServiceRegistry.initialize(
            config=config,
            memory_service=MagicMock(),
            cache_service=MagicMock(),
            audit_service=MagicMock(),
            providers=MagicMock(),
        )
        # No governance service
        reg.rate_limiter = MagicMock()
        reg.rate_limiter.check_rate_limit = AsyncMock(return_value=True)

        result = await reg.check_access("user1", "store_memory")
        assert result is None

        # Called without max_requests override (uses config default)
        reg.rate_limiter.check_rate_limit.assert_called_once_with(
            "user1", "store_memory",
        )
