"""Module-level singleton holding initialized services."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from agent_memory.core.config import MCPConfig
    from agent_memory.providers.manager import ProviderManager
    from agent_memory.services.audit import AuditService
    from agent_memory.services.cache import CacheService
    from agent_memory.services.memory import MemoryService


class ServiceRegistry:
    """Populated once during lifespan startup.  Tools import and call
    ``ServiceRegistry.get()`` to access services — no parameters needed.
    """

    _instance: ClassVar["ServiceRegistry | None"] = None

    def __init__(self) -> None:
        self.config: MCPConfig | None = None
        self.memory_service: MemoryService | None = None
        self.cache_service: CacheService | None = None
        self.audit_service: AuditService | None = None
        self.providers: ProviderManager | None = None
        self.governance_service = None
        self.rate_limiter = None
        self.prompt_library = None
        self.decision_service = None

    @classmethod
    def initialize(
        cls,
        config: MCPConfig,
        memory_service: MemoryService,
        cache_service: CacheService,
        audit_service: AuditService,
        providers: ProviderManager,
    ) -> "ServiceRegistry":
        instance = cls()
        instance.config = config
        instance.memory_service = memory_service
        instance.cache_service = cache_service
        instance.audit_service = audit_service
        instance.providers = providers
        cls._instance = instance
        return instance

    @classmethod
    def get(cls) -> "ServiceRegistry":
        if cls._instance is None:
            raise RuntimeError("ServiceRegistry not initialized.")
        return cls._instance

    # Operations classified as "search" for governance limit mapping
    _SEARCH_OPERATIONS = frozenset({
        "recall_memory", "hybrid_search", "search_web", "check_cache",
    })

    async def check_access(self, user_id: str, operation: str, role: str | None = None) -> str | None:
        """Check governance and rate limits. Returns error string or None if OK."""
        effective_role = role or (self.config.auth_default_role if self.config else "end_user")

        profile = None
        if self.governance_service is not None:
            allowed = await self.governance_service.check_allowed(user_id, effective_role, operation)
            if not allowed:
                return f"Operation '{operation}' not allowed for role '{effective_role}'"
            profile = await self.governance_service.get_profile(effective_role)

        if self.rate_limiter is not None:
            if profile is not None:
                if operation in self._SEARCH_OPERATIONS:
                    max_requests = profile.get("max_searches_per_day")
                else:
                    max_requests = profile.get("max_memories_per_day")
                within_limit = await self.rate_limiter.check_rate_limit(
                    user_id, operation, max_requests=max_requests,
                )
            else:
                within_limit = await self.rate_limiter.check_rate_limit(user_id, operation)
            if not within_limit:
                return f"Rate limit exceeded for '{operation}'"

        return None
