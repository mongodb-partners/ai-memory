"""Public API exports. REQ-E-082."""


class TestPublicExports:
    def test_top_level_imports(self):
        # TC-EXP-001
        from agent_memory import (
            AccessError,
            AsyncMemory,
            ConfigError,
            Memory,
            MemoryConfig,
            NotFoundError,
            RateLimitError,
        )

        assert AsyncMemory is not None
        assert Memory is not None
        assert MemoryConfig is not None
        assert issubclass(RateLimitError, AccessError)
        assert NotFoundError is not None
        assert ConfigError is not None

    def test_version_is_4(self):
        import agent_memory

        assert agent_memory.__version__.startswith("4.")
