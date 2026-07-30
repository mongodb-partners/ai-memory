"""Dual-transport entrypoint. REQ-E-080.

TRANSPORT=both serves MCP + REST from one shared AsyncMemory; mcp/rest serve a
single shell. We assert which shells are built and that one facade is shared,
without binding real sockets.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.shells.runner import build_shells


def _config(transport: str) -> MemoryConfig:
    return MemoryConfig(
        mongodb_connection_string="mongodb://localhost:27017",
        transport=transport, _env_file=None,
    )


@pytest.fixture
def _patched(monkeypatch):
    import agent_memory.shells.runner as runner

    facade = MagicMock()
    create = AsyncMock(return_value=facade)
    monkeypatch.setattr(runner.AsyncMemory, "create", create)
    monkeypatch.setattr(runner, "create_mcp", lambda config, app: ("mcp", app))
    # `config` is captured, not ignored: `build_shells` used to call
    # `create_app(app)` with no config, which silently disabled REST auth. The
    # stub accepts it so a regression to the positional-only call fails here.
    monkeypatch.setattr(runner, "create_app", lambda app, config=None: ("rest", app, config))
    return runner, facade, create


class TestBuildShells:
    async def test_both_shares_one_facade(self, _patched):
        runner, facade, create = _patched
        shells = await runner.build_shells(_config("both"))
        create.assert_awaited_once()
        assert set(shells) == {"mcp", "rest"}
        # both shells bound to the same facade instance
        assert shells["mcp"][1] is facade
        assert shells["rest"][1] is facade

    async def test_rest_shell_receives_the_config(self, _patched):
        """Without the config, `create_app` builds a no-op auth dependency.

        That made `TRANSPORT=rest` serve every route unauthenticated regardless of
        `AUTH_ENABLED`, while the MCP shell built from the same config enforced it.
        """
        runner, facade, create = _patched
        cfg = _config("rest")
        shells = await runner.build_shells(cfg)
        assert shells["rest"][2] is cfg, (
            "build_shells must pass config to create_app, or REST auth is disabled"
        )

    async def test_mcp_only(self, _patched):
        runner, facade, create = _patched
        shells = await runner.build_shells(_config("mcp"))
        assert set(shells) == {"mcp"}

    async def test_rest_only(self, _patched):
        runner, facade, create = _patched
        shells = await runner.build_shells(_config("rest"))
        assert set(shells) == {"rest"}

    async def test_legacy_transport_maps_to_mcp(self, _patched):
        runner, facade, create = _patched
        shells = await runner.build_shells(_config("streamable-http"))
        assert set(shells) == {"mcp"}

    async def test_unknown_transport_raises(self, _patched):
        runner, facade, create = _patched
        with pytest.raises(ValueError):
            await runner.build_shells(_config("carrier-pigeon"))


class TestCombinedApp:
    """TRANSPORT=both must serve mcp+rest off ONE shared facade (one Atlas pool).

    Regression guard for the live-test finding: the old run() built two separate
    facades and mounted MCP without lifespan chaining.
    """

    def test_combined_app_creates_single_facade(self, monkeypatch):
        import agent_memory.shells.runner as runner

        created = []
        facade = MagicMock()
        facade.close = AsyncMock()

        async def _create(config):
            created.append(config)
            return facade

        monkeypatch.setattr(runner.AsyncMemory, "create", _create)
        # avoid real FastMCP/tool wiring
        mcp_obj = MagicMock()
        mcp_obj.http_app = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(runner, "create_mcp", lambda config, app: mcp_obj)
        monkeypatch.setattr(runner, "create_app", lambda app, config=None: MagicMock(mount=MagicMock()))

        cfg = _config("both")
        api = runner.build_combined_app(cfg)
        # lifespan not yet run → no facade created until the app starts
        import asyncio

        async def _drive():
            async with api.router.lifespan_context(api):
                pass

        asyncio.run(_drive())
        assert len(created) == 1, f"expected exactly ONE shared facade, got {len(created)}"
        facade.close.assert_awaited_once()


class TestRun:
    """run() dispatch — uvicorn/mcp are mocked, no sockets bound."""

    def test_run_mcp(self, monkeypatch):
        import agent_memory.shells.runner as runner

        built = MagicMock()
        monkeypatch.setattr(runner, "create_mcp", lambda config: built)
        runner.run(_config("mcp"))
        built.run.assert_called_once()

    def test_run_rest(self, monkeypatch):
        import agent_memory.shells.runner as runner
        import agent_memory.shells.rest.app as rest

        monkeypatch.setattr(rest, "create_managed_app", lambda config: "rest-app")
        uvicorn_run = MagicMock()
        monkeypatch.setattr("uvicorn.run", uvicorn_run)
        runner.run(_config("rest"))
        uvicorn_run.assert_called_once()

    def test_run_both_serves_combined_app(self, monkeypatch):
        import agent_memory.shells.runner as runner

        combined = MagicMock()
        monkeypatch.setattr(runner, "build_combined_app", lambda config: combined)
        uvicorn_run = MagicMock()
        monkeypatch.setattr("uvicorn.run", uvicorn_run)
        runner.run(_config("both"))
        uvicorn_run.assert_called_once()
        # the combined single-facade app is what gets served
        assert uvicorn_run.call_args.args[0] is combined
