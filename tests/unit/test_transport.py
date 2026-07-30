"""Dual-transport entrypoint. REQ-E-080.

TRANSPORT=both serves MCP + REST from one shared AsyncMemory; mcp/rest serve a
single shell. We assert which shells are built and that one facade is shared,
without binding real sockets.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.config import MemoryConfig


def _config(transport: str, **overrides) -> MemoryConfig:
    return MemoryConfig(
        mongodb_connection_string="mongodb://localhost:27017",
        transport=transport, _env_file=None, **overrides,
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
        runner, *_ = _patched
        cfg = _config("rest")
        shells = await runner.build_shells(cfg)
        assert shells["rest"][2] is cfg, (
            "build_shells must pass config to create_app, or REST auth is disabled"
        )

    async def test_mcp_only(self, _patched):
        runner, *_ = _patched
        shells = await runner.build_shells(_config("mcp"))
        assert set(shells) == {"mcp"}

    async def test_rest_only(self, _patched):
        runner, *_ = _patched
        shells = await runner.build_shells(_config("rest"))
        assert set(shells) == {"rest"}

    async def test_legacy_transport_maps_to_mcp(self, _patched):
        runner, *_ = _patched
        shells = await runner.build_shells(_config("streamable-http"))
        assert set(shells) == {"mcp"}

    async def test_unknown_transport_raises(self, _patched):
        runner, *_ = _patched
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
        import agent_memory.shells.rest.app as rest
        import agent_memory.shells.runner as runner

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


class TestTheBindAddressIsNotHardcoded:
    """`0.0.0.0` was a literal at all three bind sites, so there was no way to
    ask a deployed shell to listen on loopback only. Each transport is checked
    separately because a guard added to one branch left the others open.
    """

    def test_mcp_binds_the_configured_host(self, monkeypatch):
        import agent_memory.shells.runner as runner

        built = MagicMock()
        monkeypatch.setattr(runner, "create_mcp", lambda config: built)
        runner.run(_config("mcp", host="127.0.0.5"))
        assert built.run.call_args.kwargs["host"] == "127.0.0.5"

    def test_rest_binds_the_configured_host(self, monkeypatch):
        import agent_memory.shells.rest.app as rest
        import agent_memory.shells.runner as runner

        monkeypatch.setattr(rest, "create_managed_app", lambda config: "rest-app")
        uvicorn_run = MagicMock()
        monkeypatch.setattr("uvicorn.run", uvicorn_run)
        runner.run(_config("rest", host="127.0.0.5"))
        assert uvicorn_run.call_args.kwargs["host"] == "127.0.0.5"

    def test_both_binds_the_configured_host(self, monkeypatch):
        import agent_memory.shells.runner as runner

        monkeypatch.setattr(runner, "build_combined_app", lambda config: MagicMock())
        uvicorn_run = MagicMock()
        monkeypatch.setattr("uvicorn.run", uvicorn_run)
        runner.run(_config("both", host="127.0.0.5"))
        assert uvicorn_run.call_args.kwargs["host"] == "127.0.0.5"

    def test_the_default_is_loopback(self):
        """A default of `0.0.0.0` is the finding. Asserted on the config itself
        so it holds for every caller, not only `run()`."""
        assert _config("mcp").host == "127.0.0.1"


class TestARoutableAddressRequiresAuth:
    """The Critical: shells bound every interface while auth defaulted off, so
    any reachable client could name any `user_id` and read or erase its data.

    Every case asserts what happened at the *bind* — that a refusal actually
    prevented the socket rather than only logging — because the failure being
    guarded is a process that comes up healthy and open.
    """

    @staticmethod
    def _attempt(monkeypatch, transport: str, **overrides):
        """Run the given transport with every bind site mocked.

        Returns the list of bind calls, so "refused" is provable as an empty
        list rather than inferred from the exception alone.
        """
        import agent_memory.shells.rest.app as rest
        import agent_memory.shells.runner as runner

        binds: list = []
        mcp = MagicMock()
        mcp.run = lambda **kw: binds.append(kw)
        monkeypatch.setattr(runner, "create_mcp", lambda config: mcp)
        monkeypatch.setattr(rest, "create_managed_app", lambda config: "rest-app")
        monkeypatch.setattr(runner, "build_combined_app", lambda config: MagicMock())
        monkeypatch.setattr("uvicorn.run", lambda app, **kw: binds.append(kw))
        runner.run(_config(transport, **overrides))
        return binds

    @pytest.mark.parametrize("transport", ["mcp", "rest", "both"])
    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", ""])
    def test_refuses_every_transport_on_every_routable_host(
        self, monkeypatch, transport, host
    ):
        with pytest.raises(RuntimeError, match="authentication disabled"):
            self._attempt(monkeypatch, transport, host=host)

    def test_the_refusal_names_all_three_ways_out(self, monkeypatch):
        """An error that only says "no" gets worked around by whatever the
        operator tries first, which here is turning off the wrong thing."""
        with pytest.raises(RuntimeError) as exc:
            self._attempt(monkeypatch, "rest", host="0.0.0.0")
        message = str(exc.value)
        assert "AUTH_ENABLED=true" in message
        assert "127.0.0.1" in message
        assert "ALLOW_UNAUTHENTICATED_NETWORK_ACCESS=true" in message

    def test_nothing_is_bound_when_it_refuses(self, monkeypatch):
        binds: list = []
        try:
            binds = self._attempt(monkeypatch, "rest", host="0.0.0.0")
        except RuntimeError:
            pass
        assert binds == [], "refused but still bound a socket"

    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]", "127.0.0.2"]
    )
    def test_loopback_serves_unauthenticated(self, monkeypatch, host):
        """Local development must keep working with no flags. `127.0.0.2` is in
        the loopback range and as unroutable as `127.0.0.1`; refusing it would
        break a working setup for nothing."""
        assert self._attempt(monkeypatch, "rest", host=host), "loopback was refused"

    def test_auth_enabled_serves_a_routable_address(self, monkeypatch):
        binds = self._attempt(
            monkeypatch, "rest", host="0.0.0.0",
            auth_enabled=True, auth_secret="x" * 32,
        )
        assert binds and binds[0]["host"] == "0.0.0.0"

    def test_the_explicit_assertion_serves_and_warns(self, monkeypatch, caplog):
        """The escape hatch has to work — an internal deployment behind its own
        gateway is real — but it must leave evidence on every start, because a
        flag set for a spike is how this reaches production unnoticed."""
        import logging

        with caplog.at_level(logging.WARNING, logger="agent_memory.shells.runner"):
            binds = self._attempt(
                monkeypatch, "rest", host="0.0.0.0",
                allow_unauthenticated_network_access=True,
            )
        assert binds and binds[0]["host"] == "0.0.0.0"
        assert any(
            "UNAUTHENTICATED" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        ), "served unauthenticated on a routable address without warning"

    def test_the_assertion_is_off_by_default(self):
        assert _config("rest").allow_unauthenticated_network_access is False
