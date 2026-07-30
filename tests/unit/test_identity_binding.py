"""A token may only act as itself, and its role must reach the access check.

Three defects, one root cause: authentication was wired up and then not used.

1. **Cross-tenant access (IDOR).** ``_build_auth_dependency`` returned a verified
   ``AccessToken``, but it was attached as ``dependencies=protected`` — a form
   FastAPI runs for its side effects and whose return value no handler receives.
   Every handler read ``user_id`` from the query string or the JSON body, and the
   query layer then scoped faithfully to whoever was named. Any valid token could
   read, edit, or wipe any other tenant's memory. The MCP shell had the same shape
   with ``user_id`` as a tool argument.

2. **Fail-open auth.** ``AUTH_ENABLED=true`` with an empty ``AUTH_SECRET`` logged
   one warning at startup and served every route unauthenticated. And
   ``build_shells`` called ``create_app(app)`` with no config at all, so
   ``TRANSPORT=rest`` was unauthenticated whatever the config said.

3. **Roles never reached governance.** ``_check_access`` took a ``role``
   parameter; ``_run`` never passed one. Every caller was evaluated as
   ``auth_default_role``, so with governance on the ``admin`` profile was
   unreachable through any transport and ``end_user`` could not delete.

``auth_user_id_claim`` and ``auth_role_claim`` were in the config for exactly this
and were read nowhere in the codebase — the fix was designed and never wired.

REQ-E-143 (identity comes from the token, not the request).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from fastmcp.server.auth import AccessToken

from agent_memory.auth.identity import Caller, IdentityError, resolve_caller
from agent_memory.config import MemoryConfig
from agent_memory.shells.rest.app import create_app

SECRET = "s" * 40


def _config(**overrides) -> MemoryConfig:
    defaults = {
        "mongodb_connection_string": "mongodb://localhost:27017",
        "_env_file": None,
    }
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def _auth_config(**overrides) -> MemoryConfig:
    return _config(auth_enabled=True, auth_secret=SECRET, **overrides)


def _token(user_id: str, **claims) -> AccessToken:
    return AccessToken(
        token="t", client_id=user_id, scopes=["memory-mcp"],
        claims={"sub": user_id, **claims},
    )


def _facade():
    """A facade recording every call, so we can read back the user_id it got."""
    app = MagicMock()
    app.config = _auth_config()
    for name, value in (
        ("add", {"count": 1}),
        ("recall", {"results": [], "count": 0}),
        ("search", {"results": [], "count": 0}),
        ("delete", {"deleted_count": 0}),
        ("remember_decision", {"key": "k", "status": "stored"}),
        ("recall_decision", {"key": "k", "value": "v"}),
        ("log_activity", {"enqueued": True, "thread_id": "t1"}),
        ("recall_activity", {"results": [], "count": 0}),
        ("get_thread", {"results": [], "count": 0}),
        ("get_activity_by_correlation", {"results": [], "count": 0}),
        ("set_activity_retention", {"ttl_seconds": 7200}),
        ("health", {}),
        ("wipe_user_data", {}),
    ):
        setattr(app, name, AsyncMock(return_value=value))
    app.activity_stats = MagicMock(return_value={"enqueued": 0})
    return app


def _bearer(user_id: str, config: MemoryConfig, **claims) -> dict:
    from agent_memory.auth.api_keys import APIKeyManager
    from agent_memory.auth.token_verifier import MemoryMCPTokenVerifier

    verifier = MemoryMCPTokenVerifier(
        secret=config.auth_secret, api_key_manager=APIKeyManager()
    )
    token = verifier.create_token(user_id=user_id)
    if claims:
        # create_token only emits sub/iss/iat/exp; re-sign to add role claims.
        import jwt

        payload = jwt.decode(token, config.auth_secret, algorithms=["HS256"],
                             issuer="memory-mcp")
        payload.update(claims)
        token = jwt.encode(payload, config.auth_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


class TestResolveCallerIsTheOnlyDecider:
    """The pure function both shells share. Everything else delegates to it."""

    def test_token_identity_wins_over_a_matching_request(self) -> None:
        caller = resolve_caller(_token("alice"), "alice", _auth_config())
        assert caller.user_id == "alice"
        assert caller.authenticated is True

    def test_a_request_naming_someone_else_is_refused(self) -> None:
        """The finding, stated as one assertion.

        Refused rather than silently rewritten to the token's own id: a request
        for someone else's data is a request whose author is confused about whose
        data it is, and answering a different question hides that.
        """
        with pytest.raises(IdentityError):
            resolve_caller(_token("alice"), "bob", _auth_config())

    def test_an_absent_request_user_id_takes_the_token_identity(self) -> None:
        assert resolve_caller(_token("alice"), None, _auth_config()).user_id == "alice"

    def test_the_configured_claim_is_the_one_read(self) -> None:
        """`auth_user_id_claim` existed and was read nowhere. It is read here."""
        access = AccessToken(token="t", client_id="wrong", scopes=[],
                             claims={"sub": "wrong", "tenant": "right"})
        config = _auth_config(auth_user_id_claim="tenant")
        assert resolve_caller(access, None, config).user_id == "right"

    def test_client_id_is_the_fallback_when_the_claim_is_absent(self) -> None:
        access = AccessToken(token="t", client_id="alice", scopes=[], claims={})
        assert resolve_caller(access, None, _auth_config()).user_id == "alice"

    def test_a_token_identifying_nobody_is_refused(self) -> None:
        """"No identity" cannot be read as "any identity"."""
        access = AccessToken(token="t", client_id="", scopes=[], claims={})
        with pytest.raises(IdentityError):
            resolve_caller(access, "alice", _auth_config())

    def test_the_role_claim_is_carried(self) -> None:
        caller = resolve_caller(_token("alice", role="admin"), None, _auth_config())
        assert caller.role == "admin"

    def test_the_configured_role_claim_is_the_one_read(self) -> None:
        access = _token("alice", **{"https://ex/roles": "admin"})
        config = _auth_config(auth_role_claim="https://ex/roles")
        assert resolve_caller(access, None, config).role == "admin"

    def test_no_role_claim_means_use_the_default(self) -> None:
        """None, not "" — the facade reads it as "fall back to auth_default_role"."""
        assert resolve_caller(_token("alice"), None, _auth_config()).role is None

    def test_auth_off_takes_the_request_at_its_word(self) -> None:
        caller = resolve_caller(None, "alice", _config())
        assert caller == Caller(user_id="alice")
        assert caller.authenticated is False

    def test_auth_off_still_needs_a_user_id(self) -> None:
        with pytest.raises(IdentityError):
            resolve_caller(None, None, _config())

    def test_authenticated_is_stored_not_inferred(self) -> None:
        """A real JWT can carry no role and no scopes.

        `_reconcile`-style logic that infers "unauthenticated" from empty
        role/scopes would re-open the cross-tenant hole for exactly the tokens
        that look least remarkable, so the flag is a stored fact.
        """
        access = AccessToken(token="t", client_id="alice", scopes=[],
                             claims={"sub": "alice"})
        caller = resolve_caller(access, None, _auth_config())
        assert caller.role is None and caller.scopes == ()
        assert caller.authenticated is True


class TestRestRefusesCrossTenantRequests:
    """Driven through real HTTP, because the bug was in the wiring, not the logic.

    A unit test of `resolve_caller` cannot catch `dependencies=protected` — the
    resolver was correct and simply not called. These go through TestClient so the
    dependency graph is the thing under test.
    """

    def _client(self, facade, config=None):
        config = config or _auth_config()
        return TestClient(create_app(facade, config=config)), config

    def test_query_param_naming_another_user_is_403(self) -> None:
        facade = _facade()
        client, cfg = self._client(facade)
        r = client.get("/memories/recall", params={"user_id": "bob", "query": "q"},
                       headers=_bearer("alice", cfg))
        assert r.status_code == 403, "alice read bob's memories"
        facade.recall.assert_not_awaited()

    def test_body_naming_another_user_is_403(self) -> None:
        facade = _facade()
        client, cfg = self._client(facade)
        r = client.post("/memories", headers=_bearer("alice", cfg), json={
            "user_id": "bob", "conversation_id": "c1", "messages": [{"content": "x"}],
        })
        assert r.status_code == 403, "alice wrote into bob's memory"
        facade.add.assert_not_awaited()

    def test_delete_naming_another_user_is_403(self) -> None:
        """The worst case: a token deleting someone else's memories."""
        facade = _facade()
        client, cfg = self._client(facade)
        r = client.request("DELETE", "/memories",
                           params={"user_id": "bob", "confirm": True},
                           headers=_bearer("alice", cfg))
        assert r.status_code == 403
        facade.delete.assert_not_awaited()

    def test_retention_naming_another_user_is_403(self) -> None:
        facade = _facade()
        client, cfg = self._client(facade)
        r = client.put("/activity/retention", headers=_bearer("alice", cfg),
                       json={"user_id": "bob", "ttl_seconds": 60})
        assert r.status_code == 403
        facade.set_activity_retention.assert_not_awaited()

    @pytest.mark.parametrize(
        "method,path,params",
        [
            ("get", "/memories/search", {"query": "q"}),
            ("get", "/decisions", {"key": "k"}),
            ("get", "/activity/search", {"query": "q"}),
            ("get", "/activity/thread/t1", {}),
            ("get", "/activity/correlation/c1", {}),
        ],
    )
    def test_every_get_route_refuses_a_foreign_user_id(self, method, path, params) -> None:
        """Parametrized because one unconverted handler is the whole vulnerability."""
        facade = _facade()
        client, cfg = self._client(facade)
        r = getattr(client, method)(path, params={**params, "user_id": "bob"},
                                    headers=_bearer("alice", cfg))
        assert r.status_code == 403, f"{path} served bob's data to alice"

    def test_the_facade_receives_the_token_identity_not_the_request(self) -> None:
        facade = _facade()
        client, cfg = self._client(facade)
        r = client.get("/memories/recall", params={"query": "q"},
                       headers=_bearer("alice", cfg))
        assert r.status_code == 200
        assert facade.recall.await_args.args[0] == "alice"

    def test_the_role_claim_reaches_the_facade(self) -> None:
        """H1: `_check_access` accepted a role that nothing ever passed."""
        facade = _facade()
        client, cfg = self._client(facade)
        r = client.get("/memories/recall", params={"query": "q"},
                       headers=_bearer("alice", cfg, role="admin"))
        assert r.status_code == 200
        assert facade.recall.await_args.kwargs["role"] == "admin"

    def test_no_token_is_still_401(self) -> None:
        client, _ = self._client(_facade())
        assert client.get("/memories/recall", params={"query": "q"}).status_code == 401

    def test_health_stays_open(self) -> None:
        client, _ = self._client(_facade())
        assert client.get("/health").status_code == 200


class TestAuthOffKeepsTheSingleTenantPosture:
    """With auth disabled the caller names itself. That is the documented default.

    These exist so the fix cannot be read as "auth is now mandatory" — a library
    used in-process by a single-tenant app must keep working unchanged.
    """

    def _client(self, facade):
        return TestClient(create_app(facade, config=_config()))

    def test_query_user_id_is_honoured(self) -> None:
        facade = _facade()
        r = self._client(facade).get("/memories/recall",
                                     params={"user_id": "alice", "query": "q"})
        assert r.status_code == 200
        assert facade.recall.await_args.args[0] == "alice"

    def test_body_user_id_is_honoured(self) -> None:
        facade = _facade()
        r = self._client(facade).post("/memories", json={
            "user_id": "alice", "conversation_id": "c1", "messages": [{"content": "x"}],
        })
        assert r.status_code == 200
        assert facade.add.await_args.args[0] == "alice"

    def test_no_config_at_all_still_works(self) -> None:
        """`create_app(facade)` with no config is how the tests and demos build it."""
        facade = _facade()
        client = TestClient(create_app(facade))
        r = client.get("/memories/recall", params={"user_id": "alice", "query": "q"})
        assert r.status_code == 200

    def test_a_missing_user_id_is_400_not_500(self) -> None:
        """Malformed, not forbidden: there is no identity to compare against."""
        facade = _facade()
        r = self._client(facade).get("/memories/recall", params={"query": "q"})
        assert r.status_code == 400


class TestConfigFailsClosed:
    """C2: the two configurations that silently served everything."""

    def test_auth_enabled_without_a_secret_refuses_to_construct(self) -> None:
        with pytest.raises(ValueError, match="AUTH_SECRET"):
            _config(auth_enabled=True, auth_secret="")

    def test_auth_enabled_with_a_secret_is_fine(self) -> None:
        assert _auth_config().auth_enabled is True

    def test_require_auth_refuses_to_start_unauthenticated(self) -> None:
        with pytest.raises(ValueError, match="AUTH_ENABLED"):
            _config(require_auth_for_multi_tenant=True)

    def test_require_auth_with_auth_on_is_fine(self) -> None:
        assert _auth_config(require_auth_for_multi_tenant=True).auth_enabled is True

    def test_the_default_posture_is_unchanged(self) -> None:
        """Auth off by default. The library's primary use is in-process."""
        cfg = _config()
        assert cfg.auth_enabled is False
        assert cfg.require_auth_for_multi_tenant is False


class TestMcpToolsResolveIdentityToo:
    """The MCP shell had the same hole with `user_id` as a tool argument.

    Tools are registered as closures on a FastMCP instance, so these drive them
    through a recording double rather than a live server: the assertion is that
    each tool routes `user_id` through `resolve_caller` before touching the facade.
    """

    def _tools(self, facade, access=None, monkeypatch=None):
        from agent_memory.shells.mcp import tools as tools_mod

        registered: dict = {}

        class _Mcp:
            def tool(self, name=None, description=None):
                def deco(fn):
                    registered[name] = fn
                    return fn

                return deco

        monkeypatch.setattr(
            "fastmcp.server.dependencies.get_access_token", lambda: access
        )
        tools_mod.register_all_tools(_Mcp(), facade)
        return registered

    @pytest.mark.asyncio
    async def test_a_tool_naming_another_user_returns_an_error(self, monkeypatch) -> None:
        facade = _facade()
        tools = self._tools(facade, access=_token("alice"), monkeypatch=monkeypatch)
        result = await tools["recall_memory"](user_id="bob", query="q")
        assert "error" in result, "the tool served bob's memories to alice"
        facade.recall.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_token_identity_is_what_reaches_the_facade(self, monkeypatch) -> None:
        facade = _facade()
        tools = self._tools(facade, access=_token("alice"), monkeypatch=monkeypatch)
        await tools["recall_memory"](user_id="alice", query="q")
        assert facade.recall.await_args.args[0] == "alice"

    @pytest.mark.asyncio
    async def test_the_role_claim_reaches_the_facade(self, monkeypatch) -> None:
        facade = _facade()
        tools = self._tools(facade, access=_token("alice", role="admin"),
                            monkeypatch=monkeypatch)
        await tools["recall_memory"](user_id="alice", query="q")
        assert facade.recall.await_args.kwargs["role"] == "admin"

    @pytest.mark.asyncio
    async def test_wipe_cannot_target_another_user(self, monkeypatch) -> None:
        facade = _facade()
        tools = self._tools(facade, access=_token("alice"), monkeypatch=monkeypatch)
        result = await tools["wipe_user_data"](user_id="bob", confirm=True)
        assert "error" in result
        facade.wipe_user_data.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_token_context_keeps_the_caller_supplied_id(self, monkeypatch) -> None:
        """stdio transport and in-process calls have no request and no token."""
        facade = _facade()
        tools = self._tools(facade, access=None, monkeypatch=monkeypatch)
        await tools["recall_memory"](user_id="alice", query="q")
        assert facade.recall.await_args.args[0] == "alice"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool,kwargs",
        [
            ("store_memory", {"conversation_id": "c1", "messages": []}),
            ("hybrid_search", {"query": "q"}),
            ("delete_memory", {"confirm": True}),
            ("check_cache", {"query": "q"}),
            ("store_cache", {"query": "q", "response": "r"}),
            ("cache_invalidate", {"invalidate_all": True}),
            ("store_decision", {"key": "k", "value": "v"}),
            ("recall_decision", {"key": "k"}),
            ("log_activity", {"thread_id": "t1", "messages": []}),
            ("search_activity", {"query": "q"}),
            ("get_thread", {"thread_id": "t1"}),
            ("get_correlation", {"correlation_id": "c1"}),
            ("set_activity_retention", {"ttl_seconds": 60}),
            ("memory_health", {}),
        ],
    )
    async def test_every_tool_refuses_a_foreign_user_id(self, tool, kwargs, monkeypatch) -> None:
        """Parametrized over the whole surface: one missed tool is the whole hole."""
        facade = _facade()
        tools = self._tools(facade, access=_token("alice"), monkeypatch=monkeypatch)
        result = await tools[tool](user_id="bob", **kwargs)
        assert isinstance(result, dict) and "error" in result, (
            f"{tool} acted as bob on alice's token"
        )


class TestAutoCaptureCannotWriteCrossTenant:
    """A refused tool call must not be captured into the account it named.

    The tools refuse a foreign `user_id` by *returning* `{"error": ...}` rather
    than raising, so `wrap_tools`' wrapper sees an ordinary result and fires
    auto-capture anyway. Capture then read `params["user_id"]` — the raw tool
    argument, the one value identity resolution exists to distrust — so Alice
    could call any wrapped tool with `user_id="bob"`, be denied, and still have
    the refusal text embedded and stored under Bob.

    That is a cross-tenant *write* reachable by any authenticated caller, and an
    injection vector besides: the attacker controls `params`, so they control the
    text that lands in the victim's memory and is later recalled to an LLM.
    """

    def _wrapped(self, facade, config, access, monkeypatch):
        """Register the real tools, then wrap them the way the lifespan does."""
        from agent_memory.shells.mcp import tools as tools_mod
        from agent_memory.shells.mcp.auto_capture import AutoCaptureMiddleware, wrap_tools

        registered: dict = {}

        class _Component:
            def __init__(self, name, fn):
                self.name = name
                self.fn = fn

        class _Mcp:
            def __init__(self):
                self.local_provider = MagicMock()

            def tool(self, name=None, description=None):
                def deco(fn):
                    registered[name] = _Component(name, fn)
                    return fn

                return deco

        monkeypatch.setattr(
            "fastmcp.server.dependencies.get_access_token", lambda: access
        )
        mcp = _Mcp()
        tools_mod.register_all_tools(mcp, facade)
        mcp.local_provider._components = {
            f"tool:{name}": comp for name, comp in registered.items()
        }
        capture = AutoCaptureMiddleware(facade, config)
        wrap_tools(mcp, capture)
        return {name: comp.fn for name, comp in registered.items()}, capture

    def _capture_config(self) -> MemoryConfig:
        return _auth_config(
            auto_capture_enabled=True,
            auto_capture_tools=["recall_memory", "hybrid_search"],
            auto_capture_min_length=1,
        )

    @pytest.mark.asyncio
    async def test_refused_call_is_not_stored_under_the_named_victim(
        self, monkeypatch
    ) -> None:
        config = self._capture_config()
        facade = _facade()
        facade.config = config
        tools, capture = self._wrapped(
            facade, config, _token("alice"), monkeypatch
        )

        result = await tools["recall_memory"](user_id="bob", query="q")
        assert "error" in result, "precondition: the tool must refuse"
        await capture.drain(2.0)

        stored = [c.args[0] for c in facade.add.await_args_list]
        assert "bob" not in stored, (
            f"auto-capture wrote into a foreign tenant: add() called for {stored}"
        )

    @pytest.mark.asyncio
    async def test_a_refusal_is_not_stored_at_all(self, monkeypatch) -> None:
        """Not even under the caller's own account.

        Storing a denial as a memory records an operation that did not happen, and
        the text is attacker-chosen. There is no account this belongs in.
        """
        config = self._capture_config()
        facade = _facade()
        facade.config = config
        tools, capture = self._wrapped(
            facade, config, _token("alice"), monkeypatch
        )

        await tools["recall_memory"](user_id="bob", query="q")
        await capture.drain(2.0)
        facade.add.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_allowed_call_is_still_captured_under_the_token(
        self, monkeypatch
    ) -> None:
        """The fix must not disable auto-capture for legitimate calls."""
        config = self._capture_config()
        facade = _facade()
        facade.config = config
        tools, capture = self._wrapped(
            facade, config, _token("alice"), monkeypatch
        )

        await tools["recall_memory"](user_id="alice", query="q")
        await capture.drain(2.0)
        facade.add.assert_awaited_once()
        assert facade.add.await_args.args[0] == "alice"

    @pytest.mark.asyncio
    async def test_capture_uses_the_token_not_the_argument(self, monkeypatch) -> None:
        """With auth on, the token decides even when both are present.

        `recall_memory` honours a matching `user_id`, so this passes through the
        allowed path — the assertion is about *which* value capture used.
        """
        config = self._capture_config()
        facade = _facade()
        facade.config = config
        tools, capture = self._wrapped(
            facade, config, _token("alice"), monkeypatch
        )

        await tools["recall_memory"](user_id="alice", query="q")
        await capture.drain(2.0)
        assert facade.add.await_args.args[0] == "alice"

    @pytest.mark.asyncio
    async def test_single_tenant_capture_still_uses_the_supplied_id(
        self, monkeypatch
    ) -> None:
        """Auth off is the documented single-tenant posture; capture must work."""
        config = _config(
            auto_capture_enabled=True,
            auto_capture_tools=["recall_memory"],
            auto_capture_min_length=1,
        )
        facade = _facade()
        facade.config = config
        tools, capture = self._wrapped(facade, config, None, monkeypatch)

        await tools["recall_memory"](user_id="solo", query="q")
        await capture.drain(2.0)
        facade.add.assert_awaited_once()
        assert facade.add.await_args.args[0] == "solo"
