"""The demo seed script's reset paths, which the booth runs twice.

One asymmetry drives every test here. The demo has two users: the seeded one,
which proves recall, and a second one, which proves per-user *isolation* by
recalling nothing. `seed()` can only wipe as a prelude to planting, so before
`--wipe-only` there was no way to express "empty this user and stop".

That mattered because unseeded is not the same as empty. The isolation beat types
a question at the second user with memory ON, and that question is stored — so
after one rehearsal the "empty" user held 2 short-term documents, 1 episode, and 1
cache entry, one of them the demo's own question verbatim. The next run can then
recall its own residue and return hits where the entire point is zero. Nothing
errors; the beat just stops proving anything.

So these tests pin the two properties a reset needs: it clears every collection a
user's data spans, and it plants nothing afterwards.

Which collections those are moved once. `wipe_user_data` originally cleared three
collections while promising "all user data", so the script deleted `episodes`
itself. The library now covers every user-scoped collection it owns — `episodes`
among them — leaving only `demo_response_cache`, which is the demo's own table and
genuinely invisible to the library. Deleting `episodes` here as well was harmless
in itself, but the server's `/reset` did the same thing and then reported *its*
count, overwriting the library's: a reset that cleared nine episodes reported zero.

Hence the direction these tests now pin. `demo_response_cache` must be deleted by
the script, and `episodes` must **not** be — because a second delete is how the
real count got lost.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_DEMO_ROOT = Path(__file__).resolve().parents[2] / "examples" / "memory-ui"
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

pytest.importorskip(
    "sse_starlette",
    reason="demo tooling; install the 'demo' extra",
)


def _import_demo(module_name: str):
    """Import a demo module without letting its `load_dotenv` reach `os.environ`.

    Both `demo.seed` and `server.app` load the repository's real `.env` at module
    scope, which is right for a script and wrong for a test process: importing one
    would publish live configuration into `os.environ` for the rest of the session,
    and any test asserting a *default* would then read the developer's actual value
    instead. That failure is order-dependent and lands in unrelated files — three
    config tests broke this way the first time, and ten the second, when
    `server.app` was imported inside a test body where this guard could not see it.
    All of them passed in isolation.

    So neutralize the call for the duration of the import, and snapshot the
    environment around it to prove nothing leaked either way. Import demo modules
    only through here.
    """
    before = dict(os.environ)
    with patch("dotenv.load_dotenv", return_value=False):
        module = importlib.import_module(module_name)
    assert dict(os.environ) == before, f"importing {module_name} mutated os.environ"
    return module


_seed = _import_demo("demo.seed")
_wipe = _seed._wipe
main = _seed.main

# Imported here rather than inside the route test, so it goes through the guard
# above. `server.app` calls `load_dotenv` at module scope exactly as the seed
# script does.
_server_app = _import_demo("server.app")

# The collections the *demo* owns and the library cannot see, so the script has to
# name them itself. `memories` and `episodes` are the library's, reached through
# `wipe_user_data` rather than a direct delete.
DEMO_OWNED_COLLECTIONS = ("demo_response_cache",)

# Cleared by `wipe_user_data`, so a second delete here would double-count. That is
# not hypothetical: the server's `/reset` overwrote the library's episode count
# with its own zero.
LIBRARY_OWNED_COLLECTIONS = ("memories", "episodes", "audit_log", "semantic_cache")


@pytest.fixture(autouse=True)
def _no_shutdown_residue():
    """Leave the process's SSE shutdown state clean around every test here.

    Several tests in this file run the demo's real lifespan, and its *exit* sets
    the module-level `SHUTDOWN` event — in production the process is exiting, so
    nothing cleared it. Left set, any later test that streams a turn gets a
    `shutdown` error frame instead of its own output. That has bitten twice: seven
    tests in `test_demo_ui_server.py` failed on residue from this file, and then
    the last test in `TestASecondLifecycleStillServesTurns` hung on residue from
    the test before it — waiting on a generator the producer refused to start.
    Both were order-dependent and both passed in isolation.

    The production fix (`reset_shutdown_state` at lifespan startup) makes a second
    *lifecycle* clean, which is the bug that was reported. It cannot help a test
    that streams a turn without running a lifespan at all, so this stays.
    """
    import server.sse as sse

    sse.reset_shutdown_state()
    try:
        yield
    finally:
        sse.reset_shutdown_state()


def _fake_db() -> tuple[MagicMock, dict[str, MagicMock]]:
    """A `db[name]` mapping whose collections record their `delete_many` calls."""
    collections: dict[str, MagicMock] = {}

    def getitem(name: str) -> MagicMock:
        if name not in collections:
            col = MagicMock()
            col.delete_many = AsyncMock(
                return_value=MagicMock(deleted_count=0)
            )
            collections[name] = col
        return collections[name]

    db = MagicMock()
    db.__getitem__.side_effect = getitem
    return db, collections


def _fake_memory() -> MagicMock:
    memory = MagicMock()
    memory.wipe_user_data = AsyncMock(return_value={"memories_deleted": 7})
    memory.close = AsyncMock()
    return memory


class TestWipeCoversEveryCollection:
    """A partial wipe is worse than none: it looks clean and leaves residue."""

    @pytest.mark.asyncio
    async def test_library_memories_are_wiped_through_the_facade(self) -> None:
        memory, (db, _) = _fake_memory(), _fake_db()

        await _wipe(memory, db, "alex")

        memory.wipe_user_data.assert_awaited_once_with("alex", confirm=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("collection", DEMO_OWNED_COLLECTIONS)
    async def test_demo_owned_collections_are_wiped_explicitly(
        self, collection: str
    ) -> None:
        """`wipe_user_data` does not know about these, so the script must.

        Both were found populated for the supposedly-empty user after a rehearsal.
        The cache entry is the subtler one: it does not appear in a memory count,
        but it can serve the isolation beat a cached answer.
        """
        memory, (db, collections) = _fake_memory(), _fake_db()

        await _wipe(memory, db, "alex")

        collections[collection].delete_many.assert_awaited_once_with(
            {"user_id": "alex"}
        )

    @pytest.mark.asyncio
    async def test_the_filter_is_scoped_to_one_user(self) -> None:
        """A reset that dropped the whole collection would take the demo user too."""
        memory, (db, collections) = _fake_memory(), _fake_db()

        await _wipe(memory, db, "alex")

        for name in DEMO_OWNED_COLLECTIONS:
            (filter_arg,) = collections[name].delete_many.await_args.args
            assert filter_arg == {"user_id": "alex"}
            assert filter_arg != {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("collection", LIBRARY_OWNED_COLLECTIONS)
    async def test_library_collections_are_not_deleted_twice(
        self, collection: str
    ) -> None:
        """The library clears these, so a second delete here can only mislead.

        `episodes` is the one that moved: the script used to delete it and report
        its own count. Harmless in the script, but the server's `/reset` did the
        same and *spread* the library's result first, so its zero replaced the real
        figure and a reset that cleared nine episodes displayed none.
        """
        memory, (db, collections) = _fake_memory(), _fake_db()

        await _wipe(memory, db, "alex")

        assert collection not in collections, (
            f"_wipe deleted {collection!r} directly; wipe_user_data already covers "
            f"it, and a second delete overwrites the real count with zero"
        )

    @pytest.mark.asyncio
    async def test_the_logged_counts_come_from_the_library(self) -> None:
        """What the presenter reads on the terminal has to be the real number."""
        memory, (db, _) = _fake_memory(), _fake_db()
        memory.wipe_user_data = AsyncMock(
            return_value={"memories_deleted": 20, "episodes_deleted": 3}
        )

        with patch.object(_seed.log, "info") as logged:
            await _wipe(memory, db, "alex")

        rendered = logged.call_args.args[0] % logged.call_args.args[1:]
        assert "memories=20" in rendered, rendered
        assert "episodes=3" in rendered, rendered


class TestAPartialWipeStopsTheSeed:
    """A half-cleared user is the state that must not be reported as ready.

    `wipe_user_data` raises `PartialWipeError` rather than returning it, because
    the audit status is derived from whether the call raised. For this script the
    consequence is different but no less real: seeding on top of a half-cleared
    user leaves exactly the stale documents a recall beat can surface.
    """

    def test_it_is_reported_as_not_ready_and_exits_nonzero(
        self, monkeypatch, capsys
    ) -> None:
        """Through the same channel as the other pre-flight failures.

        A raw traceback on the morning of a talk reads as a broken script rather
        than as "run this again".
        """
        from agent_memory.services.admin import PartialWipeError

        def boom(coro):
            coro.close()
            raise PartialWipeError(
                {"user_id": "alex", "memories_deleted": 2}, {"episodes": "down"}
            )

        monkeypatch.setattr("demo.seed.asyncio.run", boom)
        monkeypatch.setattr(sys, "argv", ["seed", "--user", "alex"])

        assert main() == 1
        err = capsys.readouterr().err
        assert "NOT READY TO PRESENT" in err
        assert "episodes" in err, "the operator is not told what is still there"
        assert "safe to repeat" in err, "the operator is not told what to do"

    def test_a_clean_run_is_unaffected(self, monkeypatch) -> None:
        monkeypatch.setattr("demo.seed.asyncio.run", lambda coro: coro.close() or 0)
        monkeypatch.setattr(sys, "argv", ["seed", "--user", "alex"])

        assert main() == 0


class TestWipeOnlyPlantsNothing:
    """The flag exists so that emptying a user does not also populate it."""

    def test_wipe_only_never_reaches_the_seed_path(self, monkeypatch) -> None:
        """`--wipe-only` must call `wipe_only`, not `seed`.

        Routing it to `seed()` would wipe *and re-plant*, giving the second user
        the demo user's memories and inverting the isolation beat — the exact
        failure the flag was added to prevent.
        """
        calls: list[str] = []
        monkeypatch.setattr(
            "demo.seed.wipe_only", lambda user: calls.append(f"wipe:{user}") or 0
        )
        monkeypatch.setattr(
            "demo.seed.seed",
            lambda *a, **k: pytest.fail("seed() ran under --wipe-only"),
        )
        monkeypatch.setattr("demo.seed.asyncio.run", lambda coro: coro)
        monkeypatch.setattr(sys, "argv", ["seed", "--user", "alex", "--wipe-only"])

        assert main() == 0
        assert calls == ["wipe:alex"]

    def test_the_default_path_still_seeds(self, monkeypatch) -> None:
        """Without the flag, nothing changes — this is an addition, not a rewrite."""
        calls: list[str] = []
        monkeypatch.setattr(
            "demo.seed.wipe_only",
            lambda user: pytest.fail("wipe_only ran without --wipe-only"),
        )
        monkeypatch.setattr(
            "demo.seed.seed",
            lambda user, **kwargs: calls.append(f"seed:{user}") or 0,
        )
        monkeypatch.setattr("demo.seed.asyncio.run", lambda coro: coro)
        monkeypatch.setattr(sys, "argv", ["seed", "--user", "ai4-demo"])

        assert main() == 0
        assert calls == ["seed:ai4-demo"]

    def test_wipe_only_with_keep_is_rejected(self, monkeypatch) -> None:
        """`--keep` means "add to existing data"; together they contradict.

        Silently letting one win would make the reset's outcome depend on flag
        order, which is not a thing to discover at a booth.
        """
        monkeypatch.setattr(
            "demo.seed.wipe_only",
            lambda user: pytest.fail("wiped despite a contradictory flag"),
        )
        monkeypatch.setattr("demo.seed.asyncio.run", lambda coro: coro)
        monkeypatch.setattr(
            sys, "argv", ["seed", "--user", "alex", "--wipe-only", "--keep"]
        )

        # argparse's `error()` exits rather than raising something catchable.
        with pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code != 0



class TestTheResetRouteReportsRealCounts:
    """The on-stage reset. The same duplicate delete, with a visible consequence.

    The route spread the library's result and *then* overwrote `episodes_deleted`
    from its own second delete — which necessarily found nothing, because the
    library had already cleared the collection. The number the presenter reads was
    structurally guaranteed to be zero.

    Driven through the real route, via the real lifespan with only the external
    dependencies faked. The defect was the order of two dict writes; a test that
    restates that order cannot see it.
    """

    @staticmethod
    async def _client(*, failing: set[str] | None = None):
        """The real app, with Atlas and the providers replaced."""

        from agent_memory.services.admin import AdminService
        from tests.unit.test_erasure_is_final import _DB

        db = _DB(failing=failing or set())
        memory = MagicMock()
        memory._db_manager.db = db
        memory.close = AsyncMock()
        admin = AdminService(db)

        async def wipe(user_id, confirm=False):
            return await admin.wipe_user_data(user_id)

        memory.wipe_user_data = AsyncMock(side_effect=wipe)

        cache = MagicMock()
        cache.ensure_indexes = AsyncMock()
        cache.clear = AsyncMock(return_value=2)

        with patch("server.app.MemoryConfig") as config_cls, \
             patch("server.app.AsyncMemory") as memory_cls, \
             patch("server.app.DemoResponseCache", return_value=cache), \
             patch("server.app.TurnRunner"):
            config_cls.from_env.return_value = MagicMock(llm_provider="bedrock")
            memory_cls.create = AsyncMock(return_value=memory)

            app = _server_app.create_app()
            # The routes 503 until the lifespan has populated `state`, so it has
            # to actually run — that is also what makes this the real wiring
            # rather than a hand-built dict. Starlette's own TestClient runs it;
            # a bare httpx transport does not.
            from starlette.testclient import TestClient

            # No `SHUTDOWN` bookkeeping here any more. This helper used to save and
            # restore the flag, because running the lifespan runs its *shutdown*,
            # which sets a module-level event that nothing cleared — seven tests in
            # `test_demo_ui_server.py` failed on the residue, none of them touching
            # this file, and all of them passing in isolation. The lifespan clears
            # it on the way in now, which is the production fix as well: see
            # `TestASecondLifecycleStillServesTurns`.
            with TestClient(app) as client:
                yield client, db

    @pytest.mark.asyncio
    async def test_the_episode_count_is_the_librarys_own(self) -> None:
        async for client, db in self._client():
            db["episodes"].docs.extend([{"user_id": "ai4-demo"}] * 9)
            db["memories"].docs.extend([{"user_id": "ai4-demo"}] * 4)

            r = client.post("/reset",
                            json={"user_id": "ai4-demo", "confirm": True})

            assert r.status_code == 200, r.text
            body = r.json()
            assert body["episodes_deleted"] == 9, (
                f"the route reported {body['episodes_deleted']} episodes where the "
                f"library deleted 9 — its own duplicate delete overwrote the count"
            )
            assert body["memories_deleted"] == 4
            assert body["demo_cache_deleted"] == 2
            assert body["complete"] is True

    @pytest.mark.asyncio
    async def test_a_partial_wipe_is_a_conflict_not_a_crash(self) -> None:
        """409 with the counts. A 500 would say "the server is broken" when what
        happened is that some of the user's data is still there — and the
        difference decides whether the operator retries or starts debugging."""
        async for client, db in self._client(failing={"decisions"}):
            db["memories"].docs.extend([{"user_id": "ai4-demo"}] * 3)

            r = client.post("/reset",
                            json={"user_id": "ai4-demo", "confirm": True})

            assert r.status_code == 409, r.text
            detail = r.json()["detail"]
            assert detail["complete"] is False
            assert detail["memories_deleted"] == 3
            assert detail["failed_collections"] == ["decisions"]

    @pytest.mark.asyncio
    async def test_the_confirm_gate_still_holds(self) -> None:
        async for client, db in self._client():
            db["memories"].docs.append({"user_id": "ai4-demo"})

            r = client.post("/reset", json={"user_id": "ai4-demo"})

            assert r.status_code == 400
            assert db["memories"].docs, "a reset ran without confirmation"


class TestASecondLifecycleStillServesTurns:
    """Shutdown state is per *process*, so it has to be cleared on the way in.

    ``SHUTDOWN`` is a module-level ``asyncio.Event``. Shutdown set it and nothing
    cleared it, on the reasoning that the process is exiting — true of
    ``uvicorn server.app:app``, and false of anything that runs the lifespan twice
    in one interpreter: ``pytest``'s ``TestClient``, a reload-on-edit dev loop, an
    embedding host that mounts the app, a rehearsal harness that restarts the demo
    between passes.

    What that costs is a server that looks healthy and refuses everything.
    ``_producer`` checks ``SHUTDOWN.is_set()`` before the first frame, so every
    ``/chat`` request in the second lifecycle returns a well-formed stream whose
    only content is a ``shutdown`` error. ``/health`` is fine, the log is quiet, and
    on stage that is a demo that has stopped working with nothing to point at.

    This suite is where it was found: `_client` above had to save and restore the
    flag by hand, or seven tests in another file failed. That workaround is gone.
    """

    @pytest.mark.asyncio
    async def test_a_turn_after_a_previous_shutdown_is_not_refused(self) -> None:
        from server.sse import SHUTDOWN, sse_response

        # Exactly the state a previous lifecycle's shutdown leaves behind.
        SHUTDOWN.set()
        async for _client, _db in TestTheResetRouteReportsRealCounts._client():
            assert SHUTDOWN.is_set() is False, (
                "the lifespan started with a previous shutdown still in force; "
                "every turn in this lifecycle would answer with a shutdown frame"
            )

            # Not just the flag — the frame a client would actually receive.
            async def drive():
                yield {"event": "token", "data": "hello"}

            response = sse_response(drive, "cid-second-life", 5.0)
            frames = [f async for f in response.body_iterator]
            assert [f["event"] for f in frames] == [
                "correlation", "token", "done",
            ], frames

    @pytest.mark.asyncio
    async def test_shutdown_still_stops_new_turns(self) -> None:
        """Paired. Clearing the flag unconditionally — rather than only at startup
        — would satisfy the test above and disable the drain the flag exists for:
        a turn arriving mid-shutdown would be accepted against a closing database.
        """
        from server.sse import SHUTDOWN, sse_response

        async for _client, _db in TestTheResetRouteReportsRealCounts._client():
            pass
        # The lifespan has exited, so shutdown is in force again.
        assert SHUTDOWN.is_set() is True, "shutdown no longer refuses new turns"

        async def drive():  # pragma: no cover - must never be entered
            yield {"event": "token", "data": "should not run"}

        frames = [
            f async for f in sse_response(drive, "cid", 5.0).body_iterator
        ]
        assert frames == [
            {"event": "correlation", "data": "cid"},
            {"event": "error", "data": "shutdown"},
        ]

    @pytest.mark.asyncio
    async def test_streams_from_a_dead_loop_are_not_carried_forward(
        self, caplog
    ) -> None:
        """A stream that outlived its drain timeout stays in `_IN_FLIGHT`.

        Its task is bound to an event loop that no longer exists, so the next
        lifespan's `drain_in_flight` would gather a cross-loop task during
        shutdown — a failure at the least convenient moment, and one that is
        logged and dropped rather than reported.

        Discarding it is the right call — there is no loop left to drain it on —
        but it is not free: that stream may have owed a memory write-back, and
        dropping it silently means a turn vanishes with no record that it did. So
        the log line is asserted, not just the discard.
        """
        import logging

        import server.sse as sse

        stale = MagicMock()  # a task object from a loop that is gone
        sse._IN_FLIGHT.add(stale)
        try:
            with caplog.at_level(logging.WARNING, logger="server.sse"):
                async for _c, _db in TestTheResetRouteReportsRealCounts._client():
                    assert stale not in sse._IN_FLIGHT, (
                        "a stream from a previous lifespan survived into this one"
                    )
            assert any(
                "previous lifespan" in r.message and r.levelno >= logging.WARNING
                for r in caplog.records
            ), "a leftover stream was dropped without a word"
        finally:
            sse._IN_FLIGHT.discard(stale)

    @pytest.mark.asyncio
    async def test_a_clean_startup_says_nothing(self) -> None:
        """Paired with the warning above. A line on every ordinary startup is a
        line the presenter learns to ignore, which is how the real one gets
        missed."""
        import logging

        import server.sse as sse

        assert sse._IN_FLIGHT == set()
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append
        logger = logging.getLogger("server.sse")
        logger.addHandler(handler)
        try:
            async for _c, _db in TestTheResetRouteReportsRealCounts._client():
                pass
        finally:
            logger.removeHandler(handler)
        assert [r for r in records if r.levelno >= logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_shutdown_drains_before_closing_the_database(self) -> None:
        """What makes clearing `_IN_FLIGHT` at startup safe.

        Startup can discard leftovers only because shutdown is supposed to have
        drained them. Drop the drain and the two changes compose into silent loss:
        every unfinished stream is abandoned at shutdown and then thrown away at
        the next startup, so a turn's memory write-back disappears with a warning
        that names the wrong cause.

        Ordering is the assertion. Draining after `memory.close()` would be no
        drain at all — the streams still writing would find the connection gone.
        """
        from agent_memory.services.admin import AdminService
        from tests.unit.test_erasure_is_final import _DB

        order: list[str] = []
        db = _DB(failing=set())
        memory = MagicMock()
        memory._db_manager.db = db
        memory.wipe_user_data = AsyncMock(
            side_effect=AdminService(db).wipe_user_data
        )

        async def _close():
            order.append("close")

        memory.close = _close

        async def _drain(timeout: float = 10.0):
            order.append("drain")

        cache = MagicMock()
        cache.ensure_indexes = AsyncMock()

        with patch("server.app.MemoryConfig") as config_cls, \
             patch("server.app.AsyncMemory") as memory_cls, \
             patch("server.app.DemoResponseCache", return_value=cache), \
             patch("server.app.drain_in_flight", _drain), \
             patch("server.app.TurnRunner"):
            config_cls.from_env.return_value = MagicMock(llm_provider="bedrock")
            memory_cls.create = AsyncMock(return_value=memory)
            app = _server_app.create_app()
            async with app.router.lifespan_context(app):
                assert order == []

        assert order == ["drain", "close"]

    @pytest.mark.asyncio
    async def test_a_live_stream_is_not_discarded_by_a_reset(self) -> None:
        """Paired. `reset_shutdown_state` clearing `_IN_FLIGHT` is only safe at
        startup, when by definition nothing this loop owns is registered. Called
        anywhere else it would drop live streams from the drain set, so shutdown
        would close the database under a turn that was still writing.
        """
        import server.sse as sse
        from server.sse import sse_response

        started, release = asyncio.Event(), asyncio.Event()

        async def drive():
            started.set()
            await release.wait()
            yield {"event": "token", "data": "done waiting"}

        response = sse_response(drive, "cid-live", 5.0)
        consumer = asyncio.create_task(
            _collect(response.body_iterator)
        )
        await started.wait()
        assert len(sse._IN_FLIGHT) == 1
        release.set()
        await consumer
        # Registration is by done-callback, so it clears itself on completion —
        # which is what makes a *leftover* entry evidence of a dead loop.
        assert sse._IN_FLIGHT == set()


async def _collect(iterator) -> list:
    return [frame async for frame in iterator]


class TestTheDemoModulesDoNotLeakTheEnvironment:
    """Importing a demo module must not publish the real `.env` into the process.

    Both demo entry points call `load_dotenv` at module scope, which is correct for
    something you run from a shell and hostile inside a test session: the live
    `VOYAGE_API_KEY`, endpoint, and embedding dimension become `os.environ` for
    every test that follows, so anything asserting a *default* reads the
    developer's actual value. The failures land in unrelated files and every one of
    them passes in isolation.

    This has happened twice — first from `demo.seed`, then from `server.app`
    imported inside a test body where the module-scope guard could not apply. So it
    is asserted rather than documented.

    Asserted at the level of "every demo module this file imports goes through the
    guard", because re-importing to observe the leak directly is not reproducible:
    the values are already in `os.environ` from the guarded import at module scope,
    and `load_dotenv` will not overwrite them.
    """

    # Every demo module this file pulls in. A new one added without going through
    # `_import_demo` is the failure being prevented.
    DEMO_MODULES = ("demo.seed", "server.app")

    @pytest.mark.parametrize("module_name", DEMO_MODULES)
    def test_the_module_was_imported_through_the_guard(self, module_name: str) -> None:
        """`_import_demo` asserts the environment is unchanged around the import,
        so a module reaching this file any other way is what to catch."""
        assert module_name in sys.modules, (
            f"{module_name} is not imported; if this file stopped needing it, drop "
            f"it from DEMO_MODULES"
        )

    @pytest.mark.parametrize("module_name", DEMO_MODULES)
    def test_the_module_really_does_load_a_dotenv(self, module_name: str) -> None:
        """Otherwise the guard is load-bearing for nothing and the next person
        removes it. Read from the source, since the call already happened."""
        module = sys.modules[module_name]
        source = Path(module.__file__).read_text()
        assert "load_dotenv(" in source, (
            f"{module_name} no longer loads a .env at import — the guard in "
            f"_import_demo may now be unnecessary, but confirm before removing it"
        )

    def test_the_guard_restores_load_dotenv_afterwards(self) -> None:
        """It patches a shared module. Leaving it patched would silently stop the
        *next* test's legitimate dotenv use."""
        import dotenv

        _import_demo("demo.seed")  # already imported; exercises the patch path
        assert not isinstance(dotenv.load_dotenv, MagicMock)

    def test_no_live_credential_reached_the_environment(self) -> None:
        """The concrete consequence, stated as the thing that must not be true.

        A leaked `VOYAGE_API_KEY` made `test_voyage_defaults` assert the real key
        against `None`, which also printed it into the failure output.
        """
        for key in ("VOYAGE_API_KEY", "VOYAGE_API_URL", "EMBEDDING_DIMENSION"):
            assert key not in os.environ, (
                f"{key} leaked into os.environ from a demo module import; tests "
                f"asserting defaults will read the live value instead"
            )
