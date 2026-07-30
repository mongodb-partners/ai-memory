"""Tests for ambient per-user scoping. REQ-E-096."""

import asyncio
import threading

import pytest

from agent_memory.core.context import current_user_id, scoped_user


class TestScopedUser:
    def test_default_is_none(self):
        # TC-EP-CTX-001
        assert current_user_id() is None

    def test_scope_binds_and_restores(self):
        # TC-EP-CTX-002
        with scoped_user("alice"):
            assert current_user_id() == "alice"
        assert current_user_id() is None

    def test_scopes_nest(self):
        # TC-EP-CTX-003
        with scoped_user("alice"):
            with scoped_user("bob"):
                assert current_user_id() == "bob"
            assert current_user_id() == "alice"
        assert current_user_id() is None

    def test_exception_still_restores(self):
        # TC-EP-CTX-004: a leaked id would attribute the next request's memory
        # to the wrong user.
        with pytest.raises(RuntimeError):
            with scoped_user("alice"):
                raise RuntimeError("boom")
        assert current_user_id() is None


class TestIsolation:
    async def test_concurrent_tasks_do_not_leak(self):
        # TC-EP-CTX-005: each asyncio Task gets its own copy of the ContextVar.
        seen: dict[str, str | None] = {}

        async def run(user_id: str) -> None:
            with scoped_user(user_id):
                await asyncio.sleep(0.01)  # force interleaving
                seen[user_id] = current_user_id()

        await asyncio.gather(run("alice"), run("bob"), run("carol"))
        assert seen == {"alice": "alice", "bob": "bob", "carol": "carol"}

    async def test_outer_scope_is_unaffected_by_a_child_task(self):
        # TC-EP-CTX-006
        async def child() -> None:
            with scoped_user("bob"):
                await asyncio.sleep(0)

        with scoped_user("alice"):
            await asyncio.create_task(child())
            assert current_user_id() == "alice"

    def test_threads_do_not_share(self):
        # TC-EP-CTX-007
        seen: dict[str, str | None] = {}

        def run(user_id: str) -> None:
            with scoped_user(user_id):
                seen[user_id] = current_user_id()

        threads = [
            threading.Thread(target=run, args=(name,)) for name in ("alice", "bob")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert seen == {"alice": "alice", "bob": "bob"}

    def test_a_thread_does_not_inherit_the_callers_scope(self):
        # TC-EP-CTX-008: threads start from a fresh context, so a worker thread
        # must be given the user id explicitly.
        seen: list[str | None] = []

        def run() -> None:
            seen.append(current_user_id())

        with scoped_user("alice"):
            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
        assert seen == [None]
