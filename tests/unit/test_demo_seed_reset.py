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

So these tests pin the two properties a reset needs: it clears *all three*
collections (the library's `wipe_user_data` covers `memories`, but `episodes` and
the demo's response cache are outside its user-data contract and have to be named
explicitly), and it plants nothing afterwards.
"""

from __future__ import annotations

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


def _import_seed():
    """Import `demo.seed` without letting its `load_dotenv` reach `os.environ`.

    The script loads the repository's real `.env` at module scope, which is right
    for a script and wrong for a test process: importing it would publish live
    configuration into `os.environ` for the rest of the session, and any test
    asserting a *default* would then read the developer's actual value instead.
    That failure is order-dependent and lands in unrelated files — three config
    tests broke this way, all of them passing in isolation.

    So neutralize the call for the duration of the import, and snapshot the
    environment around it to prove nothing leaked either way.
    """
    before = dict(os.environ)
    with patch("dotenv.load_dotenv", return_value=False):
        module = importlib.import_module("demo.seed")
    assert dict(os.environ) == before, "importing demo.seed mutated os.environ"
    return module


_seed = _import_seed()
_wipe = _seed._wipe
main = _seed.main

# The three collections a user's data spans. `memories` is the library's own —
# reached through `wipe_user_data` rather than a direct delete — while these two
# belong to the demo and are invisible to it.
DEMO_OWNED_COLLECTIONS = ("episodes", "demo_response_cache")


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
