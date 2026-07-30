"""Ambient per-user scoping — a ContextVar that survives async and thread hops.

Memory is per-user, always. But the call site that knows the user (an HTTP
handler, a message consumer) is usually not the call site that logs the turn
(a callback deep inside an agent loop, with no argument for it). This module
carries the user id across that gap without threading a parameter through
every layer.

::

    with scoped_user(request.user_id):
        await agent.run(prompt)          # anything inside can call current_user_id()

``ContextVar`` is the right primitive rather than a module global: each asyncio
Task and each thread gets its own copy, so concurrent requests in one process
cannot read each other's user id.

The variable itself is module-private on purpose. Exposing it would invite
``_user_id_var.set(x)`` without a matching ``reset()``, which leaks the id into
whatever the framework runs next on that task — the exact bug this is meant to
prevent. Use the context manager.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_user_id_var: ContextVar[str | None] = ContextVar(
    "agent_memory_user_id", default=None
)


def current_user_id() -> str | None:
    """Return the user id in scope, or ``None`` outside any ``scoped_user``."""
    return _user_id_var.get()


@contextmanager
def scoped_user(user_id: str) -> Iterator[None]:
    """Bind ``user_id`` for the duration of the block.

    Nests correctly — the previous value is restored on exit, including on an
    exception — because the token from ``set()`` is always reset in ``finally``.
    """
    token = _user_id_var.set(user_id)
    try:
        yield
    finally:
        _user_id_var.reset(token)


__all__ = ["current_user_id", "scoped_user"]
