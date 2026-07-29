# How to scope memory to a user

Every memory operation is scoped to a user. `user_id` is the first positional
argument on every method, and there is no unscoped read — not a convention you
should follow, but a signature you cannot avoid.

That covers the easy case. The hard case is when the code that *knows* the user is
not the code that *logs the turn*.

## The problem

An HTTP handler knows the user id. A callback buried inside an agent loop is where
the turn actually gets logged — and it has no argument for it. Threading
`user_id` down through every layer works, but it means every intermediate function
grows a parameter it does not use, and any framework in the middle that does not
forward it breaks the chain.

## `scoped_user`

```python
from agent_memory.core.context import scoped_user, current_user_id

@app.post("/chat")
async def chat(request):
    with scoped_user(request.user_id):
        await run_agent(request.message)      # anything inside can read it


# ...many layers down, with no user_id parameter in sight
async def on_turn_end(messages):
    user_id = current_user_id()
    if user_id:
        await memory.log_activity(user_id, thread_id, messages)
```

`current_user_id()` returns `None` outside any `scoped_user` block. Treat that as
"do not log" rather than "log unscoped" — a turn without a `user_id` is discarded
by the write path anyway, so guarding explicitly just makes the intent visible.

## Why `ContextVar` and not a global

A module-level global would be shared by every concurrent request in the process.
Two users hitting the same server would overwrite each other's id, and the bug
would appear as cross-tenant data — the worst possible failure for a memory
system, and one that only shows up under concurrency.

`ContextVar` gives each asyncio Task and each thread its own copy. Concurrent
requests cannot see each other's value. This is the same primitive
`contextvars`-based request-id propagation uses, for the same reason.

## Why the variable is private

`agent_memory.core.context` exports `current_user_id()` and `scoped_user()`, and
nothing else. The `ContextVar` itself is module-private on purpose.

An exposed variable invites `_user_id_var.set(x)` without a matching `reset()`.
That leaks the id into whatever the framework schedules next on that task —
which, in an async server reusing tasks, can be a different user's request. The
context manager always resets its token in a `finally`, including on an
exception, so the leak is not possible through the public surface.

Nesting therefore works as you would expect:

```python
with scoped_user("u1"):
    assert current_user_id() == "u1"
    with scoped_user("u2"):
        assert current_user_id() == "u2"
    assert current_user_id() == "u1"      # restored
```

## Where the boundary belongs

Set the scope once, at the edge, where the user id is first authenticated:

```python
# FastAPI middleware — one place, every route
@app.middleware("http")
async def bind_user(request, call_next):
    user_id = request.headers.get("x-user-id")
    if not user_id:
        return await call_next(request)
    with scoped_user(user_id):
        return await call_next(request)
```

Not inside the agent, and not per-tool. The point of the edge is that there is
exactly one place to get it right.

## What it does not do

`scoped_user` does **not** make `user_id` optional on the facade. `log_activity`
and every other method still take it explicitly. The context var is for code that
cannot receive it as an argument; passing it directly is better wherever you can.

It also crosses `asyncio.create_task` correctly (the child task copies the context
at creation), but **not** `loop.run_in_executor` unless you propagate the context
yourself:

```python
ctx = contextvars.copy_context()
await loop.run_in_executor(None, lambda: ctx.run(sync_work))
```

Without that, the worker thread sees `None` — which fails closed (no `user_id`, no
document) rather than misattributing the turn.

## Isolation on the read side

Scoping the write path is half of tenant isolation. The read path enforces the
other half: every episodic query includes `user_id` as a mandatory pre-filter, in
the `$vectorSearch` filter *and* in the `$search` compound clause, so neither
branch of the hybrid pipeline can surface another user's turn.

This is why the vector index declares `user_id` as a filter field. An undeclared
filter field in a `$vectorSearch` pre-filter does not raise — the branch silently
returns nothing. In this case the failure direction is safe (no results rather
than the wrong results), but it means an index misconfiguration looks exactly like
an empty collection.

## See also

- [The document shape](../reference/episodic-document-shape.md) — required fields
  and the filter-field declarations
- [Architecture](../explanation/architecture.md) — where the access check sits
