"""Demo backend for the agent-memory sample UI.

Deliberately *not* a package inside ``agent_memory``: this is an example, and
importing it must never become a way to depend on it. Nothing here is shipped in
the wheel or the sdist.

Run it from ``examples/memory-ui``::

    uv run --extra demo uvicorn server.app:app --reload --port 8100

The parent directory is named with a hyphen and is not importable, which is why
the run target is ``server.app`` from inside it rather than a dotted path from
the repository root.

This file stays import-free on purpose. ``app.py`` pulls in FastAPI and
``agent_memory``; making the package import do that would mean a test collecting
``server/`` needed the demo extra installed just to read a helper.
"""

from __future__ import annotations
