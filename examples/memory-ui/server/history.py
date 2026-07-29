"""In-process conversation history, bounded per thread.

Chat history is not memory. Keeping it in a dict — lost on restart, never
persisted, never searched — is the point rather than a shortcut: it makes the
distinction the talk rests on visible in the code. What survives a new thread is
what lives in Atlas; what dies with the process is the transcript.

It also keeps the demo honest. When the memory panel shows a fact being recalled
across threads, that fact cannot have come from here, because "new thread" empties
this and touches nothing in the database.
"""

from __future__ import annotations

from collections import OrderedDict, deque

# Turns retained per thread. Two exchanges of context is enough for pronouns to
# resolve; more would blur the line between transcript and memory.
MAX_TURNS_PER_THREAD = 8

# Threads retained overall, evicted oldest-first. A bound rather than a TTL
# because this is a demo process that may run all morning: without it, every
# thread id ever typed stays resident.
MAX_THREADS = 200


class ConversationHistory:
    """Bounded per-thread transcript. Not persisted, not searchable, not memory."""

    def __init__(self) -> None:
        self._threads: OrderedDict[tuple[str, str], deque[tuple[str, str]]] = (
            OrderedDict()
        )

    def append(self, user_id: str, thread_id: str, role: str, text: str) -> None:
        if not text:
            return
        key = (user_id, thread_id)
        turns = self._threads.get(key)
        if turns is None:
            turns = deque(maxlen=MAX_TURNS_PER_THREAD * 2)
            self._threads[key] = turns
        turns.append((role, text))
        self._threads.move_to_end(key)
        while len(self._threads) > MAX_THREADS:
            self._threads.popitem(last=False)

    def turns(
        self, user_id: str, thread_id: str, *, limit: int
    ) -> list[tuple[str, str]]:
        """Return the last ``limit`` exchanges as ``(role, text)`` pairs.

        ``limit`` counts exchanges, not messages, so the returned list holds up
        to ``2 * limit`` entries and always starts on a user turn — a history
        beginning with an assistant message is rejected by some providers.
        """
        turns = list(self._threads.get((user_id, thread_id), ()))
        if not turns:
            return []
        tail = turns[-(limit * 2) :]
        while tail and tail[0][0] != "user":
            tail.pop(0)
        return tail

    def clear(self, user_id: str, thread_id: str | None = None) -> None:
        """Forget one thread, or every thread for a user."""
        if thread_id is not None:
            self._threads.pop((user_id, thread_id), None)
            return
        for key in [k for k in self._threads if k[0] == user_id]:
            del self._threads[key]

    def thread_count(self) -> int:
        return len(self._threads)
