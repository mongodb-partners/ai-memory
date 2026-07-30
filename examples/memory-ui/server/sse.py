"""SSE transport: turn an async generator of frames into a resilient stream.

This module knows nothing about memory or LLMs. It handles the parts of
server-sent events that are easy to get subtly wrong:

* the leading ``correlation`` frame, so the client has a trace id before any
  content arrives
* a wall-clock timeout per turn, applied per-frame rather than to the whole
  stream, so a model that stops mid-answer is caught
* a trailing ``done``, and distinct ``error`` codes for timeout vs shutdown vs
  an unexpected failure
* in-flight registration, so a shutdown can cancel live streams instead of
  killing the process underneath them
* generator close on every exit path, including client disconnect

The producer/consumer split via a bounded queue is what makes cancellation work:
the consumer is the response body, so when the client disconnects the consumer
stops, and its ``finally`` cancels the producer. Driving the generator directly
from the response body leaves the producer running after a disconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from sse_starlette.sse import EventSourceResponse

log = logging.getLogger(__name__)

# Set once at shutdown. A turn that has not yet produced its first frame is
# refused outright; a turn already streaming is left alone and bounded by
# `drain_in_flight` instead. Cutting a live turn short would lose both the tail of
# the answer and its memory write-back, and the write is the thing worth keeping.
#
# Module-level, and therefore per *process* rather than per app instance — which is
# only safe because `reset_shutdown_state()` clears it on the way in. See there.
SHUTDOWN = asyncio.Event()

_IN_FLIGHT: set[asyncio.Task[Any]] = set()


def reset_shutdown_state() -> None:
    """Clear process-global shutdown state. Call at the start of every lifespan.

    ``SHUTDOWN`` and ``_IN_FLIGHT`` are module globals, so they outlive the app
    instance that set them. Shutdown sets ``SHUTDOWN`` and nothing ever cleared it,
    on the assumption that the process is exiting — an assumption the code does not
    enforce and that is wrong whenever a lifespan runs twice in one interpreter:

    * ``_producer`` checks ``SHUTDOWN.is_set()`` before the first frame, so a
      second lifecycle answers **every** turn with a ``shutdown`` error frame. The
      server is up, ``/health`` is fine, and each chat request returns a
      well-formed stream containing nothing but a refusal.
    * A stream that missed the drain timeout stays in ``_IN_FLIGHT`` holding a task
      bound to the *previous* event loop. The next ``drain_in_flight`` gathers it
      and gets a cross-loop failure during shutdown, when there is least attention
      to spare.

    Neither symptom points at the previous lifecycle, and the second is discarded
    rather than reported. Stale entries are dropped rather than awaited: their loop
    is gone, so there is nothing left to drain — the log line says so, because
    silently discarding a stream that may still have owed a memory write is worth
    one.
    """
    SHUTDOWN.clear()
    if _IN_FLIGHT:
        log.warning(
            "discarding %d stream(s) left over from a previous lifespan; their "
            "event loop is gone, so they cannot be drained",
            len(_IN_FLIGHT),
        )
        _IN_FLIGHT.clear()

# Bounded so a slow client applies backpressure to the model loop instead of
# letting the queue grow without limit.
_QUEUE_SIZE = 64


class ShutdownInterrupted(RuntimeError):
    """Raised inside a driver when the server begins draining."""


async def drain_in_flight(timeout: float = 10.0) -> None:
    """Wait for live streams to finish during shutdown. Call after setting
    ``SHUTDOWN``, and before closing the database."""
    if not _IN_FLIGHT:
        return
    log.info("waiting on %d in-flight stream(s)", len(_IN_FLIGHT))
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(
            asyncio.gather(*_IN_FLIGHT, return_exceptions=True), timeout=timeout
        )
    if _IN_FLIGHT:
        log.warning("%d stream(s) did not finish draining", len(_IN_FLIGHT))


def sse_response(
    drive: Any, correlation_id: str, timeout: float
) -> EventSourceResponse:
    """Wrap a frame-yielding async generator factory in the SSE machinery.

    ``drive`` is a zero-argument callable returning an async generator of
    ``{"event": ..., "data": ...}`` dicts. It is a factory rather than a
    generator so nothing starts running until the response is being consumed.
    """

    async def _with_timeout() -> AsyncIterator[dict[str, Any]]:
        agen = drive()
        iterator = agen.__aiter__()
        deadline = time.monotonic() + timeout if timeout > 0 else None
        try:
            while True:
                if deadline is None:
                    try:
                        frame = await iterator.__anext__()
                    except StopAsyncIteration:
                        return
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError
                    try:
                        frame = await asyncio.wait_for(
                            iterator.__anext__(), timeout=remaining
                        )
                    except StopAsyncIteration:
                        return
                yield frame
        finally:
            # The driver holds an open Bedrock response and a Motor cursor;
            # neither is released by garbage collection alone.
            with contextlib.suppress(Exception):
                await agen.aclose()

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_SIZE)

    # Termination is signalled out of band rather than by a sentinel frame. An
    # in-band sentinel has to be enqueued, and the only way to enqueue it from a
    # cancellation handler is `put_nowait` — which raises `QueueFull` exactly when
    # the queue is full, silently dropping the one frame the consumer is waiting
    # for. The stream would then hang until the client gave up. An Event cannot
    # fail to be set.
    finished = asyncio.Event()

    async def _producer() -> None:
        try:
            await queue.put({"event": "correlation", "data": correlation_id})
            try:
                if SHUTDOWN.is_set():
                    # Checked here rather than in the route so the client still
                    # gets a well-formed stream (correlation, then error) instead
                    # of a bare HTTP status it has no handler for.
                    raise ShutdownInterrupted
                async for frame in _with_timeout():
                    await queue.put(frame)
                await queue.put({"event": "done", "data": "[DONE]"})
            except TimeoutError:
                log.warning("turn timeout (%ss) cid=%s", timeout, correlation_id)
                await queue.put({"event": "error", "data": "turn_timeout"})
            except ShutdownInterrupted:
                await queue.put({"event": "error", "data": "shutdown"})
            except asyncio.CancelledError:
                # put_nowait, not await: awaiting during cancellation can hang.
                with contextlib.suppress(Exception):
                    queue.put_nowait({"event": "error", "data": "shutdown"})
                raise
            except Exception:
                log.exception("stream failed cid=%s", correlation_id)
                await queue.put(
                    {"event": "error", "data": f"internal_error cid={correlation_id}"}
                )
        finally:
            finished.set()

    producer = asyncio.create_task(_producer(), name=f"sse-{correlation_id}")
    _IN_FLIGHT.add(producer)
    producer.add_done_callback(_IN_FLIGHT.discard)

    async def _event_stream() -> AsyncGenerator[dict[str, Any], None]:
        finished_wait = asyncio.ensure_future(finished.wait())
        try:
            while True:
                # Drain whatever is already buffered before consulting `finished`.
                # Checking the flag first would truncate the tail: the producer
                # sets it immediately after enqueuing `done`, so the last frames
                # of every successful turn — including the episodic write — would
                # never reach the client.
                try:
                    yield queue.get_nowait()
                    continue
                except asyncio.QueueEmpty:
                    pass
                if finished.is_set():
                    return
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {getter, finished_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    yield getter.result()
                else:
                    # Producer is done; loop back to drain the buffer, then exit.
                    getter.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await getter
        finally:
            finished_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await finished_wait
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await producer

    return EventSourceResponse(_event_stream())
