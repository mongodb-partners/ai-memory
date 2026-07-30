"""AWS Bedrock implementations of EmbeddingProvider and LLMProvider."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator

import boto3
from botocore.config import Config as BotocoreConfig

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    parse_importance,
)

logger = logging.getLogger(__name__)

# Bedrock returns a transient ServiceUnavailableException under load. boto3's
# default (3 attempts, legacy mode) gives up sooner than the model recovers, and
# the caller sees a hard failure on a request that would have succeeded. Adaptive
# mode adds client-side rate limiting on top of the retry, which is the right
# behaviour when several workers share one account's quota.
_RETRY_CONFIG = BotocoreConfig(retries={"max_attempts": 5, "mode": "adaptive"})

# Converse reports a rejected sampling parameter in snake_case, but the request
# field is camelCase. Mapping both ways is what lets the error message name the
# key to remove.
_SAMPLING_PARAM_ALIASES = {
    "temperature": "temperature",
    "top_p": "topP",
    "topp": "topP",
    "top_k": "topK",
    "topk": "topK",
}

_DEPRECATED_PARAM_RE = re.compile(r"`([a-zA-Z_]+)`\s+is\s+deprecated", re.IGNORECASE)


def _build_runtime_client(config: MCPConfig):
    kwargs: dict = {
        "service_name": "bedrock-runtime",
        "region_name": config.aws_region,
        "config": _RETRY_CONFIG,
    }
    if config.aws_access_key_id:
        kwargs["aws_access_key_id"] = config.aws_access_key_id
    if config.aws_secret_access_key:
        kwargs["aws_secret_access_key"] = config.aws_secret_access_key
    return boto3.client(**kwargs)


class BedrockEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings via Amazon Bedrock (Titan Embed Text)."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._client = _build_runtime_client(config)

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a single embedding vector. Runs boto3 in a thread."""
        return await asyncio.to_thread(self._invoke_embedding, text)

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts concurrently."""
        tasks = [self.generate_embedding(t) for t in texts]
        return await asyncio.gather(*tasks)

    def _invoke_embedding(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text})
        response = self._client.invoke_model(
            modelId=self._config.embedding_model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["embedding"]


class BedrockLLMProvider(LLMProvider):
    """LLM calls via Amazon Bedrock (Claude Sonnet)."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._client = _build_runtime_client(config)
        # Sampling parameters this model rejects, learned from its own error and
        # cached per instance so the retry happens at most once per model.
        self._unsupported_sampling: set[str] = set()

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat request to the LLM."""
        return await asyncio.to_thread(self._invoke_converse, messages, **kwargs)

    def user_turn(self, text: str) -> list[dict]:
        """Converse requires content blocks; a bare string is rejected outright."""
        return [{"role": "user", "content": [{"text": text}]}]

    async def assess_importance(self, content: str, prompt: str | None = None) -> float:
        """Ask the LLM to rate importance on a 1-10 scale, normalize to 0.1-1.0."""
        if prompt:
            text = prompt.format(content=content)
        else:
            text = (
                "Rate the importance of the following memory on a scale of 1-10, "
                "where 1 is trivial and 10 is critically important. "
                "Respond with ONLY a single integer.\n\n"
                f"Memory: {content}"
            )
        response = await self.complete(text)
        # Handles both scales, because `prompt` may ask for either. See
        # `parse_importance` — a naive `\d+` silently floors every 0.0-1.0 reply.
        return parse_importance(response)

    async def generate_summary(self, content: str, max_length: int = 100, prompt: str | None = None) -> str:
        """Ask the LLM to summarize content."""
        if prompt:
            text = prompt.format(content=content)
        else:
            text = (
                f"Summarize the following text in {max_length} words or fewer. "
                "Be concise and capture the key points.\n\n"
                f"Text: {content}"
            )
        return await self.complete(text)

    async def chat_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Stream text deltas from Bedrock's ``converse_stream``.

        boto3's event stream is a blocking iterator, so it cannot be consumed
        directly from the event loop. A worker thread drains it into a queue and
        this coroutine yields from the queue — the alternative, calling
        ``next()`` inside ``to_thread`` per event, pays a thread hop per token.

        The sentinel is a tuple ``(kind, payload)`` rather than a bare ``None``
        so a failure inside the thread is re-raised here instead of being
        swallowed into a silently-truncated answer.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=256)

        def _open_stream():
            """Start the stream, retrying once without rejected sampling params.

            The retry is safe here because it happens before any token is
            yielded: ``converse_stream`` validates the request and returns the
            event stream in one call, so a ValidationException means nothing was
            produced and nothing has reached the consumer.
            """
            payload = self._converse_kwargs(messages, **kwargs)
            try:
                return self._client.converse_stream(**payload)["stream"]
            except Exception as exc:
                learned = self._note_unsupported_sampling(exc)
                if not learned or not self._strip_unsupported_sampling(
                    payload, learned
                ):
                    raise
                return self._client.converse_stream(**payload)["stream"]

        def _pump() -> None:
            try:
                stream = _open_stream()
                for event in stream:
                    delta = event.get("contentBlockDelta")
                    if delta:
                        text = delta.get("delta", {}).get("text")
                        if text:
                            # The queue is bounded, so a slow consumer applies
                            # backpressure to the pump rather than buffering an
                            # unbounded response in memory.
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("text", text)), loop
                            ).result()
            # `BaseException`, not `Exception`, and deliberately broad: this runs on
            # a worker thread, where an escaping exception is lost and the consumer
            # below waits forever on a queue nothing will fill. Nothing is
            # swallowed — the exception is handed to the loop and re-raised there.
            except BaseException as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", exc)), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("done", None)), loop
                ).result()

        pump = asyncio.create_task(asyncio.to_thread(_pump))
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "text":
                    yield payload  # type: ignore[misc]
                elif kind == "error":
                    raise payload  # type: ignore[misc]
                else:
                    return
        finally:
            # A consumer that breaks early (client disconnect) must not leave
            # the pump thread holding an open HTTP response.
            if not pump.done():
                pump.cancel()

    def _converse_kwargs(self, messages: list[dict], **kwargs) -> dict:
        """Build the Converse API payload, honouring caller overrides.

        Previously ``**kwargs`` was accepted and then silently discarded, so a
        caller passing a system prompt or a temperature got neither and no
        error. Anything the Converse API accepts (``system``,
        ``inferenceConfig``, ``toolConfig``) now passes through, and ``modelId``
        is overridable for a per-call model switch.

        Sampling parameters already known to be rejected by this model are
        stripped — see :meth:`_strip_unsupported_sampling`.
        """
        payload: dict = {"modelId": self._config.llm_model, "messages": messages}
        payload.update(kwargs)
        if self._unsupported_sampling:
            self._strip_unsupported_sampling(payload, self._unsupported_sampling)
        return payload

    @staticmethod
    def _strip_unsupported_sampling(payload: dict, names: set[str]) -> bool:
        """Remove named keys from ``inferenceConfig``. True if anything changed.

        Copies before mutating: the caller may be reusing one ``inferenceConfig``
        dict across calls, and editing it in place would silently reconfigure
        every other model sharing it.
        """
        config = payload.get("inferenceConfig")
        if not isinstance(config, dict):
            return False
        present = names & set(config)
        if not present:
            return False
        payload["inferenceConfig"] = {
            key: value for key, value in config.items() if key not in present
        }
        return True

    def _note_unsupported_sampling(self, exc: Exception) -> set[str]:
        """Extract rejected sampling parameter names from a ValidationException.

        The newest Claude models on Bedrock reject ``temperature`` and ``top_p``
        outright — the API's own words are "`temperature` is deprecated for this
        model" — while every earlier model accepts them. A library cannot carry a
        static per-model table for that: the list changes with each release, and
        being wrong in the conservative direction means silently dropping a
        parameter a model does honour.

        So the model is asked, and the answer is remembered. Returns the names
        that were newly learned, empty if the error was something else.
        """
        if type(exc).__name__ != "ValidationException":
            return set()
        message = str(exc)
        learned: set[str] = set()
        for raw in _DEPRECATED_PARAM_RE.findall(message):
            field = _SAMPLING_PARAM_ALIASES.get(raw.lower().replace("_", ""))
            field = field or _SAMPLING_PARAM_ALIASES.get(raw.lower())
            if field:
                learned.add(field)
        new = learned - self._unsupported_sampling
        if new:
            self._unsupported_sampling |= new
            logger.info(
                "Bedrock model %s rejects %s; dropping it for subsequent calls.",
                self._config.llm_model,
                ", ".join(sorted(new)),
            )
        return new

    @staticmethod
    def _extract_text(response: dict) -> str:
        """Join every text block in a Converse response, ignoring the rest.

        Indexing ``content[0]["text"]`` breaks on reasoning-capable models: they
        return a ``reasoningContent`` block *before* the answer, so element zero
        has no ``text`` key at all and the call dies with ``KeyError: 'text'``.
        Concatenating the text blocks is correct for every model — a
        single-block response is just the one-element case — and it also handles
        a response split across several text blocks.

        Reasoning content is deliberately dropped rather than returned: it is not
        the answer, and a caller parsing an importance score out of it would
        read the model's deliberation instead of its conclusion.
        """
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(
            block["text"] for block in blocks if isinstance(block.get("text"), str)
        )

    def _invoke_converse(self, messages: list[dict], **kwargs) -> str:
        payload = self._converse_kwargs(messages, **kwargs)
        try:
            response = self._client.converse(**payload)
        except Exception as exc:
            learned = self._note_unsupported_sampling(exc)
            if not learned or not self._strip_unsupported_sampling(payload, learned):
                raise
            response = self._client.converse(**payload)

        text = self._extract_text(response)
        if not text and response.get("stopReason") == "max_tokens":
            # A reasoning model can spend the entire token budget thinking and
            # return no text at all. Silence here would surface as an empty
            # summary or a default importance score, with nothing in the logs to
            # explain it.
            logger.warning(
                "Bedrock model %s hit maxTokens before emitting any text "
                "(reasoning consumed the budget); raise maxTokens.",
                self._config.llm_model,
            )
        return text
