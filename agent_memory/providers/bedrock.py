"""AWS Bedrock implementations of EmbeddingProvider and LLMProvider."""

import asyncio
import json
import re
from collections.abc import AsyncIterator

import boto3

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import EmbeddingProvider, LLMProvider


class BedrockEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings via Amazon Bedrock (Titan Embed Text)."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        kwargs: dict = {"service_name": "bedrock-runtime", "region_name": config.aws_region}
        if config.aws_access_key_id:
            kwargs["aws_access_key_id"] = config.aws_access_key_id
        if config.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = config.aws_secret_access_key
        self._client = boto3.client(**kwargs)

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
        kwargs: dict = {"service_name": "bedrock-runtime", "region_name": config.aws_region}
        if config.aws_access_key_id:
            kwargs["aws_access_key_id"] = config.aws_access_key_id
        if config.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = config.aws_secret_access_key
        self._client = boto3.client(**kwargs)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat request to the LLM."""
        return await asyncio.to_thread(self._invoke_converse, messages, **kwargs)

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
        messages = [
            {
                "role": "user",
                "content": [{"text": text}],
            }
        ]
        response = await self.chat(messages)
        # Extract numeric value, normalize 1-10 → 0.1-1.0
        match = re.search(r"\d+", response)
        if match:
            score = int(match.group())
            return max(0.1, min(1.0, score / 10.0))
        return 0.5  # Default on parse failure

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
        messages = [
            {
                "role": "user",
                "content": [{"text": text}],
            }
        ]
        return await self.chat(messages)

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

        def _pump() -> None:
            try:
                stream = self._client.converse_stream(
                    **self._converse_kwargs(messages, **kwargs)
                )["stream"]
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
            except BaseException as exc:  # noqa: BLE001 - re-raised on the loop
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
        """
        payload: dict = {"modelId": self._config.llm_model, "messages": messages}
        payload.update(kwargs)
        return payload

    def _invoke_converse(self, messages: list[dict], **kwargs) -> str:
        response = self._client.converse(**self._converse_kwargs(messages, **kwargs))
        return response["output"]["message"]["content"][0]["text"]
