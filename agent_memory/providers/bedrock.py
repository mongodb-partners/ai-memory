"""AWS Bedrock implementations of EmbeddingProvider and LLMProvider."""

import asyncio
import json
import re

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

    def _invoke_converse(self, messages: list[dict], **kwargs) -> str:
        response = self._client.converse(
            modelId=self._config.llm_model,
            messages=messages,
        )
        return response["output"]["message"]["content"][0]["text"]
