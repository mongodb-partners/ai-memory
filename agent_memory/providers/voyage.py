"""Voyage AI embedding provider using async httpx."""

import logging

import httpx

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_VOYAGE_BATCH_LIMIT = 128


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings via the Voyage AI REST API.

    Uses async httpx with Bearer token authentication.
    Automatically splits large batches into chunks of 128.
    """

    def __init__(self, config: MCPConfig) -> None:
        self._api_key = config.voyage_api_key or ""
        self._base_url = config.voyage_base_url
        self._model = config.voyage_model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a single embedding vector (input_type='query')."""
        embeddings = await self._embed_batch([text], input_type="query")
        return embeddings[0]

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (input_type='document').

        Automatically splits into chunks of 128 per API call.
        """
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), _VOYAGE_BATCH_LIMIT):
            batch = texts[start : start + _VOYAGE_BATCH_LIMIT]
            batch_embeddings = await self._embed_batch(batch, input_type="document")
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    async def _embed_batch(
        self, inputs: list[str], input_type: str = "document"
    ) -> list[list[float]]:
        """Call the Voyage API for a single batch."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": inputs,
            "input_type": input_type,
        }

        response = await self._client.post(
            self._base_url, headers=headers, json=payload,
        )
        response.raise_for_status()

        data = response.json()
        items = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]
