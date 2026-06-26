"""Tests for EmbeddingProvider, LLMProvider, ProviderManager, and Bedrock/Voyage implementations."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import EmbeddingProvider, LLMProvider
from agent_memory.providers.bedrock import BedrockEmbeddingProvider, BedrockLLMProvider
from agent_memory.providers.manager import ProviderManager
from agent_memory.providers.voyage import VoyageEmbeddingProvider


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


class TestEmbeddingProviderInterface:
    """TC-007: EmbeddingProvider abstract interface."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()


class TestBedrockEmbeddingProvider:
    """TC-008: BedrockEmbeddingProvider generates embeddings."""

    def _make_provider(self, config=None):
        config = config or _make_config()
        with patch("agent_memory.providers.bedrock.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            provider = BedrockEmbeddingProvider(config)
            provider._client = mock_client
            return provider, mock_client

    async def test_generate_embedding_returns_vector(self):
        provider, mock_client = self._make_provider()
        fake_embedding = [0.1] * 1536
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=MagicMock(
                return_value=json.dumps({"embedding": fake_embedding}).encode()
            ))
        }

        result = await provider.generate_embedding("test text")
        assert len(result) == 1536
        assert result == fake_embedding

    async def test_generate_embeddings_batch(self):
        provider, mock_client = self._make_provider()
        fake_embedding = [0.1] * 1536
        mock_client.invoke_model.return_value = {
            "body": MagicMock(read=MagicMock(
                return_value=json.dumps({"embedding": fake_embedding}).encode()
            ))
        }

        results = await provider.generate_embeddings_batch(["text1", "text2"])
        assert len(results) == 2
        assert all(len(e) == 1536 for e in results)


class TestBedrockEmbeddingProviderError:
    """TC-009: BedrockEmbeddingProvider error handling."""

    async def test_generate_embedding_propagates_error(self):
        config = _make_config()
        with patch("agent_memory.providers.bedrock.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_client.invoke_model.side_effect = Exception("Bedrock unavailable")
            mock_boto3.client.return_value = mock_client
            provider = BedrockEmbeddingProvider(config)
            provider._client = mock_client

            with pytest.raises(Exception, match="Bedrock unavailable"):
                await provider.generate_embedding("test")


class TestLLMProviderInterface:
    """TC-010: LLMProvider abstract interface."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            LLMProvider()


class TestBedrockLLMProvider:
    """TC-011: BedrockLLMProvider importance and summary."""

    def _make_provider(self, config=None):
        config = config or _make_config()
        with patch("agent_memory.providers.bedrock.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            provider = BedrockLLMProvider(config)
            provider._client = mock_client
            return provider, mock_client

    async def test_assess_importance_returns_float(self):
        provider, mock_client = self._make_provider()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "7"}]}}
        }

        result = await provider.assess_importance("important content")
        assert 0.0 <= result <= 1.0

    async def test_generate_summary_returns_string(self):
        provider, mock_client = self._make_provider()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "This is a summary."}]}}
        }

        result = await provider.generate_summary("long content here")
        assert isinstance(result, str)
        assert len(result) > 0


class TestBedrockLLMProviderParseFailure:
    """assess_importance returns 0.5 default when LLM response has no number."""

    async def test_assess_importance_returns_default_on_no_number(self):
        config = _make_config()
        with patch("agent_memory.providers.bedrock.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_client.converse.return_value = {
                "output": {"message": {"content": [{"text": "I cannot rate this."}]}}
            }
            mock_boto3.client.return_value = mock_client
            provider = BedrockLLMProvider(config)
            provider._client = mock_client

            result = await provider.assess_importance("test")
            assert result == 0.5


class TestBedrockLLMProviderError:
    """TC-012: BedrockLLMProvider error handling."""

    async def test_assess_importance_returns_default_on_error(self):
        config = _make_config()
        with patch("agent_memory.providers.bedrock.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_client.converse.side_effect = Exception("LLM down")
            mock_boto3.client.return_value = mock_client
            provider = BedrockLLMProvider(config)
            provider._client = mock_client

            with pytest.raises(Exception, match="LLM down"):
                await provider.assess_importance("test")


class TestProviderManager:
    """TC-013: ProviderManager creates correct providers."""

    def test_creates_bedrock_providers(self):
        config = _make_config(embedding_provider="bedrock", llm_provider="bedrock")
        with patch("agent_memory.providers.bedrock.boto3"):
            manager = ProviderManager(config)
            assert isinstance(manager.embedding, BedrockEmbeddingProvider)
            assert isinstance(manager.llm, BedrockLLMProvider)


class TestProviderManagerUnknown:
    """TC-014: ProviderManager raises for unknown providers."""

    def test_unknown_embedding_provider_raises(self):
        config = _make_config(embedding_provider="unknown")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            ProviderManager(config)

    def test_unknown_llm_provider_raises(self):
        config = _make_config(llm_provider="unknown")
        with patch("agent_memory.providers.bedrock.boto3"):
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                ProviderManager(config)


# ─── Voyage Embedding Provider ──────────────────────────────


def _voyage_response(embeddings: list[list[float]], total_tokens: int = 100) -> dict:
    """Build a Voyage API response dict."""
    return {
        "data": [
            {"index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ],
        "model": "voyage-3",
        "usage": {"total_tokens": total_tokens},
    }


class TestVoyageEmbeddingProvider:
    """REQ-VP-001: VoyageEmbeddingProvider via async httpx."""

    def test_implements_embedding_provider(self):
        """Voyage provider is a proper EmbeddingProvider subclass."""
        config = _make_config(
            embedding_provider="voyage",
            voyage_api_key="test-key",
        )
        provider = VoyageEmbeddingProvider(config)
        assert isinstance(provider, EmbeddingProvider)

    async def test_generate_embedding_returns_vector(self):
        """Single embedding call returns correct-dimension vector."""
        config = _make_config(
            voyage_api_key="test-key",
            voyage_model="voyage-3",
            embedding_dimension=1024,
        )
        provider = VoyageEmbeddingProvider(config)
        fake_emb = [0.1] * 1024

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _voyage_response([fake_emb])
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate_embedding("test text")

        assert len(result) == 1024
        assert result == fake_emb

    async def test_generate_embeddings_batch(self):
        """Batch embedding returns one vector per input text."""
        config = _make_config(
            voyage_api_key="test-key",
            voyage_model="voyage-3",
            embedding_dimension=1024,
        )
        provider = VoyageEmbeddingProvider(config)
        fake_embs = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _voyage_response(fake_embs)
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response):
            results = await provider.generate_embeddings_batch(
                ["text1", "text2", "text3"]
            )

        assert len(results) == 3
        assert all(len(e) == 1024 for e in results)

    async def test_uses_document_input_type_for_batch(self):
        """Batch embedding uses input_type='document' by default."""
        config = _make_config(
            voyage_api_key="test-key",
            voyage_model="voyage-3",
            embedding_dimension=1024,
        )
        provider = VoyageEmbeddingProvider(config)
        fake_emb = [0.1] * 1024

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _voyage_response([fake_emb])
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)
        with patch.object(provider._client, "post", mock_post):
            await provider.generate_embeddings_batch(["text1"])

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["input_type"] == "document"

    async def test_uses_query_input_type_for_single(self):
        """Single embedding uses input_type='query'."""
        config = _make_config(
            voyage_api_key="test-key",
            voyage_model="voyage-3",
            embedding_dimension=1024,
        )
        provider = VoyageEmbeddingProvider(config)
        fake_emb = [0.1] * 1024

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _voyage_response([fake_emb])
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)
        with patch.object(provider._client, "post", mock_post):
            await provider.generate_embedding("search query")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["input_type"] == "query"

    async def test_sends_bearer_token(self):
        """API key is sent as Bearer token in Authorization header."""
        config = _make_config(
            voyage_api_key="my-secret-key",
            voyage_model="voyage-3",
            embedding_dimension=1024,
        )
        provider = VoyageEmbeddingProvider(config)
        fake_emb = [0.1] * 1024

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _voyage_response([fake_emb])
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)
        with patch.object(provider._client, "post", mock_post):
            await provider.generate_embedding("test")

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer my-secret-key"

    async def test_batching_splits_large_input(self):
        """Inputs exceeding 128 items are split into multiple API calls."""
        config = _make_config(
            voyage_api_key="test-key",
            voyage_model="voyage-3",
            embedding_dimension=1024,
        )
        provider = VoyageEmbeddingProvider(config)

        # 200 texts → should result in 2 API calls (128 + 72)
        texts = [f"text-{i}" for i in range(200)]

        def make_response(*args, **kwargs):
            payload = kwargs.get("json", {})
            n = len(payload.get("input", []))
            fake_embs = [[0.1] * 1024 for _ in range(n)]
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            resp.json.return_value = _voyage_response(fake_embs)
            resp.raise_for_status = MagicMock()
            return resp

        mock_post = AsyncMock(side_effect=make_response)
        with patch.object(provider._client, "post", mock_post):
            results = await provider.generate_embeddings_batch(texts)

        assert len(results) == 200
        assert mock_post.call_count == 2  # ceil(200/128) = 2

    async def test_http_error_propagates(self):
        """HTTP errors from the Voyage API propagate as exceptions."""
        config = _make_config(
            voyage_api_key="test-key",
            voyage_model="voyage-3",
            embedding_dimension=1024,
        )
        provider = VoyageEmbeddingProvider(config)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "rate limited", request=MagicMock(), response=mock_response,
        )

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                await provider.generate_embedding("test")


class TestVoyageConfig:
    """REQ-VP-002: Voyage config fields in MCPConfig."""

    def test_config_has_voyage_fields(self):
        config = _make_config(
            voyage_api_key="pa-test",
            voyage_base_url="https://custom.api/v1/embeddings",
            voyage_model="voyage-code-3",
        )
        assert config.voyage_api_key == "pa-test"
        assert config.voyage_base_url == "https://custom.api/v1/embeddings"
        assert config.voyage_model == "voyage-code-3"

    def test_voyage_defaults(self):
        config = _make_config()
        assert config.voyage_api_key is None
        assert config.voyage_base_url == "https://api.voyageai.com/v1/embeddings"
        assert config.voyage_model == "voyage-3"


class TestProviderManagerVoyage:
    """REQ-VP-003: ProviderManager selects Voyage when configured."""

    def test_creates_voyage_provider(self):
        config = _make_config(
            embedding_provider="voyage",
            voyage_api_key="test-key",
        )
        with patch("agent_memory.providers.bedrock.boto3"):
            manager = ProviderManager(config)
        assert isinstance(manager.embedding, VoyageEmbeddingProvider)
