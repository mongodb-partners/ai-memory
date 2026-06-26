"""Voyage endpoint + dimension handling (live-test fixes).

Issue #2: the MongoDB-issued Voyage key targets the Atlas embeddings gateway
(https://ai.mongodb.com/v1/embeddings), whose voyage-3 model emits 1024 dims,
not the 1536 default. Both the public Voyage endpoint and the MongoDB endpoint
must be selectable via env (VOYAGE_BASE_URL), and the dimension must follow the
model so create()'s guard does not reject a correctly-configured voyage setup.
"""

from agent_memory.config import MemoryConfig
from agent_memory.providers.manager import ProviderManager


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


class TestVoyageDimensionSync:
    """ProviderManager aligns embedding_dimension to the voyage model."""

    def test_voyage_3_sets_dimension_1024(self):
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-3",
                      voyage_api_key="k")
        ProviderManager(cfg)
        assert cfg.embedding_dimension == 1024

    def test_voyage_3_large_sets_dimension_1024(self):
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-3-large",
                      voyage_api_key="k")
        ProviderManager(cfg)
        assert cfg.embedding_dimension == 1024

    def test_explicit_dimension_is_respected(self):
        # An operator who pins a non-default voyage model dim keeps control.
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-custom",
                      voyage_api_key="k", embedding_dimension=2048)
        ProviderManager(cfg)
        assert cfg.embedding_dimension == 2048

    def test_embedding_model_synced_to_voyage_model(self):
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-3",
                      voyage_api_key="k")
        ProviderManager(cfg)
        assert cfg.embedding_model == "voyage-3"


class TestVoyageEndpointSelection:
    """Both the public and MongoDB Atlas endpoints are env-selectable."""

    def test_default_endpoint_is_public_voyage(self):
        cfg = _config()
        assert cfg.voyage_base_url == "https://api.voyageai.com/v1/embeddings"

    def test_mongodb_endpoint_via_env(self, monkeypatch):
        monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://h:27017")
        monkeypatch.setenv("VOYAGE_BASE_URL", "https://ai.mongodb.com/v1/embeddings")
        cfg = MemoryConfig.from_env(_env_file=None)
        assert cfg.voyage_base_url == "https://ai.mongodb.com/v1/embeddings"
