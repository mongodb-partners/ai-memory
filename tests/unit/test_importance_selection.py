"""Scorer selection and injection. REQ-E-162, REQ-E-164, REQ-E-171.

The test that matters most here is `test_scorer_built_after_embedding_provider`.
`_create_embedding_provider` mutates the config for Voyage — overwriting
`embedding_model` and `embedding_dimension` — so a scorer constructed before it
would read Titan's defaults on a Voyage deployment, match no artifact, and fall
back to lexical. Nothing errors; the scores just get worse.
"""

import os
from unittest.mock import patch

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.exceptions import ConfigError
from agent_memory.providers.manager import ProviderManager, select_artifact_name
from agent_memory.services.importance import (
    ImportanceScorer,
    LLMScorer,
    LocalScorer,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.upper().startswith(("IMPORTANCE_", "EMBEDDING_", "VOYAGE_", "AWS_")):
            monkeypatch.delenv(key, raising=False)


def _config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


class TestSelectArtifactName:
    def test_bedrock_titan_1536(self):
        assert select_artifact_name(_config()) == "titan-1536"

    def test_voyage_3_1024(self):
        config = _config(
            embedding_provider="voyage",
            embedding_model="voyage-3",
            embedding_dimension=1024,
        )
        assert select_artifact_name(config) == "voyage-3-1024"

    def test_unknown_model_falls_back_to_lexical(self):
        config = _config(embedding_model="some-new-embedder")
        assert select_artifact_name(config) == "lexical"

    def test_right_model_wrong_dimension_falls_back_to_lexical(self):
        """A 512-dim voyage-3-lite must not load 1024 coefficients."""
        config = _config(
            embedding_provider="voyage",
            embedding_model="voyage-3-lite",
            embedding_dimension=512,
        )
        assert select_artifact_name(config) == "lexical"

    def test_titan_at_wrong_dimension_falls_back(self):
        config = _config(embedding_dimension=1024)
        assert select_artifact_name(config) == "lexical"

    def test_openai_falls_back_to_lexical(self):
        """No artifact shipped for OpenAI yet. Lexical is worse than a trained
        head and better than a constant."""
        config = _config(
            embedding_provider="openai", embedding_model="text-embedding-3-small"
        )
        assert select_artifact_name(config) == "lexical"


class TestProviderManagerSelection:
    """Embedding and LLM providers are stubbed — this is about the scorer."""

    @pytest.fixture(autouse=True)
    def _stub_providers(self):
        with patch.object(
            ProviderManager, "_create_embedding_provider", return_value=object()
        ) as emb, patch.object(
            ProviderManager, "_create_llm_provider", return_value=object()
        ):
            self.emb = emb
            yield

    def test_default_config_selects_llm_scorer(self):
        manager = ProviderManager(_config())
        assert isinstance(manager.scorer, LLMScorer)

    def test_llm_scorer_wraps_the_llm_provider(self):
        manager = ProviderManager(_config())
        assert manager.scorer._llm is manager.llm

    def test_local_config_selects_local_scorer(self):
        manager = ProviderManager(_config(importance_scorer="local"))
        assert isinstance(manager.scorer, LocalScorer)

    def test_local_scorer_loads_the_selected_bundled_artifact(self):
        manager = ProviderManager(_config(importance_scorer="local"))
        assert manager.scorer.artifact.model == "amazon.titan-embed-text-v1"

    def test_local_scorer_honors_explicit_path(self, tmp_path):
        import json

        from agent_memory.services.importance import (
            LEXICAL_FEATURE_COUNT,
            SCHEMA_VERSION,
        )

        path = tmp_path / "custom.json"
        path.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "kind": "lexical",
            "coefficients": [0.5] * LEXICAL_FEATURE_COUNT,
            "intercept": 0.1,
            "squash": "logistic",
            "training": {},
        }))
        manager = ProviderManager(
            _config(importance_scorer="local", importance_model_path=str(path))
        )
        assert manager.scorer.artifact.coefficients[0] == 0.5

    def test_missing_explicit_path_raises(self, tmp_path):
        """Refuse to start rather than silently fall back. The operator named a
        file; a typo'd path that quietly loads different coefficients is worse
        than a startup failure."""
        with pytest.raises(ConfigError, match="not found"):
            ProviderManager(_config(
                importance_scorer="local",
                importance_model_path=str(tmp_path / "absent.json"),
            ))

    @pytest.mark.parametrize("scorer_kind", ["llm", "local"])
    def test_scorer_satisfies_the_protocol(self, scorer_kind):
        manager = ProviderManager(_config(importance_scorer=scorer_kind))
        assert isinstance(manager.scorer, ImportanceScorer)


class TestConstructionOrder:
    def test_scorer_built_after_embedding_provider(self):
        """`_create_embedding_provider` rewrites `embedding_model` and
        `embedding_dimension` for Voyage. A scorer built first reads Titan's
        defaults and silently selects the lexical artifact."""
        from agent_memory.providers import manager as manager_mod

        real_select = manager_mod.select_artifact_name
        seen = {}

        def fake_embedding(self, config):
            config.embedding_model = "voyage-3"
            config.embedding_dimension = 1024
            return object()

        def record(config):
            seen["name"] = real_select(config)
            return seen["name"]

        with patch.object(
            ProviderManager, "_create_embedding_provider", fake_embedding
        ), patch.object(
            ProviderManager, "_create_llm_provider", lambda self, c: object()
        ), patch.object(manager_mod, "select_artifact_name", record):
            ProviderManager(_config(
                embedding_provider="voyage", importance_scorer="local"
            ))

        assert seen["name"] == "voyage-3-1024", (
            "scorer selection ran before the embedding provider rewrote the "
            "config — a Voyage deployment would silently get lexical scoring"
        )

    def test_voyage_end_to_end_selects_voyage_artifact(self):
        """The real integration, with only the network-touching provider stubbed.
        Config defaults are Titan's; only `_create_embedding_provider` knows
        otherwise."""
        from agent_memory.providers.voyage import VoyageEmbeddingProvider

        with patch.object(
            VoyageEmbeddingProvider, "__init__", lambda self, c: None
        ), patch.object(
            ProviderManager, "_create_llm_provider", lambda self, c: object()
        ):
            manager = ProviderManager(_config(
                embedding_provider="voyage",
                voyage_api_key="test-key",
                importance_scorer="local",
            ))
        assert manager.scorer.artifact.model == "voyage-3"
        assert manager.scorer.artifact.dimension == 1024
