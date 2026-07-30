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
    """We ship no trained embedding head, so `_BUNDLED_ARTIFACTS` is empty and
    every config resolves to `lexical`. That is the intended answer rather than a
    lookup miss — a lexical model trained on real labels beats the constant an
    untrained embedding head returns."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {},  # bedrock/titan defaults
            {"embedding_provider": "voyage", "embedding_model": "voyage-4",
             "embedding_dimension": 1024},
            {"embedding_provider": "voyage", "embedding_model": "voyage-3",
             "embedding_dimension": 1024},
            {"embedding_provider": "openai",
             "embedding_model": "text-embedding-3-small"},
            {"embedding_model": "some-new-embedder"},
        ],
        ids=["titan", "voyage-4", "voyage-3", "openai", "unknown"],
    )
    def test_every_embedder_selects_lexical(self, overrides):
        assert select_artifact_name(_config(**overrides)) == "lexical"

    def test_no_embedding_head_is_bundled(self):
        """Pins the reason the parametrize above is uniform, so a future entry
        makes this fail and forces those expectations to be revisited rather than
        passing on a stale assumption."""
        from agent_memory.providers.manager import _BUNDLED_ARTIFACTS

        assert _BUNDLED_ARTIFACTS == {}

    def test_dimension_is_part_of_the_key(self):
        """The mismatch guard, kept live against a stand-in map: voyage-3 is 1024
        and voyage-3-lite is 512, and loading 1024 coefficients against a
        512-vector is what selection exists to prevent. Patched rather than
        deleted so the lookup keeps being exercised while the real map is empty."""
        from agent_memory.providers import manager as manager_mod

        stand_in = {("voyage", "voyage-3", 1024): "voyage-3-1024"}
        with patch.object(manager_mod, "_BUNDLED_ARTIFACTS", stand_in):
            matched = _config(
                embedding_provider="voyage",
                embedding_model="voyage-3",
                embedding_dimension=1024,
            )
            assert select_artifact_name(matched) == "voyage-3-1024"

            wrong_dimension = _config(
                embedding_provider="voyage",
                embedding_model="voyage-3",
                embedding_dimension=512,
            )
            assert select_artifact_name(wrong_dimension) == "lexical"


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
        """`lexical` for every embedder, and it must be the *trained* file — an
        all-zero artifact would load fine and score every memory alike."""
        manager = ProviderManager(_config(importance_scorer="local"))
        artifact = manager.scorer.artifact
        assert artifact.kind == "lexical"
        assert any(c != 0.0 for c in artifact.coefficients)

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
    """The ordering `select_artifact_name` depends on, asserted without depending
    on `_BUNDLED_ARTIFACTS` having entries.

    These used to assert the selected *name* (`"voyage-3-1024"`), which only fails
    on a wrong order while some triple is registered. With the map empty every
    order returns `"lexical"` and that assertion would pass vacuously — so they now
    check the config state selection actually observed, which is the property that
    matters and is what breaks the moment a head is added back.
    """

    def _observed_config_at_selection(self, config) -> dict:
        """Run construction, capturing the (provider, model, dimension) triple
        `select_artifact_name` saw."""
        from agent_memory.providers import manager as manager_mod

        real_select = manager_mod.select_artifact_name
        seen: dict = {}

        def record(cfg):
            seen["triple"] = (
                cfg.embedding_provider,
                cfg.embedding_model,
                cfg.embedding_dimension,
            )
            seen["name"] = real_select(cfg)
            return seen["name"]

        with patch.object(manager_mod, "select_artifact_name", record):
            ProviderManager(config)
        return seen

    def test_scorer_built_after_embedding_provider(self):
        """`_create_embedding_provider` rewrites `embedding_model` and
        `embedding_dimension` for Voyage. A scorer built first reads Titan's
        defaults — so if any Voyage triple is ever registered, it would match
        nothing and silently downgrade to lexical."""
        def fake_embedding(self, config):
            config.embedding_model = "voyage-4"
            config.embedding_dimension = 1024
            return object()

        with patch.object(
            ProviderManager, "_create_embedding_provider", fake_embedding
        ), patch.object(
            ProviderManager, "_create_llm_provider", lambda self, c: object()
        ):
            seen = self._observed_config_at_selection(
                _config(embedding_provider="voyage", importance_scorer="local")
            )

        assert seen["triple"] == ("voyage", "voyage-4", 1024), (
            "scorer selection ran before the embedding provider rewrote the "
            "config — it saw Titan's defaults, so a Voyage deployment would "
            "silently get lexical scoring once a Voyage artifact is bundled"
        )

    def test_voyage_end_to_end_reaches_selection_with_voyage_config(self):
        """The real integration, with only the network-touching provider stubbed.
        Config defaults are Titan's; only `_create_embedding_provider` knows
        otherwise."""
        from agent_memory.providers.voyage import VoyageEmbeddingProvider

        with patch.object(
            VoyageEmbeddingProvider, "__init__", lambda self, c: None
        ), patch.object(
            ProviderManager, "_create_llm_provider", lambda self, c: object()
        ):
            seen = self._observed_config_at_selection(_config(
                embedding_provider="voyage",
                voyage_api_key="test-key",
                importance_scorer="local",
            ))

        provider, _, dimension = seen["triple"]
        assert (provider, dimension) == ("voyage", 1024)
        assert seen["name"] == "lexical"

    def test_a_registered_voyage_artifact_would_be_selected(self):
        """The end-to-end path with a stand-in registration, so the wiring from
        `_create_embedding_provider`'s config rewrite through to a *matched*
        artifact stays covered while we bundle no embedding head."""
        from agent_memory.providers import manager as manager_mod
        from agent_memory.providers.voyage import VoyageEmbeddingProvider

        with patch.object(
            VoyageEmbeddingProvider, "__init__", lambda self, c: None
        ), patch.object(
            ProviderManager, "_create_llm_provider", lambda self, c: object()
        ):
            config = _config(
                embedding_provider="voyage",
                voyage_api_key="test-key",
                importance_scorer="local",
            )
            seen = self._observed_config_at_selection(config)
            stand_in = {seen["triple"]: "voyage-stand-in"}
            with patch.object(manager_mod, "_BUNDLED_ARTIFACTS", stand_in):
                assert select_artifact_name(config) == "voyage-stand-in"
