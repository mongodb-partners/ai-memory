"""Scorer selection and injection. REQ-E-162, REQ-E-164, REQ-E-171.

The class that matters most here is
`TestSelectionDoesNotDependOnConstructionOrder`. Selection reads the embedder's
model and dimension, and those two values used to be *installed* on the config by
`_create_embedding_provider` for Voyage, so a scorer constructed before it read
Titan's defaults, matched no artifact, and fell back to lexical. Nothing errored;
the scores just got worse. `select_artifact_name` now resolves them itself, and
those tests assert the resulting stronger property: the answer is the same
whether or not any provider has been built.
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


class TestSelectionDoesNotDependOnConstructionOrder:
    """Selection is correct whenever it runs — the stronger claim.

    This class used to assert the opposite shape: that the scorer was built *after*
    the embedding provider, because the Voyage arm of `_create_embedding_provider`
    rewrote `embedding_model` and `embedding_dimension` on the config and selection
    read those fields. Order was load-bearing, enforced by a comment, and getting
    it wrong silently downgraded a Voyage deployment to lexical scoring.

    `select_artifact_name` now resolves the model and dimension itself, so there is
    no order to get wrong. The tests assert that: selection sees Voyage's values
    whether or not any provider has been constructed.
    """

    def _resolved_triple(self, config) -> tuple:
        from agent_memory.providers.manager import resolve_embedding

        resolved = resolve_embedding(config)
        return (config.embedding_provider, resolved.model, resolved.dimension)

    def test_selection_sees_voyage_before_any_provider_is_built(self):
        """The case the old ordering rule existed to prevent, now simply correct.

        No `ProviderManager` at all: nothing has had a chance to rewrite anything,
        and selection still resolves voyage-4 at 1024 rather than Titan's defaults.
        """
        config = _config(
            embedding_provider="voyage",
            voyage_api_key="test-key",
            importance_scorer="local",
        )

        assert self._resolved_triple(config) == ("voyage", "voyage-3", 1024), (
            "selection read the config's declared fields — Titan's defaults on a "
            "Voyage deployment — so it would match nothing and silently downgrade "
            "to lexical once a Voyage artifact is bundled"
        )

    def test_the_answer_is_the_same_after_construction(self):
        """Constructing providers changes nothing, which is the point.

        With the write-back, before-vs-after differed and only "after" was right.
        """
        from agent_memory.providers.voyage import VoyageEmbeddingProvider

        config = _config(
            embedding_provider="voyage",
            voyage_api_key="test-key",
            importance_scorer="local",
        )
        before = self._resolved_triple(config)

        with patch.object(
            VoyageEmbeddingProvider, "__init__", lambda self, c: None
        ), patch.object(
            ProviderManager, "_create_llm_provider", lambda self, c: object()
        ):
            ProviderManager(config)

        assert self._resolved_triple(config) == before

    def test_a_registered_voyage_artifact_would_be_selected(self):
        """A stand-in registration, so the path from a Voyage config through to a
        *matched* artifact stays covered while we bundle no embedding head.

        Without this the parametrized "everything selects lexical" tests would be
        the only coverage, and they pass whether resolution works or not.
        """
        from agent_memory.providers import manager as manager_mod

        config = _config(
            embedding_provider="voyage",
            voyage_api_key="test-key",
            importance_scorer="local",
        )
        stand_in = {self._resolved_triple(config): "voyage-stand-in"}

        with patch.object(manager_mod, "_BUNDLED_ARTIFACTS", stand_in):
            assert select_artifact_name(config) == "voyage-stand-in"

    def test_the_scorer_still_builds_from_a_voyage_config(self):
        """The integration the old end-to-end test also covered: real
        `_create_embedding_provider`, only the network-touching constructor
        stubbed, and a `local` scorer that has to load an artifact."""
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

        assert isinstance(manager.scorer, ImportanceScorer)
        assert manager.embedding_spec.dimension == 1024
