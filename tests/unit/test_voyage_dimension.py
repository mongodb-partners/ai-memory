"""Voyage endpoint + dimension handling (live-test fixes).

Issue #2: the MongoDB-issued Voyage key targets the Atlas embeddings gateway
(https://ai.mongodb.com/v1/embeddings), whose voyage-3 model emits 1024 dims,
not the 1536 default. Both the public Voyage endpoint and the MongoDB endpoint
must be selectable via env (VOYAGE_BASE_URL), and the dimension must follow the
model so create()'s guard does not reject a correctly-configured voyage setup.

The alignment used to be performed by writing back onto the config object inside
``_create_embedding_provider``, so these tests asserted ``cfg.embedding_dimension
== 1024`` *after* constructing a ``ProviderManager``. They now assert the same
behaviour against ``resolve_embedding``, which returns the derived values instead
of installing them — see ``TestTheConfigIsNotRewritten`` for why that matters.
"""

from agent_memory.config import MemoryConfig
from agent_memory.providers.manager import (
    ProviderManager,
    resolve_embedding,
)


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


class TestVoyageDimensionSync:
    """The resolved dimension follows the voyage model."""

    def test_voyage_3_sets_dimension_1024(self):
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-3",
                      voyage_api_key="k")
        assert resolve_embedding(cfg).dimension == 1024

    def test_voyage_4_family_sets_dimension_1024(self):
        """voyage-4 is what the Atlas gateway serves today; all three variants
        are 1024, verified against the live gateway."""
        for model in ("voyage-4", "voyage-4-large", "voyage-4-lite"):
            cfg = _config(embedding_provider="voyage", voyage_model=model,
                          voyage_api_key="k")
            resolved = resolve_embedding(cfg)
            assert resolved.dimension == 1024, model
            assert resolved.model == model

    def test_voyage_3_large_sets_dimension_1024(self):
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-3-large",
                      voyage_api_key="k")
        assert resolve_embedding(cfg).dimension == 1024

    def test_explicit_dimension_is_respected(self):
        # An operator who pins a non-default voyage model dim keeps control.
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-custom",
                      voyage_api_key="k", embedding_dimension=2048)
        assert resolve_embedding(cfg).dimension == 2048

    def test_a_pinned_default_value_is_still_a_pin(self):
        """The case the old ``== _DEFAULT_EMBEDDING_DIMENSION`` test could not see.

        Pinning was detected by comparing the declared dimension against 1536, so
        an operator who set ``EMBEDDING_DIMENSION=1536`` deliberately on a Voyage
        deployment was indistinguishable from one who left it alone — and had it
        silently rewritten to 1024. That is the operator most likely to have an
        existing 1536-dim index with vectors already in it, and the rewrite would
        re-provision the index out from under them.

        Pinning is now read from ``model_fields_set``, which records what was
        actually supplied rather than what it happens to equal.
        """
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-4",
                      voyage_api_key="k", embedding_dimension=1536)
        assert resolve_embedding(cfg).dimension == 1536

    def test_an_unpinned_default_still_follows_the_model(self):
        """The paired case: nothing supplied, so the model's dimension wins.

        Stated alongside the test above because the two differ only in whether the
        value was supplied, and a fix that read "pinned" as "always keep the
        declared value" would pass one and fail the other.
        """
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-4",
                      voyage_api_key="k")
        assert cfg.embedding_dimension == 1536, "precondition: the declared default"
        assert resolve_embedding(cfg).dimension == 1024

    def test_embedding_model_synced_to_voyage_model(self):
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-3",
                      voyage_api_key="k")
        assert resolve_embedding(cfg).model == "voyage-3"


class TestTheConfigIsNotRewritten:
    """Constructing providers must not edit the config it was handed.

    ``_create_embedding_provider`` used to assign ``config.embedding_model`` and
    ``config.embedding_dimension`` in place. Three consequences, none of which
    raised:

    * The caller's own object changed under them. A library caller that built one
      config and passed it to two facades, or inspected it afterwards to log what
      it had configured, saw values it never set.
    * Every downstream reader of those fields became order-dependent on the
      factory having already run. ``select_artifact_name`` carried a comment
      saying so, and the startup guard and index provisioning depended on it
      silently — a correct-looking read of ``config.embedding_dimension`` before
      that point yields Titan's 1536 on a Voyage deployment, provisions a
      1536-dim index, and writes 1024-dim vectors that ``$vectorSearch`` accepts
      and never returns.
    * Construction stopped being idempotent in a way that mattered for the pin
      check: the first pass overwrote 1536 with 1024, so a second pass saw a
      "non-default" value and treated it as pinned.
    """

    def test_constructing_providers_leaves_the_config_alone(self):
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-4",
                      voyage_api_key="k")
        before = cfg.model_dump()

        ProviderManager(cfg)

        assert cfg.model_dump() == before, (
            "ProviderManager edited the config it was given; the fields it used "
            "to rewrite are derived values and belong to `resolve_embedding`"
        )

    def test_the_manager_publishes_the_resolved_spec(self):
        """Callers need somewhere to read the dimension actually in force."""
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-4",
                      voyage_api_key="k")

        manager = ProviderManager(cfg)

        assert manager.embedding_spec.dimension == 1024
        assert manager.embedding_spec.model == "voyage-4"

    def test_resolution_is_idempotent(self):
        """Twice must give the same answer as once.

        With the write-back, the second call read a config the first had already
        modified — and a 1024 it had installed itself now looked like an operator's
        deliberate pin.
        """
        cfg = _config(embedding_provider="voyage", voyage_model="voyage-4",
                      voyage_api_key="k")

        assert resolve_embedding(cfg) == resolve_embedding(cfg)
        ProviderManager(cfg)
        assert resolve_embedding(cfg).dimension == 1024


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
