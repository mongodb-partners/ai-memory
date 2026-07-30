"""Artifact loading contract for local importance scoring. REQ-E-163, REQ-E-171.

These tests exist because a bad artifact is a silent failure mode. A file with
1024 coefficients loaded against a 1536-dim embedder would score the overlapping
prefix and return a plausible number, and the only symptom would be memories
being forgotten or promoted wrongly weeks later — the same class of invisible
defect recorded in `test_importance_parsing.py`.
"""

import json
import pathlib

import pytest

from agent_memory.exceptions import ConfigError
from agent_memory.services.importance import (
    LEXICAL_FEATURE_COUNT,
    SCHEMA_VERSION,
    artifact_dir,
    bundled_artifact_path,
    load_artifact,
)


def _valid_embedding_artifact(**overrides) -> dict:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "kind": "embedding_linear",
        "embedding": {"provider": "bedrock", "model": "test-model", "dimension": 3},
        "coefficients": [0.1, 0.2, 0.3],
        "intercept": 0.4,
        "squash": "logistic",
        "training": {"labels": ["synthetic"], "n_samples": 0},
    }
    doc.update(overrides)
    return doc


def _write(tmp_path: pathlib.Path, doc: dict, name: str = "a.json") -> pathlib.Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


class TestValidArtifacts:
    def test_loads_embedding_artifact(self, tmp_path):
        art = load_artifact(_write(tmp_path, _valid_embedding_artifact()))
        assert art.kind == "embedding_linear"
        assert art.coefficients == (0.1, 0.2, 0.3)
        assert art.intercept == 0.4
        assert art.provider == "bedrock"
        assert art.model == "test-model"
        assert art.dimension == 3

    def test_loads_lexical_artifact(self, tmp_path):
        doc = {
            "schema_version": SCHEMA_VERSION,
            "kind": "lexical",
            "coefficients": [0.0] * LEXICAL_FEATURE_COUNT,
            "intercept": 0.5,
            "squash": "logistic",
            "training": {},
        }
        art = load_artifact(_write(tmp_path, doc))
        assert art.kind == "lexical"
        assert art.dimension is None
        assert len(art.coefficients) == LEXICAL_FEATURE_COUNT

    def test_coefficients_are_immutable(self, tmp_path):
        """A shared Artifact must not be mutable by one scorer."""
        art = load_artifact(_write(tmp_path, _valid_embedding_artifact()))
        assert isinstance(art.coefficients, tuple)

    def test_missing_training_block_normalizes_to_empty_dict(self, tmp_path):
        """Callers read `artifact.training.get(...)`, so None would be a
        TypeError at startup on a hand-written artifact."""
        doc = _valid_embedding_artifact()
        del doc["training"]
        assert load_artifact(_write(tmp_path, doc)).training == {}


class TestRejections:
    """Every rejection must name the file and the specific problem.

    "Failed to load model" sends an operator to read source. "coefficient count
    2 does not match declared dimension 3" sends them to the file.
    """

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_artifact(tmp_path / "nope.json")

    def test_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        # Escaped: `match=` is a regex, so an unescaped `.` would also accept
        # "badXjson" — a weaker assertion than the one this test means to make.
        with pytest.raises(ConfigError, match=r"bad\.json"):
            load_artifact(path)

    def test_unknown_schema_version(self, tmp_path):
        path = _write(tmp_path, _valid_embedding_artifact(schema_version=99))
        with pytest.raises(ConfigError, match="schema_version"):
            load_artifact(path)

    def test_unknown_kind(self, tmp_path):
        path = _write(tmp_path, _valid_embedding_artifact(kind="neural_net"))
        with pytest.raises(ConfigError, match="kind"):
            load_artifact(path)

    def test_coefficient_count_mismatch(self, tmp_path):
        doc = _valid_embedding_artifact(coefficients=[0.1, 0.2])
        path = _write(tmp_path, doc)
        with pytest.raises(ConfigError, match="does not match"):
            load_artifact(path)

    def test_lexical_wrong_feature_count(self, tmp_path):
        doc = {
            "schema_version": SCHEMA_VERSION,
            "kind": "lexical",
            "coefficients": [0.0] * (LEXICAL_FEATURE_COUNT - 1),
            "intercept": 0.5,
            "training": {},
        }
        with pytest.raises(ConfigError, match="does not match"):
            load_artifact(_write(tmp_path, doc))

    def test_non_numeric_coefficient(self, tmp_path):
        doc = _valid_embedding_artifact(coefficients=[0.1, "oops", 0.3])
        with pytest.raises(ConfigError, match="numeric"):
            load_artifact(_write(tmp_path, doc))

    def test_boolean_coefficient_is_not_numeric(self, tmp_path):
        """`bool` is an `int` subclass, so a naive isinstance check accepts True
        and then silently treats it as 1.0."""
        doc = _valid_embedding_artifact(coefficients=[0.1, True, 0.3])
        with pytest.raises(ConfigError, match="numeric"):
            load_artifact(_write(tmp_path, doc))

    def test_embedding_artifact_without_embedding_block(self, tmp_path):
        doc = _valid_embedding_artifact()
        del doc["embedding"]
        with pytest.raises(ConfigError, match="embedding"):
            load_artifact(_write(tmp_path, doc))

    def test_unknown_squash(self, tmp_path):
        doc = _valid_embedding_artifact(squash="softmax")
        with pytest.raises(ConfigError, match="squash"):
            load_artifact(_write(tmp_path, doc))

    def test_empty_coefficients(self, tmp_path):
        doc = _valid_embedding_artifact(coefficients=[])
        with pytest.raises(ConfigError, match="coefficients"):
            load_artifact(_write(tmp_path, doc))

    def test_json_array_is_not_an_artifact(self, tmp_path):
        path = tmp_path / "arr.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ConfigError, match="JSON object"):
            load_artifact(path)


class TestBundledArtifacts:
    """Cheap integrity checks on the files we ship. Catches a hand-edited file."""

    def test_artifact_dir_exists(self):
        assert artifact_dir().is_dir()

    def _bundled_names(self) -> list[str]:
        """Discovered rather than listed, so adding an artifact arms these checks
        on it without anyone remembering to extend a parametrize list."""
        return sorted(p.stem for p in artifact_dir().glob("*.json"))

    def test_every_bundled_artifact_loads(self):
        names = self._bundled_names()
        assert names, "no bundled artifacts found"
        for name in names:
            art = load_artifact(bundled_artifact_path(name))
            assert art.kind in ("embedding_linear", "lexical"), name

    def test_embedding_artifacts_declare_a_matching_dimension(self):
        """Vacuous while we ship no embedding head — and that is the point of
        discovering names instead of listing them. Commit one whose declared
        dimension disagrees with its coefficient count and this fails."""
        for name in self._bundled_names():
            art = load_artifact(bundled_artifact_path(name))
            if art.kind == "embedding_linear":
                assert art.dimension == len(art.coefficients), name
                assert art.provider, name
                assert art.model, name

    def test_no_bundled_artifact_is_a_constant(self):
        """All-zero coefficients score every memory the intercept, which reads as
        working — no error, a plausible number — while pinning every memory below
        the 0.6 promotion threshold or above it. Two shipped placeholders had
        exactly this shape; they were deleted rather than trained, and this keeps
        an untrained stand-in from being committed as if it scored anything."""
        for name in self._bundled_names():
            art = load_artifact(bundled_artifact_path(name))
            assert any(c != 0.0 for c in art.coefficients), (
                f"{name} has all-zero coefficients: it ignores its input and "
                "returns a constant for every memory"
            )

    def test_bundled_lexical_has_seven_features(self):
        art = load_artifact(bundled_artifact_path("lexical"))
        assert len(art.coefficients) == LEXICAL_FEATURE_COUNT

    def test_lexical_records_what_it_was_trained_on(self):
        """`lexical` is the artifact every deployment now gets, so an operator has
        to be able to read its provenance off the file."""
        art = load_artifact(bundled_artifact_path("lexical"))
        assert art.training.get("labels")
        assert art.training.get("metrics", {}).get("forget_agreement") is not None
