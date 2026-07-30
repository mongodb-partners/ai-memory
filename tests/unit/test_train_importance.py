"""Pure-function contracts in the offline trainer.
REQ-E-167, REQ-E-168, REQ-E-169, REQ-E-170.

The trainer talks to HuggingFace, an LLM, and MongoDB — none of which belongs in a
unit test, and mocking all three would only assert the mocks. These tests cover the
four functions where a quiet mistake would produce a trained-looking model that
silently deletes memories:

- label derivation must not treat *unlabeled* as *negative*
- the composite metric must rank calibration above correlation
- the emitted artifact must load through the real runtime loader

`scripts/` is not a package, so the module is loaded by path.
"""

import importlib.util
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train_importance.py"

pytest.importorskip("sklearn", reason="trainer tests need the `training` extra")


def _load_trainer():
    spec = importlib.util.spec_from_file_location("train_importance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


class TestBenchmarkLabelDerivation:
    """REQ-E-167. A turn cited by a later question is positive. A turn in a session
    no question draws on is negative. Everything else is *unlabeled* and must be
    dropped, not defaulted."""

    SESSIONS = [
        {
            "session_id": "s1",
            "turns": [
                {"turn_id": "t1", "content": "I'm allergic to penicillin."},
                {"turn_id": "t2", "content": "Nice weather."},
            ],
        },
        {
            "session_id": "s2",
            "turns": [{"turn_id": "t3", "content": "Anything at all."}],
        },
    ]
    QUESTIONS = [{"evidence_turn_ids": ["t1"], "evidence_session_ids": ["s1"]}]

    def test_cited_turn_is_positive(self, trainer):
        labels = dict(trainer.derive_benchmark_labels(self.SESSIONS, self.QUESTIONS))
        assert labels["I'm allergic to penicillin."] == 1.0

    def test_turn_in_an_uncited_session_is_negative(self, trainer):
        labels = dict(trainer.derive_benchmark_labels(self.SESSIONS, self.QUESTIONS))
        assert labels["Anything at all."] == 0.0

    def test_uncited_turn_in_a_cited_session_is_dropped(self, trainer):
        """The load-bearing assertion. 'Nice weather' sits in a session a question
        drew on, which that question did not cite, so we cannot tell whether it was
        useless or merely unasked-about. Labeling it 0 is how a trainer learns to
        forget everything while every aggregate metric still looks plausible."""
        labels = dict(trainer.derive_benchmark_labels(self.SESSIONS, self.QUESTIONS))
        assert "Nice weather." not in labels

    def test_no_questions_yields_no_labels(self, trainer):
        """With nothing cited, every session is 'uncited' — which would label the
        entire corpus negative. Refuse instead."""
        assert trainer.derive_benchmark_labels(self.SESSIONS, []) == []


class TestMongoLabelDerivation:
    """REQ-E-170. Labels from signals the documents already carry."""

    NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)

    def _doc(self, **kw):
        doc = {
            "created_at": self.NOW - timedelta(days=60),
            "access_count": 0,
            "memory_type": "long_term",
            "is_deleted": False,
        }
        doc.update(kw)
        return doc

    def test_frequently_accessed_is_high(self, trainer):
        label = trainer.label_from_mongo_document(
            self._doc(access_count=25), now=self.NOW
        )
        assert label is not None and label > 0.7

    def test_soft_deleted_by_consolidation_is_low(self, trainer):
        label = trainer.label_from_mongo_document(
            self._doc(is_deleted=True, deleted_reason="low_importance"), now=self.NOW
        )
        assert label is not None and label < 0.3

    def test_old_and_never_accessed_is_low(self, trainer):
        label = trainer.label_from_mongo_document(
            self._doc(access_count=0, created_at=self.NOW - timedelta(days=180)),
            now=self.NOW,
        )
        assert label is not None and label < 0.4

    def test_recent_and_unaccessed_is_unlabeled(self, trainer):
        """The one that matters. A memory created an hour ago with zero accesses
        has not had the chance to be useful. Scoring it 0 teaches the model that
        everything new is worthless — and new is when scoring happens."""
        label = trainer.label_from_mongo_document(
            self._doc(created_at=self.NOW - timedelta(hours=1)), now=self.NOW
        )
        assert label is None

    def test_labels_are_in_range(self, trainer):
        for doc in [
            self._doc(access_count=1000),
            self._doc(access_count=0, created_at=self.NOW - timedelta(days=900)),
            self._doc(is_deleted=True, deleted_reason="low_importance"),
        ]:
            label = trainer.label_from_mongo_document(doc, now=self.NOW)
            assert label is None or 0.0 <= label <= 1.0


class TestCompositeScore:
    """REQ-E-169. Calibration outranks ranking, because consolidation compares
    against absolute thresholds rather than sorting."""

    def test_calibrated_beats_better_correlated_but_offset(self, trainer):
        """A model with Spearman 0.85 and a +0.15 mean offset promotes nearly
        everything: the promotion threshold is 0.6, and shifting the whole
        distribution up by 0.15 moves a large slice of the store across it."""
        offset = {
            "spearman": 0.85, "mae": 0.20, "mean_offset": 0.15,
            "forget_agreement": 0.55, "promote_agreement": 0.50,
        }
        calibrated = {
            "spearman": 0.70, "mae": 0.08, "mean_offset": 0.01,
            "forget_agreement": 0.92, "promote_agreement": 0.90,
        }
        assert trainer.composite_score(calibrated) > trainer.composite_score(offset)

    def test_mean_offset_sign_does_not_matter(self, trainer):
        """A -0.15 offset forgets too much; +0.15 promotes too much. Both are
        equally wrong, so the metric must use the magnitude."""
        base = {"spearman": 0.8, "mae": 0.1, "forget_agreement": 0.8,
                "promote_agreement": 0.8}
        up = trainer.composite_score({**base, "mean_offset": 0.15})
        down = trainer.composite_score({**base, "mean_offset": -0.15})
        assert up == pytest.approx(down)

    def test_perfect_model_scores_highest(self, trainer):
        perfect = {"spearman": 1.0, "mae": 0.0, "mean_offset": 0.0,
                   "forget_agreement": 1.0, "promote_agreement": 1.0}
        worst = {"spearman": -1.0, "mae": 1.0, "mean_offset": 1.0,
                 "forget_agreement": 0.0, "promote_agreement": 0.0}
        assert trainer.composite_score(perfect) > trainer.composite_score(worst)


class TestEvaluate:
    def test_reports_the_operational_columns(self, trainer):
        y_true = [0.05, 0.3, 0.7, 0.95]
        metrics = trainer.evaluate(y_true, list(y_true))
        for key in ("spearman", "mae", "mean_offset", "forget_agreement",
                    "promote_agreement", "mean_pred", "mean_label"):
            assert key in metrics, key

    def test_identical_predictions_agree_completely(self, trainer):
        y_true = [0.05, 0.3, 0.7, 0.95]
        metrics = trainer.evaluate(y_true, list(y_true))
        assert metrics["forget_agreement"] == 1.0
        assert metrics["promote_agreement"] == 1.0
        assert metrics["mae"] == pytest.approx(0.0)

    def test_forget_agreement_is_measured_at_the_real_threshold(self, trainer):
        """0.1 is `forgetting_score_threshold`. A prediction of 0.11 against a
        label of 0.05 disagrees about deleting the memory, which is the decision
        the number exists to inform."""
        metrics = trainer.evaluate([0.05], [0.11])
        assert metrics["forget_agreement"] == 0.0


class TestArtifactRoundTrip:
    """The trainer must not be able to emit something the runtime rejects. Uses the
    real loader rather than re-checking the shape by hand."""

    def test_emitted_lexical_artifact_loads(self, trainer, tmp_path):
        import json

        from agent_memory.services.importance import (
            LEXICAL_FEATURE_COUNT,
            load_artifact,
        )

        doc = trainer.build_artifact(
            "lexical",
            [0.1] * LEXICAL_FEATURE_COUNT,
            0.2,
            training={"labels": ["test"], "n_samples": 1},
        )
        path = tmp_path / "out.json"
        path.write_text(json.dumps(doc))
        artifact = load_artifact(path)
        assert artifact.kind == "lexical"
        assert artifact.intercept == pytest.approx(0.2)

    def test_emitted_embedding_artifact_loads(self, trainer, tmp_path):
        import json

        from agent_memory.services.importance import load_artifact

        doc = trainer.build_artifact(
            "embedding_linear",
            [0.1, 0.2, 0.3],
            0.0,
            embedding={"provider": "bedrock", "model": "m", "dimension": 3},
            training={"labels": ["test"], "n_samples": 1},
        )
        path = tmp_path / "out.json"
        path.write_text(json.dumps(doc))
        assert load_artifact(path).dimension == 3

    def test_artifact_declares_the_current_schema_version(self, trainer):
        from agent_memory.services.importance import SCHEMA_VERSION

        doc = trainer.build_artifact("lexical", [0.0] * 7, 0.0, training={})
        assert doc["schema_version"] == SCHEMA_VERSION

    def test_training_block_records_the_metrics(self, trainer):
        """An operator has to be able to read calibration off the artifact before
        switching a production deployment onto it."""
        doc = trainer.build_artifact(
            "lexical", [0.0] * 7, 0.0,
            training={"metrics": {"forget_agreement": 0.9}, "n_samples": 10},
        )
        assert doc["training"]["metrics"]["forget_agreement"] == 0.9


class TestDiscriminationGate:
    """The check that catches what `composite_score` cannot: a model that is
    beautifully calibrated and separates nothing."""

    def test_a_constant_model_has_no_margin(self, trainer):
        """All-zero coefficients score every input identically, so the worst durable
        case cannot beat the best expiring one. This is the shipped placeholder, and
        it must not pass."""
        margin = trainer.discrimination_margin([0.0] * 7, 0.5)
        assert margin == pytest.approx(0.0)

    def test_a_temporal_positive_model_fails(self, trainer):
        """The real defect, pinned. These are the coefficients a nonzero
        --benchmark-weight produces: `temporal` +2.14 from the benchmarks' cited
        'yesterday'. It promotes 'busy today and tomorrow' over 'our policy is'."""
        coefficients = [-2.02, -2.11, -0.11, 1.34, 2.14, -1.55, 3.01]
        assert trainer.discrimination_margin(coefficients, -0.82) < 0

    def test_margin_compares_worst_durable_to_best_expiring(self, trainer):
        """Not mean-vs-mean. A model can have both group means ordered correctly
        while an individual pair is inverted, and an inverted pair is a memory
        consolidation deletes for the wrong reason."""
        from agent_memory.services.importance import lexical_features

        coefficients = [0.0, 0.0, 3.0, 1.0, -3.0, -2.0, 0.0]
        intercept = -0.5
        margin = trainer.discrimination_margin(coefficients, intercept)
        scores = trainer.predict(
            coefficients,
            intercept,
            [lexical_features(t) for t, _ in trainer.DISCRIMINATION_CASES],
        )
        durable = [
            s for s, (_, k) in zip(scores, trainer.DISCRIMINATION_CASES) if k
        ]
        expiring = [
            s for s, (_, k) in zip(scores, trainer.DISCRIMINATION_CASES) if not k
        ]
        assert margin == pytest.approx(min(durable) - max(expiring))

    def test_cases_are_held_out_from_training(self, trainer):
        """If a gate case were in SYNTHETIC_CONTENT, the model would be fitted on
        the thing that judges it and the gate would report on memorization."""
        synthetic = set(trainer.SYNTHETIC_CONTENT)
        overlap = [t for t, _ in trainer.DISCRIMINATION_CASES if t in synthetic]
        assert overlap == []

    def test_both_classes_are_represented(self, trainer):
        """A gate with no expiring cases, or no durable ones, has an undefined
        margin and would raise rather than fail informatively."""
        keeps = [k for _, k in trainer.DISCRIMINATION_CASES]
        assert any(keeps) and not all(keeps)


class TestServableLabelDispatch:
    """REQ-E-168. The rescale is correct for benchmark indicators and wrong for LLM
    scores that already sit in the servable range."""

    def test_llm_labels_pass_through_unchanged(self, trainer):
        """An affine rescale of an already-servable label shifts it upward — 0.3
        becomes 0.37 — biasing the fitted model toward promotion against exactly the
        threshold that matters."""
        import numpy as np

        out = trainer._servable_labels(np.array([0.2, 0.3, 0.6, 1.0]), ["llm"] * 4)
        assert list(out) == pytest.approx([0.2, 0.3, 0.6, 1.0])

    def test_benchmark_labels_are_rescaled(self, trainer):
        import numpy as np

        out = trainer._servable_labels(np.array([0.0, 1.0]), ["benchmark"] * 2)
        assert list(out) == pytest.approx([0.1, 1.0])

    def test_mixed_stages_are_dispatched_per_row(self, trainer):
        import numpy as np

        out = trainer._servable_labels(
            np.array([0.0, 0.3]), ["benchmark", "llm"]
        )
        assert list(out) == pytest.approx([0.1, 0.3])

    def test_output_is_always_servable(self, trainer):
        import numpy as np

        for sources in (["llm"], ["benchmark"], ["mongodb"]):
            out = trainer._servable_labels(np.array([-0.5, 0.0, 0.5, 2.0]), sources * 4)
            assert out.min() >= trainer.MIN_IMPORTANCE
            assert out.max() <= trainer.MAX_IMPORTANCE


class TestFeatureNamesAreRecorded:
    def test_lexical_artifact_records_feature_names(self, trainer):
        """Positional coefficients with no names in the file is how a reordering
        becomes undiagnosable. Names are documentation the artifact carries."""
        from agent_memory.services.importance import LEXICAL_FEATURE_NAMES

        doc = trainer.build_artifact("lexical", [0.0] * 7, 0.0, training={})
        assert tuple(doc["training"]["feature_names"]) == LEXICAL_FEATURE_NAMES
