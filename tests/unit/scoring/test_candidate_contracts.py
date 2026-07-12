from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.scoring import (
    LEGACY_UNTRACKED_SCORE_VERSION,
    CommunicationScores,
    ScoringFunctional,
    ScoringFunctionalStatus,
    ScoringModelManifest,
    float64_array_digest,
    mechanistic_strength,
    pair_softmin,
    score_communication,
)

_MANIFEST_VALUES = {
    "score_version": "mechanistic_softmin_v2",
    "basis_id": "basis-1",
    "coefficient_digest": "coefficients-1",
    "receptor_gate_manifest_id": "receptor-gates-1",
    "target_weight_manifest_id": "target-weights-1",
    "downstream_functional_id": "downstream-1",
    "sender_functional_id": "sender-1",
    "availability_transform_id": "availability-transform-1",
    "precision_transform_id": "precision-transform-1",
    "filter_universe_id": "filter-universe-1",
    "tuning_manifest_id": "tuning-1",
}


def _manifest(**overrides: str) -> ScoringModelManifest:
    values = {**_MANIFEST_VALUES, **overrides}
    return ScoringModelManifest(**values)


def _functional(
    manifest: ScoringModelManifest | None = None,
    *,
    status: ScoringFunctionalStatus = ScoringFunctionalStatus.EXPLORATORY_IN_SAMPLE,
) -> ScoringFunctional:
    return ScoringFunctional(
        contrast_name="treated-v-control",
        contrast_contexts=("control", "treated"),
        training_subject_ids=("train-1", "train-2"),
        interaction_ids=("i1",),
        target_ids=("G1",),
        score_version=(
            LEGACY_UNTRACKED_SCORE_VERSION
            if manifest is None
            else manifest.score_version
        ),
        model_manifest=manifest,
        status=status,
        fold_id="fold-1" if status is ScoringFunctionalStatus.OUT_OF_FOLD else None,
    )


def _availability(functional: ScoringFunctional) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-c", "sample-t"],
            "subject_id": ["test-c", "test-t"],
            "context": ["control", "treated"],
            "sender": ["S", "S"],
            "receiver": ["R", "R"],
            "interaction_id": ["i1", "i1"],
            "state_availability": [0.8, 0.8],
            "ecosystem_availability": [0.4, 0.4],
            "sender_component": [0.5, 0.5],
            "scoring_function_id": [functional.scoring_function_id] * 2,
        }
    )


def _downstream(functional: ScoringFunctional) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["sample-c", "sample-t"],
            "subject_id": ["test-c", "test-t"],
            "context": ["control", "treated"],
            "receiver": ["R", "R"],
            "interaction_id": ["i1", "i1"],
            "downstream_activity": [0.25, 0.25],
            "prior_quality": [1.0, 1.0],
            "scoring_function_id": [functional.scoring_function_id] * 2,
        }
    )


def test_float64_array_digest_is_dtype_endian_and_order_stable() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="F")
    little = np.asarray(values, dtype="<f8", order="C")
    big = np.asarray(values, dtype=">f8", order="C")

    assert float64_array_digest(values) == float64_array_digest(little)
    assert float64_array_digest(values) == float64_array_digest(big)
    assert float64_array_digest(values) != float64_array_digest(values + 1.0)
    with pytest.raises(ValueError, match="finite"):
        float64_array_digest(np.asarray([np.nan]))


@pytest.mark.parametrize("field_name", tuple(_MANIFEST_VALUES))
def test_every_learned_artifact_changes_manifest_and_functional_id(
    field_name: str,
) -> None:
    first_manifest = _manifest()
    changed_manifest = _manifest(
        **{field_name: f"{_MANIFEST_VALUES[field_name]}-changed"}
    )

    assert first_manifest.model_manifest_id != changed_manifest.model_manifest_id
    assert _functional(first_manifest).scoring_function_id != _functional(
        changed_manifest
    ).scoring_function_id


def test_model_manifest_is_stable_serializable_and_validated() -> None:
    first = _manifest()
    second = _manifest()

    assert first.model_manifest_id == second.model_manifest_id
    assert first.to_dict()["model_manifest_id"] == first.model_manifest_id
    assert set(first.to_dict()) == {"model_manifest_id", *_MANIFEST_VALUES}
    with pytest.raises(ValueError, match="basis_id"):
        _manifest(basis_id=" ")


def test_tracked_functional_requires_matching_manifest() -> None:
    legacy = _functional()
    assert legacy.score_version == LEGACY_UNTRACKED_SCORE_VERSION
    assert legacy.model_manifest_id is None
    assert legacy.to_dict()["model_manifest"] is None

    with pytest.raises(ValueError, match="requires a ScoringModelManifest"):
        ScoringFunctional(
            contrast_name="treated-v-control",
            contrast_contexts=("control", "treated"),
            training_subject_ids=("train-1",),
            interaction_ids=("i1",),
            target_ids=("G1",),
            score_version="mechanistic_softmin_v2",
        )
    with pytest.raises(ValueError, match="must match"):
        ScoringFunctional(
            contrast_name="treated-v-control",
            contrast_contexts=("control", "treated"),
            training_subject_ids=("train-1",),
            interaction_ids=("i1",),
            target_ids=("G1",),
            score_version="wrong-version",
            model_manifest=_manifest(),
        )


def test_tracked_provenance_is_preserved_and_common_across_contexts() -> None:
    functional = _functional(
        _manifest(), status=ScoringFunctionalStatus.OUT_OF_FOLD
    )
    result = score_communication(
        _availability(functional), _downstream(functional), functional
    )

    assert set(result.table["score_version"]) == {functional.score_version}
    assert set(result.table["model_manifest_id"]) == {
        functional.model_manifest_id
    }
    inconsistent = result.table.copy()
    inconsistent.loc[
        inconsistent["context"] == "treated", "model_manifest_id"
    ] = "different-model"
    with pytest.raises(ValueError, match="model_manifest_id provenance"):
        CommunicationScores(inconsistent, functional)


def test_pair_softmin_is_exact_at_zero_limited_and_missing_aware() -> None:
    assert pair_softmin(0.4, 0.4) == pytest.approx(0.4)
    limited = pair_softmin(0.8, 0.1, power=8.0)
    raised = pair_softmin(0.8, 0.2, power=8.0)
    assert limited is not None and raised is not None
    assert 0.1 <= limited < raised < 0.8
    assert pair_softmin(0.8, 0.0) == 0.0
    assert pair_softmin(None, 0.5) is None

    with pytest.raises(ValueError, match="power"):
        pair_softmin(0.5, 0.5, power=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pair_softmin(1.1, 0.5)


def test_mechanistic_strength_keeps_lr_core_sender_unresolved_and_conserved() -> None:
    core, adjusted, unresolved = mechanistic_strength(
        availability=0.8,
        incremental_downstream=0.2,
        prior_quality=1.0,
        sender_weight=None,
    )
    assert core is not None
    assert adjusted == core
    assert unresolved is None

    sender_strengths = [
        mechanistic_strength(
            availability=0.8,
            incremental_downstream=0.2,
            prior_quality=1.0,
            sender_weight=weight,
        )[2]
        for weight in (0.25, 0.75)
    ]
    assert all(value is not None for value in sender_strengths)
    assert sum(value or 0.0 for value in sender_strengths) == pytest.approx(adjusted)

    same_core, half_quality, sender_resolved = mechanistic_strength(
        availability=0.8,
        incremental_downstream=0.2,
        prior_quality=0.5,
        sender_weight=0.25,
    )
    assert same_core == core
    assert half_quality == pytest.approx(core * 0.5)
    assert sender_resolved == pytest.approx(core * 0.5 * 0.25)


def test_mechanistic_missingness_does_not_replace_missing_with_zero() -> None:
    core, adjusted, sender_resolved = mechanistic_strength(
        availability=0.8,
        incremental_downstream=0.2,
        prior_quality=None,
        sender_weight=0.5,
    )
    assert core is not None
    assert adjusted is None
    assert sender_resolved is None

    assert mechanistic_strength(
        availability=None,
        incremental_downstream=0.2,
        prior_quality=1.0,
        sender_weight=0.5,
    ) == (None, None, None)
