from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from crychic.core import ContractError
from crychic.scoring import (
    FoldLatentNuisanceArtifact,
    FrozenLatentNuisanceSpec,
    fit_fold_latent_nuisance,
)


def _inputs() -> dict[str, object]:
    rng = np.random.default_rng(7103)
    n_rows = 16
    n_features = 10
    first = np.tile(np.asarray([-2.0, -1.0, 1.0, 2.0]), 4)
    second = np.repeat(np.asarray([-1.5, -0.5, 0.5, 1.5]), 4)
    control_loadings = np.asarray(
        [
            [1.1, 0.2],
            [0.9, -0.3],
            [-0.7, 0.8],
            [0.5, 1.0],
            [-0.4, -0.6],
            [0.8, -0.7],
        ]
    )
    target_loadings = np.asarray(
        [[1.4, -0.2], [-0.8, 1.1], [0.7, 0.9], [-1.2, -0.5]]
    )
    scores = np.column_stack([first, second])
    response = np.column_stack(
        [scores @ control_loadings.T, scores @ target_loadings.T]
    )
    response += rng.normal(0.0, 0.015, size=(n_rows, n_features))
    family_basis = np.zeros((n_features, 2), dtype=np.float64)
    family_basis[6:8, 0] = 1.0
    family_basis[8:, 1] = 1.0
    return {
        "response_matrix": response,
        "feature_ids": tuple(f"g{index:02d}" for index in range(n_features)),
        "family_basis": family_basis,
        "nuisance_matrix": np.ones((n_rows, 1), dtype=np.float64),
        "sample_ids": tuple(f"sample-{index:02d}" for index in range(n_rows)),
        "subject_ids": tuple(f"subject-{index // 2:02d}" for index in range(n_rows)),
        "context_ids": tuple("AB"[index % 2] for index in range(n_rows)),
        "training_weights": np.linspace(0.8, 1.2, n_rows),
        "feature_center": np.zeros(n_features, dtype=np.float64),
        "feature_scale": np.linspace(0.8, 1.3, n_features),
        "precision_weights": np.linspace(0.5, 1.5, n_features),
        "fold_id": "repeat-01/fold-02",
        "receiver": "receiver-A",
        "contrast_name": "treated-vs-control",
        "spec": FrozenLatentNuisanceSpec(
            max_components=2,
            min_control_features=5,
            min_training_subjects=4,
            svd_rcond=1e-10,
            minimum_explained_fraction=0.01,
        ),
    }


def _fit(values: dict[str, object]) -> FoldLatentNuisanceArtifact:
    return fit_fold_latent_nuisance(**values)  # type: ignore[arg-type]


def test_spec_identity_is_stable_and_detects_forced_mutation() -> None:
    first = FrozenLatentNuisanceSpec(max_components=3)
    second = FrozenLatentNuisanceSpec(max_components=3)

    assert first.spec_id == second.spec_id
    assert first.to_dict()["algorithm_contract"].endswith("v1")
    with pytest.raises(TypeError, match="producer-owned"):
        FoldLatentNuisanceArtifact()

    object.__setattr__(first, "max_components", 4)
    with pytest.raises(ContractError) as caught:
        first.to_dict()
    assert caught.value.details.code == "latent_nuisance_spec_integrity_violation"


def test_training_row_order_is_canonical_and_signs_are_deterministic() -> None:
    values = _inputs()
    first = _fit(values)
    order = np.arange(15, -1, -1)
    reordered = dict(values)
    for name in (
        "response_matrix",
        "nuisance_matrix",
        "training_weights",
        "sample_ids",
        "subject_ids",
        "context_ids",
    ):
        source = values[name]
        if isinstance(source, np.ndarray):
            reordered[name] = source[order]
        else:
            reordered[name] = tuple(source[index] for index in order)  # type: ignore[index]
    second = _fit(reordered)

    assert first.status == second.status == "observed"
    assert first.artifact_id == second.artifact_id
    assert first.training_row_manifest_id == second.training_row_manifest_id
    np.testing.assert_array_equal(first.factor_scores, second.factor_scores)
    np.testing.assert_array_equal(first.coordinate_basis, second.coordinate_basis)
    for component in range(first.control_score_basis.shape[1]):
        loadings = first.control_score_basis[:, component]
        maximum = np.max(np.abs(loadings))
        tied = np.flatnonzero(np.abs(loadings) == maximum)
        anchor = min(
            tied,
            key=lambda index: first.control_feature_ids[int(index)],
        )
        assert loadings[anchor] > 0.0
    for array in (
        first.coordinate_basis,
        first.factor_scores,
        first.control_score_basis,
        first.singular_values,
        first.explained_variance_fractions,
    ):
        assert not array.flags.writeable


def test_api_has_no_heldout_input_and_target_columns_cannot_change_scores() -> None:
    assert not any(
        "heldout" in parameter
        for parameter in inspect.signature(fit_fold_latent_nuisance).parameters
    )
    values = _inputs()
    first = _fit(values)
    poisoned = dict(values)
    changed = np.asarray(values["response_matrix"]).copy()
    changed[:, 6:] += np.arange(changed.shape[0])[:, np.newaxis] * np.asarray(
        [20.0, -30.0, 40.0, -50.0]
    )
    poisoned["response_matrix"] = changed
    second = _fit(poisoned)

    assert first.status == second.status == "observed"
    assert first.response_digest != second.response_digest
    assert first.artifact_id != second.artifact_id
    assert first.control_score_input_digest == second.control_score_input_digest
    assert first.control_residual_digest == second.control_residual_digest
    assert first.program_ids == second.program_ids
    np.testing.assert_array_equal(first.factor_scores, second.factor_scores)
    np.testing.assert_array_equal(
        first.control_score_basis, second.control_score_basis
    )


def test_insufficient_controls_and_zero_control_rank_are_typed_not_estimable() -> None:
    values = _inputs()
    family_basis = np.asarray(values["family_basis"]).copy()
    family_basis[:4, 0] = 1.0
    too_few = _fit({**values, "family_basis": family_basis})

    constant = np.asarray(values["response_matrix"]).copy()
    constant[:, :6] = 3.0
    zero_rank = _fit({**values, "response_matrix": constant})

    assert too_few.status == "not_estimable"
    assert too_few.reason_code == "latent_nuisance_insufficient_control_features"
    assert too_few.coordinate_basis.shape == (10, 0)
    assert zero_rank.status == "not_estimable"
    assert zero_rank.reason_code == "latent_nuisance_control_variance_not_estimable"
    assert zero_rank.program_ids == ()


def test_rank_deficient_static_basis_is_typed_not_estimable() -> None:
    values = _inputs()
    static = np.column_stack(
        [np.linspace(-1.0, 1.0, 10), np.linspace(-1.0, 1.0, 10)]
    )
    result = _fit(
        {
            **values,
            "static_coordinate_basis": static,
            "static_coordinate_basis_id": "static-basis-1",
            "static_program_ids": ("static-1", "static-2"),
        }
    )

    assert result.status == "not_estimable"
    assert result.reason_code == "latent_nuisance_static_basis_rank_not_estimable"
    assert result.static_numerical_rank == 1


def test_learned_basis_is_precision_orthogonal_to_static_span() -> None:
    values = _inputs()
    response = np.asarray(values["response_matrix"]).copy()
    static = np.asarray(
        [1.0, -0.8, 0.6, -0.4, 0.2, 0.5, -0.9, 0.7, -0.3, 1.1]
    )[:, np.newaxis]
    static_score = np.linspace(-2.5, 2.5, response.shape[0])
    response += static_score[:, np.newaxis] * static.T
    spec = replace(values["spec"], max_components=1)  # type: ignore[arg-type]
    result = _fit(
        {
            **values,
            "response_matrix": response,
            "spec": spec,
            "static_coordinate_basis": static,
            "static_coordinate_basis_id": "reviewed-static-basis-1",
            "static_program_ids": ("reviewed-static-program-1",),
        }
    )

    assert result.status == "observed"
    precision = np.asarray(values["precision_weights"])
    cross_product = static.T @ (precision[:, np.newaxis] * result.coordinate_basis)
    np.testing.assert_allclose(cross_product, 0.0, atol=1e-10, rtol=0.0)
    assert result.static_numerical_rank == 1
    assert result.static_orthogonality_max_abs < 1e-10
