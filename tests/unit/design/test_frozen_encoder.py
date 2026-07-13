from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.design import (
    ContextGraph,
    FrozenDesignApplication,
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    balanced_contrast,
    factorial_interaction_contrast,
    fit_frozen_design_encoder,
)


def _metadata(subjects: tuple[str, ...], *, site: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, subject in enumerate(subjects):
        for condition in ("ctrl", "stim"):
            rows.append(
                {
                    "sample_id": f"{subject}:{condition}",
                    "subject_id": subject,
                    "condition": condition,
                    "site": site[index],
                    "age": 30.0 + 5.0 * index,
                    "constant": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _contrast():
    return balanced_contrast(("stim",), ("ctrl",), name="stim_vs_ctrl")


def test_frozen_encoder_fits_reference_coding_and_fixed_contrast() -> None:
    metadata = _metadata(("p1", "p2", "p3"), site=("a", "b", "a"))

    encoder = fit_frozen_design_encoder(
        metadata,
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )

    assert encoder.training_subject_ids == ("p1", "p2", "p3")
    assert encoder.training_sample_subject_ids == (
        "p1",
        "p1",
        "p2",
        "p2",
        "p3",
        "p3",
    )
    assert len(encoder.training_sample_context_ids) == len(encoder.training_sample_ids)
    assert len(set(encoder.training_sample_context_ids)) == 2
    assert len(encoder.nuisance_column_ids) == 3
    assert set(np.unique(encoder.training_context_regressor)) == {0.0, 1.0}
    assert np.isfinite(encoder.training_nuisance_matrix).all()
    assert encoder.training_nuisance_matrix.flags.writeable is False
    assert encoder.context_regressor_id
    assert encoder.nuisance_design_id
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenDesignEncoder()
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenDesignApplication()


def test_frozen_encoder_applies_train_levels_and_scales_without_refit() -> None:
    training = _metadata(("p1", "p2", "p3"), site=("a", "b", "a"))
    encoder = fit_frozen_design_encoder(
        training,
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )
    heldout = _metadata(("q1", "q2"), site=("a", "b"))
    heldout.loc[heldout["subject_id"].eq("q2"), "age"] = 80.0

    applied = apply_frozen_design_encoder(encoder, heldout)

    assert applied.status == "observed"
    assert applied.reason_code is None
    assert applied.encoder_id == encoder.encoder_id
    assert applied.context_regressor_id == encoder.context_regressor_id
    assert applied.nuisance_design_id == encoder.nuisance_design_id
    assert applied.application_id
    assert applied.sample_subject_ids == ("q1", "q1", "q2", "q2")
    assert len(set(applied.sample_context_ids)) == 2
    assert applied.nuisance_matrix.shape == (4, 3)
    assert np.max(applied.nuisance_matrix[:, 2]) > 5
    applied.require_compatible(encoder)


def test_application_identity_is_row_order_invariant_and_binds_rows() -> None:
    training = _metadata(("p1", "p2", "p3"), site=("a", "b", "a"))
    encoder = fit_frozen_design_encoder(
        training,
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )
    heldout = _metadata(("q1", "q2"), site=("a", "b"))

    original = apply_frozen_design_encoder(encoder, heldout)
    shuffled = apply_frozen_design_encoder(
        encoder, heldout.sample(frac=1.0, random_state=7)
    )

    assert shuffled.application_id == original.application_id
    assert shuffled.sample_ids == original.sample_ids
    assert shuffled.sample_subject_ids == original.sample_subject_ids
    assert shuffled.sample_context_ids == original.sample_context_ids

    poisoned = apply_frozen_design_encoder(encoder, heldout)
    object.__setattr__(
        poisoned,
        "sample_subject_ids",
        tuple(reversed(poisoned.sample_subject_ids)),
    )
    with pytest.raises(ContractError) as captured:
        poisoned.to_dict()
    assert (
        captured.value.details.code == "frozen_design_application_integrity_violation"
    )


def test_frozen_design_arrays_are_byte_backed_and_poison_is_detected() -> None:
    encoder = fit_frozen_design_encoder(
        _metadata(("p1", "p2", "p3"), site=("a", "b", "a")),
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )
    applied = apply_frozen_design_encoder(
        encoder,
        _metadata(("q1", "q2"), site=("a", "b")),
    )

    for values in (
        encoder.training_nuisance_matrix,
        encoder.training_context_regressor,
        applied.nuisance_matrix,
        applied.context_regressor,
    ):
        assert values.flags.writeable is False
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)

    object.__setattr__(
        applied,
        "context_regressor",
        applied.context_regressor.copy(),
    )
    with pytest.raises(ContractError) as captured:
        applied.require_compatible(encoder)
    assert (
        captured.value.details.code == "frozen_design_application_integrity_violation"
    )


def test_training_application_binds_encoder_matrices_and_rows() -> None:
    encoder = fit_frozen_design_encoder(
        _metadata(("p1", "p2", "p3"), site=("a", "b", "a")),
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )

    training = encoder.training_application()

    assert training.application_scope == "training"
    assert training.sample_ids == encoder.training_sample_ids
    assert training.sample_subject_ids == encoder.training_sample_subject_ids
    assert training.sample_context_ids == encoder.training_sample_context_ids
    assert np.array_equal(
        training.nuisance_matrix,
        encoder.training_nuisance_matrix,
    )
    assert np.array_equal(
        training.context_regressor,
        encoder.training_context_regressor,
    )
    training.require_compatible(encoder)


def test_encoder_integrity_rejects_poisoned_training_rows_and_matrix() -> None:
    original = fit_frozen_design_encoder(
        _metadata(("p1", "p2", "p3"), site=("a", "b", "a")),
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )
    row_poisoned = fit_frozen_design_encoder(
        _metadata(("p1", "p2", "p3"), site=("a", "b", "a")),
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )
    object.__setattr__(
        row_poisoned,
        "training_sample_subject_ids",
        tuple(reversed(row_poisoned.training_sample_subject_ids)),
    )
    with pytest.raises(ContractError) as row_error:
        row_poisoned.training_application()
    assert row_error.value.details.code == "frozen_design_encoder_integrity_violation"

    matrix_poisoned = fit_frozen_design_encoder(
        _metadata(("p1", "p2", "p3"), site=("a", "b", "a")),
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )
    changed = matrix_poisoned.training_nuisance_matrix.copy()
    changed[0, 0] += 1.0
    object.__setattr__(matrix_poisoned, "training_nuisance_matrix", changed)
    with pytest.raises(ContractError) as matrix_error:
        matrix_poisoned.training_application()
    assert (
        matrix_error.value.details.code == "frozen_design_encoder_integrity_violation"
    )
    original.training_application()


def test_application_rejects_another_encoder_lineage() -> None:
    training = _metadata(("p1", "p2", "p3"), site=("a", "b", "a"))
    encoder = fit_frozen_design_encoder(
        training,
        contrast=_contrast(),
        context_keys=("condition",),
    )
    different = fit_frozen_design_encoder(
        training,
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("age",),
    )
    applied = apply_frozen_design_encoder(
        encoder,
        _metadata(("q1", "q2"), site=("a", "b")),
    )

    with pytest.raises(ContractError) as captured:
        applied.require_compatible(different)
    assert captured.value.details.code == "frozen_design_application_scope_mismatch"


def test_heldout_encoding_does_not_trust_mutable_patsy_state() -> None:
    training = _metadata(("p1", "p2", "p3"), site=("a", "b", "a"))
    encoder = fit_frozen_design_encoder(
        training,
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site",),
        formula="~ site + condition",
    )
    heldout = _metadata(("q1", "q2"), site=("a", "b"))
    expected = apply_frozen_design_encoder(encoder, heldout)

    # Patsy state is deliberately not retained as a trusted encoder input.
    assert not hasattr(encoder, "_design_info")
    repeated = apply_frozen_design_encoder(encoder, heldout)

    assert repeated.application_id == expected.application_id
    np.testing.assert_array_equal(
        repeated.context_regressor, expected.context_regressor
    )


def test_unseen_heldout_category_is_not_estimable_and_never_reference() -> None:
    encoder = fit_frozen_design_encoder(
        _metadata(("p1", "p2", "p3"), site=("a", "b", "a")),
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site",),
    )
    heldout = _metadata(("q1", "q2"), site=("a", "new-site"))

    applied = apply_frozen_design_encoder(encoder, heldout)

    assert applied.status == "not_estimable"
    assert applied.reason_code == "unseen_heldout_level:site"
    assert np.isnan(applied.nuisance_matrix).all()
    assert np.isnan(applied.context_regressor).all()


def test_constant_training_covariate_new_heldout_value_is_not_estimable() -> None:
    training = _metadata(("p1", "p2"), site=("a", "a"))
    encoder = fit_frozen_design_encoder(
        training,
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("constant",),
        formula="~ condition",
    )
    heldout = _metadata(("q1", "q2"), site=("a", "a"))
    heldout.loc[heldout["subject_id"].eq("q2"), "constant"] = 2.0

    applied = apply_frozen_design_encoder(encoder, heldout)

    assert applied.status == "not_estimable"
    assert applied.reason_code == (
        "heldout_value_outside_constant_training_support:constant"
    )


def test_default_formula_rejects_a_constant_declared_covariate() -> None:
    with pytest.raises(ValueError, match="design_rank_deficient"):
        fit_frozen_design_encoder(
            _metadata(("p1", "p2"), site=("a", "a")),
            contrast=_contrast(),
            context_keys=("condition",),
            covariates=("constant",),
        )


def test_heldout_poison_cannot_change_training_encoder_identity() -> None:
    training = _metadata(("p1", "p2", "p3"), site=("a", "b", "a"))
    encoder = fit_frozen_design_encoder(
        training,
        contrast=_contrast(),
        context_keys=("condition",),
        covariates=("site", "age"),
    )
    heldout_a = _metadata(("q1", "q2"), site=("a", "b"))
    heldout_b = heldout_a.copy()
    heldout_b["age"] = 1e9

    applied_a = apply_frozen_design_encoder(encoder, heldout_a)
    applied_b = apply_frozen_design_encoder(encoder, heldout_b)

    assert applied_a.encoder_id == applied_b.encoder_id == encoder.encoder_id
    assert not np.array_equal(
        applied_a.nuisance_matrix,
        applied_b.nuisance_matrix,
    )


def test_application_rejects_training_subject_overlap() -> None:
    training = _metadata(("p1", "p2"), site=("a", "b"))
    encoder = fit_frozen_design_encoder(
        training,
        contrast=_contrast(),
        context_keys=("condition",),
    )

    with pytest.raises(ValueError, match="overlaps training subjects"):
        apply_frozen_design_encoder(encoder, training)


def test_unbalanced_three_context_coefficient_is_equal_context_emm() -> None:
    contexts = ["A", *(["B"] * 3), *(["C"] * 6)]
    means = {"A": 10.0, "B": 2.0, "C": 4.0}
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(len(contexts))],
            "subject_id": [f"p{index}" for index in range(len(contexts))],
            "condition": contexts,
        }
    )
    contrast = balanced_contrast(("A",), ("B", "C"), name="A_vs_BC")

    encoder = fit_frozen_design_encoder(
        metadata,
        contrast=contrast,
        context_keys=("condition",),
    )
    design = np.column_stack(
        (encoder.training_nuisance_matrix, encoder.training_context_regressor)
    )
    response = np.asarray([means[value] for value in contexts], dtype=float)
    coefficient = np.linalg.lstsq(design, response, rcond=None)[0][-1]

    assert coefficient == pytest.approx(7.0)

    shifted = response + np.asarray(
        [
            0.0 if value == "A" else (100.0 if value == "B" else -100.0)
            for value in contexts
        ]
    )
    shifted_coefficient = np.linalg.lstsq(design, shifted, rcond=None)[0][-1]
    assert shifted_coefficient == pytest.approx(7.0)


def test_local_contrast_keeps_known_outside_contexts_observed() -> None:
    training = pd.DataFrame(
        {
            "sample_id": [
                f"p{index}:{context}" for index in range(2) for context in "ABCD"
            ],
            "subject_id": [f"p{index}" for index in range(2) for _ in "ABCD"],
            "condition": list("ABCD") * 2,
        }
    )
    encoder = fit_frozen_design_encoder(
        training,
        contrast=balanced_contrast(("A",), ("B",), name="local_A_B"),
        context_keys=("condition",),
    )
    heldout = pd.DataFrame(
        {
            "sample_id": [f"q:{context}" for context in "ABCD"],
            "subject_id": ["q"] * 4,
            "condition": list("ABCD"),
        }
    )

    applied = apply_frozen_design_encoder(encoder, heldout)

    assert applied.status == "observed"
    by_context = dict(
        zip(
            heldout.sort_values("sample_id")["condition"],
            applied.context_regressor,
            strict=True,
        )
    )
    assert by_context["C"] == pytest.approx(0.0)
    assert by_context["D"] == pytest.approx(0.0)


def test_interaction_formula_coefficient_matches_difference_in_differences() -> None:
    rows: list[dict[str, object]] = []
    cell_means = {
        ("ctrl", "core"): 0.0,
        ("stim", "core"): 1.0,
        ("ctrl", "edge"): 2.0,
        ("stim", "edge"): 5.0,
    }
    for replicate in range(2):
        for condition, region in cell_means:
            rows.append(
                {
                    "sample_id": f"p{replicate}:{condition}:{region}",
                    "subject_id": f"p{replicate}",
                    "condition": condition,
                    "region": region,
                    "response": cell_means[(condition, region)],
                }
            )
    metadata = pd.DataFrame(rows)
    nodes = tuple(
        (("condition", condition), ("region", region))
        for condition, region in cell_means
    )
    contrast = factorial_interaction_contrast(
        ContextGraph.complete(nodes),
        "condition",
        "stim",
        "ctrl",
        "region",
        "edge",
        "core",
    )
    encoder = fit_frozen_design_encoder(
        metadata,
        contrast=contrast,
        context_keys=("condition", "region"),
        formula="~ condition * region",
    )
    design = np.column_stack(
        (encoder.training_nuisance_matrix, encoder.training_context_regressor)
    )

    coefficient = np.linalg.lstsq(
        design, metadata.sort_values("sample_id")["response"].to_numpy(), rcond=None
    )[0][-1]

    assert coefficient == pytest.approx(2.0)


def test_numeric_categorical_registry_freezes_levels() -> None:
    training = pd.DataFrame(
        {
            "sample_id": [
                f"p{p}:{c}:{b}" for p in range(2) for c in (0, 1) for b in (0, 1)
            ],
            "subject_id": [f"p{p}" for p in range(2) for _ in range(4)],
            "condition": [c for _ in range(2) for c in (0, 1) for _ in range(2)],
            "batch": [b for _ in range(2) for _ in range(2) for b in (0, 1)],
        }
    )
    encoder = fit_frozen_design_encoder(
        training,
        contrast=balanced_contrast((1,), (0,)),
        context_keys=("condition",),
        covariates=("batch",),
        categorical_covariates=("batch",),
        formula="~ batch * condition",
    )
    heldout = training.iloc[:2].copy()
    heldout["sample_id"] = ["q:0", "q:1"]
    heldout["subject_id"] = "q"
    heldout.loc[heldout.index[1], "batch"] = 2

    applied = apply_frozen_design_encoder(encoder, heldout)

    assert dict(encoder.factor_levels)["batch"] == (0, 1)
    assert applied.status == "not_estimable"
    assert applied.reason_code == "unseen_heldout_level:batch"


def test_encoder_identity_hashes_assignments_but_not_row_order() -> None:
    training = _metadata(("p1", "p2", "p3"), site=("a", "b", "a"))
    kwargs = {
        "contrast": _contrast(),
        "context_keys": ("condition",),
        "covariates": ("site", "age"),
    }
    original = fit_frozen_design_encoder(training, **kwargs)
    shuffled = fit_frozen_design_encoder(
        training.sample(frac=1.0, random_state=5), **kwargs
    )
    context_changed = training.copy()
    context_changed.loc[[0, 1], "condition"] = ["stim", "ctrl"]
    changed_context = fit_frozen_design_encoder(context_changed, **kwargs)
    covariate_changed = training.copy()
    covariate_changed.loc[covariate_changed["subject_id"].eq("p1"), "site"] = "b"
    covariate_changed.loc[covariate_changed["subject_id"].eq("p2"), "site"] = "a"
    changed_covariate = fit_frozen_design_encoder(covariate_changed, **kwargs)

    assert shuffled.encoder_id == original.encoder_id
    assert changed_context.encoder_id != original.encoder_id
    assert changed_covariate.encoder_id != original.encoder_id


def test_encoder_identity_hashes_input_key_schema() -> None:
    training = _metadata(("p1", "p2", "p3"), site=("a", "b", "a"))
    renamed = training.assign(
        sid=training["sample_id"],
        pid=training["subject_id"],
    )
    common = {
        "contrast": _contrast(),
        "context_keys": ("condition",),
        "covariates": ("site", "age"),
    }

    default_keys = fit_frozen_design_encoder(renamed, **common)
    renamed_keys = fit_frozen_design_encoder(
        renamed,
        sample_key="sid",
        subject_key="pid",
        **common,
    )

    assert renamed_keys.encoder_id != default_keys.encoder_id
    assert renamed_keys.context_regressor_id != default_keys.context_regressor_id
    assert renamed_keys.nuisance_design_id != default_keys.nuisance_design_id
