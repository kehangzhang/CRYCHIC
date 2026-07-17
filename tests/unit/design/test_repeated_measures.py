from __future__ import annotations

import pandas as pd
import pytest

from crychic.core import ContractError
from crychic.design import (
    FrozenRepeatedMeasuresDesign,
    RepeatedMeasuresDesignSpec,
    balanced_contrast,
    freeze_repeated_measures_design,
)


def _mixed_metadata() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    allocations = {
        "p1": ("control", "case"),
        "p2": ("control", "case"),
        "p3": ("control",),
        "p4": ("control",),
        "p5": ("case",),
        "p6": ("case",),
    }
    for subject, contexts in allocations.items():
        for context in contexts:
            rows.append(
                {
                    "sample_id": f"{subject}-{context}",
                    "subject_id": subject,
                    "condition": context,
                }
            )
    rows.append(
        {
            "sample_id": "p1-control-technical-replicate",
            "subject_id": "p1",
            "condition": "control",
        }
    )
    return pd.DataFrame(rows)


def _spec(*, subject_fixed_effects: bool = False) -> RepeatedMeasuresDesignSpec:
    return RepeatedMeasuresDesignSpec(
        context_keys=("condition",),
        subject_fixed_effects=subject_fixed_effects,
        min_subjects_per_context=3,
        min_subject_clusters=6,
    )


def _contrast():
    return balanced_contrast(("case",), ("control",), name="case_vs_control")


def test_mixed_allocation_freezes_subject_clusters_and_collapses_replicates() -> None:
    design = freeze_repeated_measures_design(
        _mixed_metadata(),
        contrast=_contrast(),
        spec=_spec(),
    )

    assert design.estimable
    assert design.reason_code is None
    assert len(design.sample_ids) == 9
    assert len(design.cell_ids) == 8
    assert sorted(design.cell_sample_counts) == [1, 1, 1, 1, 1, 1, 1, 2]
    assert design.n_subject_clusters == 6
    assert design.n_contrast_subject_clusters == 6
    assert design.n_repeated_subject_clusters == 2
    assert design.n_complete_contrast_subjects == 2
    assert design.context_subject_counts[0][1] == 4
    assert design.context_subject_counts[1][1] == 4
    assert not design.design_matrix.flags.writeable
    assert not design.contrast_vector.flags.writeable
    assert not design.sample_cell_indices.flags.writeable
    assert design.to_dict()["model_version"] == (
        "formula_ols_subject_cluster_cr1_v1"
    )


def test_design_identity_is_sample_row_order_invariant() -> None:
    metadata = _mixed_metadata()
    first = freeze_repeated_measures_design(
        metadata,
        contrast=_contrast(),
        spec=_spec(),
    )
    second = freeze_repeated_measures_design(
        metadata.sample(frac=1.0, random_state=17),
        contrast=_contrast(),
        spec=_spec(),
    )

    assert first.design_id == second.design_id


def test_multi_context_partial_repetition_supports_pairwise_contrast() -> None:
    allocations = {
        "p1": ("a", "b", "c"),
        "p2": ("a", "b"),
        "p3": ("b", "c"),
        "p4": ("a",),
        "p5": ("c",),
        "p6": ("a", "c"),
    }
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{subject}-{context}",
                "subject_id": subject,
                "condition": context,
            }
            for subject, contexts in allocations.items()
            for context in contexts
        ]
    )
    spec = RepeatedMeasuresDesignSpec(
        context_keys=("condition",),
        min_subjects_per_context=3,
        min_subject_clusters=5,
    )
    contrast = balanced_contrast(("c",), ("a",), name="c_vs_a")

    design = freeze_repeated_measures_design(
        metadata,
        contrast=contrast,
        spec=spec,
    )

    assert design.estimable
    assert design.n_contrast_subject_clusters == 6
    assert design.n_complete_contrast_subjects == 2
    assert design.n_repeated_subject_clusters == 2


def test_subject_fixed_effects_reject_between_subject_contrast() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{context}-{index}",
                "subject_id": f"{context}-{index}",
                "condition": context,
            }
            for context in ("control", "case")
            for index in range(4)
        ]
    )

    design = freeze_repeated_measures_design(
        metadata,
        contrast=_contrast(),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            subject_fixed_effects=True,
            min_subjects_per_context=3,
            min_subject_clusters=6,
        ),
    )

    assert not design.estimable
    assert design.reason_code == "subject_fixed_effects_absorb_contrast"


def test_subject_fixed_effects_keep_within_subject_contrast_estimable() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"p{index}-{context}",
                "subject_id": f"p{index}",
                "condition": context,
            }
            for index in range(6)
            for context in ("control", "case")
        ]
    )

    design = freeze_repeated_measures_design(
        metadata,
        contrast=_contrast(),
        spec=_spec(subject_fixed_effects=True),
    )

    assert design.estimable
    assert design.residual_df == 5
    assert design.n_complete_contrast_subjects == 6


def test_confounded_repeated_formula_is_rank_deficient_and_fails_closed() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"p{index}-{condition}",
                "subject_id": f"p{index}",
                "condition": condition,
                "batch": "batch-control" if condition == "control" else "batch-case",
            }
            for index in range(6)
            for condition in ("control", "case")
        ]
    )

    design = freeze_repeated_measures_design(
        metadata,
        contrast=_contrast(),
        spec=RepeatedMeasuresDesignSpec(
            context_keys=("condition",),
            covariates=("batch",),
            categorical_covariates=("batch",),
            formula="~ condition + batch",
            min_subjects_per_context=3,
            min_subject_clusters=6,
        ),
    )

    assert not design.estimable
    assert design.reason_code == "design_rank_deficient"
    assert design.design_rank < design.design_matrix.shape[1]


def test_frozen_design_constructor_is_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenRepeatedMeasuresDesign()


def test_forced_support_diagnostic_mutation_fails_integrity_check() -> None:
    design = freeze_repeated_measures_design(
        _mixed_metadata(),
        contrast=_contrast(),
        spec=_spec(),
    )
    object.__setattr__(design, "n_subject_clusters", 600)

    with pytest.raises(ContractError) as error:
        design.to_dict()
    assert error.value.details.code == (
        "repeated_measures_design_integrity_violation"
    )


def test_forced_spec_mutation_fails_integrity_check() -> None:
    design = freeze_repeated_measures_design(
        _mixed_metadata(),
        contrast=_contrast(),
        spec=_spec(),
    )
    object.__setattr__(design.spec, "min_subject_clusters", 600)

    with pytest.raises(ContractError) as error:
        design.to_dict()
    assert error.value.details.code == (
        "repeated_measures_design_integrity_violation"
    )
