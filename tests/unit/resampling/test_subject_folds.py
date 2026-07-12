from __future__ import annotations

import pandas as pd
import pytest

from crychic.core import SeedLineage
from crychic.design import balanced_contrast
from crychic.resampling import (
    DesignFoldChecker,
    FoldEstimabilityResult,
    FoldManifest,
    FoldPlanningError,
    plan_subject_folds,
)


def _checker(*, with_batch: bool = False) -> DesignFoldChecker:
    return DesignFoldChecker(
        context_keys=("condition",),
        covariates=("batch",) if with_batch else (),
        formula="~ batch + condition" if with_batch else "~ condition",
        contrasts=(
            balanced_contrast(
                ("stim",),
                ("ctrl",),
                name="stim_vs_ctrl",
            ),
        ),
    )


def _paired_metadata(n_subjects: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"{subject}-{condition}",
                "subject_id": subject,
                "condition": condition,
                "batch": int(subject.removeprefix("p")) % 2,
            }
            for subject in (f"p{index}" for index in range(n_subjects))
            for condition in ("ctrl", "stim")
        ]
    )


def test_paired_subjects_are_never_split_and_have_exactly_one_oof_fold() -> None:
    metadata = _paired_metadata()
    plan = plan_subject_folds(
        metadata,
        design_checker=_checker(),
        context_keys=("condition",),
        allowed_n_splits=(4, 3, 2),
        seed_lineage=SeedLineage(20260713),
    )

    assert plan.effective_n_splits == 4
    test_counts = dict.fromkeys(plan.subject_ids, 0)
    for fold in plan:
        assert not set(fold.train_subject_ids).intersection(fold.test_subject_ids)
        assert set(fold.context_support.values()) == {6}
        assert set(fold.test_context_support.values()) == {2}
        for subject in fold.test_subject_ids:
            test_counts[subject] += 1
            observed = set(
                metadata.loc[metadata["subject_id"].eq(subject), "condition"]
            )
            assert observed == {"ctrl", "stim"}
    assert set(test_counts.values()) == {1}


def test_fold_plan_is_deterministic_and_input_order_invariant() -> None:
    metadata = _paired_metadata()
    kwargs = {
        "design_checker": _checker(),
        "context_keys": ("condition",),
        "allowed_n_splits": (4, 3, 2),
        "seed_lineage": SeedLineage(91),
    }

    first = plan_subject_folds(metadata, **kwargs)
    shuffled = plan_subject_folds(
        metadata.sample(frac=1.0, random_state=7).reset_index(drop=True),
        **kwargs,
    )

    assert shuffled.to_dict() == first.to_dict()


def test_independent_groups_are_stratified_by_context_and_batch() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{condition}-{batch}-{index}",
                "subject_id": f"{condition}-{batch}-{index}",
                "condition": condition,
                "batch": batch,
            }
            for condition in ("ctrl", "stim")
            for batch in ("a", "b")
            for index in range(3)
        ]
    )
    plan = plan_subject_folds(
        metadata,
        design_checker=_checker(with_batch=True),
        context_keys=("condition",),
        strata_keys=("batch",),
        allowed_n_splits=(3, 2),
        seed_lineage=SeedLineage(3),
    )

    assert plan.effective_n_splits == 3
    for fold in plan:
        held = metadata.loc[metadata["subject_id"].isin(fold.test_subject_ids)]
        support = held.groupby(["condition", "batch"], observed=True).size()
        assert len(support) == 4
        assert set(support) == {1}


def test_fold_count_reduction_uses_only_predeclared_candidates() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{condition}-{index}",
                "subject_id": f"{condition}-{index}",
                "condition": condition,
            }
            for condition in ("ctrl", "stim")
            for index in range(3)
        ]
    )
    plan = plan_subject_folds(
        metadata,
        design_checker=_checker(),
        context_keys=("condition",),
        allowed_n_splits=(4, 3, 2),
        seed_lineage=SeedLineage(5),
    )

    assert plan.requested_n_splits == 4
    assert plan.effective_n_splits == 3
    assert plan.reduction_reason_code == "requested_4_not_estimable_selected_3"
    assert plan.rejected_candidate_reasons[0][0] == 4

    with pytest.raises(FoldPlanningError, match="no predeclared"):
        plan_subject_folds(
            metadata,
            design_checker=_checker(),
            context_keys=("condition",),
            allowed_n_splits=(4,),
            seed_lineage=SeedLineage(5),
        )


def test_training_fold_design_rank_is_reaudited_and_can_block_all_k() -> None:
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{condition}-{index}",
                "subject_id": f"{condition}-{index}",
                "condition": condition,
                "batch": int(condition == "stim"),
            }
            for condition in ("ctrl", "stim")
            for index in range(4)
        ]
    )

    with pytest.raises(FoldPlanningError, match="fold_design_not_estimable"):
        plan_subject_folds(
            metadata,
            design_checker=_checker(with_batch=True),
            context_keys=("condition",),
            allowed_n_splits=(4, 2),
            seed_lineage=SeedLineage(12),
        )


def test_declared_strata_must_be_immutable_within_subject() -> None:
    metadata = _paired_metadata(4)
    metadata.loc[metadata["subject_id"].eq("p0"), "batch"] = [0, 1]

    with pytest.raises(ValueError, match="immutable within subject"):
        plan_subject_folds(
            metadata,
            design_checker=_checker(),
            context_keys=("condition",),
            strata_keys=("batch",),
            allowed_n_splits=(2,),
        )


def test_fold_manifest_rejects_subject_overlap() -> None:
    check = FoldEstimabilityResult(
        design_matrix_id="design-1",
        contrast_ids=("contrast-1",),
        estimable=True,
        reason_code=None,
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        FoldManifest(
            repeat_id="repeat-0",
            train_subject_ids=("p1", "p2"),
            test_subject_ids=("p2",),
            design_matrix_id=check.design_matrix_id,
            contrast_ids=check.contrast_ids,
            context_support={"ctrl": 2, "stim": 2},
            test_context_support={"ctrl": 1, "stim": 1},
            estimable=True,
            reason_code=None,
            seed_lineage=SeedLineage(1).derive("fold"),
            requested_n_splits=2,
            effective_n_splits=2,
            fold_index=0,
        )
