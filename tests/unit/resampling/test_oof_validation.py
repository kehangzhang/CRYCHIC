from __future__ import annotations

import pandas as pd
import pytest

from crychic.core import ContractError, SeedLineage
from crychic.design import balanced_contrast
from crychic.resampling import (
    DesignFoldChecker,
    plan_subject_folds,
    validate_oof_subject_coverage,
)


def _plan(*, multiple_contrasts: bool = False):
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{subject}-{condition}",
                "subject_id": subject,
                "condition": condition,
            }
            for subject in ("p1", "p2", "p3", "p4")
            for condition in ("ctrl", "stim")
        ]
    )
    contrasts = [balanced_contrast(("stim",), ("ctrl",), name="stim_vs_ctrl")]
    if multiple_contrasts:
        contrasts.append(balanced_contrast(("ctrl",), ("stim",), name="ctrl_vs_stim"))
    checker = DesignFoldChecker(
        context_keys=("condition",),
        covariates=(),
        formula="~ condition",
        contrasts=tuple(contrasts),
    )
    return plan_subject_folds(
        metadata,
        design_checker=checker,
        context_keys=("condition",),
        allowed_n_splits=(2,),
        seed_lineage=SeedLineage(17),
    )


def _table(*, multiple_contrasts: bool = False):
    plan = _plan(multiple_contrasts=multiple_contrasts)
    rows: list[dict[str, object]] = []
    for fold in plan:
        for contrast_id in fold.contrast_ids:
            for subject in fold.test_subject_ids:
                for context in ("ctrl", "stim"):
                    rows.append(
                        {
                            "subject_id": subject,
                            "fold_id": fold.fold_id,
                            "contrast_id": contrast_id,
                            "context": context,
                            "functional_status": "out_of_fold",
                            "scoring_function_id": (
                                f"function-{fold.fold_index}-{contrast_id}"
                            ),
                            "model_manifest_id": (
                                f"model-{fold.fold_index}-{contrast_id}"
                            ),
                        }
                    )
    return plan, pd.DataFrame(rows)


def _contexts(plan) -> dict[str, tuple[str, str]]:
    return {
        contrast_id: ("ctrl", "stim")
        for fold in plan
        for contrast_id in fold.contrast_ids
    }


def test_complete_oof_table_passes_exact_assignment_and_functional_audit() -> None:
    plan, table = _table()

    audit = validate_oof_subject_coverage(
        table,
        plan,
        contrast_contexts=_contexts(plan),
    )

    assert audit.fold_plan_id == plan.plan_id
    assert audit.subject_ids == plan.subject_ids
    assert audit.n_rows == 8
    assert audit.to_dict()["common_functional_validated"] is True


def test_oof_audit_rejects_forced_provenance_mutation() -> None:
    plan, table = _table()
    audit = validate_oof_subject_coverage(
        table,
        plan,
        contrast_contexts=_contexts(plan),
    )
    object.__setattr__(audit, "subject_ids", ("POISON",))

    with pytest.raises(ContractError) as error:
        audit.to_dict()
    assert error.value.details.code == "oof_coverage_audit_integrity_violation"


def test_training_subject_in_test_fold_is_rejected_as_leakage() -> None:
    plan, table = _table()
    fold = plan.folds[0]
    table.loc[0, "subject_id"] = fold.train_subject_ids[0]

    with pytest.raises(ValueError, match="leaks training subject"):
        validate_oof_subject_coverage(
            table,
            plan,
            contrast_contexts=_contexts(plan),
        )


def test_every_subject_must_have_exactly_one_oof_fold() -> None:
    plan, table = _table()
    table = table.loc[~table["subject_id"].eq(plan.subject_ids[0])].copy()

    with pytest.raises(ValueError, match="subject coverage mismatch"):
        validate_oof_subject_coverage(
            table,
            plan,
            contrast_contexts=_contexts(plan),
        )


def test_fold_contrast_rejects_multiple_model_manifests() -> None:
    plan, table = _table()
    first_fold = table["fold_id"].iloc[0]
    selected = table["fold_id"].eq(first_fold) & table["context"].eq("stim")
    table.loc[selected, "model_manifest_id"] = "different-model"

    with pytest.raises(ValueError, match="multiple model_manifest_id"):
        validate_oof_subject_coverage(
            table,
            plan,
            contrast_contexts=_contexts(plan),
        )


def test_each_fold_must_contain_every_compared_context() -> None:
    plan, table = _table()
    first_fold = table["fold_id"].iloc[0]
    table = table.loc[
        ~(table["fold_id"].eq(first_fold) & table["context"].eq("stim"))
    ].copy()

    with pytest.raises(ValueError, match="context coverage"):
        validate_oof_subject_coverage(
            table,
            plan,
            contrast_contexts=_contexts(plan),
        )


def test_each_fold_must_contain_every_manifest_contrast() -> None:
    plan, table = _table(multiple_contrasts=True)
    fold = plan.folds[0]
    missing_contrast = fold.contrast_ids[0]
    table = table.loc[
        ~(table["fold_id"].eq(fold.fold_id) & table["contrast_id"].eq(missing_contrast))
    ].copy()

    with pytest.raises(ValueError, match="contrast coverage does not match"):
        validate_oof_subject_coverage(
            table,
            plan,
            contrast_contexts=_contexts(plan),
        )


def test_each_fold_contrast_must_cover_every_test_subject() -> None:
    plan, table = _table(multiple_contrasts=True)
    fold = plan.folds[0]
    contrast_id = fold.contrast_ids[0]
    missing_subject = fold.test_subject_ids[0]
    table = table.loc[
        ~(
            table["fold_id"].eq(fold.fold_id)
            & table["contrast_id"].eq(contrast_id)
            & table["subject_id"].eq(missing_subject)
        )
    ].copy()

    with pytest.raises(ValueError, match="test subject coverage mismatch"):
        validate_oof_subject_coverage(
            table,
            plan,
            contrast_contexts=_contexts(plan),
        )
