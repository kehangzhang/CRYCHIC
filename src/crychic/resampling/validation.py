"""Audits for merged out-of-fold sample score tables."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id

from .contracts import SubjectFoldPlan


def _context_key(value: object) -> str:
    return str(canonical_json(value))


@dataclass(frozen=True, slots=True, kw_only=True)
class OOFCoverageAudit:
    """Successful exact subject/fold/common-functional coverage audit."""

    fold_plan_id: str
    subject_ids: tuple[str, ...]
    fold_ids: tuple[str, ...]
    contrast_ids: tuple[str, ...]
    n_rows: int
    audit_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in ("fold_plan_id",):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("subject_ids", "fold_ids", "contrast_ids"):
            values = tuple(getattr(self, field_name))
            if not values or any(not value for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique values")
            object.__setattr__(self, field_name, tuple(sorted(values)))
        if isinstance(self.n_rows, bool) or not isinstance(self.n_rows, int):
            raise ValueError("n_rows must be an integer")
        if self.n_rows < len(self.subject_ids):
            raise ValueError("OOF audit must contain at least one row per subject")
        payload = {
            "contrast_ids": list(self.contrast_ids),
            "fold_ids": list(self.fold_ids),
            "fold_plan_id": self.fold_plan_id,
            "n_rows": self.n_rows,
            "subject_ids": list(self.subject_ids),
        }
        object.__setattr__(self, "audit_id", stable_id("oof_coverage_audit", payload))

    def _require_intact(self) -> None:
        """Reject forced mutation of persisted OOF coverage provenance."""

        try:
            repeated = OOFCoverageAudit(
                fold_plan_id=self.fold_plan_id,
                subject_ids=self.subject_ids,
                fold_ids=self.fold_ids,
                contrast_ids=self.contrast_ids,
                n_rows=self.n_rows,
            )
            valid = (
                isinstance(self.subject_ids, tuple)
                and isinstance(self.fold_ids, tuple)
                and isinstance(self.contrast_ids, tuple)
                and self.fold_plan_id == repeated.fold_plan_id
                and self.subject_ids == repeated.subject_ids
                and self.fold_ids == repeated.fold_ids
                and self.contrast_ids == repeated.contrast_ids
                and self.n_rows == repeated.n_rows
                and self.audit_id == repeated.audit_id
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "OOF coverage audit failed integrity validation",
                code="oof_coverage_audit_integrity_violation",
                field="audit_id",
                remediation="Recompute coverage from the intact fold plan and rows",
            ) from error
        if not valid:
            raise ContractError(
                "OOF coverage audit failed integrity validation",
                code="oof_coverage_audit_integrity_violation",
                field="audit_id",
                remediation="Recompute coverage from the intact fold plan and rows",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "audit_id": self.audit_id,
            "fold_plan_id": self.fold_plan_id,
            "subject_ids": list(self.subject_ids),
            "fold_ids": list(self.fold_ids),
            "contrast_ids": list(self.contrast_ids),
            "n_rows": self.n_rows,
            "coverage_complete": True,
            "common_functional_validated": True,
        }


def validate_oof_subject_coverage(
    table: pd.DataFrame,
    plan: SubjectFoldPlan,
    *,
    contrast_contexts: Mapping[str, Sequence[Hashable]],
    subject_column: str = "subject_id",
    fold_column: str = "fold_id",
    contrast_id_column: str = "contrast_id",
    context_column: str = "context",
    functional_status_column: str = "functional_status",
    scoring_function_column: str = "scoring_function_id",
    model_manifest_column: str = "model_manifest_id",
) -> OOFCoverageAudit:
    """Reject leakage, incomplete OOF coverage, or mixed fold functionals."""

    required = {
        subject_column,
        fold_column,
        contrast_id_column,
        context_column,
        functional_status_column,
        scoring_function_column,
        model_manifest_column,
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"OOF score table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("OOF score table must not be empty")
    if table.loc[:, list(required)].isna().any().any():
        raise ValueError("OOF provenance and assignment fields must not be missing")
    if set(table[functional_status_column].astype(str)) != {"out_of_fold"}:
        raise ValueError("OOF score rows must have functional_status='out_of_fold'")

    folds_by_id = {fold.fold_id: fold for fold in plan.folds}
    observed_fold_ids = set(table[fold_column].astype(str))
    unknown_folds = observed_fold_ids.difference(folds_by_id)
    if unknown_folds:
        raise ValueError(
            f"OOF score table contains unknown fold IDs: {sorted(unknown_folds)}"
        )
    missing_folds = set(folds_by_id).difference(observed_fold_ids)
    if missing_folds:
        raise ValueError(f"OOF score table is missing folds: {sorted(missing_folds)}")

    subject_fold_sets: dict[str, set[str]] = {}
    for row in (
        table.loc[:, [subject_column, fold_column]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        subject, fold_id = map(str, row)
        fold = folds_by_id[fold_id]
        if subject in fold.train_subject_ids:
            raise ValueError(
                f"OOF score leaks training subject {subject!r} into fold {fold_id!r}"
            )
        if subject not in fold.test_subject_ids:
            raise ValueError(
                f"subject {subject!r} is not assigned to test fold {fold_id!r}"
            )
        subject_fold_sets.setdefault(subject, set()).add(fold_id)
    repeated = {
        subject: sorted(folds)
        for subject, folds in subject_fold_sets.items()
        if len(folds) != 1
    }
    if repeated:
        raise ValueError(f"subjects occur in multiple OOF folds: {repeated}")
    observed_subjects = set(subject_fold_sets)
    expected_subjects = set(plan.subject_ids)
    if observed_subjects != expected_subjects:
        missing_subjects = sorted(expected_subjects.difference(observed_subjects))
        extra_subjects = sorted(observed_subjects.difference(expected_subjects))
        raise ValueError(
            "OOF subject coverage mismatch; "
            f"missing={missing_subjects}, extra={extra_subjects}"
        )

    declared_contexts = {
        str(contrast_id): {_context_key(value) for value in contexts}
        for contrast_id, contexts in contrast_contexts.items()
    }
    if any(not contexts for contexts in declared_contexts.values()):
        raise ValueError("every contrast must declare at least one context")
    expected_contrasts = {
        str(contrast_id) for fold in plan.folds for contrast_id in fold.contrast_ids
    }
    if set(declared_contexts) != expected_contrasts:
        missing_contrasts = sorted(expected_contrasts.difference(declared_contexts))
        extra_contrasts = sorted(set(declared_contexts).difference(expected_contrasts))
        raise ValueError(
            "contrast_contexts must exactly match fold manifests; "
            f"missing={missing_contrasts}, extra={extra_contrasts}"
        )
    observed_contrasts = set(table[contrast_id_column].astype(str))
    if observed_contrasts != set(declared_contexts):
        raise ValueError("OOF contrast IDs must exactly match contrast_contexts")
    for group_fold_id, fold_table in table.groupby(
        fold_column, sort=False, observed=True
    ):
        fold = folds_by_id[str(group_fold_id)]
        fold_contrasts = set(fold_table[contrast_id_column].astype(str))
        expected_fold_contrasts = set(fold.contrast_ids)
        if fold_contrasts != expected_fold_contrasts:
            missing_contrasts = sorted(
                expected_fold_contrasts.difference(fold_contrasts)
            )
            extra_contrasts = sorted(fold_contrasts.difference(expected_fold_contrasts))
            raise ValueError(
                f"fold {group_fold_id!r} contrast coverage does not match its "
                f"manifest; missing={missing_contrasts}, extra={extra_contrasts}"
            )
        for contrast_id in sorted(expected_fold_contrasts):
            group = fold_table.loc[
                fold_table[contrast_id_column].astype(str).eq(contrast_id)
            ]
            observed_test_subjects = set(group[subject_column].astype(str))
            expected_test_subjects = set(fold.test_subject_ids)
            if observed_test_subjects != expected_test_subjects:
                missing_subjects = sorted(
                    expected_test_subjects.difference(observed_test_subjects)
                )
                extra_subjects = sorted(
                    observed_test_subjects.difference(expected_test_subjects)
                )
                raise ValueError(
                    f"fold {group_fold_id!r}, contrast {contrast_id!r} test subject "
                    f"coverage mismatch; missing={missing_subjects}, "
                    f"extra={extra_subjects}"
                )
            for column in (scoring_function_column, model_manifest_column):
                if group[column].astype(str).nunique(dropna=False) != 1:
                    raise ValueError(
                        f"fold {group_fold_id!r}, contrast {contrast_id!r} uses "
                        "multiple "
                        f"{column} values"
                    )
            observed_contexts = {
                _context_key(value) for value in group[context_column].tolist()
            }
            expected_contexts = declared_contexts[str(contrast_id)]
            if observed_contexts != expected_contexts:
                raise ValueError(
                    f"fold {group_fold_id!r}, contrast {contrast_id!r} context "
                    "coverage does not match its registered comparison"
                )
    return OOFCoverageAudit(
        fold_plan_id=plan.plan_id,
        subject_ids=tuple(observed_subjects),
        fold_ids=tuple(observed_fold_ids),
        contrast_ids=tuple(observed_contrasts),
        n_rows=len(table),
    )


__all__ = ["OOFCoverageAudit", "validate_oof_subject_coverage"]
