"""Deterministic estimability-aware subject fold planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from crychic.core import SeedLineage, canonical_json, stable_id
from crychic.design import ContrastSpec, audit_sample_design, canonical_context

from .contracts import (
    FoldEstimabilityResult,
    FoldManifest,
    FoldPlanningError,
    SubjectFoldPlan,
)


class FoldEstimabilityChecker(Protocol):
    """Minimum training-only design audit required by the fold planner."""

    @property
    def contrast_ids(self) -> tuple[str, ...]: ...

    @property
    def required_columns(self) -> tuple[str, ...]: ...

    def __call__(self, training_metadata: pd.DataFrame) -> FoldEstimabilityResult: ...


def _plain(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _typed_key(value: object) -> str:
    plain = _plain(value)
    return canonical_json(
        {
            "type": f"{type(plain).__module__}.{type(plain).__qualname__}",
            "value": plain,
        }
    )


def _context_label(row: Mapping[object, object], context_keys: tuple[str, ...]) -> str:
    return canonical_json(dict(canonical_context(row, context_keys)))


def _sample_table(
    metadata: pd.DataFrame,
    *,
    sample_key: str,
    subject_key: str,
    context_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
    additional_keys: tuple[str, ...],
) -> pd.DataFrame:
    required = (
        sample_key,
        subject_key,
        *context_keys,
        *strata_keys,
        *additional_keys,
    )
    missing = set(required).difference(metadata.columns)
    if missing:
        raise ValueError(f"sample metadata is missing fields: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("sample metadata must not be empty")
    table = metadata.copy(deep=True)
    if table.isna().any().any():
        raise ValueError("fold-planning fields must not contain missing values")
    subjects = table[subject_key]
    if any(not isinstance(value, str) or not value.strip() for value in subjects):
        raise ValueError("subject identifiers must be non-empty strings")
    table[subject_key] = subjects.astype(str).str.strip()

    for sample_id, group in table.groupby(sample_key, sort=False, dropna=False):
        for field_name in (
            subject_key,
            *context_keys,
            *strata_keys,
            *additional_keys,
        ):
            keys = {_typed_key(value) for value in group[field_name].tolist()}
            if len(keys) != 1:
                raise ValueError(
                    f"sample {sample_id!r} maps to multiple {field_name!r} values"
                )
    table = table.drop_duplicates(sample_key, keep="first")
    table["__sample_sort_key"] = [
        _typed_key(value) for value in table[sample_key].tolist()
    ]
    table = table.sort_values("__sample_sort_key", kind="stable").drop(
        columns="__sample_sort_key"
    )
    return table.reset_index(drop=True)


def _subject_strata(
    table: pd.DataFrame,
    *,
    subject_key: str,
    context_keys: tuple[str, ...],
    strata_keys: tuple[str, ...],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for subject, group in table.groupby(subject_key, sort=False, observed=True):
        subject_id = str(subject)
        strata: list[tuple[str, str]] = []
        for field_name in strata_keys:
            values = {_typed_key(value) for value in group[field_name].tolist()}
            if len(values) != 1:
                raise ValueError(
                    f"stratification field {field_name!r} must be immutable within "
                    f"subject {subject_id!r}"
                )
            strata.append((field_name, next(iter(values))))
        contexts = tuple(
            sorted(
                {
                    _context_label(row, context_keys)
                    for row in group.to_dict(orient="records")
                }
            )
        )
        result[subject_id] = canonical_json(
            {"contexts": list(contexts), "strata": strata}
        )
    return result


def _assign_subjects(
    subject_strata: Mapping[str, str],
    *,
    n_splits: int,
    lineage: SeedLineage,
) -> tuple[tuple[str, ...], ...]:
    grouped: dict[str, list[str]] = {}
    for subject, stratum in subject_strata.items():
        grouped.setdefault(stratum, []).append(subject)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for stratum in sorted(grouped):
        stratum_id = stable_id("fold_stratum", {"stratum": stratum})
        stratum_lineage = lineage.derive("stratum", stratum_id)
        ordered = sorted(
            grouped[stratum],
            key=lambda subject: stable_id(
                "fold_subject_order",
                {"seed": stratum_lineage.seed, "subject_id": subject},
            ),
        )
        offset = stratum_lineage.seed % n_splits
        for index, subject in enumerate(ordered):
            folds[(offset + index) % n_splits].append(subject)
    return tuple(tuple(sorted(values)) for values in folds)


def _context_support(
    table: pd.DataFrame,
    *,
    subject_key: str,
    context_keys: tuple[str, ...],
) -> dict[str, int]:
    labels = [
        _context_label(row, context_keys) for row in table.to_dict(orient="records")
    ]
    working = pd.DataFrame(
        {"context": labels, "subject": table[subject_key].astype(str).tolist()}
    ).drop_duplicates()
    counts = working.groupby("context", sort=True, observed=True)["subject"].nunique()
    return {str(context): int(value) for context, value in counts.items()}


def _missing_support_reason(
    support: Mapping[str, int],
    *,
    required_contexts: tuple[str, ...],
    minimum: int,
    scope: str,
) -> str | None:
    failed = [
        context for context in required_contexts if support.get(context, 0) < minimum
    ]
    if not failed:
        return None
    return f"insufficient_{scope}_context_support:" + ",".join(failed)


@dataclass(frozen=True, slots=True, kw_only=True)
class DesignFoldChecker:
    """Rebuild and audit the declared design using training samples only."""

    context_keys: tuple[str, ...]
    covariates: tuple[str, ...]
    formula: str
    contrasts: tuple[ContrastSpec, ...]
    categorical_covariates: tuple[str, ...] = ()
    sample_key: str = "sample_id"

    def __post_init__(self) -> None:
        categorical_covariates = tuple(self.categorical_covariates)
        if not self.context_keys:
            raise ValueError("context_keys must not be empty")
        if len(set((*self.context_keys, *self.covariates))) != len(
            (*self.context_keys, *self.covariates)
        ):
            raise ValueError("context and covariate fields must be unique")
        if not self.formula.strip():
            raise ValueError("formula must be a non-empty string")
        if not self.contrasts:
            raise ValueError("contrasts must not be empty")
        if len(set(categorical_covariates)) != len(categorical_covariates):
            raise ValueError("categorical_covariates must contain unique fields")
        unknown_categorical = set(categorical_covariates).difference(self.covariates)
        if unknown_categorical:
            raise ValueError(
                "categorical_covariates must be declared covariates; unknown="
                f"{sorted(unknown_categorical)}"
            )
        object.__setattr__(self, "categorical_covariates", categorical_covariates)

    @property
    def contrast_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                stable_id("contrast", contrast.to_dict()) for contrast in self.contrasts
            )
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((self.sample_key, *self.context_keys, *self.covariates))
        )

    def __call__(self, training_metadata: pd.DataFrame) -> FoldEstimabilityResult:
        audit = audit_sample_design(
            training_metadata,
            context_keys=self.context_keys,
            covariates=self.covariates,
            categorical_covariates=self.categorical_covariates,
            formula=self.formula,
            sample_key=self.sample_key,
        )
        matrix_payload = {
            "column_names": list(audit.column_names),
            "categorical_covariates": list(audit.categorical_covariates),
            "context_nodes": list(audit.context_nodes),
            "formula": audit.formula,
            "matrix": audit.design_matrix.to_numpy(dtype=float).tolist(),
            "sample_ids": [_typed_key(value) for value in audit.sample_ids],
        }
        design_matrix_id = stable_id("design_matrix", matrix_payload)
        contrast_table = audit.contrast_table(self.contrasts)
        failed_contrasts = sorted(
            contrast_table.loc[~contrast_table["estimable"], "contrast"]
            .astype(str)
            .tolist()
        )
        if not audit.ready:
            reason = "fold_design_not_estimable:" + ",".join(audit.reason_codes)
        elif failed_contrasts:
            reason = "fold_contrast_not_estimable:" + ",".join(failed_contrasts)
        else:
            reason = None
        return FoldEstimabilityResult(
            design_matrix_id=design_matrix_id,
            contrast_ids=self.contrast_ids,
            estimable=reason is None,
            reason_code=reason,
        )


def _candidate_folds(
    table: pd.DataFrame,
    *,
    subject_key: str,
    context_keys: tuple[str, ...],
    subject_strata: Mapping[str, str],
    n_splits: int,
    requested_n_splits: int,
    repeat_id: str,
    seed_lineage: SeedLineage,
    min_train_subjects_per_context: int,
    min_test_subjects_per_context: int,
    checker: FoldEstimabilityChecker,
) -> tuple[FoldManifest, ...]:
    subject_ids = tuple(sorted(subject_strata))
    required_contexts = tuple(
        sorted(
            _context_support(table, subject_key=subject_key, context_keys=context_keys)
        )
    )
    test_sets = _assign_subjects(
        subject_strata,
        n_splits=n_splits,
        lineage=seed_lineage.derive("crossfit", repeat_id, f"k={n_splits}"),
    )
    manifests: list[FoldManifest] = []
    for fold_index, test_subjects in enumerate(test_sets):
        train_subjects = tuple(
            subject for subject in subject_ids if subject not in set(test_subjects)
        )
        train = table.loc[table[subject_key].isin(train_subjects)].copy()
        test = table.loc[table[subject_key].isin(test_subjects)].copy()
        train_support = _context_support(
            train, subject_key=subject_key, context_keys=context_keys
        )
        test_support = _context_support(
            test, subject_key=subject_key, context_keys=context_keys
        )
        for context in required_contexts:
            train_support.setdefault(context, 0)
            test_support.setdefault(context, 0)
        reason = None
        if not train_subjects or not test_subjects:
            reason = "empty_subject_partition"
        if reason is None:
            reason = _missing_support_reason(
                train_support,
                required_contexts=required_contexts,
                minimum=min_train_subjects_per_context,
                scope="training",
            )
        if reason is None:
            reason = _missing_support_reason(
                test_support,
                required_contexts=required_contexts,
                minimum=min_test_subjects_per_context,
                scope="test",
            )
        if reason is None:
            check = checker(train)
            reason = check.reason_code
        else:
            check = FoldEstimabilityResult(
                design_matrix_id=stable_id(
                    "design_matrix_unavailable",
                    {
                        "fold_index": fold_index,
                        "n_splits": n_splits,
                        "reason_code": reason,
                        "train_subject_ids": list(train_subjects),
                    },
                ),
                contrast_ids=checker.contrast_ids,
                estimable=False,
                reason_code=reason,
            )
        fold_lineage = seed_lineage.derive(
            "crossfit", repeat_id, f"k={n_splits}", f"fold={fold_index}"
        )
        manifests.append(
            FoldManifest(
                repeat_id=repeat_id,
                train_subject_ids=train_subjects,
                test_subject_ids=test_subjects,
                design_matrix_id=check.design_matrix_id,
                contrast_ids=check.contrast_ids,
                context_support=train_support,
                test_context_support=test_support,
                estimable=reason is None,
                reason_code=reason,
                seed_lineage=fold_lineage,
                requested_n_splits=requested_n_splits,
                effective_n_splits=n_splits,
                fold_index=fold_index,
            )
        )
    return tuple(manifests)


def plan_subject_folds(
    sample_metadata: pd.DataFrame,
    *,
    design_checker: FoldEstimabilityChecker,
    subject_key: str = "subject_id",
    sample_key: str = "sample_id",
    context_keys: Sequence[str] = ("context",),
    strata_keys: Sequence[str] = (),
    allowed_n_splits: Sequence[int] = (5, 4, 3, 2),
    min_train_subjects_per_context: int = 2,
    min_test_subjects_per_context: int = 1,
    repeat_id: str = "repeat-0",
    seed_lineage: SeedLineage | None = None,
) -> SubjectFoldPlan:
    """Select the first predeclared K whose every subject fold is estimable."""

    allowed = tuple(allowed_n_splits)
    if (
        not allowed
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in allowed
        )
        or tuple(sorted(set(allowed), reverse=True)) != allowed
    ):
        raise ValueError(
            "allowed_n_splits must be unique integers >= 2 in descending order"
        )
    for field_name, value in (
        ("min_train_subjects_per_context", min_train_subjects_per_context),
        ("min_test_subjects_per_context", min_test_subjects_per_context),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be an integer >= 1")
    if not isinstance(repeat_id, str) or not repeat_id.strip():
        raise ValueError("repeat_id must be a non-empty string")
    contexts = tuple(context_keys)
    strata = tuple(strata_keys)
    if not contexts:
        raise ValueError("context_keys must not be empty")
    if len(set((*contexts, *strata))) != len((*contexts, *strata)):
        raise ValueError("context_keys and strata_keys must be unique and disjoint")
    lineage = seed_lineage or SeedLineage(0)
    table = _sample_table(
        sample_metadata,
        sample_key=sample_key,
        subject_key=subject_key,
        context_keys=contexts,
        strata_keys=strata,
        additional_keys=design_checker.required_columns,
    )
    subject_strata = _subject_strata(
        table,
        subject_key=subject_key,
        context_keys=contexts,
        strata_keys=strata,
    )
    if len(subject_strata) < 2:
        raise FoldPlanningError(
            "cross-fitting requires at least two subjects",
            code="insufficient_crossfit_subjects",
            field=subject_key,
            remediation="Provide at least two independent subject blocks",
        )

    rejected: list[tuple[int, str]] = []
    requested = allowed[0]
    for n_splits in allowed:
        if n_splits > len(subject_strata):
            rejected.append((n_splits, "fold_count_exceeds_subject_count"))
            continue
        folds = _candidate_folds(
            table,
            subject_key=subject_key,
            context_keys=contexts,
            subject_strata=subject_strata,
            n_splits=n_splits,
            requested_n_splits=requested,
            repeat_id=repeat_id.strip(),
            seed_lineage=lineage,
            min_train_subjects_per_context=min_train_subjects_per_context,
            min_test_subjects_per_context=min_test_subjects_per_context,
            checker=design_checker,
        )
        failed = [
            f"fold_{fold.fold_index}:{fold.reason_code}"
            for fold in folds
            if not fold.estimable
        ]
        if failed:
            rejected.append((n_splits, "|".join(failed)))
            continue
        reduction_reason = (
            None
            if n_splits == requested
            else f"requested_{requested}_not_estimable_selected_{n_splits}"
        )
        return SubjectFoldPlan(
            repeat_id=repeat_id.strip(),
            requested_n_splits=requested,
            effective_n_splits=n_splits,
            allowed_n_splits=allowed,
            subject_ids=tuple(subject_strata),
            folds=folds,
            seed_lineage=lineage,
            reduction_reason_code=reduction_reason,
            rejected_candidate_reasons=tuple(rejected),
        )
    reason = ";".join(f"k={value}:{code}" for value, code in rejected)
    raise FoldPlanningError(
        "no predeclared subject fold count is estimable: " + reason,
        code="no_estimable_subject_fold_plan",
        field="allowed_n_splits",
        remediation=(
            "Increase independent subject support or revise the design before fitting"
        ),
    )


__all__ = [
    "DesignFoldChecker",
    "FoldEstimabilityChecker",
    "plan_subject_folds",
]
