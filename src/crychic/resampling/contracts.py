"""Immutable subject-level cross-fitting plans and fold diagnostics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from crychic.core import ContractError, SeedLineage, stable_id


def _names(
    values: tuple[str, ...], *, field_name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if not values and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = tuple(sorted(value.strip() for value in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return normalized


def _support(values: Mapping[str, int], *, field_name: str) -> Mapping[str, int]:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    normalized: dict[str, int] = {}
    for context, count in values.items():
        if not isinstance(context, str) or not context:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field_name} counts must be non-negative integers")
        normalized[context] = count
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True, kw_only=True)
class FoldEstimabilityResult:
    """Design and contrast audit result for one candidate training fold."""

    design_matrix_id: str
    contrast_ids: tuple[str, ...]
    estimable: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.design_matrix_id, str)
            or not self.design_matrix_id.strip()
        ):
            raise ValueError("design_matrix_id must be a non-empty string")
        contrast_ids = _names(tuple(self.contrast_ids), field_name="contrast_ids")
        if self.estimable == (self.reason_code is not None):
            raise ValueError(
                "reason_code must be present exactly when a fold is not estimable"
            )
        if self.reason_code is not None and not self.reason_code.strip():
            raise ValueError("reason_code must be non-empty when present")
        object.__setattr__(self, "design_matrix_id", self.design_matrix_id.strip())
        object.__setattr__(self, "contrast_ids", contrast_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class FoldManifest:
    """One immutable subject-block training/test split."""

    repeat_id: str
    train_subject_ids: tuple[str, ...]
    test_subject_ids: tuple[str, ...]
    design_matrix_id: str
    contrast_ids: tuple[str, ...]
    context_support: Mapping[str, int]
    test_context_support: Mapping[str, int]
    estimable: bool
    reason_code: str | None
    seed_lineage: SeedLineage
    requested_n_splits: int
    effective_n_splits: int
    fold_index: int
    fold_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.repeat_id, str) or not self.repeat_id.strip():
            raise ValueError("repeat_id must be a non-empty string")
        train = _names(
            tuple(self.train_subject_ids),
            field_name="train_subject_ids",
            allow_empty=not self.estimable,
        )
        test = _names(
            tuple(self.test_subject_ids),
            field_name="test_subject_ids",
            allow_empty=not self.estimable,
        )
        overlap = set(train).intersection(test)
        if overlap:
            raise ValueError(
                f"training and test subjects must be disjoint: {sorted(overlap)}"
            )
        if (
            not isinstance(self.design_matrix_id, str)
            or not self.design_matrix_id.strip()
        ):
            raise ValueError("design_matrix_id must be a non-empty string")
        contrasts = _names(tuple(self.contrast_ids), field_name="contrast_ids")
        train_support = _support(self.context_support, field_name="context_support")
        test_support = _support(
            self.test_context_support, field_name="test_context_support"
        )
        if set(train_support) != set(test_support):
            raise ValueError("training and test context support must use the same keys")
        for field_name, value in (
            ("requested_n_splits", self.requested_n_splits),
            ("effective_n_splits", self.effective_n_splits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{field_name} must be an integer >= 2")
        if self.effective_n_splits > self.requested_n_splits:
            raise ValueError("effective_n_splits cannot exceed requested_n_splits")
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < self.effective_n_splits
        ):
            raise ValueError("fold_index must be within the effective fold count")
        if self.estimable == (self.reason_code is not None):
            raise ValueError(
                "reason_code must be present exactly when a fold is not estimable"
            )
        if self.reason_code is not None and not self.reason_code.strip():
            raise ValueError("reason_code must be non-empty when present")
        if self.estimable and (not train or not test):
            raise ValueError("an estimable fold requires non-empty train and test sets")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")

        payload = {
            "contrast_ids": list(contrasts),
            "design_matrix_id": self.design_matrix_id.strip(),
            "effective_n_splits": self.effective_n_splits,
            "fold_index": self.fold_index,
            "repeat_id": self.repeat_id.strip(),
            "seed_lineage": self.seed_lineage.to_dict(),
            "test_subject_ids": list(test),
            "train_subject_ids": list(train),
        }
        object.__setattr__(self, "repeat_id", self.repeat_id.strip())
        object.__setattr__(self, "train_subject_ids", train)
        object.__setattr__(self, "test_subject_ids", test)
        object.__setattr__(self, "design_matrix_id", self.design_matrix_id.strip())
        object.__setattr__(self, "contrast_ids", contrasts)
        object.__setattr__(self, "context_support", train_support)
        object.__setattr__(self, "test_context_support", test_support)
        object.__setattr__(self, "fold_id", stable_id("subject_fold", payload))

    def to_dict(self) -> dict[str, object]:
        """Return a manifest-ready fold representation."""

        return {
            "fold_id": self.fold_id,
            "repeat_id": self.repeat_id,
            "train_subject_ids": list(self.train_subject_ids),
            "test_subject_ids": list(self.test_subject_ids),
            "design_matrix_id": self.design_matrix_id,
            "contrast_ids": list(self.contrast_ids),
            "context_support": dict(self.context_support),
            "test_context_support": dict(self.test_context_support),
            "estimable": self.estimable,
            "reason_code": self.reason_code,
            "seed_lineage": self.seed_lineage.to_dict(),
            "requested_n_splits": self.requested_n_splits,
            "effective_n_splits": self.effective_n_splits,
            "fold_index": self.fold_index,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SubjectFoldPlan:
    """Selected fold count plus every rejected pre-fit candidate."""

    repeat_id: str
    requested_n_splits: int
    effective_n_splits: int
    allowed_n_splits: tuple[int, ...]
    subject_ids: tuple[str, ...]
    folds: tuple[FoldManifest, ...]
    seed_lineage: SeedLineage
    reduction_reason_code: str | None = None
    rejected_candidate_reasons: tuple[tuple[int, str], ...] = ()
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        allowed = tuple(self.allowed_n_splits)
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
        if self.requested_n_splits != allowed[0]:
            raise ValueError("requested_n_splits must equal the first allowed value")
        if self.effective_n_splits not in allowed:
            raise ValueError(
                "effective_n_splits must be predeclared in allowed_n_splits"
            )
        subjects = _names(tuple(self.subject_ids), field_name="subject_ids")
        folds = tuple(self.folds)
        if len(folds) != self.effective_n_splits:
            raise ValueError("fold count must equal effective_n_splits")
        if any(not fold.estimable for fold in folds):
            raise ValueError(
                "a selected SubjectFoldPlan may contain only estimable folds"
            )
        if any(fold.repeat_id != self.repeat_id for fold in folds):
            raise ValueError("every fold must use the plan repeat_id")
        if any(fold.effective_n_splits != self.effective_n_splits for fold in folds):
            raise ValueError("every fold must use the selected effective_n_splits")
        expected = set(subjects)
        test_counts = dict.fromkeys(subjects, 0)
        for fold in folds:
            if set(fold.train_subject_ids).union(fold.test_subject_ids) != expected:
                raise ValueError(
                    "every fold must partition the complete subject universe"
                )
            for subject in fold.test_subject_ids:
                test_counts[subject] += 1
        if any(count != 1 for count in test_counts.values()):
            raise ValueError("every subject must occur in exactly one test fold")
        reduced = self.effective_n_splits != self.requested_n_splits
        if reduced != (self.reduction_reason_code is not None):
            raise ValueError(
                "reduction_reason_code must be present exactly when K is reduced"
            )
        rejected = tuple(self.rejected_candidate_reasons)
        rejected_counts = tuple(value for value, _ in rejected)
        if len(set(rejected_counts)) != len(rejected_counts):
            raise ValueError("rejected candidate fold counts must be unique")
        if any(
            value not in allowed or value <= self.effective_n_splits
            for value in rejected_counts
        ):
            raise ValueError(
                "rejected candidates must be predeclared fold counts above selected K"
            )
        if any(not reason for _, reason in rejected):
            raise ValueError("rejected candidate reasons must be non-empty")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        payload = {
            "allowed_n_splits": list(allowed),
            "effective_n_splits": self.effective_n_splits,
            "fold_ids": [fold.fold_id for fold in folds],
            "repeat_id": self.repeat_id,
            "seed_lineage": self.seed_lineage.to_dict(),
            "subject_ids": list(subjects),
        }
        object.__setattr__(self, "allowed_n_splits", allowed)
        object.__setattr__(self, "subject_ids", subjects)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "rejected_candidate_reasons", rejected)
        object.__setattr__(self, "plan_id", stable_id("subject_fold_plan", payload))

    def __iter__(self) -> Iterator[FoldManifest]:
        return iter(self.folds)

    def to_dict(self) -> dict[str, object]:
        """Return a persisted plan with fold and reduction provenance."""

        return {
            "plan_id": self.plan_id,
            "repeat_id": self.repeat_id,
            "requested_n_splits": self.requested_n_splits,
            "effective_n_splits": self.effective_n_splits,
            "allowed_n_splits": list(self.allowed_n_splits),
            "subject_ids": list(self.subject_ids),
            "seed_lineage": self.seed_lineage.to_dict(),
            "reduction_reason_code": self.reduction_reason_code,
            "rejected_candidate_reasons": [
                {"n_splits": n_splits, "reason_code": reason}
                for n_splits, reason in self.rejected_candidate_reasons
            ],
            "folds": [fold.to_dict() for fold in self.folds],
        }


class FoldPlanningError(ContractError):
    """Raised when no predeclared subject-level fold count is estimable."""


__all__ = [
    "FoldEstimabilityResult",
    "FoldManifest",
    "FoldPlanningError",
    "SubjectFoldPlan",
]
