"""Subject-block exchangeability maps, permutations, and bootstrap plans."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from crychic.core import ContractError, SeedLineage, canonical_json, stable_id


class ExchangeabilityDesign(StrEnum):
    """Reviewed subject allocation that determines legal null operations."""

    INDEPENDENT = "independent_subject_context_labels_v1"
    PAIRED_BINARY = "paired_binary_within_subject_contexts_v1"


class ContextPermutationOperation(StrEnum):
    """Factor-specific context operations supported by the first formal planner."""

    BETWEEN_SUBJECT_WITHIN_STRATUM = (
        "between_subject_context_label_permutation_within_stratum_v1"
    )
    WITHIN_SUBJECT_BINARY_SWAP = "within_subject_binary_context_swap_v1"


def _names(
    values: Sequence[str], *, field_name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(values)
    if not result and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in result
    ):
        raise ValueError(f"{field_name} must contain canonical non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _scalar(value: object, *, field_name: str) -> Hashable:
    resolved = value.item() if isinstance(value, np.generic) else value
    if resolved is None or resolved is pd.NA or resolved is pd.NaT:
        raise ValueError(f"{field_name} cannot contain missing values")
    if isinstance(resolved, float) and not math.isfinite(resolved):
        raise ValueError(f"{field_name} cannot contain non-finite values")
    if not isinstance(resolved, Hashable):
        raise TypeError(f"{field_name} values must be hashable")
    try:
        canonical_json(resolved)
    except TypeError as error:
        raise TypeError(
            f"{field_name} values must be canonical JSON scalars"
        ) from error
    return resolved


def _context_key(context: tuple[tuple[str, Hashable], ...]) -> str:
    return canonical_json(dict(context))


@dataclass(frozen=True, slots=True, kw_only=True)
class ExchangeabilityMap:
    """Frozen sample-to-subject/context blocks and their only legal operation."""

    exchangeability_id: str
    design: ExchangeabilityDesign
    operation: ContextPermutationOperation
    sample_key: str
    subject_key: str
    context_keys: tuple[str, ...]
    strata_keys: tuple[str, ...]
    immutable_covariates: tuple[str, ...]
    sample_ids: tuple[str, ...]
    sample_subject_ids: tuple[str, ...]
    sample_contexts: tuple[tuple[tuple[str, Hashable], ...], ...]
    subject_ids: tuple[str, ...]
    subject_strata: tuple[tuple[Hashable, ...], ...]
    context_nodes: tuple[tuple[tuple[str, Hashable], ...], ...]

    def __post_init__(self) -> None:
        design = ExchangeabilityDesign(self.design)
        operation = ContextPermutationOperation(self.operation)
        if (design is ExchangeabilityDesign.INDEPENDENT) != (
            operation is ContextPermutationOperation.BETWEEN_SUBJECT_WITHIN_STRATUM
        ):
            raise ValueError("exchangeability design and operation do not match")
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "operation", operation)
        samples = _names(self.sample_ids, field_name="sample_ids")
        subjects = _names(self.subject_ids, field_name="subject_ids")
        if tuple(sorted(samples)) != samples or tuple(sorted(subjects)) != subjects:
            raise ValueError("sample_ids and subject_ids must be sorted")
        if (
            len(self.sample_subject_ids) != len(samples)
            or len(self.sample_contexts) != len(samples)
            or len(self.subject_strata) != len(subjects)
        ):
            raise ValueError("exchangeability arrays do not align")
        if set(self.sample_subject_ids) != set(subjects):
            raise ValueError("every subject must own at least one sample block")
        if any(subject not in set(subjects) for subject in self.sample_subject_ids):
            raise ValueError("sample subject is absent from subject_ids")
        contexts = tuple(self.context_nodes)
        if len(contexts) < 2 or len(set(contexts)) != len(contexts):
            raise ValueError("context_nodes must contain at least two unique values")
        if set(self.sample_contexts) != set(contexts):
            raise ValueError("sample contexts must exactly cover context_nodes")
        if any(len(values) != len(self.strata_keys) for values in self.subject_strata):
            raise ValueError("subject_strata must align with strata_keys")
        expected = stable_id(
            "exchangeability_map",
            self._identity_payload(),
            schema_version="1",
        )
        if self.exchangeability_id != expected:
            raise ContractError(
                "Exchangeability map identity is not intact",
                code="exchangeability_map_integrity_violation",
                field="exchangeability_id",
                remediation="Rebuild the map from sample metadata",
            )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "design": self.design.value,
            "operation": self.operation.value,
            "sample_key": self.sample_key,
            "subject_key": self.subject_key,
            "context_keys": list(self.context_keys),
            "strata_keys": list(self.strata_keys),
            "immutable_covariates": list(self.immutable_covariates),
            "sample_ids": list(self.sample_ids),
            "sample_subject_ids": list(self.sample_subject_ids),
            "sample_contexts": [dict(context) for context in self.sample_contexts],
            "subject_ids": list(self.subject_ids),
            "subject_strata": [list(values) for values in self.subject_strata],
            "context_nodes": [dict(context) for context in self.context_nodes],
            "cell_level_resampling_allowed": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "exchangeability_id": self.exchangeability_id,
            **self._identity_payload(),
        }


def build_exchangeability_map(
    sample_metadata: pd.DataFrame,
    *,
    sample_key: str,
    subject_key: str,
    context_keys: Sequence[str],
    strata_keys: Sequence[str] = (),
    immutable_covariates: Sequence[str] = (),
) -> ExchangeabilityMap:
    """Audit metadata and freeze the legal subject-block null operation."""

    if not isinstance(sample_metadata, pd.DataFrame):
        raise TypeError("sample_metadata must be a pandas DataFrame")
    contexts = tuple(sorted(_names(context_keys, field_name="context_keys")))
    strata = tuple(
        sorted(_names(strata_keys, field_name="strata_keys", allow_empty=True))
    )
    immutable = tuple(
        sorted(
            _names(
                immutable_covariates,
                field_name="immutable_covariates",
                allow_empty=True,
            )
        )
    )
    if set(strata).difference(immutable):
        raise ValueError("strata_keys must be declared immutable_covariates")
    required = {sample_key, subject_key, *contexts, *strata, *immutable}
    missing = required.difference(sample_metadata.columns)
    if missing:
        raise ValueError(f"sample metadata is missing fields: {sorted(missing)}")
    table = sample_metadata.loc[:, list(required)].copy()
    if table.isna().any().any():
        raise ValueError("exchangeability fields cannot contain missing values")
    sample_ids = tuple(str(value) for value in table[sample_key])
    subject_ids_by_row = tuple(str(value) for value in table[subject_key])
    if any(not value or value != value.strip() for value in sample_ids):
        raise ValueError("sample IDs must be canonical non-empty strings")
    if any(not value or value != value.strip() for value in subject_ids_by_row):
        raise ValueError("subject IDs must be canonical non-empty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_metadata must contain one row per unique sample")
    order = np.argsort(np.asarray(sample_ids, dtype=object), kind="stable")
    table = table.iloc[order].reset_index(drop=True)
    sample_ids = tuple(str(value) for value in table[sample_key])
    subject_ids_by_row = tuple(str(value) for value in table[subject_key])
    sample_contexts = tuple(
        tuple(
            (key, _scalar(row[key], field_name=key))
            for key in sorted(contexts)
        )
        for _, row in table.iterrows()
    )
    context_nodes = tuple(sorted(set(sample_contexts), key=_context_key))
    if len(context_nodes) < 2:
        raise ContractError(
            "Context permutation requires at least two observed contexts",
            code="exchangeability_single_context",
            field="context_keys",
            remediation="Declare a multi-context analysis",
        )
    subjects = tuple(sorted(set(subject_ids_by_row)))
    rows_by_subject: dict[str, list[int]] = defaultdict(list)
    for row_index, subject in enumerate(subject_ids_by_row):
        rows_by_subject[subject].append(row_index)

    subject_strata: list[tuple[Hashable, ...]] = []
    subject_context_sets: list[frozenset[tuple[tuple[str, Hashable], ...]]] = []
    for subject in subjects:
        indexes = rows_by_subject[subject]
        for covariate in immutable:
            covariate_values = {
                _scalar(table.iloc[index][covariate], field_name=covariate)
                for index in indexes
            }
            if len(covariate_values) != 1:
                raise ContractError(
                    "Immutable covariates must be constant within subject",
                    code="exchangeability_immutable_covariate_varies",
                    field=covariate,
                    remediation="Correct metadata or remove the invalid stratum",
                )
        subject_strata.append(
            tuple(
                _scalar(table.iloc[indexes[0]][key], field_name=key)
                for key in strata
            )
        )
        subject_context_sets.append(
            frozenset(sample_contexts[index] for index in indexes)
        )

    if all(len(values) == 1 for values in subject_context_sets):
        design = ExchangeabilityDesign.INDEPENDENT
        operation = ContextPermutationOperation.BETWEEN_SUBJECT_WITHIN_STRATUM
        stratum_labels: dict[tuple[Hashable, ...], set[object]] = defaultdict(set)
        stratum_sizes: dict[tuple[Hashable, ...], int] = defaultdict(int)
        for context_set, stratum in zip(
            subject_context_sets, subject_strata, strict=True
        ):
            stratum_labels[stratum].add(next(iter(context_set)))
            stratum_sizes[stratum] += 1
        if not any(
            stratum_sizes[key] >= 2 and len(labels) >= 2
            for key, labels in stratum_labels.items()
        ):
            raise ContractError(
                "No stratum contains exchangeable subjects from multiple contexts",
                code="exchangeability_no_permutable_stratum",
                field="strata_keys",
                remediation="Use coarser valid strata or add subject replication",
            )
    elif all(values == frozenset(context_nodes) for values in subject_context_sets):
        if len(context_nodes) != 2:
            raise ContractError(
                "Multi-context within-subject permutation requires an explicit map",
                code="multi_context_exchangeability_not_declared",
                field="context_keys",
                remediation="Declare factor-specific legal operations before testing",
            )
        design = ExchangeabilityDesign.PAIRED_BINARY
        operation = ContextPermutationOperation.WITHIN_SUBJECT_BINARY_SWAP
    else:
        raise ContractError(
            "Mixed paired/unpaired context allocation has no reviewed permutation",
            code="mixed_paired_unpaired_exchangeability_unsupported",
            field="subject_key,context_keys",
            remediation="Use a reviewed mixed-design exchangeability procedure",
        )

    payload = {
        "design": design.value,
        "operation": operation.value,
        "sample_key": sample_key,
        "subject_key": subject_key,
        "context_keys": list(contexts),
        "strata_keys": list(strata),
        "immutable_covariates": list(immutable),
        "sample_ids": list(sample_ids),
        "sample_subject_ids": list(subject_ids_by_row),
        "sample_contexts": [dict(context) for context in sample_contexts],
        "subject_ids": list(subjects),
        "subject_strata": [list(values) for values in subject_strata],
        "context_nodes": [dict(context) for context in context_nodes],
        "cell_level_resampling_allowed": False,
    }
    return ExchangeabilityMap(
        exchangeability_id=stable_id(
            "exchangeability_map", payload, schema_version="1"
        ),
        design=design,
        operation=operation,
        sample_key=sample_key,
        subject_key=subject_key,
        context_keys=contexts,
        strata_keys=strata,
        immutable_covariates=immutable,
        sample_ids=sample_ids,
        sample_subject_ids=subject_ids_by_row,
        sample_contexts=sample_contexts,
        subject_ids=subjects,
        subject_strata=tuple(subject_strata),
        context_nodes=context_nodes,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextPermutationPlan:
    """One subject-block context relabeling with complete sample coverage."""

    permutation_id: str
    exchangeability_id: str
    resample_index: int
    operation: ContextPermutationOperation
    sample_ids: tuple[str, ...]
    permuted_contexts: tuple[tuple[tuple[str, Hashable], ...], ...]
    changed_subject_ids: tuple[str, ...]
    seed_lineage: SeedLineage

    def __post_init__(self) -> None:
        if (
            isinstance(self.resample_index, bool)
            or not isinstance(self.resample_index, int)
            or self.resample_index < 0
        ):
            raise ValueError("resample_index must be a non-negative integer")
        operation = ContextPermutationOperation(self.operation)
        object.__setattr__(self, "operation", operation)
        samples = _names(self.sample_ids, field_name="sample_ids")
        changed = _names(
            self.changed_subject_ids,
            field_name="changed_subject_ids",
            allow_empty=True,
        )
        if len(self.permuted_contexts) != len(samples):
            raise ValueError("permuted_contexts must align with sample_ids")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        expected = stable_id(
            "context_permutation",
            self._identity_payload(),
            schema_version="1",
        )
        if self.permutation_id != expected:
            raise ContractError(
                "Context permutation identity is not intact",
                code="context_permutation_integrity_violation",
                field="permutation_id",
                remediation="Regenerate the plan from its exchangeability map",
            )
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "changed_subject_ids", changed)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "exchangeability_id": self.exchangeability_id,
            "resample_index": self.resample_index,
            "operation": self.operation.value,
            "sample_ids": list(self.sample_ids),
            "permuted_contexts": [
                dict(context) for context in self.permuted_contexts
            ],
            "changed_subject_ids": list(self.changed_subject_ids),
            "seed_lineage": self.seed_lineage.to_dict(),
            "resampling_unit": "subject_block",
        }

    def to_dict(self) -> dict[str, object]:
        return {"permutation_id": self.permutation_id, **self._identity_payload()}


def _subject_rows(exchangeability: ExchangeabilityMap) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(exchangeability.sample_subject_ids):
        rows[subject].append(index)
    return rows


def plan_context_permutations(
    exchangeability: ExchangeabilityMap,
    *,
    n_permutations: int,
    seed_lineage: SeedLineage,
) -> tuple[ContextPermutationPlan, ...]:
    """Generate call-order-independent legal context relabeling plans."""

    if not isinstance(exchangeability, ExchangeabilityMap):
        raise TypeError("exchangeability must be an ExchangeabilityMap")
    if (
        isinstance(n_permutations, bool)
        or not isinstance(n_permutations, int)
        or n_permutations < 1
    ):
        raise ValueError("n_permutations must be an integer >= 1")
    if not isinstance(seed_lineage, SeedLineage):
        raise TypeError("seed_lineage must be a SeedLineage")
    rows_by_subject = _subject_rows(exchangeability)
    stratum_by_subject = dict(
        zip(
            exchangeability.subject_ids,
            exchangeability.subject_strata,
            strict=True,
        )
    )
    plans: list[ContextPermutationPlan] = []
    for resample_index in range(n_permutations):
        child_seed = seed_lineage.derive(
            "context_permutation",
            exchangeability.exchangeability_id,
            f"resample={resample_index}",
        )
        rng = child_seed.python_random()
        permuted = list(exchangeability.sample_contexts)
        changed: set[str] = set()
        if exchangeability.design is ExchangeabilityDesign.INDEPENDENT:
            subjects_by_stratum: dict[tuple[Hashable, ...], list[str]] = defaultdict(
                list
            )
            for subject in exchangeability.subject_ids:
                subjects_by_stratum[stratum_by_subject[subject]].append(subject)
            original_context = {
                subject: exchangeability.sample_contexts[rows_by_subject[subject][0]]
                for subject in exchangeability.subject_ids
            }
            for stratum in sorted(subjects_by_stratum, key=canonical_json):
                subjects = subjects_by_stratum[stratum]
                labels = [original_context[subject] for subject in subjects]
                rng.shuffle(labels)
                for subject, label in zip(subjects, labels, strict=True):
                    if label != original_context[subject]:
                        changed.add(subject)
                    for row_index in rows_by_subject[subject]:
                        permuted[row_index] = label
        else:
            left, right = exchangeability.context_nodes
            for subject in exchangeability.subject_ids:
                if rng.getrandbits(1):
                    changed.add(subject)
                    for row_index in rows_by_subject[subject]:
                        current = exchangeability.sample_contexts[row_index]
                        permuted[row_index] = right if current == left else left
        payload = {
            "exchangeability_id": exchangeability.exchangeability_id,
            "resample_index": resample_index,
            "operation": exchangeability.operation.value,
            "sample_ids": list(exchangeability.sample_ids),
            "permuted_contexts": [dict(context) for context in permuted],
            "changed_subject_ids": sorted(changed),
            "seed_lineage": child_seed.to_dict(),
            "resampling_unit": "subject_block",
        }
        plans.append(
            ContextPermutationPlan(
                permutation_id=stable_id(
                    "context_permutation", payload, schema_version="1"
                ),
                exchangeability_id=exchangeability.exchangeability_id,
                resample_index=resample_index,
                operation=exchangeability.operation,
                sample_ids=exchangeability.sample_ids,
                permuted_contexts=tuple(permuted),
                changed_subject_ids=tuple(sorted(changed)),
                seed_lineage=child_seed,
            )
        )
    return tuple(plans)


def apply_context_permutation(
    metadata: pd.DataFrame,
    exchangeability: ExchangeabilityMap,
    plan: ContextPermutationPlan,
) -> pd.DataFrame:
    """Return a copy with sample-block context labels replaced by one legal plan."""

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    if not isinstance(exchangeability, ExchangeabilityMap):
        raise TypeError("exchangeability must be an ExchangeabilityMap")
    if not isinstance(plan, ContextPermutationPlan):
        raise TypeError("plan must be a ContextPermutationPlan")
    if plan.exchangeability_id != exchangeability.exchangeability_id or (
        plan.sample_ids != exchangeability.sample_ids
    ):
        raise ContractError(
            "Permutation plan does not belong to the exchangeability map",
            code="context_permutation_parent_mismatch",
            field="exchangeability_id,sample_ids",
            remediation="Apply a plan to its exact source sample universe",
        )
    required = {exchangeability.sample_key, *exchangeability.context_keys}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing fields: {sorted(missing)}")
    observed_samples = {str(value) for value in metadata[exchangeability.sample_key]}
    if observed_samples != set(plan.sample_ids):
        raise ContractError(
            "Metadata sample universe differs from the permutation plan",
            code="context_permutation_sample_universe_mismatch",
            field=exchangeability.sample_key,
            remediation="Use the same sample blocks used to construct the map",
        )
    context_by_sample = dict(
        zip(plan.sample_ids, plan.permuted_contexts, strict=True)
    )
    result = metadata.copy(deep=True)
    sample_values = result[exchangeability.sample_key].astype(str)
    for context_key in exchangeability.context_keys:
        result[context_key] = sample_values.map(
            lambda sample_id, key=context_key: dict(context_by_sample[sample_id])[key]
        )
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapSubjectDraw:
    """One source subject copied into one uniquely named bootstrap block."""

    source_subject_id: str
    bootstrap_subject_id: str
    stratum: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        for field_name in ("source_subject_id", "bootstrap_subject_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a canonical non-empty string")
        stratum = tuple(self.stratum)
        for value in stratum:
            _scalar(value, field_name="stratum")
        object.__setattr__(self, "stratum", stratum)


@dataclass(frozen=True, slots=True, kw_only=True)
class SubjectBootstrapPlan:
    """Stratified subject-block draws with replacement."""

    bootstrap_id: str
    exchangeability_id: str
    resample_index: int
    draws: tuple[BootstrapSubjectDraw, ...]
    seed_lineage: SeedLineage

    def __post_init__(self) -> None:
        if (
            isinstance(self.resample_index, bool)
            or not isinstance(self.resample_index, int)
            or self.resample_index < 0
        ):
            raise ValueError("resample_index must be a non-negative integer")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise TypeError("seed_lineage must be a SeedLineage")
        if not self.draws or any(
            not isinstance(draw, BootstrapSubjectDraw) for draw in self.draws
        ):
            raise ValueError("subject bootstrap plan must contain draws")
        if len({draw.bootstrap_subject_id for draw in self.draws}) != len(self.draws):
            raise ValueError("bootstrap subject IDs must be unique")
        expected = stable_id(
            "subject_bootstrap",
            self._identity_payload(),
            schema_version="1",
        )
        if self.bootstrap_id != expected:
            raise ContractError(
                "Subject bootstrap identity is not intact",
                code="subject_bootstrap_integrity_violation",
                field="bootstrap_id",
                remediation="Regenerate the subject-block bootstrap plan",
            )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "exchangeability_id": self.exchangeability_id,
            "resample_index": self.resample_index,
            "draws": [
                {
                    "source_subject_id": draw.source_subject_id,
                    "bootstrap_subject_id": draw.bootstrap_subject_id,
                    "stratum": list(draw.stratum),
                }
                for draw in self.draws
            ],
            "seed_lineage": self.seed_lineage.to_dict(),
            "resampling_unit": "subject_block",
            "with_replacement": True,
        }

    def to_dict(self) -> dict[str, object]:
        return {"bootstrap_id": self.bootstrap_id, **self._identity_payload()}


def plan_subject_bootstraps(
    exchangeability: ExchangeabilityMap,
    *,
    n_bootstraps: int,
    seed_lineage: SeedLineage,
) -> tuple[SubjectBootstrapPlan, ...]:
    """Draw whole subjects within immutable strata, preserving stratum sizes."""

    if not isinstance(exchangeability, ExchangeabilityMap):
        raise TypeError("exchangeability must be an ExchangeabilityMap")
    if (
        isinstance(n_bootstraps, bool)
        or not isinstance(n_bootstraps, int)
        or n_bootstraps < 1
    ):
        raise ValueError("n_bootstraps must be an integer >= 1")
    if not isinstance(seed_lineage, SeedLineage):
        raise TypeError("seed_lineage must be a SeedLineage")
    strata_by_subject = dict(
        zip(
            exchangeability.subject_ids,
            exchangeability.subject_strata,
            strict=True,
        )
    )
    subjects_by_stratum: dict[tuple[Hashable, ...], list[str]] = defaultdict(list)
    for subject in exchangeability.subject_ids:
        subjects_by_stratum[strata_by_subject[subject]].append(subject)
    plans: list[SubjectBootstrapPlan] = []
    for resample_index in range(n_bootstraps):
        child_seed = seed_lineage.derive(
            "subject_bootstrap",
            exchangeability.exchangeability_id,
            f"resample={resample_index}",
        )
        rng = child_seed.python_random()
        draws: list[BootstrapSubjectDraw] = []
        draw_index = 0
        for stratum in sorted(subjects_by_stratum, key=canonical_json):
            subjects = subjects_by_stratum[stratum]
            for _ in subjects:
                source = rng.choice(subjects)
                draws.append(
                    BootstrapSubjectDraw(
                        source_subject_id=source,
                        bootstrap_subject_id=(
                            f"bootstrap_{resample_index:06d}_{draw_index:06d}"
                        ),
                        stratum=stratum,
                    )
                )
                draw_index += 1
        payload = {
            "exchangeability_id": exchangeability.exchangeability_id,
            "resample_index": resample_index,
            "draws": [
                {
                    "source_subject_id": draw.source_subject_id,
                    "bootstrap_subject_id": draw.bootstrap_subject_id,
                    "stratum": list(draw.stratum),
                }
                for draw in draws
            ],
            "seed_lineage": child_seed.to_dict(),
            "resampling_unit": "subject_block",
            "with_replacement": True,
        }
        plans.append(
            SubjectBootstrapPlan(
                bootstrap_id=stable_id(
                    "subject_bootstrap", payload, schema_version="1"
                ),
                exchangeability_id=exchangeability.exchangeability_id,
                resample_index=resample_index,
                draws=tuple(draws),
                seed_lineage=child_seed,
            )
        )
    return tuple(plans)


__all__ = [
    "BootstrapSubjectDraw",
    "ContextPermutationOperation",
    "ContextPermutationPlan",
    "ExchangeabilityDesign",
    "ExchangeabilityMap",
    "SubjectBootstrapPlan",
    "apply_context_permutation",
    "build_exchangeability_map",
    "plan_context_permutations",
    "plan_subject_bootstraps",
]
