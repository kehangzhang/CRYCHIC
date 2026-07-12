"""Sample-level formula, EMM, rank, and contrast estimability diagnostics."""

from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
import patsy  # type: ignore[import-untyped]
from scipy import linalg

from crychic.core import canonical_json

from .contrasts import ContrastSpec


class DesignStatus(StrEnum):
    """Whether a declared sample-level design is usable."""

    READY = "ready"
    BLOCKED = "blocked"


class SubjectDesign(StrEnum):
    """Study-level subject allocation across referenced contexts."""

    INDEPENDENT = "independent_subjects"
    PAIRED = "fully_paired_subjects"
    MIXED = "mixed_paired_unpaired_subjects"
    REPEATED_MULTI_CONTEXT = "repeated_multi_context_subjects"


class DesignAuditError(ValueError):
    """Raised when a formula is invalid or exceeds the reviewed grammar."""


@dataclass(frozen=True, slots=True)
class SubjectDesignAudit:
    """Allocation audit that is independent of receiver-specific missingness."""

    design: SubjectDesign
    context_nodes: tuple[Hashable, ...]
    subject_sets: tuple[frozenset[Hashable], ...]
    status: DesignStatus
    reason_code: str | None

    def __post_init__(self) -> None:
        design = SubjectDesign(self.design)
        status = DesignStatus(self.status)
        if len(self.context_nodes) < 2:
            raise ValueError("subject design audit requires at least two contexts")
        if len(set(self.context_nodes)) != len(self.context_nodes):
            raise ValueError("subject design contexts must be unique")
        if len(self.subject_sets) != len(self.context_nodes):
            raise ValueError("subject sets must align one-to-one with contexts")
        if status is DesignStatus.BLOCKED and not self.reason_code:
            raise ValueError("blocked subject design requires a reason_code")
        if status is DesignStatus.READY and self.reason_code is not None:
            raise ValueError("ready subject design cannot have a reason_code")
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "status", status)

    @property
    def ready(self) -> bool:
        return self.status is DesignStatus.READY

    @property
    def paired(self) -> bool:
        return self.design is SubjectDesign.PAIRED

    def subjects_for(self, context: Hashable) -> frozenset[Hashable]:
        """Return the study-level subjects observed in one referenced context."""

        try:
            index = self.context_nodes.index(context)
        except ValueError as error:
            raise KeyError(context) from error
        return self.subject_sets[index]


def audit_subject_design(
    sample_metadata: pd.DataFrame,
    context_nodes: Sequence[Hashable],
    *,
    context_key: str,
    subject_key: str,
) -> SubjectDesignAudit:
    """Classify independent, fully paired, and unsupported repeated designs.

    This audit uses the complete sample table, before filtering any receiver.
    Receiver-specific state missingness can therefore trigger a paired
    complete-case analysis only when the underlying study is fully paired.
    """

    nodes = tuple(context_nodes)
    if len(nodes) < 2 or len(set(nodes)) != len(nodes):
        raise ValueError("context_nodes must contain at least two unique contexts")
    missing = {context_key, subject_key}.difference(sample_metadata.columns)
    if missing:
        raise DesignAuditError(
            f"sample metadata is missing subject-design field(s): {sorted(missing)}"
        )
    if sample_metadata[[context_key, subject_key]].isna().any().any():
        raise DesignAuditError(
            "sample metadata subject and context fields must not contain missing values"
        )

    subject_sets = tuple(
        frozenset(
            sample_metadata.loc[
                sample_metadata[context_key].map(
                    lambda value, expected=node: value == expected
                ),
                subject_key,
            ].tolist()
        )
        for node in nodes
    )
    pairwise_overlap = any(
        left.intersection(right)
        for left, right in itertools.combinations(subject_sets, 2)
    )
    if not pairwise_overlap:
        design = SubjectDesign.INDEPENDENT
        status = DesignStatus.READY
        reason = None
    elif len(nodes) == 2 and subject_sets[0] == subject_sets[1]:
        design = SubjectDesign.PAIRED
        status = DesignStatus.READY
        reason = None
    elif len(nodes) == 2:
        design = SubjectDesign.MIXED
        status = DesignStatus.BLOCKED
        reason = "mixed_paired_unpaired_design_not_supported"
    else:
        design = SubjectDesign.REPEATED_MULTI_CONTEXT
        status = DesignStatus.BLOCKED
        reason = "repeated_multi_context_design_not_supported"
    return SubjectDesignAudit(
        design=design,
        context_nodes=nodes,
        subject_sets=subject_sets,
        status=status,
        reason_code=reason,
    )


def default_design_formula(
    context_keys: Sequence[str], covariates: Sequence[str] = ()
) -> str:
    """Return the reviewed default: covariates plus the context factorial."""

    contexts = tuple(context_keys)
    if not contexts:
        raise ValueError("context_keys must contain at least one field")
    terms = [*covariates, " * ".join(contexts)]
    return "~ " + " + ".join(terms)


def _factor_names(formula: str, allowed: set[str]) -> tuple[str, ...]:
    try:
        description = patsy.ModelDesc.from_formula(formula)
    except Exception as error:
        raise DesignAuditError(f"invalid design formula: {error}") from error
    if description.lhs_termlist:
        raise DesignAuditError("design formula must be one-sided and start with '~'")
    names: list[str] = []
    for term in description.rhs_termlist:
        for factor in term.factors:
            code = str(factor.code).strip()
            if code not in allowed:
                raise DesignAuditError(
                    "design formula factors must be declared context/covariate field "
                    f"names; unsupported expression {code!r}"
                )
            names.append(code)
    return tuple(dict.fromkeys(names))


def _stable_level_key(value: object) -> str:
    plain = value.item() if isinstance(value, np.generic) else value
    result: str = canonical_json(
        {
            "type": f"{type(plain).__module__}.{type(plain).__qualname__}",
            "value": plain,
        }
    )
    return result


def _levels(series: pd.Series) -> tuple[object, ...]:
    return tuple(sorted(series.drop_duplicates().tolist(), key=_stable_level_key))


def _prepare_table(
    sample_metadata: pd.DataFrame,
    *,
    context_keys: tuple[str, ...],
    covariates: tuple[str, ...],
) -> pd.DataFrame:
    required = (*context_keys, *covariates)
    missing = set(required).difference(sample_metadata.columns)
    if missing:
        raise DesignAuditError(
            f"sample metadata is missing design field(s): {sorted(missing)}"
        )
    if sample_metadata.empty:
        raise DesignAuditError("sample metadata must contain at least one row")
    table = sample_metadata.loc[:, list(required)].copy(deep=True)
    if table.isna().any().any():
        raise DesignAuditError(
            "sample metadata design fields must not contain missing values"
        )
    for key in context_keys:
        table[key] = pd.Categorical(table[key], categories=list(_levels(table[key])))
    for key in covariates:
        series = table[key]
        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            table[key] = pd.Categorical(series, categories=list(_levels(series)))
    return table


def _reference_grid(
    table: pd.DataFrame,
    *,
    context_keys: tuple[str, ...],
    covariates: tuple[str, ...],
    max_rows: int,
) -> pd.DataFrame:
    factor_values: list[tuple[object, ...]] = []
    factor_names: list[str] = []
    for key in (*context_keys, *covariates):
        series = table[key]
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = tuple(series.cat.categories.tolist())
        else:
            values = (float(pd.to_numeric(series).mean()),)
        factor_names.append(key)
        factor_values.append(values)
    n_rows = math.prod(len(values) for values in factor_values)
    if n_rows > max_rows:
        raise DesignAuditError(
            f"balanced EMM reference grid would contain {n_rows} rows; "
            f"maximum is {max_rows}"
        )
    grid = pd.DataFrame(
        itertools.product(*factor_values),
        columns=factor_names,
    )
    for key in (*context_keys, *covariates):
        source = table[key]
        if isinstance(source.dtype, pd.CategoricalDtype):
            grid[key] = pd.Categorical(
                grid[key], categories=source.cat.categories.tolist()
            )
    return grid


def _context_node(row: Mapping[str, Any], keys: tuple[str, ...]) -> Hashable:
    values = tuple((key, row[key]) for key in sorted(keys))
    return values[0][1] if len(values) == 1 else values


def _rank_diagnostics(
    matrix: np.ndarray, column_names: tuple[str, ...]
) -> tuple[int, tuple[str, ...], float | None]:
    _, triangular, pivots = linalg.qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    tolerance = max(matrix.shape) * np.finfo(float).eps * diagonal.max(initial=0.0)
    rank = int(np.sum(diagonal > tolerance))
    aliased = tuple(column_names[int(index)] for index in pivots[rank:])
    if rank < len(column_names):
        condition_number = None
    else:
        singular = np.linalg.svd(matrix, compute_uv=False)
        condition_number = float(singular[0] / singular[-1])
    return rank, aliased, condition_number


@dataclass(frozen=True, slots=True)
class DesignAudit:
    """Auditable formula matrix and balanced estimated-marginal-mean grid."""

    formula: str
    context_keys: tuple[str, ...]
    covariates: tuple[str, ...]
    factor_names: tuple[str, ...]
    status: DesignStatus
    reason_codes: tuple[str, ...]
    rank: int
    n_columns: int
    condition_number: float | None
    aliased_columns: tuple[str, ...]
    column_names: tuple[str, ...]
    sample_ids: tuple[Hashable, ...]
    design_matrix: pd.DataFrame
    reference_grid: pd.DataFrame
    emm_matrix: pd.DataFrame
    context_nodes: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        status = DesignStatus(self.status)
        if self.design_matrix.shape[1] != self.n_columns:
            raise ValueError("design matrix width does not match n_columns")
        if tuple(self.design_matrix.columns) != self.column_names:
            raise ValueError("design matrix columns do not match column_names")
        if len(self.sample_ids) != len(self.design_matrix):
            raise ValueError("sample_ids must align one-to-one with design rows")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")
        if self.emm_matrix.shape != (len(self.context_nodes), self.n_columns):
            raise ValueError("EMM matrix shape does not match contexts x coefficients")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_codes", tuple(sorted(set(self.reason_codes))))
        for name in ("design_matrix", "reference_grid", "emm_matrix"):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))

    @property
    def ready(self) -> bool:
        return self.status is DesignStatus.READY

    def coefficient_contrast(self, contrast: ContrastSpec) -> np.ndarray:
        """Map a context-mean contrast onto the formula coefficient scale."""

        unknown = set(contrast.weights).difference(self.context_nodes)
        if unknown:
            raise ValueError(
                f"contrast {contrast.name!r} references context(s) absent from EMM "
                f"grid: {sorted(map(repr, unknown))}"
            )
        weights = np.asarray(
            [contrast.weights.get(node, 0.0) for node in self.context_nodes],
            dtype=float,
        )
        result: np.ndarray = weights @ self.emm_matrix.to_numpy(dtype=float)
        return result

    def contrast_estimable(
        self, contrast: ContrastSpec, *, tolerance: float = 1e-8
    ) -> bool:
        """Test whether a contrast lies in the observed design row space."""

        if not contrast.estimable or not self.ready:
            return False
        vector = self.coefficient_contrast(contrast)
        if np.linalg.norm(vector) <= tolerance:
            return False
        matrix = self.design_matrix.to_numpy(dtype=float)
        projection = np.linalg.pinv(matrix) @ matrix
        residual = vector - vector @ projection
        scale = max(1.0, float(np.linalg.norm(vector)))
        return bool(np.linalg.norm(residual) <= tolerance * scale)

    def contrast_table(self, contrasts: Sequence[ContrastSpec]) -> pd.DataFrame:
        """Return one explicit estimability row per registered contrast."""

        rows: list[dict[str, object]] = []
        for contrast in contrasts:
            try:
                estimable = self.contrast_estimable(contrast)
                reason = None if estimable else "contrast_not_estimable"
                vector_norm = float(np.linalg.norm(self.coefficient_contrast(contrast)))
            except ValueError:
                estimable = False
                reason = "contrast_context_absent_from_reference_grid"
                vector_norm = math.nan
            rows.append(
                {
                    "contrast": contrast.name,
                    "family": contrast.family,
                    "mode": contrast.mode,
                    "estimable": estimable,
                    "reason_code": reason,
                    "coefficient_vector_norm": vector_norm,
                }
            )
        return pd.DataFrame(rows)


def audit_sample_design(
    sample_metadata: pd.DataFrame,
    *,
    context_keys: Sequence[str],
    covariates: Sequence[str] = (),
    formula: str | None = None,
    sample_key: str | None = None,
    condition_number_warning: float = 1e8,
    max_reference_rows: int = 10_000,
) -> DesignAudit:
    """Build and audit a reviewed sample-level formula and balanced EMM grid."""

    contexts = tuple(context_keys)
    covariate_names = tuple(covariates)
    if not contexts:
        raise DesignAuditError("context_keys must contain at least one field")
    if len(set((*contexts, *covariate_names))) != len((*contexts, *covariate_names)):
        raise DesignAuditError("context and covariate fields must be unique")
    if not math.isfinite(condition_number_warning) or condition_number_warning <= 1:
        raise ValueError("condition_number_warning must be finite and greater than 1")
    if isinstance(max_reference_rows, bool) or max_reference_rows < 1:
        raise ValueError("max_reference_rows must be an integer >= 1")
    declared_formula = formula or default_design_formula(contexts, covariate_names)
    factor_names = _factor_names(declared_formula, set((*contexts, *covariate_names)))
    table = _prepare_table(
        sample_metadata,
        context_keys=contexts,
        covariates=covariate_names,
    )
    if sample_key is None:
        sample_ids: tuple[Hashable, ...] = tuple(sample_metadata.index)
    else:
        if sample_key not in sample_metadata:
            raise DesignAuditError(
                f"sample metadata is missing sample key {sample_key!r}"
            )
        sample_ids = tuple(sample_metadata[sample_key].tolist())
    if len(set(sample_ids)) != len(sample_ids):
        raise DesignAuditError("sample design rows must have unique sample IDs")
    try:
        matrix = patsy.dmatrix(declared_formula, table, return_type="dataframe")
    except Exception as error:
        raise DesignAuditError(f"design matrix construction failed: {error}") from error
    column_names = tuple(map(str, matrix.columns))
    numeric = matrix.to_numpy(dtype=float)
    rank, aliased, condition_number = _rank_diagnostics(numeric, column_names)

    grid = _reference_grid(
        table,
        context_keys=contexts,
        covariates=covariate_names,
        max_rows=max_reference_rows,
    )
    try:
        grid_matrix = np.asarray(
            patsy.build_design_matrices(
                [matrix.design_info], grid, return_type="dataframe"
            )[0],
            dtype=float,
        )
    except Exception as error:
        raise DesignAuditError(
            f"EMM reference-grid encoding failed: {error}"
        ) from error

    context_nodes: list[Hashable] = []
    emm_rows: list[np.ndarray] = []
    context_group_keys: str | list[str] = (
        contexts[0] if len(contexts) == 1 else list(contexts)
    )
    grouped = grid.groupby(context_group_keys, observed=True, dropna=False, sort=False)
    for values, indices in grouped.indices.items():
        key_values = values if isinstance(values, tuple) else (values,)
        mapping = dict(zip(contexts, key_values, strict=True))
        context_nodes.append(_context_node(mapping, contexts))
        emm_rows.append(grid_matrix[np.asarray(indices, dtype=int)].mean(axis=0))

    reasons: list[str] = []
    if set(contexts).difference(factor_names):
        reasons.append("context_factor_omitted")
    if rank < len(column_names):
        reasons.append("design_rank_deficient")
    if condition_number is not None and condition_number > condition_number_warning:
        reasons.append("design_ill_conditioned")
    blocking = {"context_factor_omitted", "design_rank_deficient"}
    status = (
        DesignStatus.BLOCKED if blocking.intersection(reasons) else DesignStatus.READY
    )
    return DesignAudit(
        formula=declared_formula,
        context_keys=contexts,
        covariates=covariate_names,
        factor_names=factor_names,
        status=status,
        reason_codes=tuple(reasons),
        rank=rank,
        n_columns=len(column_names),
        condition_number=condition_number,
        aliased_columns=aliased,
        column_names=column_names,
        sample_ids=sample_ids,
        design_matrix=pd.DataFrame(numeric, columns=column_names, index=table.index),
        reference_grid=grid,
        emm_matrix=pd.DataFrame(np.vstack(emm_rows), columns=column_names),
        context_nodes=tuple(context_nodes),
    )


__all__ = [
    "DesignAudit",
    "DesignAuditError",
    "DesignStatus",
    "SubjectDesign",
    "SubjectDesignAudit",
    "audit_sample_design",
    "audit_subject_design",
    "default_design_formula",
]
