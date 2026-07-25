"""Sparse uncertainty-aware empirical-Bayes hypergraph shrinkage for M5."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from crychic.core import canonical_digest, stable_id

from .hypergraph_prior import FrozenHypergraphPrior

HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_VERSION = (
    "sparse_incidence_uncertainty_aware_eb_m5_v2"
)
HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_COLUMNS = (
    "edge_id",
    "raw_effect",
    "raw_standard_error",
    "topology_mean",
    "prior_variance",
    "shrinkage_factor",
    "posterior_effect",
    "posterior_standard_error",
    "posterior_se_scope",
    "status",
    "reason_code",
    "prior_id",
    "spec_id",
    "formal_inference_allowed",
    "score_version",
    "shrinkage_record_id",
)
_SCHEMA_VERSION = "2.0.0"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class UncertaintyAwareHypergraphShrinkageV2Spec:
    """Frozen sparse-ridge and EB residual-variance policy."""

    node_ridge_penalty: float = 1.0
    minimum_prior_variance: float = 1.0e-8
    minimum_observed_edges: int = 4
    maximum_iterations: int = 100
    convergence_tolerance: float = 1.0e-8
    variance_update_damping: float = 0.5
    tau2_estimator: str = "zero_mean_residual_second_moment_v1"
    posterior_se_scope: str = "conditional_on_fitted_topology_mean"
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        ridge = _finite(self.node_ridge_penalty, field_name="node_ridge_penalty")
        variance_floor = _finite(
            self.minimum_prior_variance,
            field_name="minimum_prior_variance",
        )
        tolerance = _finite(
            self.convergence_tolerance,
            field_name="convergence_tolerance",
        )
        damping = _finite(
            self.variance_update_damping,
            field_name="variance_update_damping",
        )
        if ridge <= 0.0 or variance_floor <= 0.0 or tolerance <= 0.0:
            raise ValueError(
                "M5 v2 ridge, variance floor, and tolerance must be positive"
            )
        if not 0.0 < damping <= 1.0:
            raise ValueError("variance_update_damping must lie in (0, 1]")
        for field_name, minimum in (
            ("minimum_observed_edges", 3),
            ("maximum_iterations", 1),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        if self.tau2_estimator != "zero_mean_residual_second_moment_v1":
            raise ValueError("tau2_estimator is unsupported")
        if self.posterior_se_scope != "conditional_on_fitted_topology_mean":
            raise ValueError("posterior_se_scope is unsupported")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "node_ridge_penalty", ridge)
        object.__setattr__(self, "minimum_prior_variance", variance_floor)
        object.__setattr__(self, "convergence_tolerance", tolerance)
        object.__setattr__(self, "variance_update_damping", damping)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "uncertainty_aware_hypergraph_shrinkage_v2_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "convergence_tolerance": self.convergence_tolerance,
            "maximum_iterations": self.maximum_iterations,
            "minimum_observed_edges": self.minimum_observed_edges,
            "minimum_prior_variance": self.minimum_prior_variance,
            "node_ridge_penalty": self.node_ridge_penalty,
            "posterior_se_scope": self.posterior_se_scope,
            "schema_version": self.schema_version,
            "tau2_estimator": self.tau2_estimator,
            "variance_update_damping": self.variance_update_damping,
            "version": HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class UncertaintyAwareHypergraphShrinkageV2Fit:
    """Fit diagnostics for one sparse H-prior empirical-Bayes application."""

    prior_id: str
    spec: UncertaintyAwareHypergraphShrinkageV2Spec
    input_digest: str
    output_digest: str
    edge_count: int
    observed_edge_count: int
    incidence_column_count: int
    incidence_nnz: int
    prior_variance: float
    iterations: int
    converged: bool
    maximum_variance_update: float
    formal_inference_allowed: bool = False
    fit_id: str = field(init=False)

    def __post_init__(self) -> None:
        prior_id = _name(self.prior_id, field_name="prior_id")
        if not isinstance(self.spec, UncertaintyAwareHypergraphShrinkageV2Spec):
            raise TypeError("spec must be UncertaintyAwareHypergraphShrinkageV2Spec")
        for field_name in ("input_digest", "output_digest"):
            value = _name(getattr(self, field_name), field_name=field_name)
            if len(value) != 64:
                raise ValueError(f"{field_name} must be a canonical digest")
        for field_name, minimum in (
            ("edge_count", 1),
            ("observed_edge_count", 1),
            ("incidence_column_count", 1),
            ("incidence_nnz", 1),
            ("iterations", 1),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        if self.observed_edge_count > self.edge_count:
            raise ValueError("observed_edge_count cannot exceed edge_count")
        prior_variance = _finite(self.prior_variance, field_name="prior_variance")
        maximum_update = _finite(
            self.maximum_variance_update,
            field_name="maximum_variance_update",
        )
        if prior_variance <= 0.0 or maximum_update < 0.0:
            raise ValueError("M5 v2 variance diagnostics are invalid")
        if self.formal_inference_allowed is not False:
            raise ValueError("M5 v2 shrinkage cannot claim formal inference")
        object.__setattr__(self, "prior_id", prior_id)
        object.__setattr__(self, "prior_variance", prior_variance)
        object.__setattr__(self, "maximum_variance_update", maximum_update)
        object.__setattr__(
            self,
            "fit_id",
            stable_id(
                "uncertainty_aware_hypergraph_shrinkage_v2_fit",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "edge_count": self.edge_count,
            "formal_inference_allowed": False,
            "incidence_column_count": self.incidence_column_count,
            "incidence_nnz": self.incidence_nnz,
            "input_digest": self.input_digest,
            "iterations": self.iterations,
            "maximum_variance_update": self.maximum_variance_update,
            "observed_edge_count": self.observed_edge_count,
            "output_digest": self.output_digest,
            "prior_id": self.prior_id,
            "prior_variance": self.prior_variance,
            "spec_id": self.spec.spec_id,
            "version": HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_VERSION,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "fit_id": self.fit_id,
            **self._identity_payload(),
            "spec": self.spec.to_dict(),
        }


def sparse_hypergraph_incidence_v2(
    prior: FrozenHypergraphPrior,
) -> tuple[sparse.csr_matrix, tuple[str, ...]]:
    """Return intercept-plus-view incidence without constructing dense blocks."""

    if not isinstance(prior, FrozenHypergraphPrior):
        raise TypeError("prior must be FrozenHypergraphPrior")
    edge_count = len(prior.edges)
    row_indices: list[int] = list(range(edge_count))
    column_indices: list[int] = [0] * edge_count
    column_names = ["intercept"]
    next_column = 1
    for view_index, view_name in enumerate(prior.view_names):
        levels = tuple(sorted({edge.memberships[view_index] for edge in prior.edges}))
        level_columns = {
            level: next_column + index for index, level in enumerate(levels)
        }
        column_names.extend(f"{view_name}:{level}" for level in levels)
        for row_index, edge in enumerate(prior.edges):
            row_indices.append(row_index)
            column_indices.append(level_columns[edge.memberships[view_index]])
        next_column += len(levels)
    values: npt.NDArray[np.float64] = np.ones(len(row_indices), dtype=float)
    incidence = sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(edge_count, next_column),
        dtype=float,
    ).tocsr()
    return incidence, tuple(column_names)


def _canonical_input_digest(
    edge_ids: tuple[str, ...],
    effects: np.ndarray,
    standard_errors: np.ndarray,
) -> str:
    records = [
        {
            "edge_id": edge_id,
            "effect": (
                None
                if not math.isfinite(effect)
                else {"float_hex": float(effect).hex()}
            ),
            "standard_error": (
                None
                if not math.isfinite(standard_error)
                else {"float_hex": float(standard_error).hex()}
            ),
        }
        for edge_id, effect, standard_error in zip(
            edge_ids, effects, standard_errors, strict=True
        )
    ]
    return cast(str, canonical_digest(records))


def _solve_topology_mean(
    incidence: sparse.csr_matrix,
    values: np.ndarray,
    standard_error_squared: np.ndarray,
    prior_variance: float,
    *,
    node_ridge_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.reciprocal(standard_error_squared + prior_variance)
    weighted = incidence.multiply(weights[:, np.newaxis])
    penalty = np.full(incidence.shape[1], node_ridge_penalty, dtype=float)
    penalty[0] = 0.0
    left = (incidence.T @ weighted).tocsc() + sparse.diags(penalty, format="csc")
    right = np.asarray(incidence.T @ (weights * values)).ravel()
    coefficients = np.asarray(spsolve(left, right), dtype=float)
    if np.any(~np.isfinite(coefficients)):
        raise np.linalg.LinAlgError("sparse M5 coefficient solve was non-finite")
    return coefficients, np.asarray(incidence @ coefficients).ravel()


def fit_uncertainty_aware_hypergraph_shrinkage_v2(
    estimates: pd.DataFrame,
    *,
    prior: FrozenHypergraphPrior,
    spec: UncertaintyAwareHypergraphShrinkageV2Spec | None = None,
    edge_id_column: str = "edge_id",
    effect_column: str = "effect",
    standard_error_column: str = "standard_error",
) -> tuple[pd.DataFrame, UncertaintyAwareHypergraphShrinkageV2Fit]:
    """Shrink edge effects toward a sparse fitted topology mean with SEs."""

    if not isinstance(estimates, pd.DataFrame):
        raise TypeError("estimates must be a pandas DataFrame")
    if not isinstance(prior, FrozenHypergraphPrior):
        raise TypeError("prior must be FrozenHypergraphPrior")
    resolved = spec or UncertaintyAwareHypergraphShrinkageV2Spec()
    if not isinstance(resolved, UncertaintyAwareHypergraphShrinkageV2Spec):
        raise TypeError(
            "spec must be UncertaintyAwareHypergraphShrinkageV2Spec or None"
        )
    edge_id_column = _name(edge_id_column, field_name="edge_id_column")
    effect_column = _name(effect_column, field_name="effect_column")
    standard_error_column = _name(
        standard_error_column, field_name="standard_error_column"
    )
    required = {edge_id_column, effect_column, standard_error_column}
    missing = required.difference(estimates.columns)
    if missing:
        raise ValueError(f"M5 v2 estimates are missing columns: {sorted(missing)}")
    source = estimates.loc[
        :, [edge_id_column, effect_column, standard_error_column]
    ].copy()
    if source.empty or source[edge_id_column].isna().any():
        raise ValueError("M5 v2 estimates require non-empty edge IDs")
    source[edge_id_column] = source[edge_id_column].astype(str)
    if (
        source[edge_id_column].eq("").any()
        or source[edge_id_column].str.strip().ne(source[edge_id_column]).any()
        or source[edge_id_column].duplicated().any()
    ):
        raise ValueError("M5 v2 edge IDs must be canonical and unique")
    source = source.sort_values(edge_id_column, kind="stable", ignore_index=True)
    edge_ids = tuple(edge.edge_id for edge in prior.edges)
    if tuple(source[edge_id_column]) != edge_ids:
        raise ValueError("M5 v2 estimate universe differs from frozen H_prior")
    effect = pd.to_numeric(source[effect_column], errors="coerce")
    standard_error = pd.to_numeric(source[standard_error_column], errors="coerce")
    invalid_effect = source[effect_column].notna() & effect.isna()
    invalid_se = source[standard_error_column].notna() & standard_error.isna()
    if invalid_effect.any() or invalid_se.any():
        raise ValueError("M5 v2 effect inputs must be numeric or NA")
    effects = effect.to_numpy(dtype=float)
    standard_errors = standard_error.to_numpy(dtype=float)
    paired_missing = np.isnan(effects) == np.isnan(standard_errors)
    if not paired_missing.all():
        raise ValueError("M5 v2 effects and standard errors must be jointly observed")
    observed = np.isfinite(effects) & np.isfinite(standard_errors)
    if (
        np.isinf(effects).any()
        or np.isinf(standard_errors).any()
        or (standard_errors[observed] <= 0.0).any()
    ):
        raise ValueError("observed M5 v2 effects require finite positive SEs")
    if int(observed.sum()) < resolved.minimum_observed_edges:
        raise ValueError("M5 v2 has insufficient observed edges")

    full_incidence, column_names = sparse_hypergraph_incidence_v2(prior)
    incidence = full_incidence[observed]
    observed_effects = effects[observed]
    observed_se2 = np.square(standard_errors[observed])
    initial_variance = max(
        resolved.minimum_prior_variance,
        float(np.var(observed_effects, ddof=0) - np.mean(observed_se2)),
    )
    prior_variance = initial_variance
    converged = False
    maximum_update = math.inf
    iterations = 0
    coefficients = np.zeros(incidence.shape[1], dtype=float)
    observed_topology_mean: npt.NDArray[np.float64] = np.zeros(
        len(observed_effects), dtype=float
    )
    for iteration in range(1, resolved.maximum_iterations + 1):
        coefficients, observed_topology_mean = _solve_topology_mean(
            incidence,
            observed_effects,
            observed_se2,
            prior_variance,
            node_ridge_penalty=resolved.node_ridge_penalty,
        )
        residual = observed_effects - observed_topology_mean
        moment = max(
            resolved.minimum_prior_variance,
            float(np.mean(np.square(residual) - observed_se2)),
        )
        updated = (
            resolved.variance_update_damping * moment
            + (1.0 - resolved.variance_update_damping) * prior_variance
        )
        maximum_update = abs(updated - prior_variance)
        prior_variance = updated
        iterations = iteration
        if maximum_update <= resolved.convergence_tolerance * max(1.0, prior_variance):
            converged = True
            break
    coefficients, observed_topology_mean = _solve_topology_mean(
        incidence,
        observed_effects,
        observed_se2,
        prior_variance,
        node_ridge_penalty=resolved.node_ridge_penalty,
    )
    topology_mean = np.asarray(full_incidence @ coefficients).ravel()
    shrinkage_factor: npt.NDArray[np.float64] = np.full(
        len(effects), np.nan, dtype=float
    )
    posterior_effect: npt.NDArray[np.float64] = np.full(
        len(effects), np.nan, dtype=float
    )
    posterior_standard_error: npt.NDArray[np.float64] = np.full(
        len(effects), np.nan, dtype=float
    )
    shrinkage_factor[observed] = prior_variance / (
        prior_variance + np.square(standard_errors[observed])
    )
    posterior_effect[observed] = (
        shrinkage_factor[observed] * effects[observed]
        + (1.0 - shrinkage_factor[observed]) * topology_mean[observed]
    )
    posterior_standard_error[observed] = np.sqrt(
        shrinkage_factor[observed] * np.square(standard_errors[observed])
    )

    output = pd.DataFrame(
        {
            "edge_id": edge_ids,
            "raw_effect": effects,
            "raw_standard_error": standard_errors,
            "topology_mean": topology_mean,
            "prior_variance": prior_variance,
            "shrinkage_factor": shrinkage_factor,
            "posterior_effect": posterior_effect,
            "posterior_standard_error": posterior_standard_error,
            "posterior_se_scope": resolved.posterior_se_scope,
            "status": np.where(observed, "observed", "not_estimable"),
            "reason_code": np.where(observed, None, "input_effect_not_estimable"),
            "prior_id": prior.prior_id,
            "spec_id": resolved.spec_id,
            "formal_inference_allowed": False,
            "score_version": HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_VERSION,
        }
    )
    output["shrinkage_record_id"] = [
        stable_id(
            "uncertainty_aware_hypergraph_shrinkage_v2_record",
            {
                "edge_id": edge_id,
                "posterior_effect": (
                    None if not is_observed else float(posterior_value)
                ),
                "posterior_standard_error": (
                    None if not is_observed else float(posterior_se)
                ),
                "prior_id": prior.prior_id,
                "spec_id": resolved.spec_id,
            },
            schema_version=_SCHEMA_VERSION,
        )
        for edge_id, posterior_value, posterior_se, is_observed in zip(
            edge_ids,
            posterior_effect,
            posterior_standard_error,
            observed,
            strict=True,
        )
    ]
    output = output.loc[:, list(HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_COLUMNS)]
    input_digest = _canonical_input_digest(edge_ids, effects, standard_errors)
    output_digest = cast(
        str,
        canonical_digest(
            [
                {
                    "edge_id": edge_id,
                    "posterior_effect": (
                        None
                        if not math.isfinite(posterior_value)
                        else {"float_hex": float(posterior_value).hex()}
                    ),
                    "posterior_standard_error": (
                        None
                        if not math.isfinite(posterior_se)
                        else {"float_hex": float(posterior_se).hex()}
                    ),
                }
                for edge_id, posterior_value, posterior_se in zip(
                    edge_ids,
                    posterior_effect,
                    posterior_standard_error,
                    strict=True,
                )
            ]
        ),
    )
    fit = UncertaintyAwareHypergraphShrinkageV2Fit(
        prior_id=prior.prior_id,
        spec=resolved,
        input_digest=input_digest,
        output_digest=output_digest,
        edge_count=len(edge_ids),
        observed_edge_count=int(observed.sum()),
        incidence_column_count=len(column_names),
        incidence_nnz=int(full_incidence.nnz),
        prior_variance=prior_variance,
        iterations=iterations,
        converged=converged,
        maximum_variance_update=maximum_update,
    )
    return output, fit


__all__ = [
    "HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_COLUMNS",
    "HYPERGRAPH_UNCERTAINTY_SHRINKAGE_V2_VERSION",
    "UncertaintyAwareHypergraphShrinkageV2Fit",
    "UncertaintyAwareHypergraphShrinkageV2Spec",
    "fit_uncertainty_aware_hypergraph_shrinkage_v2",
    "sparse_hypergraph_incidence_v2",
]
