"""Preregistered E5 hypergraph topology component swaps for v7."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from benchmarks.simulation.v7_metrics import METRIC_COLUMNS
from crychic.core import canonical_digest
from crychic.resources import ResourceBundle
from crychic.scoring import (
    FrozenHypergraphPrior,
    UncertaintyAwareHypergraphShrinkageV2Spec,
    fit_uncertainty_aware_hypergraph_shrinkage_v2,
    freeze_hypergraph_prior,
    permute_hypergraph_prior_degree_matched,
    rewire_hypergraph_prior_degree_matched,
    select_hypergraph_prior_views,
)

SCHEMA_VERSION = "crychic-suggest-next2-v7-e5-hypergraph-swap-v1"
E5_GENERATOR_ID = "M5"
E5_INFERENCE_ID = "I1"
E5_ARMS = (
    "no_prior",
    "ligand_only_prior",
    "receptor_only_prior",
    "pathway_only_prior",
    "full_hypergraph",
    "degree_matched_permuted_hypergraph",
    "rewired_10pct",
    "rewired_25pct",
    "rewired_50pct",
    "pairwise_clique_expansion",
    "tensor_factorization",
)
E5_METRICS = (
    "effect_mse",
    "effect_spearman",
    "direction_accuracy",
    "ci_coverage",
    "weak_signal_power",
    "exact_hyperedge_ap",
    "topology_permutation_advantage",
)
E5_EDGE_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "contrast_name",
    "score_view",
    "event_id",
    "sender",
    "receiver",
    "interaction_id",
    "raw_effect",
    "raw_standard_error",
    "topology_mean",
    "prior_variance",
    "shrinkage_factor",
    "posterior_effect",
    "posterior_standard_error",
    "posterior_se_scope",
    "truth_known",
    "truth_label",
    "truth_effect",
    "status",
    "reason_code",
    "prior_id",
    "fit_id",
    "formal_inference_allowed",
)
E5_FIT_COLUMNS = (
    "schema_version",
    "dataset_id",
    "dgp_family",
    "design_kind",
    "contrast_name",
    "score_view",
    "method_kind",
    "prior_id",
    "fit_id",
    "edge_count",
    "observed_edge_count",
    "incidence_column_count",
    "incidence_nnz",
    "prior_variance",
    "iterations",
    "converged",
    "tensor_rank",
    "formal_inference_allowed",
)
E5_TOPOLOGY_COLUMNS = (
    "schema_version",
    "dataset_id",
    "score_view",
    "prior_id",
    "parent_prior_id",
    "topology_kind",
    "view_names",
    "requested_rewire_fraction",
    "changed_edge_fraction",
    "degree_profiles_equal",
    "outcome_blind",
)

_TOPOLOGY_VIEWS = ("sender", "ligand", "receptor", "receiver", "pathway")
_REWIRE_FRACTIONS = {
    "rewired_10pct": 0.10,
    "rewired_25pct": 0.25,
    "rewired_50pct": 0.50,
}
_TENSOR_RANK = 2
_TENSOR_RIDGE = 0.10
_TENSOR_MAXIMUM_OUTER_ITERATIONS = 6
_TENSOR_OPTIMIZER_MAXIMUM_ITERATIONS = 300
_TENSOR_CONVERGENCE_TOLERANCE = 1.0e-6
_MINIMUM_PRIOR_VARIANCE = 1.0e-8


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


def _derived_seed(root_seed: int, namespace: str) -> int:
    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise ValueError("root_seed must be an integer")
    label = _name(namespace, field="namespace")
    digest = hashlib.sha256(
        f"crychic:v7:e5:{root_seed}:{label}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _effect_universe(effects: pd.DataFrame) -> pd.DataFrame:
    required = {
        "generator_id",
        "score_view",
        "resolution",
        "inference_id",
        "contrast_name",
        "event_id",
        "sender",
        "receiver",
        "interaction_id",
        "effect",
        "standard_error",
        "status",
    }
    missing = required.difference(effects.columns)
    if not isinstance(effects, pd.DataFrame) or missing:
        raise ValueError(f"E5 effects are invalid: missing={sorted(missing)}")
    selected = effects.loc[
        effects["generator_id"].eq("G3")
        & effects["score_view"].eq("primary_sender_detection")
        & effects["resolution"].eq("sender_lr_receiver_child")
        & effects["inference_id"].eq("I1"),
        [
            "contrast_name",
            "event_id",
            "sender",
            "receiver",
            "interaction_id",
            "effect",
            "standard_error",
            "status",
        ],
    ].copy()
    key = ["contrast_name", "event_id"]
    if selected.empty or selected.duplicated(key).any():
        raise ValueError("E5 requires one G3-I1 effect per contrast and event")
    mapping = ["event_id", "sender", "receiver", "interaction_id"]
    if selected.loc[:, mapping].drop_duplicates().duplicated("event_id").any():
        raise ValueError("E5 event IDs do not identify one biological edge")
    event_sets = selected.groupby("contrast_name", observed=True)["event_id"].apply(
        lambda values: tuple(sorted(map(str, values)))
    )
    if event_sets.nunique() != 1:
        raise ValueError("E5 contrasts do not share one frozen event universe")
    observed = selected["status"].astype(str).eq("observed")
    effect = pd.to_numeric(selected["effect"], errors="coerce")
    standard_error = pd.to_numeric(selected["standard_error"], errors="coerce")
    valid = (
        observed
        & effect.notna()
        & standard_error.notna()
        & np.isfinite(effect)
        & np.isfinite(standard_error)
        & standard_error.ge(0.0)
    )
    selected["effect"] = effect.where(valid, np.nan)
    selected["standard_error"] = standard_error.where(valid, np.nan)
    selected["status"] = np.where(valid, "observed", "not_estimable")
    return selected.sort_values(key, kind="stable", ignore_index=True)


def _resource_topology(
    effects: pd.DataFrame,
    resource: ResourceBundle,
) -> pd.DataFrame:
    if not isinstance(resource, ResourceBundle):
        raise TypeError("resource must be a ResourceBundle")
    events = effects.loc[
        :, ["event_id", "sender", "receiver", "interaction_id"]
    ].drop_duplicates()
    lookup = pd.DataFrame.from_records(
        [
            {
                "interaction_id": interaction.interaction_id,
                "ligand": interaction.ligand_name,
                "receptor": interaction.receptor_name,
                "pathway": interaction.pathway or "__unannotated_pathway__",
            }
            for interaction in resource.interactions
        ]
    )
    if lookup.empty or lookup["interaction_id"].duplicated().any():
        raise ValueError("E5 resource interaction identities are not unique")
    topology = events.merge(
        lookup,
        on="interaction_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if topology.loc[:, list(_TOPOLOGY_VIEWS)].isna().any(axis=None):
        raise ValueError("E5 resource does not annotate the complete event universe")
    return topology.rename(columns={"event_id": "edge_id"}).sort_values(
        "edge_id", kind="stable", ignore_index=True
    )


def _pairwise_clique_prior(full: FrozenHypergraphPrior) -> FrozenHypergraphPrior:
    records: list[dict[str, str]] = []
    pair_names = tuple(
        f"{left}__{right}" for left, right in itertools.combinations(full.view_names, 2)
    )
    for edge in full.edges:
        values = dict(zip(full.view_names, edge.memberships, strict=True))
        record = {"edge_id": edge.edge_id}
        for left, right in itertools.combinations(full.view_names, 2):
            record[f"{left}__{right}"] = (
                f"{left}:{values[left]}|{right}:{values[right]}"
            )
        records.append(record)
    frozen = freeze_hypergraph_prior(
        pd.DataFrame.from_records(records), view_columns=pair_names
    )
    return FrozenHypergraphPrior(
        view_names=frozen.view_names,
        edges=frozen.edges,
        source_digest=str(
            canonical_digest(
                {
                    "parent_prior_id": full.prior_id,
                    "representation": "all_unordered_view_pairs",
                }
            )
        ),
        topology_kind="pairwise_clique_expansion",
        parent_prior_id=full.prior_id,
    )


def _prior_controls(
    full: FrozenHypergraphPrior,
    *,
    root_seed: int,
) -> dict[str, FrozenHypergraphPrior]:
    controls = {
        "ligand_only_prior": select_hypergraph_prior_views(
            full, view_names=("ligand",)
        ),
        "receptor_only_prior": select_hypergraph_prior_views(
            full, view_names=("receptor",)
        ),
        "pathway_only_prior": select_hypergraph_prior_views(
            full, view_names=("pathway",)
        ),
        "full_hypergraph": full,
        "degree_matched_permuted_hypergraph": (
            permute_hypergraph_prior_degree_matched(
                full,
                seed=_derived_seed(root_seed, "degree-matched-permutation"),
            )
        ),
        "pairwise_clique_expansion": _pairwise_clique_prior(full),
    }
    for arm, fraction in _REWIRE_FRACTIONS.items():
        controls[arm] = rewire_hypergraph_prior_degree_matched(
            full,
            fraction=fraction,
            seed=_derived_seed(root_seed, arm),
        )
    return controls


def _changed_edge_fraction(
    full: FrozenHypergraphPrior,
    control: FrozenHypergraphPrior,
) -> float | None:
    if full.view_names != control.view_names:
        return None
    return float(
        np.mean(
            [
                source.memberships != target.memberships
                for source, target in zip(full.edges, control.edges, strict=True)
            ]
        )
    )


def _topology_diagnostics(
    *,
    dataset_id: str,
    full: FrozenHypergraphPrior,
    controls: dict[str, FrozenHypergraphPrior],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for arm, prior in controls.items():
        fraction = _REWIRE_FRACTIONS.get(arm)
        comparable = full.view_names == prior.view_names
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "score_view": arm,
                "prior_id": prior.prior_id,
                "parent_prior_id": prior.parent_prior_id,
                "topology_kind": prior.topology_kind,
                "view_names": "|".join(prior.view_names),
                "requested_rewire_fraction": fraction,
                "changed_edge_fraction": _changed_edge_fraction(full, prior),
                "degree_profiles_equal": (
                    full.degree_profile() == prior.degree_profile()
                    if comparable
                    else pd.NA
                ),
                "outcome_blind": True,
            }
        )
    records.extend(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "score_view": "no_prior",
                "prior_id": None,
                "parent_prior_id": None,
                "topology_kind": "none",
                "view_names": None,
                "requested_rewire_fraction": None,
                "changed_edge_fraction": None,
                "degree_profiles_equal": pd.NA,
                "outcome_blind": True,
            },
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "score_view": "tensor_factorization",
                "prior_id": full.prior_id,
                "parent_prior_id": full.prior_id,
                "topology_kind": "fixed_rank_cp_tensor_on_declared_modes",
                "view_names": "|".join(full.view_names),
                "requested_rewire_fraction": None,
                "changed_edge_fraction": 0.0,
                "degree_profiles_equal": True,
                "outcome_blind": True,
            },
        ]
    )
    result = pd.DataFrame.from_records(records, columns=E5_TOPOLOGY_COLUMNS)
    return result.sort_values("score_view", kind="stable", ignore_index=True)


@dataclass(frozen=True, slots=True)
class _TensorFit:
    topology_mean: npt.NDArray[np.float64]
    prior_variance: float
    posterior_effect: npt.NDArray[np.float64]
    posterior_standard_error: npt.NDArray[np.float64]
    shrinkage_factor: npt.NDArray[np.float64]
    iterations: int
    converged: bool


def _tensor_row_indices(
    prior: FrozenHypergraphPrior,
) -> tuple[tuple[npt.NDArray[np.int64], ...], tuple[int, ...]]:
    indices: list[npt.NDArray[np.int64]] = []
    sizes: list[int] = []
    for mode in range(len(prior.view_names)):
        levels = tuple(sorted({edge.memberships[mode] for edge in prior.edges}))
        lookup = {level: index for index, level in enumerate(levels)}
        indices.append(
            np.asarray([lookup[edge.memberships[mode]] for edge in prior.edges])
        )
        sizes.append(len(levels))
    return tuple(indices), tuple(sizes)


def _cp_prediction(
    factors: list[npt.NDArray[np.float64]],
    row_indices: tuple[npt.NDArray[np.int64], ...],
    intercept: float,
) -> npt.NDArray[np.float64]:
    components = np.ones((len(row_indices[0]), _TENSOR_RANK), dtype=float)
    for factor, indices in zip(factors, row_indices, strict=True):
        components *= factor[indices]
    return intercept + components.sum(axis=1)


def _fit_tensor_factorization(
    estimates: pd.DataFrame,
    *,
    prior: FrozenHypergraphPrior,
    seed: int,
) -> _TensorFit:
    source = estimates.sort_values("edge_id", kind="stable", ignore_index=True)
    if tuple(source["edge_id"].astype(str)) != tuple(
        edge.edge_id for edge in prior.edges
    ):
        raise ValueError("tensor factorization event universe differs from H_prior")
    effects = pd.to_numeric(source["effect"], errors="coerce").to_numpy(float)
    standard_errors = pd.to_numeric(source["standard_error"], errors="coerce").to_numpy(
        float
    )
    observed = np.isfinite(effects) & np.isfinite(standard_errors)
    if int(observed.sum()) < 4 or (standard_errors[observed] < 0.0).any():
        raise ValueError("tensor factorization requires four valid effects and SEs")
    row_indices, mode_sizes = _tensor_row_indices(prior)
    observed_indices = tuple(indices[observed] for indices in row_indices)
    values = effects[observed]
    se2 = np.square(standard_errors[observed])
    prior_variance = max(
        _MINIMUM_PRIOR_VARIANCE,
        float(np.var(values, ddof=0) - np.mean(se2)),
    )
    rng = np.random.default_rng(seed)
    component_scale = max(float(np.std(values)), 1.0e-3) ** (1.0 / len(mode_sizes))
    factors = [
        rng.normal(scale=component_scale, size=(size, _TENSOR_RANK))
        for size in mode_sizes
    ]
    intercept = float(np.mean(values))
    offsets = np.cumsum([1, *(size * _TENSOR_RANK for size in mode_sizes)], dtype=int)
    parameters = np.concatenate(
        [np.asarray([intercept]), *(factor.ravel() for factor in factors)]
    )

    def unpack(
        vector: npt.NDArray[np.float64],
    ) -> tuple[float, list[npt.NDArray[np.float64]]]:
        local_factors = [
            vector[offsets[mode] : offsets[mode + 1]].reshape(
                mode_sizes[mode], _TENSOR_RANK
            )
            for mode in range(len(mode_sizes))
        ]
        return float(vector[0]), local_factors

    converged = False
    completed_iterations = 0
    for _outer_iteration in range(_TENSOR_MAXIMUM_OUTER_ITERATIONS):
        weights = np.reciprocal(se2 + prior_variance)

        def objective_gradient(
            vector: npt.NDArray[np.float64],
            fixed_weights: npt.NDArray[np.float64] = weights,
        ) -> tuple[float, npt.NDArray[np.float64]]:
            local_intercept, local_factors = unpack(vector)
            components = np.ones((len(values), _TENSOR_RANK), dtype=float)
            for factor, indices in zip(local_factors, observed_indices, strict=True):
                components *= factor[indices]
            residual = local_intercept + components.sum(axis=1) - values
            weighted_residual = fixed_weights * residual
            gradient = np.zeros_like(vector)
            gradient[0] = weighted_residual.sum()
            objective = 0.5 * float(np.sum(fixed_weights * np.square(residual)))
            for mode, (factor, indices) in enumerate(
                zip(local_factors, observed_indices, strict=True)
            ):
                other = np.ones((len(values), _TENSOR_RANK), dtype=float)
                for other_mode, (other_factor, other_indices) in enumerate(
                    zip(local_factors, observed_indices, strict=True)
                ):
                    if other_mode != mode:
                        other *= other_factor[other_indices]
                factor_gradient = _TENSOR_RIDGE * factor.copy()
                np.add.at(
                    factor_gradient,
                    indices,
                    weighted_residual[:, np.newaxis] * other,
                )
                gradient[offsets[mode] : offsets[mode + 1]] = factor_gradient.ravel()
                objective += 0.5 * _TENSOR_RIDGE * float(np.square(factor).sum())
            return objective, gradient

        optimized = minimize(
            objective_gradient,
            parameters,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": _TENSOR_OPTIMIZER_MAXIMUM_ITERATIONS,
                "ftol": 1.0e-7,
                "gtol": 1.0e-5,
                "maxls": 40,
            },
        )
        parameters = np.asarray(optimized.x, dtype=float)
        completed_iterations += int(optimized.nit)
        intercept, factors = unpack(parameters)
        prediction = _cp_prediction(factors, observed_indices, intercept)
        residual = values - prediction
        moment = max(
            _MINIMUM_PRIOR_VARIANCE,
            float(np.mean(np.square(residual) - se2)),
        )
        variance_converged = abs(moment - prior_variance) <= (
            _TENSOR_CONVERGENCE_TOLERANCE * max(1.0, prior_variance)
        )
        prior_variance = moment
        if bool(optimized.success) and variance_converged:
            converged = True
            break
    topology_mean = _cp_prediction(factors, row_indices, intercept)
    if not np.isfinite(topology_mean).all():
        raise np.linalg.LinAlgError("tensor topology mean is non-finite")
    shrinkage_factor = np.full(len(effects), np.nan, dtype=float)
    posterior_effect = np.full(len(effects), np.nan, dtype=float)
    posterior_standard_error = np.full(len(effects), np.nan, dtype=float)
    shrinkage_factor[observed] = prior_variance / (prior_variance + se2)
    posterior_effect[observed] = (
        shrinkage_factor[observed] * effects[observed]
        + (1.0 - shrinkage_factor[observed]) * topology_mean[observed]
    )
    posterior_standard_error[observed] = np.sqrt(shrinkage_factor[observed] * se2)
    return _TensorFit(
        topology_mean=topology_mean,
        prior_variance=prior_variance,
        posterior_effect=posterior_effect,
        posterior_standard_error=posterior_standard_error,
        shrinkage_factor=shrinkage_factor,
        iterations=completed_iterations,
        converged=converged,
    )


def _raw_result(estimates: pd.DataFrame) -> pd.DataFrame:
    source = estimates.sort_values("edge_id", kind="stable", ignore_index=True)
    observed = source["effect"].notna() & source["standard_error"].notna()
    return pd.DataFrame(
        {
            "edge_id": source["edge_id"].astype(str),
            "raw_effect": source["effect"],
            "raw_standard_error": source["standard_error"],
            "topology_mean": np.nan,
            "prior_variance": np.nan,
            "shrinkage_factor": np.where(observed, 1.0, np.nan),
            "posterior_effect": source["effect"],
            "posterior_standard_error": source["standard_error"],
            "posterior_se_scope": "raw_I1_HC3_CR2_diagnostic",
            "status": np.where(observed, "observed", "not_estimable"),
            "reason_code": np.where(observed, None, "input_effect_not_estimable"),
            "prior_id": None,
            "fit_id": None,
        }
    )


def _tensor_result(
    estimates: pd.DataFrame,
    *,
    prior: FrozenHypergraphPrior,
    seed: int,
) -> tuple[pd.DataFrame, _TensorFit]:
    source = estimates.sort_values("edge_id", kind="stable", ignore_index=True)
    fit = _fit_tensor_factorization(source, prior=prior, seed=seed)
    observed = source["effect"].notna() & source["standard_error"].notna()
    return (
        pd.DataFrame(
            {
                "edge_id": source["edge_id"].astype(str),
                "raw_effect": source["effect"],
                "raw_standard_error": source["standard_error"],
                "topology_mean": fit.topology_mean,
                "prior_variance": fit.prior_variance,
                "shrinkage_factor": fit.shrinkage_factor,
                "posterior_effect": fit.posterior_effect,
                "posterior_standard_error": fit.posterior_standard_error,
                "posterior_se_scope": ("conditional_on_fitted_rank2_cp_tensor_mean"),
                "status": np.where(observed, "observed", "not_estimable"),
                "reason_code": np.where(observed, None, "input_effect_not_estimable"),
                "prior_id": prior.prior_id,
                "fit_id": str(
                    canonical_digest(
                        {
                            "prior_id": prior.prior_id,
                            "rank": _TENSOR_RANK,
                            "ridge": _TENSOR_RIDGE,
                            "seed": seed,
                            "posterior": [
                                None if not math.isfinite(value) else float(value)
                                for value in fit.posterior_effect
                            ],
                        }
                    )
                ),
            }
        ),
        fit,
    )


def _truth_table(truth: pd.DataFrame) -> pd.DataFrame:
    required = {
        "contrast_name",
        "sender",
        "receiver",
        "interaction_id",
        "truth_score_effect",
        "truth_causal_sender",
    }
    missing = required.difference(truth.columns)
    if not isinstance(truth, pd.DataFrame) or missing:
        raise ValueError(f"E5 truth is invalid: missing={sorted(missing)}")
    result = truth.loc[:, list(required)].copy()
    key = ["contrast_name", "sender", "receiver", "interaction_id"]
    if result.empty or result.duplicated(key).any():
        raise ValueError("E5 truth keys must be non-empty and unique")
    return result.rename(
        columns={
            "truth_score_effect": "truth_effect",
            "truth_causal_sender": "truth_label",
        }
    )


def _edge_table(
    fitted: pd.DataFrame,
    *,
    topology: pd.DataFrame,
    truth: pd.DataFrame,
    dataset_id: str,
    dgp_family: str,
    design_kind: str,
    contrast_name: str,
    arm: str,
) -> pd.DataFrame:
    result = fitted.merge(
        topology.loc[:, ["edge_id", "sender", "receiver", "interaction_id"]],
        on="edge_id",
        validate="one_to_one",
    )
    local_truth = truth.loc[truth["contrast_name"].eq(contrast_name)]
    result = result.merge(
        local_truth.drop(columns="contrast_name"),
        on=["sender", "receiver", "interaction_id"],
        how="left",
        validate="one_to_one",
    )
    result["truth_known"] = result["truth_label"].notna()
    result.insert(0, "score_view", arm)
    result.insert(0, "contrast_name", contrast_name)
    result.insert(0, "design_kind", design_kind)
    result.insert(0, "dgp_family", dgp_family)
    result.insert(0, "dataset_id", dataset_id)
    result.insert(0, "schema_version", SCHEMA_VERSION)
    result = result.rename(columns={"edge_id": "event_id"})
    result["formal_inference_allowed"] = False
    return result.loc[:, list(E5_EDGE_COLUMNS)].sort_values(
        "event_id", kind="stable", ignore_index=True
    )


def _metric_row(
    edge: pd.DataFrame,
    *,
    metric: str,
    value: float | None,
    reason_code: str | None,
    n_observed: int,
    n_positive: int,
    n_negative: int,
) -> dict[str, object]:
    first = edge.iloc[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": first["dataset_id"],
        "dgp_family": first["dgp_family"],
        "design_kind": first["design_kind"],
        "generator_id": E5_GENERATOR_ID,
        "score_view": first["score_view"],
        "estimand": "uncertainty_aware_hypergraph_shrunken_edge_effect",
        "resolution": "sender_lr_receiver_child",
        "inference_id": E5_INFERENCE_ID,
        "contrast_name": first["contrast_name"],
        "metric": metric,
        "value": value,
        "status": "observed" if value is not None else "not_estimable",
        "reason_code": reason_code if value is None else None,
        "n_observed": n_observed,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _method_metrics(edge: pd.DataFrame) -> list[dict[str, object]]:
    effect = pd.to_numeric(edge["posterior_effect"], errors="coerce")
    standard_error = pd.to_numeric(edge["posterior_standard_error"], errors="coerce")
    truth_effect = pd.to_numeric(edge["truth_effect"], errors="coerce")
    known = (
        edge["truth_known"].astype(bool)
        & edge["status"].eq("observed")
        & effect.notna()
        & standard_error.notna()
        & truth_effect.notna()
        & np.isfinite(effect)
        & np.isfinite(standard_error)
        & np.isfinite(truth_effect)
    )
    labels = edge.loc[known, "truth_label"].astype(bool).to_numpy()
    predicted = effect.loc[known].to_numpy(float)
    observed_truth = truth_effect.loc[known].to_numpy(float)
    observed_se = standard_error.loc[known].to_numpy(float)
    count = len(predicted)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - labels.sum())
    rows: list[dict[str, object]] = []
    mse = float(np.mean(np.square(predicted - observed_truth))) if count else None
    rows.append(
        _metric_row(
            edge,
            metric="effect_mse",
            value=mse,
            reason_code="no_known_effect_pairs" if not count else None,
            n_observed=count,
            n_positive=n_positive,
            n_negative=n_negative,
        )
    )
    spearman = None
    if (
        count >= 3
        and len(np.unique(predicted)) > 1
        and len(np.unique(observed_truth)) > 1
    ):
        spearman = float(spearmanr(predicted, observed_truth).statistic)
    rows.append(
        _metric_row(
            edge,
            metric="effect_spearman",
            value=spearman,
            reason_code=(
                "fewer_than_three_or_constant_effects" if spearman is None else None
            ),
            n_observed=count,
            n_positive=n_positive,
            n_negative=n_negative,
        )
    )
    nonzero = observed_truth != 0.0
    direction = (
        float(np.mean(np.sign(predicted[nonzero]) == np.sign(observed_truth[nonzero])))
        if nonzero.any()
        else None
    )
    rows.append(
        _metric_row(
            edge,
            metric="direction_accuracy",
            value=direction,
            reason_code="no_nonzero_truth_effects" if direction is None else None,
            n_observed=int(nonzero.sum()),
            n_positive=n_positive,
            n_negative=n_negative,
        )
    )
    lower = predicted - 1.96 * observed_se
    upper = predicted + 1.96 * observed_se
    coverage = (
        float(np.mean((observed_truth >= lower) & (observed_truth <= upper)))
        if count
        else None
    )
    rows.append(
        _metric_row(
            edge,
            metric="ci_coverage",
            value=coverage,
            reason_code="no_known_effect_intervals" if coverage is None else None,
            n_observed=count,
            n_positive=n_positive,
            n_negative=n_negative,
        )
    )
    weak = np.zeros(count, dtype=bool)
    if nonzero.any():
        threshold = float(np.median(np.abs(observed_truth[nonzero])))
        weak = nonzero & (np.abs(observed_truth) <= threshold)
    detected = ((observed_truth > 0.0) & (lower > 0.0)) | (
        (observed_truth < 0.0) & (upper < 0.0)
    )
    weak_power = float(np.mean(detected[weak])) if weak.any() else None
    rows.append(
        _metric_row(
            edge,
            metric="weak_signal_power",
            value=weak_power,
            reason_code=(
                "no_bottom_half_nonzero_effects" if weak_power is None else None
            ),
            n_observed=int(weak.sum()),
            n_positive=n_positive,
            n_negative=n_negative,
        )
    )
    exact_ap = (
        float(average_precision_score(labels.astype(int), np.abs(predicted)))
        if n_positive and n_negative
        else None
    )
    rows.append(
        _metric_row(
            edge,
            metric="exact_hyperedge_ap",
            value=exact_ap,
            reason_code="truth_has_single_class" if exact_ap is None else None,
            n_observed=count,
            n_positive=n_positive,
            n_negative=n_negative,
        )
    )
    return rows


def _topology_advantage_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (_, _contrast), group in metrics.loc[
        metrics["metric"].eq("effect_mse")
    ].groupby(["dataset_id", "contrast_name"], observed=True, sort=True):
        baseline = group.loc[
            group["score_view"].eq("degree_matched_permuted_hypergraph")
            & group["status"].eq("observed")
        ]
        if len(baseline) != 1:
            raise ValueError("E5 lacks one observed permuted-topology MSE")
        baseline_mse = float(baseline.iloc[0]["value"])
        for row in group.itertuples(index=False):
            value = (
                baseline_mse - float(row.value)
                if row.status == "observed" and pd.notna(row.value)
                else None
            )
            records.append(
                {
                    **row._asdict(),
                    "metric": "topology_permutation_advantage",
                    "value": value,
                    "status": "observed" if value is not None else "not_estimable",
                    "reason_code": (
                        None if value is not None else "effect_mse_not_estimable"
                    ),
                }
            )
    return pd.DataFrame.from_records(records, columns=METRIC_COLUMNS)


@dataclass(frozen=True, slots=True)
class V7E5HypergraphSwapResult:
    """Complete descriptive E5 result on one generated dataset."""

    edge_estimates: pd.DataFrame
    metrics: pd.DataFrame
    fit_diagnostics: pd.DataFrame
    topology_diagnostics: pd.DataFrame

    def __post_init__(self) -> None:
        expected = (
            (self.edge_estimates, E5_EDGE_COLUMNS, "edge_estimates"),
            (self.metrics, METRIC_COLUMNS, "metrics"),
            (self.fit_diagnostics, E5_FIT_COLUMNS, "fit_diagnostics"),
            (
                self.topology_diagnostics,
                E5_TOPOLOGY_COLUMNS,
                "topology_diagnostics",
            ),
        )
        for table, columns, name in expected:
            if not isinstance(table, pd.DataFrame) or tuple(table.columns) != columns:
                raise ValueError(f"E5 {name} does not match its frozen contract")
            object.__setattr__(self, name, table.copy(deep=True))
        for table in (self.edge_estimates, self.metrics, self.fit_diagnostics):
            if set(table["score_view"].astype(str)) != set(E5_ARMS):
                raise ValueError("E5 result does not cover every preregistered arm")
        if set(self.metrics["metric"].astype(str)) != set(E5_METRICS):
            raise ValueError("E5 result does not cover every preregistered metric")
        if self.edge_estimates["formal_inference_allowed"].any():
            raise ValueError("E5 component swaps cannot claim formal inference")


def run_v7_e5_hypergraph_swap(
    effects: pd.DataFrame,
    *,
    resource: ResourceBundle,
    truth: pd.DataFrame,
    dataset_id: str,
    dgp_family: str,
    design_kind: str,
    root_seed: int,
) -> V7E5HypergraphSwapResult:
    """Run all E5 arms on the same G3-I1 edge-effect universe."""

    dataset = _name(dataset_id, field="dataset_id")
    family = _name(dgp_family, field="dgp_family")
    design = _name(design_kind, field="design_kind")
    source = _effect_universe(effects)
    topology = _resource_topology(source, resource)
    full = freeze_hypergraph_prior(topology, view_columns=_TOPOLOGY_VIEWS)
    controls = _prior_controls(full, root_seed=root_seed)
    topology_diagnostics = _topology_diagnostics(
        dataset_id=dataset,
        full=full,
        controls=controls,
    )
    truth_table = _truth_table(truth)
    spec = UncertaintyAwareHypergraphShrinkageV2Spec(minimum_observed_edges=4)
    edge_parts: list[pd.DataFrame] = []
    fit_records: list[dict[str, object]] = []
    for contrast_name, local in source.groupby(
        "contrast_name", observed=True, sort=True
    ):
        contrast = str(contrast_name)
        estimates = local.loc[:, ["event_id", "effect", "standard_error"]].rename(
            columns={"event_id": "edge_id"}
        )
        raw = _raw_result(estimates)
        method_results: dict[str, pd.DataFrame] = {"no_prior": raw}
        fit_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset,
                "dgp_family": family,
                "design_kind": design,
                "contrast_name": contrast,
                "score_view": "no_prior",
                "method_kind": "raw_G3_I1_effect",
                "prior_id": None,
                "fit_id": None,
                "edge_count": len(raw),
                "observed_edge_count": int(raw["status"].eq("observed").sum()),
                "incidence_column_count": 0,
                "incidence_nnz": 0,
                "prior_variance": None,
                "iterations": 0,
                "converged": True,
                "tensor_rank": None,
                "formal_inference_allowed": False,
            }
        )
        for arm, prior in controls.items():
            result, fit = fit_uncertainty_aware_hypergraph_shrinkage_v2(
                estimates,
                prior=prior,
                spec=spec,
            )
            method_results[arm] = result.loc[
                :,
                [
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
                ],
            ].assign(fit_id=fit.fit_id)
            fit_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": dataset,
                    "dgp_family": family,
                    "design_kind": design,
                    "contrast_name": contrast,
                    "score_view": arm,
                    "method_kind": (
                        "pairwise_clique_ridge_eb"
                        if arm == "pairwise_clique_expansion"
                        else "sparse_incidence_ridge_eb"
                    ),
                    "prior_id": prior.prior_id,
                    "fit_id": fit.fit_id,
                    "edge_count": fit.edge_count,
                    "observed_edge_count": fit.observed_edge_count,
                    "incidence_column_count": fit.incidence_column_count,
                    "incidence_nnz": fit.incidence_nnz,
                    "prior_variance": fit.prior_variance,
                    "iterations": fit.iterations,
                    "converged": fit.converged,
                    "tensor_rank": None,
                    "formal_inference_allowed": False,
                }
            )
        tensor_seed = _derived_seed(root_seed, f"tensor:{contrast}")
        tensor, tensor_fit = _tensor_result(
            estimates,
            prior=full,
            seed=tensor_seed,
        )
        method_results["tensor_factorization"] = tensor
        tensor_fit_id = str(tensor["fit_id"].iloc[0])
        fit_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset,
                "dgp_family": family,
                "design_kind": design,
                "contrast_name": contrast,
                "score_view": "tensor_factorization",
                "method_kind": "fixed_rank_cp_tensor_als_eb",
                "prior_id": full.prior_id,
                "fit_id": tensor_fit_id,
                "edge_count": len(tensor),
                "observed_edge_count": int(tensor["status"].eq("observed").sum()),
                "incidence_column_count": sum(
                    len({edge.memberships[index] for edge in full.edges})
                    for index in range(len(full.view_names))
                )
                * _TENSOR_RANK,
                "incidence_nnz": len(full.edges) * len(full.view_names) * _TENSOR_RANK,
                "prior_variance": tensor_fit.prior_variance,
                "iterations": tensor_fit.iterations,
                "converged": tensor_fit.converged,
                "tensor_rank": _TENSOR_RANK,
                "formal_inference_allowed": False,
            }
        )
        if set(method_results) != set(E5_ARMS):
            raise RuntimeError("E5 did not construct every preregistered arm")
        for arm in E5_ARMS:
            edge_parts.append(
                _edge_table(
                    method_results[arm],
                    topology=topology,
                    truth=truth_table,
                    dataset_id=dataset,
                    dgp_family=family,
                    design_kind=design,
                    contrast_name=contrast,
                    arm=arm,
                )
            )
    edge_estimates = pd.concat(edge_parts, ignore_index=True).loc[
        :, list(E5_EDGE_COLUMNS)
    ]
    metric_records: list[dict[str, object]] = []
    for _, group in edge_estimates.groupby(
        ["contrast_name", "score_view"], observed=True, sort=True
    ):
        metric_records.extend(_method_metrics(group))
    metrics = pd.DataFrame.from_records(metric_records, columns=METRIC_COLUMNS)
    metrics = pd.concat(
        [metrics, _topology_advantage_rows(metrics)], ignore_index=True
    ).sort_values(
        ["score_view", "contrast_name", "metric"],
        kind="stable",
        ignore_index=True,
    )
    fit_diagnostics = pd.DataFrame.from_records(
        fit_records, columns=E5_FIT_COLUMNS
    ).sort_values(["score_view", "contrast_name"], kind="stable", ignore_index=True)
    edge_estimates = edge_estimates.sort_values(
        ["score_view", "contrast_name", "event_id"],
        kind="stable",
        ignore_index=True,
    )
    return V7E5HypergraphSwapResult(
        edge_estimates=edge_estimates,
        metrics=metrics.loc[:, list(METRIC_COLUMNS)],
        fit_diagnostics=fit_diagnostics.loc[:, list(E5_FIT_COLUMNS)],
        topology_diagnostics=topology_diagnostics.loc[:, list(E5_TOPOLOGY_COLUMNS)],
    )


__all__ = [
    "E5_ARMS",
    "E5_EDGE_COLUMNS",
    "E5_FIT_COLUMNS",
    "E5_GENERATOR_ID",
    "E5_INFERENCE_ID",
    "E5_METRICS",
    "E5_TOPOLOGY_COLUMNS",
    "SCHEMA_VERSION",
    "V7E5HypergraphSwapResult",
    "run_v7_e5_hypergraph_swap",
]
