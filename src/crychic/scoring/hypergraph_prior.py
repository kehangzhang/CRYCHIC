"""Outcome-blind frozen molecular hypergraph prior and descriptive shrinkage."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crychic.core import canonical_digest, stable_id

HYPERGRAPH_PRIOR_VERSION = "frozen_multiview_hprior_m5_v1"
HYPERGRAPH_SHRINKAGE_VERSION = "additive_incidence_ridge_m5_v1"
_SCHEMA_VERSION = "1.0.0"


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a canonical non-empty string")
    return value


def _names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(_name(value, field_name=field_name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique names")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenHyperedgePrior:
    """One outcome-blind edge and its ordered molecular memberships."""

    edge_id: str
    memberships: tuple[str, ...]

    def __post_init__(self) -> None:
        edge_id = _name(self.edge_id, field_name="edge_id")
        memberships = tuple(
            _name(value, field_name="membership") for value in self.memberships
        )
        if not memberships:
            raise ValueError("hyperedge memberships cannot be empty")
        object.__setattr__(self, "edge_id", edge_id)
        object.__setattr__(self, "memberships", memberships)

    def to_dict(self, view_names: Sequence[str]) -> dict[str, object]:
        if len(view_names) != len(self.memberships):
            raise ValueError("view names and memberships differ")
        return {
            "edge_id": self.edge_id,
            "memberships": dict(zip(view_names, self.memberships, strict=True)),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenHypergraphPrior:
    """Immutable molecular H_prior frozen before any outcome fit."""

    view_names: tuple[str, ...]
    edges: tuple[FrozenHyperedgePrior, ...]
    source_digest: str
    topology_kind: str = "declared"
    parent_prior_id: str | None = None
    permutation_seed: int | None = None
    schema_version: str = _SCHEMA_VERSION
    prior_id: str = field(init=False)

    def __post_init__(self) -> None:
        views = _names(self.view_names, field_name="view_names")
        edges = tuple(self.edges)
        if not edges or any(
            not isinstance(edge, FrozenHyperedgePrior) for edge in edges
        ):
            raise ValueError("edges must contain frozen hyperedge priors")
        edges = tuple(sorted(edges, key=lambda edge: edge.edge_id))
        edge_ids = tuple(edge.edge_id for edge in edges)
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("frozen hypergraph edge IDs must be unique")
        if any(len(edge.memberships) != len(views) for edge in edges):
            raise ValueError("every hyperedge must cover every declared view")
        if len(self.source_digest) != 64:
            raise ValueError("source_digest must be a canonical digest")
        topology_kind = _name(self.topology_kind, field_name="topology_kind")
        parent = (
            None
            if self.parent_prior_id is None
            else _name(self.parent_prior_id, field_name="parent_prior_id")
        )
        seed = self.permutation_seed
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("permutation_seed must be an integer or None")
        if topology_kind == "declared" and (parent is not None or seed is not None):
            raise ValueError("declared priors cannot have permutation parents")
        if topology_kind == "degree_matched_permutation" and (
            parent is None or seed is None
        ):
            raise ValueError("permuted priors require parent and seed")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "view_names", views)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "topology_kind", topology_kind)
        object.__setattr__(self, "parent_prior_id", parent)
        object.__setattr__(
            self,
            "prior_id",
            stable_id(
                "frozen_hypergraph_prior",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_version": HYPERGRAPH_PRIOR_VERSION,
            "view_names": list(self.view_names),
            "edges": [edge.to_dict(self.view_names) for edge in self.edges],
            "source_digest": self.source_digest,
            "topology_kind": self.topology_kind,
            "parent_prior_id": self.parent_prior_id,
            "permutation_seed": self.permutation_seed,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {"prior_id": self.prior_id, **self._identity_payload()}

    def degree_profile(self) -> dict[str, dict[str, int]]:
        return {
            view: dict(
                sorted(Counter(edge.memberships[index] for edge in self.edges).items())
            )
            for index, view in enumerate(self.view_names)
        }


def freeze_hypergraph_prior(
    edge_table: pd.DataFrame,
    *,
    view_columns: Sequence[str],
    edge_id_column: str = "edge_id",
) -> FrozenHypergraphPrior:
    """Freeze an outcome-blind edge-by-view table as H_prior."""

    if not isinstance(edge_table, pd.DataFrame):
        raise TypeError("edge_table must be a pandas DataFrame")
    views = _names(view_columns, field_name="view_columns")
    edge_id_column = _name(edge_id_column, field_name="edge_id_column")
    required = {edge_id_column, *views}
    missing = required.difference(edge_table.columns)
    if missing:
        raise ValueError(f"hypergraph prior is missing columns: {sorted(missing)}")
    source = edge_table.loc[:, [edge_id_column, *views]].copy()
    if source.empty or source.isna().any().any():
        raise ValueError("hypergraph prior columns must be complete")
    source = source.astype(str)
    for column in source.columns:
        invalid = source[column].eq("") | source[column].str.strip().ne(source[column])
        if invalid.any():
            raise ValueError("hypergraph prior labels must be canonical")
    if source[edge_id_column].duplicated().any():
        raise ValueError("hypergraph prior edge IDs must be unique")
    source = source.sort_values(edge_id_column, kind="stable", ignore_index=True)
    digest_records = source.rename(columns={edge_id_column: "edge_id"}).to_dict(
        orient="records"
    )
    edges = tuple(
        FrozenHyperedgePrior(
            edge_id=str(row[0]),
            memberships=tuple(map(str, row[1:])),
        )
        for row in source.itertuples(index=False, name=None)
    )
    return FrozenHypergraphPrior(
        view_names=views,
        edges=edges,
        source_digest=canonical_digest(digest_records),
    )


def select_hypergraph_prior_views(
    prior: FrozenHypergraphPrior, *, view_names: Sequence[str]
) -> FrozenHypergraphPrior:
    """Create an authenticated partial-view control from one frozen prior."""

    if not isinstance(prior, FrozenHypergraphPrior):
        raise TypeError("prior must be a FrozenHypergraphPrior")
    selected = _names(view_names, field_name="view_names")
    if not set(selected).issubset(prior.view_names):
        raise ValueError("selected views must be present in the parent prior")
    indices = tuple(prior.view_names.index(view) for view in selected)
    records = [
        {
            "edge_id": edge.edge_id,
            **{
                view: edge.memberships[index]
                for view, index in zip(selected, indices, strict=True)
            },
        }
        for edge in prior.edges
    ]
    subset = freeze_hypergraph_prior(
        pd.DataFrame.from_records(records), view_columns=selected
    )
    return FrozenHypergraphPrior(
        view_names=subset.view_names,
        edges=subset.edges,
        source_digest=canonical_digest(
            {"parent_prior_id": prior.prior_id, "view_names": list(selected)},
        ),
        topology_kind=f"declared_view_subset:{','.join(selected)}",
    )


def permute_hypergraph_prior_degree_matched(
    prior: FrozenHypergraphPrior, *, seed: int
) -> FrozenHypergraphPrior:
    """Independently permute view stubs, preserving every node degree exactly."""

    if not isinstance(prior, FrozenHypergraphPrior):
        raise TypeError("prior must be a FrozenHypergraphPrior")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    memberships = np.asarray([edge.memberships for edge in prior.edges], dtype=object)
    permuted = memberships.copy()
    changed = False
    for index in range(len(prior.view_names)):
        values = memberships[:, index]
        if len(set(map(str, values))) < 2:
            continue
        order = rng.permutation(len(values))
        if np.array_equal(order, np.arange(len(values))):
            order = np.roll(order, 1)
        permuted[:, index] = values[order]
        changed = changed or not np.array_equal(permuted[:, index], values)
    if not changed:
        raise ValueError("degree-matched permutation cannot alter this prior")
    edges = tuple(
        FrozenHyperedgePrior(edge_id=edge.edge_id, memberships=tuple(map(str, row)))
        for edge, row in zip(prior.edges, permuted, strict=True)
    )
    result = FrozenHypergraphPrior(
        view_names=prior.view_names,
        edges=edges,
        source_digest=canonical_digest(
            {"parent_prior_id": prior.prior_id, "seed": seed},
        ),
        topology_kind="degree_matched_permutation",
        parent_prior_id=prior.prior_id,
        permutation_seed=seed,
    )
    if result.degree_profile() != prior.degree_profile():
        raise RuntimeError("degree-matched permutation changed node degrees")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class HypergraphShrinkageSpec:
    """Fixed node-ridge and edge-residual policy for H_prior shrinkage."""

    node_ridge_penalty: float = 1.0
    edge_residual_penalty: float = 1.0
    schema_version: str = _SCHEMA_VERSION
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        node_penalty = float(self.node_ridge_penalty)
        edge_penalty = float(self.edge_residual_penalty)
        if not math.isfinite(node_penalty) or node_penalty <= 0.0:
            raise ValueError("node_ridge_penalty must be finite and positive")
        if not math.isfinite(edge_penalty) or edge_penalty < 0.0:
            raise ValueError("edge_residual_penalty must be finite and non-negative")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        object.__setattr__(self, "node_ridge_penalty", node_penalty)
        object.__setattr__(self, "edge_residual_penalty", edge_penalty)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "hypergraph_shrinkage_spec",
                self._identity_payload(),
                schema_version=self.schema_version,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_version": HYPERGRAPH_SHRINKAGE_VERSION,
            "node_ridge_penalty": self.node_ridge_penalty,
            "edge_residual_penalty": self.edge_residual_penalty,
            "model": "beta_equals_incidence_theta_plus_edge_residual",
            "solver": "closed_form_weighted_ridge_v1",
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {"spec_id": self.spec_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True, kw_only=True)
class HypergraphShrinkageFit:
    """Immutable fit diagnostics binding one H_prior and estimate table."""

    prior_id: str
    spec_id: str
    input_digest: str
    output_digest: str
    edge_count: int
    view_count: int
    converged: bool
    iterations: int
    maximum_update: float
    formal_inference_allowed: bool = False
    fit_id: str = field(init=False)

    def __post_init__(self) -> None:
        _name(self.prior_id, field_name="prior_id")
        _name(self.spec_id, field_name="spec_id")
        if len(self.input_digest) != 64 or len(self.output_digest) != 64:
            raise ValueError("shrinkage digests must be canonical")
        for value, name in (
            (self.edge_count, "edge_count"),
            (self.view_count, "view_count"),
            (self.iterations, "iterations"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.maximum_update) or self.maximum_update < 0.0:
            raise ValueError("maximum_update must be finite and non-negative")
        if self.formal_inference_allowed is not False:
            raise ValueError("H_prior shrinkage is descriptive")
        object.__setattr__(
            self,
            "fit_id",
            stable_id(
                "hypergraph_shrinkage_fit",
                self._identity_payload(),
                schema_version=_SCHEMA_VERSION,
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "algorithm_version": HYPERGRAPH_SHRINKAGE_VERSION,
            "prior_id": self.prior_id,
            "spec_id": self.spec_id,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "edge_count": self.edge_count,
            "view_count": self.view_count,
            "converged": self.converged,
            "iterations": self.iterations,
            "maximum_update": self.maximum_update,
            "formal_inference_allowed": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {"fit_id": self.fit_id, **self._identity_payload()}


def _incidence_matrix(prior: FrozenHypergraphPrior) -> np.ndarray:
    blocks: list[np.ndarray] = []
    rows = np.arange(len(prior.edges))
    for index in range(len(prior.view_names)):
        codes, levels = pd.factorize(
            np.asarray([edge.memberships[index] for edge in prior.edges], dtype=object),
            sort=True,
        )
        block = np.zeros((len(prior.edges), len(levels)), dtype=float)
        block[rows, codes] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def fit_hypergraph_prior_shrinkage(
    estimates: pd.DataFrame,
    *,
    prior: FrozenHypergraphPrior,
    spec: HypergraphShrinkageSpec | None = None,
    edge_id_column: str = "edge_id",
    estimate_column: str = "estimate",
    precision_column: str | None = None,
) -> tuple[pd.DataFrame, HypergraphShrinkageFit]:
    """Fit ``beta = B theta + delta`` using a frozen incidence matrix."""

    if not isinstance(estimates, pd.DataFrame):
        raise TypeError("estimates must be a pandas DataFrame")
    if not isinstance(prior, FrozenHypergraphPrior):
        raise TypeError("prior must be a FrozenHypergraphPrior")
    resolved = spec or HypergraphShrinkageSpec()
    if not isinstance(resolved, HypergraphShrinkageSpec):
        raise TypeError("spec must be a HypergraphShrinkageSpec or None")
    edge_id_column = _name(edge_id_column, field_name="edge_id_column")
    estimate_column = _name(estimate_column, field_name="estimate_column")
    required = {edge_id_column, estimate_column}
    if precision_column is not None:
        precision_column = _name(precision_column, field_name="precision_column")
        required.add(precision_column)
    missing = required.difference(estimates.columns)
    if missing:
        raise ValueError(f"shrinkage estimates are missing columns: {sorted(missing)}")
    source = estimates.loc[:, sorted(required)].copy()
    if source.empty or source[edge_id_column].isna().any():
        raise ValueError("shrinkage estimates require non-empty edge IDs")
    source[edge_id_column] = source[edge_id_column].astype(str)
    if source[edge_id_column].duplicated().any():
        raise ValueError("shrinkage estimates require one row per edge")
    source = source.sort_values(edge_id_column, kind="stable", ignore_index=True)
    prior_ids = tuple(edge.edge_id for edge in prior.edges)
    if tuple(source[edge_id_column]) != prior_ids:
        raise ValueError("estimate edge universe differs from frozen H_prior")
    values = pd.to_numeric(source[estimate_column], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("edge estimates must be finite")
    if precision_column is None:
        precision = np.ones(len(values), dtype=float)
    else:
        precision = pd.to_numeric(source[precision_column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(precision).all() or (precision <= 0.0).any():
            raise ValueError("observation precision must be finite and positive")
    if resolved.edge_residual_penalty == 0.0:
        estimate = values.copy()
    else:
        incidence = _incidence_matrix(prior)
        design = np.column_stack([np.ones(len(values), dtype=float), incidence])
        effective_weight = (
            precision
            * resolved.edge_residual_penalty
            / (precision + resolved.edge_residual_penalty)
        )
        penalty = np.full(design.shape[1], resolved.node_ridge_penalty, dtype=float)
        penalty[0] = 0.0
        left = design.T @ (effective_weight[:, np.newaxis] * design)
        left += np.diag(penalty)
        right = design.T @ (effective_weight * values)
        coefficients = np.linalg.solve(left, right)
        node_prediction = design @ coefficients
        edge_residual = (
            precision
            / (precision + resolved.edge_residual_penalty)
            * (values - node_prediction)
        )
        estimate = node_prediction + edge_residual
    maximum_update = float(np.max(np.abs(estimate - values)))
    converged = True
    iterations = 1
    input_records = [
        {
            "edge_id": edge_id,
            "estimate": float(value),
            "precision": float(weight),
        }
        for edge_id, value, weight in zip(prior_ids, values, precision, strict=True)
    ]
    output = pd.DataFrame(
        {
            "edge_id": prior_ids,
            "raw_estimate": values,
            "shrunk_estimate": estimate,
            "prior_id": prior.prior_id,
            "spec_id": resolved.spec_id,
            "formal_inference_allowed": False,
            "score_version": HYPERGRAPH_SHRINKAGE_VERSION,
        }
    )
    output["shrinkage_record_id"] = [
        stable_id(
            "hypergraph_shrinkage_record",
            {
                "edge_id": edge_id,
                "raw_estimate": float(raw),
                "shrunk_estimate": float(shrunk),
                "prior_id": prior.prior_id,
                "spec_id": resolved.spec_id,
            },
            schema_version=_SCHEMA_VERSION,
        )
        for edge_id, raw, shrunk in zip(prior_ids, values, estimate, strict=True)
    ]
    output_digest = canonical_digest(
        output.loc[:, ["edge_id", "raw_estimate", "shrunk_estimate"]].to_dict(
            orient="records"
        )
    )
    fit = HypergraphShrinkageFit(
        prior_id=prior.prior_id,
        spec_id=resolved.spec_id,
        input_digest=canonical_digest(input_records),
        output_digest=output_digest,
        edge_count=len(prior_ids),
        view_count=len(prior.view_names),
        converged=converged,
        iterations=iterations,
        maximum_update=maximum_update,
    )
    return output, fit


__all__ = [
    "HYPERGRAPH_PRIOR_VERSION",
    "HYPERGRAPH_SHRINKAGE_VERSION",
    "FrozenHyperedgePrior",
    "FrozenHypergraphPrior",
    "HypergraphShrinkageFit",
    "HypergraphShrinkageSpec",
    "fit_hypergraph_prior_shrinkage",
    "freeze_hypergraph_prior",
    "permute_hypergraph_prior_degree_matched",
    "select_hypergraph_prior_views",
]
