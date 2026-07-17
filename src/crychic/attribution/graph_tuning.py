"""Subject-blocked penalty selection for experimental graph-fused attribution."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy import sparse

from crychic.core import ContractError, stable_id
from crychic.design import ContextGraph

from .contracts import FamilyFirstBasis
from .graph_fused import (
    GraphFusedFamilyAttributionResult,
    fit_graph_fused_family_attribution,
    solve_graph_fused_nonnegative_elastic_net,
)

_SELECTION_RULE = "paired_subject_delta_one_se_fusion_l1_l2_priority_v1"
_VALIDATION_ESTIMAND = "subject_equal_full_context_weighted_prediction_loss_v1"


def _penalty(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be numeric, not boolean")
    try:
        resolved = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{field_name} must be a finite non-negative scalar"
        ) from error
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{field_name} must be a finite non-negative scalar")
    return 0.0 if resolved == 0.0 else resolved


def _names(
    values: Sequence[str], *, field_name: str, minimum: int = 1
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    resolved = tuple(values)
    if len(resolved) < minimum or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in resolved
    ):
        raise ValueError(
            f"{field_name} must contain at least {minimum} canonical names"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field_name} must contain unique values")
    return resolved


def _frozen_vector(values: np.ndarray, *, length: int, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype="<f8").copy(order="C")
    if array.shape != (length,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{field_name} must be a finite aligned vector")
    array[array == 0.0] = 0.0
    immutable = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.frombuffer(array.tobytes(order="C"), dtype="<f8"),
    )
    immutable.setflags(write=False)
    return immutable


def _frozen_matrix(
    values: sparse.spmatrix, *, shape: tuple[int, int], field_name: str
) -> sparse.csc_matrix:
    matrix = sparse.csc_matrix(values, dtype="<f8", copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if (
        matrix.shape != shape
        or np.any(~np.isfinite(matrix.data))
        or np.any(matrix.data < 0)
    ):
        raise ValueError(
            f"{field_name} must be a finite non-negative aligned sparse matrix"
        )
    for values_array in (matrix.data, matrix.indices, matrix.indptr):
        values_array.setflags(write=False)
    return matrix


def _array_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _matrix_digest(matrix: sparse.csc_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    return digest.hexdigest()


def _graph_id(graph: ContextGraph) -> str:
    identifier: str = stable_id("context_graph", graph.to_dict(), schema_version="1")
    return identifier


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphPenaltyCandidate:
    """One explicitly pre-registered absolute graph-penalty candidate."""

    lambda1: float
    lambda2: float
    lambda_f: float
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            "lambda1": _penalty(self.lambda1, field_name="lambda1"),
            "lambda2": _penalty(self.lambda2, field_name="lambda2"),
            "lambda_f": _penalty(self.lambda_f, field_name="lambda_f"),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "candidate_id",
            stable_id("graph_penalty_candidate", values, schema_version="1"),
        )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "candidate_id": self.candidate_id,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "lambda_f": self.lambda_f,
        }

    def _require_intact(self) -> None:
        try:
            repeated = GraphPenaltyCandidate(
                lambda1=self.lambda1,
                lambda2=self.lambda2,
                lambda_f=self.lambda_f,
            )
            valid = self.candidate_id == repeated.candidate_id
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Graph penalty candidate identity is not intact",
                code="graph_penalty_candidate_integrity_violation",
                field="candidate_id",
                remediation="Recreate the pre-registered candidate",
            ) from error
        if not valid:
            raise ContractError(
                "Graph penalty candidate identity is not intact",
                code="graph_penalty_candidate_integrity_violation",
                field="candidate_id",
                remediation="Recreate the pre-registered candidate",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphPenaltyTuningSpec:
    """Cartesian grid and deterministic paired one-SE selection policy."""

    lambda1_values: tuple[float, ...] = (0.0,)
    lambda2_values: tuple[float, ...] = (0.0,)
    lambda_f_values: tuple[float, ...] = (0.0,)
    coefficient_selection_tolerance: float = 1e-8
    selection_rule: str = _SELECTION_RULE
    candidates: tuple[GraphPenaltyCandidate, ...] = field(init=False)
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        grids: dict[str, tuple[float, ...]] = {}
        for field_name in ("lambda1_values", "lambda2_values", "lambda_f_values"):
            supplied = tuple(getattr(self, field_name))
            if not supplied:
                raise ValueError(f"{field_name} must not be empty")
            grids[field_name] = tuple(
                sorted(
                    {_penalty(value, field_name=field_name) for value in supplied},
                    reverse=True,
                )
            )
        selection_tolerance = _penalty(
            self.coefficient_selection_tolerance,
            field_name="coefficient_selection_tolerance",
        )
        if selection_tolerance <= 0:
            raise ValueError("coefficient_selection_tolerance must be positive")
        if self.selection_rule != _SELECTION_RULE:
            raise ValueError(f"selection_rule must be {_SELECTION_RULE!r}")
        candidates = tuple(
            GraphPenaltyCandidate(lambda1=l1, lambda2=l2, lambda_f=lf)
            for lf in grids["lambda_f_values"]
            for l1 in grids["lambda1_values"]
            for l2 in grids["lambda2_values"]
        )
        payload = {
            **{name: list(values) for name, values in grids.items()},
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "coefficient_selection_tolerance": selection_tolerance,
            "selection_rule": self.selection_rule,
            "penalty_scale": "absolute_on_normalized_family_basis_v1",
        }
        for name, values in grids.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "coefficient_selection_tolerance", selection_tolerance)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "spec_id",
            stable_id("graph_penalty_tuning_spec", payload, schema_version="1"),
        )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "spec_id": self.spec_id,
            "lambda1_values": list(self.lambda1_values),
            "lambda2_values": list(self.lambda2_values),
            "lambda_f_values": list(self.lambda_f_values),
            "coefficient_selection_tolerance": (self.coefficient_selection_tolerance),
            "selection_rule": self.selection_rule,
            "penalty_scale": "absolute_on_normalized_family_basis_v1",
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    def _require_intact(self) -> None:
        try:
            for candidate in self.candidates:
                candidate._require_intact()
            repeated = GraphPenaltyTuningSpec(
                lambda1_values=self.lambda1_values,
                lambda2_values=self.lambda2_values,
                lambda_f_values=self.lambda_f_values,
                coefficient_selection_tolerance=(self.coefficient_selection_tolerance),
                selection_rule=self.selection_rule,
            )
            valid = (
                self.candidates == repeated.candidates
                and self.spec_id == repeated.spec_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Graph penalty tuning specification identity is not intact",
                code="graph_penalty_tuning_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the pre-registered graph penalty grid",
            ) from error
        if not valid:
            raise ContractError(
                "Graph penalty tuning specification identity is not intact",
                code="graph_penalty_tuning_spec_integrity_violation",
                field="spec_id",
                remediation="Recreate the pre-registered graph penalty grid",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphTuningFold:
    """One inner-training problem and subject-indexed held-out responses."""

    fold_id: str
    graph: ContextGraph
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    validation_subject_ids: tuple[str, ...]
    matrices: tuple[sparse.csc_matrix, ...]
    training_responses: tuple[np.ndarray, ...]
    training_precision_weights: tuple[np.ndarray, ...]
    validation_responses: tuple[tuple[np.ndarray, ...], ...]
    validation_precision_weights: tuple[tuple[np.ndarray, ...], ...]
    family_parent_set_id: str | None = None
    fold_data_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fold_id, str)
            or not self.fold_id
            or self.fold_id != self.fold_id.strip()
        ):
            raise ValueError("fold_id must be a non-empty canonical string")
        if not isinstance(self.graph, ContextGraph):
            raise TypeError("graph must be a ContextGraph")
        features = _names(self.feature_ids, field_name="feature_ids")
        families = _names(self.family_ids, field_name="family_ids")
        training_subjects = tuple(
            sorted(
                _names(
                    self.training_subject_ids,
                    field_name="training_subject_ids",
                    minimum=2,
                )
            )
        )
        validation_subjects = _names(
            self.validation_subject_ids,
            field_name="validation_subject_ids",
            minimum=1,
        )
        if validation_subjects != tuple(sorted(validation_subjects)):
            raise ValueError(
                "validation_subject_ids must be sorted to align validation arrays"
            )
        if set(training_subjects).intersection(validation_subjects):
            raise ContractError(
                "Graph tuning subjects must be disjoint within every fold",
                code="graph_tuning_subject_leakage",
                field="training_subject_ids,validation_subject_ids",
                remediation="Construct subject-blocked inner folds",
            )
        parent_set_id = self.family_parent_set_id
        if parent_set_id is not None and (
            not isinstance(parent_set_id, str)
            or not parent_set_id
            or parent_set_id != parent_set_id.strip()
        ):
            raise ValueError("family_parent_set_id must be a canonical identifier")
        context_count = len(self.graph.nodes)
        if any(
            len(group) != context_count
            for group in (
                self.matrices,
                self.training_responses,
                self.training_precision_weights,
            )
        ):
            raise ValueError("training arrays must align with graph.nodes")
        if len(self.validation_responses) != len(validation_subjects) or len(
            self.validation_precision_weights
        ) != len(validation_subjects):
            raise ValueError("validation arrays must align with validation subjects")
        matrices = tuple(
            _frozen_matrix(
                matrix,
                shape=(len(features), len(families)),
                field_name="matrices",
            )
            for matrix in self.matrices
        )
        training_responses = tuple(
            _frozen_vector(
                values,
                length=len(features),
                field_name="training_responses",
            )
            for values in self.training_responses
        )
        if any(np.any(values < 0) for values in training_responses):
            raise ValueError("training_responses must be direction-compatible")
        training_precision = tuple(
            _frozen_vector(
                values,
                length=len(features),
                field_name="training_precision_weights",
            )
            for values in self.training_precision_weights
        )
        if any(
            np.any(values < 0) or not np.any(values > 0)
            for values in training_precision
        ):
            raise ValueError("training precision must be non-negative and not all zero")

        validation_responses: list[tuple[np.ndarray, ...]] = []
        validation_precision: list[tuple[np.ndarray, ...]] = []
        for subject_responses, subject_precision in zip(
            self.validation_responses,
            self.validation_precision_weights,
            strict=True,
        ):
            if (
                len(subject_responses) != context_count
                or len(subject_precision) != context_count
            ):
                raise ValueError("each validation subject must cover every context")
            responses = tuple(
                _frozen_vector(
                    values,
                    length=len(features),
                    field_name="validation_responses",
                )
                for values in subject_responses
            )
            if any(np.any(values < 0) for values in responses):
                raise ValueError("validation responses must be direction-compatible")
            precision = tuple(
                _frozen_vector(
                    values,
                    length=len(features),
                    field_name="validation_precision_weights",
                )
                for values in subject_precision
            )
            if any(
                np.any(values < 0) or not np.any(values > 0) for values in precision
            ):
                raise ValueError(
                    "validation precision must be non-negative and not all zero"
                )
            validation_responses.append(responses)
            validation_precision.append(precision)
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "family_ids", families)
        object.__setattr__(self, "training_subject_ids", training_subjects)
        object.__setattr__(self, "validation_subject_ids", validation_subjects)
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "training_responses", training_responses)
        object.__setattr__(self, "training_precision_weights", training_precision)
        object.__setattr__(self, "validation_responses", tuple(validation_responses))
        object.__setattr__(
            self, "validation_precision_weights", tuple(validation_precision)
        )
        object.__setattr__(self, "family_parent_set_id", parent_set_id)
        object.__setattr__(
            self,
            "fold_data_id",
            stable_id(
                "graph_tuning_fold",
                self._identity_payload(),
                schema_version="1",
            ),
        )

    def _identity_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "fold_id": self.fold_id,
            "graph_id": _graph_id(self.graph),
            "feature_ids": list(self.feature_ids),
            "family_ids": list(self.family_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "validation_subject_ids": list(self.validation_subject_ids),
            "matrix_nnz": [int(matrix.nnz) for matrix in self.matrices],
            "matrix_digests": [_matrix_digest(matrix) for matrix in self.matrices],
            "training_response_digests": [
                _array_digest(values) for values in self.training_responses
            ],
            "training_precision_digests": [
                _array_digest(values) for values in self.training_precision_weights
            ],
            "validation_responses": [
                [_array_digest(values) for values in subject]
                for subject in self.validation_responses
            ],
            "validation_precision": [
                [_array_digest(values) for values in subject]
                for subject in self.validation_precision_weights
            ],
        }
        if self.family_parent_set_id is not None:
            payload["family_parent_set_id"] = self.family_parent_set_id
        return payload

    def _require_intact(self) -> None:
        try:
            expected = stable_id(
                "graph_tuning_fold",
                self._identity_payload(),
                schema_version="1",
            )
            valid = self.fold_data_id == expected
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Graph tuning fold identity is not intact",
                code="graph_tuning_fold_integrity_violation",
                field="fold_data_id",
                remediation="Rebuild the fold from immutable inner-training inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Graph tuning fold identity is not intact",
                code="graph_tuning_fold_integrity_violation",
                field="fold_data_id",
                remediation="Rebuild the fold from immutable inner-training inputs",
            )

    @classmethod
    def from_mappings(
        cls,
        *,
        fold_id: str,
        graph: ContextGraph,
        feature_ids: Sequence[str],
        family_ids: Sequence[str],
        training_subject_ids: Sequence[str],
        matrices: Mapping[Hashable, sparse.spmatrix],
        training_responses: Mapping[Hashable, np.ndarray],
        training_precision_weights: Mapping[Hashable, np.ndarray],
        validation_responses: Mapping[str, Mapping[Hashable, np.ndarray]],
        validation_precision_weights: Mapping[str, Mapping[Hashable, np.ndarray]],
        family_parent_set_id: str | None = None,
    ) -> GraphTuningFold:
        """Freeze mapping inputs in canonical graph-node and subject order."""

        nodes = tuple(graph.nodes)
        mapping_groups: tuple[Mapping[Hashable, object], ...] = (
            cast(Mapping[Hashable, object], matrices),
            cast(Mapping[Hashable, object], training_responses),
            cast(Mapping[Hashable, object], training_precision_weights),
        )
        if any(set(values) != set(nodes) for values in mapping_groups):
            raise ContractError(
                "Graph tuning training mappings must exactly cover graph nodes",
                code="graph_tuning_context_mismatch",
                field="graph.nodes",
                remediation="Provide one aligned training value per context",
            )
        validation_subjects = tuple(sorted(validation_responses))
        if set(validation_precision_weights) != set(validation_subjects):
            raise ValueError("validation response and precision subjects must match")
        for subject in validation_subjects:
            if set(validation_responses[subject]) != set(nodes) or set(
                validation_precision_weights[subject]
            ) != set(nodes):
                raise ContractError(
                    "Every graph tuning validation subject must cover graph nodes",
                    code="graph_tuning_context_mismatch",
                    field="validation_responses",
                    remediation="Use complete subject x context validation responses",
                )
        return cls(
            fold_id=fold_id,
            graph=graph,
            feature_ids=tuple(feature_ids),
            family_ids=tuple(family_ids),
            training_subject_ids=tuple(training_subject_ids),
            validation_subject_ids=validation_subjects,
            matrices=tuple(sparse.csc_matrix(matrices[node]) for node in nodes),
            training_responses=tuple(training_responses[node] for node in nodes),
            training_precision_weights=tuple(
                training_precision_weights[node] for node in nodes
            ),
            validation_responses=tuple(
                tuple(validation_responses[subject][node] for node in nodes)
                for subject in validation_subjects
            ),
            validation_precision_weights=tuple(
                tuple(validation_precision_weights[subject][node] for node in nodes)
                for subject in validation_subjects
            ),
            family_parent_set_id=family_parent_set_id,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphPenaltyFoldEvaluation:
    """One candidate's fitted coefficients and subject-level held-out losses."""

    candidate_id: str
    fold_data_id: str
    validation_subject_ids: tuple[str, ...]
    subject_losses: np.ndarray
    selected_coefficients: np.ndarray
    status: str
    reason_code: str | None
    evaluation_id: str

    @property
    def observed(self) -> bool:
        return self.status == "observed"


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphPenaltyCandidateSummary:
    """Paired validation summary relative to the empirical best candidate."""

    candidate_id: str
    mean_loss: float
    mean_paired_delta: float
    paired_delta_se: float
    within_one_se: bool
    summary_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphPenaltyTuningResult:
    """Experimental graph-aware selection and family stability artifact."""

    tuning_id: str
    spec: GraphPenaltyTuningSpec
    graph_id: str
    fold_data_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    validation_subject_ids: tuple[str, ...]
    evaluations: tuple[GraphPenaltyFoldEvaluation, ...]
    summaries: tuple[GraphPenaltyCandidateSummary, ...]
    best_candidate_id: str
    selected_candidate_id: str
    context_nodes: tuple[Hashable, ...]
    family_ids: tuple[str, ...]
    selection_frequency: np.ndarray
    experimental: bool = True

    def __post_init__(self) -> None:
        frequency = np.asarray(self.selection_frequency, dtype="<f8").copy()
        if frequency.shape != (len(self.context_nodes), len(self.family_ids)) or (
            np.any(~np.isfinite(frequency))
            or np.any(frequency < 0)
            or np.any(frequency > 1)
        ):
            raise ValueError("selection_frequency must align and lie in [0, 1]")
        if not self.experimental:
            raise ValueError("graph penalty tuning remains experimental until G2")
        frequency.setflags(write=False)
        object.__setattr__(self, "selection_frequency", frequency)

    @property
    def selected_candidate(self) -> GraphPenaltyCandidate:
        return next(
            candidate
            for candidate in self.spec.candidates
            if candidate.candidate_id == self.selected_candidate_id
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tuning_id": self.tuning_id,
            "spec": self.spec.to_dict(),
            "graph_id": self.graph_id,
            "fold_data_ids": list(self.fold_data_ids),
            "training_subject_ids": list(self.training_subject_ids),
            "validation_subject_ids": list(self.validation_subject_ids),
            "best_candidate_id": self.best_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "validation_loss_estimand": _VALIDATION_ESTIMAND,
            "selection_frequency": self.selection_frequency.tolist(),
            "experimental": True,
            "formal_inference_allowed": False,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class TunedGraphFusedFamilyAttribution:
    """Full outer-training graph fit bound to its inner selection artifact."""

    tuned_attribution_id: str
    tuning: GraphPenaltyTuningResult
    attribution: GraphFusedFamilyAttributionResult
    graph_id: str
    selected_candidate_id: str
    experimental: bool = True

    def __post_init__(self) -> None:
        candidate = self.tuning.selected_candidate
        if (
            self.graph_id != self.tuning.graph_id
            or self.selected_candidate_id != candidate.candidate_id
            or self.attribution.family_ids != self.tuning.family_ids
            or self.attribution.context_nodes != self.tuning.context_nodes
            or self.attribution.lambda1 != candidate.lambda1
            or self.attribution.lambda2 != candidate.lambda2
            or self.attribution.lambda_f != candidate.lambda_f
        ):
            raise ContractError(
                "Tuned graph attribution parents or selected penalty do not match",
                code="tuned_graph_attribution_parent_mismatch",
                field="tuning,attribution",
                remediation="Refit from one intact graph tuning result",
            )
        expected = stable_id(
            "tuned_graph_fused_family_attribution",
            {
                "tuning_id": self.tuning.tuning_id,
                "attribution_id": self.attribution.attribution_id,
                "graph_id": self.graph_id,
                "selected_candidate_id": self.selected_candidate_id,
                "formal_inference_allowed": False,
            },
            schema_version="1",
        )
        if self.tuned_attribution_id != expected:
            raise ContractError(
                "Tuned graph attribution identity is not intact",
                code="tuned_graph_attribution_integrity_violation",
                field="tuned_attribution_id",
                remediation="Recreate the result through the producer function",
            )
        if not self.experimental:
            raise ValueError("tuned graph attribution remains experimental until G2")

    def to_dict(self) -> dict[str, object]:
        return {
            "tuned_attribution_id": self.tuned_attribution_id,
            "tuning_id": self.tuning.tuning_id,
            "attribution_id": self.attribution.attribution_id,
            "graph_id": self.graph_id,
            "selected_candidate_id": self.selected_candidate_id,
            "experimental": True,
            "formal_inference_allowed": False,
        }


def _evaluate_candidate(
    fold: GraphTuningFold,
    candidate: GraphPenaltyCandidate,
    *,
    coefficient_tolerance: float,
    solver_tolerance: float,
    max_iterations: int,
) -> GraphPenaltyFoldEvaluation:
    nodes = tuple(fold.graph.nodes)
    matrices = dict(zip(nodes, fold.matrices, strict=True))
    responses = dict(zip(nodes, fold.training_responses, strict=True))
    precision = dict(zip(nodes, fold.training_precision_weights, strict=True))
    solution = solve_graph_fused_nonnegative_elastic_net(
        matrices,
        responses,
        fold.graph,
        precision_weights=precision,
        lambda1=candidate.lambda1,
        lambda2=candidate.lambda2,
        lambda_f=candidate.lambda_f,
        tolerance=solver_tolerance,
        max_iterations=max_iterations,
    )
    status = "observed" if solution.diagnostics.converged else "failed"
    reason = (
        None
        if status == "observed"
        else (solution.diagnostics.failure_reason or "graph_solver_not_converged")
    )
    losses: list[float] = []
    if status == "observed":
        for subject_responses, subject_precision in zip(
            fold.validation_responses,
            fold.validation_precision_weights,
            strict=True,
        ):
            numerator = 0.0
            denominator = 0.0
            for response, weight, predicted in zip(
                subject_responses,
                subject_precision,
                solution.predicted,
                strict=True,
            ):
                numerator += float(np.dot(weight, np.square(response - predicted)))
                denominator += float(weight.sum())
            losses.append(numerator / denominator)
    frozen_losses = np.asarray(losses, dtype="<f8")
    frozen_losses.setflags(write=False)
    selected = np.asarray(solution.coefficients > coefficient_tolerance, dtype=bool)
    selected.setflags(write=False)
    payload = {
        "candidate_id": candidate.candidate_id,
        "fold_data_id": fold.fold_data_id,
        "validation_subject_ids": list(fold.validation_subject_ids),
        "subject_losses": frozen_losses.tolist(),
        "selected_coefficients": selected.astype(int).tolist(),
        "status": status,
        "reason_code": reason,
        "solver_backend": solution.diagnostics.backend,
        "solver_status": solution.diagnostics.status.value,
    }
    return GraphPenaltyFoldEvaluation(
        candidate_id=candidate.candidate_id,
        fold_data_id=fold.fold_data_id,
        validation_subject_ids=fold.validation_subject_ids,
        subject_losses=frozen_losses,
        selected_coefficients=selected,
        status=status,
        reason_code=reason,
        evaluation_id=stable_id(
            "graph_penalty_fold_evaluation", payload, schema_version="1"
        ),
    )


def tune_graph_fused_penalties(
    folds: Sequence[GraphTuningFold],
    spec: GraphPenaltyTuningSpec,
    *,
    solver_tolerance: float = 1e-7,
    max_iterations: int = 1_000,
) -> GraphPenaltyTuningResult:
    """Select graph penalties using only subject-blocked inner validation loss."""

    if not isinstance(spec, GraphPenaltyTuningSpec):
        raise TypeError("spec must be a GraphPenaltyTuningSpec")
    spec._require_intact()
    inner_folds = tuple(folds)
    if len(inner_folds) < 2 or any(
        not isinstance(fold, GraphTuningFold) for fold in inner_folds
    ):
        raise ValueError("folds must contain at least two GraphTuningFold values")
    for fold in inner_folds:
        fold._require_intact()
    if not math.isfinite(solver_tolerance) or solver_tolerance <= 0:
        raise ValueError("solver_tolerance must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or (max_iterations < 1)
    ):
        raise ValueError("max_iterations must be an integer >= 1")
    graph_id = _graph_id(inner_folds[0].graph)
    feature_ids = inner_folds[0].feature_ids
    family_ids = inner_folds[0].family_ids
    if any(
        _graph_id(fold.graph) != graph_id
        or fold.feature_ids != feature_ids
        or fold.family_ids != family_ids
        for fold in inner_folds
    ):
        raise ContractError(
            "Graph tuning folds must share one topology and feature/family axis",
            code="graph_tuning_fold_mismatch",
            field="folds",
            remediation="Freeze the graph and family universe before inner tuning",
        )
    fold_ids = tuple(fold.fold_data_id for fold in inner_folds)
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("graph tuning folds must be unique")
    validation_subjects = tuple(
        subject for fold in inner_folds for subject in fold.validation_subject_ids
    )
    if len(set(validation_subjects)) != len(validation_subjects):
        raise ContractError(
            "Each subject may validate exactly one inner graph-tuning fold",
            code="graph_tuning_validation_subject_reused",
            field="validation_subject_ids",
            remediation="Use a disjoint inner fold partition",
        )
    outer_subjects = set(validation_subjects)
    for fold in inner_folds:
        outer_subjects.update(fold.training_subject_ids)
        if set(fold.training_subject_ids).union(fold.validation_subject_ids) != set(
            validation_subjects
        ):
            raise ContractError(
                "Every graph tuning fold must partition the same outer subjects",
                code="graph_tuning_partition_mismatch",
                field="training_subject_ids,validation_subject_ids",
                remediation="Build all inner folds from one outer-training subject set",
            )
    if outer_subjects != set(validation_subjects):
        raise RuntimeError("graph tuning outer-subject validation failed")

    evaluations = tuple(
        _evaluate_candidate(
            fold,
            candidate,
            coefficient_tolerance=spec.coefficient_selection_tolerance,
            solver_tolerance=solver_tolerance,
            max_iterations=max_iterations,
        )
        for candidate in spec.candidates
        for fold in inner_folds
    )
    evaluations_by_candidate = {
        candidate.candidate_id: tuple(
            evaluation
            for evaluation in evaluations
            if evaluation.candidate_id == candidate.candidate_id
        )
        for candidate in spec.candidates
    }
    eligible_candidates = tuple(
        candidate
        for candidate in spec.candidates
        if all(
            evaluation.observed
            for evaluation in evaluations_by_candidate[candidate.candidate_id]
        )
    )
    if not eligible_candidates:
        raise ContractError(
            "No graph penalty candidate converged in every inner fold",
            code="graph_tuning_no_estimable_candidate",
            field="candidates",
            remediation="Inspect solver diagnostics or simplify the candidate grid",
        )

    losses_by_candidate: dict[str, np.ndarray] = {}
    for candidate in eligible_candidates:
        losses = np.concatenate(
            [
                evaluation.subject_losses
                for evaluation in evaluations_by_candidate[candidate.candidate_id]
            ]
        )
        losses_by_candidate[candidate.candidate_id] = losses
    best = min(
        eligible_candidates,
        key=lambda candidate: (
            float(np.mean(losses_by_candidate[candidate.candidate_id])),
            -candidate.lambda_f,
            -candidate.lambda1,
            -candidate.lambda2,
            candidate.candidate_id,
        ),
    )
    best_losses = losses_by_candidate[best.candidate_id]
    summaries: list[GraphPenaltyCandidateSummary] = []
    for candidate in eligible_candidates:
        losses = losses_by_candidate[candidate.candidate_id]
        delta = losses - best_losses
        mean_delta = float(np.mean(delta))
        delta_se = (
            float(np.std(delta, ddof=1) / math.sqrt(delta.size))
            if delta.size > 1
            else 0.0
        )
        within_one_se = mean_delta <= delta_se + 1e-12
        mean_loss = float(np.mean(losses))
        payload = {
            "candidate_id": candidate.candidate_id,
            "mean_loss": mean_loss,
            "mean_paired_delta": mean_delta,
            "paired_delta_se": delta_se,
            "within_one_se": within_one_se,
        }
        summaries.append(
            GraphPenaltyCandidateSummary(
                candidate_id=candidate.candidate_id,
                mean_loss=mean_loss,
                mean_paired_delta=mean_delta,
                paired_delta_se=delta_se,
                within_one_se=within_one_se,
                summary_id=stable_id(
                    "graph_penalty_candidate_summary", payload, schema_version="1"
                ),
            )
        )
    summary_by_candidate = {summary.candidate_id: summary for summary in summaries}
    selected = min(
        (
            candidate
            for candidate in eligible_candidates
            if summary_by_candidate[candidate.candidate_id].within_one_se
        ),
        key=lambda candidate: (
            -candidate.lambda_f,
            -candidate.lambda1,
            -candidate.lambda2,
            candidate.candidate_id,
        ),
    )
    selected_evaluations = evaluations_by_candidate[selected.candidate_id]
    selection_frequency = np.mean(
        np.stack(
            [evaluation.selected_coefficients for evaluation in selected_evaluations]
        ),
        axis=0,
    )
    payload = {
        "spec_id": spec.spec_id,
        "graph_id": graph_id,
        "fold_data_ids": list(fold_ids),
        "evaluation_ids": [evaluation.evaluation_id for evaluation in evaluations],
        "summary_ids": [summary.summary_id for summary in summaries],
        "best_candidate_id": best.candidate_id,
        "selected_candidate_id": selected.candidate_id,
        "validation_loss_estimand": _VALIDATION_ESTIMAND,
        "selection_frequency": selection_frequency.tolist(),
        "formal_inference_allowed": False,
    }
    return GraphPenaltyTuningResult(
        tuning_id=stable_id("graph_penalty_tuning", payload, schema_version="1"),
        spec=spec,
        graph_id=graph_id,
        fold_data_ids=fold_ids,
        training_subject_ids=tuple(sorted(outer_subjects)),
        validation_subject_ids=tuple(sorted(validation_subjects)),
        evaluations=evaluations,
        summaries=tuple(summaries),
        best_candidate_id=best.candidate_id,
        selected_candidate_id=selected.candidate_id,
        context_nodes=tuple(inner_folds[0].graph.nodes),
        family_ids=family_ids,
        selection_frequency=selection_frequency,
    )


def fit_tuned_graph_fused_family_attribution(
    family_bases: Mapping[Hashable, FamilyFirstBasis],
    signed_responses: Mapping[Hashable, np.ndarray],
    graph: ContextGraph,
    tuning_folds: Sequence[GraphTuningFold],
    tuning_spec: GraphPenaltyTuningSpec,
    *,
    precision_weights: Mapping[Hashable, np.ndarray] | None = None,
    solver_tolerance: float = 1e-7,
    max_iterations: int = 1_000,
) -> TunedGraphFusedFamilyAttribution:
    """Tune within outer training, then refit the selected graph model in full."""

    tuning = tune_graph_fused_penalties(
        tuning_folds,
        tuning_spec,
        solver_tolerance=solver_tolerance,
        max_iterations=max_iterations,
    )
    if _graph_id(graph) != tuning.graph_id:
        raise ContractError(
            "Final graph must equal the topology frozen for inner tuning",
            code="tuned_graph_attribution_parent_mismatch",
            field="graph",
            remediation="Use the exact outer-training context graph in both stages",
        )
    nodes = tuple(graph.nodes)
    if set(family_bases) != set(nodes):
        raise ContractError(
            "Final family bases must exactly cover the tuned context graph",
            code="tuned_graph_attribution_parent_mismatch",
            field="family_bases",
            remediation="Provide one full outer-training family basis per context",
        )
    first_basis = family_bases[nodes[0]]
    if (
        first_basis.feature_ids != tuning_folds[0].feature_ids
        or first_basis.family_ids != tuning.family_ids
    ):
        raise ContractError(
            "Final family axes must match the axes frozen for graph tuning",
            code="tuned_graph_attribution_parent_mismatch",
            field="feature_ids,family_ids",
            remediation="Keep one common outer-training family universe",
        )
    selected = tuning.selected_candidate
    attribution = fit_graph_fused_family_attribution(
        family_bases,
        signed_responses,
        graph,
        precision_weights=precision_weights,
        lambda1=selected.lambda1,
        lambda2=selected.lambda2,
        lambda_f=selected.lambda_f,
        tolerance=solver_tolerance,
        max_iterations=max_iterations,
    )
    payload = {
        "tuning_id": tuning.tuning_id,
        "attribution_id": attribution.attribution_id,
        "graph_id": tuning.graph_id,
        "selected_candidate_id": selected.candidate_id,
        "formal_inference_allowed": False,
    }
    return TunedGraphFusedFamilyAttribution(
        tuned_attribution_id=stable_id(
            "tuned_graph_fused_family_attribution", payload, schema_version="1"
        ),
        tuning=tuning,
        attribution=attribution,
        graph_id=tuning.graph_id,
        selected_candidate_id=selected.candidate_id,
    )


__all__ = [
    "GraphPenaltyCandidate",
    "GraphPenaltyCandidateSummary",
    "GraphPenaltyFoldEvaluation",
    "GraphPenaltyTuningResult",
    "GraphPenaltyTuningSpec",
    "GraphTuningFold",
    "TunedGraphFusedFamilyAttribution",
    "fit_tuned_graph_fused_family_attribution",
    "tune_graph_fused_penalties",
]
