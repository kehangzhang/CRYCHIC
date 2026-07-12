"""Response-aligned, normalized, receptor-gated TargetPrior bases."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import sparse

from crychic.core import ContractError, stable_id
from crychic.resources import TargetPrior

from .contracts import BasisBuildReport, GatedTargetBasis


def _validated_gates(
    driver_ids: tuple[str, ...], receptor_gates: Mapping[str, float]
) -> np.ndarray:
    expected = set(driver_ids)
    observed = set(receptor_gates)
    missing = expected.difference(observed)
    unknown = observed.difference(expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)!r}")
        if unknown:
            details.append(f"unknown={sorted(unknown)!r}")
        raise ContractError(
            "Receptor gate keys must exactly match prior drivers: "
            + ", ".join(details),
            code="receptor_gate_mismatch",
            field="receptor_gates",
            remediation="Provide one fold-frozen gate for every TargetPrior driver",
        )
    values: list[float] = []
    for driver in driver_ids:
        raw = receptor_gates[driver]
        if isinstance(raw, bool):
            raise ContractError(
                f"Receptor gate for {driver!r} must be numeric, not boolean",
                code="invalid_receptor_gate",
                field="receptor_gates",
                remediation="Provide a continuous gate in [0, 1]",
            )
        value = float(raw)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ContractError(
                f"Receptor gate for {driver!r} must lie in [0, 1]",
                code="invalid_receptor_gate",
                field="receptor_gates",
                remediation="Provide a fold-frozen receptor availability gate",
            )
        values.append(value)
    result: np.ndarray = np.asarray(values, dtype=float)
    return result


def build_gated_target_basis(
    prior: TargetPrior,
    feature_ids: Sequence[str],
    receptor_gates: Mapping[str, float],
) -> GatedTargetBasis:
    """Align a positive prior, L2-normalize columns, then apply receptor gates."""

    if not isinstance(prior, TargetPrior):
        raise TypeError("prior must be a TargetPrior")
    if prior.direction != 1:
        raise ContractError(
            "V0.1 attribution only supports a positive TargetPrior direction",
            code="unsupported_signed_prior",
            field="direction",
            remediation="Use an explicitly reviewed signed-channel implementation",
        )
    features = tuple(map(str, feature_ids))
    if not features or len(set(features)) != len(features):
        raise ContractError(
            "Response feature IDs must be non-empty and unique",
            code="invalid_basis_features",
            field="feature_ids",
            remediation="Collapse duplicate response genes before attribution",
        )
    gates = _validated_gates(prior.driver_ids, receptor_gates)
    feature_index = {gene: index for index, gene in enumerate(features)}
    normalized_data: list[float] = []
    normalized_indices: list[int] = []
    gated_data: list[float] = []
    gated_indices: list[int] = []
    normalized_indptr = [0]
    gated_indptr = [0]
    norms: list[float] = []
    zero_norm: list[str] = []
    for driver_index, driver in enumerate(prior.driver_ids):
        start, stop = prior.indptr[driver_index : driver_index + 2]
        weights_by_row: dict[int, float] = {}
        for position in range(start, stop):
            target = prior.target_ids[prior.target_indices[position]]
            row = feature_index.get(target)
            weight = prior.weights[position]
            if row is not None and weight > 0:
                weights_by_row[row] = weights_by_row.get(row, 0.0) + weight
        column = sorted(weights_by_row.items())
        norm = math.sqrt(sum(weight * weight for _, weight in column))
        norms.append(norm)
        if norm == 0:
            zero_norm.append(driver)
        else:
            gate = gates[driver_index]
            for row, weight in column:
                normalized = weight / norm
                normalized_indices.append(row)
                normalized_data.append(normalized)
                if gate > 0:
                    gated_indices.append(row)
                    gated_data.append(normalized * gate)
        normalized_indptr.append(len(normalized_data))
        gated_indptr.append(len(gated_data))

    shape = (len(features), len(prior.driver_ids))
    normalized_profiles = sparse.csc_matrix(
        (
            np.asarray(normalized_data, dtype=float),
            np.asarray(normalized_indices, dtype=np.int32),
            np.asarray(normalized_indptr, dtype=np.int64),
        ),
        shape=shape,
    )
    gated_matrix = sparse.csc_matrix(
        (
            np.asarray(gated_data, dtype=float),
            np.asarray(gated_indices, dtype=np.int32),
            np.asarray(gated_indptr, dtype=np.int64),
        ),
        shape=shape,
    )
    unmatched = tuple(sorted(set(prior.target_ids).difference(features)))
    report = BasisBuildReport(
        prior_targets=len(prior.target_ids),
        response_features=len(features),
        matched_targets=len(set(prior.target_ids).intersection(features)),
        unmatched_prior_targets=unmatched,
        zero_norm_drivers=tuple(zero_norm),
        zero_gate_drivers=tuple(
            driver
            for driver, gate in zip(prior.driver_ids, gates, strict=True)
            if gate == 0
        ),
    )
    basis_id = stable_id(
        "target_basis",
        {
            "driver_ids": prior.driver_ids,
            "feature_ids": features,
            "gates": gates.tolist(),
            "prior_manifest": prior.manifest_digest,
            "prior_resource_id": prior.resource_id,
            "prior_version": prior.version,
        },
    )
    return GatedTargetBasis(
        basis_id=basis_id,
        feature_ids=features,
        driver_ids=prior.driver_ids,
        normalized_profiles=normalized_profiles,
        matrix=gated_matrix,
        receptor_gates=gates,
        pre_normalization_norms=np.asarray(norms, dtype=float),
        prior_resource_id=prior.resource_id,
        prior_version=prior.version,
        report=report,
    )
