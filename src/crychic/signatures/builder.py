"""Build descriptive gene signatures and sparse driver contributions."""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import cast

import numpy as np
import pandas as pd

from crychic.attribution import AttributionResult, GatedTargetBasis
from crychic.core import ContractError, stable_id
from crychic.response import ResponseEstimate, ResponseStatus

from .contracts import (
    CONTRIBUTION_COLUMNS,
    GENE_SIGNATURE_COLUMNS,
    DirectionAgreement,
    SignatureStatus,
    SignatureTable,
)

_DIRECTION_RULE = "positive_prior_agreement_v1"


def _key_component(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_key_component(item) for item in value]}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def _selected_response_rows(
    response: ResponseEstimate,
    receiver: Hashable,
    contrast: str,
) -> pd.DataFrame:
    receiver_mask = response.contrasts["receiver"].map(
        lambda value: value == receiver
    )
    selected = response.contrasts.loc[
        receiver_mask & (response.contrasts["contrast"] == contrast)
    ].copy(deep=True)
    if selected.empty:
        raise ContractError(
            "ResponseEstimate has no rows for the requested receiver and contrast",
            code="unknown_signature_response",
            field="receiver,contrast",
            remediation="Select an emitted response receiver and contrast",
        )
    if selected["gene"].duplicated().any():
        raise ContractError(
            "ResponseEstimate contains duplicate receiver/contrast/gene rows",
            code="duplicate_signature_response",
            field="gene",
            remediation="Preserve the response table primary key",
        )
    indexed = selected.set_index("gene", drop=False)
    missing = set(response.feature_ids).difference(indexed.index)
    if missing:
        raise ContractError(
            f"ResponseEstimate is missing feature rows: {sorted(missing)!r}",
            code="incomplete_signature_response",
            field="gene",
            remediation="Retain the complete continuous response feature vector",
        )
    result: pd.DataFrame = indexed.loc[list(response.feature_ids)].reset_index(
        drop=True
    )
    return result


def _direction(value: float, tolerance: float) -> str:
    if not math.isfinite(value):
        return DirectionAgreement.NOT_AVAILABLE.value
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "zero"


def _agreement(observed: float, predicted: float, tolerance: float) -> str:
    if not math.isfinite(observed) or not math.isfinite(predicted):
        return DirectionAgreement.NOT_AVAILABLE.value
    if predicted <= tolerance:
        return DirectionAgreement.NO_PREDICTION.value
    if observed > tolerance:
        return DirectionAgreement.AGREES.value
    return DirectionAgreement.DISCORDANT.value


def _rank_absolute(
    frame: pd.DataFrame,
    value_column: str,
    output_column: str,
    *,
    group_column: str | None = None,
) -> None:
    ranks: dict[int, int] = {}
    if group_column is None:
        groups = [frame]
    else:
        groups = [
            frame.loc[frame[group_column] == value]
            for value in sorted(map(str, frame[group_column].unique()))
        ]
    for group in groups:
        finite_indices = [
            index
            for index in group.index
            if math.isfinite(
                float(cast(float, frame.at[index, value_column]))
            )
        ]
        ordered = sorted(
            finite_indices,
            key=lambda index: (
                -abs(float(cast(float, frame.at[index, value_column]))),
                str(frame.at[index, "gene"]),
            ),
        )
        ranks.update({index: rank for rank, index in enumerate(ordered, start=1)})
    frame[output_column] = pd.array(
        [ranks.get(index, pd.NA) for index in frame.index], dtype="Int64"
    )


def _validate_attribution_alignment(
    attribution: AttributionResult,
    basis: GatedTargetBasis,
    observed: np.ndarray,
) -> None:
    if attribution.basis_id != basis.basis_id:
        raise ContractError(
            "AttributionResult and GatedTargetBasis IDs do not match",
            code="signature_basis_mismatch",
            field="basis_id",
            remediation="Use the exact basis that produced the attribution result",
        )
    if (
        attribution.feature_ids != basis.feature_ids
        or attribution.driver_ids != basis.driver_ids
    ):
        raise ContractError(
            "AttributionResult and basis feature/driver axes do not match",
            code="signature_basis_mismatch",
            field="feature_ids",
            remediation="Do not reorder a fitted basis or attribution result",
        )
    if not np.allclose(
        attribution.signed_response, observed, rtol=1e-10, atol=1e-12
    ):
        raise ContractError(
            "Attribution signed response differs from the selected ResponseEstimate",
            code="signature_response_mismatch",
            field="signed_response",
            remediation=(
                "Use the same response field, receiver, and contrast as fitting"
            ),
        )


def build_signature_table(
    response: ResponseEstimate,
    attribution: AttributionResult | None,
    basis: GatedTargetBasis | None,
    *,
    context: Hashable,
    receiver: Hashable,
    contrast: str,
    gene_namespace: str,
    observed_field: str = "z_score",
    fold_id: str | None = None,
    direction_tolerance: float = 1e-12,
) -> SignatureTable:
    """Build gene and driver signature tables without fitting or significance tests."""

    if not isinstance(response, ResponseEstimate):
        raise TypeError("response must be a ResponseEstimate")
    if not gene_namespace.strip():
        raise ContractError(
            "gene_namespace cannot be empty",
            code="missing_signature_namespace",
            field="gene_namespace",
            remediation="Record the response and prior gene namespace",
        )
    if observed_field not in response.contrasts.columns:
        raise ContractError(
            f"ResponseEstimate lacks observed field {observed_field!r}",
            code="unknown_signature_response_field",
            field="observed_field",
            remediation="Use effect or z_score from the response producer contract",
        )
    if not math.isfinite(direction_tolerance) or direction_tolerance < 0:
        raise ContractError(
            "direction_tolerance must be finite and non-negative",
            code="invalid_direction_tolerance",
            field="direction_tolerance",
            remediation="Use a fixed non-negative numerical direction tolerance",
        )
    if (attribution is None) != (basis is None):
        raise ContractError(
            "AttributionResult and GatedTargetBasis must be supplied together",
            code="incomplete_signature_attribution",
            field="attribution,basis",
            remediation="Provide both fitted artifacts or neither for an empty result",
        )

    selected = _selected_response_rows(response, receiver, contrast)
    observed = selected[observed_field].to_numpy(dtype=float)
    response_ok = (
        (selected["status"] == ResponseStatus.OK.value).to_numpy(dtype=bool)
        & np.isfinite(observed)
    )
    if attribution is not None and basis is not None:
        if not np.all(response_ok):
            raise ContractError(
                "Attribution cannot be joined to non-estimable response genes",
                code="signature_response_not_estimable",
                field="status",
                remediation="Fit attribution only on a complete continuous response",
            )
        _validate_attribution_alignment(attribution, basis, observed)

    context_id = stable_id("signature_context", _key_component(context))
    receiver_key = _key_component(receiver)
    basis_id = None if attribution is None else attribution.basis_id
    solver_status = (
        "not_run"
        if attribution is None
        else attribution.diagnostics.status.value
    )
    accepted = attribution is not None and attribution.succeeded
    failed_reason: str | None = (
        None
        if attribution is None or attribution.succeeded
        else attribution.diagnostics.failure_reason
        or attribution.diagnostics.status.value
    )
    gene_rows: list[dict[str, object]] = []
    signature_ids: dict[str, str] = {}
    for index, gene in enumerate(response.feature_ids):
        signature_id = stable_id(
            "gene_signature",
            {
                "basis_id": basis_id,
                "context_id": context_id,
                "contrast": contrast,
                "fold_id": fold_id,
                "gene": gene,
                "gene_namespace": gene_namespace,
                "receiver": receiver_key,
                "rule": _DIRECTION_RULE,
            },
        )
        signature_ids[gene] = signature_id
        observed_value = float(observed[index])
        reason: str | None
        if not response_ok[index]:
            status = SignatureStatus.RESPONSE_NOT_ESTIMABLE
            response_reason = selected.iloc[index]["reason_code"]
            reason = (
                "response_not_estimable"
                if pd.isna(response_reason)
                else str(response_reason)
            )
            predicted = residual = consistent = math.nan
        elif attribution is None:
            status = SignatureStatus.EMPTY_ATTRIBUTION
            reason = "no_attribution"
            predicted = 0.0
            residual = observed_value
            consistent = 0.0
        elif not accepted:
            status = SignatureStatus.ATTRIBUTION_FAILED
            reason = failed_reason
            predicted = residual = consistent = math.nan
        else:
            status = SignatureStatus.OK
            reason = None
            predicted = float(attribution.predicted[index])
            residual = float(attribution.residual[index])
            consistent = predicted if observed_value > direction_tolerance else 0.0
        gene_rows.append(
            {
                "signature_id": signature_id,
                "context_id": context_id,
                "context": context,
                "receiver": receiver,
                "contrast": contrast,
                "gene": gene,
                "gene_namespace": gene_namespace,
                "observed": observed_value,
                "predicted": predicted,
                "residual": residual,
                "direction_consistent_predicted": consistent,
                "observed_direction": _direction(
                    observed_value, direction_tolerance
                ),
                "predicted_direction": _direction(predicted, direction_tolerance),
                "direction_agreement": _agreement(
                    observed_value, predicted, direction_tolerance
                ),
                "status": status.value,
                "reason_code": reason,
                "solver_status": solver_status,
                "basis_id": basis_id,
                "fold_id": fold_id,
                "response_scale": response.value_scale,
                "response_source": response.expression_source,
                "direction_rule_version": _DIRECTION_RULE,
                "observed_rank": pd.NA,
                "predicted_rank": pd.NA,
                "residual_rank": pd.NA,
            }
        )
    genes = pd.DataFrame(gene_rows, columns=GENE_SIGNATURE_COLUMNS)
    _rank_absolute(genes, "observed", "observed_rank")
    _rank_absolute(genes, "predicted", "predicted_rank")
    _rank_absolute(genes, "residual", "residual_rank")

    contribution_rows: list[dict[str, object]] = []
    if accepted and attribution is not None and basis is not None:
        family_by_driver = {
            estimate.driver_id: estimate.family_id
            for estimate in attribution.driver_estimates
        }
        for driver_index, driver in enumerate(basis.driver_ids):
            coefficient = float(attribution.coefficients[driver_index])
            start, stop = basis.matrix.indptr[driver_index : driver_index + 2]
            for position in range(start, stop):
                gene_index = int(basis.matrix.indices[position])
                gene = basis.feature_ids[gene_index]
                weight = float(basis.matrix.data[position])
                contribution = coefficient * weight
                observed_value = float(observed[gene_index])
                consistent = (
                    contribution
                    if observed_value > direction_tolerance
                    else 0.0
                )
                signature_id = signature_ids[gene]
                contribution_rows.append(
                    {
                        "contribution_id": stable_id(
                            "driver_contribution",
                            {
                                "driver_id": driver,
                                "signature_id": signature_id,
                            },
                        ),
                        "signature_id": signature_id,
                        "context_id": context_id,
                        "context": context,
                        "receiver": receiver,
                        "contrast": contrast,
                        "gene": gene,
                        "gene_namespace": gene_namespace,
                        "driver_id": driver,
                        "family_id": family_by_driver[driver],
                        "coefficient": coefficient,
                        "basis_weight": weight,
                        "contribution": contribution,
                        "direction_consistent_contribution": consistent,
                        "direction_agreement": _agreement(
                            observed_value, contribution, direction_tolerance
                        ),
                        "status": SignatureStatus.OK.value,
                        "reason_code": None,
                        "solver_status": solver_status,
                        "basis_id": basis_id,
                        "fold_id": fold_id,
                        "direction_rule_version": _DIRECTION_RULE,
                        "contribution_rank": pd.NA,
                    }
                )
        contributions = pd.DataFrame(
            contribution_rows, columns=CONTRIBUTION_COLUMNS
        )
        if not contributions.empty:
            _rank_absolute(
                contributions,
                "contribution",
                "contribution_rank",
                group_column="driver_id",
            )
        reconstructed = np.asarray(
            basis.matrix @ attribution.coefficients
        ).ravel()
        if not np.allclose(
            reconstructed, attribution.predicted, rtol=1e-9, atol=1e-10
        ):
            raise ContractError(
                "Coefficient-times-basis contributions do not reconstruct prediction",
                code="invalid_contribution_reconstruction",
                field="basis_id",
                remediation="Use the exact fitted basis without re-normalization",
            )
    else:
        contributions = pd.DataFrame(columns=CONTRIBUTION_COLUMNS)

    return SignatureTable(
        genes=genes,
        contributions=contributions,
        gene_namespace=gene_namespace,
        direction_rule_version=_DIRECTION_RULE,
    )
