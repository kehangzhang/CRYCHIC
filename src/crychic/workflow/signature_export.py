"""Lossless post-fit gene signatures from subject-cross-fit artifacts.

This adapter never refits a response or attribution model.  It reconstructs
the exact held-out gene coordinate used by the incremental functional, then
allocates its fitted family contribution with already-frozen LR and sender
weights.  Unavailable allocations remain explicit in the availability audit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, cast

import numpy as np
import pandas as pd

from crychic.core import ContractError, canonical_json, stable_id
from crychic.design import node_context_fields
from crychic.signatures import (
    FittedSignatureComponentRecord,
    LayeredSignatureStatus,
    LayeredSignatureTable,
    SignatureComponentLevel,
    SignatureFeatureKind,
    build_layered_signature_table,
)

from .crossfit import CrossFitArtifacts, CrossFitFoldArtifacts

SignatureScoreMode = Literal["state", "ecosystem"]

CROSSFIT_SIGNATURE_SCHEMA_VERSION = "0.1.0"
CROSSFIT_SIGNATURE_ID_SCHEMA_VERSION = "1"
GENE_MODEL_COORDINATE_VERSION = (
    "training_standardized_autonomous_orthogonal_null_residual_gene_v1"
)
SUBJECT_CONTEXT_AGGREGATION_VERSION = (
    "technical_row_mean_within_subject_context_then_equal_subject_mean_v1"
)
WITHIN_FAMILY_ENTROPY_AGGREGATION_VERSION = (
    "maximum_resolved_sample_entropy_within_outer_fold_context_v1"
)


class SignatureExportLayer(StrEnum):
    """A released layer in the cross-fit signature export."""

    RECEIVER_CONTEXT = "receiver_context"
    LR_ATTRIBUTED = "lr_attributed"
    SENDER_LR_RECEIVER = "sender_lr_receiver"


class SignatureExportAvailabilityStatus(StrEnum):
    """Whether one exact fold-context layer was losslessly reconstructed."""

    AVAILABLE = "available"
    NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True, slots=True)
class SignatureLayerAvailability:
    """Typed availability of one fold, receiver, context, and signature layer."""

    repeat_id: str
    fold_id: str
    contrast: str
    receiver: str
    context_id: str
    context_json: str
    mode: SignatureScoreMode
    layer: SignatureExportLayer
    status: SignatureExportAvailabilityStatus
    reason_code: str | None
    scoring_functional_id: str | None
    source_artifact_id: str
    component_count: int
    availability_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "repeat_id",
            "fold_id",
            "contrast",
            "receiver",
            "context_id",
            "source_artifact_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty identifier")
        if self.mode not in {"state", "ecosystem"}:
            raise ValueError("mode must be state or ecosystem")
        try:
            layer = SignatureExportLayer(self.layer)
            status = SignatureExportAvailabilityStatus(self.status)
        except ValueError as error:
            raise ValueError("signature availability enum is invalid") from error
        if (status is SignatureExportAvailabilityStatus.AVAILABLE) == (
            self.reason_code is not None
        ):
            raise ValueError(
                "reason_code is required exactly when a signature layer is unavailable"
            )
        if (
            isinstance(self.component_count, bool)
            or not isinstance(self.component_count, int)
            or self.component_count < 0
        ):
            raise ValueError("component_count must be a non-negative integer")
        if status is SignatureExportAvailabilityStatus.NOT_ESTIMABLE and (
            self.component_count != 0
        ):
            raise ValueError("unavailable layers cannot claim released components")
        if self.scoring_functional_id is not None and (
            not isinstance(self.scoring_functional_id, str)
            or not self.scoring_functional_id.strip()
        ):
            raise ValueError("scoring_functional_id must be missing or non-empty")
        try:
            decoded = json.loads(self.context_json)
        except json.JSONDecodeError as error:
            raise ValueError("context_json must contain valid JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError("context_json must encode an object")
        normalized_context = canonical_json(decoded)
        object.__setattr__(self, "context_json", normalized_context)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "availability_id",
            stable_id(
                "crossfit_signature_layer_availability",
                {
                    "component_count": self.component_count,
                    "context_id": self.context_id,
                    "context_json": normalized_context,
                    "contrast": self.contrast,
                    "fold_id": self.fold_id,
                    "layer": layer.value,
                    "mode": self.mode,
                    "reason_code": self.reason_code,
                    "receiver": self.receiver,
                    "repeat_id": self.repeat_id,
                    "scoring_functional_id": self.scoring_functional_id,
                    "source_artifact_id": self.source_artifact_id,
                    "status": status.value,
                },
                schema_version=CROSSFIT_SIGNATURE_ID_SCHEMA_VERSION,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a manifest-ready availability record."""

        return {
            "availability_id": self.availability_id,
            "repeat_id": self.repeat_id,
            "fold_id": self.fold_id,
            "contrast": self.contrast,
            "receiver": self.receiver,
            "context_id": self.context_id,
            "context_json": self.context_json,
            "mode": self.mode,
            "layer": self.layer.value,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "scoring_functional_id": self.scoring_functional_id,
            "source_artifact_id": self.source_artifact_id,
            "component_count": self.component_count,
        }


@dataclass(frozen=True, slots=True)
class CrossFitLayeredSignatureExport:
    """Descriptive post-fit signatures plus exact per-layer availability."""

    crossfit_id: str
    repeat_id: str
    mode: SignatureScoreMode
    table: LayeredSignatureTable | None
    availability: tuple[SignatureLayerAvailability, ...]
    entropy_threshold: float
    coordinate_version: str = GENE_MODEL_COORDINATE_VERSION
    aggregation_version: str = SUBJECT_CONTEXT_AGGREGATION_VERSION
    schema_version: str = CROSSFIT_SIGNATURE_SCHEMA_VERSION
    export_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.crossfit_id or not self.repeat_id:
            raise ValueError("crossfit_id and repeat_id must be non-empty")
        if self.mode not in {"state", "ecosystem"}:
            raise ValueError("mode must be state or ecosystem")
        if not math.isfinite(self.entropy_threshold) or not (
            0 <= self.entropy_threshold <= 1
        ):
            raise ValueError("entropy_threshold must lie in [0, 1]")
        records = tuple(self.availability)
        if not records or any(
            not isinstance(item, SignatureLayerAvailability) for item in records
        ):
            raise ValueError("availability must contain typed records")
        ordered = tuple(sorted(records, key=lambda item: item.availability_id))
        if len({item.availability_id for item in ordered}) != len(ordered):
            raise ValueError("availability records must be unique")
        if any(
            item.repeat_id != self.repeat_id or item.mode != self.mode
            for item in ordered
        ):
            raise ValueError("availability scope does not match its export")
        released = sum(item.component_count for item in ordered)
        if (self.table is None) != (released == 0):
            raise ValueError(
                "table must be present exactly when at least one component is released"
            )
        if self.table is not None and not isinstance(self.table, LayeredSignatureTable):
            raise TypeError("table must be a LayeredSignatureTable or None")
        if self.table is not None:
            expected_counts = {
                SignatureExportLayer.RECEIVER_CONTEXT: len(
                    self.table.receiver_context
                ),
                SignatureExportLayer.LR_ATTRIBUTED: len(self.table.lr_attributed),
                SignatureExportLayer.SENDER_LR_RECEIVER: len(
                    self.table.sender_lr_receiver
                ),
            }
            observed_counts = {
                layer: sum(
                    item.component_count
                    for item in ordered
                    if item.layer is layer
                    and item.status is SignatureExportAvailabilityStatus.AVAILABLE
                )
                for layer in SignatureExportLayer
            }
            if observed_counts != expected_counts:
                raise ValueError(
                    "availability component counts do not match released table rows"
                )
        object.__setattr__(self, "availability", ordered)
        object.__setattr__(
            self,
            "export_id",
            stable_id(
                "crossfit_layered_signature_export",
                {
                    "aggregation_version": self.aggregation_version,
                    "availability_ids": [item.availability_id for item in ordered],
                    "coordinate_version": self.coordinate_version,
                    "crossfit_id": self.crossfit_id,
                    "entropy_threshold": self.entropy_threshold,
                    "mode": self.mode,
                    "repeat_id": self.repeat_id,
                    "schema_version": self.schema_version,
                    "signature_ids": (
                        []
                        if self.table is None
                        else sorted(
                            [
                                *self.table.receiver_context["signature_id"].astype(str),
                                *self.table.lr_attributed["signature_id"].astype(str),
                                *self.table.sender_lr_receiver["signature_id"].astype(
                                    str
                                ),
                            ]
                        )
                    ),
                },
                schema_version=CROSSFIT_SIGNATURE_ID_SCHEMA_VERSION,
            ),
        )

    @property
    def inference_eligible(self) -> bool:
        """This post-fit export never releases formal inference."""

        return False

    def availability_table(self) -> pd.DataFrame:
        """Return the deterministic typed availability audit as a table."""

        return pd.DataFrame([item.to_dict() for item in self.availability]).sort_values(
            "availability_id", ignore_index=True
        )

    def to_manifest(self) -> dict[str, object]:
        """Return descriptive export lineage without expanding signature rows."""

        return {
            "export_id": self.export_id,
            "crossfit_id": self.crossfit_id,
            "repeat_id": self.repeat_id,
            "mode": self.mode,
            "coordinate_version": self.coordinate_version,
            "aggregation_version": self.aggregation_version,
            "entropy_aggregation_version": (
                WITHIN_FAMILY_ENTROPY_AGGREGATION_VERSION
            ),
            "entropy_threshold": self.entropy_threshold,
            "schema_version": self.schema_version,
            "inference_eligible": False,
            "availability": [item.to_dict() for item in self.availability],
        }


@dataclass(frozen=True, slots=True)
class _HeldoutGeneCoordinate:
    sample_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    observed: np.ndarray
    residualized_context: np.ndarray
    family_gene_effects: tuple[np.ndarray, ...]
    predicted: np.ndarray


class _UnavailableCoordinate(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _project_autonomous_span(
    values: np.ndarray,
    autonomous_basis: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    """Reproduce the frozen precision-weighted autonomous projection."""

    observed = np.asarray(values, dtype=np.float64)
    programs = np.asarray(autonomous_basis, dtype=np.float64)
    weights = np.asarray(precision, dtype=np.float64)
    if programs.shape[1] == 0:
        return cast(np.ndarray, np.array(observed, dtype=np.float64, copy=True))
    sqrt_precision = np.sqrt(weights)
    weighted_programs = sqrt_precision[:, np.newaxis] * programs
    column_scales = np.max(np.abs(weighted_programs), axis=0, initial=0.0)
    supported_columns = column_scales > 0.0
    normalized_programs = np.zeros_like(weighted_programs)
    if np.any(supported_columns):
        scaled = (
            weighted_programs[:, supported_columns]
            / column_scales[supported_columns]
        )
        scaled_norms = np.sqrt(np.sum(scaled * scaled, axis=0))
        normalized_programs[:, supported_columns] = scaled / scaled_norms
    left_vectors, singular_values, _ = np.linalg.svd(
        normalized_programs,
        full_matrices=False,
    )
    threshold = (
        0.0 if not singular_values.size else 1e-12 * float(singular_values[0])
    )
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    if numerical_rank != programs.shape[1]:
        raise _UnavailableCoordinate("autonomous_program_support_not_estimable")
    weighted_values = sqrt_precision[:, np.newaxis] * observed.T
    orthonormal_span = left_vectors[:, :numerical_rank]
    weighted_values -= orthonormal_span @ (orthonormal_span.T @ weighted_values)
    residual = np.array(observed.T, dtype=np.float64, copy=True)
    supported = sqrt_precision > 0.0
    residual[supported] = (
        weighted_values[supported] / sqrt_precision[supported, np.newaxis]
    )
    return cast(np.ndarray, np.asarray(residual.T, dtype=np.float64))


def _heldout_gene_coordinate(
    fold: CrossFitFoldArtifacts,
    chain_index: int,
    design_index: int,
) -> _HeldoutGeneCoordinate:
    response = fold.receiver_response_applications[chain_index]
    design = fold.design_applications[design_index]
    model = fold.receiver_incremental_models[chain_index]
    application = fold.receiver_incremental_applications[chain_index]
    if response.status != "ok":
        raise _UnavailableCoordinate(
            response.reason_code or "heldout_gene_response_not_estimable"
        )
    if design.status != "observed":
        raise _UnavailableCoordinate(
            design.reason_code or "heldout_design_not_estimable"
        )
    functional = model.diagnostic_functional
    diagnostic = application.diagnostic_application
    if functional is None:
        raise _UnavailableCoordinate(
            model.diagnostic_reason_code or "incremental_functional_not_estimable"
        )
    if diagnostic is None or diagnostic.status != "observed":
        raise _UnavailableCoordinate(
            application.diagnostic_reason_code
            or "heldout_incremental_application_not_estimable"
        )
    if (
        response.sample_ids != design.sample_ids
        or response.sample_subject_ids != design.sample_subject_ids
        or response.sample_context_ids != design.sample_context_ids
        or response.feature_ids != functional.feature_ids
        or diagnostic.sample_ids != response.sample_ids
        or diagnostic.sample_subject_ids != response.sample_subject_ids
        or diagnostic.sample_context_ids != response.sample_context_ids
    ):
        raise ContractError(
            "Signature coordinate parents do not share exact held-out rows",
            code="signature_export_parent_mismatch",
            field="sample_ids",
            remediation="Rerun subject cross-fit from intact parents",
        )
    standardized = (response.sample_values - functional.feature_center) / (
        functional.feature_scale
    )
    projected = _project_autonomous_span(
        standardized,
        functional.autonomous_basis,
        functional.precision_weights,
    )
    null_prediction = design.nuisance_matrix @ (
        functional.null_nuisance_coefficients
    )
    residualized_context = design.context_regressor - design.nuisance_matrix @ (
        functional.context_regressor_nuisance_coefficients
    )
    observed = projected - null_prediction
    effects = tuple(
        np.asarray(functional.family_basis.getcol(index).toarray()).ravel()
        * float(functional.family_coefficients[index])
        for index in range(len(functional.family_ids))
    )
    predicted_effect = np.sum(np.stack(effects, axis=0), axis=0)
    predicted = np.outer(residualized_context, predicted_effect)
    null_losses = np.sum(
        observed * observed * functional.precision_weights,
        axis=1,
    )
    full_residual = observed - predicted
    full_losses = np.sum(
        full_residual * full_residual * functional.precision_weights,
        axis=1,
    )
    if not np.allclose(
        null_losses,
        diagnostic.sample_null_losses,
        rtol=1e-10,
        atol=1e-12,
    ) or not np.allclose(
        full_losses,
        diagnostic.sample_full_losses,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ContractError(
            "Reconstructed gene coordinate does not reproduce held-out losses",
            code="signature_export_coordinate_mismatch",
            field="sample_full_losses",
            remediation=(
                "Use a signature adapter matching the fitted coordinate version"
            ),
        )
    return _HeldoutGeneCoordinate(
        sample_ids=response.sample_ids,
        subject_ids=response.sample_subject_ids,
        context_ids=response.sample_context_ids,
        feature_ids=functional.feature_ids,
        family_ids=functional.family_ids,
        observed=observed,
        residualized_context=np.asarray(residualized_context, dtype=np.float64),
        family_gene_effects=effects,
        predicted=predicted,
    )


def _subject_equal_mean(values: np.ndarray, subject_ids: tuple[str, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] != len(subject_ids) or not len(subject_ids):
        raise ValueError("subject-equal values must align with non-empty row IDs")
    summaries = [
        np.mean(
            array[np.asarray([value == subject for value in subject_ids], dtype=bool)],
            axis=0,
        )
        for subject in sorted(set(subject_ids))
    ]
    return cast(
        np.ndarray,
        np.asarray(np.mean(np.stack(summaries, axis=0), axis=0), dtype=np.float64),
    )


def _normalized_entropy(weights: np.ndarray) -> float:
    if len(weights) <= 1:
        return 0.0
    positive = weights[weights > 0]
    return float(-np.sum(positive * np.log(positive)) / math.log(len(weights)))


def _context_mapping(
    fold: CrossFitFoldArtifacts,
    design_index: int,
) -> dict[str, str]:
    encoder = fold.design_encoders[design_index]
    mapping: dict[str, str] = {}
    for node in encoder.contrast.weights:
        context_id, context_json = node_context_fields(node, encoder.context_keys)
        if context_id in mapping and mapping[context_id] != context_json:
            raise ContractError(
                "One context ID maps to multiple frozen context objects",
                code="signature_export_context_mismatch",
                field="context_id",
            )
        mapping[context_id] = context_json
    return mapping


def _source_artifact_id(
    artifacts: CrossFitArtifacts,
    fold: CrossFitFoldArtifacts,
    chain_index: int,
    mode: SignatureScoreMode,
) -> str:
    common = fold.family_common_applications[chain_index]
    return stable_id(
        "crossfit_signature_source_artifacts",
        {
            "common_application_id": common.application_id,
            "coordinate_version": GENE_MODEL_COORDINATE_VERSION,
            "crossfit_id": artifacts.crossfit_id,
            "design_application_id": (
                fold.receiver_incremental_applications[
                    chain_index
                ].design_application_id
            ),
            "incremental_application_id": (
                fold.receiver_incremental_applications[chain_index].application_id
            ),
            "mode": mode,
            "response_application_id": (
                fold.receiver_response_applications[chain_index].application_id
            ),
        },
        schema_version=CROSSFIT_SIGNATURE_ID_SCHEMA_VERSION,
    )


def _availability(
    *,
    artifacts: CrossFitArtifacts,
    fold_id: str,
    contrast: str,
    receiver: str,
    context_id: str,
    context_json: str,
    mode: SignatureScoreMode,
    layer: SignatureExportLayer,
    reason_code: str | None,
    scoring_functional_id: str | None,
    source_artifact_id: str,
    component_count: int,
) -> SignatureLayerAvailability:
    return SignatureLayerAvailability(
        repeat_id=artifacts.spec.repeat_id,
        fold_id=fold_id,
        contrast=contrast,
        receiver=receiver,
        context_id=context_id,
        context_json=context_json,
        mode=mode,
        layer=layer,
        status=(
            SignatureExportAvailabilityStatus.AVAILABLE
            if reason_code is None
            else SignatureExportAvailabilityStatus.NOT_ESTIMABLE
        ),
        reason_code=reason_code,
        scoring_functional_id=scoring_functional_id,
        source_artifact_id=source_artifact_id,
        component_count=component_count if reason_code is None else 0,
    )


def _component_record(
    *,
    artifacts: CrossFitArtifacts,
    fold: CrossFitFoldArtifacts,
    source_artifact_id: str,
    family_common_application_id: str,
    scoring_functional_id: str,
    mode: SignatureScoreMode,
    context_id: str,
    context_json: str,
    receiver: str,
    contrast: str,
    feature_id: str,
    family_id: str,
    entropy: float,
    observed: float,
    predicted: float,
    residual: float,
    contribution: float | None,
    component_level: SignatureComponentLevel,
    component_status: LayeredSignatureStatus,
    reason_code: str | None,
    interaction_id: str | None = None,
    sender: str | None = None,
    entropy_status: str = "resolved_from_frozen_lr_weights",
) -> FittedSignatureComponentRecord:
    source_record_id = stable_id(
        "crossfit_fitted_signature_component",
        {
            "component_level": component_level.value,
            "context_id": context_id,
            "contrast": contrast,
            "family_id": family_id,
            "feature_id": feature_id,
            "fold_id": fold.fold_id,
            "interaction_id": interaction_id,
            "mode": mode,
            "receiver": receiver,
            "repeat_id": artifacts.spec.repeat_id,
            "sender": sender,
            "source_artifact_id": source_artifact_id,
        },
        schema_version=CROSSFIT_SIGNATURE_ID_SCHEMA_VERSION,
    )
    provenance = {
        "aggregation_version": SUBJECT_CONTEXT_AGGREGATION_VERSION,
        "coordinate_version": GENE_MODEL_COORDINATE_VERSION,
        "crossfit_id": artifacts.crossfit_id,
        "descriptive_only": True,
        "entropy_aggregation_version": (
            WITHIN_FAMILY_ENTROPY_AGGREGATION_VERSION
        ),
        "entropy_status": entropy_status,
        "family_common_application_id": family_common_application_id,
        "mode": mode,
        "no_model_refit": True,
        "scalar_score_fields_used_as_gene_contribution": [],
        "weight_sources": {
            "lr": "within_family_lr_weight",
            "sender": "assignment_weight",
        },
    }
    return FittedSignatureComponentRecord(
        component_level=component_level,
        source_record_id=source_record_id,
        context_id=context_id,
        context_json=context_json,
        receiver=receiver,
        contrast=contrast,
        feature_id=feature_id,
        feature_kind=SignatureFeatureKind.GENE,
        feature_namespace=fold.training.target_prior.gene_namespace.value,
        family_id=family_id,
        interaction_id=interaction_id,
        sender=sender,
        observed=observed,
        predicted=predicted,
        residual=residual,
        attributed_contribution=contribution,
        response_status=LayeredSignatureStatus.OK,
        component_status=component_status,
        reason_code=reason_code,
        within_family_entropy=entropy,
        uncertainty=None,
        uncertainty_kind=None,
        source_artifact_id=source_artifact_id,
        scoring_functional_id=scoring_functional_id,
        fold_id=fold.fold_id,
        repeat_id=artifacts.spec.repeat_id,
        provenance_json=canonical_json(provenance),
    )


def _component_status(
    contribution: float,
    *,
    estimable: bool,
    unavailable_reason: str | None,
    zero_reason: str,
) -> tuple[LayeredSignatureStatus, float | None, str | None]:
    if not estimable:
        return (
            LayeredSignatureStatus.NOT_ESTIMABLE,
            None,
            unavailable_reason or "fitted_family_not_estimable",
        )
    if contribution == 0.0:
        return LayeredSignatureStatus.STRUCTURAL_ZERO, 0.0, zero_reason
    return LayeredSignatureStatus.OK, contribution, None


def _resolved_lr_weights(
    member: pd.DataFrame,
    *,
    sample_ids: tuple[str, ...],
    family_id: str,
    interaction_ids: tuple[str, ...],
) -> tuple[np.ndarray, float]:
    local = member.loc[
        member["sample_id"].astype(str).isin(sample_ids)
        & member["family_id"].astype(str).eq(family_id)
        & member["interaction_id"].astype(str).isin(interaction_ids)
    ].copy(deep=True)
    expected = {
        (sample_id, interaction_id)
        for sample_id in sample_ids
        for interaction_id in interaction_ids
    }
    observed = set(
        zip(
            local["sample_id"].astype(str),
            local["interaction_id"].astype(str),
            strict=True,
        )
    )
    if observed != expected or local.duplicated(
        ["sample_id", "interaction_id"]
    ).any():
        raise _UnavailableCoordinate(
            f"lr_weight_coverage_incomplete:{family_id}"
        )
    matrix: np.ndarray = np.empty(
        (len(sample_ids), len(interaction_ids)), dtype=np.float64
    )
    entropies: list[float] = []
    for sample_index, sample_id in enumerate(sample_ids):
        rows = local.loc[local["sample_id"].astype(str).eq(sample_id)].set_index(
            "interaction_id"
        )
        values = pd.to_numeric(
            rows.loc[list(interaction_ids), "within_family_lr_weight"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise _UnavailableCoordinate(
                f"within_family_lr_weight_not_estimable:{family_id}"
            )
        if not math.isclose(
            float(np.sum(values)), 1.0, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise _UnavailableCoordinate(
                f"within_family_lr_weight_not_conservative:{family_id}"
            )
        entropy = _normalized_entropy(values)
        reported = pd.to_numeric(
            rows.loc[list(interaction_ids), "within_family_entropy"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        finite_reported = reported[np.isfinite(reported)]
        if finite_reported.size and (
            not np.allclose(finite_reported, entropy, rtol=1e-10, atol=1e-12)
        ):
            raise ContractError(
                "Reported within-family entropy disagrees with frozen LR weights",
                code="signature_export_entropy_mismatch",
                field="within_family_entropy",
                remediation="Reapply family-common scoring from intact weights",
            )
        matrix[sample_index] = values
        entropies.append(entropy)
    return matrix, max(entropies)


def _resolved_sender_weights(
    sender: pd.DataFrame,
    *,
    sample_ids: tuple[str, ...],
    family_id: str,
    interaction_id: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    local = sender.loc[
        sender["sample_id"].astype(str).isin(sample_ids)
        & sender["family_id"].astype(str).eq(family_id)
        & sender["interaction_id"].astype(str).eq(interaction_id)
    ].copy(deep=True)
    sender_sets = {
        sample_id: tuple(
            sorted(
                local.loc[
                    local["sample_id"].astype(str).eq(sample_id), "sender"
                ].astype(str)
            )
        )
        for sample_id in sample_ids
    }
    unique_sets = set(sender_sets.values())
    if len(unique_sets) != 1 or not unique_sets or not next(iter(unique_sets)):
        raise _UnavailableCoordinate(
            f"sender_candidate_coverage_incomplete:{family_id}:{interaction_id}"
        )
    sender_ids = next(iter(unique_sets))
    expected = {
        (sample_id, sender_id)
        for sample_id in sample_ids
        for sender_id in sender_ids
    }
    observed = set(
        zip(
            local["sample_id"].astype(str),
            local["sender"].astype(str),
            strict=True,
        )
    )
    if observed != expected or local.duplicated(["sample_id", "sender"]).any():
        raise _UnavailableCoordinate(
            f"sender_weight_coverage_incomplete:{family_id}:{interaction_id}"
        )
    matrix: np.ndarray = np.empty(
        (len(sample_ids), len(sender_ids)), dtype=np.float64
    )
    for sample_index, sample_id in enumerate(sample_ids):
        rows = local.loc[local["sample_id"].astype(str).eq(sample_id)].set_index(
            "sender"
        )
        values = pd.to_numeric(
            rows.loc[list(sender_ids), "assignment_weight"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise _UnavailableCoordinate(
                f"sender_assignment_weight_not_estimable:{family_id}:{interaction_id}"
            )
        if not math.isclose(
            float(np.sum(values)), 1.0, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise _UnavailableCoordinate(
                f"sender_assignment_weight_not_conservative:{family_id}:"
                f"{interaction_id}"
            )
        matrix[sample_index] = values
    return sender_ids, matrix


def _unavailable_context_records(
    *,
    artifacts: CrossFitArtifacts,
    fold: CrossFitFoldArtifacts,
    contrast: str,
    receiver: str,
    context_id: str,
    context_json: str,
    mode: SignatureScoreMode,
    reason_code: str,
    scoring_functional_id: str | None,
    source_artifact_id: str,
) -> list[SignatureLayerAvailability]:
    return [
        _availability(
            artifacts=artifacts,
            fold_id=fold.fold_id,
            contrast=contrast,
            receiver=receiver,
            context_id=context_id,
            context_json=context_json,
            mode=mode,
            layer=layer,
            reason_code=reason_code,
            scoring_functional_id=scoring_functional_id,
            source_artifact_id=source_artifact_id,
            component_count=0,
        )
        for layer in SignatureExportLayer
    ]


def _export_chain(
    artifacts: CrossFitArtifacts,
    fold: CrossFitFoldArtifacts,
    *,
    chain_index: int,
    design_index: int,
    mode: SignatureScoreMode,
    entropy_threshold: float,
) -> tuple[LayeredSignatureTable | None, list[SignatureLayerAvailability]]:
    model = fold.receiver_incremental_models[chain_index]
    common_functional = fold.family_common_functionals[chain_index]
    common_application = fold.family_common_applications[chain_index]
    contrast = model.contrast_name
    receiver = model.receiver
    scoring_id = common_functional.family_common_functional_id
    source_id = _source_artifact_id(artifacts, fold, chain_index, mode)
    contexts = _context_mapping(fold, design_index)
    response = fold.receiver_response_applications[chain_index]
    context_ids = tuple(sorted(set(response.sample_context_ids)))
    for context_id in context_ids:
        if context_id not in contexts:
            raise ContractError(
                "Held-out response contains a context outside the frozen contrast",
                code="signature_export_context_mismatch",
                field="context_id",
            )
    try:
        coordinate = _heldout_gene_coordinate(fold, chain_index, design_index)
    except _UnavailableCoordinate as unavailable:
        audit: list[SignatureLayerAvailability] = []
        for context_id in context_ids:
            audit.extend(
                _unavailable_context_records(
                    artifacts=artifacts,
                    fold=fold,
                    contrast=contrast,
                    receiver=receiver,
                    context_id=context_id,
                    context_json=contexts[context_id],
                    mode=mode,
                    reason_code=unavailable.reason_code,
                    scoring_functional_id=scoring_id,
                    source_artifact_id=source_id,
                )
            )
        return None, audit
    if common_functional.incremental_functional is None or (
        common_functional.incremental_functional.incremental_functional_id
        != model.diagnostic_functional.incremental_functional_id
        if model.diagnostic_functional is not None
        else True
    ):
        raise ContractError(
            "Family-common scoring does not share the reconstructed functional",
            code="signature_export_parent_mismatch",
            field="incremental_functional_id",
        )
    interactions_by_family: dict[str, tuple[str, ...]] = {}
    for family_id in coordinate.family_ids:
        interaction_ids = tuple(
            sorted(
                item.interaction_id
                for item in common_functional.interactions
                if item.family_id == family_id
            )
        )
        if not interaction_ids:
            raise ContractError(
                "An active fitted family has no frozen LR members",
                code="signature_export_parent_mismatch",
                field="family_id",
            )
        interactions_by_family[family_id] = interaction_ids
    member = common_application.member_scores
    sender = common_application.sender_scores
    member = member.loc[member["mode"].astype(str).eq(mode)].copy(deep=True)
    sender = sender.loc[sender["mode"].astype(str).eq(mode)].copy(deep=True)
    family_records: list[FittedSignatureComponentRecord] = []
    lr_records: list[FittedSignatureComponentRecord] = []
    sender_records: list[FittedSignatureComponentRecord] = []
    audit = []
    assert model.diagnostic_functional is not None
    fitted = model.diagnostic_functional
    for context_id in context_ids:
        context_mask = np.asarray(
            [value == context_id for value in coordinate.context_ids], dtype=bool
        )
        indices = np.flatnonzero(context_mask)
        sample_ids = tuple(coordinate.sample_ids[index] for index in indices)
        subject_ids = tuple(coordinate.subject_ids[index] for index in indices)
        observed = _subject_equal_mean(coordinate.observed[indices], subject_ids)
        predicted = _subject_equal_mean(coordinate.predicted[indices], subject_ids)
        residual = observed - predicted
        family_scalar = float(
            _subject_equal_mean(
                coordinate.residualized_context[indices], subject_ids
            )
        )
        family_contributions = tuple(
            gene_effect * family_scalar
            for gene_effect in coordinate.family_gene_effects
        )
        lr_weights: dict[str, tuple[np.ndarray, float]] = {}
        try:
            for family_id in coordinate.family_ids:
                lr_weights[family_id] = _resolved_lr_weights(
                    member,
                    sample_ids=sample_ids,
                    family_id=family_id,
                    interaction_ids=interactions_by_family[family_id],
                )
        except _UnavailableCoordinate as unavailable:
            receiver_only_family_records: list[
                FittedSignatureComponentRecord
            ] = []
            for family_index, family_id in enumerate(coordinate.family_ids):
                family_contribution = family_contributions[family_index]
                for feature_index, feature_id in enumerate(coordinate.feature_ids):
                    family_value = float(family_contribution[feature_index])
                    family_status, released_family, family_reason = _component_status(
                        family_value,
                        estimable=bool(fitted.family_estimable[family_index]),
                        unavailable_reason=fitted.family_reason_codes[family_index],
                        zero_reason="fitted_family_gene_contribution_zero",
                    )
                    receiver_only_family_records.append(
                        _component_record(
                            artifacts=artifacts,
                            fold=fold,
                            source_artifact_id=source_id,
                            family_common_application_id=(
                                common_application.application_id
                            ),
                            scoring_functional_id=scoring_id,
                            mode=mode,
                            context_id=context_id,
                            context_json=contexts[context_id],
                            receiver=receiver,
                            contrast=contrast,
                            feature_id=feature_id,
                            family_id=family_id,
                            entropy=0.0,
                            observed=float(observed[feature_index]),
                            predicted=float(predicted[feature_index]),
                            residual=float(residual[feature_index]),
                            contribution=released_family,
                            component_level=SignatureComponentLevel.FAMILY,
                            component_status=family_status,
                            reason_code=family_reason,
                            entropy_status=(
                                "not_available_receiver_only_placeholder_not_released"
                            ),
                        )
                    )
            family_records.extend(receiver_only_family_records)
            audit.append(
                _availability(
                    artifacts=artifacts,
                    fold_id=fold.fold_id,
                    contrast=contrast,
                    receiver=receiver,
                    context_id=context_id,
                    context_json=contexts[context_id],
                    mode=mode,
                    layer=SignatureExportLayer.RECEIVER_CONTEXT,
                    reason_code=None,
                    scoring_functional_id=scoring_id,
                    source_artifact_id=source_id,
                    component_count=len(coordinate.feature_ids),
                )
            )
            for layer in (
                SignatureExportLayer.LR_ATTRIBUTED,
                SignatureExportLayer.SENDER_LR_RECEIVER,
            ):
                audit.append(
                    _availability(
                        artifacts=artifacts,
                        fold_id=fold.fold_id,
                        contrast=contrast,
                        receiver=receiver,
                        context_id=context_id,
                        context_json=contexts[context_id],
                        mode=mode,
                        layer=layer,
                        reason_code=unavailable.reason_code,
                        scoring_functional_id=scoring_id,
                        source_artifact_id=source_id,
                        component_count=0,
                    )
                )
            continue
        context_family_records: list[FittedSignatureComponentRecord] = []
        context_lr_records: list[FittedSignatureComponentRecord] = []
        context_sender_records: list[FittedSignatureComponentRecord] = []
        sender_unavailable_reason: str | None = None
        for family_index, family_id in enumerate(coordinate.family_ids):
            weight_matrix, entropy = lr_weights[family_id]
            gene_effect = coordinate.family_gene_effects[family_index]
            family_contribution = family_contributions[family_index]
            interaction_ids = interactions_by_family[family_id]
            lr_contributions: dict[str, np.ndarray] = {}
            for interaction_index, interaction_id in enumerate(interaction_ids):
                lr_scalar = float(
                    _subject_equal_mean(
                        coordinate.residualized_context[indices]
                        * weight_matrix[:, interaction_index],
                        subject_ids,
                    )
                )
                lr_contributions[interaction_id] = gene_effect * lr_scalar
            if not np.allclose(
                np.sum(np.stack(tuple(lr_contributions.values())), axis=0),
                family_contribution,
                rtol=1e-9,
                atol=1e-10,
            ):
                raise ContractError(
                    "LR weights do not conserve the fitted family gene contribution",
                    code="signature_export_lr_conservation_failure",
                    field="within_family_lr_weight",
                )
            sender_allocations: dict[
                str, tuple[tuple[str, ...], np.ndarray]
            ] = {}
            if sender_unavailable_reason is None:
                try:
                    for interaction_id in interaction_ids:
                        sender_allocations[interaction_id] = _resolved_sender_weights(
                            sender,
                            sample_ids=sample_ids,
                            family_id=family_id,
                            interaction_id=interaction_id,
                        )
                except _UnavailableCoordinate as unavailable:
                    sender_unavailable_reason = unavailable.reason_code
                    context_sender_records.clear()
                    sender_allocations.clear()
            for feature_index, feature_id in enumerate(coordinate.feature_ids):
                family_value = float(family_contribution[feature_index])
                family_status, released_family, family_reason = _component_status(
                    family_value,
                    estimable=bool(fitted.family_estimable[family_index]),
                    unavailable_reason=fitted.family_reason_codes[family_index],
                    zero_reason="fitted_family_gene_contribution_zero",
                )
                context_family_records.append(
                    _component_record(
                        artifacts=artifacts,
                        fold=fold,
                        source_artifact_id=source_id,
                        family_common_application_id=(
                            common_application.application_id
                        ),
                        scoring_functional_id=scoring_id,
                        mode=mode,
                        context_id=context_id,
                        context_json=contexts[context_id],
                        receiver=receiver,
                        contrast=contrast,
                        feature_id=feature_id,
                        family_id=family_id,
                        entropy=entropy,
                        observed=float(observed[feature_index]),
                        predicted=float(predicted[feature_index]),
                        residual=float(residual[feature_index]),
                        contribution=released_family,
                        component_level=SignatureComponentLevel.FAMILY,
                        component_status=family_status,
                        reason_code=family_reason,
                    )
                )
                for interaction_index, interaction_id in enumerate(interaction_ids):
                    lr_value = float(lr_contributions[interaction_id][feature_index])
                    lr_status, released_lr, lr_reason = _component_status(
                        lr_value,
                        estimable=bool(fitted.family_estimable[family_index]),
                        unavailable_reason=fitted.family_reason_codes[family_index],
                        zero_reason="fitted_lr_gene_contribution_zero",
                    )
                    context_lr_records.append(
                        _component_record(
                            artifacts=artifacts,
                            fold=fold,
                            source_artifact_id=source_id,
                            family_common_application_id=(
                                common_application.application_id
                            ),
                            scoring_functional_id=scoring_id,
                            mode=mode,
                            context_id=context_id,
                            context_json=contexts[context_id],
                            receiver=receiver,
                            contrast=contrast,
                            feature_id=feature_id,
                            family_id=family_id,
                            entropy=entropy,
                            observed=float(observed[feature_index]),
                            predicted=float(predicted[feature_index]),
                            residual=float(residual[feature_index]),
                            contribution=released_lr,
                            component_level=SignatureComponentLevel.LR,
                            component_status=lr_status,
                            reason_code=lr_reason,
                            interaction_id=interaction_id,
                        )
                    )
                    if sender_unavailable_reason is not None:
                        continue
                    sender_ids, assignment_matrix = sender_allocations[interaction_id]
                    sender_values: list[float] = []
                    for sender_index, sender_id in enumerate(sender_ids):
                        sender_scalar = float(
                            _subject_equal_mean(
                                coordinate.residualized_context[indices]
                                * weight_matrix[:, interaction_index]
                                * assignment_matrix[:, sender_index],
                                subject_ids,
                            )
                        )
                        sender_value = float(gene_effect[feature_index] * sender_scalar)
                        sender_values.append(sender_value)
                        sender_status, released_sender, sender_reason = (
                            _component_status(
                                sender_value,
                                estimable=bool(fitted.family_estimable[family_index]),
                                unavailable_reason=(
                                    fitted.family_reason_codes[family_index]
                                ),
                                zero_reason="fitted_sender_gene_contribution_zero",
                            )
                        )
                        context_sender_records.append(
                            _component_record(
                                artifacts=artifacts,
                                fold=fold,
                                source_artifact_id=source_id,
                                family_common_application_id=(
                                    common_application.application_id
                                ),
                                scoring_functional_id=scoring_id,
                                mode=mode,
                                context_id=context_id,
                                context_json=contexts[context_id],
                                receiver=receiver,
                                contrast=contrast,
                                feature_id=feature_id,
                                family_id=family_id,
                                entropy=entropy,
                                observed=float(observed[feature_index]),
                                predicted=float(predicted[feature_index]),
                                residual=float(residual[feature_index]),
                                contribution=released_sender,
                                component_level=(
                                    SignatureComponentLevel.SENDER_LR
                                ),
                                component_status=sender_status,
                                reason_code=sender_reason,
                                interaction_id=interaction_id,
                                sender=sender_id,
                            )
                        )
                    if not math.isclose(
                        sum(sender_values),
                        lr_value,
                        rel_tol=1e-9,
                        abs_tol=1e-10,
                    ):
                        raise ContractError(
                            "Sender weights do not conserve the fitted LR contribution",
                            code="signature_export_sender_conservation_failure",
                            field="assignment_weight",
                        )
        family_records.extend(context_family_records)
        lr_records.extend(context_lr_records)
        if sender_unavailable_reason is None:
            sender_records.extend(context_sender_records)
        audit.append(
            _availability(
                artifacts=artifacts,
                fold_id=fold.fold_id,
                contrast=contrast,
                receiver=receiver,
                context_id=context_id,
                context_json=contexts[context_id],
                mode=mode,
                layer=SignatureExportLayer.RECEIVER_CONTEXT,
                reason_code=None,
                scoring_functional_id=scoring_id,
                source_artifact_id=source_id,
                component_count=len(coordinate.feature_ids),
            )
        )
        audit.append(
            _availability(
                artifacts=artifacts,
                fold_id=fold.fold_id,
                contrast=contrast,
                receiver=receiver,
                context_id=context_id,
                context_json=contexts[context_id],
                mode=mode,
                layer=SignatureExportLayer.LR_ATTRIBUTED,
                reason_code=None,
                scoring_functional_id=scoring_id,
                source_artifact_id=source_id,
                component_count=len(context_lr_records),
            )
        )
        audit.append(
            _availability(
                artifacts=artifacts,
                fold_id=fold.fold_id,
                contrast=contrast,
                receiver=receiver,
                context_id=context_id,
                context_json=contexts[context_id],
                mode=mode,
                layer=SignatureExportLayer.SENDER_LR_RECEIVER,
                reason_code=sender_unavailable_reason,
                scoring_functional_id=scoring_id,
                source_artifact_id=source_id,
                component_count=len(context_sender_records),
            )
        )
    if not family_records:
        return None, audit
    return (
        build_layered_signature_table(
            family_records,
            lr_records,
            sender_records,
            entropy_threshold=entropy_threshold,
        ),
        audit,
    )


def _combine_tables(
    tables: list[LayeredSignatureTable],
    *,
    entropy_threshold: float,
) -> LayeredSignatureTable | None:
    if not tables:
        return None
    return LayeredSignatureTable(
        receiver_context=pd.concat(
            [table.receiver_context for table in tables], ignore_index=True
        ).sort_values("signature_id", ignore_index=True),
        lr_attributed=pd.concat(
            [table.lr_attributed for table in tables], ignore_index=True
        ).sort_values("signature_id", ignore_index=True),
        sender_lr_receiver=pd.concat(
            [table.sender_lr_receiver for table in tables], ignore_index=True
        ).sort_values("signature_id", ignore_index=True),
        entropy_threshold=entropy_threshold,
    )


def export_crossfit_layered_signatures(
    artifacts: CrossFitArtifacts,
    *,
    mode: SignatureScoreMode = "state",
    entropy_threshold: float = 0.8,
) -> CrossFitLayeredSignatureExport:
    """Reconstruct three descriptive signature layers without any new fitting.

    Rows remain outer-fold scoped through their fold and scoring-functional
    lineage.  Technical rows are averaged within subject/context before equal
    subject averaging.  LR and sender allocation is performed before this
    aggregation, so every released child layer exactly conserves its parent.
    """

    if not isinstance(artifacts, CrossFitArtifacts):
        raise TypeError("artifacts must be CrossFitArtifacts")
    if mode not in {"state", "ecosystem"}:
        raise ValueError("mode must be state or ecosystem")
    if not math.isfinite(entropy_threshold) or not 0 <= entropy_threshold <= 1:
        raise ValueError("entropy_threshold must lie in [0, 1]")
    artifacts._require_intact()
    tables: list[LayeredSignatureTable] = []
    audit: list[SignatureLayerAvailability] = []
    for fold in sorted(artifacts.folds, key=lambda item: item.fold_id):
        design_by_contrast = {
            encoder.contrast.name: index
            for index, encoder in enumerate(fold.design_encoders)
        }
        if not fold.family_common_functionals:
            for chain_index, model in enumerate(fold.receiver_incremental_models):
                response = fold.receiver_response_applications[chain_index]
                design_index = design_by_contrast[model.contrast_name]
                contexts = _context_mapping(fold, design_index)
                source_id = stable_id(
                    "crossfit_signature_unavailable_source",
                    {
                        "crossfit_id": artifacts.crossfit_id,
                        "fold_id": fold.fold_id,
                        "incremental_training_artifact_id": (
                            model.training_artifact_id
                        ),
                        "mode": mode,
                    },
                    schema_version=CROSSFIT_SIGNATURE_ID_SCHEMA_VERSION,
                )
                for context_id in sorted(set(response.sample_context_ids)):
                    audit.extend(
                        _unavailable_context_records(
                            artifacts=artifacts,
                            fold=fold,
                            contrast=model.contrast_name,
                            receiver=model.receiver,
                            context_id=context_id,
                            context_json=contexts[context_id],
                            mode=mode,
                            reason_code=(
                                "family_common_functional_not_produced_without_"
                                "penalty_tuning"
                            ),
                            scoring_functional_id=None,
                            source_artifact_id=source_id,
                        )
                    )
            continue
        for chain_index, model in enumerate(fold.receiver_incremental_models):
            design_index = design_by_contrast[model.contrast_name]
            table, chain_audit = _export_chain(
                artifacts,
                fold,
                chain_index=chain_index,
                design_index=design_index,
                mode=mode,
                entropy_threshold=entropy_threshold,
            )
            if table is not None:
                tables.append(table)
            audit.extend(chain_audit)
    return CrossFitLayeredSignatureExport(
        crossfit_id=artifacts.crossfit_id,
        repeat_id=artifacts.spec.repeat_id,
        mode=mode,
        table=_combine_tables(tables, entropy_threshold=entropy_threshold),
        availability=tuple(audit),
        entropy_threshold=entropy_threshold,
    )


__all__ = [
    "CROSSFIT_SIGNATURE_SCHEMA_VERSION",
    "GENE_MODEL_COORDINATE_VERSION",
    "SUBJECT_CONTEXT_AGGREGATION_VERSION",
    "WITHIN_FAMILY_ENTROPY_AGGREGATION_VERSION",
    "CrossFitLayeredSignatureExport",
    "SignatureExportAvailabilityStatus",
    "SignatureExportLayer",
    "SignatureLayerAvailability",
    "SignatureScoreMode",
    "export_crossfit_layered_signatures",
]
