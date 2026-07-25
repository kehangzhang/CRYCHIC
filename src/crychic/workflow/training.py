"""Producer-owned training boundary for future subject-level cross-fitting.

This module intentionally implements only the first real fold-trained stage:
the interaction availability universe.  It does not construct an OOF scoring
functional or claim that the remaining response/attribution stages are
cross-fitted.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from crychic.availability import (
    AvailabilityParameters,
    BatchAvailability,
    FrozenInteractionUniverse,
    estimate_bundle_availability,
)
from crychic.core import ContractError, CrychicConfig, canonical_json, stable_id
from crychic.core._validation import (
    record_validation,
    validation_is_cached,
    validation_scope,
)
from crychic.data import (
    ExpressionTransform,
    InputSchema,
    ValidatedInput,
    validate_anndata,
)
from crychic.design import ContrastSpec, balanced_contrast
from crychic.pseudobulk import (
    ExploratoryAggregate,
    MissingnessReason,
    PseudobulkDataset,
    aggregate_pseudobulk,
)
from crychic.pseudobulk.aggregation import _unit_id
from crychic.resources import ResourceBundle, TargetPrior
from crychic.sender import (
    ContrastCommonSenderFunctional,
    ContrastCommonSenderParameters,
    fit_contrast_common_sender_functional,
    freeze_common_sender_candidate_manifest,
)

_PRODUCER_MARKER = "crychic.workflow.training.v1"
_PARTIAL_STATUS = "training_only_partial_not_oof"
_ROOT_INPUT_PRODUCER = "crychic.workflow.sanitized_raw_input_identity.v1"
_ROOT_SNAPSHOT_PRODUCER = "crychic.workflow.sanitized_raw_input_snapshot.v1"


def _availability_parameter_payload(
    parameters: AvailabilityParameters,
) -> dict[str, float]:
    return {
        "complex_epsilon": parameters.complex_epsilon,
        "complex_power": parameters.complex_power,
        "detection_alpha": parameters.detection.alpha,
        "detection_beta": parameters.detection.beta,
        "detection_exponent": parameters.detection.exponent,
        "hill_coefficient": parameters.hill.coefficient,
        "hill_half_saturation": parameters.hill.half_saturation,
    }


@lru_cache(maxsize=8)
def _resource_bundle_content_id(resource_bundle: ResourceBundle) -> str:
    """Bind the in-memory interaction content used during held-out application."""

    result: str = stable_id("resource_bundle_content", asdict(resource_bundle))
    return result


@lru_cache(maxsize=8)
def _target_prior_content_id(target_prior: TargetPrior) -> str:
    """Bind the complete in-memory target prior used by training children."""

    result: str = stable_id("target_prior_content", asdict(target_prior))
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class FoldTrainingSpec:
    """Pre-registered settings for the implemented fold-training stage."""

    min_cells: int = 10
    min_pooled_availability: float = 0.01
    max_interactions: int | None = None
    availability_parameters: AvailabilityParameters = field(
        default_factory=AvailabilityParameters
    )
    sender_parameters: ContrastCommonSenderParameters = field(
        default_factory=ContrastCommonSenderParameters
    )
    sender_contrasts: tuple[ContrastSpec, ...] | None = None
    schema_version: str = "1.0.0"
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_cells, bool)
            or not isinstance(self.min_cells, int)
            or self.min_cells < 1
        ):
            raise ValueError("min_cells must be an integer >= 1")
        threshold = float(self.min_pooled_availability)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("min_pooled_availability must lie in [0, 1]")
        maximum = self.max_interactions
        if maximum is not None and (
            isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
        ):
            raise ValueError("max_interactions must be a positive integer or None")
        if not isinstance(self.availability_parameters, AvailabilityParameters):
            raise TypeError(
                "availability_parameters must be an AvailabilityParameters instance"
            )
        if not isinstance(self.sender_parameters, ContrastCommonSenderParameters):
            raise TypeError(
                "sender_parameters must be ContrastCommonSenderParameters"
            )
        self.sender_parameters._require_intact()
        contrasts = self.sender_contrasts
        if contrasts is not None:
            contrasts = tuple(contrasts)
            if not contrasts:
                raise ValueError("sender_contrasts must be non-empty when declared")
            if any(not isinstance(contrast, ContrastSpec) for contrast in contrasts):
                raise TypeError("sender_contrasts must contain ContrastSpec values")
            if any(
                not contrast.estimable or len(contrast.weights) < 2
                for contrast in contrasts
            ):
                raise ValueError(
                    "sender_contrasts must be estimable multi-context contrasts"
                )
            contrast_ids = [
                stable_id("contrast", contrast.to_dict()) for contrast in contrasts
            ]
            if len(contrast_ids) != len(set(contrast_ids)):
                raise ValueError("sender_contrasts must be unique")
            contrasts = tuple(
                contrast
                for _, contrast in sorted(
                    zip(contrast_ids, contrasts, strict=True),
                    key=lambda item: item[0],
                )
            )
        if self.schema_version != "1.0.0":
            raise ValueError("FoldTrainingSpec schema_version must be 1.0.0")
        payload: dict[str, object] = {
            "availability_parameters": _availability_parameter_payload(
                self.availability_parameters
            ),
            "max_interactions": maximum,
            "min_cells": self.min_cells,
            "min_pooled_availability": threshold,
            "schema_version": self.schema_version,
            "sender_parameter_manifest_id": (
                self.sender_parameters.parameter_manifest_id
            ),
        }
        if contrasts is not None:
            payload["sender_contrasts"] = [
                contrast.to_dict() for contrast in contrasts
            ]
        object.__setattr__(self, "sender_contrasts", contrasts)
        object.__setattr__(self, "min_pooled_availability", threshold)
        object.__setattr__(
            self,
            "spec_id",
            stable_id(
                "fold_training_spec",
                payload,
                schema_version=self.schema_version,
            ),
        )

    def _require_intact(self) -> None:
        """Reject forced mutation of the fold-training policy."""

        try:
            self.sender_parameters._require_intact()
            repeated = FoldTrainingSpec(
                min_cells=self.min_cells,
                min_pooled_availability=self.min_pooled_availability,
                max_interactions=self.max_interactions,
                availability_parameters=self.availability_parameters,
                sender_parameters=self.sender_parameters,
                sender_contrasts=self.sender_contrasts,
                schema_version=self.schema_version,
            )
            valid = (
                self.min_cells == repeated.min_cells
                and self.min_pooled_availability
                == repeated.min_pooled_availability
                and self.max_interactions == repeated.max_interactions
                and self.availability_parameters
                == repeated.availability_parameters
                and self.sender_parameters.parameter_manifest_id
                == repeated.sender_parameters.parameter_manifest_id
                and self.sender_contrasts == repeated.sender_contrasts
                and self.schema_version == repeated.schema_version
                and self.spec_id == repeated.spec_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Fold-training specification failed integrity validation",
                code="fold_training_spec_integrity_violation",
                field="spec_id",
                remediation="Rebuild FoldTrainingSpec from the declared policy",
            ) from error
        if not valid:
            raise ContractError(
                "Fold-training specification failed integrity validation",
                code="fold_training_spec_integrity_violation",
                field="spec_id",
                remediation="Rebuild FoldTrainingSpec from the declared policy",
            )


@dataclass(frozen=True, slots=True, init=False)
class TrainingArtifacts:
    """Producer-owned, partial fold artifacts that make no OOF claim."""

    config: CrychicConfig
    resource_bundle: ResourceBundle
    target_prior: TargetPrior
    spec: FoldTrainingSpec
    training_subject_ids: tuple[str, ...]
    training_sample_ids: tuple[str, ...]
    cell_type_ids: tuple[str, ...]
    training_input_digest: str
    resource_bundle_content_id: str
    target_prior_content_id: str
    frozen_interaction_universe: FrozenInteractionUniverse
    sender_functionals: tuple[ContrastCommonSenderFunctional, ...]
    sender_availability_input_digests: tuple[tuple[str, str], ...]
    completed_stages: tuple[str, ...]
    remaining_stages: tuple[str, ...]
    certification_status: str
    training_artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "TrainingArtifacts are producer-owned; use fit_training_artifacts()"
        )

    @classmethod
    def _from_training(
        cls,
        *,
        config: CrychicConfig,
        resource_bundle: ResourceBundle,
        target_prior: TargetPrior,
        spec: FoldTrainingSpec,
        training_subject_ids: tuple[str, ...],
        training_sample_ids: tuple[str, ...],
        cell_type_ids: tuple[str, ...],
        training_input_digest: str,
        frozen_interaction_universe: FrozenInteractionUniverse,
        sender_functionals: tuple[ContrastCommonSenderFunctional, ...],
        sender_availability_input_digests: tuple[tuple[str, str], ...],
    ) -> TrainingArtifacts:
        if not isinstance(config, CrychicConfig):
            raise TypeError("config must be a CrychicConfig")
        if not isinstance(resource_bundle, ResourceBundle):
            raise TypeError("resource_bundle must be a ResourceBundle")
        if not isinstance(target_prior, TargetPrior):
            raise TypeError("target_prior must be a TargetPrior")
        if not isinstance(spec, FoldTrainingSpec):
            raise TypeError("spec must be a FoldTrainingSpec")
        spec._require_intact()
        if not isinstance(frozen_interaction_universe, FrozenInteractionUniverse):
            raise TypeError(
                "frozen_interaction_universe must be a FrozenInteractionUniverse"
            )
        frozen_interaction_universe._require_intact()
        if resource_bundle.species is not target_prior.species:
            raise ValueError("resource bundle and target prior species must match")
        if resource_bundle.gene_namespace is not target_prior.gene_namespace:
            raise ValueError("resource bundle and target prior namespace must match")
        if (
            frozen_interaction_universe.resource_id != resource_bundle.resource_id
            or frozen_interaction_universe.resource_version
            != resource_bundle.version
            or frozen_interaction_universe.resource_manifest_digest
            != resource_bundle.manifest_digest
            or frozen_interaction_universe.min_pooled_availability
            != spec.min_pooled_availability
            or frozen_interaction_universe.max_interactions
            != spec.max_interactions
        ):
            raise ValueError(
                "frozen interaction universe does not match its resource and spec"
            )
        if not isinstance(training_input_digest, str) or not training_input_digest:
            raise ValueError("training_input_digest must be a non-empty identifier")
        subjects = tuple(sorted(training_subject_ids))
        samples = tuple(sorted(training_sample_ids))
        cell_types = tuple(sorted(cell_type_ids))
        if not subjects or len(subjects) != len(set(subjects)):
            raise ValueError("training subjects must be non-empty and unique")
        if not samples or len(samples) != len(set(samples)):
            raise ValueError("training samples must be non-empty and unique")
        if not cell_types or len(cell_types) != len(set(cell_types)):
            raise ValueError("training cell types must be non-empty and unique")
        if subjects != frozen_interaction_universe.training_subject_ids:
            raise ValueError(
                "frozen interaction universe subjects do not match the raw input"
            )
        functionals = tuple(sender_functionals)
        if any(
            not isinstance(functional, ContrastCommonSenderFunctional)
            for functional in functionals
        ):
            raise TypeError(
                "sender_functionals must contain ContrastCommonSenderFunctional"
            )
        for functional in functionals:
            functional._require_intact()
        sender_digests = tuple(sorted(tuple(sender_availability_input_digests)))
        if any(
            not isinstance(contrast_id, str)
            or not contrast_id
            or not isinstance(input_digest, str)
            or not input_digest
            for contrast_id, input_digest in sender_digests
        ):
            raise ValueError(
                "sender availability input digests must contain non-empty IDs"
            )
        if len({contrast_id for contrast_id, _ in sender_digests}) != len(
            sender_digests
        ):
            raise ValueError("sender availability digest contrasts must be unique")
        expected_sender_digests = tuple(
            sorted(
                (
                    functional.contrast_manifest_id,
                    functional.training_availability_digest,
                )
                for functional in functionals
            )
        )
        if sender_digests != expected_sender_digests:
            raise ValueError(
                "sender availability input digests do not match their functionals"
            )
        if functionals:
            if any(
                not set(functional.training_subject_ids).issubset(subjects)
                or functional.training_input_digest != training_input_digest
                or functional.filter_universe_id
                != frozen_interaction_universe.filter_universe_id
                or functional.frozen_interaction_ids
                != frozen_interaction_universe.interaction_ids
                or functional.parameters.parameter_manifest_id
                != spec.sender_parameters.parameter_manifest_id
                for functional in functionals
            ):
                raise ValueError(
                    "sender functional provenance does not match training artifacts"
                )
            contrast_ids = {item.contrast_manifest_id for item in functionals}
            if len(contrast_ids) != len(functionals):
                raise ValueError("sender functionals must bind unique contrasts")
            completed: tuple[str, ...] = (
                "availability_filter",
                "common_sender_functional",
            )
        else:
            completed = ("availability_filter",)
        remaining: tuple[str, ...] = (
            "receptor_gate",
            "response_precision",
            "family_basis",
            "attribution_tuning",
            "incremental_downstream",
            "common_scoring_functional",
        )
        if not functionals:
            remaining = (*remaining[:-1], "common_sender_functional", remaining[-1])
        resource_bundle_content_id = _resource_bundle_content_id(resource_bundle)
        target_prior_content_id = _target_prior_content_id(target_prior)
        payload = {
            "cell_type_ids": list(cell_types),
            "completed_stages": list(completed),
            "config_digest": config.digest,
            "filter_universe_id": (frozen_interaction_universe.filter_universe_id),
            "remaining_stages": list(remaining),
            "resource_bundle_content_id": resource_bundle_content_id,
            "resource_manifest_digest": resource_bundle.manifest_digest,
            "sender_functional_ids": [
                functional.sender_functional_id for functional in functionals
            ],
            "sender_availability_input_digests": [
                list(value) for value in sender_digests
            ],
            "spec_id": spec.spec_id,
            "target_prior_manifest_digest": target_prior.manifest_digest,
            "target_prior_content_id": target_prior_content_id,
            "training_input_digest": training_input_digest,
            "training_sample_ids": list(samples),
            "training_subject_ids": list(subjects),
            "certification_status": _PARTIAL_STATUS,
        }
        artifact_id = stable_id("partial_fold_training_artifact", payload)
        self = object.__new__(cls)
        values: dict[str, Any] = {
            "config": config,
            "resource_bundle": resource_bundle,
            "target_prior": target_prior,
            "spec": spec,
            "training_subject_ids": subjects,
            "training_sample_ids": samples,
            "cell_type_ids": cell_types,
            "training_input_digest": training_input_digest,
            "resource_bundle_content_id": resource_bundle_content_id,
            "target_prior_content_id": target_prior_content_id,
            "frozen_interaction_universe": frozen_interaction_universe,
            "sender_functionals": functionals,
            "sender_availability_input_digests": sender_digests,
            "completed_stages": completed,
            "remaining_stages": remaining,
            "certification_status": _PARTIAL_STATUS,
            "training_artifact_id": artifact_id,
            "_producer_marker": _PRODUCER_MARKER,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        return self

    @property
    def is_oof_certified(self) -> bool:
        """Return false until the complete train/apply pipeline exists."""

        return False

    def _require_producer_owned(self) -> None:
        if self._producer_marker != _PRODUCER_MARKER:
            raise TypeError("TrainingArtifacts were not produced by this workflow")
        self._require_intact()

    def _require_intact(self) -> None:
        """Reject forced mutation of training scope, resources, or child artifacts."""

        try:
            repeated = TrainingArtifacts._from_training(
                config=self.config,
                resource_bundle=self.resource_bundle,
                target_prior=self.target_prior,
                spec=self.spec,
                training_subject_ids=self.training_subject_ids,
                training_sample_ids=self.training_sample_ids,
                cell_type_ids=self.cell_type_ids,
                training_input_digest=self.training_input_digest,
                frozen_interaction_universe=self.frozen_interaction_universe,
                sender_functionals=self.sender_functionals,
                sender_availability_input_digests=(
                    self.sender_availability_input_digests
                ),
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and isinstance(self.training_subject_ids, tuple)
                and isinstance(self.training_sample_ids, tuple)
                and isinstance(self.cell_type_ids, tuple)
                and isinstance(self.sender_functionals, tuple)
                and isinstance(self.sender_availability_input_digests, tuple)
                and isinstance(self.completed_stages, tuple)
                and isinstance(self.remaining_stages, tuple)
                and self.training_subject_ids == repeated.training_subject_ids
                and self.training_sample_ids == repeated.training_sample_ids
                and self.cell_type_ids == repeated.cell_type_ids
                and self.training_input_digest == repeated.training_input_digest
                and self.resource_bundle_content_id
                == repeated.resource_bundle_content_id
                and self.target_prior_content_id == repeated.target_prior_content_id
                and self.sender_functionals == repeated.sender_functionals
                and self.sender_availability_input_digests
                == repeated.sender_availability_input_digests
                and self.completed_stages == repeated.completed_stages
                and self.remaining_stages == repeated.remaining_stages
                and self.certification_status == repeated.certification_status
                and self.training_artifact_id == repeated.training_artifact_id
                and not self.is_oof_certified
            )
        except (
            AttributeError,
            ContractError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Training artifact failed scope or child integrity validation",
                code="training_artifact_integrity_violation",
                field="training_artifact_id",
                remediation="Refit the training artifact from intact raw inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Training artifact failed scope or child integrity validation",
                code="training_artifact_integrity_violation",
                field="training_artifact_id",
                remediation="Refit the training artifact from intact raw inputs",
            )


@dataclass(frozen=True, slots=True)
class _FoldTrainingResult:
    """Fold-local training artifacts plus their already fitted availability."""

    artifacts: TrainingArtifacts
    training_availability: BatchAvailability

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, TrainingArtifacts):
            raise TypeError("artifacts must be TrainingArtifacts")
        if not isinstance(self.training_availability, BatchAvailability):
            raise TypeError("training_availability must be BatchAvailability")
        if (
            self.training_availability.application_subject_ids
            != self.artifacts.training_subject_ids
            or self.training_availability.frozen_interaction_universe.to_dict()
            != self.artifacts.frozen_interaction_universe.to_dict()
        ):
            raise ValueError(
                "fold-local availability does not match its training artifacts"
            )


Aggregate: TypeAlias = PseudobulkDataset | ExploratoryAggregate


@dataclass(frozen=True, slots=True)
class _PreparedRawFold:
    schema: InputSchema
    aggregate: Aggregate
    sample_metadata: pd.DataFrame
    subject_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    cell_type_ids: tuple[str, ...]
    input_digest: str
    config_digest: str
    min_cells: int
    requested_cell_type_ids: tuple[str, ...] | None
    excluded_cell_type_ids: tuple[str, ...]
    root_input_identity_id: str | None

    def require_compatible(
        self,
        config: CrychicConfig,
        *,
        min_cells: int,
        cell_types: tuple[str, ...] | None,
    ) -> None:
        requested = None if cell_types is None else tuple(sorted(cell_types))
        if (
            self.config_digest != config.digest
            or self.min_cells != min_cells
            or self.requested_cell_type_ids != requested
        ):
            raise ContractError(
                "Prepared raw fold does not match the requested execution policy",
                code="prepared_raw_fold_policy_mismatch",
                field="prepared_fold",
                remediation="Reprepare the physical fold with the exact policy",
            )


def _input_schema(config: CrychicConfig) -> InputSchema:
    return InputSchema(
        context_keys=tuple(config.context_keys),
        counts_layer=config.counts_layer,
        sample_key=config.sample_key,
        subject_key=config.subject_key,
        cell_type_key=config.cell_type_key,
        covariates=tuple(config.covariates),
        expression_layer=config.expression_layer,
        expression_source=config.expression_source,
        expression_transform=cast(
            ExpressionTransform | None, config.expression_transform
        ),
        normalized_zero_is_nondetection=config.normalized_zero_is_nondetection,
        species=config.species,
        gene_namespace=config.gene_namespace,
        allow_duplicate_genes=config.allow_duplicate_genes,
    )


def _copy_matrix(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        source = sparse.csr_matrix(matrix)
        if source.has_canonical_format and source.has_sorted_indices:
            return source.copy()
        result = sparse.csr_matrix(matrix, dtype=np.float64).copy()
        result.sum_duplicates()
        result.sort_indices()
        return result
    return np.asarray(matrix).copy()


def _auxiliary_sample_columns(
    validated: ValidatedInput,
    values: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("auxiliary_sample_columns must be a sequence, not a string")
    columns = tuple(sorted(values))
    if len(columns) != len(set(columns)) or any(
        not isinstance(column, str)
        or not column
        or column != column.strip()
        for column in columns
    ):
        raise ValueError(
            "auxiliary_sample_columns must contain unique canonical names"
        )
    schema = validated.schema
    declared = {
        schema.sample_key,
        schema.subject_key,
        schema.cell_type_key,
        *schema.context_keys,
        *schema.covariates,
    }
    overlap = declared.intersection(columns)
    if overlap:
        raise ValueError(
            "auxiliary_sample_columns duplicate declared input fields: "
            f"{sorted(overlap)}"
        )
    missing = set(columns).difference(validated.adata.obs.columns)
    if missing:
        raise ValueError(
            f"auxiliary sample metadata is missing fields: {sorted(missing)}"
        )
    grouped = validated.adata.obs.groupby(
        schema.sample_key,
        observed=True,
        sort=False,
    )
    for column in columns:
        values_for_column = validated.adata.obs[column]
        if values_for_column.isna().any():
            raise ValueError(
                f"auxiliary sample metadata {column!r} contains missing values"
            )
        if grouped[column].nunique(dropna=False).ne(1).any():
            raise ValueError(
                f"auxiliary sample metadata {column!r} varies within sample"
            )
        for value in values_for_column:
            canonical_json(_plain_metadata_value(value))
    return columns


def _sanitize_validated_input(
    validated: ValidatedInput,
    *,
    auxiliary_sample_columns: Sequence[str] = (),
) -> AnnData:
    """Copy only declared input fields and the selected expression matrix."""

    schema = validated.schema
    auxiliary = _auxiliary_sample_columns(validated, auxiliary_sample_columns)
    required_obs = [
        schema.sample_key,
        schema.subject_key,
        schema.cell_type_key,
        *schema.context_keys,
        *schema.covariates,
        *auxiliary,
    ]
    obs = validated.adata.obs.loc[:, required_obs].copy(deep=True)
    var = pd.DataFrame(index=validated.adata.var_names.copy())
    n_obs = len(obs)
    n_vars = len(var)
    selected = _copy_matrix(validated.matrix)
    empty = sparse.csr_matrix((n_obs, n_vars), dtype=np.float64)
    if validated.is_counts:
        if schema.counts_layer is None:  # pragma: no cover - validated contract
            raise RuntimeError("count input is missing its declared layer")
        sanitized = AnnData(X=empty, obs=obs, var=var)
        sanitized.layers[schema.counts_layer] = selected
    elif schema.expression_layer is None:
        sanitized = AnnData(X=selected, obs=obs, var=var)
    else:
        sanitized = AnnData(X=empty, obs=obs, var=var)
        sanitized.layers[schema.expression_layer] = selected
    return sanitized


def _plain_metadata_value(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _update_canonical_digest(digest: Any, value: object) -> None:
    encoded = canonical_json(value).encode("ascii")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _feature_id_digest(feature_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for position, feature_id in enumerate(feature_ids):
        _update_canonical_digest(
            digest,
            {"feature_id": feature_id, "position": position},
        )
    return digest.hexdigest()


def _row_expression_digests(matrix: Any) -> tuple[str, ...]:
    result: list[str] = []
    if sparse.issparse(matrix):
        canonical = sparse.csr_matrix(matrix)
        if not canonical.has_canonical_format or not canonical.has_sorted_indices:
            canonical = sparse.csr_matrix(matrix, dtype=np.float64).copy()
            canonical.sum_duplicates()
            canonical.sort_indices()
        for row_index in range(canonical.shape[0]):
            start = int(canonical.indptr[row_index])
            stop = int(canonical.indptr[row_index + 1])
            digest = hashlib.sha256()
            digest.update(np.asarray([canonical.shape[1]], dtype="<i8").tobytes())
            digest.update(
                np.asarray(canonical.indices[start:stop], dtype="<i8").tobytes()
            )
            digest.update(
                np.asarray(canonical.data[start:stop], dtype="<f8").tobytes()
            )
            result.append(digest.hexdigest())
    else:
        canonical = np.asarray(matrix, dtype="<f8", order="C")
        for row in canonical:
            digest = hashlib.sha256()
            digest.update(np.asarray([canonical.shape[1]], dtype="<i8").tobytes())
            digest.update(np.asarray(row, dtype="<f8").tobytes())
            result.append(digest.hexdigest())
    return tuple(result)


def _matrix_content_digest(matrix: Any) -> str:
    digest = hashlib.sha256()
    shape = tuple(int(value) for value in matrix.shape)
    digest.update(np.asarray(shape, dtype="<i8").tobytes())
    if sparse.issparse(matrix):
        canonical = sparse.csr_matrix(matrix)
        if not canonical.has_canonical_format or not canonical.has_sorted_indices:
            canonical = sparse.csr_matrix(matrix, dtype=np.float64).copy()
            canonical.sum_duplicates()
            canonical.sort_indices()
        digest.update(np.asarray(canonical.indptr, dtype="<i8").tobytes())
        digest.update(np.asarray(canonical.indices, dtype="<i8").tobytes())
        digest.update(np.asarray(canonical.data, dtype="<f8").tobytes())
    else:
        canonical = np.asarray(matrix, dtype="<f8", order="C")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _subject_content_digests(
    validated: ValidatedInput,
    config: CrychicConfig,
) -> tuple[tuple[str, str], ...]:
    schema = validated.schema
    columns = (
        schema.sample_key,
        schema.subject_key,
        schema.cell_type_key,
        *schema.context_keys,
        *schema.covariates,
    )
    table = validated.adata.obs.loc[:, list(columns)]
    expression_digests = _row_expression_digests(validated.matrix)
    if len(expression_digests) != len(table):
        raise RuntimeError("expression and observation rows do not align")
    rows_by_subject: dict[str, list[str]] = {}
    samples_by_subject: dict[str, set[str]] = {}
    for position, (obs_name, row) in enumerate(table.iterrows()):
        subject_id = str(row[schema.subject_key])
        sample_id = str(row[schema.sample_key])
        row_digest = hashlib.sha256()
        _update_canonical_digest(
            row_digest,
            {
                "expression_digest": expression_digests[position],
                "obs_name": str(obs_name),
                "values": [
                    [column, _plain_metadata_value(row[column])] for column in columns
                ],
            },
        )
        rows_by_subject.setdefault(subject_id, []).append(row_digest.hexdigest())
        samples_by_subject.setdefault(subject_id, set()).add(sample_id)
    feature_digest = _feature_id_digest(validated.feature_ids)
    result: list[tuple[str, str]] = []
    for subject_id in sorted(rows_by_subject):
        subject_digest: str = stable_id(
            "sanitized_raw_subject_content",
            {
                "config_digest": config.digest,
                "feature_id_digest": feature_digest,
                "row_digests": sorted(rows_by_subject[subject_id]),
                "sample_ids": sorted(samples_by_subject[subject_id]),
                "subject_id": subject_id,
            },
            schema_version="1",
        )
        result.append((subject_id, subject_digest))
    if not result:
        raise ValueError("sanitized raw input must contain at least one subject")
    return tuple(result)


def _scope_input_digest(
    config_digest: str,
    subject_content_digests: tuple[tuple[str, str], ...],
) -> str:
    result: str = stable_id(
        "sanitized_raw_fold_input",
        {
            "config_digest": config_digest,
            "subject_content_digests": [
                [subject_id, digest]
                for subject_id, digest in subject_content_digests
            ],
        },
        schema_version="2",
    )
    return result


def _validated_raw_input_digest(
    validated: ValidatedInput,
    config: CrychicConfig,
) -> str:
    return _scope_input_digest(
        config.digest,
        _subject_content_digests(validated, config),
    )


@dataclass(frozen=True, slots=True, init=False)
class SanitizedRawInputIdentity:
    """Producer-owned identity of the complete sanitized raw-count input."""

    config_digest: str
    input_digest: str
    subject_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    subject_content_digests: tuple[tuple[str, str], ...]
    identity_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "SanitizedRawInputIdentity is producer-owned; use the raw-input "
            "workflow"
        )

    @validation_scope()
    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            payload = {
                "config_digest": self.config_digest,
                "input_digest": self.input_digest,
                "sample_ids": list(self.sample_ids),
                "subject_content_digests": [
                    list(value) for value in self.subject_content_digests
                ],
                "subject_ids": list(self.subject_ids),
            }
            valid = (
                self._producer_marker == _ROOT_INPUT_PRODUCER
                and isinstance(self.config_digest, str)
                and bool(self.config_digest)
                and isinstance(self.input_digest, str)
                and bool(self.input_digest)
                and isinstance(self.subject_ids, tuple)
                and bool(self.subject_ids)
                and len(self.subject_ids) == len(set(self.subject_ids))
                and isinstance(self.sample_ids, tuple)
                and bool(self.sample_ids)
                and len(self.sample_ids) == len(set(self.sample_ids))
                and isinstance(self.subject_content_digests, tuple)
                and tuple(
                    subject_id for subject_id, _ in self.subject_content_digests
                )
                == self.subject_ids
                and self.input_digest
                == _scope_input_digest(
                    self.config_digest,
                    self.subject_content_digests,
                )
                and self.identity_id
                == stable_id(
                    "sanitized_raw_input_identity",
                    payload,
                    schema_version="1",
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Sanitized raw-input identity failed integrity validation",
                code="root_input_identity_integrity_violation",
                field="identity_id",
                remediation="Rebuild the identity from the complete raw input",
            ) from error
        if not valid:
            raise ContractError(
                "Sanitized raw-input identity failed integrity validation",
                code="root_input_identity_integrity_violation",
                field="identity_id",
                remediation="Rebuild the identity from the complete raw input",
            )
        record_validation(self)

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {
            "identity_id": self.identity_id,
            "config_digest": self.config_digest,
            "input_digest": self.input_digest,
            "subject_ids": list(self.subject_ids),
            "sample_ids": list(self.sample_ids),
            "subject_content_digests": [
                {"subject_id": subject_id, "digest": digest}
                for subject_id, digest in self.subject_content_digests
            ],
        }

    def scope_digest(self, subject_ids: tuple[str, ...]) -> str:
        """Derive one fold scope digest from the root subject manifest."""

        self._require_intact()
        requested = tuple(sorted(subject_ids))
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("scope subject_ids must be non-empty and unique")
        by_subject = dict(self.subject_content_digests)
        unknown = set(requested).difference(by_subject)
        if unknown:
            raise ValueError(f"scope contains unknown subjects: {sorted(unknown)}")
        return _scope_input_digest(
            self.config_digest,
            tuple((subject_id, by_subject[subject_id]) for subject_id in requested),
        )


def _validated_raw_input_identity(
    validated: ValidatedInput,
    config: CrychicConfig,
) -> SanitizedRawInputIdentity:
    metadata = validated.report.sample_metadata
    subjects = tuple(sorted(metadata[config.subject_key].astype(str).unique()))
    samples = tuple(sorted(metadata[config.sample_key].astype(str).unique()))
    subject_content_digests = _subject_content_digests(validated, config)
    input_digest = _scope_input_digest(config.digest, subject_content_digests)
    payload = {
        "config_digest": config.digest,
        "input_digest": input_digest,
        "sample_ids": list(samples),
        "subject_content_digests": [list(value) for value in subject_content_digests],
        "subject_ids": list(subjects),
    }
    self = object.__new__(SanitizedRawInputIdentity)
    object.__setattr__(self, "config_digest", config.digest)
    object.__setattr__(self, "input_digest", input_digest)
    object.__setattr__(self, "subject_ids", subjects)
    object.__setattr__(self, "sample_ids", samples)
    object.__setattr__(self, "subject_content_digests", subject_content_digests)
    object.__setattr__(
        self,
        "identity_id",
        stable_id("sanitized_raw_input_identity", payload, schema_version="1"),
    )
    object.__setattr__(self, "_producer_marker", _ROOT_INPUT_PRODUCER)
    self._require_intact()
    return self


@dataclass(frozen=True, slots=True, init=False)
class SanitizedRawInputSnapshot:
    """Producer-owned defensive raw-count snapshot plus its subject manifest."""

    adata: AnnData = field(repr=False)
    identity: SanitizedRawInputIdentity
    subject_key: str
    sample_key: str
    counts_layer: str
    metadata_digest: str
    expression_digest: str
    expression_shape: tuple[int, int]
    snapshot_id: str
    _expression_matrix: Any = field(repr=False)
    _producer_marker: str = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "SanitizedRawInputSnapshot is producer-owned; use the raw-input "
            "workflow"
        )

    @validation_scope()
    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            self.identity._require_intact()
            subjects = tuple(
                sorted(self.adata.obs[self.subject_key].astype(str).unique())
            )
            samples = tuple(
                sorted(self.adata.obs[self.sample_key].astype(str).unique())
            )
            selected = self.adata.layers[self.counts_layer]
            snapshot_payload = {
                "counts_layer": self.counts_layer,
                "expression_digest": self.expression_digest,
                "expression_shape": list(self.expression_shape),
                "identity_id": self.identity.identity_id,
                "metadata_digest": self.metadata_digest,
                "sample_key": self.sample_key,
                "subject_key": self.subject_key,
            }
            valid = (
                self._producer_marker == _ROOT_SNAPSHOT_PRODUCER
                and isinstance(self.adata, AnnData)
                and not self.adata.uns
                and not self.adata.obsm
                and isinstance(self.subject_key, str)
                and bool(self.subject_key)
                and isinstance(self.sample_key, str)
                and bool(self.sample_key)
                and isinstance(self.counts_layer, str)
                and bool(self.counts_layer)
                and selected is self._expression_matrix
                and tuple(int(value) for value in selected.shape)
                == self.expression_shape
                and _matrix_is_read_only(selected)
                and _matrix_content_digest(selected) == self.expression_digest
                and _snapshot_metadata_digest(self.adata) == self.metadata_digest
                and subjects == self.identity.subject_ids
                and samples == self.identity.sample_ids
                and self.snapshot_id
                == stable_id(
                    "sanitized_raw_input_snapshot",
                    snapshot_payload,
                    schema_version="1",
                )
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Sanitized raw-input snapshot failed integrity validation",
                code="root_input_snapshot_integrity_violation",
                field="snapshot",
                remediation="Rebuild the defensive snapshot from raw input",
            ) from error
        if not valid:
            raise ContractError(
                "Sanitized raw-input snapshot failed integrity validation",
                code="root_input_snapshot_integrity_violation",
                field="snapshot",
                remediation="Rebuild the defensive snapshot from raw input",
            )
        record_validation(self)


def _snapshot_metadata_digest(adata: AnnData) -> str:
    digest = hashlib.sha256()
    _update_canonical_digest(
        digest,
        {
            "obs_columns": [str(column) for column in adata.obs.columns],
            "obs_dtypes": [str(dtype) for dtype in adata.obs.dtypes],
            "var_index_name": str(adata.var_names.name),
        },
    )
    obs_hash = pd.util.hash_pandas_object(
        adata.obs,
        index=True,
        categorize=True,
    ).to_numpy(dtype="<u8")
    var_hash = pd.util.hash_pandas_object(
        adata.var_names,
        index=True,
        categorize=True,
    ).to_numpy(dtype="<u8")
    digest.update(obs_hash.tobytes())
    digest.update(var_hash.tobytes())
    return digest.hexdigest()


def _freeze_matrix(matrix: Any) -> None:
    if sparse.issparse(matrix):
        matrix.data.setflags(write=False)
        matrix.indices.setflags(write=False)
        matrix.indptr.setflags(write=False)
    else:
        np.asarray(matrix).setflags(write=False)


def _matrix_is_read_only(matrix: Any) -> bool:
    if sparse.issparse(matrix):
        return not (
            matrix.data.flags.writeable
            or matrix.indices.flags.writeable
            or matrix.indptr.flags.writeable
        )
    return not np.asarray(matrix).flags.writeable


def _sanitized_raw_input_snapshot(
    adata: AnnData,
    config: CrychicConfig,
    *,
    auxiliary_sample_columns: Sequence[str] = (),
) -> SanitizedRawInputSnapshot:
    """Copy declared raw input and bind its producer-owned subject manifest."""

    caller_view = validate_anndata(adata, _input_schema(config))
    if not caller_view.is_counts:
        raise ValueError(
            "subject cross-fitting requires raw counts; normalized-only input "
            "cannot certify train-only preprocessing"
        )
    sanitized = _sanitize_validated_input(
        caller_view,
        auxiliary_sample_columns=auxiliary_sample_columns,
    )
    validated = validate_anndata(sanitized, _input_schema(config))
    if validated.report.duplicate_genes:
        raise ValueError("fold training does not support duplicate gene identifiers")
    if config.counts_layer is None:  # pragma: no cover - validated counts contract
        raise RuntimeError("cross-fit snapshot requires a declared counts layer")
    selected = validated.matrix
    _freeze_matrix(selected)
    self = object.__new__(SanitizedRawInputSnapshot)
    object.__setattr__(self, "adata", sanitized)
    object.__setattr__(
        self,
        "identity",
        _validated_raw_input_identity(validated, config),
    )
    object.__setattr__(self, "subject_key", config.subject_key)
    object.__setattr__(self, "sample_key", config.sample_key)
    object.__setattr__(self, "counts_layer", config.counts_layer)
    object.__setattr__(self, "metadata_digest", _snapshot_metadata_digest(sanitized))
    object.__setattr__(self, "expression_digest", _matrix_content_digest(selected))
    object.__setattr__(
        self,
        "expression_shape",
        tuple(int(value) for value in selected.shape),
    )
    object.__setattr__(self, "_expression_matrix", selected)
    snapshot_payload = {
        "counts_layer": self.counts_layer,
        "expression_digest": self.expression_digest,
        "expression_shape": list(self.expression_shape),
        "identity_id": self.identity.identity_id,
        "metadata_digest": self.metadata_digest,
        "sample_key": self.sample_key,
        "subject_key": self.subject_key,
    }
    object.__setattr__(
        self,
        "snapshot_id",
        stable_id(
            "sanitized_raw_input_snapshot",
            snapshot_payload,
            schema_version="1",
        ),
    )
    object.__setattr__(self, "_producer_marker", _ROOT_SNAPSHOT_PRODUCER)
    record_validation(self)
    return self


def _sanitized_raw_input_identity(
    adata: AnnData,
    config: CrychicConfig,
) -> SanitizedRawInputIdentity:
    """Build a typed identity without fitting any learned stage."""

    return _sanitized_raw_input_snapshot(adata, config).identity


def _sanitized_raw_input_digest(adata: AnnData, config: CrychicConfig) -> str:
    """Digest the exact sanitized raw input without fitting any learned stage."""

    return _sanitized_raw_input_identity(adata, config).input_digest


def _empty_scope_pseudobulk(
    validated: ValidatedInput,
    *,
    cell_types: tuple[str, ...],
) -> PseudobulkDataset:
    schema = validated.schema
    rows: list[dict[str, object]] = []
    for _, sample_row in validated.report.sample_metadata.iterrows():
        sample_id = sample_row[schema.sample_key]
        subject_id = sample_row[schema.subject_key]
        context = tuple(
            (key, sample_row[key]) for key in schema.context_keys
        )
        for cell_type in cell_types:
            row: dict[str, object] = {
                "unit_id": _unit_id(sample_id, context, cell_type),
                "sample_id": sample_id,
                "subject_id": subject_id,
                "cell_type": cell_type,
                "context": context,
                "matrix_row": pd.NA,
                "n_cells": 0,
                "cell_proportion": np.nan,
                "state_eligible": False,
                "abundance_eligible": False,
                "missingness_reason": MissingnessReason.SAMPLING_ZERO.value,
                "library_size": pd.NA,
                "median_umi": pd.NA,
            }
            for context_key, context_value in context:
                if context_key != "context":
                    row[context_key] = context_value
            rows.append(row)
    metadata = pd.DataFrame(rows).sort_values(
        "unit_id", kind="stable", ignore_index=True
    )
    n_features = len(validated.feature_ids)
    return PseudobulkDataset(
        counts=sparse.csr_matrix((0, n_features), dtype=np.int64),
        detection_fraction=sparse.csr_matrix((0, n_features), dtype=float),
        unit_metadata=metadata,
        feature_ids=validated.feature_ids,
        matrix_unit_ids=(),
        source_location=validated.report.expression_location,
    )


def _prepare_raw_fold(
    adata: AnnData,
    config: CrychicConfig,
    *,
    min_cells: int,
    cell_types: tuple[str, ...] | None = None,
    root_input_identity: SanitizedRawInputIdentity | None = None,
) -> _PreparedRawFold:
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData instance")
    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig instance")
    schema = _input_schema(config)
    caller_view = validate_anndata(adata, schema)
    scope_metadata = caller_view.report.sample_metadata
    scope_subjects = tuple(
        sorted(scope_metadata[config.subject_key].astype(str).unique())
    )
    scope_samples = tuple(
        sorted(scope_metadata[config.sample_key].astype(str).unique())
    )
    sanitized_adata = _sanitize_validated_input(caller_view)
    excluded_cell_types: tuple[str, ...] = ()
    all_requested_types_absent = False
    if cell_types is not None:
        requested_cell_types = tuple(sorted(cell_types))
        observed_input_cell_types = set(
            sanitized_adata.obs[config.cell_type_key].astype(str).unique()
        )
        excluded_cell_types = tuple(
            sorted(observed_input_cell_types.difference(requested_cell_types))
        )
        if excluded_cell_types:
            retained = sanitized_adata.obs[config.cell_type_key].astype(str).isin(
                requested_cell_types
            )
            if not retained.any():
                all_requested_types_absent = True
            else:
                sanitized_adata = sanitized_adata[retained].copy()
    validated = validate_anndata(sanitized_adata, schema)
    if validated.report.duplicate_genes:
        raise ValueError("fold training does not support duplicate gene identifiers")
    aggregate = (
        _empty_scope_pseudobulk(
            validated,
            cell_types=tuple(sorted(cell_types or ())),
        )
        if all_requested_types_absent
        else aggregate_pseudobulk(
            validated,
            min_cells=min_cells,
            cell_types=cell_types,
        )
    )
    subjects = scope_subjects
    samples = scope_samples
    observed_cell_types = tuple(
        sorted(aggregate.unit_metadata["cell_type"].astype(str).unique())
    )
    if root_input_identity is None:
        input_digest = _validated_raw_input_digest(
            caller_view if excluded_cell_types else validated,
            config,
        )
    else:
        root_input_identity._require_intact()
        if root_input_identity.config_digest != config.digest:
            raise ValueError("root input identity does not match the fold config")
        if not set(samples).issubset(root_input_identity.sample_ids):
            raise ValueError("fold samples are outside the root input identity")
        input_digest = root_input_identity.scope_digest(subjects)
    return _PreparedRawFold(
        schema=validated.schema,
        aggregate=aggregate,
        sample_metadata=validated.report.sample_metadata.copy(deep=True),
        subject_ids=subjects,
        sample_ids=samples,
        cell_type_ids=observed_cell_types,
        input_digest=input_digest,
        config_digest=config.digest,
        min_cells=min_cells,
        requested_cell_type_ids=(
            None if cell_types is None else tuple(sorted(cell_types))
        ),
        excluded_cell_type_ids=excluded_cell_types,
        root_input_identity_id=(
            None
            if root_input_identity is None
            else root_input_identity.identity_id
        ),
    )


def _prepare_validated_root_fold(
    validated: ValidatedInput,
    config: CrychicConfig,
    *,
    min_cells: int,
    root_input_identity: SanitizedRawInputIdentity,
) -> _PreparedRawFold:
    """Aggregate one immutable sanitized root before subject partitioning."""

    if not isinstance(validated, ValidatedInput) or not validated.is_counts:
        raise TypeError("validated must be a raw-count ValidatedInput")
    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig")
    if not isinstance(root_input_identity, SanitizedRawInputIdentity):
        raise TypeError("root_input_identity must be SanitizedRawInputIdentity")
    root_input_identity._require_intact()
    if root_input_identity.config_digest != config.digest:
        raise ValueError("root input identity does not match the root fold config")
    if validated.report.duplicate_genes:
        raise ValueError("fold training does not support duplicate gene identifiers")
    sample_metadata = validated.report.sample_metadata.copy(deep=True)
    subjects = tuple(
        sorted(sample_metadata[config.subject_key].astype(str).unique())
    )
    samples = tuple(sorted(sample_metadata[config.sample_key].astype(str).unique()))
    if subjects != root_input_identity.subject_ids:
        raise ValueError("validated root subjects differ from the root input identity")
    aggregate = aggregate_pseudobulk(validated, min_cells=min_cells)
    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("subject cross-fit root aggregate must be count pseudobulk")
    cell_types = tuple(
        sorted(
            aggregate.unit_metadata.loc[
                aggregate.unit_metadata["n_cells"].astype(int) > 0,
                "cell_type",
            ]
            .astype(str)
            .unique()
        )
    )
    return _PreparedRawFold(
        schema=validated.schema,
        aggregate=aggregate,
        sample_metadata=sample_metadata,
        subject_ids=subjects,
        sample_ids=samples,
        cell_type_ids=cell_types,
        input_digest=root_input_identity.scope_digest(subjects),
        config_digest=config.digest,
        min_cells=min_cells,
        requested_cell_type_ids=None,
        excluded_cell_type_ids=(),
        root_input_identity_id=root_input_identity.identity_id,
    )


def _subset_prepared_raw_fold(
    root: _PreparedRawFold,
    config: CrychicConfig,
    *,
    subject_ids: tuple[str, ...],
    min_cells: int,
    root_input_identity: SanitizedRawInputIdentity,
    cell_types: tuple[str, ...] | None = None,
) -> _PreparedRawFold:
    """Select a physical subject fold from one pre-aggregated raw-count root."""

    if not isinstance(root, _PreparedRawFold):
        raise TypeError("root must be a _PreparedRawFold")
    if not isinstance(root.aggregate, PseudobulkDataset):
        raise TypeError("root aggregate must be a PseudobulkDataset")
    root.require_compatible(config, min_cells=min_cells, cell_types=None)
    root_input_identity._require_intact()
    subjects = tuple(sorted(subject_ids))
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("subject_ids must be non-empty and unique")
    if not set(subjects).issubset(root.subject_ids):
        raise ValueError("fold subjects are outside the pre-aggregated root")
    scope_metadata = root.sample_metadata.loc[
        root.sample_metadata[config.subject_key].astype(str).isin(subjects)
    ].copy(deep=True)
    observed_subjects = tuple(
        sorted(scope_metadata[config.subject_key].astype(str).unique())
    )
    if observed_subjects != subjects:
        raise ValueError("pre-aggregated fold does not cover its subject manifest")
    samples = tuple(sorted(scope_metadata[config.sample_key].astype(str).unique()))

    root_units = root.aggregate.unit_metadata
    scope_units = root_units.loc[
        root_units["sample_id"].astype(str).isin(samples)
    ].copy(deep=True)
    observed_cell_types = {
        str(value)
        for value in scope_units.loc[
            scope_units["n_cells"].astype(int) > 0,
            "cell_type",
        ]
    }
    requested = None if cell_types is None else tuple(sorted(cell_types))
    selected_cell_types = (
        tuple(sorted(observed_cell_types)) if requested is None else requested
    )
    if not selected_cell_types:
        raise ValueError("fold contains no selected cell types")
    excluded_cell_types = (
        ()
        if requested is None
        else tuple(sorted(observed_cell_types.difference(requested)))
    )
    units = scope_units.loc[
        scope_units["cell_type"].astype(str).isin(selected_cell_types)
    ].copy(deep=True)
    sample_totals = units.groupby("sample_id", observed=True, sort=False)[
        "n_cells"
    ].transform("sum")
    has_observed_units = bool((sample_totals > 0).any())
    if has_observed_units:
        retained_samples = set(units.loc[sample_totals > 0, "sample_id"])
        units = units.loc[units["sample_id"].isin(retained_samples)].copy(deep=True)
        scope_metadata = scope_metadata.loc[
            scope_metadata[config.sample_key].isin(retained_samples)
        ].copy(deep=True)
        sample_totals = units.groupby("sample_id", observed=True, sort=False)[
            "n_cells"
        ].transform("sum")
    else:
        units["state_eligible"] = False
        units["abundance_eligible"] = False
        units["missingness_reason"] = MissingnessReason.SAMPLING_ZERO.value
    units["cell_proportion"] = units["n_cells"].astype(float).div(
        sample_totals.astype(float).replace(0.0, np.nan)
    )
    units = units.sort_values("unit_id", kind="stable", ignore_index=True)
    matrix_rows = units["matrix_row"].notna()
    root_matrix_rows = units.loc[matrix_rows, "matrix_row"].astype(int).to_numpy()
    matrix_unit_ids = tuple(units.loc[matrix_rows, "unit_id"].astype(str))
    row_by_id = {
        unit_id: row for row, unit_id in enumerate(matrix_unit_ids)
    }
    if has_observed_units:
        units["matrix_row"] = pd.array(
            units["unit_id"].map(row_by_id).tolist(),
            dtype="Int64",
        )
    else:
        units["matrix_row"] = pd.Series(
            [pd.NA] * len(units),
            index=units.index,
            dtype=object,
        )
    aggregate = PseudobulkDataset(
        counts=sparse.csr_matrix(root.aggregate.counts[root_matrix_rows]),
        detection_fraction=sparse.csr_matrix(
            root.aggregate.detection_fraction[root_matrix_rows]
        ),
        unit_metadata=units,
        feature_ids=root.aggregate.feature_ids,
        matrix_unit_ids=matrix_unit_ids,
        source_location=root.aggregate.source_location,
    )
    return _PreparedRawFold(
        schema=root.schema,
        aggregate=aggregate,
        sample_metadata=scope_metadata,
        subject_ids=subjects,
        sample_ids=samples,
        cell_type_ids=selected_cell_types,
        input_digest=root_input_identity.scope_digest(subjects),
        config_digest=config.digest,
        min_cells=min_cells,
        requested_cell_type_ids=requested,
        excluded_cell_type_ids=excluded_cell_types,
        root_input_identity_id=root_input_identity.identity_id,
    )


def _fit_interaction_universe(
    prepared: _PreparedRawFold,
    resource_bundle: ResourceBundle,
    spec: FoldTrainingSpec,
) -> BatchAvailability:
    return estimate_bundle_availability(
        prepared.aggregate,
        resource_bundle,
        context_keys=prepared.schema.context_keys,
        parameters=spec.availability_parameters,
        min_pooled_availability=spec.min_pooled_availability,
        max_interactions=spec.max_interactions,
    )


def _planned_sender_contrasts(
    sample_interactions: pd.DataFrame,
    context_keys: tuple[str, ...],
) -> tuple[ContrastSpec, ...]:
    nodes: set[Hashable] = set()
    for row in sample_interactions.itertuples(index=False):
        if len(context_keys) == 1:
            node: Hashable = getattr(row, context_keys[0])
        else:
            node = tuple((key, getattr(row, key)) for key in context_keys)
        nodes.add(node)
    ordered = tuple(sorted(nodes, key=canonical_json))
    return tuple(
        balanced_contrast(
            (left,),
            (right,),
            name=(
                "sender_pair:"
                f"{canonical_json(left)}-vs-{canonical_json(right)}"
            ),
            family="sender_pairwise",
        )
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1 :]
    )


def _fit_training_result_from_prepared(
    prepared: _PreparedRawFold,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: FoldTrainingSpec,
) -> _FoldTrainingResult:
    if not isinstance(prepared, _PreparedRawFold):
        raise TypeError("prepared must be a _PreparedRawFold")
    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig")
    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior")
    if not isinstance(spec, FoldTrainingSpec):
        raise TypeError("spec must be a FoldTrainingSpec")
    spec._require_intact()
    prepared.require_compatible(
        config,
        min_cells=spec.min_cells,
        cell_types=None,
    )
    if resource_bundle.species is not target_prior.species:
        raise ValueError("resource bundle and target prior species must match")
    if resource_bundle.gene_namespace is not target_prior.gene_namespace:
        raise ValueError("resource bundle and target prior namespace must match")

    availability = _fit_interaction_universe(prepared, resource_bundle, spec)
    sender_contrasts = spec.sender_contrasts or _planned_sender_contrasts(
        availability.sample_interactions,
        tuple(config.context_keys),
    )
    sender_candidate_manifest = (
        freeze_common_sender_candidate_manifest(
            availability.sample_interactions,
            frozen_interaction_ids=(
                availability.frozen_interaction_universe.interaction_ids
            ),
        )
        if sender_contrasts
        else ()
    )
    sender_functionals = tuple(
        fit_contrast_common_sender_functional(
            availability.sample_interactions,
            contrast=contrast,
            context_keys=tuple(config.context_keys),
            frozen_interaction_universe=availability.frozen_interaction_universe,
            frozen_candidate_sender_manifest=sender_candidate_manifest,
            training_input_digest=prepared.input_digest,
            parameters=spec.sender_parameters,
        )
        for contrast in sender_contrasts
    )
    sender_availability_input_digests = tuple(
        sorted(
            (
                functional.contrast_manifest_id,
                functional.training_availability_digest,
            )
            for functional in sender_functionals
        )
    )
    artifacts = TrainingArtifacts._from_training(
        config=config,
        resource_bundle=resource_bundle,
        target_prior=target_prior,
        spec=spec,
        training_subject_ids=prepared.subject_ids,
        training_sample_ids=prepared.sample_ids,
        cell_type_ids=prepared.cell_type_ids,
        training_input_digest=prepared.input_digest,
        frozen_interaction_universe=availability.frozen_interaction_universe,
        sender_functionals=sender_functionals,
        sender_availability_input_digests=sender_availability_input_digests,
    )
    return _FoldTrainingResult(
        artifacts=artifacts,
        training_availability=availability,
    )


def _fit_training_artifacts_from_prepared(
    prepared: _PreparedRawFold,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: FoldTrainingSpec,
) -> TrainingArtifacts:
    """Compatibility wrapper returning only the persisted training artifacts."""

    return _fit_training_result_from_prepared(
        prepared,
        config,
        resource_bundle,
        target_prior,
        spec=spec,
    ).artifacts


def fit_training_artifacts(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: FoldTrainingSpec,
) -> TrainingArtifacts:
    """Fit the implemented stage from a raw training-fold AnnData only.

    The function derives biological subject provenance from ``adata.obs``.  It
    deliberately accepts no fold manifest, subject list, precomputed matrix,
    basis, functional, or provenance identifier.
    """

    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig")
    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior")
    if not isinstance(spec, FoldTrainingSpec):
        raise TypeError("spec must be a FoldTrainingSpec")
    spec._require_intact()
    prepared = _prepare_raw_fold(adata, config, min_cells=spec.min_cells)
    return _fit_training_artifacts_from_prepared(
        prepared,
        config,
        resource_bundle,
        target_prior,
        spec=spec,
    )


__all__ = ["FoldTrainingSpec", "TrainingArtifacts", "fit_training_artifacts"]
