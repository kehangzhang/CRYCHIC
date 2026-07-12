"""Producer-owned training boundary for future subject-level cross-fitting.

This module intentionally implements only the first real fold-trained stage:
the interaction availability universe.  It does not construct an OOF scoring
functional or claim that the remaining response/attribution stages are
cross-fitted.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
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
from crychic.core import CrychicConfig, canonical_json, stable_id
from crychic.data import (
    ExpressionTransform,
    InputSchema,
    ValidatedInput,
    validate_anndata,
)
from crychic.pseudobulk import (
    ExploratoryAggregate,
    PseudobulkDataset,
    aggregate_pseudobulk,
)
from crychic.resources import ResourceBundle, TargetPrior

_PRODUCER_MARKER = "crychic.workflow.training.v1"
_PARTIAL_STATUS = "training_only_partial_not_oof"


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


@dataclass(frozen=True, slots=True, kw_only=True)
class FoldTrainingSpec:
    """Pre-registered settings for the implemented fold-training stage."""

    min_cells: int = 10
    min_pooled_availability: float = 0.01
    max_interactions: int | None = None
    availability_parameters: AvailabilityParameters = field(
        default_factory=AvailabilityParameters
    )
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
        }
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
    frozen_interaction_universe: FrozenInteractionUniverse
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
    ) -> TrainingArtifacts:
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
        completed = ("availability_filter",)
        remaining = (
            "receptor_gate",
            "response_precision",
            "family_basis",
            "attribution_tuning",
            "incremental_downstream",
            "common_sender_functional",
            "common_scoring_functional",
        )
        payload = {
            "cell_type_ids": list(cell_types),
            "completed_stages": list(completed),
            "config_digest": config.digest,
            "filter_universe_id": (frozen_interaction_universe.filter_universe_id),
            "resource_manifest_digest": resource_bundle.manifest_digest,
            "spec_id": spec.spec_id,
            "target_prior_manifest_digest": target_prior.manifest_digest,
            "training_input_digest": training_input_digest,
            "training_sample_ids": list(samples),
            "training_subject_ids": list(subjects),
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
            "frozen_interaction_universe": frozen_interaction_universe,
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


Aggregate: TypeAlias = PseudobulkDataset | ExploratoryAggregate


@dataclass(frozen=True, slots=True)
class _PreparedRawFold:
    validated: ValidatedInput
    aggregate: Aggregate
    subject_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    cell_type_ids: tuple[str, ...]
    input_digest: str


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
        return matrix.copy()
    return np.asarray(matrix).copy()


def _sanitize_validated_input(validated: ValidatedInput) -> AnnData:
    """Copy only declared input fields and the selected expression matrix."""

    schema = validated.schema
    required_obs = [
        schema.sample_key,
        schema.subject_key,
        schema.cell_type_key,
        *schema.context_keys,
        *schema.covariates,
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


def _matrix_digest(matrix: Any) -> str:
    digest = hashlib.sha256()
    shape = tuple(int(value) for value in matrix.shape)
    digest.update(np.asarray(shape, dtype="<i8").tobytes())
    if sparse.issparse(matrix):
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


def _plain_metadata_value(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _update_canonical_digest(digest: Any, value: object) -> None:
    encoded = canonical_json(value).encode("ascii")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _declared_obs_digest(validated: ValidatedInput) -> str:
    schema = validated.schema
    columns = (
        schema.sample_key,
        schema.subject_key,
        schema.cell_type_key,
        *schema.context_keys,
        *schema.covariates,
    )
    digest = hashlib.sha256()
    _update_canonical_digest(digest, {"columns": list(columns)})
    table = validated.adata.obs.loc[:, list(columns)]
    for position, (obs_name, row) in enumerate(table.iterrows()):
        _update_canonical_digest(
            digest,
            {
                "obs_name": str(obs_name),
                "position": position,
                "values": [
                    [column, _plain_metadata_value(row[column])] for column in columns
                ],
            },
        )
    return digest.hexdigest()


def _feature_id_digest(feature_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for position, feature_id in enumerate(feature_ids):
        _update_canonical_digest(
            digest,
            {"feature_id": feature_id, "position": position},
        )
    return digest.hexdigest()


def _prepare_raw_fold(
    adata: AnnData,
    config: CrychicConfig,
    *,
    min_cells: int,
    cell_types: tuple[str, ...] | None = None,
) -> _PreparedRawFold:
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData instance")
    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig instance")
    schema = _input_schema(config)
    caller_view = validate_anndata(adata, schema)
    sanitized_adata = _sanitize_validated_input(caller_view)
    validated = validate_anndata(sanitized_adata, schema)
    if validated.report.duplicate_genes:
        raise ValueError("fold training does not support duplicate gene identifiers")
    aggregate = aggregate_pseudobulk(
        validated,
        min_cells=min_cells,
        cell_types=cell_types,
    )
    metadata = validated.report.sample_metadata
    subjects = tuple(sorted(metadata[config.subject_key].astype(str).unique()))
    samples = tuple(sorted(metadata[config.sample_key].astype(str).unique()))
    observed_cell_types = tuple(
        sorted(aggregate.unit_metadata["cell_type"].astype(str).unique())
    )
    input_digest = stable_id(
        "sanitized_raw_fold_input",
        {
            "config_digest": config.digest,
            "declared_obs_digest": _declared_obs_digest(validated),
            "expression_matrix_digest": _matrix_digest(validated.matrix),
            "feature_id_digest": _feature_id_digest(validated.feature_ids),
            "sample_ids": list(samples),
            "subject_ids": list(subjects),
        },
    )
    return _PreparedRawFold(
        validated=validated,
        aggregate=aggregate,
        subject_ids=subjects,
        sample_ids=samples,
        cell_type_ids=observed_cell_types,
        input_digest=input_digest,
    )


def _fit_interaction_universe(
    prepared: _PreparedRawFold,
    resource_bundle: ResourceBundle,
    spec: FoldTrainingSpec,
) -> BatchAvailability:
    return estimate_bundle_availability(
        prepared.aggregate,
        resource_bundle,
        context_keys=prepared.validated.schema.context_keys,
        parameters=spec.availability_parameters,
        min_pooled_availability=spec.min_pooled_availability,
        max_interactions=spec.max_interactions,
    )


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

    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior")
    if not isinstance(spec, FoldTrainingSpec):
        raise TypeError("spec must be a FoldTrainingSpec")
    if resource_bundle.species is not target_prior.species:
        raise ValueError("resource bundle and target prior species must match")
    if resource_bundle.gene_namespace is not target_prior.gene_namespace:
        raise ValueError("resource bundle and target prior namespace must match")

    prepared = _prepare_raw_fold(adata, config, min_cells=spec.min_cells)
    availability = _fit_interaction_universe(prepared, resource_bundle, spec)
    return TrainingArtifacts._from_training(
        config=config,
        resource_bundle=resource_bundle,
        target_prior=target_prior,
        spec=spec,
        training_subject_ids=prepared.subject_ids,
        training_sample_ids=prepared.sample_ids,
        cell_type_ids=prepared.cell_type_ids,
        training_input_digest=prepared.input_digest,
        frozen_interaction_universe=availability.frozen_interaction_universe,
    )


__all__ = ["FoldTrainingSpec", "TrainingArtifacts", "fit_training_artifacts"]
