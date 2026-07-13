"""Producer-owned training boundary for future subject-level cross-fitting.

This module intentionally implements only the first real fold-trained stage:
the interaction availability universe.  It does not construct an OOF scoring
functional or claim that the remaining response/attribution stages are
cross-fitted.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable
from dataclasses import asdict, dataclass, field
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
from crychic.data import (
    ExpressionTransform,
    InputSchema,
    ValidatedInput,
    validate_anndata,
)
from crychic.design import ContrastSpec, balanced_contrast
from crychic.pseudobulk import (
    ExploratoryAggregate,
    PseudobulkDataset,
    aggregate_pseudobulk,
)
from crychic.resources import ResourceBundle, TargetPrior
from crychic.sender import (
    ContrastCommonSenderFunctional,
    ContrastCommonSenderParameters,
    fit_contrast_common_sender_functional,
)

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


def _resource_bundle_content_id(resource_bundle: ResourceBundle) -> str:
    """Bind the in-memory interaction content used during held-out application."""

    return stable_id("resource_bundle_content", asdict(resource_bundle))


def _target_prior_content_id(target_prior: TargetPrior) -> str:
    """Bind the complete in-memory target prior used by training children."""

    return stable_id("target_prior_content", asdict(target_prior))


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
    frozen_interaction_universe: FrozenInteractionUniverse
    sender_functionals: tuple[ContrastCommonSenderFunctional, ...]
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
        if functionals:
            if any(
                functional.training_subject_ids != subjects
                or functional.filter_universe_id
                != frozen_interaction_universe.filter_universe_id
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
        payload = {
            "cell_type_ids": list(cell_types),
            "completed_stages": list(completed),
            "config_digest": config.digest,
            "filter_universe_id": (frozen_interaction_universe.filter_universe_id),
            "remaining_stages": list(remaining),
            "resource_bundle_content_id": _resource_bundle_content_id(
                resource_bundle
            ),
            "resource_manifest_digest": resource_bundle.manifest_digest,
            "sender_functional_ids": [
                functional.sender_functional_id for functional in functionals
            ],
            "spec_id": spec.spec_id,
            "target_prior_manifest_digest": target_prior.manifest_digest,
            "target_prior_content_id": _target_prior_content_id(target_prior),
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
            "frozen_interaction_universe": frozen_interaction_universe,
            "sender_functionals": functionals,
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
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and isinstance(self.training_subject_ids, tuple)
                and isinstance(self.training_sample_ids, tuple)
                and isinstance(self.cell_type_ids, tuple)
                and isinstance(self.sender_functionals, tuple)
                and isinstance(self.completed_stages, tuple)
                and isinstance(self.remaining_stages, tuple)
                and self.training_subject_ids == repeated.training_subject_ids
                and self.training_sample_ids == repeated.training_sample_ids
                and self.cell_type_ids == repeated.cell_type_ids
                and self.training_input_digest == repeated.training_input_digest
                and self.sender_functionals == repeated.sender_functionals
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
    spec._require_intact()
    if resource_bundle.species is not target_prior.species:
        raise ValueError("resource bundle and target prior species must match")
    if resource_bundle.gene_namespace is not target_prior.gene_namespace:
        raise ValueError("resource bundle and target prior namespace must match")

    prepared = _prepare_raw_fold(adata, config, min_cells=spec.min_cells)
    availability = _fit_interaction_universe(prepared, resource_bundle, spec)
    sender_contrasts = spec.sender_contrasts or _planned_sender_contrasts(
        availability.sample_interactions,
        tuple(config.context_keys),
    )
    sender_functionals = tuple(
        fit_contrast_common_sender_functional(
            availability.sample_interactions,
            contrast=contrast,
            context_keys=tuple(config.context_keys),
            filter_universe_id=availability.filter_universe_id,
            parameters=spec.sender_parameters,
        )
        for contrast in sender_contrasts
    )
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
        sender_functionals=sender_functionals,
    )


__all__ = ["FoldTrainingSpec", "TrainingArtifacts", "fit_training_artifacts"]
