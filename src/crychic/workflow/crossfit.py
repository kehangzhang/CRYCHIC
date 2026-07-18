"""Public subject-blocked orchestration for train/apply scoring stages.

Separate exact-coverage audits verify contrast-common sender rows, the typed
receiver response/precision/incremental chain, and family-common held-out
tables. A complete trusted and tuned chain can be certified as descriptive
OOF scoring; formal inference remains a separate full-pipeline calibration.
"""

from __future__ import annotations

import copyreg
import json
import math
import os
import pickle
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Hashable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from crychic.attribution import (
    DirectionalContrastPairSpec,
    FrozenReceiverFamilyLRHypothesisUniverse,
    FrozenReceiverFamilyOpportunityUniverse,
    GainCalibrationSpec,
    PenaltyTuningSpec,
    PrecisionTransformResult,
    ReceiverFamilyTrainingArtifact,
    fit_receiver_family_training_artifacts,
    fit_response_precision,
    freeze_receiver_family_lr_hypothesis_universe,
    freeze_receiver_family_opportunity_universe,
)
from crychic.attribution.frozen_family import _receiver_family_validation_scope
from crychic.attribution.precision import (
    _raw_vector_digest,
    fit_repeated_response_precision,
)
from crychic.core import (
    ContractError,
    CrychicConfig,
    SeedLineage,
    canonical_json,
    stable_id,
)
from crychic.core._validation import (
    isolated_validation_scope,
    record_validation,
    validation_is_cached,
    validation_scope,
)
from crychic.data import InputMode, validate_anndata
from crychic.design import (
    ContrastSpec,
    FrozenDesignApplication,
    FrozenDesignEncoder,
    SubjectDesign,
    apply_frozen_design_encoder,
    audit_subject_design,
    canonical_context,
    default_design_formula,
    fit_frozen_design_encoder,
    node_context_fields,
    plain_context_value,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.resampling import (
    DesignFoldChecker,
    FoldManifest,
    OOFCoverageAudit,
    SubjectFoldPlan,
    plan_subject_folds,
    validate_oof_subject_coverage,
)
from crychic.resources import ResourceBundle, TargetPrior
from crychic.response import (
    FoldGeneResponseApplication,
    FoldGeneResponseArtifact,
    ReceiverAutonomousProgramResource,
    RepeatedMeasuresCR2ReceiverEffect,
    RepeatedMeasuresReceiverEffect,
    apply_fold_gene_response,
    fit_fold_gene_response,
)
from crychic.response.repeated_fold import (
    RepeatedMeasuresFoldResponseApplication,
    RepeatedMeasuresFoldResponseArtifact,
    apply_repeated_measures_fold_response,
    fit_repeated_measures_cr2_fold_response,
)
from crychic.scoring import (
    FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS,
    CrossReceiverCommonScoringApplication,
    CrossReceiverCommonScoringFunctional,
    FamilyCommonScoringApplication,
    FamilyCommonScoringFunctional,
    FrozenLatentNuisanceSpec,
    PlannedScoringCollectionManifest,
    ReceiverFamilyScoringApplication,
    ReceiverFamilyScoringArtifact,
    ReceiverProgramApplication,
    ReceiverProgramTrainingArtifact,
    ReceiverScoringFunctionalManifest,
    ScoringCollectionDocument,
    ScoringCollectionManifest,
    apply_cross_receiver_common_scoring_functional,
    apply_family_common_scoring_functional,
    apply_receiver_family_scoring_artifact,
    apply_receiver_program_training_artifact,
    family_common_edge_evidence_digest,
    family_common_sender_application_digest,
    fit_cross_receiver_common_scoring_functional,
    fit_family_common_scoring_functional,
    fit_receiver_family_scoring_artifact,
    fit_receiver_program_training_artifact,
    mark_family_common_scoring_application_not_estimable,
    mark_family_common_scoring_not_estimable,
    mark_receiver_family_application_not_estimable,
    mark_receiver_family_scoring_not_estimable,
    mark_receiver_program_application_not_estimable,
    mark_receiver_program_training_not_estimable,
)
from crychic.sender import (
    COMMON_SENDER_APPLICATION_COLUMNS,
    CommonSenderApplication,
    ContrastCommonSenderFunctional,
    SenderContrastSupportStatus,
    apply_contrast_common_sender_functional,
    interaction_ligand_contrast_gate,
)

from .application import (
    TrainingArtifactApplication,
    _apply_training_artifacts_from_prepared,
)
from .directional import (
    DirectionalCrossFitBinding,
    build_directional_crossfit_binding,
)
from .receiver_incremental import (
    ReceiverIncrementalApplication,
    ReceiverIncrementalTrainingArtifact,
    _autonomous_verification_status,
    _receiver_incremental_feature_scale,
    apply_receiver_incremental_training_artifact,
    fit_receiver_incremental_training_artifact,
)
from .receiver_universe import (
    FrozenReceiverUniverse,
    ReceiverTrainingSupportRecord,
    ReceiverTrainingSupportStatus,
    assess_receiver_training_support,
    freeze_receiver_universe,
)
from .training import (
    FoldTrainingSpec,
    SanitizedRawInputIdentity,
    SanitizedRawInputSnapshot,
    TrainingArtifacts,
    _fit_interaction_universe,
    _fit_training_artifacts_from_prepared,
    _input_schema,
    _prepare_validated_root_fold,
    _PreparedRawFold,
    _resource_bundle_content_id,
    _sanitized_raw_input_snapshot,
    _subset_prepared_raw_fold,
    _target_prior_content_id,
)

if TYPE_CHECKING:
    from .certification import CrossFitOOFCertificationAudit

_STAGE_NAME = "contrast_common_sender_application"
_RECEIVER_STAGE_NAME = "receiver_incremental_application_diagnostic"
_STAGE_STATUS = "verified_train_only_oof_partial_pipeline"
_PRODUCER_MARKER = "crychic.workflow.crossfit.v1"
_REMAINING_PUBLIC_STAGES = (
    "attribution_tuning",
    "common_scoring_functional",
    "family_attribution",
    "incremental_downstream",
    "subject_blocked_inner_tuning",
    "selected_penalty_inner_oof_gain_calibration",
)

_WorkItemT = TypeVar("_WorkItemT")
_WorkResultT = TypeVar("_WorkResultT")


def _restore_mapping_proxy(
    value: dict[object, object],
) -> MappingProxyType[object, object]:
    return MappingProxyType(value)


def _reduce_mapping_proxy(
    value: MappingProxyType[object, object],
) -> tuple[
    Callable[[dict[object, object]], MappingProxyType[object, object]],
    tuple[dict[object, object]],
]:
    return _restore_mapping_proxy, (dict(value),)


_ReceiverResponseArtifact: TypeAlias = (
    FoldGeneResponseArtifact | RepeatedMeasuresFoldResponseArtifact
)
_ReceiverResponseApplication: TypeAlias = (
    FoldGeneResponseApplication | RepeatedMeasuresFoldResponseApplication
)
_CPM_SCALE = 1_000_000.0
_MAX_SEED = 2**63 - 1
AutonomousProgramUseScope = Literal[
    "algorithm_diagnostic",
    "biological_analysis",
]
_DEFAULT_AUTONOMOUS_PROGRAM_USE_SCOPE: AutonomousProgramUseScope = (
    "algorithm_diagnostic"
)
_FAMILY_EDGE_EVIDENCE_POLICY = (
    "max_sender_local_availability_with_frozen_train_only_ligand_gate_v2"
)
_FAMILY_BINDING_PRODUCER = "crychic.family_common_crossfit_binding.v2"


def _response_backend_manifest(
    response: _ReceiverResponseArtifact,
) -> dict[str, object]:
    """Build a compact, replayable audit of the response backend and support."""

    response._require_intact()
    raw_precision = np.asarray(response.raw_precision, dtype=np.float64)
    supported_indices = np.flatnonzero(
        np.isfinite(raw_precision) & (raw_precision > 0.0)
    ).astype(int)
    supported_index_set = set(supported_indices.tolist())
    feature_diagnostics: list[dict[str, object]] = []
    repeated_design_manifest: dict[str, object] | None = None
    n_backend_eligible_features: int
    if isinstance(response, RepeatedMeasuresFoldResponseArtifact):
        repeated_design_manifest = response.repeated_design.to_dict()
        repeated_effect = response.repeated_effect
        n_backend_eligible_features = 0
        if isinstance(repeated_effect, RepeatedMeasuresCR2ReceiverEffect):
            artifact_kind = "repeated_cr2_fold_response_v1"
            for index, cr2_effect in enumerate(repeated_effect.feature_effects):
                formal_backend_eligible = cr2_effect.formal_backend_eligible
                if formal_backend_eligible:
                    n_backend_eligible_features += 1
                feature_diagnostics.append(
                    {
                        "backend": cr2_effect.backend,
                        "cluster_df": cr2_effect.cluster_df,
                        "effect_id": cr2_effect.effect_id,
                        "feature_id": cr2_effect.feature_id,
                        "feature_index": index,
                        "formal_backend_eligible": formal_backend_eligible,
                        "n_effective_clusters": cr2_effect.n_effective_clusters,
                        "precision_supported": index in supported_index_set,
                        "reason_code": cr2_effect.reason_code,
                        "status": cr2_effect.status.value,
                    }
                )
        elif isinstance(repeated_effect, RepeatedMeasuresReceiverEffect):
            artifact_kind = "repeated_cr1_fold_response_v1"
            for index, cr1_effect in enumerate(repeated_effect.feature_effects):
                feature_diagnostics.append(
                    {
                        "backend": response.method,
                        "cluster_df": cr1_effect.cluster_df,
                        "effect_id": cr1_effect.effect_id,
                        "feature_id": cr1_effect.feature_id,
                        "feature_index": index,
                        "formal_backend_eligible": False,
                        "n_effective_clusters": (
                            cr1_effect.n_contrast_subject_clusters
                        ),
                        "precision_supported": index in supported_index_set,
                        "reason_code": cr1_effect.reason_code,
                        "status": cr1_effect.status,
                    }
                )
        else:  # pragma: no cover - artifact integrity owns this closed union
            raise TypeError("unsupported repeated receiver effect backend")
    else:
        artifact_kind = "fold_gene_response_v2"
        n_backend_eligible_features = len(supported_indices)
    feature_diagnostics_digest = stable_id(
        "crossfit_response_feature_diagnostics",
        {
            "feature_diagnostics": feature_diagnostics,
            "feature_ids": list(response.feature_ids),
            "response_artifact_id": response.artifact_id,
        },
        schema_version="1",
        digest_length=64,
    )
    payload: dict[str, object] = {
        "artifact_kind": artifact_kind,
        "contrast_name": response.contrast_name,
        "feature_diagnostics": feature_diagnostics,
        "feature_diagnostics_digest": feature_diagnostics_digest,
        "feature_ids": list(response.feature_ids),
        "fold_id": response.fold_id,
        "formal_inference_allowed": False,
        "method": response.method,
        "n_backend_eligible_features": n_backend_eligible_features,
        "n_features": len(response.feature_ids),
        "n_precision_supported_features": len(supported_indices),
        "precision_parent_raw_digest": _raw_vector_digest(raw_precision),
        "precision_supported_feature_indices": supported_indices.tolist(),
        "reason_code": response.reason_code,
        "receiver": response.receiver,
        "repeated_design_manifest": repeated_design_manifest,
        "response_artifact_id": response.artifact_id,
        "response_identity": response._identity_payload(),
        "schema_version": "1.0.0",
        "status": response.status,
    }
    return {
        "response_backend_manifest_id": stable_id(
            "crossfit_response_backend_manifest",
            payload,
            schema_version="1",
        ),
        **payload,
    }


def _precision_audit_manifest(
    precision: PrecisionTransformResult,
) -> dict[str, object]:
    """Persist exact precision provenance and its ordered-value digest."""

    provenance = precision.to_dict()
    payload: dict[str, object] = {
        "numeric_values_persisted": False,
        "ordered_values_digest": precision.transformed_precision_digest,
        "provenance": provenance,
        "replay_scope": "lineage_and_ordered_value_digest_v1",
        "schema_version": "1.0.0",
    }
    return {
        "precision_audit_manifest_id": stable_id(
            "crossfit_precision_audit_manifest",
            payload,
            schema_version="1",
        ),
        **payload,
    }


_FAMILY_SCORE_MODES = ("state", "ecosystem")
_COVERAGE_COLUMNS = (
    "subject_id",
    "sample_id",
    "fold_id",
    "contrast_id",
    "contrast_context",
    "functional_status",
    "sender_functional_id",
    "design_encoder_id",
    "design_status",
    "training_artifact_id",
    "stage",
)
_ASSIGNMENT_PROVENANCE_COLUMNS = (
    "fold_id",
    "contrast_id",
    "training_artifact_id",
    "functional_status",
    "stage",
)
_RECEIVER_COVERAGE_COLUMNS = (
    "subject_id",
    "sample_id",
    "fold_id",
    "contrast_id",
    "contrast_context",
    "receiver",
    "receiver_universe_id",
    "receiver_training_support_id",
    "receiver_training_support_status",
    "receiver_training_support_reason_code",
    "response_artifact_id",
    "precision_transform_id",
    "incremental_training_artifact_id",
    "response_application_id",
    "design_application_id",
    "incremental_application_id",
    "diagnostic_status",
    "diagnostic_reason_code",
    "official_incremental_status",
    "reason_code",
    "stage",
)


def _table_cell_token(value: object) -> dict[str, object]:
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing", "value": None}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "missing", "value": None}
        if not math.isfinite(value):
            raise ValueError("cross-fit tables cannot contain infinite values")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": canonical_json(value),
    }


def _optional_table_text(value: object) -> str | None:
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))
    ):
        return None
    return str(value)


def _table_digest(table_name: str, table: pd.DataFrame) -> str:
    rows = [
        [_table_cell_token(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    rows.sort(key=canonical_json)
    result: str = stable_id(
        "crossfit_table",
        {
            "columns": [str(column) for column in table.columns],
            "rows": rows,
            "table_name": table_name,
        },
        schema_version="1",
        digest_length=64,
    )
    return result


def _contrast_id(contrast: ContrastSpec) -> str:
    result: str = stable_id("contrast", contrast.to_dict())
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossFitSpec:
    """Pre-registered policy for one subject-blocked cross-fit run."""

    contrasts: tuple[ContrastSpec, ...]
    directional_pairs: tuple[DirectionalContrastPairSpec, ...] = ()
    predeclared_receiver_ids: tuple[str, ...] | None = None
    repeat_index: int = 0
    outer_fold_partition_seed: int | None = None
    training_spec: FoldTrainingSpec = field(default_factory=FoldTrainingSpec)
    strata_keys: tuple[str, ...] = ()
    allowed_n_splits: tuple[int, ...] = (5, 4, 3, 2)
    min_train_subjects_per_context: int = 2
    min_test_subjects_per_context: int = 1
    receptor_gate_threshold: float = 0.1
    family_cosine_threshold: float = 0.95
    downstream_minimum_scale: float = 0.25
    autonomous_program_resource: ReceiverAutonomousProgramResource | None = None
    latent_nuisance_spec: FrozenLatentNuisanceSpec | None = None
    autonomous_program_use_scope: AutonomousProgramUseScope = (
        _DEFAULT_AUTONOMOUS_PROGRAM_USE_SCOPE
    )
    penalty_tuning_spec: PenaltyTuningSpec | None = None
    gain_calibration_spec: GainCalibrationSpec | None = None
    schema_version: str = "1.0.0"
    spec_id: str = field(init=False)
    repeat_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 0
        ):
            raise ValueError("repeat_index must be a non-negative integer")
        partition_seed = self.outer_fold_partition_seed
        if partition_seed is not None and (
            isinstance(partition_seed, bool)
            or not isinstance(partition_seed, int)
            or not 0 <= partition_seed <= _MAX_SEED
        ):
            raise ValueError(
                f"outer_fold_partition_seed must be None or an integer between 0 "
                f"and {_MAX_SEED}"
            )
        contrasts = tuple(self.contrasts)
        if not contrasts or any(
            not isinstance(contrast, ContrastSpec) for contrast in contrasts
        ):
            raise TypeError("contrasts must contain at least one ContrastSpec")
        if any(
            not contrast.estimable or len(contrast.weights) < 2
            for contrast in contrasts
        ):
            raise ValueError("cross-fit contrasts must be estimable and multi-context")
        identified = [(_contrast_id(contrast), contrast) for contrast in contrasts]
        if len({contrast_id for contrast_id, _ in identified}) != len(identified):
            raise ValueError("cross-fit contrasts must be unique")
        if len({contrast.name for contrast in contrasts}) != len(contrasts):
            raise ValueError("cross-fit contrast names must be unique")
        contrasts = tuple(
            contrast for _, contrast in sorted(identified, key=lambda item: item[0])
        )
        directional_pairs = tuple(self.directional_pairs)
        if any(
            not isinstance(pair, DirectionalContrastPairSpec)
            for pair in directional_pairs
        ):
            raise TypeError(
                "directional_pairs must contain DirectionalContrastPairSpec values"
            )
        contrast_ids = {_contrast_id(contrast) for contrast in contrasts}
        directional_contrast_ids: list[str] = []
        for pair in directional_pairs:
            pair._require_intact()
            pair_contrast_ids = (
                pair.forward_contrast_id,
                pair.reverse_contrast_id,
            )
            if not set(pair_contrast_ids).issubset(contrast_ids):
                raise ValueError(
                    "directional pair contrasts must be registered in contrasts"
                )
            directional_contrast_ids.extend(pair_contrast_ids)
        if len({pair.pair_spec_id for pair in directional_pairs}) != len(
            directional_pairs
        ):
            raise ValueError("directional_pairs must be unique")
        if len(set(directional_contrast_ids)) != len(directional_contrast_ids):
            raise ValueError("a contrast may belong to at most one directional pair")
        directional_pairs = tuple(
            sorted(directional_pairs, key=lambda pair: pair.pair_spec_id)
        )
        predeclared_receivers = self.predeclared_receiver_ids
        if predeclared_receivers is not None:
            if type(predeclared_receivers) is not tuple:
                raise TypeError(
                    "predeclared_receiver_ids must be a tuple of identifiers"
                )
            predeclared_receivers = tuple(predeclared_receivers)
            if not predeclared_receivers or any(
                not isinstance(receiver, str)
                or not receiver
                or receiver != receiver.strip()
                for receiver in predeclared_receivers
            ):
                raise ValueError(
                    "predeclared_receiver_ids must contain canonical non-empty IDs"
                )
            if len(predeclared_receivers) != len(set(predeclared_receivers)):
                raise ValueError("predeclared_receiver_ids must be unique")
            predeclared_receivers = tuple(sorted(predeclared_receivers))
        if not isinstance(self.training_spec, FoldTrainingSpec):
            raise TypeError("training_spec must be a FoldTrainingSpec")
        self.training_spec._require_intact()
        declared = self.training_spec.sender_contrasts
        if declared is not None and {
            _contrast_id(contrast) for contrast in declared
        } != {_contrast_id(contrast) for contrast in contrasts}:
            raise ValueError(
                "training_spec.sender_contrasts must match CrossFitSpec.contrasts"
            )
        training_spec = replace(self.training_spec, sender_contrasts=contrasts)
        strata = tuple(self.strata_keys)
        if any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in strata
        ):
            raise ValueError("strata_keys must contain non-empty field names")
        if len(set(strata)) != len(strata):
            raise ValueError("strata_keys must be unique")
        allowed = tuple(self.allowed_n_splits)
        if (
            not allowed
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in allowed
            )
            or tuple(sorted(set(allowed), reverse=True)) != allowed
        ):
            raise ValueError(
                "allowed_n_splits must be unique integers >= 2 in descending order"
            )
        for field_name in (
            "min_train_subjects_per_context",
            "min_test_subjects_per_context",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be an integer >= 1")
        gate_threshold = float(self.receptor_gate_threshold)
        family_threshold = float(self.family_cosine_threshold)
        minimum_scale = float(self.downstream_minimum_scale)
        autonomous_resource = self.autonomous_program_resource
        latent_nuisance_spec = self.latent_nuisance_spec
        autonomous_use_scope = self.autonomous_program_use_scope
        if not isinstance(autonomous_use_scope, str) or autonomous_use_scope not in {
            "algorithm_diagnostic",
            "biological_analysis",
        }:
            raise ValueError(
                "autonomous_program_use_scope must be algorithm_diagnostic or "
                "biological_analysis"
            )
        if autonomous_resource is not None:
            if type(autonomous_resource) is not ReceiverAutonomousProgramResource:
                raise TypeError(
                    "autonomous_program_resource must be a "
                    "producer-owned ReceiverAutonomousProgramResource"
                )
            autonomous_resource._require_producer_owned()
            if (
                autonomous_use_scope == "biological_analysis"
                and not autonomous_resource.is_biological_reference_trusted
            ):
                raise ValueError(
                    "biological_analysis requires a registered biological-reference "
                    "autonomous program resource"
                )
        if latent_nuisance_spec is not None:
            if not isinstance(latent_nuisance_spec, FrozenLatentNuisanceSpec):
                raise TypeError(
                    "latent_nuisance_spec must be a FrozenLatentNuisanceSpec"
                )
            latent_nuisance_spec._require_intact()
        tuning_spec = self.penalty_tuning_spec
        if tuning_spec is not None:
            if not isinstance(tuning_spec, PenaltyTuningSpec):
                raise TypeError("penalty_tuning_spec must be a PenaltyTuningSpec")
            tuning_spec._require_intact()
            gain_calibration_spec = self.gain_calibration_spec
            if gain_calibration_spec is None:
                gain_calibration_spec = GainCalibrationSpec()
            elif not isinstance(gain_calibration_spec, GainCalibrationSpec):
                raise TypeError("gain_calibration_spec must be a GainCalibrationSpec")
            gain_calibration_spec._require_intact()
        else:
            gain_calibration_spec = self.gain_calibration_spec
            if gain_calibration_spec is not None:
                raise ValueError("gain_calibration_spec requires penalty_tuning_spec")
        if not np.isfinite(gate_threshold) or not 0 < gate_threshold <= 1:
            raise ValueError("receptor_gate_threshold must be finite in (0, 1]")
        if not np.isfinite(family_threshold) or not 0 <= family_threshold <= 1:
            raise ValueError("family_cosine_threshold must be finite in [0, 1]")
        if not np.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("downstream_minimum_scale must be finite and positive")
        if self.schema_version != "1.0.0":
            raise ValueError("CrossFitSpec schema_version must be 1.0.0")
        payload = {
            "allowed_n_splits": list(allowed),
            "contrasts": [contrast.to_dict() for contrast in contrasts],
            "min_test_subjects_per_context": self.min_test_subjects_per_context,
            "min_train_subjects_per_context": self.min_train_subjects_per_context,
            "receptor_gate_threshold": gate_threshold,
            "family_cosine_threshold": family_threshold,
            "downstream_minimum_scale": minimum_scale,
            "schema_version": self.schema_version,
            "strata_keys": list(strata),
            "training_spec_id": training_spec.spec_id,
        }
        if directional_pairs:
            payload["directional_pair_spec_ids"] = [
                pair.pair_spec_id for pair in directional_pairs
            ]
        if predeclared_receivers is not None:
            payload["predeclared_receiver_ids"] = list(predeclared_receivers)
        if autonomous_resource is not None:
            payload["autonomous_program_resource_id"] = autonomous_resource.artifact_id
        if latent_nuisance_spec is not None:
            payload["latent_nuisance_spec_id"] = latent_nuisance_spec.spec_id
        if autonomous_use_scope != _DEFAULT_AUTONOMOUS_PROGRAM_USE_SCOPE:
            payload["autonomous_program_use_scope"] = autonomous_use_scope
        if tuning_spec is not None:
            payload["penalty_tuning_spec_id"] = tuning_spec.spec_id
            assert gain_calibration_spec is not None
            payload["gain_calibration_spec_id"] = gain_calibration_spec.spec_id
        if partition_seed is not None:
            payload["outer_fold_partition_seed"] = partition_seed
        spec_id = stable_id(
            "subject_crossfit_spec", payload, schema_version=self.schema_version
        )
        object.__setattr__(self, "contrasts", contrasts)
        object.__setattr__(self, "directional_pairs", directional_pairs)
        object.__setattr__(self, "predeclared_receiver_ids", predeclared_receivers)
        object.__setattr__(self, "training_spec", training_spec)
        object.__setattr__(self, "strata_keys", strata)
        object.__setattr__(self, "allowed_n_splits", allowed)
        object.__setattr__(self, "receptor_gate_threshold", gate_threshold)
        object.__setattr__(self, "family_cosine_threshold", family_threshold)
        object.__setattr__(self, "downstream_minimum_scale", minimum_scale)
        object.__setattr__(self, "autonomous_program_resource", autonomous_resource)
        object.__setattr__(self, "latent_nuisance_spec", latent_nuisance_spec)
        object.__setattr__(
            self,
            "autonomous_program_use_scope",
            autonomous_use_scope,
        )
        object.__setattr__(self, "penalty_tuning_spec", tuning_spec)
        object.__setattr__(self, "gain_calibration_spec", gain_calibration_spec)
        object.__setattr__(self, "outer_fold_partition_seed", partition_seed)
        object.__setattr__(self, "spec_id", spec_id)
        object.__setattr__(
            self,
            "repeat_id",
            stable_id(
                "subject_crossfit_repeat",
                {"spec_id": spec_id, "repeat": self.repeat_index},
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the pre-registered cross-fit policy manifest."""

        self._require_intact()
        result: dict[str, object] = {
            "spec_id": self.spec_id,
            "repeat_id": self.repeat_id,
            "repeat_index": self.repeat_index,
            "schema_version": self.schema_version,
            "contrasts": [contrast.to_dict() for contrast in self.contrasts],
            "training_spec_id": self.training_spec.spec_id,
            "strata_keys": list(self.strata_keys),
            "allowed_n_splits": list(self.allowed_n_splits),
            "min_train_subjects_per_context": (self.min_train_subjects_per_context),
            "min_test_subjects_per_context": self.min_test_subjects_per_context,
            "receptor_gate_threshold": self.receptor_gate_threshold,
            "family_cosine_threshold": self.family_cosine_threshold,
            "downstream_minimum_scale": self.downstream_minimum_scale,
            "autonomous_program_resource": (
                None
                if self.autonomous_program_resource is None
                else self.autonomous_program_resource.to_dict()
            ),
            "latent_nuisance_spec": (
                None
                if self.latent_nuisance_spec is None
                else self.latent_nuisance_spec.to_dict()
            ),
            "penalty_tuning_spec": (
                None
                if self.penalty_tuning_spec is None
                else self.penalty_tuning_spec.to_dict()
            ),
            "gain_calibration_spec": (
                None
                if self.gain_calibration_spec is None
                else self.gain_calibration_spec.to_dict()
            ),
        }
        if self.directional_pairs:
            result["directional_pairs"] = [
                pair.to_dict() for pair in self.directional_pairs
            ]
        if self.predeclared_receiver_ids is not None:
            result["predeclared_receiver_ids"] = list(self.predeclared_receiver_ids)
        if self.outer_fold_partition_seed is not None:
            result["outer_fold_partition_seed"] = self.outer_fold_partition_seed
        if self.autonomous_program_use_scope != _DEFAULT_AUTONOMOUS_PROGRAM_USE_SCOPE:
            result["autonomous_program_use_scope"] = self.autonomous_program_use_scope
        return result

    def _require_intact(self) -> None:
        """Reject forced mutation of the frozen cross-fit policy."""

        try:
            repeated = CrossFitSpec(
                contrasts=self.contrasts,
                directional_pairs=self.directional_pairs,
                predeclared_receiver_ids=self.predeclared_receiver_ids,
                repeat_index=self.repeat_index,
                outer_fold_partition_seed=self.outer_fold_partition_seed,
                training_spec=self.training_spec,
                strata_keys=self.strata_keys,
                allowed_n_splits=self.allowed_n_splits,
                min_train_subjects_per_context=(self.min_train_subjects_per_context),
                min_test_subjects_per_context=self.min_test_subjects_per_context,
                receptor_gate_threshold=self.receptor_gate_threshold,
                family_cosine_threshold=self.family_cosine_threshold,
                downstream_minimum_scale=self.downstream_minimum_scale,
                autonomous_program_resource=self.autonomous_program_resource,
                latent_nuisance_spec=self.latent_nuisance_spec,
                autonomous_program_use_scope=self.autonomous_program_use_scope,
                penalty_tuning_spec=self.penalty_tuning_spec,
                gain_calibration_spec=self.gain_calibration_spec,
                schema_version=self.schema_version,
            )
            valid = (
                isinstance(self.contrasts, tuple)
                and isinstance(self.directional_pairs, tuple)
                and isinstance(self.strata_keys, tuple)
                and isinstance(self.allowed_n_splits, tuple)
                and self.contrasts == repeated.contrasts
                and tuple(pair.pair_spec_id for pair in self.directional_pairs)
                == tuple(pair.pair_spec_id for pair in repeated.directional_pairs)
                and self.predeclared_receiver_ids == repeated.predeclared_receiver_ids
                and self.repeat_index == repeated.repeat_index
                and self.outer_fold_partition_seed == repeated.outer_fold_partition_seed
                and self.training_spec.spec_id == repeated.training_spec.spec_id
                and self.strata_keys == repeated.strata_keys
                and self.allowed_n_splits == repeated.allowed_n_splits
                and self.autonomous_program_use_scope
                == repeated.autonomous_program_use_scope
                and (
                    None
                    if self.latent_nuisance_spec is None
                    else self.latent_nuisance_spec.spec_id
                )
                == (
                    None
                    if repeated.latent_nuisance_spec is None
                    else repeated.latent_nuisance_spec.spec_id
                )
                and (
                    None
                    if self.gain_calibration_spec is None
                    else self.gain_calibration_spec.spec_id
                )
                == (
                    None
                    if repeated.gain_calibration_spec is None
                    else repeated.gain_calibration_spec.spec_id
                )
                and self.spec_id == repeated.spec_id
                and self.repeat_id == repeated.repeat_id
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Cross-fit specification failed integrity validation",
                code="crossfit_spec_integrity_violation",
                field="spec_id",
                remediation="Rebuild CrossFitSpec from the declared policy",
            ) from error
        if not valid:
            raise ContractError(
                "Cross-fit specification failed integrity validation",
                code="crossfit_spec_integrity_violation",
                field="spec_id",
                remediation="Rebuild CrossFitSpec from the declared policy",
            )


def _outer_fold_partition_lineage(
    spec: CrossFitSpec,
) -> SeedLineage | None:
    seed = spec.outer_fold_partition_seed
    if seed is None:
        return None
    return SeedLineage(seed).derive(
        "subject_crossfit_outer_partition_v1",
        f"repeat={spec.repeat_index}",
    )


def _inner_tuning_partition_base_lineage(
    spec: CrossFitSpec,
    fold: FoldManifest,
) -> SeedLineage | None:
    tuning_spec = spec.penalty_tuning_spec
    outer_lineage = _outer_fold_partition_lineage(spec)
    if tuning_spec is None or outer_lineage is None:
        return None
    outer_scope_id = stable_id(
        "subject_crossfit_outer_partition_scope",
        {
            "partition_seed_lineage": outer_lineage.to_dict(),
            "test_subject_ids": list(fold.test_subject_ids),
            "train_subject_ids": list(fold.train_subject_ids),
        },
        schema_version="1",
    )
    return SeedLineage(tuning_spec.root_seed).derive(
        "receiver_incremental_inner_partition_v1",
        tuning_spec.spec_id,
        outer_scope_id,
    )


def _receiver_inner_partition_lineage(
    base_lineage: SeedLineage | None,
    *,
    receiver: str,
    contrast_id: str,
) -> SeedLineage | None:
    if base_lineage is None:
        return None
    return base_lineage.derive(
        "receiver_contrast",
        f"receiver={receiver}",
        f"contrast_id={contrast_id}",
    )


@dataclass(frozen=True, slots=True, init=False)
class _FamilyCommonCrossFitBinding:
    """Bind one family-common child to its exact public held-out inputs."""

    family_common_functional_id: str
    family_common_application_id: str
    training_application_id: str
    availability_sample_interactions_digest: str
    edge_evidence_policy_id: str
    edge_evidence_digest: str
    sender_input_digest: str
    binding_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "Family-common cross-fit bindings are producer-owned by "
            "run_subject_crossfit()"
        )

    @classmethod
    def _from_workflow(
        cls,
        functional: FamilyCommonScoringFunctional,
        application: FamilyCommonScoringApplication,
        training_application: TrainingArtifactApplication,
        edge_evidence: pd.DataFrame,
        sender_application: CommonSenderApplication,
    ) -> _FamilyCommonCrossFitBinding:
        functional._require_intact()
        application._require_intact()
        training_application._require_intact()
        sender_application = CommonSenderApplication(
            sender_application.table, sender_application.functional
        )
        edge_digest = family_common_edge_evidence_digest(edge_evidence)
        sender_digest = family_common_sender_application_digest(sender_application)
        if (
            application.edge_evidence_digest != edge_digest
            or application.sender_application_digest != sender_digest
        ):
            raise ContractError(
                "Family-common application inputs do not match its cross-fit binding",
                code="family_common_crossfit_input_mismatch",
                field="application_id",
                remediation=(
                    "Reapply the common functional to canonical held-out inputs"
                ),
            )
        values = {
            "family_common_functional_id": (functional.family_common_functional_id),
            "family_common_application_id": application.application_id,
            "training_application_id": training_application.application_id,
            "availability_sample_interactions_digest": _table_digest(
                "family_common_source_availability",
                training_application.availability.sample_interactions,
            ),
            "edge_evidence_policy_id": _FAMILY_EDGE_EVIDENCE_POLICY,
            "edge_evidence_digest": edge_digest,
            "sender_input_digest": sender_digest,
            "_producer_marker": _FAMILY_BINDING_PRODUCER,
        }
        self = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "binding_id",
            stable_id(
                "family_common_crossfit_binding",
                self._identity_payload(),
                schema_version="2",
            ),
        )
        return self

    def _identity_payload(self) -> dict[str, object]:
        return {
            "availability_sample_interactions_digest": (
                self.availability_sample_interactions_digest
            ),
            "edge_evidence_digest": self.edge_evidence_digest,
            "edge_evidence_policy_id": self.edge_evidence_policy_id,
            "family_common_application_id": self.family_common_application_id,
            "family_common_functional_id": self.family_common_functional_id,
            "sender_input_digest": self.sender_input_digest,
            "training_application_id": self.training_application_id,
        }

    def _require_intact(self) -> None:
        valid = (
            self._producer_marker == _FAMILY_BINDING_PRODUCER
            and self.edge_evidence_policy_id == _FAMILY_EDGE_EVIDENCE_POLICY
            and all(
                isinstance(value, str) and bool(value) and value == value.strip()
                for value in self._identity_payload().values()
            )
            and self.binding_id
            == stable_id(
                "family_common_crossfit_binding",
                self._identity_payload(),
                schema_version="2",
            )
        )
        if not valid:
            raise ContractError(
                "Family-common cross-fit binding failed integrity validation",
                code="family_common_crossfit_binding_integrity_violation",
                field="binding_id",
                remediation="Rerun the public subject cross-fit workflow",
            )

    def to_dict(self) -> dict[str, object]:
        self._require_intact()
        return {"binding_id": self.binding_id, **self._identity_payload()}


def _string_key_set(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    table_name: str,
) -> set[tuple[str, ...]]:
    keys = [
        tuple(str(value) for value in row)
        for row in table.loc[:, list(columns)].itertuples(index=False, name=None)
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{table_name} contains duplicate exact-coverage keys")
    return set(keys)


def _require_sample_lineage(
    table: pd.DataFrame,
    expected_samples: dict[str, tuple[str, str]],
    *,
    table_name: str,
) -> None:
    observed: dict[str, tuple[str, str]] = {}
    for sample_id, group in table.groupby("sample_id", observed=True, sort=False):
        subjects = tuple(sorted(set(group["subject_id"].astype(str))))
        contexts = tuple(sorted(set(group["context_id"].astype(str))))
        if len(subjects) != 1 or len(contexts) != 1:
            raise ValueError(f"{table_name} sample lineage is not unique")
        observed[str(sample_id)] = (subjects[0], contexts[0])
    if observed != expected_samples:
        raise ValueError(f"{table_name} does not exactly cover held-out samples")


def _require_family_common_exact_coverage(
    functional: FamilyCommonScoringFunctional,
    application: FamilyCommonScoringApplication,
    design_application: FrozenDesignApplication,
    tables: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    expected_samples = {
        str(sample_id): (str(subject_id), str(context_id))
        for sample_id, subject_id, context_id in zip(
            design_application.sample_ids,
            design_application.sample_subject_ids,
            design_application.sample_context_ids,
            strict=True,
        )
        if str(context_id) in functional.context_ids
    }
    expected_subjects = tuple(
        sorted({subject_id for subject_id, _ in expected_samples.values()})
    )
    if application.heldout_subject_ids != expected_subjects:
        raise ValueError(
            "family-common application subjects do not exactly cover heldout fold"
        )

    attribution, differential, family_scores, member_scores, sender_scores = tables
    modes = tuple(sorted(_FAMILY_SCORE_MODES))
    membership = {
        item.interaction_id: (item.family_id, item.driver_id)
        for item in functional.interactions
    }
    scored_family_ids = tuple(
        sorted({family_id for family_id, _ in membership.values()})
    )

    if _string_key_set(
        attribution,
        ("family_id",),
        table_name="family attribution",
    ) != {(family_id,) for family_id in functional.family_ids}:
        raise ValueError("family attribution does not cover the frozen family set")
    if _string_key_set(
        differential,
        ("subject_id", "family_id"),
        table_name="subject differential",
    ) != {
        (subject_id, family_id)
        for subject_id in expected_subjects
        for family_id in functional.family_ids
    }:
        raise ValueError("subject differential does not have exact held-out coverage")

    expected_family_keys = {
        (sample_id, functional.receiver, family_id, mode)
        for sample_id in expected_samples
        for family_id in scored_family_ids
        for mode in modes
    }
    if (
        _string_key_set(
            family_scores,
            ("sample_id", "receiver", "family_id", "mode"),
            table_name="family scores",
        )
        != expected_family_keys
    ):
        raise ValueError("family scores do not have exact sample/family/mode coverage")

    expected_member_keys = {
        (
            sample_id,
            functional.receiver,
            family_id,
            driver_id,
            interaction_id,
            mode,
        )
        for sample_id in expected_samples
        for interaction_id, (family_id, driver_id) in membership.items()
        for mode in modes
    }
    if (
        _string_key_set(
            member_scores,
            (
                "sample_id",
                "receiver",
                "family_id",
                "driver_id",
                "interaction_id",
                "mode",
            ),
            table_name="member scores",
        )
        != expected_member_keys
    ):
        raise ValueError("member scores do not have exact sample/interaction coverage")

    sender_candidates = {
        (prior.interaction_id, prior.sender)
        for prior in functional.sender_functional.candidate_priors
        if prior.receiver == functional.receiver and prior.interaction_id in membership
    }
    expected_sender_keys = {
        (
            sample_id,
            functional.receiver,
            membership[interaction_id][0],
            membership[interaction_id][1],
            interaction_id,
            mode,
            sender,
        )
        for sample_id in expected_samples
        for interaction_id, sender in sender_candidates
        for mode in modes
    }
    if (
        _string_key_set(
            sender_scores,
            (
                "sample_id",
                "receiver",
                "family_id",
                "driver_id",
                "interaction_id",
                "mode",
                "sender",
            ),
            table_name="sender scores",
        )
        != expected_sender_keys
    ):
        raise ValueError("sender scores do not have exact candidate-sender coverage")

    for table, table_name in (
        (family_scores, "family scores"),
        (member_scores, "member scores"),
    ):
        _require_sample_lineage(
            table,
            expected_samples,
            table_name=table_name,
        )
    if expected_sender_keys:
        _require_sample_lineage(
            sender_scores,
            expected_samples,
            table_name="sender scores",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossFitFoldArtifacts:
    """Training and held-out application artifacts for one planned fold."""

    fold_id: str
    training: TrainingArtifacts
    application: TrainingArtifactApplication
    receiver_training_support: tuple[ReceiverTrainingSupportRecord, ...]
    design_encoders: tuple[FrozenDesignEncoder, ...]
    design_applications: tuple[FrozenDesignApplication, ...]
    receiver_family_models: tuple[ReceiverFamilyScoringArtifact, ...]
    receiver_family_applications: tuple[ReceiverFamilyScoringApplication, ...]
    receiver_program_models: tuple[ReceiverProgramTrainingArtifact, ...]
    receiver_program_applications: tuple[ReceiverProgramApplication, ...]
    receiver_responses: tuple[_ReceiverResponseArtifact, ...]
    response_precisions: tuple[PrecisionTransformResult, ...]
    receiver_incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...]
    receiver_response_applications: tuple[_ReceiverResponseApplication, ...]
    receiver_incremental_applications: tuple[ReceiverIncrementalApplication, ...]
    family_common_functionals: tuple[FamilyCommonScoringFunctional, ...]
    family_common_applications: tuple[FamilyCommonScoringApplication, ...]
    family_common_bindings: tuple[_FamilyCommonCrossFitBinding, ...]
    directional_response_bindings: tuple[DirectionalCrossFitBinding, ...] = ()
    cross_receiver_common_functionals: tuple[
        CrossReceiverCommonScoringFunctional, ...
    ] = ()
    cross_receiver_common_applications: tuple[
        CrossReceiverCommonScoringApplication, ...
    ] = ()

    @validation_scope()
    def __post_init__(self) -> None:
        if validation_is_cached(self):
            return
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise ValueError("fold_id must be a non-empty identifier")
        if not isinstance(self.training, TrainingArtifacts):
            raise TypeError("training must be TrainingArtifacts")
        self.training._require_intact()
        if not isinstance(self.application, TrainingArtifactApplication):
            raise TypeError("application must be TrainingArtifactApplication")
        self.application._require_intact()
        if self.application.training_artifact_id != self.training.training_artifact_id:
            raise ValueError("fold application is not bound to its training artifacts")
        support_records = tuple(self.receiver_training_support)
        if not support_records or any(
            not isinstance(record, ReceiverTrainingSupportRecord)
            for record in support_records
        ):
            raise TypeError(
                "receiver_training_support must contain complete support records"
            )
        for record in support_records:
            record._require_intact()
        support_receivers = tuple(record.receiver_id for record in support_records)
        if support_receivers != tuple(sorted(support_receivers)) or len(
            support_receivers
        ) != len(set(support_receivers)):
            raise ValueError(
                "receiver training-support records must be canonical and unique"
            )
        if any(
            record.outer_fold_id != self.fold_id
            or record.training_cell_type_ids != self.training.cell_type_ids
            for record in support_records
        ):
            raise ValueError(
                "receiver training-support records do not match the fold training axis"
            )
        observed_support = tuple(
            record.receiver_id
            for record in support_records
            if record.status is ReceiverTrainingSupportStatus.OBSERVED
        )
        if observed_support != self.training.cell_type_ids:
            raise ValueError(
                "observed receiver support must exactly match training cell types"
            )
        object.__setattr__(self, "receiver_training_support", support_records)
        encoders = tuple(self.design_encoders)
        applications = tuple(self.design_applications)
        if not encoders or len(encoders) != len(applications):
            raise ValueError("fold design encoders and applications must align")
        if len({encoder.encoder_id for encoder in encoders}) != len(encoders):
            raise ValueError("fold design encoders must be contrast-unique")
        for encoder, design_application in zip(encoders, applications, strict=True):
            encoder.to_dict()
            design_application.require_compatible(encoder)
            if encoder.training_subject_ids != self.training.training_subject_ids:
                raise ValueError("design encoder training subjects do not match fold")
            if design_application.encoder_id != encoder.encoder_id:
                raise ValueError("design application does not match its encoder")
            if design_application.subject_ids != self.application.heldout_subject_ids:
                raise ValueError(
                    "design application subjects do not match heldout fold"
                )
        object.__setattr__(self, "design_encoders", encoders)
        object.__setattr__(self, "design_applications", applications)
        receiver_models = tuple(self.receiver_family_models)
        receiver_applications = tuple(self.receiver_family_applications)
        if not receiver_models or len(receiver_models) != len(receiver_applications):
            raise ValueError("fold receiver-family models and applications must align")
        model_keys = [
            (
                model.contrast_name,
                model.receiver_family_artifact.receiver,
            )
            for model in receiver_models
        ]
        if len(model_keys) != len(set(model_keys)):
            raise ValueError("fold receiver-family models must be receiver-unique")
        expected_model_keys = {
            (encoder.contrast.name, receiver)
            for encoder in encoders
            for receiver in self.training.cell_type_ids
        }
        if set(model_keys) != expected_model_keys:
            raise ValueError(
                "fold receiver-family models must cover every contrast and receiver"
            )
        for model, receiver_application in zip(
            receiver_models, receiver_applications, strict=True
        ):
            model._require_producer_owned()
            model.receiver_family_artifact._require_producer_owned()
            receiver_application._require_intact()
            if model.training_subject_ids != self.training.training_subject_ids:
                raise ValueError(
                    "receiver-family training subjects do not match the fold"
                )
            if receiver_application.training_artifact_id != model.training_artifact_id:
                raise ValueError(
                    "receiver-family application does not match its training model"
                )
            if receiver_application.active_family_ids != model.active_family_ids:
                raise ValueError(
                    "receiver-family application families do not match its model"
                )
            receiver_downstream = receiver_application.downstream_application
            if receiver_downstream is not None and (
                model.downstream_functional is None
                or receiver_downstream.downstream_functional_id
                != model.downstream_functional.downstream_functional_id
            ):
                raise ValueError(
                    "receiver-family application functional does not match its model"
                )
            if not set(receiver_application.heldout_subject_ids).issubset(
                self.application.heldout_subject_ids
            ):
                raise ValueError(
                    "receiver-family application subjects are outside heldout fold"
                )
            if set(receiver_application.heldout_subject_ids).intersection(
                model.training_subject_ids
            ):
                raise ValueError(
                    "receiver-family application overlaps its training subjects"
                )
        object.__setattr__(self, "receiver_family_models", receiver_models)
        object.__setattr__(self, "receiver_family_applications", receiver_applications)
        program_models = tuple(self.receiver_program_models)
        program_applications = tuple(self.receiver_program_applications)
        if len(program_models) != len(receiver_models) or len(
            program_applications
        ) != len(receiver_models):
            raise ValueError("fold receiver-program parent chains must align")
        design_by_name = {
            encoder.contrast.name: (encoder, application)
            for encoder, application in zip(encoders, applications, strict=True)
        }
        for family_model, program_model, program_application in zip(
            receiver_models,
            program_models,
            program_applications,
            strict=True,
        ):
            program_model._require_intact()
            program_application._require_intact()
            key = (
                family_model.contrast_name,
                family_model.receiver_family_artifact.receiver,
            )
            if (program_model.contrast_name, program_model.receiver) != key:
                raise ValueError("receiver-program training keys do not align")
            if (
                program_model.receiver_family_artifact.training_artifact_id
                != family_model.receiver_family_artifact.training_artifact_id
                or program_application.training_artifact.training_artifact_id
                != program_model.training_artifact_id
                or program_application.heldout_subject_ids
                != self.application.heldout_subject_ids
            ):
                raise ValueError("receiver-program parent lineage is invalid")
            encoder, design_application = design_by_name[program_model.contrast_name]
            context_ids = {
                node_context_fields(node, encoder.context_keys)[0]
                for node in encoder.contrast.weights
            }
            expected_rows = {
                (sample_id, subject_id, context_id)
                for sample_id, subject_id, context_id in zip(
                    design_application.sample_ids,
                    design_application.sample_subject_ids,
                    design_application.sample_context_ids,
                    strict=True,
                )
                if context_id in context_ids
            }
            observed_rows = set(
                zip(
                    program_application.sample_ids,
                    program_application.sample_subject_ids,
                    program_application.sample_context_ids,
                    strict=True,
                )
            )
            if observed_rows != expected_rows:
                raise ValueError(
                    "receiver-program application does not exactly cover heldout rows"
                )
        object.__setattr__(self, "receiver_program_models", program_models)
        object.__setattr__(self, "receiver_program_applications", program_applications)
        responses = tuple(self.receiver_responses)
        precisions = tuple(self.response_precisions)
        incremental_models = tuple(self.receiver_incremental_models)
        response_applications = tuple(self.receiver_response_applications)
        incremental_applications = tuple(self.receiver_incremental_applications)
        parent_groups = (
            responses,
            precisions,
            incremental_models,
            response_applications,
            incremental_applications,
        )
        if any(len(group) != len(receiver_models) for group in parent_groups):
            raise ValueError("fold receiver incremental parent chains must align")
        for (
            family_model,
            response,
            precision,
            incremental_model,
            response_application,
            incremental_application,
        ) in zip(
            receiver_models,
            responses,
            precisions,
            incremental_models,
            response_applications,
            incremental_applications,
            strict=True,
        ):
            receiver = family_model.receiver_family_artifact.receiver
            key = (family_model.contrast_name, receiver)
            if (response.contrast_name, response.receiver) != key or (
                incremental_model.contrast_name,
                incremental_model.receiver,
            ) != key:
                raise ValueError("receiver incremental training keys do not align")
            encoder, design_application = design_by_name[family_model.contrast_name]
            response.require_compatible(encoder)
            precision.require_response_compatible(response)
            incremental_model._require_intact()
            response_application.require_compatible(response, design_application)
            incremental_application._require_intact()
            if (
                response.fold_id != self.fold_id
                or response.encoder_id != encoder.encoder_id
                or response.training_subject_ids != self.training.training_subject_ids
                or response.training_input_digest != self.training.training_input_digest
                or precision.response_artifact_id != response.artifact_id
                or precision.precision_transform_id
                != incremental_model.precision_transform_id
                or incremental_model.response_artifact_id != response.artifact_id
                or incremental_model.receiver_family_training_artifact_id
                != family_model.receiver_family_artifact.training_artifact_id
                or incremental_model.encoder_id != encoder.encoder_id
                or incremental_model.context_regressor_id
                != encoder.context_regressor_id
                or incremental_model.nuisance_design_id != encoder.nuisance_design_id
                or incremental_model.nuisance_column_ids != encoder.nuisance_column_ids
            ):
                raise ValueError(
                    "receiver incremental training parent chain is invalid"
                )
            if (
                response_application.training_response_id != response.artifact_id
                or response_application.design_application_id
                != design_application.application_id
                or incremental_application.training_artifact_id
                != incremental_model.training_artifact_id
                or incremental_application.response_application_id
                != response_application.application_id
                or incremental_application.design_application_id
                != design_application.application_id
            ):
                raise ValueError(
                    "receiver incremental held-out parent chain is invalid"
                )
            if (
                response_application.subject_ids != self.application.heldout_subject_ids
                or incremental_application.heldout_subject_ids
                != self.application.heldout_subject_ids
            ):
                raise ValueError("receiver incremental applications do not cover fold")
            if set(incremental_application.heldout_subject_ids).intersection(
                incremental_model.training_subject_ids
            ):
                raise ValueError("receiver incremental application overlaps training")
        object.__setattr__(self, "receiver_responses", responses)
        object.__setattr__(self, "response_precisions", precisions)
        object.__setattr__(self, "receiver_incremental_models", incremental_models)
        object.__setattr__(
            self, "receiver_response_applications", response_applications
        )
        object.__setattr__(
            self, "receiver_incremental_applications", incremental_applications
        )
        directional_bindings = tuple(self.directional_response_bindings)
        if any(
            not isinstance(binding, DirectionalCrossFitBinding)
            for binding in directional_bindings
        ):
            raise TypeError(
                "directional_response_bindings must contain "
                "DirectionalCrossFitBinding values"
            )
        response_ids = {response.artifact_id for response in responses}
        incremental_model_ids = {
            model.training_artifact_id for model in incremental_models
        }
        incremental_application_ids = {
            application.application_id for application in incremental_applications
        }
        directional_keys: list[tuple[str, str]] = []
        for binding in directional_bindings:
            binding._require_intact()
            if (
                binding.fold_id != self.fold_id
                or binding.receiver not in self.training.cell_type_ids
                or binding.training_subject_ids != self.training.training_subject_ids
                or binding.heldout_subject_ids != self.application.heldout_subject_ids
                or binding.forward_response_id not in response_ids
                or binding.reverse_response_id not in response_ids
                or binding.forward_training_artifact_id not in incremental_model_ids
                or binding.reverse_training_artifact_id not in incremental_model_ids
                or binding.forward_application_id not in incremental_application_ids
                or binding.reverse_application_id not in incremental_application_ids
            ):
                raise ValueError(
                    "directional response binding does not belong to this fold"
                )
            directional_keys.append((binding.pair_spec_id, binding.receiver))
        if len(directional_keys) != len(set(directional_keys)) or len(
            {binding.binding_id for binding in directional_bindings}
        ) != len(directional_bindings):
            raise ValueError(
                "directional response bindings must be pair/receiver unique"
            )
        object.__setattr__(self, "directional_response_bindings", directional_bindings)
        common_functionals = tuple(self.family_common_functionals)
        common_applications = tuple(self.family_common_applications)
        common_bindings = tuple(self.family_common_bindings)
        if (
            bool(common_functionals) != bool(common_applications)
            or bool(common_functionals) != bool(common_bindings)
            or (common_functionals and len(common_functionals) != len(receiver_models))
            or len(common_functionals) != len(common_applications)
            or len(common_functionals) != len(common_bindings)
        ):
            raise ValueError(
                "fold family-common functional/application/input chains must align"
            )
        if common_functionals:
            sender_by_contrast = {
                functional.contrast_name: functional
                for functional in self.training.sender_functionals
            }
            sender_applications: dict[
                tuple[str, str, tuple[str, ...]], CommonSenderApplication
            ] = {}
            for (
                family_model,
                program_model,
                program_application,
                incremental_model,
                incremental_application,
                common_functional,
                common_application,
                common_binding,
            ) in zip(
                receiver_models,
                program_models,
                program_applications,
                incremental_models,
                incremental_applications,
                common_functionals,
                common_applications,
                common_bindings,
                strict=True,
            ):
                common_functional._require_intact()
                common_tables = common_application._validated_tables()
                common_binding._require_intact()
                expected_key = (
                    family_model.contrast_name,
                    family_model.receiver_family_artifact.receiver,
                    self.fold_id,
                )
                observed_key = (
                    common_functional.contrast_name,
                    common_functional.receiver,
                    common_functional.fold_id,
                )
                if observed_key != expected_key:
                    raise ValueError("family-common scoring keys do not match the fold")
                if (
                    common_functional.receiver_family.training_artifact_id
                    != family_model.receiver_family_artifact.training_artifact_id
                    or common_functional.receiver_program_artifact is None
                    or common_functional.receiver_program_artifact.training_artifact_id
                    != program_model.training_artifact_id
                    or common_functional.receiver_incremental_training_artifact_id
                    != incremental_model.training_artifact_id
                    or common_application.functional.family_common_functional_id
                    != common_functional.family_common_functional_id
                ):
                    raise ValueError(
                        "family-common parent lineage does not match the fold"
                    )
                tuning = incremental_model.penalty_tuning_artifact
                if tuning is None:
                    raise ValueError(
                        "family-common scoring is missing its tuning parent"
                    )
                sender_functional = sender_by_contrast.get(
                    common_functional.contrast_name
                )
                training_diagnostic = (
                    incremental_model.diagnostic_functional is not None
                    and tuning.is_oof_certified
                    and incremental_model.selected_penalty_candidate_id
                    == tuning.selected_candidate_id
                    and incremental_model.diagnostic_functional.penalty_candidate_id
                    == tuning.selected_candidate_id
                    and incremental_model.selected_resolved_penalty_id
                    == incremental_model.diagnostic_functional.resolved_penalty_id
                )
                expected_incremental_reason = (
                    None
                    if training_diagnostic
                    else incremental_model.diagnostic_reason_code
                    or tuning.reason_code
                    or incremental_model.reason_code
                    or "incremental_training_not_estimable"
                )
                if (
                    sender_functional is None
                    or common_functional.sender_functional.sender_functional_id
                    != sender_functional.sender_functional_id
                    or common_functional.tuning_manifest_id != tuning.tuning_id
                    or common_functional.selected_penalty_id
                    != incremental_model.selected_resolved_penalty_id
                    or common_functional.autonomous_program_resource_id
                    != incremental_model.autonomous_program_source_id
                    or common_functional.incremental_reason_code
                    != expected_incremental_reason
                ):
                    raise ValueError(
                        "family-common authoritative training lineage is incompatible"
                    )
                common_incremental = common_functional.incremental_functional
                diagnostic_incremental = incremental_model.diagnostic_functional
                common_incremental_id = (
                    None
                    if common_incremental is None
                    else common_incremental.incremental_functional_id
                )
                diagnostic_incremental_id = (
                    None
                    if diagnostic_incremental is None
                    else diagnostic_incremental.incremental_functional_id
                )
                if common_incremental_id is not None and (
                    common_incremental_id != diagnostic_incremental_id
                ):
                    raise ValueError(
                        "family-common incremental functional is incompatible"
                    )
                diagnostic_application = incremental_application.diagnostic_application
                application_diagnostic = (
                    training_diagnostic
                    and diagnostic_application is not None
                    and incremental_application.diagnostic_status == "observed"
                )
                expected_incremental_application_id = (
                    diagnostic_application.application_id
                    if application_diagnostic and diagnostic_application is not None
                    else None
                )
                if (
                    common_application.incremental_application_id
                    != expected_incremental_application_id
                ):
                    raise ValueError(
                        "family-common held-out application is incompatible"
                    )
                if (
                    common_application.receiver_program_application_id
                    != program_application.application_id
                ):
                    raise ValueError(
                        "family-common receiver-program application is incompatible"
                    )
                expected_heldout_reason = (
                    None
                    if application_diagnostic
                    else incremental_application.diagnostic_reason_code
                    or common_functional.incremental_reason_code
                    or incremental_application.reason_code
                    or "heldout_incremental_not_estimable"
                )
                if common_application.heldout_reason_code != expected_heldout_reason:
                    raise ValueError(
                        "family-common held-out reason does not match its parent"
                    )
                _, design_application = design_by_name[common_functional.contrast_name]
                _require_family_common_exact_coverage(
                    common_functional,
                    common_application,
                    design_application,
                    common_tables,
                )
                edge_evidence = _family_common_edge_evidence(
                    common_functional,
                    design_application,
                    self.application.availability.sample_interactions,
                )
                sender_key = (
                    common_functional.contrast_name,
                    common_functional.receiver,
                    common_functional.interaction_ids,
                )
                sender_application = sender_applications.get(sender_key)
                if sender_application is None:
                    sender_application = _completed_common_sender_application(
                        common_functional.sender_functional,
                        design_application,
                        self.application.availability.sample_interactions,
                        receiver=common_functional.receiver,
                        interaction_ids=common_functional.interaction_ids,
                    )
                    sender_applications[sender_key] = sender_application
                expected_binding = _FamilyCommonCrossFitBinding._from_workflow(
                    common_functional,
                    common_application,
                    self.application,
                    edge_evidence,
                    sender_application,
                )
                if common_binding != expected_binding:
                    raise ValueError(
                        "family-common inputs are not bound to held-out availability"
                    )
        object.__setattr__(self, "family_common_functionals", common_functionals)
        object.__setattr__(self, "family_common_applications", common_applications)
        object.__setattr__(self, "family_common_bindings", common_bindings)
        global_functionals = tuple(self.cross_receiver_common_functionals)
        global_applications = tuple(self.cross_receiver_common_applications)
        if bool(global_functionals) != bool(global_applications):
            raise ValueError(
                "cross-receiver common functionals and applications must align"
            )
        if global_functionals and not common_functionals:
            raise ValueError(
                "cross-receiver common scoring must track the family-common stage"
            )
        all_receivers_supported = all(
            record.status is ReceiverTrainingSupportStatus.OBSERVED
            for record in support_records
        )
        if common_functionals and all_receivers_supported and not global_functionals:
            raise ValueError(
                "complete receiver support requires cross-receiver common scoring"
            )
        if global_functionals and not all_receivers_supported:
            raise ValueError(
                "cross-receiver common scoring cannot shrink an incomplete "
                "receiver axis"
            )
        if global_functionals:
            if len(global_functionals) != len(encoders) or len(
                global_applications
            ) != len(encoders):
                raise ValueError(
                    "cross-receiver common scoring requires one parent per contrast"
                )
            children_by_contrast = {
                contrast_name: tuple(
                    sorted(
                        (
                            functional
                            for functional in common_functionals
                            if functional.contrast_name == contrast_name
                        ),
                        key=lambda functional: functional.receiver,
                    )
                )
                for contrast_name in {encoder.contrast.name for encoder in encoders}
            }
            applications_by_child = {
                application.functional.family_common_functional_id: application
                for application in common_applications
            }
            incremental_by_key = {
                (model.contrast_name, model.receiver): model
                for model in incremental_models
            }
            global_pairs = tuple(
                sorted(
                    zip(global_functionals, global_applications, strict=True),
                    key=lambda pair: pair[0].contrast_manifest_id,
                )
            )
            for global_functional, global_application in global_pairs:
                children = children_by_contrast.get(global_functional.contrast_name)
                if children is None:
                    raise ValueError(
                        "cross-receiver common parent references an unknown contrast"
                    )
                expected_functional = fit_cross_receiver_common_scoring_functional(
                    children,
                    planned_receiver_ids=support_receivers,
                    gain_calibration_artifacts={
                        child.receiver: incremental_by_key[
                            (child.contrast_name, child.receiver)
                        ].gain_calibration_artifact
                        for child in children
                    },
                    spec=global_functional.spec,
                )
                if (
                    expected_functional.global_common_functional_id
                    != global_functional.global_common_functional_id
                ):
                    raise ValueError(
                        "cross-receiver common functional lineage is incompatible"
                    )
                child_applications = tuple(
                    applications_by_child[functional.family_common_functional_id]
                    for functional in children
                )
                source_sender_applications = tuple(
                    sender_applications[
                        (
                            functional.contrast_name,
                            functional.receiver,
                            functional.interaction_ids,
                        )
                    ]
                    for functional in children
                )
                expected_application = apply_cross_receiver_common_scoring_functional(
                    global_functional,
                    child_applications,
                    source_sender_applications,
                )
                if (
                    expected_application.application_id
                    != global_application.application_id
                    or expected_application.lr_scores_digest
                    != global_application.lr_scores_digest
                    or expected_application.sender_scores_digest
                    != global_application.sender_scores_digest
                ):
                    raise ValueError(
                        "cross-receiver common application lineage is incompatible"
                    )
            global_functionals = tuple(pair[0] for pair in global_pairs)
            global_applications = tuple(pair[1] for pair in global_pairs)
        object.__setattr__(
            self, "cross_receiver_common_functionals", global_functionals
        )
        object.__setattr__(
            self, "cross_receiver_common_applications", global_applications
        )
        record_validation(self)


def _receiver_scoring_registry(
    spec: CrossFitSpec,
    folds: tuple[CrossFitFoldArtifacts, ...],
) -> ScoringCollectionDocument:
    """Derive exact receiver registries from producer-owned fold children."""

    collections: list[ScoringCollectionManifest] = []
    plans: list[PlannedScoringCollectionManifest] = []
    for fold in folds:
        support_by_receiver = {
            record.receiver_id: record for record in fold.receiver_training_support
        }
        planned_receivers = tuple(support_by_receiver)
        family_models = {
            (model.contrast_name, model.receiver_family_artifact.receiver): model
            for model in fold.receiver_family_models
        }
        incremental_models = {
            (model.contrast_name, model.receiver): model
            for model in fold.receiver_incremental_models
        }
        functionals = {
            (functional.contrast_name, functional.receiver): functional
            for functional in fold.family_common_functionals
        }
        for contrast in spec.contrasts:
            children: list[ReceiverScoringFunctionalManifest] = []
            for receiver in planned_receivers:
                support = support_by_receiver[receiver]
                key = (contrast.name, receiver)
                family_model = family_models.get(key)
                incremental_model = incremental_models.get(key)
                if support.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE:
                    if family_model is not None or incremental_model is not None:
                        raise ValueError(
                            "outer-training-absent receiver cannot have model parents"
                        )
                    if support.reason_code is None:  # pragma: no cover - invariant
                        raise RuntimeError(
                            "not-estimable receiver support requires a reason"
                        )
                    children.append(
                        ReceiverScoringFunctionalManifest.training_not_estimable(
                            receiver=receiver,
                            filter_universe_id=(
                                fold.training.frozen_interaction_universe.filter_universe_id
                            ),
                            receiver_training_support_id=(support.support_record_id),
                            reason_code=support.reason_code,
                        )
                    )
                    continue
                if family_model is None or incremental_model is None:
                    raise ValueError(
                        "planned receiver registry is missing an authoritative model"
                    )
                receiver_family = family_model.receiver_family_artifact
                if (
                    incremental_model.receiver_family_training_artifact_id
                    != receiver_family.training_artifact_id
                    or receiver_family.filter_universe_id
                    != fold.training.frozen_interaction_universe.filter_universe_id
                ):
                    raise ValueError(
                        "planned receiver registry model universe lineage is invalid"
                    )
                functional = functionals.get(key)
                if functional is None:
                    if spec.penalty_tuning_spec is not None:
                        raise ValueError(
                            "configured family-common scoring did not exactly cover "
                            "planned receivers"
                        )
                    child = ReceiverScoringFunctionalManifest.not_produced(
                        receiver=receiver,
                        receiver_family_model_id=receiver_family.training_artifact_id,
                        receiver_incremental_model_id=(
                            incremental_model.training_artifact_id
                        ),
                        filter_universe_id=receiver_family.filter_universe_id,
                        reason_code=(
                            "family_common_functional_not_requested_without_"
                            "penalty_tuning"
                        ),
                        receiver_training_support_id=support.support_record_id,
                        receiver_training_support_status=support.status.value,
                        receiver_training_support_reason_code=support.reason_code,
                    )
                else:
                    if (
                        functional.receiver_family.training_artifact_id
                        != receiver_family.training_artifact_id
                        or functional.receiver_incremental_training_artifact_id
                        != incremental_model.training_artifact_id
                        or functional.filter_universe_id
                        != receiver_family.filter_universe_id
                    ):
                        raise ValueError(
                            "planned receiver registry functional lineage is invalid"
                        )
                    functional_status = (
                        "observed"
                        if functional.incremental_functional is not None
                        else "not_estimable"
                    )
                    child = ReceiverScoringFunctionalManifest.registered(
                        receiver=receiver,
                        receiver_family_model_id=receiver_family.training_artifact_id,
                        receiver_incremental_model_id=(
                            incremental_model.training_artifact_id
                        ),
                        filter_universe_id=functional.filter_universe_id,
                        scoring_functional_id=(functional.family_common_functional_id),
                        score_version=functional.score_version,
                        functional_status=functional_status,
                        reason_code=(
                            None
                            if functional_status == "observed"
                            else functional.incremental_reason_code
                        ),
                        receiver_training_support_id=support.support_record_id,
                        receiver_training_support_status=support.status.value,
                        receiver_training_support_reason_code=support.reason_code,
                    )
                children.append(child)
            collection = ScoringCollectionManifest.planned_receiver_registry(
                contrast=contrast.name,
                repeat_id=spec.repeat_id,
                fold_id=fold.fold_id,
                planned_receivers=planned_receivers,
                children=tuple(children),
            )
            collections.append(collection)
            plans.append(
                PlannedScoringCollectionManifest.from_collection(
                    collection,
                    filter_universe_id=(
                        fold.training.frozen_interaction_universe.filter_universe_id
                    ),
                )
            )
    return ScoringCollectionDocument(
        collections=tuple(collections), planned_collections=tuple(plans)
    )


def _require_receiver_family_on_root_axis(
    artifact: ReceiverFamilyTrainingArtifact,
    universe: FrozenReceiverFamilyOpportunityUniverse,
) -> None:
    """Reject any outer-fold family parent outside the frozen run-root axis."""

    if not isinstance(artifact, ReceiverFamilyTrainingArtifact):
        raise TypeError("artifact must be a ReceiverFamilyTrainingArtifact")
    if not isinstance(universe, FrozenReceiverFamilyOpportunityUniverse):
        raise TypeError("universe must be a FrozenReceiverFamilyOpportunityUniverse")
    artifact._require_producer_owned()
    universe._require_intact()
    basis = artifact.family_basis
    if (
        artifact.receiver not in universe.receiver_ids
        or artifact.prior_manifest_digest != universe.prior_manifest_digest
        or artifact.source_basis.feature_ids != universe.feature_ids
        or artifact.source_basis.driver_ids != universe.driver_ids
        or basis.feature_ids != universe.feature_ids
        or basis.driver_ids != universe.driver_ids
        or basis.strict_cosine_threshold != universe.cosine_threshold
        or basis.family_definitions != universe.family_definitions
        or basis.family_ids != universe.family_ids
        or artifact.frozen_family_axis_parent_id is not None
    ):
        raise ContractError(
            "Outer-fold receiver family axis differs from the frozen run root",
            code="crossfit_root_family_axis_mismatch",
            field="receiver_family_models",
            remediation=(
                "Rerun subject cross-fit from the exact target prior, root feature "
                "axis, receiver universe, and family threshold"
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class CrossFitArtifacts:
    """Partial artifacts with exact sender-stage OOF coverage verification."""

    spec: CrossFitSpec
    root_input_identity: SanitizedRawInputIdentity
    receiver_universe: FrozenReceiverUniverse
    receiver_family_opportunity_universe: FrozenReceiverFamilyOpportunityUniverse
    directional_lr_hypothesis_universe: FrozenReceiverFamilyLRHypothesisUniverse | None
    fold_plan: SubjectFoldPlan
    folds: tuple[CrossFitFoldArtifacts, ...]
    _oof_coverage: pd.DataFrame = field(repr=False)
    _oof_receiver_coverage: pd.DataFrame = field(repr=False)
    _oof_sender_assignments: pd.DataFrame = field(repr=False)
    coverage_audit: OOFCoverageAudit
    coverage_table_digest: str = field(init=False)
    receiver_coverage_table_digest: str = field(init=False)
    sender_assignment_table_digest: str = field(init=False)
    receiver_coverage_audit_id: str = field(init=False)
    receiver_scoring_registry_id: str = field(init=False)
    certification_status: str
    crossfit_id: str = field(init=False)
    _producer_marker: str = field(init=False, repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "CrossFitArtifacts are producer-owned; use run_subject_crossfit()"
        )

    @classmethod
    def _from_workflow(
        cls,
        *,
        spec: CrossFitSpec,
        root_input_identity: SanitizedRawInputIdentity,
        receiver_universe: FrozenReceiverUniverse,
        receiver_family_opportunity_universe: (FrozenReceiverFamilyOpportunityUniverse),
        directional_lr_hypothesis_universe: (
            FrozenReceiverFamilyLRHypothesisUniverse | None
        ) = None,
        fold_plan: SubjectFoldPlan,
        folds: tuple[CrossFitFoldArtifacts, ...],
        oof_coverage: pd.DataFrame,
        oof_receiver_coverage: pd.DataFrame,
        oof_sender_assignments: pd.DataFrame,
        coverage_audit: OOFCoverageAudit,
    ) -> CrossFitArtifacts:
        with _receiver_family_validation_scope(), validation_scope():
            self = object.__new__(cls)
            values: dict[str, object] = {
                "spec": spec,
                "root_input_identity": root_input_identity,
                "receiver_universe": receiver_universe,
                "receiver_family_opportunity_universe": (
                    receiver_family_opportunity_universe
                ),
                "directional_lr_hypothesis_universe": (
                    directional_lr_hypothesis_universe
                ),
                "fold_plan": fold_plan,
                "folds": folds,
                "_oof_coverage": oof_coverage,
                "_oof_receiver_coverage": oof_receiver_coverage,
                "_oof_sender_assignments": oof_sender_assignments,
                "coverage_audit": coverage_audit,
                "certification_status": _STAGE_STATUS,
                "_producer_marker": _PRODUCER_MARKER,
            }
            for name, value in values.items():
                object.__setattr__(self, name, value)
            self.__post_init__()
            return self

    def __post_init__(self) -> None:
        if self._producer_marker != _PRODUCER_MARKER:
            raise TypeError("CrossFitArtifacts were not produced by this workflow")
        if self.certification_status != _STAGE_STATUS:
            raise ValueError("partial cross-fit artifacts cannot claim another status")
        if not isinstance(self.spec, CrossFitSpec):
            raise TypeError("spec must be a CrossFitSpec")
        self.spec._require_intact()
        if not isinstance(self.root_input_identity, SanitizedRawInputIdentity):
            raise TypeError("root_input_identity must be a SanitizedRawInputIdentity")
        self.root_input_identity._require_intact()
        if not isinstance(self.receiver_universe, FrozenReceiverUniverse):
            raise TypeError("receiver_universe must be a FrozenReceiverUniverse")
        self.receiver_universe._require_intact()
        if (
            self.receiver_universe.root_input_identity_id
            != self.root_input_identity.identity_id
            or self.receiver_universe.root_input_digest
            != self.root_input_identity.input_digest
            or self.receiver_universe.root_config_digest
            != self.root_input_identity.config_digest
        ):
            raise ValueError("receiver universe does not match the root input identity")
        if (
            self.spec.predeclared_receiver_ids is not None
            and self.receiver_universe.receiver_ids
            != self.spec.predeclared_receiver_ids
        ):
            raise ValueError(
                "receiver universe does not match the predeclared receiver policy"
            )
        if not isinstance(
            self.receiver_family_opportunity_universe,
            FrozenReceiverFamilyOpportunityUniverse,
        ):
            raise TypeError(
                "receiver_family_opportunity_universe must be a "
                "FrozenReceiverFamilyOpportunityUniverse"
            )
        receiver_family_universe = self.receiver_family_opportunity_universe
        receiver_family_universe._require_intact()
        if (
            receiver_family_universe.receiver_ids != self.receiver_universe.receiver_ids
            or receiver_family_universe.receiver_universe_id
            != self.receiver_universe.universe_id
            or receiver_family_universe.receiver_axis_id
            != self.receiver_universe.receiver_axis_id
            or receiver_family_universe.root_input_identity_id
            != self.root_input_identity.identity_id
            or receiver_family_universe.root_input_digest
            != self.root_input_identity.input_digest
            or receiver_family_universe.cosine_threshold
            != self.spec.family_cosine_threshold
        ):
            raise ValueError(
                "receiver-family opportunity universe does not match the run root"
            )
        directional_lr_universe = self.directional_lr_hypothesis_universe
        if self.spec.directional_pairs:
            if not isinstance(
                directional_lr_universe,
                FrozenReceiverFamilyLRHypothesisUniverse,
            ):
                raise ValueError(
                    "directional cross-fit is missing its pre-fit LR hypothesis "
                    "universe"
                )
            directional_lr_universe._require_intact()
            if (
                directional_lr_universe.receiver_ids
                != self.receiver_universe.receiver_ids
                or directional_lr_universe.receiver_universe_id
                != self.receiver_universe.universe_id
                or directional_lr_universe.receiver_axis_id
                != self.receiver_universe.receiver_axis_id
                or directional_lr_universe.root_input_identity_id
                != self.root_input_identity.identity_id
                or directional_lr_universe.root_input_digest
                != self.root_input_identity.input_digest
                or directional_lr_universe.receiver_family_universe_id
                != receiver_family_universe.universe_id
                or directional_lr_universe.feature_axis_id
                != receiver_family_universe.feature_axis_id
                or directional_lr_universe.family_axis_id
                != receiver_family_universe.family_axis_id
                or directional_lr_universe._receiver_family_universe
                is not receiver_family_universe
            ):
                raise ValueError(
                    "directional LR hypothesis universe does not match the run root"
                )
        elif directional_lr_universe is not None:
            raise ValueError(
                "non-directional cross-fit cannot own a directional LR universe"
            )
        if not isinstance(self.fold_plan, SubjectFoldPlan):
            raise TypeError("fold_plan must be a SubjectFoldPlan")
        self.fold_plan._require_intact()
        if self.fold_plan.repeat_id != self.spec.repeat_id:
            raise ValueError("fold plan repeat_id does not match the cross-fit spec")
        expected_partition_lineage = _outer_fold_partition_lineage(self.spec)
        observed_partition_lineage = self.fold_plan.partition_seed_lineage
        if (
            None
            if observed_partition_lineage is None
            else observed_partition_lineage.to_dict()
        ) != (
            None
            if expected_partition_lineage is None
            else expected_partition_lineage.to_dict()
        ):
            raise ValueError(
                "fold plan partition seed lineage does not match the cross-fit spec"
            )
        if self.root_input_identity.subject_ids != self.fold_plan.subject_ids:
            raise ValueError("root input subjects do not match the fold plan")
        if self.fold_plan.allowed_n_splits != self.spec.allowed_n_splits:
            raise ValueError(
                "fold plan allowed_n_splits do not match the cross-fit spec"
            )
        folds = tuple(self.folds)
        if any(not isinstance(item, CrossFitFoldArtifacts) for item in folds):
            raise TypeError("folds must contain CrossFitFoldArtifacts")
        for item in folds:
            item.__post_init__()
        by_id = {fold.fold_id: fold for fold in self.fold_plan.folds}
        if len(folds) != len(by_id) or {item.fold_id for item in folds} != set(by_id):
            raise ValueError(
                "cross-fit fold artifacts must exactly cover the fold plan"
            )
        for item in folds:
            manifest = by_id[item.fold_id]
            if tuple(
                record.receiver_id for record in item.receiver_training_support
            ) != self.receiver_universe.receiver_ids or any(
                record.receiver_universe_id != self.receiver_universe.universe_id
                for record in item.receiver_training_support
            ):
                raise ValueError(
                    "fold receiver support does not cover the run-level universe"
                )
            if item.training.config.digest != self.root_input_identity.config_digest:
                raise ValueError("fold config does not match the root input identity")
            if (
                item.training.target_prior_content_id
                != receiver_family_universe.prior_content_id
                or item.training.target_prior.resource_id
                != receiver_family_universe.prior_resource_id
                or item.training.target_prior.version
                != receiver_family_universe.prior_version
                or item.training.target_prior.manifest_digest
                != receiver_family_universe.prior_manifest_digest
            ):
                raise ValueError(
                    "fold target prior does not match the root family universe"
                )
            family_parent_ids: dict[str, set[str]] = {}
            for scoring_model in item.receiver_family_models:
                receiver_family = scoring_model.receiver_family_artifact
                _require_receiver_family_on_root_axis(
                    receiver_family,
                    receiver_family_universe,
                )
                family_parent_ids.setdefault(receiver_family.receiver, set()).add(
                    receiver_family.training_artifact_id
                )
            supported_receivers = {
                record.receiver_id
                for record in item.receiver_training_support
                if record.status is ReceiverTrainingSupportStatus.OBSERVED
            }
            if set(family_parent_ids) != supported_receivers or any(
                len(parent_ids) != 1 for parent_ids in family_parent_ids.values()
            ):
                raise ValueError(
                    "fold receiver-family parents do not exactly cover supported "
                    "run receivers"
                )
            if directional_lr_universe is not None:
                expected_driver_mapping = tuple(
                    sorted(
                        (membership.interaction_id, membership.driver_id)
                        for membership in directional_lr_universe.memberships
                    )
                )
                if (
                    item.training.resource_bundle_content_id
                    != directional_lr_universe.resource_bundle_content_id
                    or item.training.target_prior_content_id
                    != directional_lr_universe.target_prior_content_id
                    or item.training.resource_bundle.manifest_digest
                    != directional_lr_universe.resource_manifest_digest
                    or item.training.target_prior.manifest_digest
                    != directional_lr_universe.target_prior_manifest_digest
                ):
                    raise ValueError(
                        "fold resources do not match the pre-fit directional LR "
                        "hypothesis universe"
                    )
                for scoring_model in item.receiver_family_models:
                    receiver_family = scoring_model.receiver_family_artifact
                    feature_axis_id = stable_id(
                        "root_feature_axis",
                        {"feature_ids": list(receiver_family.family_basis.feature_ids)},
                        schema_version="1",
                    )
                    if (
                        receiver_family.family_basis.family_ids
                        != directional_lr_universe.family_ids
                        or receiver_family.driver_by_interaction
                        != expected_driver_mapping
                        or feature_axis_id != directional_lr_universe.feature_axis_id
                    ):
                        raise ValueError(
                            "fold receiver-family model does not preserve the "
                            "pre-fit directional LR axis"
                        )
            if item.training.training_input_digest != (
                self.root_input_identity.scope_digest(manifest.train_subject_ids)
            ):
                raise ValueError(
                    "fold training input does not derive from the root identity"
                )
            if item.application.heldout_input_digest != (
                self.root_input_identity.scope_digest(manifest.test_subject_ids)
            ):
                raise ValueError(
                    "fold heldout input does not derive from the root identity"
                )
            if item.training.training_subject_ids != manifest.train_subject_ids:
                raise ValueError(
                    "training artifact subjects do not match the fold plan"
                )
            if item.application.heldout_subject_ids != manifest.test_subject_ids:
                raise ValueError("application subjects do not match the fold plan")
            observed_contrasts = {
                _contrast_id(encoder.contrast) for encoder in item.design_encoders
            }
            if observed_contrasts != set(manifest.contrast_ids):
                raise ValueError("design encoders do not match planned fold contrasts")
            expected_autonomous_id = (
                None
                if self.spec.autonomous_program_resource is None
                else self.spec.autonomous_program_resource.artifact_id
            )
            if any(
                model.autonomous_program_resource_id != expected_autonomous_id
                or model.autonomous_program_verification_status
                != _autonomous_verification_status(
                    self.spec.autonomous_program_resource,
                    self.spec.latent_nuisance_spec,
                    model.latent_nuisance_artifact,
                )
                for model in item.receiver_incremental_models
            ):
                raise ValueError(
                    "receiver incremental models do not match the cross-fit "
                    "autonomous program resource"
                )
            expected_tuning_spec_id = (
                None
                if self.spec.penalty_tuning_spec is None
                else self.spec.penalty_tuning_spec.spec_id
            )
            if any(
                model.penalty_tuning_spec_id != expected_tuning_spec_id
                for model in item.receiver_incremental_models
            ):
                raise ValueError(
                    "receiver incremental models do not match the cross-fit "
                    "penalty tuning policy"
                )
            expected_latent_nuisance_spec_id = (
                None
                if self.spec.latent_nuisance_spec is None
                else self.spec.latent_nuisance_spec.spec_id
            )
            if any(
                model.latent_nuisance_spec_id != expected_latent_nuisance_spec_id
                for model in item.receiver_incremental_models
            ):
                raise ValueError(
                    "receiver incremental models do not match the cross-fit "
                    "latent nuisance policy"
                )
            expected_gain_calibration_spec_id = (
                None
                if self.spec.gain_calibration_spec is None
                else self.spec.gain_calibration_spec.spec_id
            )
            if any(
                model.gain_calibration_spec_id != expected_gain_calibration_spec_id
                for model in item.receiver_incremental_models
            ):
                raise ValueError(
                    "receiver incremental models do not match the cross-fit gain "
                    "calibration policy"
                )
            inner_partition_base = _inner_tuning_partition_base_lineage(
                self.spec,
                manifest,
            )
            contrast_by_name = {
                contrast.name: contrast for contrast in self.spec.contrasts
            }
            for model in item.receiver_incremental_models:
                inner_plan = model.inner_fold_plan
                if inner_plan is None:
                    continue
                contrast = contrast_by_name.get(model.contrast_name)
                if contrast is None:
                    raise ValueError(
                        "receiver incremental model contrast does not match the "
                        "cross-fit specification"
                    )
                expected_inner_lineage = _receiver_inner_partition_lineage(
                    inner_partition_base,
                    receiver=model.receiver,
                    contrast_id=_contrast_id(contrast),
                )
                observed_inner_lineage = inner_plan.partition_seed_lineage
                if (
                    None
                    if observed_inner_lineage is None
                    else observed_inner_lineage.to_dict()
                ) != (
                    None
                    if expected_inner_lineage is None
                    else expected_inner_lineage.to_dict()
                ):
                    raise ValueError(
                        "inner tuning partition seed lineage does not match the "
                        "cross-fit outer partition"
                    )
            expected_common_count = (
                0
                if self.spec.penalty_tuning_spec is None
                else len(item.receiver_incremental_models)
            )
            if (
                len(item.family_common_functionals) != expected_common_count
                or len(item.family_common_applications) != expected_common_count
                or len(item.family_common_bindings) != expected_common_count
            ):
                raise ValueError(
                    "family-common scoring chains do not match the cross-fit policy"
                )
            expected_directional_keys = {
                (pair.pair_spec_id, receiver)
                for pair in self.spec.directional_pairs
                for receiver in item.training.cell_type_ids
            }
            observed_directional_keys = {
                (binding.pair_spec_id, binding.receiver)
                for binding in item.directional_response_bindings
            }
            if observed_directional_keys != expected_directional_keys:
                raise ValueError(
                    "directional response bindings do not exactly cover the "
                    "cross-fit pair/receiver policy"
                )
        coverage = self._oof_coverage.copy(deep=True)
        if tuple(coverage.columns) != _COVERAGE_COLUMNS:
            raise ValueError("oof_coverage columns do not match the stage contract")
        if set(coverage["functional_status"].astype(str)) != {"out_of_fold"}:
            raise ValueError("stage coverage must contain only out-of-fold rows")
        assignments = self._oof_sender_assignments.copy(deep=True)
        expected_assignment_columns = (
            *COMMON_SENDER_APPLICATION_COLUMNS,
            *_ASSIGNMENT_PROVENANCE_COLUMNS,
        )
        if tuple(assignments.columns) != expected_assignment_columns:
            raise ValueError(
                "oof_sender_assignments columns do not match the stage contract"
            )
        if not assignments.empty:
            if set(assignments["functional_status"].astype(str)) != {"out_of_fold"}:
                raise ValueError("sender assignment rows must be out of fold")
            if assignments["sender_application_id"].duplicated().any():
                raise ValueError("sender application IDs must be globally unique")
        if self.coverage_audit.fold_plan_id != self.fold_plan.plan_id:
            raise ValueError("coverage audit does not match the fold plan")
        self.coverage_audit._require_intact()
        if self.coverage_audit.n_rows != len(coverage):
            raise ValueError("coverage audit row count does not match its table")
        contrast_contexts = {
            _contrast_id(contrast): tuple(contrast.weights)
            for contrast in self.spec.contrasts
        }
        repeated_audit = validate_oof_subject_coverage(
            coverage,
            self.fold_plan,
            contrast_contexts=contrast_contexts,
            context_column="contrast_context",
            scoring_function_column="sender_functional_id",
            model_manifest_column="training_artifact_id",
        )
        if repeated_audit.audit_id != self.coverage_audit.audit_id:
            raise ValueError("coverage audit identity does not match its table")
        fold_artifacts = {item.fold_id: item for item in folds}
        receiver_coverage = self._oof_receiver_coverage.copy(deep=True)
        if tuple(receiver_coverage.columns) != _RECEIVER_COVERAGE_COLUMNS:
            raise ValueError(
                "oof_receiver_coverage columns do not match the stage contract"
            )
        grain = ["fold_id", "sample_id", "contrast_id", "receiver"]
        if receiver_coverage.duplicated(grain).any():
            raise ValueError("receiver coverage contains duplicate expected-grain rows")
        expected_grain: set[tuple[str, str, str, str]] = set()
        expected_parent_by_grain: dict[
            tuple[str, str, str, str], tuple[object, ...]
        ] = {}
        for item in folds:
            support_by_receiver = {
                record.receiver_id: record for record in item.receiver_training_support
            }
            for (
                response,
                precision,
                incremental_model,
                response_application,
                incremental_application,
            ) in zip(
                item.receiver_responses,
                item.response_precisions,
                item.receiver_incremental_models,
                item.receiver_response_applications,
                item.receiver_incremental_applications,
                strict=True,
            ):
                support = support_by_receiver[response.receiver]
                if support.status is not ReceiverTrainingSupportStatus.OBSERVED:
                    raise ValueError(
                        "receiver model is bound to non-observed training support"
                    )
                encoder, design_application = next(
                    (encoder, application)
                    for encoder, application in zip(
                        item.design_encoders,
                        item.design_applications,
                        strict=True,
                    )
                    if encoder.contrast.name == response.contrast_name
                )
                contrast_id = _contrast_id(encoder.contrast)
                valid_context_ids = {
                    node_context_fields(node, encoder.context_keys)[0]
                    for node in encoder.contrast.weights
                }
                for sample_id, subject_id, context_id in zip(
                    design_application.sample_ids,
                    design_application.sample_subject_ids,
                    design_application.sample_context_ids,
                    strict=True,
                ):
                    if context_id not in valid_context_ids:
                        continue
                    key = (item.fold_id, sample_id, contrast_id, response.receiver)
                    expected_grain.add(key)
                    expected_parent_by_grain[key] = (
                        subject_id,
                        canonical_json(
                            next(
                                node
                                for node in encoder.contrast.weights
                                if node_context_fields(node, encoder.context_keys)[0]
                                == context_id
                            )
                        ),
                        support.receiver_universe_id,
                        support.support_record_id,
                        support.status.value,
                        support.reason_code,
                        response.artifact_id,
                        precision.precision_transform_id,
                        incremental_model.training_artifact_id,
                        response_application.application_id,
                        design_application.application_id,
                        incremental_application.application_id,
                        incremental_application.diagnostic_status,
                        incremental_application.diagnostic_reason_code,
                        incremental_application.official_incremental_status,
                        incremental_application.reason_code,
                    )
            for support in item.receiver_training_support:
                if support.status is not ReceiverTrainingSupportStatus.NOT_ESTIMABLE:
                    continue
                for encoder, design_application in zip(
                    item.design_encoders,
                    item.design_applications,
                    strict=True,
                ):
                    contrast_id = _contrast_id(encoder.contrast)
                    context_by_id = {
                        node_context_fields(node, encoder.context_keys)[0]: node
                        for node in encoder.contrast.weights
                    }
                    for sample_id, subject_id, context_id in zip(
                        design_application.sample_ids,
                        design_application.sample_subject_ids,
                        design_application.sample_context_ids,
                        strict=True,
                    ):
                        if context_id not in context_by_id:
                            continue
                        key = (
                            item.fold_id,
                            sample_id,
                            contrast_id,
                            support.receiver_id,
                        )
                        expected_grain.add(key)
                        expected_parent_by_grain[key] = (
                            subject_id,
                            canonical_json(context_by_id[context_id]),
                            support.receiver_universe_id,
                            support.support_record_id,
                            support.status.value,
                            support.reason_code,
                            None,
                            None,
                            None,
                            None,
                            design_application.application_id,
                            None,
                            "not_estimable",
                            support.reason_code,
                            "not_estimable",
                            support.reason_code,
                        )
        observed_grain = {
            tuple(map(str, values))
            for values in receiver_coverage.loc[:, grain].itertuples(
                index=False, name=None
            )
        }
        if observed_grain != expected_grain:
            raise ValueError("receiver coverage does not match the planned exact grain")
        for row in receiver_coverage.itertuples(index=False):
            key = (
                str(row.fold_id),
                str(row.sample_id),
                str(row.contrast_id),
                str(row.receiver),
            )
            expected = expected_parent_by_grain[key]
            observed = (
                str(row.subject_id),
                canonical_json(row.contrast_context),
                str(row.receiver_universe_id),
                str(row.receiver_training_support_id),
                str(row.receiver_training_support_status),
                _optional_table_text(row.receiver_training_support_reason_code),
                _optional_table_text(row.response_artifact_id),
                _optional_table_text(row.precision_transform_id),
                _optional_table_text(row.incremental_training_artifact_id),
                _optional_table_text(row.response_application_id),
                str(row.design_application_id),
                _optional_table_text(row.incremental_application_id),
                str(row.diagnostic_status),
                _optional_table_text(row.diagnostic_reason_code),
                str(row.official_incremental_status),
                _optional_table_text(row.reason_code),
            )
            if observed != expected:
                raise ValueError(
                    "receiver coverage parent lineage is incompatible: "
                    f"{observed!r} != {expected!r}"
                )
            manifest = by_id[str(row.fold_id)]
            if (
                str(row.subject_id) not in manifest.test_subject_ids
                or str(row.subject_id) in manifest.train_subject_ids
                or str(row.stage) != _RECEIVER_STAGE_NAME
            ):
                raise ValueError(
                    "receiver coverage status or held-out scope is invalid"
                )
        receiver_audit_id = _receiver_coverage_identity(
            receiver_coverage, fold_plan_id=self.fold_plan.plan_id
        )
        for row in assignments.itertuples(index=False):
            fold_id = str(row.fold_id)
            if fold_id not in fold_artifacts:
                raise ValueError("sender assignment references an unknown fold")
            item = fold_artifacts[fold_id]
            subject_id = str(row.subject_id)
            manifest = by_id[fold_id]
            if (
                subject_id not in manifest.test_subject_ids
                or subject_id in manifest.train_subject_ids
            ):
                raise ValueError("sender assignment is not held out in its fold")
            if str(row.training_artifact_id) != item.training.training_artifact_id:
                raise ValueError(
                    "sender assignment training artifact does not match its fold"
                )
            functionals = {
                functional.sender_functional_id: _contrast_id(functional.contrast)
                for functional in item.training.sender_functionals
            }
            functional_id = str(row.sender_functional_id)
            if functionals.get(functional_id) != str(row.contrast_id):
                raise ValueError(
                    "sender assignment functional does not match its fold contrast"
                )
        for row in coverage.itertuples(index=False):
            item = fold_artifacts[str(row.fold_id)]
            encoders = {
                _contrast_id(encoder.contrast): (
                    encoder,
                    application,
                )
                for encoder, application in zip(
                    item.design_encoders,
                    item.design_applications,
                    strict=True,
                )
            }
            pair = encoders.get(str(row.contrast_id))
            if pair is None:
                raise ValueError("coverage row references an unknown design contrast")
            encoder, design_application = pair
            if (
                str(row.design_encoder_id) != encoder.encoder_id
                or str(row.design_status) != design_application.status
            ):
                raise ValueError("coverage row design provenance is incompatible")
        coverage_table_digest = _table_digest("oof_coverage", coverage)
        receiver_coverage_table_digest = _table_digest(
            "oof_receiver_coverage", receiver_coverage
        )
        sender_assignment_table_digest = _table_digest(
            "oof_sender_assignments", assignments
        )
        receiver_registry = _receiver_scoring_registry(self.spec, folds)
        if receiver_registry.registry_id is None:  # pragma: no cover - v2 constructor
            raise RuntimeError("cross-fit receiver registry is missing its identity")
        payload = {
            "coverage_audit_id": self.coverage_audit.audit_id,
            "coverage_table_digest": coverage_table_digest,
            "receiver_coverage_audit_id": receiver_audit_id,
            "receiver_coverage_table_digest": receiver_coverage_table_digest,
            "receiver_scoring_registry_id": receiver_registry.registry_id,
            "receiver_universe_id": self.receiver_universe.universe_id,
            "receiver_axis_id": self.receiver_universe.receiver_axis_id,
            "receiver_family_opportunity_universe_id": (
                receiver_family_universe.universe_id
            ),
            "family_axis_id": receiver_family_universe.family_axis_id,
            "receiver_family_opportunity_axis_id": (
                receiver_family_universe.opportunity_axis_id
            ),
            **(
                {
                    "directional_lr_hypothesis_universe_id": (
                        directional_lr_universe.universe_id
                    )
                }
                if directional_lr_universe is not None
                else {}
            ),
            "fold_applications": [
                {
                    "fold_id": item.fold_id,
                    "heldout_input_digest": item.application.heldout_input_digest,
                    "heldout_excluded_cell_type_ids": list(
                        item.application.excluded_cell_type_ids
                    ),
                    "training_application_id": item.application.application_id,
                    "training_artifact_id": item.training.training_artifact_id,
                    "receiver_training_support_ids": [
                        record.support_record_id
                        for record in item.receiver_training_support
                    ],
                    "design_encoder_ids": [
                        encoder.encoder_id for encoder in item.design_encoders
                    ],
                    "design_application_statuses": [
                        application.status for application in item.design_applications
                    ],
                    "design_application_ids": [
                        application.application_id
                        for application in item.design_applications
                    ],
                    "receiver_family_training_artifact_ids": [
                        model.training_artifact_id
                        for model in item.receiver_family_models
                    ],
                    "receiver_family_application_statuses": [
                        application.application_status
                        for application in item.receiver_family_applications
                    ],
                    "receiver_family_application_ids": [
                        application.application_id
                        for application in item.receiver_family_applications
                    ],
                    "receiver_program_training_ids": [
                        model.training_artifact_id
                        for model in item.receiver_program_models
                    ],
                    "receiver_program_application_ids": [
                        application.application_id
                        for application in item.receiver_program_applications
                    ],
                    "receiver_response_ids": [
                        response.artifact_id for response in item.receiver_responses
                    ],
                    "response_precision_ids": [
                        precision.precision_transform_id
                        for precision in item.response_precisions
                    ],
                    "receiver_incremental_training_ids": [
                        model.training_artifact_id
                        for model in item.receiver_incremental_models
                    ],
                    "receiver_response_application_ids": [
                        application.application_id
                        for application in item.receiver_response_applications
                    ],
                    "receiver_incremental_application_ids": [
                        application.application_id
                        for application in item.receiver_incremental_applications
                    ],
                    **(
                        {
                            "directional_response_binding_ids": [
                                binding.binding_id
                                for binding in item.directional_response_bindings
                            ]
                        }
                        if item.directional_response_bindings
                        else {}
                    ),
                    "family_common_functional_ids": [
                        functional.family_common_functional_id
                        for functional in item.family_common_functionals
                    ],
                    "family_common_application_ids": [
                        application.application_id
                        for application in item.family_common_applications
                    ],
                    "family_common_binding_ids": [
                        binding.binding_id for binding in item.family_common_bindings
                    ],
                    "cross_receiver_common_functional_ids": [
                        functional.global_common_functional_id
                        for functional in item.cross_receiver_common_functionals
                    ],
                    "cross_receiver_common_application_ids": [
                        application.application_id
                        for application in item.cross_receiver_common_applications
                    ],
                }
                for item in sorted(folds, key=lambda value: value.fold_id)
            ],
            "fold_plan_id": self.fold_plan.plan_id,
            "root_input_identity_id": self.root_input_identity.identity_id,
            "root_input_digest": self.root_input_identity.input_digest,
            "sender_application_ids": sorted(
                assignments["sender_application_id"].astype(str).tolist()
            ),
            "sender_assignment_table_digest": sender_assignment_table_digest,
            "spec_id": self.spec.spec_id,
            "status": self.certification_status,
        }
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "_oof_coverage", coverage)
        object.__setattr__(self, "_oof_receiver_coverage", receiver_coverage)
        object.__setattr__(self, "_oof_sender_assignments", assignments)
        object.__setattr__(self, "coverage_table_digest", coverage_table_digest)
        object.__setattr__(
            self, "receiver_coverage_table_digest", receiver_coverage_table_digest
        )
        object.__setattr__(
            self, "sender_assignment_table_digest", sender_assignment_table_digest
        )
        object.__setattr__(self, "receiver_coverage_audit_id", receiver_audit_id)
        object.__setattr__(
            self, "receiver_scoring_registry_id", receiver_registry.registry_id
        )
        object.__setattr__(self, "crossfit_id", stable_id("subject_crossfit", payload))
        record_validation(self)

    @property
    def oof_coverage(self) -> pd.DataFrame:
        """Return a defensive copy of the sender-stage OOF coverage table."""

        return self._oof_coverage.copy(deep=True)

    @property
    def root_input_digest(self) -> str:
        """Return the producer-owned complete sanitized input digest."""

        self.root_input_identity._require_intact()
        return str(self.root_input_identity.input_digest)

    @property
    def oof_receiver_coverage(self) -> pd.DataFrame:
        """Return a defensive copy of the exact receiver-stage coverage table."""

        return self._oof_receiver_coverage.copy(deep=True)

    @property
    def oof_sender_assignments(self) -> pd.DataFrame:
        """Return a defensive copy of held-out common-sender assignments."""

        return self._oof_sender_assignments.copy(deep=True)

    @property
    @validation_scope()
    def receiver_scoring_registry(self) -> ScoringCollectionDocument:
        """Return the exact producer-derived planned receiver registry."""

        self._require_intact()
        registry = _receiver_scoring_registry(self.spec, self.folds)
        if registry.registry_id != self.receiver_scoring_registry_id:
            raise ContractError(
                "Receiver scoring registry identity changed",
                code="receiver_scoring_registry_integrity_violation",
                field="receiver_scoring_registry_id",
                remediation="Rerun subject cross-fit from intact raw inputs",
            )
        return registry

    @property
    def scoring_collections(self) -> tuple[ScoringCollectionManifest, ...]:
        """Return exact planned receiver collections for every contrast/fold."""

        return tuple(self.receiver_scoring_registry.collections)

    @validation_scope()
    def _require_intact(self) -> None:
        if validation_is_cached(self):
            return
        try:
            repeated = CrossFitArtifacts._from_workflow(
                spec=self.spec,
                root_input_identity=self.root_input_identity,
                receiver_universe=self.receiver_universe,
                receiver_family_opportunity_universe=(
                    self.receiver_family_opportunity_universe
                ),
                directional_lr_hypothesis_universe=(
                    self.directional_lr_hypothesis_universe
                ),
                fold_plan=self.fold_plan,
                folds=self.folds,
                oof_coverage=self._oof_coverage,
                oof_receiver_coverage=self._oof_receiver_coverage,
                oof_sender_assignments=self._oof_sender_assignments,
                coverage_audit=self.coverage_audit,
            )
            valid = (
                self._producer_marker == _PRODUCER_MARKER
                and self.certification_status == _STAGE_STATUS
                and repeated.root_input_identity.identity_id
                == self.root_input_identity.identity_id
                and repeated.receiver_universe.universe_id
                == self.receiver_universe.universe_id
                and repeated.receiver_family_opportunity_universe.universe_id
                == self.receiver_family_opportunity_universe.universe_id
                and (
                    None
                    if repeated.directional_lr_hypothesis_universe is None
                    else repeated.directional_lr_hypothesis_universe.universe_id
                )
                == (
                    None
                    if self.directional_lr_hypothesis_universe is None
                    else self.directional_lr_hypothesis_universe.universe_id
                )
                and repeated.crossfit_id == self.crossfit_id
                and repeated.receiver_coverage_audit_id
                == self.receiver_coverage_audit_id
                and repeated.receiver_scoring_registry_id
                == self.receiver_scoring_registry_id
                and repeated.coverage_table_digest == self.coverage_table_digest
                and repeated.receiver_coverage_table_digest
                == self.receiver_coverage_table_digest
                and repeated.sender_assignment_table_digest
                == self.sender_assignment_table_digest
            )
        except (
            AttributeError,
            ContractError,
            KeyError,
            RuntimeError,
            StopIteration,
            TypeError,
            ValueError,
        ) as error:
            raise ContractError(
                "Cross-fit artifact failed table or lineage integrity validation",
                code="crossfit_artifact_integrity_violation",
                field="crossfit_id",
                remediation="Rerun subject cross-fit from intact raw inputs",
            ) from error
        if not valid:
            raise ContractError(
                "Cross-fit artifact failed table or lineage integrity validation",
                code="crossfit_artifact_integrity_violation",
                field="crossfit_id",
                remediation="Rerun subject cross-fit from intact raw inputs",
            )
        record_validation(self)

    @property
    def completed_stage_oof_verified(self) -> bool:
        """Whether the availability/sender stage passed exact OOF coverage audit."""

        return True

    @property
    @validation_scope()
    def oof_certification_audit(self) -> CrossFitOOFCertificationAudit:
        """Recompute the complete descriptive OOF readiness ledger."""

        from .certification import audit_crossfit_oof_readiness

        return audit_crossfit_oof_readiness(self)

    @property
    @validation_scope()
    def is_oof_certified(self) -> bool:
        """Whether every planned descriptive train/apply chain is OOF-ready."""

        return bool(self.oof_certification_audit.is_oof_descriptive_certified)

    @validation_scope()
    def to_manifest(self) -> dict[str, object]:
        """Return an auditable summary without embedding tabular payloads."""

        self._require_intact()
        oof_certification = self.oof_certification_audit
        common_functionals = tuple(
            functional
            for fold in self.folds
            for functional in fold.family_common_functionals
        )
        common_applications = tuple(
            application
            for fold in self.folds
            for application in fold.family_common_applications
        )
        receiver_program_models = tuple(
            model for fold in self.folds for model in fold.receiver_program_models
        )
        receiver_program_applications = tuple(
            application
            for fold in self.folds
            for application in fold.receiver_program_applications
        )
        receiver_incremental_models = tuple(
            model for fold in self.folds for model in fold.receiver_incremental_models
        )
        directional_bindings = tuple(
            binding
            for fold in self.folds
            for binding in fold.directional_response_bindings
        )
        directional_contrast_ids = {
            contrast_id
            for pair in self.spec.directional_pairs
            for contrast_id in (
                pair.forward_contrast_id,
                pair.reverse_contrast_id,
            )
        }
        calibration_statuses = [
            "missing"
            if model.gain_calibration_artifact is None
            else model.gain_calibration_artifact.status
            for model in receiver_incremental_models
        ]
        functional_statuses = [
            "observed"
            if functional.incremental_functional is not None
            else "not_estimable"
            for functional in common_functionals
        ]
        application_statuses = [
            "observed" if application.heldout_reason_code is None else "not_estimable"
            for application in common_applications
        ]
        remaining_stages = list(_REMAINING_PUBLIC_STAGES)
        if self.spec.penalty_tuning_spec is not None:
            completed_candidate_stages = {
                "attribution_tuning",
                "common_scoring_functional",
                "family_attribution",
                "incremental_downstream",
                "subject_blocked_inner_tuning",
            }
            remaining_stages = [
                stage
                for stage in remaining_stages
                if stage not in completed_candidate_stages
            ]
            if calibration_statuses and set(calibration_statuses) == {"observed"}:
                remaining_stages = [
                    stage
                    for stage in remaining_stages
                    if stage != "selected_penalty_inner_oof_gain_calibration"
                ]
        approved_nuisance_policy = bool(
            (
                self.spec.latent_nuisance_spec is None
                and self.spec.autonomous_program_resource is not None
                and self.spec.autonomous_program_resource.is_manifest_verified_trusted
            )
            or (
                self.spec.latent_nuisance_spec is not None
                and (
                    self.spec.autonomous_program_resource is None
                    or (
                        self.spec.autonomous_program_resource.is_manifest_verified_trusted
                    )
                )
            )
        )
        if not approved_nuisance_policy:
            remaining_stages.insert(-1, "receiver_autonomous_nuisance")
        if oof_certification.complete:
            remaining_stages = []
        receiver_registry = _receiver_scoring_registry(self.spec, self.folds)
        return {
            "crossfit_id": self.crossfit_id,
            "certification_status": self.certification_status,
            "completed_stage_oof_verified": self.completed_stage_oof_verified,
            "family_common_candidate_stage_connected": (
                self.spec.penalty_tuning_spec is not None
            ),
            "family_common_functional_status_counts": {
                status: functional_statuses.count(status)
                for status in sorted(set(functional_statuses))
            },
            "family_common_application_status_counts": {
                status: application_statuses.count(status)
                for status in sorted(set(application_statuses))
            },
            "gain_calibration_status_counts": {
                status: calibration_statuses.count(status)
                for status in sorted(set(calibration_statuses))
            },
            "receiver_program_training_status_counts": {
                status: sum(model.status == status for model in receiver_program_models)
                for status in sorted(
                    {model.status for model in receiver_program_models}
                )
            },
            "receiver_program_application_status_counts": {
                status: sum(
                    application.status == status
                    for application in receiver_program_applications
                )
                for status in sorted(
                    {
                        application.status
                        for application in receiver_program_applications
                    }
                )
            },
            "n_family_common_crossfit_bindings": sum(
                len(fold.family_common_bindings) for fold in self.folds
            ),
            **(
                {
                    "directional_pair_stage_connected": True,
                    "n_directional_response_bindings": len(directional_bindings),
                    "n_directional_receiver_pair_opportunities": (
                        len(self.spec.directional_pairs)
                        * len(self.receiver_universe.receiver_ids)
                        * len(self.folds)
                    ),
                    "n_directional_receiver_pair_absent_training": (
                        len(self.spec.directional_pairs)
                        * sum(
                            record.status is ReceiverTrainingSupportStatus.NOT_ESTIMABLE
                            for fold in self.folds
                            for record in fold.receiver_training_support
                        )
                    ),
                    "directional_receiver_universe_id": (
                        self.receiver_universe.universe_id
                    ),
                    "directional_receiver_axis_id": (
                        self.receiver_universe.receiver_axis_id
                    ),
                    "directional_binding_status_counts": {
                        status: sum(
                            binding.status == status for binding in directional_bindings
                        )
                        for status in sorted(
                            {binding.status for binding in directional_bindings}
                        )
                    },
                    "directional_registry_complete": len(directional_bindings)
                    == sum(
                        len(self.spec.directional_pairs)
                        * len(self.receiver_universe.receiver_ids)
                        for fold in self.folds
                    ),
                    "directional_supported_binding_registry_complete": (
                        len(directional_bindings)
                        == sum(
                            len(self.spec.directional_pairs)
                            * sum(
                                record.status is ReceiverTrainingSupportStatus.OBSERVED
                                for record in fold.receiver_training_support
                            )
                            for fold in self.folds
                        )
                    ),
                    "directional_diagnostic_complete": (
                        len(directional_bindings)
                        == len(self.spec.directional_pairs)
                        * len(self.receiver_universe.receiver_ids)
                        * len(self.folds)
                        and all(
                            binding.status == "observed"
                            for binding in directional_bindings
                        )
                    ),
                    "directional_formal_inference_allowed": False,
                }
                if self.spec.directional_pairs
                else {}
            ),
            "n_cross_receiver_common_functionals": sum(
                len(fold.cross_receiver_common_functionals) for fold in self.folds
            ),
            "n_cross_receiver_common_applications": sum(
                len(fold.cross_receiver_common_applications) for fold in self.folds
            ),
            "complete_pipeline_oof_certified": oof_certification.complete,
            "oof_certification_audit_id": oof_certification.audit_id,
            "oof_certification_audit": oof_certification.to_dict(),
            "family_edge_evidence_policy": _FAMILY_EDGE_EVIDENCE_POLICY,
            "oof_audit_scope": _STAGE_NAME,
            "spec": self.spec.to_dict(),
            "root_input_identity": self.root_input_identity.to_dict(),
            "receiver_universe": self.receiver_universe.to_dict(),
            "receiver_family_opportunity_universe_id": (
                self.receiver_family_opportunity_universe.universe_id
            ),
            "family_axis_id": self.receiver_family_opportunity_universe.family_axis_id,
            "receiver_family_opportunity_axis_id": (
                self.receiver_family_opportunity_universe.opportunity_axis_id
            ),
            "receiver_family_opportunity_universe": (
                self.receiver_family_opportunity_universe.to_dict()
            ),
            **(
                {
                    "directional_lr_hypothesis_universe": (
                        self.directional_lr_hypothesis_universe.to_dict()
                    )
                }
                if self.directional_lr_hypothesis_universe is not None
                else {}
            ),
            "fold_plan": self.fold_plan.to_dict(),
            "coverage_audit": self.coverage_audit.to_dict(),
            "coverage_table_digest": self.coverage_table_digest,
            "receiver_coverage_audit_id": self.receiver_coverage_audit_id,
            "receiver_scoring_registry_id": self.receiver_scoring_registry_id,
            "receiver_scoring_registry": receiver_registry.to_dict(),
            "receiver_coverage_table_digest": self.receiver_coverage_table_digest,
            "sender_assignment_table_digest": self.sender_assignment_table_digest,
            "fold_artifacts": [
                {
                    "fold_id": item.fold_id,
                    "training_artifact_id": item.training.training_artifact_id,
                    "receiver_training_support": [
                        record.to_dict() for record in item.receiver_training_support
                    ],
                    **(
                        {
                            "directional_design_applications": [
                                {
                                    "contrast_id": _contrast_id(encoder.contrast),
                                    "contrast_name": encoder.contrast.name,
                                    "encoder": encoder.to_dict(),
                                    "application": application.to_dict(),
                                }
                                for encoder, application in zip(
                                    item.design_encoders,
                                    item.design_applications,
                                    strict=True,
                                )
                                if _contrast_id(encoder.contrast)
                                in directional_contrast_ids
                            ]
                        }
                        if directional_contrast_ids
                        else {}
                    ),
                    "heldout_input_digest": item.application.heldout_input_digest,
                    "heldout_excluded_cell_type_ids": list(
                        item.application.excluded_cell_type_ids
                    ),
                    "training_subject_ids": list(item.training.training_subject_ids),
                    "heldout_subject_ids": list(item.application.heldout_subject_ids),
                    "receiver_family_artifacts": [
                        {
                            "receiver": model.receiver_family_artifact.receiver,
                            "contrast_name": model.contrast_name,
                            "training_artifact_id": model.training_artifact_id,
                            "certification_status": model.certification_status,
                            "application_status": application.application_status,
                        }
                        for model, application in zip(
                            item.receiver_family_models,
                            item.receiver_family_applications,
                            strict=True,
                        )
                    ],
                    "receiver_incremental_artifacts": [
                        {
                            "receiver": model.receiver,
                            "contrast_name": model.contrast_name,
                            "response_artifact_id": response.artifact_id,
                            "precision_transform_id": precision.precision_transform_id,
                            "response_backend_manifest": (
                                _response_backend_manifest(response)
                            ),
                            "precision_audit_manifest": (
                                _precision_audit_manifest(precision)
                            ),
                            "training_artifact_id": model.training_artifact_id,
                            "autonomous_program_resource_id": (
                                model.autonomous_program_resource_id
                            ),
                            "autonomous_program_source_id": (
                                model.autonomous_program_source_id
                            ),
                            "autonomous_program_verification_status": (
                                model.autonomous_program_verification_status
                            ),
                            "latent_nuisance_spec_id": (model.latent_nuisance_spec_id),
                            "latent_nuisance_artifact_id": (
                                model.latent_nuisance_artifact_id
                            ),
                            "latent_nuisance_artifact": (
                                None
                                if model.latent_nuisance_artifact is None
                                else model.latent_nuisance_artifact.to_dict()
                            ),
                            "penalty_tuning_spec_id": model.penalty_tuning_spec_id,
                            "inner_fold_plan_id": (
                                None
                                if model.inner_fold_plan is None
                                else model.inner_fold_plan.plan_id
                            ),
                            "penalty_tuning_artifact_id": (
                                None
                                if model.penalty_tuning_artifact is None
                                else model.penalty_tuning_artifact.tuning_id
                            ),
                            "gain_calibration_spec_id": (
                                model.gain_calibration_spec_id
                            ),
                            "gain_calibration_artifact_id": (
                                None
                                if model.gain_calibration_artifact is None
                                else model.gain_calibration_artifact.artifact_id
                            ),
                            "gain_calibration_status": (
                                "missing"
                                if model.gain_calibration_artifact is None
                                else model.gain_calibration_artifact.status
                            ),
                            "gain_calibration_reason_code": (
                                None
                                if model.gain_calibration_artifact is None
                                else model.gain_calibration_artifact.reason_code
                            ),
                            "selected_penalty_candidate_id": (
                                model.selected_penalty_candidate_id
                            ),
                            "selected_resolved_penalty_id": (
                                model.selected_resolved_penalty_id
                            ),
                            "certification_status": model.certification_status,
                            "response_application_id": (
                                response_application.application_id
                            ),
                            "application_id": application.application_id,
                            "application_certification_status": (
                                application.certification_status
                            ),
                            "diagnostic_status": application.diagnostic_status,
                            "official_incremental_status": (
                                application.official_incremental_status
                            ),
                            "reason_code": application.reason_code,
                        }
                        for (
                            response,
                            precision,
                            model,
                            response_application,
                            application,
                        ) in zip(
                            item.receiver_responses,
                            item.response_precisions,
                            item.receiver_incremental_models,
                            item.receiver_response_applications,
                            item.receiver_incremental_applications,
                            strict=True,
                        )
                    ],
                    **(
                        {
                            "directional_response_bindings": [
                                binding.to_dict()
                                for binding in item.directional_response_bindings
                            ]
                        }
                        if item.directional_response_bindings
                        else {}
                    ),
                    "receiver_program_artifacts": [
                        {
                            "receiver": model.receiver,
                            "contrast_name": model.contrast_name,
                            "training_artifact_id": model.training_artifact_id,
                            "training_status": model.status,
                            "training_reason_code": model.reason_code,
                            "application_id": application.application_id,
                            "application_status": application.status,
                            "application_reason_code": application.reason_code,
                            "heldout_row_manifest_id": (
                                application.heldout_row_manifest_id
                            ),
                            "heldout_input_digest": application.heldout_input_digest,
                            "reference_transform_id": (
                                None
                                if model.downstream_functional is None
                                else model.downstream_functional.reference_transform_id
                            ),
                            "reference_row_manifest_id": (
                                None
                                if model.downstream_functional is None
                                else (
                                    model.downstream_functional.reference_row_manifest_id
                                )
                            ),
                            "reference_subject_summary_digest": (
                                None
                                if model.downstream_functional is None
                                else (
                                    model.downstream_functional.reference_subject_summary_digest
                                )
                            ),
                            "reference_summary_method": (
                                None
                                if model.downstream_functional is None
                                else (
                                    model.downstream_functional.reference_summary_method
                                )
                            ),
                            "center_method": (
                                None
                                if model.downstream_functional is None
                                else model.downstream_functional.center_method
                            ),
                            "scale_method": (
                                None
                                if model.downstream_functional is None
                                else model.downstream_functional.scale_method
                            ),
                            "source_agnostic": True,
                            "receptor_agnostic": True,
                            "sender_agnostic": True,
                        }
                        for model, application in zip(
                            item.receiver_program_models,
                            item.receiver_program_applications,
                            strict=True,
                        )
                    ],
                    "family_common_scoring_artifacts": [
                        {
                            "receiver": functional.receiver,
                            "contrast_name": functional.contrast_name,
                            "functional_id": (functional.family_common_functional_id),
                            "application_id": application.application_id,
                            "binding_id": binding.binding_id,
                            "training_application_id": (
                                binding.training_application_id
                            ),
                            "edge_evidence_policy_id": (
                                binding.edge_evidence_policy_id
                            ),
                            "edge_evidence_digest": binding.edge_evidence_digest,
                            "availability_sample_interactions_digest": (
                                binding.availability_sample_interactions_digest
                            ),
                            "sender_input_digest": binding.sender_input_digest,
                            "tuning_manifest_id": functional.tuning_manifest_id,
                            "selected_penalty_id": functional.selected_penalty_id,
                            "autonomous_program_resource_id": (
                                functional.autonomous_program_resource_id
                            ),
                            "autonomous_program_source_id": (
                                functional.autonomous_program_source_id
                            ),
                            "receiver_program_training_artifact_id": (
                                None
                                if functional.receiver_program_artifact is None
                                else (
                                    functional.receiver_program_artifact.training_artifact_id
                                )
                            ),
                            "receiver_program_training_status": (
                                "not_estimable"
                                if functional.receiver_program_artifact is None
                                else functional.receiver_program_artifact.status
                            ),
                            "receiver_program_training_reason_code": (
                                functional.receiver_program_reason_code
                            ),
                            "receiver_program_application_id": (
                                application.receiver_program_application_id
                            ),
                            "functional_status": (
                                "observed"
                                if functional.incremental_functional is not None
                                else "not_estimable"
                            ),
                            "functional_reason_code": (
                                functional.incremental_reason_code
                            ),
                            "application_status": (
                                "observed"
                                if application.heldout_reason_code is None
                                else "not_estimable"
                            ),
                            "heldout_reason_code": application.heldout_reason_code,
                            "score_version": functional.score_version,
                            "family_attribution_digest": (
                                application.family_attribution_digest
                            ),
                            "subject_differential_digest": (
                                application.subject_differential_digest
                            ),
                            "family_scores_digest": application.family_scores_digest,
                            "member_scores_digest": application.member_scores_digest,
                            "sender_scores_digest": application.sender_scores_digest,
                            "n_family_attribution_rows": (
                                application.table_row_counts[0]
                            ),
                            "n_subject_differential_rows": (
                                application.table_row_counts[1]
                            ),
                            "n_family_rows": application.table_row_counts[2],
                            "n_member_rows": application.table_row_counts[3],
                            "n_sender_rows": application.table_row_counts[4],
                        }
                        for functional, application, binding in zip(
                            item.family_common_functionals,
                            item.family_common_applications,
                            item.family_common_bindings,
                            strict=True,
                        )
                    ],
                    "cross_receiver_common_scoring_artifacts": [
                        {
                            "contrast_name": functional.contrast_name,
                            "contrast_manifest_id": functional.contrast_manifest_id,
                            "functional_id": (functional.global_common_functional_id),
                            "application_id": application.application_id,
                            "receiver_ids": list(functional.receiver_ids),
                            "child_functional_ids": list(
                                functional.child_functional_ids
                            ),
                            "score_version": functional.score_version,
                            "estimand": functional.to_dict()["estimand"],
                            "functional_spec_id": functional.spec.spec_id,
                            "functional_schema_version": (
                                functional.spec.schema_version
                            ),
                            "calibration_policy": functional.to_dict()[
                                "calibration_policy"
                            ],
                            "scale_policy": functional.to_dict()["scale_policy"],
                            "sender_policy": functional.to_dict()["sender_policy"],
                            "softmin_power": functional.spec.softmin_power,
                            "epsilon": functional.spec.epsilon,
                            "receiver_gain_calibration_bindings": (
                                functional.to_dict()[
                                    "receiver_gain_calibration_bindings"
                                ]
                            ),
                            "all_receivers_gain_calibrated": (
                                functional.all_receivers_gain_calibrated
                            ),
                            "cross_receiver_percentile_rank_eligible": (
                                functional.cross_receiver_percentile_rank_eligible
                            ),
                            "lr_scores_digest": application.lr_scores_digest,
                            "sender_scores_digest": application.sender_scores_digest,
                            "n_lr_rows": application.table_row_counts[0],
                            "n_sender_rows": application.table_row_counts[1],
                            "common_functional_across_receivers": False,
                            "receiver_balanced_descriptive_collection": True,
                            "training_only_receiver_calibration": True,
                            "receiver_scale_amplification": False,
                            "formal_inference_allowed": False,
                        }
                        for functional, application in zip(
                            item.cross_receiver_common_functionals,
                            item.cross_receiver_common_applications,
                            strict=True,
                        )
                    ],
                }
                for item in self.folds
            ],
            "n_oof_coverage_rows": len(self._oof_coverage),
            "n_oof_receiver_coverage_rows": len(self._oof_receiver_coverage),
            "n_oof_sender_assignment_rows": len(self._oof_sender_assignments),
            "receiver_coverage_status_counts": {
                "diagnostic_status": {
                    str(status): int(count)
                    for status, count in sorted(
                        self._oof_receiver_coverage["diagnostic_status"]
                        .value_counts()
                        .items()
                    )
                },
                "official_incremental_status": {
                    str(status): int(count)
                    for status, count in sorted(
                        self._oof_receiver_coverage["official_incremental_status"]
                        .value_counts()
                        .items()
                    )
                },
            },
            "remaining_stages": remaining_stages,
        }


def _context_node(row: pd.Series, context_keys: tuple[str, ...]) -> Hashable:
    if len(context_keys) == 1:
        result: Hashable = plain_context_value(row[context_keys[0]])
        return result
    result = canonical_context(row, context_keys)
    return result


def _physical_subject_scope(
    sanitized: AnnData,
    *,
    subject_key: str,
    subject_ids: tuple[str, ...],
) -> AnnData:
    selected = sanitized.obs[subject_key].astype(str).isin(subject_ids).to_numpy()
    scope = sanitized[selected].copy()
    observed = tuple(sorted(scope.obs[subject_key].astype(str).unique()))
    if observed != subject_ids:
        raise ValueError("physical subject scope does not match its fold manifest")
    if scope.uns or scope.obsm:
        raise RuntimeError("sanitized fold scope retained undeclared caller state")
    return scope


def _interaction_driver_mapping(
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
) -> dict[str, str]:
    """Resolve only static resource/prior identities, never expression evidence."""

    known = set(target_prior.driver_ids)
    mapping: dict[str, str] = {}
    for interaction in resource_bundle.interactions:
        if target_prior.driver_kind == "interaction":
            candidates = {interaction.interaction_id}
        else:
            candidates = {interaction.ligand_name}
            if len(interaction.ligand_subunits) == 1:
                candidates.add(interaction.ligand_subunits[0])
        matches = sorted(candidates.intersection(known))
        if len(matches) == 1:
            mapping[interaction.interaction_id] = matches[0]
    return mapping


def _response_expression(
    prepared: _PreparedRawFold,
    *,
    receiver: str,
    contexts: set[Hashable] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Extract fixed log1p(CPM) sample expression for one receiver."""

    aggregate = prepared.aggregate
    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("public cross-fit receiver expression requires count aggregate")
    metadata = aggregate.unit_metadata.copy(deep=True).reset_index(drop=True)
    context_keys = tuple(prepared.schema.context_keys)
    context_nodes = metadata["context"].map(
        lambda value: _context_node(pd.Series(dict(value)), context_keys)
    )
    metadata["_actual_context_id"] = context_nodes.map(
        lambda node: node_context_fields(node, context_keys)[0]
    )
    selected = metadata["cell_type"].map(lambda value: str(value) == receiver)
    if contexts is not None:
        selected &= context_nodes.isin(contexts)
    selected &= metadata["state_eligible"].astype(bool)
    selected &= metadata["matrix_row"].notna()
    units = metadata.loc[selected].copy()
    if units.empty:
        raise ValueError(f"receiver {receiver!r} has no eligible sample expression")
    units["_sample_sort"] = units["sample_id"].astype(str)
    units["_unit_sort"] = units["unit_id"].astype(str)
    units = units.sort_values(
        ["_sample_sort", "_unit_sort"], kind="stable", ignore_index=True
    ).drop(columns=["_sample_sort", "_unit_sort"])
    matrix_rows = units["matrix_row"].astype(int).to_numpy()
    matrix = aggregate.counts[matrix_rows]
    if sparse.issparse(matrix):
        values = np.asarray(matrix.toarray(), dtype=np.float64)
    else:  # pragma: no cover - count aggregates are sparse by contract
        values = np.asarray(matrix, dtype=np.float64)
    library_sizes = values.sum(axis=1)
    if np.any(~np.isfinite(library_sizes)) or np.any(library_sizes <= 0):
        raise ValueError("receiver sample expression contains zero library size")
    values = np.log1p(values / library_sizes[:, None] * _CPM_SCALE)
    sample_ids = tuple(units["sample_id"].astype(str))
    subject_ids = tuple(units["subject_id"].astype(str))
    context_ids = tuple(units["_actual_context_id"].astype(str))
    return values, sample_ids, subject_ids, context_ids


def _expected_response_lineage(
    prepared: _PreparedRawFold,
    *,
    contexts: set[Hashable],
) -> tuple[tuple[str, str, str], ...]:
    """Return the raw sample metadata lineage for requested graph contexts."""

    schema = prepared.schema
    context_keys = tuple(schema.context_keys)
    rows: list[tuple[str, str, str]] = []
    for _, row in prepared.sample_metadata.iterrows():
        node = _context_node(row, context_keys)
        if node not in contexts:
            continue
        rows.append(
            (
                str(row[schema.sample_key]),
                str(row[schema.subject_key]),
                node_context_fields(node, context_keys)[0],
            )
        )
    return tuple(sorted(rows))


def _observed_response_lineage(
    sample_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(zip(sample_ids, subject_ids, context_ids, strict=True)))


def _training_response_lineage_is_valid(
    prepared: _PreparedRawFold,
    *,
    contexts: set[Hashable],
    sample_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> bool:
    """Require exact receiver-reference coverage with authoritative row lineage."""

    return (
        _training_response_lineage_reason(
            prepared,
            contexts=contexts,
            sample_ids=sample_ids,
            subject_ids=subject_ids,
            context_ids=context_ids,
        )
        is None
    )


def _training_response_lineage_reason(
    prepared: _PreparedRawFold,
    *,
    contexts: set[Hashable],
    sample_ids: tuple[str, ...],
    subject_ids: tuple[str, ...],
    context_ids: tuple[str, ...],
) -> str | None:
    """Return the fail-closed reason for an incomplete or forged reference."""

    observed = tuple(zip(sample_ids, subject_ids, context_ids, strict=True))
    if not observed or len({sample_id for sample_id, _, _ in observed}) != len(
        observed
    ):
        return "training_reference_expression_lineage_mismatch"
    expected = {
        sample_id: (subject_id, context_id)
        for sample_id, subject_id, context_id in _expected_response_lineage(
            prepared, contexts=contexts
        )
    }
    if any(
        expected.get(sample_id) != (subject_id, context_id)
        for sample_id, subject_id, context_id in observed
    ):
        return "training_reference_expression_lineage_mismatch"
    if set(sample_ids) != set(expected):
        return "training_reference_expression_incomplete_sample_coverage"
    return None


def _training_receiver_families(
    *,
    prepared: _PreparedRawFold,
    training: TrainingArtifacts,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    spec: CrossFitSpec,
    fold_id: str,
    receiver_family_opportunity_universe: FrozenReceiverFamilyOpportunityUniverse,
    directional_lr_hypothesis_universe: (
        FrozenReceiverFamilyLRHypothesisUniverse | None
    ) = None,
) -> tuple[ReceiverFamilyScoringArtifact, ...]:
    receiver_family_opportunity_universe._require_intact()
    if (
        target_prior.resource_id
        != receiver_family_opportunity_universe.prior_resource_id
        or target_prior.version != receiver_family_opportunity_universe.prior_version
        or target_prior.manifest_digest
        != receiver_family_opportunity_universe.prior_manifest_digest
        or _target_prior_content_id(target_prior)
        != receiver_family_opportunity_universe.prior_content_id
        or target_prior.driver_ids != receiver_family_opportunity_universe.driver_ids
        or prepared.aggregate.feature_ids
        != receiver_family_opportunity_universe.feature_ids
        or spec.family_cosine_threshold
        != receiver_family_opportunity_universe.cosine_threshold
    ):
        raise ContractError(
            "Receiver-family fitting inputs differ from the frozen run root",
            code="crossfit_root_family_axis_mismatch",
            field="target_prior,feature_ids,family_cosine_threshold",
            remediation="Use the exact run-root family opportunity universe",
        )
    availability = _fit_interaction_universe(
        prepared,
        resource_bundle,
        spec.training_spec,
    )
    if (
        availability.frozen_interaction_universe.to_dict()
        != training.frozen_interaction_universe.to_dict()
    ):
        raise RuntimeError(
            "receiver-family training did not reproduce the fold interaction universe"
        )
    if directional_lr_hypothesis_universe is None:
        mapping = _interaction_driver_mapping(resource_bundle, target_prior)
    else:
        directional_lr_hypothesis_universe._require_intact()
        if (
            directional_lr_hypothesis_universe.resource_bundle_content_id
            != _resource_bundle_content_id(resource_bundle)
            or directional_lr_hypothesis_universe.target_prior_content_id
            != _target_prior_content_id(target_prior)
        ):
            raise ContractError(
                "Directional receiver-family fitting received resources outside "
                "its frozen LR hypothesis universe",
                code="directional_lr_hypothesis_universe_resource_mismatch",
                field="resource_bundle,target_prior",
                remediation="Use the run-root resources frozen before fold planning",
            )
        mapping = {
            membership.interaction_id: membership.driver_id
            for membership in directional_lr_hypothesis_universe.memberships
        }
    receivers = prepared.cell_type_ids
    family_artifacts = fit_receiver_family_training_artifacts(
        availability,
        target_prior,
        receivers=receivers,
        fold_id=fold_id,
        feature_ids=prepared.aggregate.feature_ids,
        driver_by_interaction=mapping,
        receptor_gate_threshold=spec.receptor_gate_threshold,
        cosine_threshold=spec.family_cosine_threshold,
    )
    for artifact in family_artifacts:
        _require_receiver_family_on_root_axis(
            artifact,
            receiver_family_opportunity_universe,
        )
    receiver_families = {artifact.receiver: artifact for artifact in family_artifacts}
    models: list[ReceiverFamilyScoringArtifact] = []
    for contrast in spec.contrasts:
        reference_contexts = {
            context for context, weight in contrast.weights.items() if weight < 0
        }
        if not reference_contexts:  # pragma: no cover - balanced contrast contract
            raise RuntimeError("balanced contrast is missing a reference context")
        for receiver in receivers:
            receiver_family = receiver_families[receiver]
            try:
                (
                    reference_expression,
                    reference_sample_ids,
                    reference_subjects,
                    reference_context_ids,
                ) = _response_expression(
                    prepared, receiver=receiver, contexts=reference_contexts
                )
            except ValueError as error:
                if "no eligible sample expression" not in str(error):
                    raise
                models.append(
                    mark_receiver_family_scoring_not_estimable(
                        receiver_family,
                        contrast_name=contrast.name,
                        reason_code="training_reference_expression_not_estimable",
                    )
                )
                continue
            lineage_reason = _training_response_lineage_reason(
                prepared,
                contexts=reference_contexts,
                sample_ids=reference_sample_ids,
                subject_ids=reference_subjects,
                context_ids=reference_context_ids,
            )
            if lineage_reason is not None:
                models.append(
                    mark_receiver_family_scoring_not_estimable(
                        receiver_family,
                        contrast_name=contrast.name,
                        reason_code=lineage_reason,
                    )
                )
                continue
            try:
                scoring_model = fit_receiver_family_scoring_artifact(
                    receiver_family,
                    reference_expression,
                    contrast_name=contrast.name,
                    sample_ids=reference_sample_ids,
                    sample_subject_ids=reference_subjects,
                    sample_context_ids=reference_context_ids,
                    minimum_scale=spec.downstream_minimum_scale,
                )
            except ValueError as error:
                if not any(
                    message in str(error)
                    for message in (
                        "at least two complete finite samples",
                        "at least two unique reference subjects",
                    )
                ):
                    raise
                scoring_model = mark_receiver_family_scoring_not_estimable(
                    receiver_family,
                    contrast_name=contrast.name,
                    reason_code="training_reference_expression_not_estimable",
                )
            models.append(scoring_model)
    return tuple(
        sorted(
            models,
            key=lambda model: (
                model.contrast_name,
                model.receiver_family_artifact.receiver,
            ),
        )
    )


def _apply_receiver_families(
    models: tuple[ReceiverFamilyScoringArtifact, ...],
    *,
    prepared: _PreparedRawFold,
    contrasts: tuple[ContrastSpec, ...],
) -> tuple[ReceiverFamilyScoringApplication, ...]:
    contrast_contexts = {contrast.name: set(contrast.weights) for contrast in contrasts}
    heldout_subject_ids = prepared.subject_ids
    applications: list[ReceiverFamilyScoringApplication] = []
    for model in models:
        receiver = model.receiver_family_artifact.receiver
        try:
            expression, sample_ids, subject_ids, context_ids = _response_expression(
                prepared,
                receiver=receiver,
                contexts=contrast_contexts[model.contrast_name],
            )
        except ValueError as error:
            if "no eligible sample expression" not in str(error):
                raise
            applications.append(
                mark_receiver_family_application_not_estimable(
                    model,
                    heldout_subject_ids=heldout_subject_ids,
                    reason_code="heldout_receiver_expression_not_estimable",
                )
            )
            continue
        observed_lineage = _observed_response_lineage(
            sample_ids, subject_ids, context_ids
        )
        expected_lineage = _expected_response_lineage(
            prepared, contexts=contrast_contexts[model.contrast_name]
        )
        if observed_lineage != expected_lineage:
            applications.append(
                mark_receiver_family_application_not_estimable(
                    model,
                    heldout_subject_ids=heldout_subject_ids,
                    reason_code="heldout_receiver_expression_lineage_mismatch",
                )
            )
            continue
        applications.append(
            apply_receiver_family_scoring_artifact(
                model,
                expression,
                feature_ids=prepared.aggregate.feature_ids,
                sample_subject_ids=subject_ids,
            )
        )
    return tuple(applications)


def _training_receiver_programs(
    receiver_models: tuple[ReceiverFamilyScoringArtifact, ...],
    *,
    prepared: _PreparedRawFold,
    contrasts: tuple[ContrastSpec, ...],
    minimum_scale: float,
) -> tuple[ReceiverProgramTrainingArtifact, ...]:
    """Freeze receptor-agnostic target programs once per receiver/contrast."""

    contrast_by_name = {contrast.name: contrast for contrast in contrasts}
    programs: list[ReceiverProgramTrainingArtifact] = []
    for receiver_model in receiver_models:
        contrast = contrast_by_name[receiver_model.contrast_name]
        reference_contexts = {
            context for context, weight in contrast.weights.items() if weight < 0
        }
        receiver_family = receiver_model.receiver_family_artifact
        try:
            expression, sample_ids, subject_ids, context_ids = _response_expression(
                prepared,
                receiver=receiver_family.receiver,
                contexts=reference_contexts,
            )
            lineage_reason = _training_response_lineage_reason(
                prepared,
                contexts=reference_contexts,
                sample_ids=sample_ids,
                subject_ids=subject_ids,
                context_ids=context_ids,
            )
            if lineage_reason is not None:
                raise ValueError(lineage_reason)
            program = fit_receiver_program_training_artifact(
                receiver_family,
                expression,
                contrast_name=contrast.name,
                sample_ids=sample_ids,
                sample_subject_ids=subject_ids,
                reference_context_ids=context_ids,
                minimum_scale=minimum_scale,
            )
        except ValueError as error:
            if not any(
                message in str(error)
                for message in (
                    "no eligible sample expression",
                    "at least two complete finite samples",
                    "at least two unique reference subjects",
                    "training_reference_expression_lineage_mismatch",
                    "training_reference_expression_incomplete_sample_coverage",
                )
            ):
                raise
            observed_reason = str(error)
            reason_code = (
                observed_reason
                if observed_reason
                in {
                    "training_reference_expression_lineage_mismatch",
                    "training_reference_expression_incomplete_sample_coverage",
                }
                else "training_reference_expression_not_estimable"
            )
            program = mark_receiver_program_training_not_estimable(
                receiver_family,
                contrast_name=contrast.name,
                reason_code=reason_code,
            )
        programs.append(program)
    return tuple(programs)


def _apply_receiver_programs(
    programs: tuple[ReceiverProgramTrainingArtifact, ...],
    *,
    prepared: _PreparedRawFold,
    design_encoders: tuple[FrozenDesignEncoder, ...],
    design_applications: tuple[FrozenDesignApplication, ...],
) -> tuple[ReceiverProgramApplication, ...]:
    """Apply programs once per receiver without sender or receptor conditioning."""

    design_by_name = {
        encoder.contrast.name: (encoder, application)
        for encoder, application in zip(
            design_encoders, design_applications, strict=True
        )
    }
    applications: list[ReceiverProgramApplication] = []
    for program in programs:
        encoder, design_application = design_by_name[program.contrast_name]
        context_nodes = set(encoder.contrast.weights)
        context_ids = {
            node_context_fields(node, encoder.context_keys)[0] for node in context_nodes
        }
        selected = tuple(
            index
            for index, context_id in enumerate(design_application.sample_context_ids)
            if context_id in context_ids
        )
        expected_samples = tuple(
            design_application.sample_ids[index] for index in selected
        )
        expected_subjects = tuple(
            design_application.sample_subject_ids[index] for index in selected
        )
        expected_contexts = tuple(
            design_application.sample_context_ids[index] for index in selected
        )
        try:
            (
                expression,
                sample_ids,
                subject_ids,
                actual_context_ids,
            ) = _response_expression(
                prepared,
                receiver=program.receiver,
                contexts=context_nodes,
            )
        except ValueError as error:
            if "no eligible sample expression" not in str(error):
                raise
            applications.append(
                mark_receiver_program_application_not_estimable(
                    program,
                    sample_ids=expected_samples,
                    sample_subject_ids=expected_subjects,
                    sample_context_ids=expected_contexts,
                    reason_code="heldout_receiver_expression_not_estimable",
                )
            )
            continue
        expected_by_sample = {
            sample_id: (subject_id, context_id)
            for sample_id, subject_id, context_id in zip(
                expected_samples, expected_subjects, expected_contexts, strict=True
            )
        }
        observed_lineage = _observed_response_lineage(
            sample_ids, subject_ids, actual_context_ids
        )
        expected_lineage = tuple(
            sorted(
                (sample_id, subject_id, context_id)
                for sample_id, (subject_id, context_id) in expected_by_sample.items()
            )
        )
        if observed_lineage != expected_lineage:
            applications.append(
                mark_receiver_program_application_not_estimable(
                    program,
                    sample_ids=expected_samples,
                    sample_subject_ids=expected_subjects,
                    sample_context_ids=expected_contexts,
                    reason_code=(
                        "heldout_receiver_expression_incomplete_sample_coverage"
                    ),
                )
            )
            continue
        applications.append(
            apply_receiver_program_training_artifact(
                program,
                expression,
                feature_ids=prepared.aggregate.feature_ids,
                sample_ids=sample_ids,
                sample_subject_ids=subject_ids,
                sample_context_ids=actual_context_ids,
            )
        )
    return tuple(applications)


def _encoder_subject_design(encoder: FrozenDesignEncoder) -> SubjectDesign:
    context_ids = tuple(
        node_context_fields(node, encoder.context_keys)[0]
        for node in encoder.contrast.weights
    )
    metadata = pd.DataFrame(
        {
            "subject_id": encoder.training_sample_subject_ids,
            "context_id": encoder.training_sample_context_ids,
        }
    )
    return audit_subject_design(
        metadata,
        context_ids,
        context_key="context_id",
        subject_key="subject_id",
    ).design


def _fit_receiver_incremental_chains(
    models: tuple[ReceiverFamilyScoringArtifact, ...],
    *,
    aggregate: PseudobulkDataset,
    design_encoders: tuple[FrozenDesignEncoder, ...],
    training_metadata: pd.DataFrame,
    training_input_digest: str,
    fold_id: str,
    minimum_scale: float,
    min_subjects_per_context: int,
    autonomous_program_resource: ReceiverAutonomousProgramResource | None,
    latent_nuisance_spec: FrozenLatentNuisanceSpec | None,
    penalty_tuning_spec: PenaltyTuningSpec | None,
    gain_calibration_spec: GainCalibrationSpec | None,
    inner_partition_seed_lineage: SeedLineage | None,
) -> tuple[
    tuple[_ReceiverResponseArtifact, ...],
    tuple[PrecisionTransformResult, ...],
    tuple[ReceiverIncrementalTrainingArtifact, ...],
]:
    encoder_by_name = {encoder.contrast.name: encoder for encoder in design_encoders}
    receiver_jobs = _configured_jobs("CRYCHIC_RECEIVER_JOBS", default=1)
    receiver_process_jobs = _configured_jobs("CRYCHIC_RECEIVER_PROCESS_JOBS", default=1)
    use_process_workers = (
        receiver_process_jobs > 1
        and threading.current_thread() is threading.main_thread()
    )
    worker_count = receiver_process_jobs if use_process_workers else receiver_jobs
    reporter = (
        _ProgressReporter(
            total=len(models),
            label=f"fold={fold_id}:receiver_incremental",
            path=_progress_path_for_fold(fold_id),
        )
        if worker_count > 1 or _progress_enabled()
        else None
    )

    def fit_one(
        model: ReceiverFamilyScoringArtifact,
    ) -> tuple[
        _ReceiverResponseArtifact,
        PrecisionTransformResult,
        ReceiverIncrementalTrainingArtifact,
    ]:
        started = (
            None if reporter is None or use_process_workers else reporter.task_started()
        )
        try:
            encoder = encoder_by_name[model.contrast_name]
            subject_design = _encoder_subject_design(encoder)
            if subject_design in {
                SubjectDesign.MIXED,
                SubjectDesign.REPEATED_MULTI_CONTEXT,
            }:
                response: _ReceiverResponseArtifact = (
                    fit_repeated_measures_cr2_fold_response(
                        aggregate,
                        encoder,
                        training_metadata,
                        receiver=model.receiver_family_artifact.receiver,
                        fold_id=fold_id,
                        training_input_digest=training_input_digest,
                        min_subjects_per_context=max(2, min_subjects_per_context),
                        min_subject_clusters=max(6, min_subjects_per_context),
                    )
                )
            else:
                response = fit_fold_gene_response(
                    aggregate,
                    encoder,
                    receiver=model.receiver_family_artifact.receiver,
                    fold_id=fold_id,
                    training_input_digest=training_input_digest,
                    min_subjects_per_context=max(2, min_subjects_per_context),
                )
            feature_scale = (
                None
                if response.status
                != (
                    response.observed_status
                    if isinstance(response, RepeatedMeasuresFoldResponseArtifact)
                    else "ok"
                )
                else _receiver_incremental_feature_scale(
                    encoder,
                    response,
                    minimum_scale=minimum_scale,
                )
            )
            if isinstance(response, RepeatedMeasuresFoldResponseArtifact):
                precision = fit_repeated_response_precision(
                    response, feature_scale=feature_scale
                )
            else:
                precision = fit_response_precision(
                    response, feature_scale=feature_scale
                )
            incremental_model = fit_receiver_incremental_training_artifact(
                encoder,
                response,
                precision,
                model.receiver_family_artifact,
                autonomous_program_resource,
                latent_nuisance_spec=latent_nuisance_spec,
                minimum_scale=minimum_scale,
                penalty_tuning_spec=penalty_tuning_spec,
                gain_calibration_spec=gain_calibration_spec,
                inner_partition_seed_lineage=_receiver_inner_partition_lineage(
                    inner_partition_seed_lineage,
                    receiver=response.receiver,
                    contrast_id=_contrast_id(encoder.contrast),
                ),
            )
            return response, precision, incremental_model
        finally:
            if reporter is not None and started is not None:
                reporter.task_completed(
                    started,
                    detail=model.receiver_family_artifact.receiver,
                )

    workers = min(worker_count, len(models)) if models else 0
    if reporter is not None:
        reporter.event("start")
    if workers <= 1:
        fitted = tuple(fit_one(model) for model in models)
    elif use_process_workers:
        process_starts: dict[int, float] = {}

        def process_started(index: int) -> None:
            if reporter is not None:
                process_starts[index] = reporter.task_started()

        def process_completed(index: int) -> None:
            if reporter is not None:
                started_at = process_starts.pop(index, time.perf_counter())
                reporter.task_completed(
                    started_at,
                    detail=models[index].receiver_family_artifact.receiver,
                )

        fitted = _bounded_fork_map_in_order(
            fit_one,
            models,
            n_jobs=workers,
            on_start=process_started if reporter is not None else None,
            on_complete=process_completed if reporter is not None else None,
        )
    else:
        fitted_by_index: dict[
            int,
            tuple[
                _ReceiverResponseArtifact,
                PrecisionTransformResult,
                ReceiverIncrementalTrainingArtifact,
            ],
        ] = {}
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="crychic-receiver",
        ) as executor:
            futures = {
                executor.submit(fit_one, model): index
                for index, model in enumerate(models)
            }
            for future in futures:
                fitted_by_index[futures[future]] = future.result()
        fitted = tuple(fitted_by_index[index] for index in range(len(models)))
    if reporter is not None:
        reporter.event("complete")
    responses = [item[0] for item in fitted]
    precisions = [item[1] for item in fitted]
    incremental_models = [item[2] for item in fitted]
    return tuple(responses), tuple(precisions), tuple(incremental_models)


def _apply_receiver_incremental_chains(
    responses: tuple[_ReceiverResponseArtifact, ...],
    incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...],
    *,
    aggregate: PseudobulkDataset,
    design_encoders: tuple[FrozenDesignEncoder, ...],
    design_applications: tuple[FrozenDesignApplication, ...],
) -> tuple[
    tuple[_ReceiverResponseApplication, ...],
    tuple[ReceiverIncrementalApplication, ...],
]:
    design_by_name = {
        encoder.contrast.name: application
        for encoder, application in zip(
            design_encoders, design_applications, strict=True
        )
    }
    response_applications: list[_ReceiverResponseApplication] = []
    incremental_applications: list[ReceiverIncrementalApplication] = []
    for response, model in zip(responses, incremental_models, strict=True):
        design_application = design_by_name[response.contrast_name]
        if isinstance(response, RepeatedMeasuresFoldResponseArtifact):
            response_application: _ReceiverResponseApplication = (
                apply_repeated_measures_fold_response(
                    aggregate, response, design_application
                )
            )
        else:
            response_application = apply_fold_gene_response(
                aggregate, response, design_application
            )
        incremental_application = apply_receiver_incremental_training_artifact(
            model, response_application, design_application
        )
        response_applications.append(response_application)
        incremental_applications.append(incremental_application)
    return tuple(response_applications), tuple(incremental_applications)


def _build_directional_response_bindings(
    pair_specs: tuple[DirectionalContrastPairSpec, ...],
    responses: tuple[_ReceiverResponseArtifact, ...],
    incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...],
    incremental_applications: tuple[ReceiverIncrementalApplication, ...],
    *,
    receiver_ids: tuple[str, ...],
) -> tuple[DirectionalCrossFitBinding, ...]:
    """Bind exact forward/reverse chains by contrast name and receiver."""

    if not pair_specs:
        return ()
    responses_by_key = {
        (response.contrast_name, response.receiver): response for response in responses
    }
    models_by_key = {
        (model.contrast_name, model.receiver): model for model in incremental_models
    }
    applications_by_key = {
        (model.contrast_name, model.receiver): application
        for model, application in zip(
            incremental_models, incremental_applications, strict=True
        )
    }
    bindings: list[DirectionalCrossFitBinding] = []
    for pair_spec in pair_specs:
        for receiver in sorted(receiver_ids):
            forward_key = (pair_spec.forward_contrast.name, receiver)
            reverse_key = (pair_spec.reverse_contrast.name, receiver)
            try:
                binding = build_directional_crossfit_binding(
                    pair_spec,
                    responses_by_key[forward_key],
                    responses_by_key[reverse_key],
                    models_by_key[forward_key],
                    models_by_key[reverse_key],
                    applications_by_key[forward_key],
                    applications_by_key[reverse_key],
                )
            except KeyError as error:  # pragma: no cover - fold coverage invariant
                raise RuntimeError(
                    "directional pair is missing a planned receiver parent chain"
                ) from error
            bindings.append(binding)
    return tuple(
        sorted(
            bindings,
            key=lambda binding: (
                binding.pair_spec_id,
                binding.receiver,
            ),
        )
    )


def _maximum_observed(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    maximum = float(numeric.max())
    if not np.isfinite(maximum) or not 0.0 <= maximum <= 1.0:
        raise ValueError("family edge evidence must remain on the unit interval")
    return maximum


def _completed_common_sender_application(
    functional: ContrastCommonSenderFunctional,
    design_application: FrozenDesignApplication,
    availability: pd.DataFrame,
    *,
    receiver: str,
    interaction_ids: tuple[str, ...],
) -> CommonSenderApplication:
    """Apply one sender functional to an explicit held-out candidate grid."""

    selected_interactions = set(interaction_ids)
    selected_priors = tuple(
        prior
        for prior in functional.candidate_priors
        if prior.receiver == receiver and prior.interaction_id in selected_interactions
    )
    heldout_samples = {
        str(sample_id)
        for sample_id, context_id in zip(
            design_application.sample_ids,
            design_application.sample_context_ids,
            strict=True,
        )
        if context_id in functional.context_ids
    }
    candidate_senders = {prior.sender for prior in selected_priors}
    local = availability.loc[
        availability["sample_id"].astype(str).isin(heldout_samples)
        & availability["receiver"].astype(str).eq(receiver)
        & availability["interaction_id"].astype(str).isin(selected_interactions)
        & availability["sender"].astype(str).isin(candidate_senders),
        [
            "sample_id",
            "receiver",
            "interaction_id",
            "sender",
            "ligand_availability",
        ],
    ]
    lookup: dict[tuple[str, str, str, str], float | None] = {}
    if not local.empty:
        for key, group in local.groupby(
            ["sample_id", "receiver", "interaction_id", "sender"],
            observed=True,
            sort=False,
        ):
            sample_id, grouped_receiver, interaction_id, sender = key
            lookup[
                (
                    str(sample_id),
                    str(grouped_receiver),
                    str(interaction_id),
                    str(sender),
                )
            ] = _maximum_observed(group["ligand_availability"])
    rows: list[dict[str, object]] = []
    for sample_id, subject_id, context_id in zip(
        design_application.sample_ids,
        design_application.sample_subject_ids,
        design_application.sample_context_ids,
        strict=True,
    ):
        if context_id not in functional.context_ids:
            continue
        for prior in selected_priors:
            rows.append(
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "context_id": context_id,
                    "sender": prior.sender,
                    "receiver": prior.receiver,
                    "interaction_id": prior.interaction_id,
                    "ligand_availability": lookup.get(
                        (
                            sample_id,
                            prior.receiver,
                            prior.interaction_id,
                            prior.sender,
                        )
                    ),
                }
            )
    input_table = pd.DataFrame(
        rows,
        columns=(
            "sample_id",
            "subject_id",
            "context_id",
            "sender",
            "receiver",
            "interaction_id",
            "ligand_availability",
        ),
    )
    return apply_contrast_common_sender_functional(functional, input_table)


def _family_common_edge_evidence(
    functional: FamilyCommonScoringFunctional,
    design_application: FrozenDesignApplication,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    """Build exact sample/interaction/mode evidence from frozen held-out rows."""

    heldout_samples = {
        str(sample_id)
        for sample_id, context_id in zip(
            design_application.sample_ids,
            design_application.sample_context_ids,
            strict=True,
        )
        if context_id in functional.context_ids
    }
    local = availability.loc[
        availability["sample_id"].astype(str).isin(heldout_samples)
        & availability["receiver"].astype(str).eq(functional.receiver)
        & availability["interaction_id"].astype(str).isin(functional.interaction_ids),
        [
            "sample_id",
            "receiver",
            "interaction_id",
            "ligand_availability",
            "availability_state",
            "availability_ecosystem",
        ],
    ]
    grouped: dict[
        tuple[str, str, str],
        tuple[float | None, float | None, float | None],
    ] = {}
    if not local.empty:
        for key, group in local.groupby(
            ["sample_id", "receiver", "interaction_id"],
            observed=True,
            sort=False,
        ):
            sample_id, receiver, interaction_id = key
            grouped[(str(sample_id), str(receiver), str(interaction_id))] = (
                _maximum_observed(group["ligand_availability"]),
                _maximum_observed(group["availability_state"]),
                _maximum_observed(group["availability_ecosystem"]),
            )
    prevalence: dict[str, float | None] = {}
    ligand_contrast_gates = {
        interaction_id: interaction_ligand_contrast_gate(
            functional.sender_functional,
            functional.receiver,
            interaction_id,
        )
        for interaction_id in functional.interaction_ids
    }
    for interaction_id in functional.interaction_ids:
        priors = tuple(
            prior.prevalence_prior
            for prior in functional.sender_functional.candidate_priors
            if prior.receiver == functional.receiver
            and prior.interaction_id == interaction_id
        )
        prevalence[interaction_id] = (
            None
            if not priors or any(value is None for value in priors)
            else max(float(value) for value in priors if value is not None)
        )

    rows: list[dict[str, object]] = []
    for sample_id, subject_id, context_id in zip(
        design_application.sample_ids,
        design_application.sample_subject_ids,
        design_application.sample_context_ids,
        strict=True,
    ):
        if context_id not in functional.context_ids:
            continue
        for interaction_id in functional.interaction_ids:
            ligand_contrast_gate = ligand_contrast_gates[interaction_id]
            evidence = grouped.get((sample_id, functional.receiver, interaction_id))
            ligand_availability = None if evidence is None else evidence[0]
            for mode, evidence_index in (
                ("state", 1),
                ("ecosystem", 2),
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "context_id": context_id,
                        "receiver": functional.receiver,
                        "interaction_id": interaction_id,
                        "mode": mode,
                        "ligand_contrast_gate": ligand_contrast_gate.gate,
                        "ligand_contrast_gate_id": ligand_contrast_gate.gate_id,
                        "ligand_contrast_gate_status": (
                            SenderContrastSupportStatus(
                                ligand_contrast_gate.status
                            ).value
                        ),
                        "ligand_contrast_gate_reason_code": (
                            ligand_contrast_gate.reason_code
                        ),
                        "availability": (
                            None if evidence is None else evidence[evidence_index]
                        ),
                        "ligand_availability": ligand_availability,
                        "prior_quality": 1.0,
                        "subject_prevalence": prevalence[interaction_id],
                        "resource_evidence": 1.0,
                    }
                )
    return pd.DataFrame(rows, columns=FAMILY_COMMON_EDGE_EVIDENCE_COLUMNS)


def _fit_family_common_chains(
    receiver_models: tuple[ReceiverFamilyScoringArtifact, ...],
    receiver_programs: tuple[ReceiverProgramTrainingArtifact, ...],
    incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...],
    *,
    sender_functionals: tuple[ContrastCommonSenderFunctional, ...],
) -> tuple[FamilyCommonScoringFunctional, ...]:
    sender_by_contrast = {
        functional.contrast_name: functional for functional in sender_functionals
    }
    functionals: list[FamilyCommonScoringFunctional] = []
    for receiver_model, receiver_program, incremental_model in zip(
        receiver_models, receiver_programs, incremental_models, strict=True
    ):
        tuning = incremental_model.penalty_tuning_artifact
        if tuning is None:
            raise RuntimeError(
                "family-common scoring requires an explicit penalty tuning policy"
            )
        sender = sender_by_contrast[receiver_model.contrast_name]
        receiver_family = receiver_model.receiver_family_artifact
        diagnostic = incremental_model.diagnostic_functional
        diagnostic_ready = (
            diagnostic is not None
            and tuning.is_oof_certified
            and incremental_model.selected_penalty_candidate_id
            == tuning.selected_candidate_id
            and diagnostic.penalty_candidate_id == tuning.selected_candidate_id
            and incremental_model.selected_resolved_penalty_id
            == diagnostic.resolved_penalty_id
        )
        if diagnostic_ready:
            assert diagnostic is not None
            assert incremental_model.selected_resolved_penalty_id is not None
            functional = fit_family_common_scoring_functional(
                receiver_family,
                diagnostic,
                sender,
                receiver_incremental_training_artifact_id=(
                    incremental_model.training_artifact_id
                ),
                tuning_manifest_id=tuning.tuning_id,
                selected_penalty_id=(incremental_model.selected_resolved_penalty_id),
                autonomous_program_resource_id=(
                    incremental_model.autonomous_program_source_id
                ),
                receiver_program_artifact=receiver_program,
            )
        else:
            functional = mark_family_common_scoring_not_estimable(
                receiver_family,
                sender,
                fold_id=incremental_model.fold_id,
                receiver_incremental_training_artifact_id=(
                    incremental_model.training_artifact_id
                ),
                tuning_manifest_id=tuning.tuning_id,
                selected_penalty_id=(incremental_model.selected_resolved_penalty_id),
                autonomous_program_resource_id=(
                    incremental_model.autonomous_program_source_id
                ),
                reason_code=(
                    incremental_model.diagnostic_reason_code
                    or tuning.reason_code
                    or incremental_model.reason_code
                    or "incremental_training_not_estimable"
                ),
                receiver_program_artifact=receiver_program,
            )
        functionals.append(functional)
    return tuple(functionals)


def _fit_cross_receiver_common_chains(
    functionals: tuple[FamilyCommonScoringFunctional, ...],
    incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...],
    *,
    planned_receiver_ids: tuple[str, ...],
) -> tuple[CrossReceiverCommonScoringFunctional, ...]:
    """Build one global parent per contrast from complete receiver children."""

    by_contrast: dict[str, list[FamilyCommonScoringFunctional]] = {}
    for functional in functionals:
        by_contrast.setdefault(functional.contrast_name, []).append(functional)
    models_by_key = {
        (model.contrast_name, model.receiver): model for model in incremental_models
    }
    functional_keys = {
        (functional.contrast_name, functional.receiver) for functional in functionals
    }
    if len(models_by_key) != len(incremental_models) or set(models_by_key) != (
        functional_keys
    ):
        raise ValueError(
            "cross-receiver common scoring requires one incremental model per child"
        )
    result: list[CrossReceiverCommonScoringFunctional] = []
    for contrast_name, children in sorted(by_contrast.items()):
        calibration_artifacts = {}
        for child in children:
            model = models_by_key[(contrast_name, child.receiver)]
            if (
                model.training_artifact_id
                != child.receiver_incremental_training_artifact_id
            ):
                raise ValueError(
                    "cross-receiver gain calibration parent does not match its child"
                )
            calibration_artifacts[child.receiver] = model.gain_calibration_artifact
        result.append(
            fit_cross_receiver_common_scoring_functional(
                children,
                planned_receiver_ids=planned_receiver_ids,
                gain_calibration_artifacts=calibration_artifacts,
            )
        )
    if sum(len(functional.child_functionals) for functional in result) != len(
        functionals
    ):
        raise RuntimeError("cross-receiver common parents do not cover every child")
    return tuple(sorted(result, key=lambda item: item.contrast_manifest_id))


def _apply_family_common_chains(
    functionals: tuple[FamilyCommonScoringFunctional, ...],
    receiver_program_applications: tuple[ReceiverProgramApplication, ...],
    incremental_applications: tuple[ReceiverIncrementalApplication, ...],
    *,
    training_application: TrainingArtifactApplication,
    design_encoders: tuple[FrozenDesignEncoder, ...],
    design_applications: tuple[FrozenDesignApplication, ...],
    availability: pd.DataFrame,
) -> tuple[
    tuple[FamilyCommonScoringApplication, ...],
    tuple[_FamilyCommonCrossFitBinding, ...],
    tuple[CommonSenderApplication, ...],
]:
    design_by_contrast = {
        encoder.contrast.name: application
        for encoder, application in zip(
            design_encoders, design_applications, strict=True
        )
    }
    applications: list[FamilyCommonScoringApplication] = []
    bindings: list[_FamilyCommonCrossFitBinding] = []
    aligned_sender_applications: list[CommonSenderApplication] = []
    sender_applications: dict[
        tuple[str, str, tuple[str, ...]], CommonSenderApplication
    ] = {}
    for functional, receiver_program_application, incremental_application in zip(
        functionals,
        receiver_program_applications,
        incremental_applications,
        strict=True,
    ):
        design_application = design_by_contrast[functional.contrast_name]
        sender_key = (
            functional.contrast_name,
            functional.receiver,
            functional.interaction_ids,
        )
        sender_application = sender_applications.get(sender_key)
        if sender_application is None:
            sender_application = _completed_common_sender_application(
                functional.sender_functional,
                design_application,
                availability,
                receiver=functional.receiver,
                interaction_ids=functional.interaction_ids,
            )
            sender_applications[sender_key] = sender_application
        edge_evidence = _family_common_edge_evidence(
            functional,
            design_application,
            availability,
        )
        diagnostic = incremental_application.diagnostic_application
        if (
            functional.incremental_functional is None
            or diagnostic is None
            or incremental_application.diagnostic_status != "observed"
        ):
            application = mark_family_common_scoring_application_not_estimable(
                functional,
                edge_evidence,
                sender_application,
                heldout_reason_code=(
                    incremental_application.diagnostic_reason_code
                    or functional.incremental_reason_code
                    or incremental_application.reason_code
                    or "heldout_incremental_not_estimable"
                ),
                receiver_program_application=receiver_program_application,
            )
        else:
            application = apply_family_common_scoring_functional(
                functional,
                diagnostic,
                edge_evidence,
                sender_application,
                receiver_program_application=receiver_program_application,
            )
        applications.append(application)
        aligned_sender_applications.append(sender_application)
        bindings.append(
            _FamilyCommonCrossFitBinding._from_workflow(
                functional,
                application,
                training_application,
                edge_evidence,
                sender_application,
            )
        )
    return (
        tuple(applications),
        tuple(bindings),
        tuple(aligned_sender_applications),
    )


def _apply_cross_receiver_common_chains(
    functionals: tuple[CrossReceiverCommonScoringFunctional, ...],
    child_applications: tuple[FamilyCommonScoringApplication, ...],
    sender_applications: tuple[CommonSenderApplication, ...],
) -> tuple[CrossReceiverCommonScoringApplication, ...]:
    """Apply every global parent to its exact receiver child applications."""

    child_by_functional_id = {
        application.functional.family_common_functional_id: application
        for application in child_applications
    }
    sender_by_receiver_contrast = {
        (
            str(next(iter(set(application.table["receiver"].astype(str))))),
            application.functional.contrast_name,
        ): application
        for application in sender_applications
    }
    result: list[CrossReceiverCommonScoringApplication] = []
    for functional in functionals:
        children = tuple(
            child_by_functional_id[child.family_common_functional_id]
            for child in functional.child_functionals
        )
        senders = tuple(
            sender_by_receiver_contrast[(receiver, functional.contrast_name)]
            for receiver in functional.receiver_ids
        )
        result.append(
            apply_cross_receiver_common_scoring_functional(
                functional,
                children,
                senders,
            )
        )
    return tuple(result)


def _fold_coverage_rows(
    sample_metadata: pd.DataFrame,
    *,
    config: CrychicConfig,
    fold_id: str,
    test_subject_ids: tuple[str, ...],
    training: TrainingArtifacts,
    design_encoders: tuple[FrozenDesignEncoder, ...],
    design_applications: tuple[FrozenDesignApplication, ...],
) -> list[dict[str, object]]:
    heldout = sample_metadata.loc[
        sample_metadata[config.subject_key].astype(str).isin(test_subject_ids)
    ]
    rows: list[dict[str, object]] = []
    design_by_contrast = {
        _contrast_id(encoder.contrast): (encoder, application)
        for encoder, application in zip(
            design_encoders,
            design_applications,
            strict=True,
        )
    }
    for functional in training.sender_functionals:
        contrast_id = _contrast_id(functional.contrast)
        encoder, design_application = design_by_contrast[contrast_id]
        contrast_nodes = set(functional.contrast.weights)
        for _, sample in heldout.iterrows():
            node = _context_node(sample, tuple(config.context_keys))
            if node not in contrast_nodes:
                continue
            rows.append(
                {
                    "subject_id": str(sample[config.subject_key]),
                    "sample_id": str(sample[config.sample_key]),
                    "fold_id": fold_id,
                    "contrast_id": contrast_id,
                    "contrast_context": node,
                    "functional_status": "out_of_fold",
                    "sender_functional_id": functional.sender_functional_id,
                    "design_encoder_id": encoder.encoder_id,
                    "design_status": design_application.status,
                    "training_artifact_id": training.training_artifact_id,
                    "stage": _STAGE_NAME,
                }
            )
    return rows


def _receiver_coverage_rows(
    *,
    fold_id: str,
    design_encoders: tuple[FrozenDesignEncoder, ...],
    design_applications: tuple[FrozenDesignApplication, ...],
    responses: tuple[_ReceiverResponseArtifact, ...],
    precisions: tuple[PrecisionTransformResult, ...],
    incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...],
    response_applications: tuple[_ReceiverResponseApplication, ...],
    incremental_applications: tuple[ReceiverIncrementalApplication, ...],
    receiver_training_support: tuple[ReceiverTrainingSupportRecord, ...],
) -> list[dict[str, object]]:
    encoder_by_name = {
        encoder.contrast.name: (encoder, application)
        for encoder, application in zip(
            design_encoders, design_applications, strict=True
        )
    }
    rows: list[dict[str, object]] = []
    support_by_receiver = {
        record.receiver_id: record for record in receiver_training_support
    }
    for response, precision, model, response_application, application in zip(
        responses,
        precisions,
        incremental_models,
        response_applications,
        incremental_applications,
        strict=True,
    ):
        encoder, design_application = encoder_by_name[response.contrast_name]
        support = support_by_receiver[response.receiver]
        if support.status is not ReceiverTrainingSupportStatus.OBSERVED:
            raise ValueError("receiver model has a non-observed training support row")
        context_by_id = {
            node_context_fields(node, encoder.context_keys)[0]: node
            for node in encoder.contrast.weights
        }
        for sample_id, subject_id, context_id in zip(
            design_application.sample_ids,
            design_application.sample_subject_ids,
            design_application.sample_context_ids,
            strict=True,
        ):
            if context_id not in context_by_id:
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "sample_id": sample_id,
                    "fold_id": fold_id,
                    "contrast_id": _contrast_id(encoder.contrast),
                    "contrast_context": context_by_id[context_id],
                    "receiver": response.receiver,
                    "receiver_universe_id": support.receiver_universe_id,
                    "receiver_training_support_id": support.support_record_id,
                    "receiver_training_support_status": support.status.value,
                    "receiver_training_support_reason_code": support.reason_code,
                    "response_artifact_id": response.artifact_id,
                    "precision_transform_id": precision.precision_transform_id,
                    "incremental_training_artifact_id": model.training_artifact_id,
                    "response_application_id": response_application.application_id,
                    "design_application_id": design_application.application_id,
                    "incremental_application_id": application.application_id,
                    "diagnostic_status": application.diagnostic_status,
                    "diagnostic_reason_code": application.diagnostic_reason_code,
                    "official_incremental_status": (
                        application.official_incremental_status
                    ),
                    "reason_code": application.reason_code,
                    "stage": _RECEIVER_STAGE_NAME,
                }
            )
    for support in receiver_training_support:
        if support.status is not ReceiverTrainingSupportStatus.NOT_ESTIMABLE:
            continue
        if support.reason_code is None:  # pragma: no cover - support invariant
            raise RuntimeError("not-estimable receiver support requires a reason")
        for encoder, design_application in encoder_by_name.values():
            context_by_id = {
                node_context_fields(node, encoder.context_keys)[0]: node
                for node in encoder.contrast.weights
            }
            for sample_id, subject_id, context_id in zip(
                design_application.sample_ids,
                design_application.sample_subject_ids,
                design_application.sample_context_ids,
                strict=True,
            ):
                if context_id not in context_by_id:
                    continue
                rows.append(
                    {
                        "subject_id": subject_id,
                        "sample_id": sample_id,
                        "fold_id": fold_id,
                        "contrast_id": _contrast_id(encoder.contrast),
                        "contrast_context": context_by_id[context_id],
                        "receiver": support.receiver_id,
                        "receiver_universe_id": support.receiver_universe_id,
                        "receiver_training_support_id": support.support_record_id,
                        "receiver_training_support_status": support.status.value,
                        "receiver_training_support_reason_code": support.reason_code,
                        "response_artifact_id": None,
                        "precision_transform_id": None,
                        "incremental_training_artifact_id": None,
                        "response_application_id": None,
                        "design_application_id": design_application.application_id,
                        "incremental_application_id": None,
                        "diagnostic_status": "not_estimable",
                        "diagnostic_reason_code": support.reason_code,
                        "official_incremental_status": "not_estimable",
                        "reason_code": support.reason_code,
                        "stage": _RECEIVER_STAGE_NAME,
                    }
                )
    return rows


def _receiver_coverage_identity(
    coverage: pd.DataFrame,
    *,
    fold_plan_id: str,
) -> str:
    ordered = coverage.sort_values(
        ["fold_id", "sample_id", "contrast_id", "receiver", "subject_id"],
        kind="mergesort",
        ignore_index=True,
    )
    canonical_rows = [
        {
            column: (
                canonical_json(value)
                if column == "contrast_context"
                else None
                if pd.isna(value)
                else value
            )
            for column, value in row.items()
        }
        for row in ordered.to_dict(orient="records")
    ]
    payload = {
        "fold_plan_id": fold_plan_id,
        "grain": [
            "fold_id",
            "sample_id",
            "contrast_id",
            "receiver",
        ],
        "rows": canonical_rows,
        "stage": _RECEIVER_STAGE_NAME,
    }
    result: str = stable_id("receiver_oof_coverage_audit", payload, schema_version="1")
    return result


def _fold_sender_rows(
    *,
    fold_id: str,
    training: TrainingArtifacts,
    application: TrainingArtifactApplication,
) -> list[pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for assignment in application.sender_assignments:
        frame = assignment.table.copy(deep=True)
        frame["fold_id"] = fold_id
        frame["contrast_id"] = _contrast_id(assignment.functional.contrast)
        frame["training_artifact_id"] = training.training_artifact_id
        frame["functional_status"] = "out_of_fold"
        frame["stage"] = _STAGE_NAME
        rows.append(frame)
    return rows


def _positive_jobs(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer >= 1")
    return value


def _bounded_thread_map_in_order(
    function: Callable[[_WorkItemT], _WorkResultT],
    items: tuple[_WorkItemT, ...],
    *,
    n_jobs: int,
) -> tuple[_WorkResultT, ...]:
    """Run at most ``n_jobs`` submitted tasks and restore input order."""

    if not items:
        return ()
    workers = min(_positive_jobs(n_jobs, field_name="n_jobs"), len(items))
    if workers == 1:
        return tuple(function(item) for item in items)

    completed: dict[int, _WorkResultT] = {}
    item_iterator = iter(enumerate(items))
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="crychic-fold",
    ) as executor:
        pending: dict[Future[_WorkResultT], int] = {}

        def submit_next() -> bool:
            try:
                index, item = next(item_iterator)
            except StopIteration:
                return False
            pending[executor.submit(function, item)] = index
            return True

        for _ in range(workers):
            submit_next()
        try:
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                ordered_done = tuple(sorted(done, key=pending.__getitem__))
                for future in ordered_done:
                    index = pending.pop(future)
                    completed[index] = future.result()
                for _ in ordered_done:
                    submit_next()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return tuple(completed[index] for index in range(len(items)))


def _bounded_fork_map_in_order(
    function: Callable[[_WorkItemT], _WorkResultT],
    items: tuple[_WorkItemT, ...],
    *,
    n_jobs: int,
    on_start: Callable[[int], None] | None = None,
    on_complete: Callable[[int], None] | None = None,
) -> tuple[_WorkResultT, ...]:
    """Run inherited read-only tasks in bounded POSIX fork workers.

    This path is intentionally opt-in. It avoids serializing the large fold
    snapshot and bypasses the GIL for receiver-level Python work. It is only
    called from a single-threaded fold worker; callers must fall back to the
    thread map when the process already owns worker threads.
    """

    if not items:
        return ()
    workers = min(_positive_jobs(n_jobs, field_name="n_jobs"), len(items))
    if workers == 1:
        return tuple(function(item) for item in items)
    if not hasattr(os, "fork"):
        raise RuntimeError("receiver process workers require POSIX os.fork")
    # Some immutable contract fields use mappingproxy, which the standard
    # pickler cannot encode. Register the reducer only for this opt-in child
    # result channel, then restore the caller's reducer table before return.
    from types import MappingProxyType

    result_root = Path(tempfile.mkdtemp(prefix="crychic-receiver-fork-"))
    pending = iter(enumerate(items))
    active: dict[int, tuple[int, Path]] = {}
    results: dict[int, _WorkResultT] = {}

    def launch(index: int, item: _WorkItemT) -> None:
        output = result_root / f"receiver-{index:04d}.pickle"
        pid = os.fork()
        if pid == 0:
            try:
                with isolated_validation_scope():
                    result = function(item)
                payload: tuple[str, object] = ("ok", result)
            except BaseException:
                payload = ("error", traceback.format_exc())
            temporary = output.with_suffix(".tmp")
            try:
                with temporary.open("wb") as handle:
                    pickler = pickle.Pickler(handle, protocol=5)
                    pickler.dispatch_table = {
                        **copyreg.dispatch_table,
                        MappingProxyType: _reduce_mapping_proxy,
                    }
                    pickler.dump(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, output)
            except BaseException:
                traceback.print_exc(file=sys.stderr)
                os._exit(70)
            os._exit(0)
        active[pid] = (index, output)
        if on_start is not None:
            on_start(index)

    try:
        for _ in range(workers):
            try:
                index, item = next(pending)
            except StopIteration:
                break
            launch(index, item)
        while active:
            pid, wait_status = os.wait()
            if pid not in active:
                continue
            index, output = active.pop(pid)
            exit_code = os.waitstatus_to_exitcode(wait_status)
            if exit_code != 0 or not output.is_file():
                raise RuntimeError(
                    f"forked receiver {index} failed with exit code {exit_code}"
                )
            with output.open("rb") as handle:
                status, payload = pickle.load(handle)
            if status != "ok":
                raise RuntimeError(
                    f"forked receiver {index} raised an exception:\n{payload}"
                )
            results[index] = payload
            if on_complete is not None:
                on_complete(index)
            try:
                next_index, next_item = next(pending)
            except StopIteration:
                continue
            launch(next_index, next_item)
    except BaseException:
        for pid in active:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for pid in tuple(active):
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        raise
    finally:
        shutil.rmtree(result_root, ignore_errors=True)
    return tuple(results[index] for index in range(len(items)))


@dataclass(frozen=True, slots=True)
class _CrossFitFoldExecutionContext:
    snapshot: SanitizedRawInputSnapshot
    root_prepared: _PreparedRawFold
    config: CrychicConfig
    resource_bundle: ResourceBundle
    target_prior: TargetPrior
    spec: CrossFitSpec
    receiver_universe: FrozenReceiverUniverse
    receiver_family_opportunity_universe: FrozenReceiverFamilyOpportunityUniverse
    directional_lr_hypothesis_universe: FrozenReceiverFamilyLRHypothesisUniverse | None
    sample_metadata: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _CrossFitFoldExecutionResult:
    artifact: CrossFitFoldArtifacts
    coverage_rows: tuple[dict[str, object], ...]
    receiver_coverage_rows: tuple[dict[str, object], ...]
    sender_parts: tuple[pd.DataFrame, ...]


@dataclass(slots=True)
class _ProgressReporter:
    """Thread-safe lightweight progress/ETA telemetry for long fold stages."""

    total: int
    label: str
    path: Path | None = None
    started: float = field(default_factory=time.perf_counter)
    completed: int = 0
    active: int = 0
    _last_print: float = field(default=0.0, repr=False)
    _durations: list[float] = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def _write(self, payload: dict[str, object]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def event(self, event: str, *, detail: str | None = None) -> None:
        with self._lock:
            now = time.perf_counter()
            elapsed = max(0.0, now - self.started)
            average = (
                sum(self._durations) / len(self._durations) if self._durations else None
            )
            remaining = max(0, self.total - self.completed)
            eta = None if average is None else average * remaining
            payload: dict[str, object] = {
                "event": event,
                "label": self.label,
                "completed": self.completed,
                "total": self.total,
                "active_workers": self.active,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "detail": detail,
            }
            self._write(payload)
            if now - self._last_print >= 5.0 or event in {"start", "complete"}:
                eta_text = "unknown" if eta is None else f"{eta:.1f}s"
                print(
                    f"[crychic] {self.label} {self.completed}/{self.total} "
                    f"active={self.active} elapsed={elapsed:.1f}s eta={eta_text}",
                    file=sys.stderr,
                    flush=True,
                )
                self._last_print = now

    def task_started(self) -> float:
        with self._lock:
            self.active += 1
            self.event("task_start")
            return time.perf_counter()

    def task_completed(self, started: float, *, detail: str | None = None) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            self.completed += 1
            self._durations.append(max(0.0, time.perf_counter() - started))
            self.event("task_complete", detail=detail)


def _configured_jobs(name: str, *, default: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer >= 1") from error
    return _positive_jobs(value, field_name=name)


def _progress_enabled() -> bool:
    return os.environ.get("CRYCHIC_PROGRESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _progress_path_for_fold(fold_id: str) -> Path | None:
    raw = os.environ.get("CRYCHIC_PROGRESS_PATH")
    if raw is None or not raw.strip():
        return None
    path = Path(raw)
    return path.with_name(f"{path.stem}_{fold_id}{path.suffix or '.json'}")


def _run_crossfit_fold(
    fold: FoldManifest,
    *,
    context: _CrossFitFoldExecutionContext,
) -> _CrossFitFoldExecutionResult:
    """Fit and apply one outer fold from the shared immutable root snapshot."""

    snapshot = context.snapshot
    root_input_identity = snapshot.identity
    config = context.config
    resource_bundle = context.resource_bundle
    target_prior = context.target_prior
    spec = context.spec
    receiver_universe = context.receiver_universe
    receiver_family_opportunity_universe = context.receiver_family_opportunity_universe
    directional_lr_hypothesis_universe = context.directional_lr_hypothesis_universe
    sample_metadata = context.sample_metadata

    prepared_training = _subset_prepared_raw_fold(
        context.root_prepared,
        config,
        subject_ids=fold.train_subject_ids,
        min_cells=spec.training_spec.min_cells,
        root_input_identity=root_input_identity,
    )
    training = _fit_training_artifacts_from_prepared(
        prepared_training,
        config,
        resource_bundle,
        target_prior,
        spec=spec.training_spec,
    )
    receiver_training_support = assess_receiver_training_support(
        receiver_universe,
        outer_fold_id=fold.fold_id,
        training_cell_type_ids=training.cell_type_ids,
    )
    prepared_heldout = _subset_prepared_raw_fold(
        context.root_prepared,
        config,
        subject_ids=fold.test_subject_ids,
        min_cells=spec.training_spec.min_cells,
        cell_types=training.cell_type_ids,
        root_input_identity=root_input_identity,
    )
    application = _apply_training_artifacts_from_prepared(
        training,
        prepared_heldout,
    )
    training_metadata = sample_metadata.loc[
        sample_metadata[config.subject_key].astype(str).isin(fold.train_subject_ids)
    ]
    heldout_metadata = sample_metadata.loc[
        sample_metadata[config.subject_key].astype(str).isin(fold.test_subject_ids)
    ]
    design_encoders = tuple(
        fit_frozen_design_encoder(
            training_metadata,
            contrast=contrast,
            context_keys=config.context_keys,
            covariates=config.covariates,
            categorical_covariates=config.categorical_covariates,
            formula=config.design
            or default_design_formula(config.context_keys, config.covariates),
            sample_key=config.sample_key,
            subject_key=config.subject_key,
        )
        for contrast in spec.contrasts
    )
    design_applications = tuple(
        apply_frozen_design_encoder(encoder, heldout_metadata)
        for encoder in design_encoders
    )
    training_aggregate = prepared_training.aggregate
    heldout_aggregate = prepared_heldout.aggregate
    if not isinstance(training_aggregate, PseudobulkDataset) or not isinstance(
        heldout_aggregate, PseudobulkDataset
    ):
        raise RuntimeError(
            "raw-count cross-fitting requires inferential pseudobulk aggregates"
        )
    receiver_family_models = _training_receiver_families(
        prepared=prepared_training,
        training=training,
        resource_bundle=resource_bundle,
        target_prior=target_prior,
        spec=spec,
        fold_id=fold.fold_id,
        receiver_family_opportunity_universe=receiver_family_opportunity_universe,
        directional_lr_hypothesis_universe=directional_lr_hypothesis_universe,
    )
    receiver_family_applications = _apply_receiver_families(
        receiver_family_models,
        prepared=prepared_heldout,
        contrasts=spec.contrasts,
    )
    receiver_program_models = _training_receiver_programs(
        receiver_family_models,
        prepared=prepared_training,
        contrasts=spec.contrasts,
        minimum_scale=spec.downstream_minimum_scale,
    )
    receiver_program_applications = _apply_receiver_programs(
        receiver_program_models,
        prepared=prepared_heldout,
        design_encoders=design_encoders,
        design_applications=design_applications,
    )
    (
        receiver_responses,
        response_precisions,
        receiver_incremental_models,
    ) = _fit_receiver_incremental_chains(
        receiver_family_models,
        aggregate=training_aggregate,
        design_encoders=design_encoders,
        training_metadata=training_metadata,
        training_input_digest=prepared_training.input_digest,
        fold_id=fold.fold_id,
        minimum_scale=spec.downstream_minimum_scale,
        min_subjects_per_context=spec.min_train_subjects_per_context,
        autonomous_program_resource=spec.autonomous_program_resource,
        latent_nuisance_spec=spec.latent_nuisance_spec,
        penalty_tuning_spec=spec.penalty_tuning_spec,
        gain_calibration_spec=spec.gain_calibration_spec,
        inner_partition_seed_lineage=(_inner_tuning_partition_base_lineage(spec, fold)),
    )
    (
        receiver_response_applications,
        receiver_incremental_applications,
    ) = _apply_receiver_incremental_chains(
        receiver_responses,
        receiver_incremental_models,
        aggregate=heldout_aggregate,
        design_encoders=design_encoders,
        design_applications=design_applications,
    )
    directional_response_bindings = _build_directional_response_bindings(
        spec.directional_pairs,
        receiver_responses,
        receiver_incremental_models,
        receiver_incremental_applications,
        receiver_ids=training.cell_type_ids,
    )
    observed_contrasts = {
        _contrast_id(functional.contrast) for functional in training.sender_functionals
    }
    if observed_contrasts != set(fold.contrast_ids):
        raise ValueError(
            "training sender functionals do not match the planned contrasts"
        )
    if spec.penalty_tuning_spec is None:
        family_common_functionals: tuple[FamilyCommonScoringFunctional, ...] = ()
        family_common_applications: tuple[FamilyCommonScoringApplication, ...] = ()
        family_common_bindings: tuple[_FamilyCommonCrossFitBinding, ...] = ()
        cross_receiver_common_functionals: tuple[
            CrossReceiverCommonScoringFunctional, ...
        ] = ()
        cross_receiver_common_applications: tuple[
            CrossReceiverCommonScoringApplication, ...
        ] = ()
    else:
        family_common_functionals = _fit_family_common_chains(
            receiver_family_models,
            receiver_program_models,
            receiver_incremental_models,
            sender_functionals=training.sender_functionals,
        )
        all_receivers_supported = all(
            record.status is ReceiverTrainingSupportStatus.OBSERVED
            for record in receiver_training_support
        )
        cross_receiver_common_functionals = (
            _fit_cross_receiver_common_chains(
                family_common_functionals,
                receiver_incremental_models,
                planned_receiver_ids=receiver_universe.receiver_ids,
            )
            if all_receivers_supported
            else ()
        )
        (
            family_common_applications,
            family_common_bindings,
            family_common_sender_applications,
        ) = _apply_family_common_chains(
            family_common_functionals,
            receiver_program_applications,
            receiver_incremental_applications,
            training_application=application,
            design_encoders=design_encoders,
            design_applications=design_applications,
            availability=application.availability.sample_interactions,
        )
        cross_receiver_common_applications = (
            _apply_cross_receiver_common_chains(
                cross_receiver_common_functionals,
                family_common_applications,
                family_common_sender_applications,
            )
            if cross_receiver_common_functionals
            else ()
        )
    artifact = CrossFitFoldArtifacts(
        fold_id=fold.fold_id,
        training=training,
        application=application,
        receiver_training_support=receiver_training_support,
        design_encoders=design_encoders,
        design_applications=design_applications,
        receiver_family_models=receiver_family_models,
        receiver_family_applications=receiver_family_applications,
        receiver_program_models=receiver_program_models,
        receiver_program_applications=receiver_program_applications,
        receiver_responses=receiver_responses,
        response_precisions=response_precisions,
        receiver_incremental_models=receiver_incremental_models,
        receiver_response_applications=receiver_response_applications,
        receiver_incremental_applications=receiver_incremental_applications,
        family_common_functionals=family_common_functionals,
        family_common_applications=family_common_applications,
        family_common_bindings=family_common_bindings,
        directional_response_bindings=directional_response_bindings,
        cross_receiver_common_functionals=cross_receiver_common_functionals,
        cross_receiver_common_applications=cross_receiver_common_applications,
    )
    return _CrossFitFoldExecutionResult(
        artifact=artifact,
        coverage_rows=tuple(
            _fold_coverage_rows(
                sample_metadata,
                config=config,
                fold_id=fold.fold_id,
                test_subject_ids=fold.test_subject_ids,
                training=training,
                design_encoders=design_encoders,
                design_applications=design_applications,
            )
        ),
        receiver_coverage_rows=tuple(
            _receiver_coverage_rows(
                fold_id=fold.fold_id,
                design_encoders=design_encoders,
                design_applications=design_applications,
                responses=receiver_responses,
                precisions=response_precisions,
                incremental_models=receiver_incremental_models,
                response_applications=receiver_response_applications,
                incremental_applications=receiver_incremental_applications,
                receiver_training_support=receiver_training_support,
            )
        ),
        sender_parts=tuple(
            _fold_sender_rows(
                fold_id=fold.fold_id,
                training=training,
                application=application,
            )
        ),
    )


def _run_subject_crossfit(
    snapshot: SanitizedRawInputSnapshot,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: CrossFitSpec,
    n_jobs: int = 1,
    _receiver_axis_source: FrozenReceiverUniverse | None = None,
) -> CrossFitArtifacts:
    """Run subject-blocked train/apply and verify the current sender audit scope.

    The function accepts no caller-created folds, fitted values, matrices, or
    provenance identifiers.  Raw-count input is required because normalized
    input cannot establish a certified train-only transform boundary.
    """

    jobs = _positive_jobs(n_jobs, field_name="n_jobs")
    if not isinstance(snapshot, SanitizedRawInputSnapshot):
        raise TypeError("snapshot must be a SanitizedRawInputSnapshot")
    snapshot._require_intact()
    adata = snapshot.adata
    root_input_identity = snapshot.identity
    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig")
    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior")
    if not isinstance(spec, CrossFitSpec):
        raise TypeError("spec must be a CrossFitSpec")
    spec._require_intact()
    if _receiver_axis_source is not None:
        if not isinstance(_receiver_axis_source, FrozenReceiverUniverse):
            raise TypeError("_receiver_axis_source must be a FrozenReceiverUniverse")
        _receiver_axis_source._require_intact()
        if (
            spec.predeclared_receiver_ids is not None
            and spec.predeclared_receiver_ids != _receiver_axis_source.receiver_ids
        ):
            raise ContractError(
                "Cross-fit receiver declaration differs from the reused source axis",
                code="crossfit_receiver_axis_source_mismatch",
                field="predeclared_receiver_ids",
                remediation=(
                    "Use the exact source receiver IDs for every resampled child"
                ),
            )
    if spec.autonomous_program_resource is not None and (
        spec.autonomous_program_resource.species is not resource_bundle.species
        or spec.autonomous_program_resource.gene_namespace
        is not resource_bundle.gene_namespace
    ):
        raise ValueError(
            "autonomous program resource species and namespace must match LR resources"
        )

    validated = validate_anndata(adata, _input_schema(config))
    if validated.mode is not InputMode.COUNTS:
        raise ValueError(
            "subject cross-fitting requires raw counts; normalized-only input "
            "cannot certify train-only preprocessing"
        )
    sanitized = adata
    observed_root_cell_types = tuple(
        sorted(sanitized.obs[config.cell_type_key].astype(str).unique())
    )
    receiver_universe = freeze_receiver_universe(
        root_input_identity,
        observed_root_cell_types,
        predeclared_receiver_ids=(
            spec.predeclared_receiver_ids
            if _receiver_axis_source is None
            else _receiver_axis_source.receiver_ids
        ),
    )
    if (
        _receiver_axis_source is not None
        and receiver_universe.receiver_axis_id != _receiver_axis_source.receiver_axis_id
    ):
        raise ContractError(
            "Resampled cross-fit did not preserve the frozen receiver axis",
            code="crossfit_receiver_axis_reuse_mismatch",
            field="receiver_axis_id",
            remediation="Rerun the child with the complete source receiver axis",
        )
    receiver_family_opportunity_universe = freeze_receiver_family_opportunity_universe(
        target_prior,
        feature_ids=validated.feature_ids,
        receiver_ids=receiver_universe.receiver_ids,
        receiver_universe_id=receiver_universe.universe_id,
        receiver_axis_id=receiver_universe.receiver_axis_id,
        prior_content_id=_target_prior_content_id(target_prior),
        root_input_identity_id=root_input_identity.identity_id,
        root_input_digest=root_input_identity.input_digest,
        cosine_threshold=spec.family_cosine_threshold,
    )
    directional_lr_hypothesis_universe: (
        FrozenReceiverFamilyLRHypothesisUniverse | None
    ) = None
    if spec.directional_pairs:
        directional_lr_hypothesis_universe = (
            freeze_receiver_family_lr_hypothesis_universe(
                receiver_family_opportunity_universe,
                resource_bundle,
                target_prior,
                resource_bundle_content_id=(
                    _resource_bundle_content_id(resource_bundle)
                ),
                target_prior_content_id=_target_prior_content_id(target_prior),
            )
        )
    invalid_strata = set(spec.strata_keys).difference(config.covariates)
    if invalid_strata:
        raise ValueError(
            "strata_keys must be declared immutable covariates; invalid="
            f"{sorted(invalid_strata)}"
        )
    missing_strata = set(spec.strata_keys).difference(
        validated.report.sample_metadata.columns
    )
    if missing_strata:
        raise ValueError(f"sample metadata is missing strata: {sorted(missing_strata)}")

    checker = DesignFoldChecker(
        context_keys=tuple(config.context_keys),
        covariates=tuple(config.covariates),
        categorical_covariates=tuple(config.categorical_covariates),
        formula=config.design
        or default_design_formula(config.context_keys, config.covariates),
        contrasts=spec.contrasts,
        sample_key=config.sample_key,
    )
    fold_plan = plan_subject_folds(
        validated.report.sample_metadata,
        design_checker=checker,
        subject_key=config.subject_key,
        sample_key=config.sample_key,
        context_keys=config.context_keys,
        strata_keys=spec.strata_keys,
        allowed_n_splits=spec.allowed_n_splits,
        min_train_subjects_per_context=spec.min_train_subjects_per_context,
        min_test_subjects_per_context=spec.min_test_subjects_per_context,
        repeat_id=spec.repeat_id,
        seed_lineage=SeedLineage(config.random_seed).derive(
            "subject_crossfit", spec.spec_id
        ),
        partition_seed_lineage=_outer_fold_partition_lineage(spec),
    )
    if root_input_identity.config_digest != config.digest:
        raise ValueError("snapshot root identity does not match the cross-fit config")
    fold_artifacts: list[CrossFitFoldArtifacts] = []
    coverage_rows: list[dict[str, object]] = []
    receiver_coverage_rows: list[dict[str, object]] = []
    sender_parts: list[pd.DataFrame] = []
    fold_context = _CrossFitFoldExecutionContext(
        snapshot=snapshot,
        root_prepared=_prepare_validated_root_fold(
            validated,
            config,
            min_cells=spec.training_spec.min_cells,
            root_input_identity=root_input_identity,
        ),
        config=config,
        resource_bundle=resource_bundle,
        target_prior=target_prior,
        spec=spec,
        receiver_universe=receiver_universe,
        receiver_family_opportunity_universe=receiver_family_opportunity_universe,
        directional_lr_hypothesis_universe=directional_lr_hypothesis_universe,
        sample_metadata=validated.report.sample_metadata,
    )

    planned_folds = tuple(fold_plan)
    effective_jobs = min(jobs, len(planned_folds))
    fold_reporter = (
        _ProgressReporter(
            total=len(planned_folds),
            label="crossfit:outer_folds",
            path=_progress_path_for_fold("outer"),
        )
        if (_progress_enabled() or os.environ.get("CRYCHIC_PROGRESS_PATH"))
        and planned_folds
        else None
    )

    def execute(fold: FoldManifest) -> _CrossFitFoldExecutionResult:
        started = (
            fold_reporter.task_started() if fold_reporter is not None else None
        )
        try:
            with _receiver_family_validation_scope(), validation_scope():
                return _run_crossfit_fold(fold, context=fold_context)
        finally:
            if fold_reporter is not None and started is not None:
                fold_reporter.task_completed(started, detail=fold.fold_id)

    if fold_reporter is not None:
        fold_reporter.event("start")
    executions = _bounded_thread_map_in_order(
        execute,
        planned_folds,
        n_jobs=effective_jobs,
    )
    if fold_reporter is not None:
        fold_reporter.event("complete")
    for execution in executions:
        fold_artifacts.append(execution.artifact)
        coverage_rows.extend(execution.coverage_rows)
        receiver_coverage_rows.extend(execution.receiver_coverage_rows)
        sender_parts.extend(execution.sender_parts)
    coverage = pd.DataFrame(coverage_rows, columns=_COVERAGE_COLUMNS)
    receiver_coverage = pd.DataFrame(
        receiver_coverage_rows, columns=_RECEIVER_COVERAGE_COLUMNS
    )
    sender_columns = (
        *COMMON_SENDER_APPLICATION_COLUMNS,
        *_ASSIGNMENT_PROVENANCE_COLUMNS,
    )
    assignments = (
        pd.concat(sender_parts, ignore_index=True).loc[:, list(sender_columns)]
        if sender_parts
        else pd.DataFrame(columns=sender_columns)
    )
    contrast_contexts = {
        _contrast_id(contrast): tuple(contrast.weights) for contrast in spec.contrasts
    }
    audit = validate_oof_subject_coverage(
        coverage,
        fold_plan,
        contrast_contexts=contrast_contexts,
        context_column="contrast_context",
        scoring_function_column="sender_functional_id",
        model_manifest_column="training_artifact_id",
    )
    return CrossFitArtifacts._from_workflow(
        spec=spec,
        root_input_identity=root_input_identity,
        receiver_universe=receiver_universe,
        receiver_family_opportunity_universe=(receiver_family_opportunity_universe),
        directional_lr_hypothesis_universe=(directional_lr_hypothesis_universe),
        fold_plan=fold_plan,
        folds=tuple(fold_artifacts),
        oof_coverage=coverage,
        oof_receiver_coverage=receiver_coverage,
        oof_sender_assignments=assignments,
        coverage_audit=audit,
    )


@validation_scope()
def run_subject_crossfit(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: CrossFitSpec,
    n_jobs: int = 1,
) -> CrossFitArtifacts:
    """Run one public subject-blocked cross-fit from complete raw counts.

    ``n_jobs`` bounds concurrent outer folds over one immutable sanitized input
    snapshot. It affects execution only, never scientific result identity. Each
    worker retains fold-local model state, so memory may scale with the effective
    worker count and the default remains one.
    """

    jobs = _positive_jobs(n_jobs, field_name="n_jobs")
    return _run_subject_crossfit(
        _sanitized_raw_input_snapshot(adata, config),
        config,
        resource_bundle,
        target_prior,
        spec=spec,
        n_jobs=jobs,
    )


__all__ = [
    "AutonomousProgramUseScope",
    "CrossFitArtifacts",
    "CrossFitFoldArtifacts",
    "CrossFitSpec",
    "run_subject_crossfit",
]
