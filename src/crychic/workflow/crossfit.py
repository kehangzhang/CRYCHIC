"""Public subject-blocked orchestration for implemented train/apply stages.

Separate exact-coverage audits verify contrast-common sender rows and the
typed receiver response/precision/incremental diagnostic chain.  The receiver
chain remains officially not estimable until its autonomous nuisance model is
fold-frozen, so these artifacts do not certify the complete scoring pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from crychic.attribution import (
    PrecisionTransformResult,
    fit_receiver_family_training_artifacts,
    fit_response_precision,
)
from crychic.core import (
    ContractError,
    CrychicConfig,
    SeedLineage,
    canonical_json,
    stable_id,
)
from crychic.data import InputMode, validate_anndata
from crychic.design import (
    ContrastSpec,
    FrozenDesignApplication,
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    canonical_context,
    default_design_formula,
    fit_frozen_design_encoder,
    node_context_fields,
    plain_context_value,
)
from crychic.pseudobulk import PseudobulkDataset
from crychic.resampling import (
    DesignFoldChecker,
    OOFCoverageAudit,
    SubjectFoldPlan,
    plan_subject_folds,
    validate_oof_subject_coverage,
)
from crychic.resources import ResourceBundle, TargetPrior
from crychic.response import (
    FoldGeneResponseApplication,
    FoldGeneResponseArtifact,
    apply_fold_gene_response,
    fit_fold_gene_response,
)
from crychic.scoring import (
    ReceiverFamilyScoringApplication,
    ReceiverFamilyScoringArtifact,
    apply_receiver_family_scoring_artifact,
    fit_receiver_family_scoring_artifact,
    mark_receiver_family_application_not_estimable,
    mark_receiver_family_scoring_not_estimable,
)
from crychic.sender import COMMON_SENDER_APPLICATION_COLUMNS

from .application import TrainingArtifactApplication, apply_training_artifacts
from .receiver_incremental import (
    ReceiverIncrementalApplication,
    ReceiverIncrementalTrainingArtifact,
    apply_receiver_incremental_training_artifact,
    fit_receiver_incremental_training_artifact,
)
from .training import (
    FoldTrainingSpec,
    TrainingArtifacts,
    _fit_interaction_universe,
    _input_schema,
    _prepare_raw_fold,
    _PreparedRawFold,
    _sanitize_validated_input,
    fit_training_artifacts,
)

_STAGE_NAME = "contrast_common_sender_application"
_RECEIVER_STAGE_NAME = "receiver_incremental_application_diagnostic"
_STAGE_STATUS = "verified_train_only_oof_partial_pipeline"
_PRODUCER_MARKER = "crychic.workflow.crossfit.v1"
_REMAINING_PUBLIC_STAGES = (
    "attribution_tuning",
    "common_scoring_functional",
    "family_attribution",
    "incremental_downstream",
    "receiver_autonomous_nuisance",
    "subject_blocked_inner_tuning",
)
_CPM_SCALE = 1_000_000.0
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
    training_spec: FoldTrainingSpec = field(default_factory=FoldTrainingSpec)
    strata_keys: tuple[str, ...] = ()
    allowed_n_splits: tuple[int, ...] = (5, 4, 3, 2)
    min_train_subjects_per_context: int = 2
    min_test_subjects_per_context: int = 1
    receptor_gate_threshold: float = 0.1
    family_cosine_threshold: float = 0.95
    downstream_minimum_scale: float = 0.25
    schema_version: str = "1.0.0"
    spec_id: str = field(init=False)
    repeat_id: str = field(init=False)

    def __post_init__(self) -> None:
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
        if not isinstance(self.training_spec, FoldTrainingSpec):
            raise TypeError("training_spec must be a FoldTrainingSpec")
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
        spec_id = stable_id(
            "subject_crossfit_spec", payload, schema_version=self.schema_version
        )
        object.__setattr__(self, "contrasts", contrasts)
        object.__setattr__(self, "training_spec", training_spec)
        object.__setattr__(self, "strata_keys", strata)
        object.__setattr__(self, "allowed_n_splits", allowed)
        object.__setattr__(self, "receptor_gate_threshold", gate_threshold)
        object.__setattr__(self, "family_cosine_threshold", family_threshold)
        object.__setattr__(self, "downstream_minimum_scale", minimum_scale)
        object.__setattr__(self, "spec_id", spec_id)
        object.__setattr__(
            self,
            "repeat_id",
            stable_id("subject_crossfit_repeat", {"spec_id": spec_id, "repeat": 0}),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the pre-registered cross-fit policy manifest."""

        return {
            "spec_id": self.spec_id,
            "repeat_id": self.repeat_id,
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
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossFitFoldArtifacts:
    """Training and held-out application artifacts for one planned fold."""

    fold_id: str
    training: TrainingArtifacts
    application: TrainingArtifactApplication
    design_encoders: tuple[FrozenDesignEncoder, ...]
    design_applications: tuple[FrozenDesignApplication, ...]
    receiver_family_models: tuple[ReceiverFamilyScoringArtifact, ...]
    receiver_family_applications: tuple[ReceiverFamilyScoringApplication, ...]
    receiver_responses: tuple[FoldGeneResponseArtifact, ...]
    response_precisions: tuple[PrecisionTransformResult, ...]
    receiver_incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...]
    receiver_response_applications: tuple[FoldGeneResponseApplication, ...]
    receiver_incremental_applications: tuple[ReceiverIncrementalApplication, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise ValueError("fold_id must be a non-empty identifier")
        if not isinstance(self.training, TrainingArtifacts):
            raise TypeError("training must be TrainingArtifacts")
        if not isinstance(self.application, TrainingArtifactApplication):
            raise TypeError("application must be TrainingArtifactApplication")
        if self.application.training_artifact_id != self.training.training_artifact_id:
            raise ValueError("fold application is not bound to its training artifacts")
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
            if model.training_subject_ids != self.training.training_subject_ids:
                raise ValueError(
                    "receiver-family training subjects do not match the fold"
                )
            if receiver_application.training_artifact_id != model.training_artifact_id:
                raise ValueError(
                    "receiver-family application does not match its training model"
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
        design_by_name = {
            encoder.contrast.name: (encoder, application)
            for encoder, application in zip(encoders, applications, strict=True)
        }
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


@dataclass(frozen=True, slots=True, init=False)
class CrossFitArtifacts:
    """Partial artifacts with exact sender-stage OOF coverage verification."""

    spec: CrossFitSpec
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
        fold_plan: SubjectFoldPlan,
        folds: tuple[CrossFitFoldArtifacts, ...],
        oof_coverage: pd.DataFrame,
        oof_receiver_coverage: pd.DataFrame,
        oof_sender_assignments: pd.DataFrame,
        coverage_audit: OOFCoverageAudit,
    ) -> CrossFitArtifacts:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "spec": spec,
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
        if not isinstance(self.fold_plan, SubjectFoldPlan):
            raise TypeError("fold_plan must be a SubjectFoldPlan")
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
                str(row.response_artifact_id),
                str(row.precision_transform_id),
                str(row.incremental_training_artifact_id),
                str(row.response_application_id),
                str(row.design_application_id),
                str(row.incremental_application_id),
                str(row.diagnostic_status),
                (
                    None
                    if pd.isna(row.diagnostic_reason_code)
                    else str(row.diagnostic_reason_code)
                ),
                str(row.official_incremental_status),
                str(row.reason_code),
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
                or str(row.official_incremental_status) != "not_estimable"
                or str(row.reason_code) != "receiver_autonomous_nuisance_not_frozen"
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
        payload = {
            "coverage_audit_id": self.coverage_audit.audit_id,
            "coverage_table_digest": coverage_table_digest,
            "receiver_coverage_audit_id": receiver_audit_id,
            "receiver_coverage_table_digest": receiver_coverage_table_digest,
            "fold_applications": [
                {
                    "fold_id": item.fold_id,
                    "heldout_input_digest": item.application.heldout_input_digest,
                    "training_artifact_id": item.training.training_artifact_id,
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
                }
                for item in sorted(folds, key=lambda value: value.fold_id)
            ],
            "fold_plan_id": self.fold_plan.plan_id,
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
        object.__setattr__(self, "crossfit_id", stable_id("subject_crossfit", payload))

    @property
    def oof_coverage(self) -> pd.DataFrame:
        """Return a defensive copy of the sender-stage OOF coverage table."""

        return self._oof_coverage.copy(deep=True)

    @property
    def oof_receiver_coverage(self) -> pd.DataFrame:
        """Return a defensive copy of the exact receiver-stage coverage table."""

        return self._oof_receiver_coverage.copy(deep=True)

    @property
    def oof_sender_assignments(self) -> pd.DataFrame:
        """Return a defensive copy of held-out common-sender assignments."""

        return self._oof_sender_assignments.copy(deep=True)

    def _require_intact(self) -> None:
        try:
            repeated = CrossFitArtifacts._from_workflow(
                spec=self.spec,
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
                and repeated.crossfit_id == self.crossfit_id
                and repeated.receiver_coverage_audit_id
                == self.receiver_coverage_audit_id
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

    @property
    def completed_stage_oof_verified(self) -> bool:
        """Whether the availability/sender stage passed exact OOF coverage audit."""

        return True

    @property
    def is_oof_certified(self) -> bool:
        """Return false until all scoring and downstream stages are cross-fitted."""

        return False

    def to_manifest(self) -> dict[str, object]:
        """Return an auditable summary without embedding tabular payloads."""

        self._require_intact()
        return {
            "crossfit_id": self.crossfit_id,
            "certification_status": self.certification_status,
            "completed_stage_oof_verified": self.completed_stage_oof_verified,
            "complete_pipeline_oof_certified": self.is_oof_certified,
            "oof_audit_scope": _STAGE_NAME,
            "spec": self.spec.to_dict(),
            "fold_plan": self.fold_plan.to_dict(),
            "coverage_audit": self.coverage_audit.to_dict(),
            "coverage_table_digest": self.coverage_table_digest,
            "receiver_coverage_audit_id": self.receiver_coverage_audit_id,
            "receiver_coverage_table_digest": self.receiver_coverage_table_digest,
            "sender_assignment_table_digest": self.sender_assignment_table_digest,
            "fold_artifacts": [
                {
                    "fold_id": item.fold_id,
                    "training_artifact_id": item.training.training_artifact_id,
                    "heldout_input_digest": item.application.heldout_input_digest,
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
                            "training_artifact_id": model.training_artifact_id,
                            "response_application_id": (
                                response_application.application_id
                            ),
                            "application_id": application.application_id,
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
            "remaining_stages": list(_REMAINING_PUBLIC_STAGES),
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
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Extract fixed log1p(CPM) sample expression for one receiver."""

    aggregate = prepared.aggregate
    if not isinstance(aggregate, PseudobulkDataset):
        raise TypeError("public cross-fit receiver expression requires count aggregate")
    metadata = aggregate.unit_metadata.copy(deep=True).reset_index(drop=True)
    selected = metadata["cell_type"].map(lambda value: str(value) == receiver)
    if contexts is not None:
        context_nodes = metadata["context"].map(
            lambda value: _context_node(
                pd.Series(dict(value)), tuple(prepared.validated.schema.context_keys)
            )
        )
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
    return values, sample_ids, subject_ids


def _training_receiver_families(
    *,
    prepared: _PreparedRawFold,
    training: TrainingArtifacts,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    spec: CrossFitSpec,
    fold_id: str,
) -> tuple[ReceiverFamilyScoringArtifact, ...]:
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
    mapping = _interaction_driver_mapping(resource_bundle, target_prior)
    receivers = prepared.cell_type_ids
    receiver_families = {
        artifact.receiver: artifact
        for artifact in fit_receiver_family_training_artifacts(
            availability,
            target_prior,
            receivers=receivers,
            fold_id=fold_id,
            feature_ids=prepared.aggregate.feature_ids,
            driver_by_interaction=mapping,
            receptor_gate_threshold=spec.receptor_gate_threshold,
            cosine_threshold=spec.family_cosine_threshold,
        )
    }
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
            try:
                scoring_model = fit_receiver_family_scoring_artifact(
                    receiver_family,
                    reference_expression,
                    contrast_name=contrast.name,
                    sample_ids=reference_sample_ids,
                    sample_subject_ids=reference_subjects,
                    minimum_scale=spec.downstream_minimum_scale,
                )
            except ValueError as error:
                if "at least two complete finite samples" not in str(error):
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
            expression, _, subject_ids = _response_expression(
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
        observed_subjects = set(subject_ids)
        missing_subjects = set(heldout_subject_ids).difference(observed_subjects)
        if missing_subjects:
            applications.append(
                mark_receiver_family_application_not_estimable(
                    model,
                    heldout_subject_ids=heldout_subject_ids,
                    reason_code="heldout_receiver_expression_incomplete_subject_coverage",
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


def _fit_receiver_incremental_chains(
    models: tuple[ReceiverFamilyScoringArtifact, ...],
    *,
    aggregate: PseudobulkDataset,
    design_encoders: tuple[FrozenDesignEncoder, ...],
    training_input_digest: str,
    fold_id: str,
    minimum_scale: float,
    min_subjects_per_context: int,
) -> tuple[
    tuple[FoldGeneResponseArtifact, ...],
    tuple[PrecisionTransformResult, ...],
    tuple[ReceiverIncrementalTrainingArtifact, ...],
]:
    encoder_by_name = {encoder.contrast.name: encoder for encoder in design_encoders}
    responses: list[FoldGeneResponseArtifact] = []
    precisions: list[PrecisionTransformResult] = []
    incremental_models: list[ReceiverIncrementalTrainingArtifact] = []
    for model in models:
        encoder = encoder_by_name[model.contrast_name]
        response = fit_fold_gene_response(
            aggregate,
            encoder,
            receiver=model.receiver_family_artifact.receiver,
            fold_id=fold_id,
            training_input_digest=training_input_digest,
            min_subjects_per_context=max(2, min_subjects_per_context),
        )
        precision = fit_response_precision(response)
        incremental_model = fit_receiver_incremental_training_artifact(
            encoder,
            response,
            precision,
            model.receiver_family_artifact,
            minimum_scale=minimum_scale,
        )
        responses.append(response)
        precisions.append(precision)
        incremental_models.append(incremental_model)
    return tuple(responses), tuple(precisions), tuple(incremental_models)


def _apply_receiver_incremental_chains(
    responses: tuple[FoldGeneResponseArtifact, ...],
    incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...],
    *,
    aggregate: PseudobulkDataset,
    design_encoders: tuple[FrozenDesignEncoder, ...],
    design_applications: tuple[FrozenDesignApplication, ...],
) -> tuple[
    tuple[FoldGeneResponseApplication, ...],
    tuple[ReceiverIncrementalApplication, ...],
]:
    design_by_name = {
        encoder.contrast.name: application
        for encoder, application in zip(
            design_encoders, design_applications, strict=True
        )
    }
    response_applications: list[FoldGeneResponseApplication] = []
    incremental_applications: list[ReceiverIncrementalApplication] = []
    for response, model in zip(responses, incremental_models, strict=True):
        design_application = design_by_name[response.contrast_name]
        response_application = apply_fold_gene_response(
            aggregate, response, design_application
        )
        incremental_application = apply_receiver_incremental_training_artifact(
            model, response_application, design_application
        )
        response_applications.append(response_application)
        incremental_applications.append(incremental_application)
    return tuple(response_applications), tuple(incremental_applications)


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
    responses: tuple[FoldGeneResponseArtifact, ...],
    precisions: tuple[PrecisionTransformResult, ...],
    incremental_models: tuple[ReceiverIncrementalTrainingArtifact, ...],
    response_applications: tuple[FoldGeneResponseApplication, ...],
    incremental_applications: tuple[ReceiverIncrementalApplication, ...],
) -> list[dict[str, object]]:
    encoder_by_name = {
        encoder.contrast.name: (encoder, application)
        for encoder, application in zip(
            design_encoders, design_applications, strict=True
        )
    }
    rows: list[dict[str, object]] = []
    for response, precision, model, response_application, application in zip(
        responses,
        precisions,
        incremental_models,
        response_applications,
        incremental_applications,
        strict=True,
    ):
        encoder, design_application = encoder_by_name[response.contrast_name]
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


def run_subject_crossfit(
    adata: AnnData,
    config: CrychicConfig,
    resource_bundle: ResourceBundle,
    target_prior: TargetPrior,
    *,
    spec: CrossFitSpec,
) -> CrossFitArtifacts:
    """Run subject-blocked train/apply and verify the current sender audit scope.

    The function accepts no caller-created folds, fitted values, matrices, or
    provenance identifiers.  Raw-count input is required because normalized
    input cannot establish a certified train-only transform boundary.
    """

    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData instance")
    if not isinstance(config, CrychicConfig):
        raise TypeError("config must be a CrychicConfig")
    if not isinstance(resource_bundle, ResourceBundle):
        raise TypeError("resource_bundle must be a ResourceBundle")
    if not isinstance(target_prior, TargetPrior):
        raise TypeError("target_prior must be a TargetPrior")
    if not isinstance(spec, CrossFitSpec):
        raise TypeError("spec must be a CrossFitSpec")

    validated = validate_anndata(adata, _input_schema(config))
    if validated.mode is not InputMode.COUNTS:
        raise ValueError(
            "subject cross-fitting requires raw counts; normalized-only input "
            "cannot certify train-only preprocessing"
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
    )
    sanitized = _sanitize_validated_input(validated)
    fold_artifacts: list[CrossFitFoldArtifacts] = []
    coverage_rows: list[dict[str, object]] = []
    receiver_coverage_rows: list[dict[str, object]] = []
    sender_parts: list[pd.DataFrame] = []
    for fold in fold_plan:
        training_scope = _physical_subject_scope(
            sanitized,
            subject_key=config.subject_key,
            subject_ids=fold.train_subject_ids,
        )
        heldout_scope = _physical_subject_scope(
            sanitized,
            subject_key=config.subject_key,
            subject_ids=fold.test_subject_ids,
        )
        training = fit_training_artifacts(
            training_scope,
            config,
            resource_bundle,
            target_prior,
            spec=spec.training_spec,
        )
        application = apply_training_artifacts(training, heldout_scope)
        sample_metadata = validated.report.sample_metadata
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
        prepared_training = _prepare_raw_fold(
            training_scope,
            config,
            min_cells=spec.training_spec.min_cells,
            cell_types=training.cell_type_ids,
        )
        prepared_heldout = _prepare_raw_fold(
            heldout_scope,
            config,
            min_cells=spec.training_spec.min_cells,
            cell_types=training.cell_type_ids,
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
        )
        receiver_family_applications = _apply_receiver_families(
            receiver_family_models,
            prepared=prepared_heldout,
            contrasts=spec.contrasts,
        )
        (
            receiver_responses,
            response_precisions,
            receiver_incremental_models,
        ) = _fit_receiver_incremental_chains(
            receiver_family_models,
            aggregate=training_aggregate,
            design_encoders=design_encoders,
            training_input_digest=prepared_training.input_digest,
            fold_id=fold.fold_id,
            minimum_scale=spec.downstream_minimum_scale,
            min_subjects_per_context=spec.min_train_subjects_per_context,
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
        observed_contrasts = {
            _contrast_id(functional.contrast)
            for functional in training.sender_functionals
        }
        if observed_contrasts != set(fold.contrast_ids):
            raise ValueError(
                "training sender functionals do not match the planned contrasts"
            )
        fold_artifacts.append(
            CrossFitFoldArtifacts(
                fold_id=fold.fold_id,
                training=training,
                application=application,
                design_encoders=design_encoders,
                design_applications=design_applications,
                receiver_family_models=receiver_family_models,
                receiver_family_applications=receiver_family_applications,
                receiver_responses=receiver_responses,
                response_precisions=response_precisions,
                receiver_incremental_models=receiver_incremental_models,
                receiver_response_applications=receiver_response_applications,
                receiver_incremental_applications=(receiver_incremental_applications),
            )
        )
        coverage_rows.extend(
            _fold_coverage_rows(
                validated.report.sample_metadata,
                config=config,
                fold_id=fold.fold_id,
                test_subject_ids=fold.test_subject_ids,
                training=training,
                design_encoders=design_encoders,
                design_applications=design_applications,
            )
        )
        receiver_coverage_rows.extend(
            _receiver_coverage_rows(
                fold_id=fold.fold_id,
                design_encoders=design_encoders,
                design_applications=design_applications,
                responses=receiver_responses,
                precisions=response_precisions,
                incremental_models=receiver_incremental_models,
                response_applications=receiver_response_applications,
                incremental_applications=receiver_incremental_applications,
            )
        )
        sender_parts.extend(
            _fold_sender_rows(
                fold_id=fold.fold_id,
                training=training,
                application=application,
            )
        )

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
        fold_plan=fold_plan,
        folds=tuple(fold_artifacts),
        oof_coverage=coverage,
        oof_receiver_coverage=receiver_coverage,
        oof_sender_assignments=assignments,
        coverage_audit=audit,
    )


__all__ = [
    "CrossFitArtifacts",
    "CrossFitFoldArtifacts",
    "CrossFitSpec",
    "run_subject_crossfit",
]
