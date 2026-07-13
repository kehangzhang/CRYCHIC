"""Public subject-blocked orchestration for implemented train/apply stages.

The current coverage audit verifies the frozen interaction universe and
contrast-common sender rows.  Fold artifacts additionally preserve train-only
design encoders and partial receiver-family applications, but those stages do
not yet emit a separate exact-coverage audit and do not certify the still-
missing incremental downstream or common scoring stages.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from crychic.attribution import fit_receiver_family_training_artifacts
from crychic.core import CrychicConfig, SeedLineage, stable_id
from crychic.data import InputMode, validate_anndata
from crychic.design import (
    ContrastSpec,
    FrozenDesignApplication,
    FrozenDesignEncoder,
    apply_frozen_design_encoder,
    canonical_context,
    default_design_formula,
    fit_frozen_design_encoder,
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
_STAGE_STATUS = "verified_train_only_oof_partial_pipeline"
_PRODUCER_MARKER = "crychic.workflow.crossfit.v1"
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
        for model, receiver_application in zip(
            receiver_models, receiver_applications, strict=True
        ):
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


@dataclass(frozen=True, slots=True, init=False)
class CrossFitArtifacts:
    """Partial artifacts with exact sender-stage OOF coverage verification."""

    spec: CrossFitSpec
    fold_plan: SubjectFoldPlan
    folds: tuple[CrossFitFoldArtifacts, ...]
    oof_coverage: pd.DataFrame
    oof_sender_assignments: pd.DataFrame
    coverage_audit: OOFCoverageAudit
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
        oof_sender_assignments: pd.DataFrame,
        coverage_audit: OOFCoverageAudit,
    ) -> CrossFitArtifacts:
        self = object.__new__(cls)
        values: dict[str, object] = {
            "spec": spec,
            "fold_plan": fold_plan,
            "folds": folds,
            "oof_coverage": oof_coverage,
            "oof_sender_assignments": oof_sender_assignments,
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
        coverage = self.oof_coverage.copy(deep=True)
        if tuple(coverage.columns) != _COVERAGE_COLUMNS:
            raise ValueError("oof_coverage columns do not match the stage contract")
        if set(coverage["functional_status"].astype(str)) != {"out_of_fold"}:
            raise ValueError("stage coverage must contain only out-of-fold rows")
        assignments = self.oof_sender_assignments.copy(deep=True)
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
        payload = {
            "coverage_audit_id": self.coverage_audit.audit_id,
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
                    "receiver_family_training_artifact_ids": [
                        model.training_artifact_id
                        for model in item.receiver_family_models
                    ],
                    "receiver_family_application_statuses": [
                        application.application_status
                        for application in item.receiver_family_applications
                    ],
                }
                for item in sorted(folds, key=lambda value: value.fold_id)
            ],
            "fold_plan_id": self.fold_plan.plan_id,
            "spec_id": self.spec.spec_id,
            "status": self.certification_status,
        }
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "oof_coverage", coverage)
        object.__setattr__(self, "oof_sender_assignments", assignments)
        object.__setattr__(self, "crossfit_id", stable_id("subject_crossfit", payload))

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

        return {
            "crossfit_id": self.crossfit_id,
            "certification_status": self.certification_status,
            "completed_stage_oof_verified": self.completed_stage_oof_verified,
            "complete_pipeline_oof_certified": self.is_oof_certified,
            "oof_audit_scope": _STAGE_NAME,
            "spec": self.spec.to_dict(),
            "fold_plan": self.fold_plan.to_dict(),
            "coverage_audit": self.coverage_audit.to_dict(),
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
                }
                for item in self.folds
            ],
            "n_oof_coverage_rows": len(self.oof_coverage),
            "n_oof_sender_assignment_rows": len(self.oof_sender_assignments),
            "remaining_stages": sorted(
                {
                    stage
                    for item in self.folds
                    for model in item.receiver_family_models
                    for stage in model.remaining_stages
                }
            ),
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
        sender_parts.extend(
            _fold_sender_rows(
                fold_id=fold.fold_id,
                training=training,
                application=application,
            )
        )

    coverage = pd.DataFrame(coverage_rows, columns=_COVERAGE_COLUMNS)
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
        oof_sender_assignments=assignments,
        coverage_audit=audit,
    )


__all__ = [
    "CrossFitArtifacts",
    "CrossFitFoldArtifacts",
    "CrossFitSpec",
    "run_subject_crossfit",
]
