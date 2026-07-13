"""Public subject-blocked orchestration for implemented train/apply stages.

The current path verifies out-of-fold application of the frozen interaction
universe and contrast-common sender functional.  It deliberately does not
certify the still-missing downstream and common scoring stages.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field, replace

import pandas as pd
from anndata import AnnData

from crychic.core import CrychicConfig, SeedLineage, stable_id
from crychic.data import InputMode, validate_anndata
from crychic.design import (
    ContrastSpec,
    canonical_context,
    default_design_formula,
    plain_context_value,
)
from crychic.resampling import (
    DesignFoldChecker,
    OOFCoverageAudit,
    SubjectFoldPlan,
    plan_subject_folds,
    validate_oof_subject_coverage,
)
from crychic.resources import ResourceBundle, TargetPrior
from crychic.sender import COMMON_SENDER_APPLICATION_COLUMNS

from .application import TrainingArtifactApplication, apply_training_artifacts
from .training import (
    FoldTrainingSpec,
    TrainingArtifacts,
    _input_schema,
    _sanitize_validated_input,
    fit_training_artifacts,
)

_STAGE_NAME = "contrast_common_sender_application"
_STAGE_STATUS = "verified_train_only_oof_partial_pipeline"
_PRODUCER_MARKER = "crychic.workflow.crossfit.v1"
_COVERAGE_COLUMNS = (
    "subject_id",
    "sample_id",
    "fold_id",
    "contrast_id",
    "contrast_context",
    "functional_status",
    "sender_functional_id",
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
        if self.schema_version != "1.0.0":
            raise ValueError("CrossFitSpec schema_version must be 1.0.0")
        payload = {
            "allowed_n_splits": list(allowed),
            "contrasts": [contrast.to_dict() for contrast in contrasts],
            "min_test_subjects_per_context": self.min_test_subjects_per_context,
            "min_train_subjects_per_context": self.min_train_subjects_per_context,
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
            "min_train_subjects_per_context": (
                self.min_train_subjects_per_context
            ),
            "min_test_subjects_per_context": self.min_test_subjects_per_context,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossFitFoldArtifacts:
    """Training and held-out application artifacts for one planned fold."""

    fold_id: str
    training: TrainingArtifacts
    application: TrainingArtifactApplication

    def __post_init__(self) -> None:
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise ValueError("fold_id must be a non-empty identifier")
        if not isinstance(self.training, TrainingArtifacts):
            raise TypeError("training must be TrainingArtifacts")
        if not isinstance(self.application, TrainingArtifactApplication):
            raise TypeError("application must be TrainingArtifactApplication")
        if self.application.training_artifact_id != self.training.training_artifact_id:
            raise ValueError("fold application is not bound to its training artifacts")


@dataclass(frozen=True, slots=True, init=False)
class CrossFitArtifacts:
    """Verified OOF artifacts for implemented stages of the partial pipeline."""

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
        payload = {
            "coverage_audit_id": self.coverage_audit.audit_id,
            "fold_applications": [
                {
                    "fold_id": item.fold_id,
                    "heldout_input_digest": item.application.heldout_input_digest,
                    "training_artifact_id": item.training.training_artifact_id,
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
        """Whether implemented availability/sender stages passed OOF audit."""

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
                }
                for item in self.folds
            ],
            "n_oof_coverage_rows": len(self.oof_coverage),
            "n_oof_sender_assignment_rows": len(self.oof_sender_assignments),
            "remaining_stages": list(self.folds[0].training.remaining_stages),
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


def _fold_coverage_rows(
    sample_metadata: pd.DataFrame,
    *,
    config: CrychicConfig,
    fold_id: str,
    test_subject_ids: tuple[str, ...],
    training: TrainingArtifacts,
) -> list[dict[str, object]]:
    heldout = sample_metadata.loc[
        sample_metadata[config.subject_key].astype(str).isin(test_subject_ids)
    ]
    rows: list[dict[str, object]] = []
    for functional in training.sender_functionals:
        contrast_id = _contrast_id(functional.contrast)
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
    """Run verified subject-blocked OOF application of implemented stages.

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
            )
        )
        coverage_rows.extend(
            _fold_coverage_rows(
                validated.report.sample_metadata,
                config=config,
                fold_id=fold.fold_id,
                test_subject_ids=fold.test_subject_ids,
                training=training,
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
