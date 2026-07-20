"""Public facade for baseline and opt-in cross-fit/inference workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from anndata import AnnData

from crychic.attribution import AttributionSupportMethod
from crychic.availability import AvailabilityParameters
from crychic.core import CrychicConfig, FeatureUnavailableError, SeedLineage
from crychic.core._validation import validation_scope
from crychic.data import InputSchema, ValidatedInput, validate_anndata
from crychic.design import ContextGraph
from crychic.inference import (
    ActiveProbabilitySpec,
    FrozenHypothesisUniverse,
    FullPipelineEffectDistributionSpec,
    G3PCalibrationGate,
)
from crychic.inference.calibration_attestation import CalibrationReplayRegistry
from crychic.network import CommunicationHypergraph
from crychic.resampling import ActiveNullSpec
from crychic.resources import ResourceBundle, TargetPrior
from crychic.results import (
    ActiveProbabilityResult,
    BootstrapSupportDocument,
    CrychicResult,
    write_active_probability_result,
)
from crychic.results.g3p_calibration import G3PCalibrationResult
from crychic.sender import SenderEvidenceParameters
from crychic.workflow import (
    ActiveProbabilityPipelineResult,
    BaselineArtifacts,
    BaselineDryRunPlan,
    CrossFitArtifacts,
    CrossFitLayeredSignatureExport,
    CrossFitResult,
    CrossFitSpec,
    FullPipelineResamplingResult,
    RepeatedCrossFitDiagnostics,
    RepeatedCrossFitSpec,
    SignatureScoreMode,
    build_crossfit_communication_hypergraph,
    dry_run_baseline,
    export_crossfit_layered_signatures,
    fit_baseline,
    run_active_probability_pipeline,
    run_full_pipeline_resampling,
    run_repeated_subject_crossfit,
    run_subject_crossfit,
    summarize_frozen_bootstrap_support,
    validate_recommended_crossfit_spec,
    write_baseline_result,
    write_crossfit_result,
)

from .config import input_schema_from_config

_DEFAULT_AVAILABILITY_PARAMETERS = AvailabilityParameters()


class AnalysisProfile(StrEnum):
    """Versioned high-level workflow choices for :meth:`Crychic.analyze`."""

    CROSSFIT_DESCRIPTIVE_V1 = "crossfit_descriptive_v1"
    LEGACY_V01 = "legacy_v01"


class Crychic:
    """Configure baseline, cross-fit, and explicitly gated CRYCHIC workflows."""

    __slots__ = ("_config", "_resource_bundle", "_target_prior")

    def __init__(
        self,
        config: CrychicConfig,
        *,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
    ) -> None:
        if not isinstance(config, CrychicConfig):
            raise TypeError("config must be a CrychicConfig instance")
        if resource_bundle is not None and not isinstance(
            resource_bundle, ResourceBundle
        ):
            raise TypeError("resource_bundle must be a ResourceBundle or None")
        if target_prior is not None and not isinstance(target_prior, TargetPrior):
            raise TypeError("target_prior must be a TargetPrior or None")
        self._config = config
        self._resource_bundle = resource_bundle
        self._target_prior = target_prior

    @property
    def config(self) -> CrychicConfig:
        """Return the immutable analysis configuration."""

        return self._config

    @property
    def input_schema(self) -> InputSchema:
        """Return the data-owned input declaration for this configuration."""

        return input_schema_from_config(self._config)

    def validate(
        self,
        adata: AnnData,
        *,
        context_graph: ContextGraph | None = None,
    ) -> ValidatedInput:
        """Validate the AnnData contract and optional context graph."""

        validated = validate_anndata(adata, self.input_schema)
        if context_graph is not None:
            dry_run_baseline(
                adata,
                self._config,
                resource_bundle=self._resource_bundle,
                context_graph=context_graph,
                target_prior=self._target_prior,
            )
        return validated

    def dry_run(
        self,
        adata: AnnData,
        *,
        context_graph: ContextGraph | None = None,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
        min_cells: int = 10,
        min_samples_per_context: int = 2,
        min_subjects_per_context: int = 2,
    ) -> BaselineDryRunPlan:
        """Return support and resource diagnostics without fitting matrices."""

        bundle = resource_bundle or self._resource_bundle
        prior = target_prior or self._target_prior
        return dry_run_baseline(
            adata,
            self._config,
            resource_bundle=bundle,
            context_graph=context_graph,
            target_prior=prior,
            min_cells=min_cells,
            min_samples_per_context=min_samples_per_context,
            min_subjects_per_context=min_subjects_per_context,
        )

    def analyze(
        self,
        adata: AnnData,
        *,
        profile: AnalysisProfile | str = AnalysisProfile.CROSSFIT_DESCRIPTIVE_V1,
        spec: CrossFitSpec | None = None,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
        output_dir: str | Path | None = None,
    ) -> BaselineArtifacts | CrychicResult | CrossFitArtifacts | CrossFitResult:
        """Run a versioned high-level workflow without changing ``fit()``.

        The user-facing default is descriptive subject cross-fit and requires an
        explicit :class:`CrossFitSpec`. This is an API usability default, not a
        calibrated-inference or method-release default. An inestimable cross-fit
        result is returned as such and never falls back to the legacy baseline.

        Select ``legacy_v01`` explicitly to delegate to the unchanged exploratory
        baseline. ``spec`` is rejected for that profile instead of being ignored.
        """

        try:
            selected_profile = AnalysisProfile(profile)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in AnalysisProfile)
            raise ValueError(f"profile must be one of: {choices}") from exc

        if selected_profile is AnalysisProfile.CROSSFIT_DESCRIPTIVE_V1:
            if spec is None:
                raise ValueError(
                    "crossfit_descriptive_v1 requires an explicit CrossFitSpec"
                )
            if not isinstance(spec, CrossFitSpec):
                raise TypeError("spec must be a CrossFitSpec instance")
            return self.fit_descriptive(
                adata,
                spec=spec,
                resource_bundle=resource_bundle,
                target_prior=target_prior,
                output_dir=output_dir,
            )

        if spec is not None:
            raise ValueError("legacy_v01 does not accept a CrossFitSpec")
        return self.fit(
            adata,
            resource_bundle=resource_bundle,
            target_prior=target_prior,
            output_dir=output_dir,
        )

    def fit_descriptive(
        self,
        adata: AnnData,
        *,
        spec: CrossFitSpec,
        n_jobs: int = 1,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
        output_dir: str | Path | None = None,
    ) -> CrossFitArtifacts | CrossFitResult:
        """Run descriptive subject cross-fit with bounded outer-fold workers."""

        validate_recommended_crossfit_spec(spec)
        return self.fit_crossfit(
            adata,
            spec=spec,
            n_jobs=n_jobs,
            resource_bundle=resource_bundle,
            target_prior=target_prior,
            output_dir=output_dir,
        )

    def fit(
        self,
        adata: AnnData,
        *,
        context_graph: ContextGraph | None = None,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
        output_dir: str | Path | None = None,
        input_digest: str | None = None,
        git_commit: str | None = None,
        git_dirty: bool = False,
        package_version: str | None = None,
        run_id: str | None = None,
        persist_edge_evidence: bool = False,
        availability_parameters: AvailabilityParameters = (
            _DEFAULT_AVAILABILITY_PARAMETERS
        ),
        sender_parameters: SenderEvidenceParameters | None = None,
        min_cells: int = 10,
        min_samples_per_context: int = 2,
        min_subjects_per_context: int = 2,
        min_pooled_availability: float = 0.01,
        max_interactions: int | None = None,
        subject_fixed_effects: bool = False,
        lambda1: float = 0.0,
        lambda2: float = 0.0,
        cosine_threshold: float = 0.95,
        prior_quality: float = 1.0,
        component_weights: Mapping[str, float] | None = None,
        component_scales: Mapping[str, float] | None = None,
        downstream_attribution_support_method: AttributionSupportMethod | str = (
            AttributionSupportMethod.RELATIVE_COEFFICIENT_V1
        ),
    ) -> BaselineArtifacts | CrychicResult:
        """Fit the baseline and optionally atomically persist a queryable result."""

        bundle = resource_bundle or self._resource_bundle
        if bundle is None:
            raise FeatureUnavailableError(
                "A versioned ligand-receptor resource is required for fitting",
                code="resource_bundle_missing",
                field="resource_bundle",
                remediation=("Load CellChatDB or CellPhoneDB and pass resource_bundle"),
            )
        prior = target_prior or self._target_prior
        artifacts = fit_baseline(
            adata,
            self._config,
            bundle,
            context_graph=context_graph,
            target_prior=prior,
            availability_parameters=availability_parameters,
            sender_parameters=sender_parameters,
            min_cells=min_cells,
            min_samples_per_context=min_samples_per_context,
            min_subjects_per_context=min_subjects_per_context,
            min_pooled_availability=min_pooled_availability,
            max_interactions=max_interactions,
            subject_fixed_effects=subject_fixed_effects,
            lambda1=lambda1,
            lambda2=lambda2,
            cosine_threshold=cosine_threshold,
            prior_quality=prior_quality,
            component_weights=component_weights,
            component_scales=component_scales,
            downstream_attribution_support_method=(
                downstream_attribution_support_method
            ),
        )
        if output_dir is None:
            return artifacts
        return write_baseline_result(
            artifacts,
            output_dir,
            input_digest=input_digest,
            git_commit=git_commit,
            git_dirty=git_dirty,
            package_version=package_version,
            run_id=run_id,
            sender_parameters=sender_parameters,
            persist_edge_evidence=persist_edge_evidence,
        )

    @validation_scope()
    def fit_crossfit(
        self,
        adata: AnnData,
        *,
        spec: CrossFitSpec,
        n_jobs: int = 1,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
        output_dir: str | Path | None = None,
    ) -> CrossFitArtifacts | CrossFitResult:
        """Run subject-blocked family-first cross-fit and optionally persist it.

        This entry point emits held-out descriptive components and subject-family
        effects. It does not itself enable formal p/q values or communication
        probabilities; use the separate exact-parent-bound resampling or active-
        probability workflows, whose release gates remain in force.

        ``n_jobs`` bounds concurrent outer folds. Workers share the immutable
        sanitized input snapshot but retain independent fold-local model state.
        """

        if not isinstance(spec, CrossFitSpec):
            raise TypeError("spec must be a CrossFitSpec instance")
        bundle = resource_bundle or self._resource_bundle
        if bundle is None:
            raise FeatureUnavailableError(
                "A versioned ligand-receptor resource is required for cross-fitting",
                code="resource_bundle_missing",
                field="resource_bundle",
                remediation=("Load CellChatDB or CellPhoneDB and pass resource_bundle"),
            )
        prior = target_prior or self._target_prior
        if prior is None:
            raise FeatureUnavailableError(
                "A versioned ligand-target prior is required for family cross-fitting",
                code="target_prior_missing",
                field="target_prior",
                remediation="Load a NicheNet target prior and pass target_prior",
            )
        artifacts = run_subject_crossfit(
            adata,
            self._config,
            bundle,
            prior,
            spec=spec,
            n_jobs=n_jobs,
        )
        if output_dir is None:
            return artifacts
        return write_crossfit_result(artifacts, output_dir)

    def fit_active_probability(
        self,
        adata: AnnData,
        *,
        spec: CrossFitSpec,
        contrast_id_or_name: str,
        n_plans: int,
        n_jobs: int = 1,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
        output_dir: str | Path | None = None,
        universe_name: str | None = None,
        active_null_spec: ActiveNullSpec | None = None,
        active_probability_spec: ActiveProbabilitySpec | None = None,
        calibration_gate: G3PCalibrationGate | None = None,
        calibration_result: G3PCalibrationResult | None = None,
        calibration_replay_registry: CalibrationReplayRegistry | None = None,
        retain_children: bool = False,
    ) -> ActiveProbabilityPipelineResult | ActiveProbabilityResult:
        """Run the explicit ADR-013 active-null probability workflow.

        Communication probabilities are released only when the supplied
        producer-owned G3-P gate binds the exact runtime contracts and passed.
        Without such a gate, the result remains diagnostic-only.

        ``n_jobs`` bounds concurrent null reruns. Workers share the immutable
        input snapshot, but each retains fold-local model state; peak working
        memory can therefore scale with ``min(n_jobs, n_plans)``.
        """

        if not isinstance(spec, CrossFitSpec):
            raise TypeError("spec must be a CrossFitSpec instance")
        if calibration_result is not None:
            if not isinstance(calibration_result, G3PCalibrationResult):
                raise TypeError(
                    "calibration_result must be G3PCalibrationResult or None"
                )
            parent_gate = calibration_result.calibration_gate
            if (
                calibration_gate is not None
                and calibration_gate.gate_id != parent_gate.gate_id
            ):
                raise ValueError(
                    "calibration_gate differs from calibration_result.calibration_gate"
                )
            calibration_gate = parent_gate
        bundle = resource_bundle or self._resource_bundle
        if bundle is None:
            raise FeatureUnavailableError(
                "A ligand-receptor resource is required for active probability",
                code="resource_bundle_missing",
                field="resource_bundle",
                remediation="Pass a versioned CellChatDB or CellPhoneDB resource",
            )
        prior = target_prior or self._target_prior
        if prior is None:
            raise FeatureUnavailableError(
                "A ligand-target prior is required for active probability",
                code="target_prior_missing",
                field="target_prior",
                remediation="Pass a versioned NicheNet target prior",
            )
        artifacts = run_active_probability_pipeline(
            adata,
            self._config,
            bundle,
            prior,
            crossfit_spec=spec,
            contrast_id_or_name=contrast_id_or_name,
            n_plans=n_plans,
            n_jobs=n_jobs,
            universe_name=universe_name,
            active_null_spec=active_null_spec,
            active_probability_spec=active_probability_spec,
            calibration_gate=calibration_gate,
            retain_children=retain_children,
        )
        if output_dir is None:
            return artifacts
        return write_active_probability_result(
            output_dir,
            collection=artifacts.probability_collection,
            distribution=artifacts.null_rerun.distribution,
            universe=artifacts.universe,
            spec=artifacts.active_probability_spec,
            calibration_gate=artifacts.calibration_gate,
            calibration_result=(
                calibration_result
                if artifacts.calibration_gate is not None
                and artifacts.calibration_gate.comm_probability_release_allowed
                else None
            ),
            replay_registry=calibration_replay_registry,
            n_jobs=n_jobs,
        )

    def fit_repeated_crossfit(
        self,
        adata: AnnData,
        *,
        spec: RepeatedCrossFitSpec,
        n_jobs: int = 1,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
    ) -> RepeatedCrossFitDiagnostics:
        """Refit every declared subject cross-fit repeat from raw counts.

        ``n_jobs`` bounds independent complete repeats over one immutable input
        snapshot without changing the scientific result identity.
        """

        if not isinstance(spec, RepeatedCrossFitSpec):
            raise TypeError("spec must be a RepeatedCrossFitSpec instance")
        bundle = resource_bundle or self._resource_bundle
        if bundle is None:
            raise FeatureUnavailableError(
                "A ligand-receptor resource is required for repeated cross-fitting",
                code="resource_bundle_missing",
                field="resource_bundle",
                remediation="Pass a versioned CellChatDB or CellPhoneDB resource",
            )
        prior = target_prior or self._target_prior
        if prior is None:
            raise FeatureUnavailableError(
                "A ligand-target prior is required for repeated cross-fitting",
                code="target_prior_missing",
                field="target_prior",
                remediation="Pass a versioned NicheNet target prior",
            )
        return run_repeated_subject_crossfit(
            adata,
            self._config,
            bundle,
            prior,
            spec=spec,
            n_jobs=n_jobs,
        )

    def export_crossfit_signatures(
        self,
        artifacts: CrossFitArtifacts,
        *,
        mode: SignatureScoreMode = "state",
        entropy_threshold: float = 0.8,
    ) -> CrossFitLayeredSignatureExport:
        """Export descriptive fitted signatures without refitting models."""

        if not isinstance(artifacts, CrossFitArtifacts):
            raise TypeError("artifacts must be CrossFitArtifacts")
        return export_crossfit_layered_signatures(
            artifacts,
            mode=mode,
            entropy_threshold=entropy_threshold,
        )

    def export_crossfit_hypergraph(
        self,
        artifacts: CrossFitArtifacts,
        *,
        resource_bundle: ResourceBundle | None = None,
    ) -> CommunicationHypergraph:
        """Export a descriptive sender-resolved communication hypergraph."""

        if type(artifacts) is not CrossFitArtifacts:
            raise TypeError("artifacts must be producer-owned CrossFitArtifacts")
        bundle = resource_bundle or self._resource_bundle
        if bundle is None:
            raise FeatureUnavailableError(
                "A ligand-receptor resource is required for hypergraph export",
                code="resource_bundle_missing",
                field="resource_bundle",
                remediation=(
                    "Use the exact ResourceBundle fitted by cross-fit and pass it "
                    "to Crychic or this method"
                ),
            )
        return build_crossfit_communication_hypergraph(artifacts, bundle)

    def resample_crossfit(
        self,
        adata: AnnData,
        *,
        spec: CrossFitSpec,
        n_bootstraps: int = 0,
        n_permutations: int = 0,
        strata_keys: tuple[str, ...] | None = None,
        immutable_covariates: tuple[str, ...] | None = None,
        seed_lineage: SeedLineage | None = None,
        retain_children: bool = False,
        n_jobs: int = 1,
        resource_bundle: ResourceBundle | None = None,
        target_prior: TargetPrior | None = None,
    ) -> FullPipelineResamplingResult:
        """Materialize legal subject resamples and rerun the full cross-fit chain.

        ``n_jobs`` bounds concurrent reruns over the immutable source snapshot.
        Scientific result identities remain independent of execution parallelism.
        """

        if not isinstance(spec, CrossFitSpec):
            raise TypeError("spec must be a CrossFitSpec instance")
        bundle = resource_bundle or self._resource_bundle
        if bundle is None:
            raise FeatureUnavailableError(
                "A ligand-receptor resource is required for full-pipeline resampling",
                code="resource_bundle_missing",
                field="resource_bundle",
                remediation="Pass a versioned CellChatDB or CellPhoneDB resource",
            )
        prior = target_prior or self._target_prior
        if prior is None:
            raise FeatureUnavailableError(
                "A ligand-target prior is required for full-pipeline resampling",
                code="target_prior_missing",
                field="target_prior",
                remediation="Pass a versioned NicheNet target prior",
            )
        return run_full_pipeline_resampling(
            adata,
            self._config,
            bundle,
            prior,
            spec=spec,
            n_bootstraps=n_bootstraps,
            n_permutations=n_permutations,
            strata_keys=strata_keys,
            immutable_covariates=immutable_covariates,
            seed_lineage=seed_lineage,
            retain_children=retain_children,
            n_jobs=n_jobs,
        )

    def summarize_bootstrap_support(
        self,
        point_artifacts: CrossFitArtifacts,
        resampling: FullPipelineResamplingResult,
        *,
        universe: FrozenHypothesisUniverse,
        distribution_specs: (
            Sequence[FullPipelineEffectDistributionSpec]
            | Mapping[str, FullPipelineEffectDistributionSpec]
        ),
    ) -> BootstrapSupportDocument:
        """Summarize complete frozen condition-specific bootstrap support."""

        if not isinstance(point_artifacts, CrossFitArtifacts):
            raise TypeError("point_artifacts must be CrossFitArtifacts")
        if not isinstance(resampling, FullPipelineResamplingResult):
            raise TypeError("resampling must be FullPipelineResamplingResult")
        if not isinstance(universe, FrozenHypothesisUniverse):
            raise TypeError("universe must be FrozenHypothesisUniverse")
        return summarize_frozen_bootstrap_support(
            point_artifacts,
            resampling,
            universe,
            distribution_specs,
        )
