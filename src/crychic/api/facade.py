"""Public facade for the deterministic v0.1 exploratory workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from anndata import AnnData

from crychic.attribution import AttributionSupportMethod
from crychic.availability import AvailabilityParameters
from crychic.core.config import CrychicConfig
from crychic.core.errors import FeatureUnavailableError
from crychic.data import InputSchema, ValidatedInput, validate_anndata
from crychic.design import ContextGraph
from crychic.resources import ResourceBundle, TargetPrior
from crychic.results import CrychicResult
from crychic.sender import SenderEvidenceParameters
from crychic.workflow import (
    BaselineArtifacts,
    BaselineDryRunPlan,
    dry_run_baseline,
    fit_baseline,
    write_baseline_result,
)

from .config import input_schema_from_config

_DEFAULT_AVAILABILITY_PARAMETERS = AvailabilityParameters()


class Crychic:
    """Configure and run the v0.1 exploratory CRYCHIC baseline."""

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
                remediation=(
                    "Load CellChatDB or CellPhoneDB and pass resource_bundle"
                ),
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
