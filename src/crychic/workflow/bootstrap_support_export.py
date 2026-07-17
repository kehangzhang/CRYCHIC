"""Export complete frozen bootstrap-support universes to result tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from crychic.core import ContractError, stable_id
from crychic.inference.full_pipeline import FullPipelineEffectDistributionSpec
from crychic.inference.hypotheses import (
    FrozenHypothesisUniverse,
    HypothesisPrefilterStatus,
)
from crychic.results import (
    BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
    RESULT_SCHEMA_VERSION,
    BootstrapSupportDocument,
    bootstrap_support_contract,
)

from .crossfit import CrossFitArtifacts
from .frozen_selection_frequency_universe import (
    FrozenFamilySelectionFrequency,
    FrozenSelectionFrequencyUniverseCollection,
    FrozenSelectionFrequencyUniverseRecord,
    summarize_frozen_selection_frequency_universe,
)
from .frozen_specificity_support_universe import (
    FrozenSpecificitySupportUniverseCollection,
    FrozenSpecificitySupportUniverseRecord,
    summarize_frozen_specificity_support_universe,
)
from .full_pipeline_resampling import FullPipelineResamplingResult

_SCHEMA_VERSION = "1.0.0"
_REGISTRY_KIND = "frozen_bootstrap_support_lineage_v1"
_HYPOTHESIS_LEVEL = "driver_family_receiver"
_VIEW = "gene"
_MINIMUM_BOOTSTRAPS = 1_000
_SPECIFICITY_SEMANTICS = (
    "frequentist_full_pipeline_subject_bootstrap_exceedance_frequency_not_posterior_v1"
)
_SELECTION_OPPORTUNITY_GRAIN = "equal_bootstrap_mean_of_outer_fold_selection_v1"
_SELECTION_FREQUENCY_SEMANTICS = (
    "complete_subject_bootstrap_equal_plan_mean_of_outer_fold_selection_v1"
)
_SPECIFICITY_AUTHENTICATION = (
    "exact_frozen_target_point_and_subject_bootstrap_workflow_binding_v1"
)
_SELECTION_AUTHENTICATION = "exact_frozen_plan_fold_event_binding_v1"
_PREFILTER_AUTHENTICATION = (
    "exact_frozen_declaration_prefilter_and_resampling_binding_v1"
)
_ADAPTER_FAILURE_AUTHENTICATION = (
    "exact_frozen_declaration_and_resampling_adapter_failure_binding_v1"
)
_PREFILTER_SOURCE_STATUS = "frozen_prefiltered_not_estimable_v1"
_ADAPTER_FAILURE_SOURCE_STATUS = "frozen_workflow_adapter_failed_v1"


def _error(message: str, *, code: str, field: str) -> ContractError:
    return ContractError(
        message,
        code=code,
        field=field,
        remediation=(
            "Export both aggregates from the exact same intact frozen universe "
            "and full-pipeline resampling result"
        ),
    )


def _require_common_lineage(
    specificity: FrozenSpecificitySupportUniverseCollection,
    selection: FrozenSelectionFrequencyUniverseCollection,
) -> None:
    specificity._require_intact()
    selection._require_intact()
    if (
        specificity._universe is not selection._universe
        or specificity.universe_id != selection.universe_id
    ):
        raise _error(
            "Bootstrap-support aggregates use different frozen universes",
            code="bootstrap_support_export_universe_mismatch",
            field="hypothesis_universe_id",
        )
    if (
        specificity.bootstrap_plan_ids != selection.bootstrap_plan_ids
        or specificity.bootstrap_workflow_record_ids != selection.workflow_record_ids
    ):
        raise _error(
            "Bootstrap-support aggregates do not share one exact bootstrap plan set",
            code="bootstrap_support_export_plan_mismatch",
            field="bootstrap_plan_ids,workflow_record_ids",
        )
    if (
        specificity._resampling is not selection._resampling
        or specificity._resampling.result_id != selection.resampling_result_id
    ):
        raise _error(
            "Bootstrap-support aggregates use different resampling sources",
            code="bootstrap_support_export_source_mismatch",
            field="resampling_result_id",
        )
    if (
        specificity.crossfit_spec_id != selection.crossfit_spec_id
        or specificity.crossfit_spec_id != specificity._resampling.crossfit_spec_id
    ):
        raise _error(
            "Bootstrap-support aggregates use different cross-fit specifications",
            code="bootstrap_support_export_source_mismatch",
            field="crossfit_spec_id",
        )
    universe = specificity._universe
    if specificity.universe_declaration_ids != tuple(
        item.declaration_id for item in universe.declarations
    ) or selection.universe_declaration_ids != tuple(
        item.declaration_id for item in universe.declarations
    ):
        raise _error(
            "Bootstrap-support declaration coverage differs from its universe",
            code="bootstrap_support_export_universe_mismatch",
            field="declaration_id",
        )
    plans = specificity.bootstrap_plan_ids
    workflows = specificity.bootstrap_workflow_record_ids
    for specificity_record in specificity.records:
        if (
            specificity_record._universe is not universe
            or specificity_record._resampling is not specificity._resampling
            or specificity_record.bootstrap_plan_ids != plans
            or specificity_record.bootstrap_workflow_record_ids != workflows
        ):
            raise _error(
                "Specificity row differs from the export source lineage",
                code="bootstrap_support_export_source_mismatch",
                field="specificity_sources",
            )
        specificity_child = specificity_record._child_result
        if specificity_child is not None and (
            specificity_child._resampling is not specificity._resampling
            or specificity_child.bootstrap_plan_ids != plans
            or specificity_child.workflow_record_ids != workflows
        ):
            raise _error(
                "Specificity child differs from the export source lineage",
                code="bootstrap_support_export_source_mismatch",
                field="specificity_sources",
            )
    for selection_record in selection.records:
        if (
            selection_record._universe is not universe
            or selection_record._resampling is not selection._resampling
        ):
            raise _error(
                "Selection row differs from the export source lineage",
                code="bootstrap_support_export_source_mismatch",
                field="selection_sources",
            )
        selection_child = selection_record._child
        if selection_child is not None and (
            selection_child._resampling is not selection._resampling
            or selection_child.bootstrap_plan_ids != plans
            or selection_child.workflow_record_ids != workflows
        ):
            raise _error(
                "Selection child differs from the export source lineage",
                code="bootstrap_support_export_source_mismatch",
                field="selection_sources",
            )


def _specificity_source_status(
    record: FrozenSpecificitySupportUniverseRecord,
) -> str:
    child = record._child_result
    if child is not None:
        return child.source_binding_status
    if record._declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED:
        return _PREFILTER_SOURCE_STATUS
    return _ADAPTER_FAILURE_SOURCE_STATUS


def _specificity_authentication(
    record: FrozenSpecificitySupportUniverseRecord,
) -> str:
    if record._child_result is not None:
        return _SPECIFICITY_AUTHENTICATION
    if record._declaration.prefilter_status is HypothesisPrefilterStatus.FILTERED:
        return _PREFILTER_AUTHENTICATION
    return _ADAPTER_FAILURE_AUTHENTICATION


def _specificity_row(
    record: FrozenSpecificitySupportUniverseRecord,
) -> dict[str, object]:
    declaration = record._declaration
    child = record._child_result
    distribution_spec = record._distribution_spec
    return {
        "hypothesis_level": _HYPOTHESIS_LEVEL,
        "hypothesis_role": "secondary",
        "hypothesis_id": record.hypothesis_id,
        "contrast": declaration.contrast_name,
        "mode": declaration.mode.value,
        "view": _VIEW,
        "receiver": declaration.receiver,
        "family_id": declaration.family_id,
        "specificity_support": record.specificity_support,
        "status": record.status.value,
        "reason_code": record.reason_code,
        "minimum_effect": (
            None
            if distribution_spec is None
            else float(distribution_spec.minimum_effect)
        ),
        "specificity_direction": (
            None
            if distribution_spec is None
            else distribution_spec.specificity_direction.value
        ),
        "specificity_semantics": (
            _SPECIFICITY_SEMANTICS if child is None else child.specificity_semantics
        ),
        "minimum_bootstraps": (
            _MINIMUM_BOOTSTRAPS if child is None else child.minimum_bootstraps
        ),
        "n_bootstrap_total": record.n_bootstrap_total,
        "n_bootstrap_observed": record.n_bootstrap_observed,
        "n_bootstrap_not_estimable": record.n_bootstrap_not_estimable,
        "n_bootstrap_failed": record.n_bootstrap_failed,
        "result_id": record.record_id,
        "target_id": record.target_id,
        "score_target_id": record.score_target_id,
        "declaration_id": record.declaration_id,
        "hypothesis_universe_id": record.universe_id,
        "effect_scale_id": None if child is None else child.effect_scale_id,
        "point_crossfit_id": record.point_crossfit_id,
        "crossfit_spec_id": record.crossfit_spec_id,
        "point_effect_result_id": (
            None if child is None else child.point_effect_result_id
        ),
        "bootstrap_source_binding_id": (
            None if child is None else child.bootstrap_source_binding_id
        ),
        "bootstrap_plan_set_id": (
            None if child is None else child.bootstrap_plan_set_id
        ),
        "effect_spec_id": (
            None if distribution_spec is None else distribution_spec.effect_spec.spec_id
        ),
        "distribution_spec_id": record.distribution_spec_id,
        "source_distribution_id": (
            None if child is None else child.source_distribution_id
        ),
        "specificity_support_spec_id": (
            None if child is None else child.specificity_support_spec_id
        ),
        "numeric_result_id": None if child is None else child.numeric_result_id,
        "source_authenticated": True,
        "source_binding_status": _specificity_source_status(record),
        "authentication_semantics": _specificity_authentication(record),
        "specificity_support_release_allowed": (
            record.specificity_support_release_allowed
        ),
        "formal_pq_inference_allowed": False,
        "is_posterior_probability": False,
        "is_comm_probability": False,
    }


def _selection_source_binding_id(child: FrozenFamilySelectionFrequency) -> str:
    numeric = child._numeric_result
    return stable_id(
        "frozen_selection_frequency_export_binding",
        {
            "resampling_result_id": child.resampling_result_id,
            "target_id": child.target_id,
            "source_collection_id": child.source_collection_id,
            "source_event_ids": list(child.source_event_ids),
            "selection_frequency_spec_id": child.numeric_spec_id,
            "numeric_result_id": child.numeric_result_id,
            "numeric_record_ids": list(numeric.source_record_ids),
        },
        schema_version=_SCHEMA_VERSION,
    )


def _selection_source_record_set_id(
    child: FrozenFamilySelectionFrequency,
) -> str:
    return stable_id(
        "bootstrap_selection_source_record_set",
        {"record_ids": list(child._numeric_result.source_record_ids)},
        schema_version=_SCHEMA_VERSION,
    )


def _selection_source_status(
    record: FrozenSelectionFrequencyUniverseRecord,
) -> str:
    if record._child is not None:
        return record._child.source_binding_status
    if record.prefilter_status is HypothesisPrefilterStatus.FILTERED:
        return _PREFILTER_SOURCE_STATUS
    return _ADAPTER_FAILURE_SOURCE_STATUS


def _selection_authentication(
    record: FrozenSelectionFrequencyUniverseRecord,
) -> str:
    if record._child is not None:
        return _SELECTION_AUTHENTICATION
    if record.prefilter_status is HypothesisPrefilterStatus.FILTERED:
        return _PREFILTER_AUTHENTICATION
    return _ADAPTER_FAILURE_AUTHENTICATION


def _selection_row(
    record: FrozenSelectionFrequencyUniverseRecord,
) -> dict[str, object]:
    declaration = record._declaration
    child = record._child
    numeric = None if child is None else child._numeric_result
    source_event = None if child is None else child._source.events[0]
    return {
        "hypothesis_level": _HYPOTHESIS_LEVEL,
        "hypothesis_role": "primary",
        "hypothesis_id": record.hypothesis_id,
        "contrast": declaration.contrast_name,
        "mode": declaration.mode.value,
        "view": _VIEW,
        "receiver": declaration.receiver,
        "family_id": declaration.family_id,
        "selection_frequency": record.selection_frequency,
        "status": record.status.value,
        "reason_code": record.reason_code,
        "opportunity_grain": (
            _SELECTION_OPPORTUNITY_GRAIN
            if numeric is None
            else numeric.opportunity_grain
        ),
        "selection_frequency_semantics": (
            _SELECTION_FREQUENCY_SEMANTICS
            if numeric is None
            else numeric.selection_frequency_semantics
        ),
        "minimum_bootstraps": (
            _MINIMUM_BOOTSTRAPS if numeric is None else numeric.minimum_bootstraps
        ),
        "n_bootstrap_plans_total": record.n_bootstrap_plans_total,
        "n_bootstrap_plans_observed": (
            0 if numeric is None else numeric.n_bootstrap_plans_observed
        ),
        "n_bootstrap_plans_not_estimable": (
            0 if numeric is None else numeric.n_bootstrap_plans_not_estimable
        ),
        "n_bootstrap_plans_failed": (
            0 if numeric is None else numeric.n_bootstrap_plans_failed
        ),
        "n_event_rows_total": record.n_event_rows_total,
        "n_observed_event_rows": (
            0 if numeric is None else numeric.n_observed_event_rows
        ),
        "n_not_estimable_event_rows": (
            0 if numeric is None else numeric.n_not_estimable_event_rows
        ),
        "n_failed_event_rows": (0 if numeric is None else numeric.n_failed_event_rows),
        "n_selected_event_rows": record.n_selected_event_rows,
        "result_id": record.record_id,
        "target_id": record.target_id,
        "declaration_id": record.declaration_id,
        "hypothesis_universe_id": record.universe_id,
        "crossfit_spec_id": record.crossfit_spec_id,
        "resampling_result_id": record.resampling_result_id,
        "source_resampled_attribution_collection_id": (
            None if child is None else child.source_collection_id
        ),
        "source_binding_id": (
            None if child is None else _selection_source_binding_id(child)
        ),
        "selection_frequency_spec_id": (
            None if child is None else child.numeric_spec_id
        ),
        "numeric_result_id": None if child is None else child.numeric_result_id,
        "bootstrap_plan_set_id": (
            None if numeric is None else numeric.bootstrap_plan_set_id
        ),
        "opportunity_manifest_id": (
            None if child is None else child.opportunity_manifest_id
        ),
        "selection_rule_id": None if child is None else child.selection_rule_id,
        "selection_rule_semantics": (
            None if source_event is None else source_event.selection_rule_semantics
        ),
        "selection_threshold": (
            None if source_event is None else float(source_event.selection_threshold)
        ),
        "score_version": None if child is None else child.score_version,
        "source_record_set_id": (
            None if child is None else _selection_source_record_set_id(child)
        ),
        "source_authenticated": True,
        "source_binding_status": _selection_source_status(record),
        "authentication_semantics": _selection_authentication(record),
        "exact_plan_coverage": True,
        "exact_fold_coverage": True,
        "complete_universe_coverage": True,
        "selection_frequency_release_allowed": (
            record.selection_frequency_release_allowed
        ),
        "formal_pq_inference_allowed": False,
        "is_posterior_probability": False,
        "is_comm_probability": False,
    }


def _specificity_source(
    record: FrozenSpecificitySupportUniverseRecord,
) -> dict[str, object]:
    child = record._child_result
    has_included_sources = (
        record._declaration.prefilter_status is HypothesisPrefilterStatus.INCLUDED
    )
    return {
        "result_id": record.record_id,
        "hypothesis_id": record.hypothesis_id,
        "target_id": record.target_id,
        "score_target_id": record.score_target_id,
        "bootstrap_source_binding_id": (
            None if child is None else child.bootstrap_source_binding_id
        ),
        "bootstrap_plan_ids": (
            list(record.bootstrap_plan_ids) if has_included_sources else []
        ),
        "workflow_record_ids": (
            list(record.bootstrap_workflow_record_ids) if has_included_sources else []
        ),
        "effect_record_ids": [] if child is None else list(child.effect_record_ids),
    }


def _selection_source(
    record: FrozenSelectionFrequencyUniverseRecord,
) -> dict[str, object]:
    child = record._child
    return {
        "result_id": record.record_id,
        "hypothesis_id": record.hypothesis_id,
        "target_id": record.target_id,
        "source_resampled_attribution_collection_id": (
            None if child is None else child.source_collection_id
        ),
        "source_binding_id": (
            None if child is None else _selection_source_binding_id(child)
        ),
        "bootstrap_plan_ids": [] if child is None else list(child.bootstrap_plan_ids),
        "event_ids": [] if child is None else list(child.source_event_ids),
        "numeric_record_ids": (
            [] if child is None else list(child._numeric_result.source_record_ids)
        ),
    }


def _registry(
    specificity: FrozenSpecificitySupportUniverseCollection,
    selection: FrozenSelectionFrequencyUniverseCollection,
) -> dict[str, object]:
    universe = specificity._universe
    resampling = specificity._resampling
    declarations = list(universe.declarations)
    payload: dict[str, object] = {
        "extension_schema_version": BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "registry_kind": _REGISTRY_KIND,
        "hypothesis_universe": {
            "universe_id": universe.universe_id,
            "universe_name": universe.universe_name,
            "declaration_ids": [item.declaration_id for item in declarations],
            "hypothesis_ids": [item.hypothesis_id for item in declarations],
            "selection_hypothesis_ids": list(selection.applicable_hypothesis_ids),
            "specificity_hypothesis_ids": list(
                specificity.applicable_secondary_hypothesis_ids
            ),
            "declarations": [item.to_dict() for item in declarations],
        },
        "resampling_lineage": {
            "resampling_result_id": resampling.result_id,
            "exchangeability_id": resampling.exchangeability.exchangeability_id,
            "source_input_identity_id": resampling.source_input_identity_id,
            "source_input_digest": resampling.source_input_digest,
            "source_snapshot_id": resampling.source_snapshot_id,
            "config_digest": resampling.config_digest,
            "crossfit_spec_id": resampling.crossfit_spec_id,
            "resource_bundle_content_id": resampling.resource_bundle_content_id,
            "target_prior_content_id": resampling.target_prior_content_id,
            "root_seed_lineage": resampling.root_seed_lineage.to_dict(),
        },
        "bootstrap_plan_ids": list(specificity.bootstrap_plan_ids),
        "specificity_sources": [
            _specificity_source(record)
            for record in sorted(
                specificity.records,
                key=lambda item: item.hypothesis_id,
            )
        ],
        "selection_sources": [
            _selection_source(record)
            for record in sorted(
                selection.records,
                key=lambda item: item.hypothesis_id,
            )
        ],
    }
    payload["registry_id"] = stable_id(
        "bootstrap_support_registry",
        payload,
        schema_version=BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
    )
    return payload


def export_bootstrap_support_document(
    specificity: FrozenSpecificitySupportUniverseCollection,
    selection: FrozenSelectionFrequencyUniverseCollection,
) -> BootstrapSupportDocument:
    """Build released bootstrap-support tables from two exact frozen batches."""

    if not isinstance(specificity, FrozenSpecificitySupportUniverseCollection):
        raise TypeError(
            "specificity must be a FrozenSpecificitySupportUniverseCollection"
        )
    if not isinstance(selection, FrozenSelectionFrequencyUniverseCollection):
        raise TypeError(
            "selection must be a FrozenSelectionFrequencyUniverseCollection"
        )
    _require_common_lineage(specificity, selection)
    contract = bootstrap_support_contract()
    specificity_rows = [
        _specificity_row(record)
        for record in sorted(
            specificity.records,
            key=lambda item: item.hypothesis_id,
        )
    ]
    selection_rows = [
        _selection_row(record)
        for record in sorted(
            selection.records,
            key=lambda item: item.hypothesis_id,
        )
    ]
    return BootstrapSupportDocument(
        specificity_support=pd.DataFrame(
            specificity_rows,
            columns=contract.specificity.columns,
        ),
        selection_frequency=pd.DataFrame(
            selection_rows,
            columns=contract.selection.columns,
        ),
        registry=_registry(specificity, selection),
    )


def summarize_frozen_bootstrap_support(
    point_artifacts: CrossFitArtifacts,
    resampling: FullPipelineResamplingResult,
    universe: FrozenHypothesisUniverse,
    distribution_specs: (
        Sequence[FullPipelineEffectDistributionSpec]
        | Mapping[str, FullPipelineEffectDistributionSpec]
    ),
) -> BootstrapSupportDocument:
    """Run both complete frozen bootstrap summaries and export one document."""

    if not isinstance(point_artifacts, CrossFitArtifacts):
        raise TypeError("point_artifacts must be CrossFitArtifacts")
    if not isinstance(resampling, FullPipelineResamplingResult):
        raise TypeError("resampling must be FullPipelineResamplingResult")
    if not isinstance(universe, FrozenHypothesisUniverse):
        raise TypeError("universe must be FrozenHypothesisUniverse")
    specificity = summarize_frozen_specificity_support_universe(
        point_artifacts,
        resampling,
        universe,
        distribution_specs,
    )
    selection = summarize_frozen_selection_frequency_universe(
        universe,
        resampling,
    )
    return export_bootstrap_support_document(specificity, selection)


__all__ = [
    "export_bootstrap_support_document",
    "summarize_frozen_bootstrap_support",
]
