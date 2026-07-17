from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from tests.unit.workflow import (
    test_frozen_selection_frequency_universe as selection_fixtures,
)
from tests.unit.workflow import (
    test_frozen_specificity_support_universe as specificity_fixtures,
)

import crychic.workflow as workflow
import crychic.workflow.bootstrap_support_export as export_module
import crychic.workflow.frozen_selection_frequency_universe as selection_module
import crychic.workflow.frozen_specificity_support_universe as specificity_module
from crychic.core import ContractError
from crychic.results import BootstrapSupportDocument
from crychic.workflow.bootstrap_support_export import (
    export_bootstrap_support_document,
    summarize_frozen_bootstrap_support,
)
from crychic.workflow.frozen_selection_frequency_universe import (
    FrozenSelectionFrequencyUniverseCollection,
    summarize_frozen_selection_frequency_universe,
)
from crychic.workflow.frozen_specificity_support_universe import (
    FrozenSpecificitySupportUniverseCollection,
    summarize_frozen_specificity_support_universe,
)
from crychic.workflow.full_pipeline_resampling import FullPipelineResamplingResult
from crychic.workflow.resampled_attribution import (
    FrozenResampledAttributionCollection,
    ResampledFamilySelectionEvent,
)


def _specificity(
    monkeypatch: pytest.MonkeyPatch,
    chain: specificity_fixtures._BatchChain,
) -> FrozenSpecificitySupportUniverseCollection:
    specificity_fixtures._install_sources(monkeypatch, chain)
    return summarize_frozen_specificity_support_universe(
        chain.point,
        chain.resampling,
        chain.universe,
        (chain.spec,),
    )


def _selection(
    monkeypatch: pytest.MonkeyPatch,
    chain: specificity_fixtures._BatchChain,
    *,
    n_bootstraps: int,
) -> FrozenSelectionFrequencyUniverseCollection:
    monkeypatch.setattr(
        FullPipelineResamplingResult,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        FrozenResampledAttributionCollection,
        "_require_intact",
        lambda self: None,
    )
    monkeypatch.setattr(
        ResampledFamilySelectionEvent,
        "_require_intact",
        lambda self, **kwargs: None,
    )

    def summarize(
        resampling: FullPipelineResamplingResult,
        target: selection_fixtures.FrozenFamilyEffectTarget,
    ) -> FrozenResampledAttributionCollection:
        source = selection_fixtures._source_collection(
            resampling,
            target,
            n_bootstraps,
        )
        for event, plan, workflow_record in zip(
            source.events,
            resampling.plans,
            resampling.records,
            strict=True,
        ):
            object.__setattr__(event, "plan_id", plan.bootstrap_id)
            object.__setattr__(event, "resample_index", plan.resample_index)
            object.__setattr__(
                event,
                "crossfit_spec_id",
                resampling.crossfit_spec_id,
            )
            object.__setattr__(
                event,
                "full_pipeline_record_id",
                workflow_record.record_id,
            )
            object.__setattr__(
                event,
                "selection_rule_semantics",
                "outer_training_coefficient_positive_v1",
            )
            object.__setattr__(event, "selection_threshold", 0.0)
        object.__setattr__(
            source,
            "bootstrap_plan_ids",
            tuple(plan.bootstrap_id for plan in resampling.plans),
        )
        object.__setattr__(
            source,
            "full_pipeline_record_ids",
            tuple(record.record_id for record in resampling.records),
        )
        object.__setattr__(source, "crossfit_spec_id", resampling.crossfit_spec_id)
        return source

    monkeypatch.setattr(
        selection_module,
        "summarize_resampled_family_selection",
        summarize,
    )
    result = summarize_frozen_selection_frequency_universe(
        chain.universe,
        chain.resampling,
    )
    return result


def _pair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_bootstraps: int,
) -> tuple[
    specificity_fixtures._BatchChain,
    FrozenSpecificitySupportUniverseCollection,
    FrozenSelectionFrequencyUniverseCollection,
]:
    chain = specificity_fixtures._chain(n_bootstraps=n_bootstraps)
    specificity = _specificity(monkeypatch, chain)
    selection = _selection(
        monkeypatch,
        chain,
        n_bootstraps=n_bootstraps,
    )
    return chain, specificity, selection


def test_export_preserves_complete_applicable_universe_and_exact_grains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, specificity, selection = _pair(
        monkeypatch,
        n_bootstraps=1_000,
    )

    document = export_bootstrap_support_document(specificity, selection)

    assert isinstance(document, BootstrapSupportDocument)
    assert workflow.export_bootstrap_support_document is (
        export_bootstrap_support_document
    )
    assert len(document.specificity_support) == 2
    assert len(document.selection_frequency) == 1
    specificity_by_id = document.specificity_support.set_index("hypothesis_id")
    observed = specificity_by_id.loc[chain.included.hypothesis_id]
    filtered = specificity_by_id.loc[chain.filtered.hypothesis_id]
    assert observed["status"] == "observed"
    assert observed["specificity_support_release_allowed"]
    assert filtered["status"] == "not_estimable"
    assert filtered["reason_code"] == chain.filtered.filter_reason_code
    assert pd.isna(filtered["target_id"])
    assert pd.isna(filtered["numeric_result_id"])
    assert not filtered["specificity_support_release_allowed"]
    selection_row = document.selection_frequency.iloc[0]
    assert selection_row["status"] == "observed"
    assert selection_row["selection_frequency"] == 0.5
    assert selection_row["n_bootstrap_plans_observed"] == 1_000
    assert selection_row["n_observed_event_rows"] == 1_000
    assert "sender" not in document.selection_frequency.columns
    assert "interaction_id" not in document.selection_frequency.columns

    registry_universe = document.registry["hypothesis_universe"]
    assert isinstance(registry_universe, dict)
    assert len(registry_universe["declarations"]) == len(chain.universe.declarations)
    assert registry_universe["selection_hypothesis_ids"] == [
        chain.primary.hypothesis_id
    ]
    assert registry_universe["specificity_hypothesis_ids"] == list(
        specificity.applicable_secondary_hypothesis_ids
    )
    specificity_sources = document.registry["specificity_sources"]
    assert isinstance(specificity_sources, list)
    filtered_source = next(
        item
        for item in specificity_sources
        if item["hypothesis_id"] == chain.filtered.hypothesis_id
    )
    assert filtered_source["bootstrap_plan_ids"] == []
    assert filtered_source["effect_record_ids"] == []


def test_one_call_summary_composes_exact_complete_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, specificity, selection = _pair(monkeypatch, n_bootstraps=2)
    monkeypatch.setattr(
        export_module,
        "summarize_frozen_specificity_support_universe",
        lambda *args, **kwargs: specificity,
    )
    monkeypatch.setattr(
        export_module,
        "summarize_frozen_selection_frequency_universe",
        lambda *args, **kwargs: selection,
    )

    document = summarize_frozen_bootstrap_support(
        chain.point,
        chain.resampling,
        chain.universe,
        (chain.spec,),
    )

    assert workflow.summarize_frozen_bootstrap_support is (
        summarize_frozen_bootstrap_support
    )
    assert document.registry["registry_id"] == export_bootstrap_support_document(
        specificity,
        selection,
    ).registry["registry_id"]


def test_export_retains_typed_failed_rows_without_inventing_child_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = specificity_fixtures._chain(n_bootstraps=2)
    specificity_fixtures._install_sources(monkeypatch, chain)

    def fail_specificity(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ContractError(
            "synthetic specificity failure",
            code="synthetic_specificity_export_failure",
        )

    def fail_selection(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ContractError(
            "synthetic selection failure",
            code="synthetic_selection_export_failure",
        )

    monkeypatch.setattr(
        specificity_module,
        "summarize_frozen_family_specificity_support",
        fail_specificity,
    )
    specificity = summarize_frozen_specificity_support_universe(
        chain.point,
        chain.resampling,
        chain.universe,
        (chain.spec,),
    )
    monkeypatch.setattr(
        selection_module,
        "summarize_resampled_family_selection",
        fail_selection,
    )
    selection = summarize_frozen_selection_frequency_universe(
        chain.universe,
        chain.resampling,
    )

    document = export_bootstrap_support_document(specificity, selection)

    specificity_row = document.specificity_support.set_index("hypothesis_id").loc[
        chain.included.hypothesis_id
    ]
    assert specificity_row["status"] == "failed"
    assert specificity_row["reason_code"] == "synthetic_specificity_export_failure"
    assert pd.notna(specificity_row["target_id"])
    assert pd.notna(specificity_row["distribution_spec_id"])
    assert pd.isna(specificity_row["numeric_result_id"])
    assert pd.isna(specificity_row["bootstrap_source_binding_id"])
    selection_row = document.selection_frequency.iloc[0]
    assert selection_row["status"] == "failed"
    assert selection_row["reason_code"] == "synthetic_selection_export_failure"
    assert pd.isna(selection_row["target_id"])
    assert pd.isna(selection_row["numeric_result_id"])


def test_export_is_stable_and_rejects_nested_source_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, specificity, selection = _pair(monkeypatch, n_bootstraps=2)
    first = export_bootstrap_support_document(specificity, selection)
    repeated = export_bootstrap_support_document(specificity, selection)

    assert first.registry_id == repeated.registry_id
    pd.testing.assert_frame_equal(
        first.specificity_support,
        repeated.specificity_support,
    )
    pd.testing.assert_frame_equal(
        first.selection_frequency,
        repeated.selection_frequency,
    )

    reordered_universe = specificity_fixtures.freeze_hypothesis_universe(
        tuple(reversed(chain.universe.declarations)),
        universe_name=chain.universe.universe_name,
    )
    reordered_chain = replace(
        chain,
        universe=reordered_universe,
        primary=reordered_universe.declaration_for(chain.primary.hypothesis_id),
        included=reordered_universe.declaration_for(chain.included.hypothesis_id),
        filtered=reordered_universe.declaration_for(chain.filtered.hypothesis_id),
    )
    reordered_specificity = _specificity(monkeypatch, reordered_chain)
    reordered_selection = _selection(
        monkeypatch,
        reordered_chain,
        n_bootstraps=2,
    )
    reordered = export_bootstrap_support_document(
        reordered_specificity,
        reordered_selection,
    )
    assert reordered.registry_id == first.registry_id
    pd.testing.assert_frame_equal(
        reordered.specificity_support,
        first.specificity_support,
    )
    pd.testing.assert_frame_equal(
        reordered.selection_frequency,
        first.selection_frequency,
    )

    object.__setattr__(chain.included, "family_id", "forged-family")
    with pytest.raises(ContractError):
        export_bootstrap_support_document(specificity, selection)


def test_export_rejects_cross_universe_plan_and_source_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, specificity, _ = _pair(monkeypatch, n_bootstraps=2)

    other_universe_chain = specificity_fixtures._chain(n_bootstraps=2)
    other_universe_selection = _selection(
        monkeypatch,
        other_universe_chain,
        n_bootstraps=2,
    )
    with pytest.raises(ContractError) as universe_error:
        export_bootstrap_support_document(
            specificity,
            other_universe_selection,
        )
    assert universe_error.value.details.code == (
        "bootstrap_support_export_universe_mismatch"
    )

    other_source_chain = specificity_fixtures._chain(n_bootstraps=2)
    other_source_selection = _selection(
        monkeypatch,
        specificity_fixtures._BatchChain(
            point=other_source_chain.point,
            resampling=other_source_chain.resampling,
            universe=chain.universe,
            primary=chain.primary,
            included=chain.included,
            filtered=chain.filtered,
            spec=chain.spec,
            point_effect=chain.point_effect,
            bootstrap_effect=chain.bootstrap_effect,
            not_estimable_effect=chain.not_estimable_effect,
        ),
        n_bootstraps=2,
    )
    with pytest.raises(ContractError) as source_error:
        export_bootstrap_support_document(specificity, other_source_selection)
    assert source_error.value.details.code == (
        "bootstrap_support_export_source_mismatch"
    )

    other_plan_chain = specificity_fixtures._chain(n_bootstraps=3)
    other_plan_selection = _selection(
        monkeypatch,
        specificity_fixtures._BatchChain(
            point=other_plan_chain.point,
            resampling=other_plan_chain.resampling,
            universe=chain.universe,
            primary=chain.primary,
            included=chain.included,
            filtered=chain.filtered,
            spec=chain.spec,
            point_effect=chain.point_effect,
            bootstrap_effect=chain.bootstrap_effect,
            not_estimable_effect=chain.not_estimable_effect,
        ),
        n_bootstraps=3,
    )
    with pytest.raises(ContractError) as plan_error:
        export_bootstrap_support_document(specificity, other_plan_selection)
    assert plan_error.value.details.code == "bootstrap_support_export_plan_mismatch"
