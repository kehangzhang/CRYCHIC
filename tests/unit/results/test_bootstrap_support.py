from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from crychic.core import stable_id
from crychic.results.bootstrap_support import (
    BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
    BootstrapSupportDocument,
    bootstrap_support_contract,
    validate_bootstrap_support_links,
    validate_bootstrap_support_registry,
    validate_selection_frequency,
    validate_specificity_support,
)
from crychic.results.errors import ResultValidationError, ResultWriteError
from crychic.results.facade import CrychicResult
from crychic.results.persistence import write_result


def _plans() -> list[str]:
    return [f"bootstrap-{index:04d}" for index in range(1_000)]


def _specificity_row() -> dict[str, object]:
    return {
        "hypothesis_level": "driver_family_receiver",
        "hypothesis_role": "secondary",
        "hypothesis_id": "hypothesis-secondary",
        "contrast": "treated_vs_control",
        "mode": "state",
        "view": "gene",
        "receiver": "Receiver",
        "family_id": "family-1",
        "specificity_support": 0.8,
        "status": "observed",
        "reason_code": None,
        "minimum_effect": 0.1,
        "specificity_direction": "greater",
        "specificity_semantics": "frequentist-bootstrap-not-posterior-v1",
        "minimum_bootstraps": 1_000,
        "n_bootstrap_total": 1_000,
        "n_bootstrap_observed": 1_000,
        "n_bootstrap_not_estimable": 0,
        "n_bootstrap_failed": 0,
        "result_id": "specificity-result",
        "target_id": "secondary-target",
        "score_target_id": "primary-target",
        "declaration_id": "declaration-secondary",
        "hypothesis_universe_id": "universe-1",
        "effect_scale_id": "effect-scale",
        "point_crossfit_id": "point-crossfit",
        "crossfit_spec_id": "crossfit-spec",
        "point_effect_result_id": "point-effect",
        "bootstrap_source_binding_id": "specificity-binding",
        "bootstrap_plan_set_id": "plan-set",
        "effect_spec_id": "effect-spec",
        "distribution_spec_id": "distribution-spec",
        "source_distribution_id": "distribution",
        "specificity_support_spec_id": "support-spec",
        "numeric_result_id": "numeric-specificity",
        "source_authenticated": True,
        "source_binding_status": "frozen-workflow-authenticated-v1",
        "authentication_semantics": "exact-bootstrap-effect-binding-v1",
        "specificity_support_release_allowed": True,
        "formal_pq_inference_allowed": False,
        "is_posterior_probability": False,
        "is_comm_probability": False,
    }


def _selection_row() -> dict[str, object]:
    return {
        "hypothesis_level": "driver_family_receiver",
        "hypothesis_role": "primary",
        "hypothesis_id": "hypothesis-primary",
        "contrast": "all_contexts_v1",
        "mode": "state",
        "view": "gene",
        "receiver": "Receiver",
        "family_id": "family-1",
        "selection_frequency": 0.5,
        "status": "observed",
        "reason_code": None,
        "opportunity_grain": "equal_bootstrap_mean_of_outer_fold_selection_v1",
        "selection_frequency_semantics": "equal-bootstrap-selection-v1",
        "minimum_bootstraps": 1_000,
        "n_bootstrap_plans_total": 1_000,
        "n_bootstrap_plans_observed": 1_000,
        "n_bootstrap_plans_not_estimable": 0,
        "n_bootstrap_plans_failed": 0,
        "n_event_rows_total": 1_000,
        "n_observed_event_rows": 1_000,
        "n_not_estimable_event_rows": 0,
        "n_failed_event_rows": 0,
        "n_selected_event_rows": 500,
        "result_id": "selection-result",
        "target_id": "primary-target",
        "declaration_id": "declaration-primary",
        "hypothesis_universe_id": "universe-1",
        "crossfit_spec_id": "crossfit-spec",
        "resampling_result_id": "resampling-result",
        "source_resampled_attribution_collection_id": "attribution-collection",
        "source_binding_id": "selection-binding",
        "selection_frequency_spec_id": "selection-spec",
        "numeric_result_id": "numeric-selection",
        "bootstrap_plan_set_id": "plan-set",
        "opportunity_manifest_id": "opportunity-manifest",
        "selection_rule_id": "selection-rule",
        "selection_rule_semantics": "outer-training-coefficient-positive-v1",
        "selection_threshold": 0.0,
        "score_version": "family-common-v2",
        "source_record_set_id": "record-set",
        "source_authenticated": True,
        "source_binding_status": "frozen-workflow-authenticated-v1",
        "authentication_semantics": "exact-plan-fold-event-binding-v1",
        "exact_plan_coverage": True,
        "exact_fold_coverage": True,
        "complete_universe_coverage": True,
        "selection_frequency_release_allowed": True,
        "formal_pq_inference_allowed": False,
        "is_posterior_probability": False,
        "is_comm_probability": False,
    }


def _frame(row: dict[str, object], *, table: str) -> pd.DataFrame:
    contract = bootstrap_support_contract()
    columns = (
        contract.specificity.columns
        if table == "specificity"
        else contract.selection.columns
    )
    return pd.DataFrame([row], columns=columns)


def _registry() -> dict[str, object]:
    plans = _plans()
    declarations = [
        {
            "declaration_id": "declaration-primary",
            "hypothesis_id": "hypothesis-primary",
            "hypothesis_key": "primary-key",
            "endpoint": "driver_family_receiver_context_omnibus_v1",
            "contrast_name": "all_contexts_v1",
            "receiver": "Receiver",
            "family_id": "family-1",
            "mode": "state",
            "role": "primary",
            "multiplicity_family": "primary-family",
            "parent_key": None,
            "filter_stage": "pre_fit_outcome_independent",
            "prefilter_policy": "none_predeclared_v1",
            "prefilter_status": "included",
            "filter_reason_code": None,
        },
        {
            "declaration_id": "declaration-secondary",
            "hypothesis_id": "hypothesis-secondary",
            "hypothesis_key": "secondary-key",
            "endpoint": "family_common_integrated_lr_context_effect_v1",
            "contrast_name": "treated_vs_control",
            "receiver": "Receiver",
            "family_id": "family-1",
            "mode": "state",
            "role": "secondary",
            "multiplicity_family": "secondary-family",
            "parent_key": "primary-key",
            "filter_stage": "pre_fit_outcome_independent",
            "prefilter_policy": "none_predeclared_v1",
            "prefilter_status": "included",
            "filter_reason_code": None,
        },
    ]
    value: dict[str, object] = {
        "extension_schema_version": BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
        "result_schema_version": "0.1.0",
        "registry_kind": "frozen_bootstrap_support_lineage_v1",
        "hypothesis_universe": {
            "universe_id": "universe-1",
            "universe_name": "bootstrap-support-test",
            "declaration_ids": [item["declaration_id"] for item in declarations],
            "hypothesis_ids": [item["hypothesis_id"] for item in declarations],
            "selection_hypothesis_ids": ["hypothesis-primary"],
            "specificity_hypothesis_ids": ["hypothesis-secondary"],
            "declarations": declarations,
        },
        "resampling_lineage": {
            "resampling_result_id": "resampling-result",
            "exchangeability_id": "exchangeability-1",
            "source_input_identity_id": "input-identity",
            "source_input_digest": "input-digest",
            "source_snapshot_id": "snapshot-1",
            "config_digest": "config-digest",
            "crossfit_spec_id": "crossfit-spec",
            "resource_bundle_content_id": "resource-content",
            "target_prior_content_id": "target-prior-content",
            "root_seed_lineage": {
                "root_seed": 7,
                "path": ["bootstrap"],
                "derived_seed": 11,
            },
        },
        "bootstrap_plan_ids": plans,
        "specificity_sources": [
            {
                "result_id": "specificity-result",
                "hypothesis_id": "hypothesis-secondary",
                "target_id": "secondary-target",
                "score_target_id": "primary-target",
                "bootstrap_source_binding_id": "specificity-binding",
                "bootstrap_plan_ids": plans,
                "workflow_record_ids": [
                    f"workflow-{index:04d}" for index in range(1_000)
                ],
                "effect_record_ids": [f"effect-{index:04d}" for index in range(1_000)],
            }
        ],
        "selection_sources": [
            {
                "result_id": "selection-result",
                "hypothesis_id": "hypothesis-primary",
                "target_id": "primary-target",
                "source_resampled_attribution_collection_id": "attribution-collection",
                "source_binding_id": "selection-binding",
                "bootstrap_plan_ids": plans,
                "event_ids": [f"event-{index:04d}" for index in range(1_000)],
                "numeric_record_ids": [
                    f"numeric-{index:04d}" for index in range(1_000)
                ],
            }
        ],
    }
    value["registry_id"] = stable_id(
        "bootstrap_support_registry",
        value,
        schema_version=BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
    )
    return value


def _document() -> BootstrapSupportDocument:
    return BootstrapSupportDocument(
        specificity_support=_frame(_specificity_row(), table="specificity"),
        selection_frequency=_frame(_selection_row(), table="selection"),
        registry=_registry(),
    )


def _resign_registry(registry: dict[str, object]) -> None:
    payload = {key: value for key, value in registry.items() if key != "registry_id"}
    registry["registry_id"] = stable_id(
        "bootstrap_support_registry",
        payload,
        schema_version=BOOTSTRAP_SUPPORT_EXTENSION_VERSION,
    )


def _payload_with_linked_differential(
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    payload = {**result_payload, "tables": dict(result_payload["tables"])}
    differential = (
        payload["tables"]["differential"].iloc[[0, 0]].copy().reset_index(drop=True)
    )
    differential.loc[differential.index[0], "hypothesis_id"] = "hypothesis-primary"
    differential.loc[differential.index[0], "contrast"] = "all_contexts_v1"
    differential.loc[differential.index[1], "hypothesis_id"] = "hypothesis-secondary"
    differential.loc[differential.index[1], "contrast"] = "treated_vs_control"
    payload["tables"]["differential"] = differential
    return payload


def test_bootstrap_support_document_validates_exact_grains_and_registry() -> None:
    document = _document()

    assert document.registry_id.startswith("bootstrap_support_registry_")
    assert len(document.specificity_support) == 1
    assert len(document.selection_frequency) == 1
    assert "sender" not in document.selection_frequency.columns
    assert "interaction_id" not in document.selection_frequency.columns


def test_bootstrap_support_rejects_status_count_and_extra_column_mismatches() -> None:
    specificity = _frame(_specificity_row(), table="specificity")
    specificity.loc[0, "n_bootstrap_observed"] = 999
    with pytest.raises(ResultValidationError) as count_error:
        validate_specificity_support(specificity)
    assert count_error.value.details.code == "bootstrap_support_count_mismatch"

    selection = _frame(_selection_row(), table="selection")
    selection.loc[0, "status"] = "not_estimable"
    with pytest.raises(ResultValidationError) as status_error:
        validate_selection_frequency(selection)
    assert status_error.value.details.code == "bootstrap_support_status_mismatch"

    selection = _frame(_selection_row(), table="selection")
    selection["sender"] = "Sender"
    with pytest.raises(ResultValidationError) as column_error:
        validate_selection_frequency(selection)
    assert column_error.value.details.code == "invalid_bootstrap_support_table"


def test_specificity_support_retains_filtered_typed_row_without_fake_lineage() -> None:
    row = _specificity_row()
    row.update(
        {
            "specificity_support": None,
            "status": "not_estimable",
            "reason_code": "frozen_prefilter_low_support",
            "minimum_effect": None,
            "specificity_direction": None,
            "n_bootstrap_total": 0,
            "n_bootstrap_observed": 0,
            "target_id": None,
            "score_target_id": None,
            "effect_scale_id": None,
            "point_effect_result_id": None,
            "bootstrap_source_binding_id": None,
            "bootstrap_plan_set_id": None,
            "effect_spec_id": None,
            "distribution_spec_id": None,
            "source_distribution_id": None,
            "specificity_support_spec_id": None,
            "numeric_result_id": None,
            "source_binding_status": "frozen-prefiltered-not-estimable-v1",
            "specificity_support_release_allowed": False,
        }
    )

    validated = validate_specificity_support(_frame(row, table="specificity"))

    assert validated.loc[0, "status"] == "not_estimable"
    assert pd.isna(validated.loc[0, "target_id"])
    assert pd.isna(validated.loc[0, "specificity_support"])


def test_observed_specificity_requires_complete_child_lineage() -> None:
    row = _specificity_row()
    row["numeric_result_id"] = None

    with pytest.raises(ResultValidationError) as error:
        validate_specificity_support(_frame(row, table="specificity"))

    assert error.value.details.code == "bootstrap_support_status_mismatch"


def test_bootstrap_support_registry_is_tamper_evident() -> None:
    registry = deepcopy(_registry())
    registry["selection_sources"][0]["event_ids"][0] = "forged-event"  # type: ignore[index]

    with pytest.raises(ResultValidationError) as error:
        validate_bootstrap_support_registry(registry)
    assert error.value.details.code == "bootstrap_support_registry_identity_mismatch"


def test_bootstrap_support_rejects_resigned_invalid_hierarchy_and_source() -> None:
    hierarchy = deepcopy(_registry())
    hierarchy["hypothesis_universe"]["declarations"][1]["receiver"] = "Other"  # type: ignore[index]
    _resign_registry(hierarchy)
    with pytest.raises(ResultValidationError) as hierarchy_error:
        validate_bootstrap_support_registry(hierarchy)
    assert hierarchy_error.value.details.code == "bootstrap_support_universe_mismatch"

    source = deepcopy(_registry())
    source["specificity_sources"][0]["target_id"] = "other-target"  # type: ignore[index]
    _resign_registry(source)
    with pytest.raises(ResultValidationError) as source_error:
        BootstrapSupportDocument(
            specificity_support=_frame(_specificity_row(), table="specificity"),
            selection_frequency=_frame(_selection_row(), table="selection"),
            registry=source,
        )
    assert source_error.value.details.code == (
        "bootstrap_support_registry_table_mismatch"
    )


def test_bootstrap_support_links_both_hypothesis_grains_to_differential() -> None:
    document = _document()
    differential = pd.DataFrame(
        [
            {
                "hypothesis_level": "driver_family_receiver",
                "hypothesis_id": "hypothesis-primary",
                "contrast": "all_contexts_v1",
                "mode": "state",
                "view": "gene",
            },
            {
                "hypothesis_level": "driver_family_receiver",
                "hypothesis_id": "hypothesis-secondary",
                "contrast": "treated_vs_control",
                "mode": "state",
                "view": "gene",
            },
        ]
    )
    validate_bootstrap_support_links(document, differential)
    differential = differential.iloc[[0]].copy()

    with pytest.raises(ResultValidationError) as error:
        validate_bootstrap_support_links(document, differential)
    assert error.value.details.code == "bootstrap_support_differential_link_mismatch"


def test_bootstrap_support_atomic_round_trip_and_lazy_queries(
    tmp_path: Path,
    result_payload: dict[str, Any],
) -> None:
    destination = tmp_path / "with-bootstrap-support"
    payload = _payload_with_linked_differential(result_payload)

    result = write_result(
        destination,
        **payload,
        bootstrap_support=_document(),
    )
    loaded = CrychicResult.load(destination)

    assert result.has_bootstrap_support
    assert loaded.has_bootstrap_support
    assert loaded.manifest["extensions"]["bootstrap_support"][
        "extension_schema_version"
    ] == BOOTSTRAP_SUPPORT_EXTENSION_VERSION
    registry = loaded.read_bootstrap_support_registry()
    assert registry["registry_id"] == _document().registry_id
    specificity = loaded.read_specificity_support(
        filters={"hypothesis_id": "hypothesis-secondary"},
        columns=["hypothesis_id", "specificity_support", "status"],
    )
    selection = loaded.read_selection_frequency(
        filters={"hypothesis_id": "hypothesis-primary"},
        columns=["hypothesis_id", "selection_frequency", "status"],
    )
    assert specificity.to_dict(orient="records") == [
        {
            "hypothesis_id": "hypothesis-secondary",
            "specificity_support": 0.8,
            "status": "observed",
        }
    ]
    assert selection.to_dict(orient="records") == [
        {
            "hypothesis_id": "hypothesis-primary",
            "selection_frequency": 0.5,
            "status": "observed",
        }
    ]


def test_legacy_result_omits_bootstrap_support_without_shape_change(
    tmp_path: Path,
    result_payload: dict[str, Any],
) -> None:
    result = write_result(tmp_path / "legacy", **result_payload)

    assert not result.has_bootstrap_support
    assert "extensions" not in result.manifest
    with pytest.raises(KeyError, match="bootstrap_support"):
        result.read_selection_frequency()


def test_bootstrap_support_corruption_is_rejected_on_load(
    tmp_path: Path,
    result_payload: dict[str, Any],
) -> None:
    destination = tmp_path / "corrupted-bootstrap-support"
    payload = _payload_with_linked_differential(result_payload)
    write_result(destination, **payload, bootstrap_support=_document())
    registry_path = destination / "bootstrap_support_registry.json"
    registry_path.write_bytes(registry_path.read_bytes() + b"\n")

    with pytest.raises(ResultValidationError, match="does not match its manifest"):
        CrychicResult.load(destination)


def test_writer_revalidates_mutated_bootstrap_support_document(
    tmp_path: Path,
    result_payload: dict[str, Any],
) -> None:
    payload = _payload_with_linked_differential(result_payload)
    document = _document()
    document.selection_frequency.loc[0, "selection_frequency"] = 2.0

    with pytest.raises(ResultWriteError, match="marked incomplete"):
        write_result(
            tmp_path / "mutated-bootstrap-support",
            **payload,
            bootstrap_support=document,
        )
