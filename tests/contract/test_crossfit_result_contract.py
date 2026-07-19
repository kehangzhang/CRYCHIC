from __future__ import annotations

import copy
import math
from typing import Any, cast

import pandas as pd
import pytest

from crychic import CrossFitResult, run_subject_crossfit, write_crossfit_result
from crychic.attribution.gain_calibration import GAIN_CALIBRATION_PERCENTILE_POLICY
from crychic.core import canonical_digest, canonical_json, stable_id
from crychic.results import ResultValidationError
from crychic.scoring import (
    GLOBAL_COMMON_LR_SCORE_COLUMNS,
    GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    SCORING_COLLECTION_EXTENSION_VERSION,
    SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    PlannedScoringCollectionManifest,
    ReceiverScoringFunctionalManifest,
    ScoringCollectionDocument,
    ScoringCollectionManifest,
)
from crychic.workflow.crossfit_persistence import (
    CROSSFIT_COMPONENT_COLUMNS,
    CROSSFIT_COMPONENT_TABLE,
    CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_LR_TABLE,
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
    CROSSFIT_DIFFERENTIAL_COLUMNS,
    CROSSFIT_DIFFERENTIAL_TABLE,
    CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS,
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
    CROSSFIT_RECEIVER_TABLE_NAMES,
    CROSSFIT_RECEIVER_TRAINING_SUPPORT_COLUMNS,
    CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
    CROSSFIT_RESULT_SCHEMA_VERSION,
    CROSSFIT_TABLE_NAMES,
    _bundle_table_names,
    _canonical_table_scalar,
    _directional_binding_rows,
    _global_source_table_digest,
    _require_identifiers,
    _source_row_id,
    _table_columns,
    _table_schema_version,
    _validate_contrast_common_collections_manifest,
    _validate_contrast_common_lr_table,
    _validate_contrast_common_registry_links,
    _validate_contrast_common_sender_lr_table,
    _validate_contrast_common_table_lineage,
    _validate_directional_channel_registry_table,
    _validate_directional_registry_links,
    _validate_manifest,
    _validate_receiver_training_support_links,
    _validate_receiver_training_support_table,
    _validate_registry_links,
    _validate_sender_lr_cross_table_lineage,
    _validate_source_contrast_common_registry,
    _validate_source_directional_lr_hypothesis_universe,
    _validate_source_receiver_family_opportunity_universe,
    _validate_source_receiver_universe,
    _validate_source_scoring_registry,
    _validate_table,
)

_UNCERTIFIED_CLAIM_SCOPE = "heldout_family_common_diagnostic_not_complete_oof_certified"
_GLOBAL_UNCERTIFIED_CLAIM_SCOPE = (
    "heldout_cross_receiver_common_diagnostic_not_complete_oof_certified"
)


def _directional_binding(*, status: str = "observed") -> dict[str, object]:
    forward_contrast: dict[str, object] = {
        "name": "stim_vs_control",
        "family": "treatment",
        "mode": "balanced",
        "estimable": True,
        "reason_code": None,
        "weights": [
            {"context": "control", "weight": -1.0},
            {"context": "stim", "weight": 1.0},
        ],
    }
    reverse_contrast: dict[str, object] = {
        "name": "control_vs_stim",
        "family": "treatment",
        "mode": "balanced",
        "estimable": True,
        "reason_code": None,
        "weights": [
            {"context": "control", "weight": 1.0},
            {"context": "stim", "weight": -1.0},
        ],
    }
    forward_contrast_id = stable_id("contrast", forward_contrast)
    reverse_contrast_id = stable_id("contrast", reverse_contrast)
    pair_spec_payload = {
        "combination_rule": "independent_contrast_views_not_additive_v1",
        "forward_channel_name": "increased_activation_compatible",
        "forward_contrast_id": forward_contrast_id,
        "reverse_channel_name": "reduced_activation_compatible",
        "reverse_contrast_id": reverse_contrast_id,
        "schema_version": "1.0.0",
        "supports_active_inhibition_claim": False,
    }
    pair_spec_id = stable_id(
        "directional_contrast_pair_spec",
        pair_spec_payload,
        schema_version="1",
    )
    pair_spec = {
        "pair_spec_id": pair_spec_id,
        "schema_version": "1.0.0",
        "forward_contrast_id": forward_contrast_id,
        "reverse_contrast_id": reverse_contrast_id,
        "forward_contrast": forward_contrast,
        "reverse_contrast": reverse_contrast,
        "forward_channel_name": "increased_activation_compatible",
        "reverse_channel_name": "reduced_activation_compatible",
        "supports_active_inhibition_claim": False,
        "combination_rule": "independent_contrast_views_not_additive_v1",
    }
    response_pair_id = "response-pair-1" if status == "observed" else None
    response_pair: dict[str, object] | None = None
    if status == "observed":
        response_pair = {
            "response_pair_id": response_pair_id,
            "forward_channel_id": "response-channel-forward-1",
            "reverse_channel_id": "response-channel-reverse-1",
            "forward_contrast_id": forward_contrast_id,
            "reverse_contrast_id": reverse_contrast_id,
            "forward_response_id": "response-forward-1",
            "reverse_response_id": "response-reverse-1",
            "feature_ids": ["G1", "G2"],
            "prior_semantics": "nonnegative_activation_prior_v1",
            "forward_biological_claim": "direction_compatible_activation",
            "reverse_biological_claim": (
                "attenuation_of_activation_from_explicit_reverse_contrast"
            ),
            "supports_active_inhibition_claim": False,
            "combination_rule": "independent_contrast_views_not_additive_v1",
        }
    identity_payload: dict[str, object] = {
        "combination_rule": "independent_contrast_views_not_additive_v1",
        "active_inhibition_allowed": False,
        "feature_ids": ["G1", "G2"],
        "fold_id": "fold-1",
        "formal_inference_allowed": False,
        "forward_application_id": "application-forward-1",
        "forward_channel_name": "increased_activation_compatible",
        "forward_response_id": "response-forward-1",
        "forward_training_artifact_id": "model-forward-1",
        "heldout_subject_ids": ["subject-heldout-1"],
        "paired_score_comparison_allowed": False,
        "pair_spec_id": pair_spec_id,
        "reason_code": (
            None if status == "observed" else "directional_components_not_estimable"
        ),
        "receiver": "Receiver",
        "response_pair_id": response_pair_id,
        "reverse_application_id": "application-reverse-1",
        "reverse_channel_name": "reduced_activation_compatible",
        "reverse_response_id": "response-reverse-1",
        "reverse_training_artifact_id": "model-reverse-1",
        "status": status,
        "supports_active_inhibition_claim": False,
        "training_subject_ids": ["subject-training-1"],
    }
    return {
        "binding_id": stable_id(
            "directional_crossfit_binding", identity_payload, schema_version="1"
        ),
        **identity_payload,
        "pair_spec": pair_spec,
        "response_pair": response_pair,
    }


def _directional_registry_case(
    *,
    status: str = "observed",
) -> tuple[dict[str, object], pd.DataFrame]:
    binding = _directional_binding(status=status)
    rows = _directional_binding_rows(
        binding,
        crossfit_id="crossfit-1",
        spec_id="spec-1",
        repeat_id="repeat-1",
        fold_id="fold-1",
        certification_status="uncertified",
        is_oof_certified=False,
        claim_scope=_UNCERTIFIED_CLAIM_SCOPE,
    )
    manifest, _, _ = _v7_receiver_registry_case()
    source = _source(manifest)
    universe = cast(dict[str, object], source["receiver_universe"])
    source.update(
        {
            "directional_pair_stage_connected": True,
            "n_directional_response_bindings": 1,
            "n_directional_receiver_pair_opportunities": 2,
            "n_directional_receiver_pair_absent_training": 1,
            "directional_receiver_universe_id": universe["universe_id"],
            "directional_receiver_axis_id": universe["receiver_axis_id"],
            "directional_binding_status_counts": {status: 1},
            "directional_registry_complete": False,
            "directional_supported_binding_registry_complete": True,
            "directional_diagnostic_complete": status == "observed",
            "directional_formal_inference_allowed": False,
            "spec": {"directional_pairs": [binding["pair_spec"]]},
        }
    )
    folds = cast(list[dict[str, object]], source["fold_artifacts"])
    folds[0].update(
        {
            "directional_response_bindings": [binding],
            "receiver_incremental_artifacts": [
                {
                    "contrast_name": "stim_vs_control",
                    "receiver": "Receiver",
                },
                {
                    "contrast_name": "control_vs_stim",
                    "receiver": "Receiver",
                },
            ],
        }
    )
    manifest.update(
        {
            "certification_status": "uncertified",
            "complete_pipeline_oof_certified": False,
            "claim_scope": _UNCERTIFIED_CLAIM_SCOPE,
        }
    )
    return manifest, pd.DataFrame(
        rows,
        columns=CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS,
    )


def _gain_calibration_binding() -> dict[str, object]:
    payload: dict[str, object] = {
        "receiver": "Receiver",
        "source_family_common_functional_id": "functional-1",
        "gain_calibration_artifact_id": "gain-artifact-1",
        "gain_calibration_spec_id": "gain-spec-1",
        "gain_calibration_status": "observed",
        "gain_calibration_reason_code": None,
        "percentile_policy": GAIN_CALIBRATION_PERCENTILE_POLICY,
        "positive_gain_source_knots": [0.0, 0.5, 1.0],
        "positive_gain_percentile_knots": [0.0, 0.8, 1.0],
        "n_supported_families": 3,
        "n_positive_observations": 5,
        "n_distinct_positive_gains": 2,
        "tuning_id": "tuning-1",
        "outer_incremental_functional_id": "incremental-functional-1",
        "outer_selected_resolved_penalty_id": "resolved-penalty-1",
    }
    return {
        "gain_calibration_binding_id": stable_id(
            "receiver_gain_calibration_binding", payload, schema_version="1"
        ),
        **payload,
    }


def _contrast_common_lr_table() -> pd.DataFrame:
    binding = _gain_calibration_binding()
    row: dict[str, object] = {
        column: None for column in CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
    }
    row.update(
        {
            "crossfit_id": "crossfit-1",
            "spec_id": "spec-1",
            "repeat_id": "repeat-1",
            "fold_id": "fold-1",
            "contrast_id": "contrast-1",
            "contrast": "treated_vs_control",
            "contrast_common_collection_id": "collection-1",
            "global_common_application_id": "global-application-1",
            "global_common_functional_id": "global-functional-1",
            "source_family_common_functional_id": "functional-1",
            "source_family_common_application_id": "application-1",
            "sample_id": "sample-1",
            "subject_id": "subject-1",
            "context_id": "treated",
            "receiver": "Receiver",
            "family_id": "family-1",
            "driver_id": "R1",
            "interaction_id": "L1_R1",
            "mode": "state",
            "receptor_eligible": True,
            "ligand_contrast_gate_status": "supported",
            "availability": 0.6,
            "receiver_relative_family_gain": 0.5,
            "calibrated_family_gain_percentile": 0.8,
            "gain_calibration_binding_id": binding["gain_calibration_binding_id"],
            "gain_calibration_artifact_id": binding["gain_calibration_artifact_id"],
            "gain_calibration_status": "observed",
            "prior_quality": 0.9,
            "training_family_coefficient": 0.4,
            "global_lr_core_strength": 0.6661334853348485,
            "global_lr_score": 0.5995201368013636,
            "status": "observed",
            "score_version": "global-score-v1",
            "certification_status": "uncertified",
            "is_oof_certified": False,
            "formal_inference_status": "not_available_descriptive_only",
            "claim_scope": _GLOBAL_UNCERTIFIED_CLAIM_SCOPE,
        }
    )
    row["source_row_id"] = _source_row_id(
        source_table=CROSSFIT_CONTRAST_COMMON_LR_TABLE,
        application_id="global-application-1",
        sample_id="sample-1",
        subject_id="subject-1",
        context_id="treated",
        receiver="Receiver",
        family_id="family-1",
        driver_id="R1",
        interaction_id="L1_R1",
        mode="state",
    )
    return pd.DataFrame([row], columns=CROSSFIT_CONTRAST_COMMON_LR_COLUMNS)


def _contrast_common_sender_table() -> pd.DataFrame:
    binding = _gain_calibration_binding()
    rows: list[dict[str, object]] = []
    weights = (2.0 / 3.0, 1.0 / 3.0)
    normalized_entropy = -sum(
        weight * math.log(weight) for weight in weights
    ) / math.log(len(weights))
    for sender, sender_application_id, evidence, assignment_weight in (
        ("SenderA", "sender-application-1", 0.4, weights[0]),
        ("SenderB", "sender-application-2", 0.2, weights[1]),
    ):
        row: dict[str, object] = {
            column: None for column in CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
        }
        row.update(
            {
                "crossfit_id": "crossfit-1",
                "spec_id": "spec-1",
                "repeat_id": "repeat-1",
                "fold_id": "fold-1",
                "contrast_id": "contrast-1",
                "contrast": "treated_vs_control",
                "contrast_common_collection_id": "collection-1",
                "global_common_application_id": "global-application-1",
                "global_common_functional_id": "global-functional-1",
                "source_family_common_functional_id": "functional-1",
                "source_family_common_application_id": "application-1",
                "source_sender_functional_id": "sender-functional-1",
                "source_sender_application_id": sender_application_id,
                "sample_id": "sample-1",
                "subject_id": "subject-1",
                "context_id": "treated",
                "receiver": "Receiver",
                "family_id": "family-1",
                "driver_id": "R1",
                "interaction_id": "L1_R1",
                "mode": "state",
                "sender": sender,
                "global_lr_score": 0.5995201368013636,
                "gain_calibration_binding_id": binding["gain_calibration_binding_id"],
                "gain_calibration_artifact_id": binding["gain_calibration_artifact_id"],
                "gain_calibration_status": "observed",
                "ligand_availability": evidence * 2.0,
                "training_prevalence_prior": 0.5,
                "raw_sender_evidence": evidence,
                "assignment_weight": assignment_weight,
                "normalized_entropy": normalized_entropy,
                "global_sender_lr_score": (0.5995201368013636 * assignment_weight),
                "status": "observed",
                "score_version": "global-score-v1",
                "certification_status": "uncertified",
                "is_oof_certified": False,
                "formal_inference_status": "not_available_descriptive_only",
                "claim_scope": _GLOBAL_UNCERTIFIED_CLAIM_SCOPE,
            }
        )
        row["source_row_id"] = _source_row_id(
            source_table=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            application_id="global-application-1",
            sample_id="sample-1",
            subject_id="subject-1",
            context_id="treated",
            receiver="Receiver",
            family_id="family-1",
            driver_id="R1",
            interaction_id="L1_R1",
            mode="state",
            sender=sender,
        )
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
    )


def _contrast_common_manifest() -> dict[str, object]:
    binding = _gain_calibration_binding()
    application: dict[str, object] = {
        "fold_id": "fold-1",
        "global_common_functional_id": "global-functional-1",
        "global_common_application_id": "global-application-1",
        "functional_spec_id": "functional-spec-1",
        "functional_schema_version": "4.0.0",
        "filter_universe_id": "filter-universe-1",
        "context_ids": ["treated"],
        "receiver_ids": ["Receiver"],
        "training_subject_ids": ["training-subject-1"],
        "heldout_subject_ids": ["subject-1"],
        "receiver_children": [
            {
                "receiver": "Receiver",
                "family_common_functional_id": "functional-1",
                "family_common_application_id": "application-1",
            }
        ],
        "sender_functional_id": "sender-functional-1",
        "sender_lineages": [
            {
                "receiver": "Receiver",
                "sender_functional_id": "sender-functional-1",
                "sender_application_id": "sender-application-1",
            },
            {
                "receiver": "Receiver",
                "sender_functional_id": "sender-functional-1",
                "sender_application_id": "sender-application-2",
            },
        ],
        "sender_application_digests": [
            {
                "receiver": "Receiver",
                "sender_application_digest": "sender-digest-1",
            }
        ],
        "calibration_policy": "selected_penalty_inner_oof_positive_gain_ecdf_v1",
        "scale_policy": (
            "selected_penalty_inner_oof_gain_percentile_no_heldout_rescaling_v3"
        ),
        "sender_policy": "contrast_common_frozen_softmax_conserved_allocation_v1",
        "softmin_power": 4.0,
        "epsilon": 1e-12,
        "receiver_gain_calibration_bindings": [binding],
        "all_receivers_gain_calibrated": True,
        "cross_receiver_percentile_rank_eligible": True,
        "training_only_receiver_calibration": True,
        "receiver_scale_amplification": False,
        "common_functional_across_receivers": False,
        "receiver_balanced_descriptive_collection": True,
        "interaction_mapping_digest": "interaction-mapping-digest-1",
        "functional_certification_status": "heldout-diagnostic",
        "lr_scores_digest": "lr-digest-1",
        "sender_scores_digest": "sender-digest-1",
        "n_lr_rows": 1,
        "n_sender_rows": 2,
    }
    payload: dict[str, object] = {
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "contrast_id": "contrast-1",
        "contrast": "treated_vs_control",
        "score_version": "global-score-v1",
        "estimand": "receiver_balanced_descriptive_collection_v1",
        "common_functional_across_receivers": False,
        "receiver_balanced_descriptive_collection": True,
        "formal_inference_allowed": False,
        "certification_status": "uncertified",
        "is_oof_certified": False,
        "claim_scope": _GLOBAL_UNCERTIFIED_CLAIM_SCOPE,
        "fold_applications": [application],
    }
    collection = {
        "contrast_common_collection_id": stable_id(
            "contrast_common_oof_score_collection",
            payload,
            schema_version="1",
        ),
        **payload,
    }
    return {
        "schema_version": CROSSFIT_RESULT_SCHEMA_VERSION,
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "certification_status": "uncertified",
        "complete_pipeline_oof_certified": False,
        "contrast_common_stage_connected": True,
        "contrast_common_collections": [collection],
    }


def _contrast_common_lineage_case() -> tuple[
    dict[str, tuple[dict[str, object], dict[str, object]]],
    dict[str, pd.DataFrame],
]:
    manifest = _contrast_common_manifest()
    collection = cast(list[dict[str, object]], manifest["contrast_common_collections"])[
        0
    ]
    application = cast(list[dict[str, object]], collection["fold_applications"])[0]
    lr = _contrast_common_lr_table()
    sender = _contrast_common_sender_table()
    application["lr_scores_digest"] = _global_source_table_digest(
        "cross_receiver_common_lr_scores",
        lr.loc[:, list(GLOBAL_COMMON_LR_SCORE_COLUMNS)],
        GLOBAL_COMMON_LR_SCORE_COLUMNS,
    )
    application["sender_scores_digest"] = _global_source_table_digest(
        "cross_receiver_common_sender_scores",
        sender.loc[:, list(GLOBAL_COMMON_SENDER_SCORE_COLUMNS)],
        GLOBAL_COMMON_SENDER_SCORE_COLUMNS,
    )
    payload = {
        key: value
        for key, value in collection.items()
        if key != "contrast_common_collection_id"
    }
    collection_id = stable_id(
        "contrast_common_oof_score_collection",
        payload,
        schema_version="1",
    )
    collection["contrast_common_collection_id"] = collection_id
    lr.loc[:, "contrast_common_collection_id"] = collection_id
    sender.loc[:, "contrast_common_collection_id"] = collection_id
    return (
        {"global-application-1": (collection, application)},
        {
            CROSSFIT_CONTRAST_COMMON_LR_TABLE: lr,
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: sender,
        },
    )


def _contrast_common_source_case() -> tuple[
    dict[str, object],
    dict[str, tuple[dict[str, object], dict[str, object]]],
]:
    applications, _ = _contrast_common_lineage_case()
    collection, application = applications["global-application-1"]
    source_record = {
        "contrast_name": collection["contrast"],
        "contrast_manifest_id": collection["contrast_id"],
        "functional_id": application["global_common_functional_id"],
        "application_id": application["global_common_application_id"],
        "receiver_ids": application["receiver_ids"],
        "child_functional_ids": ["functional-1"],
        "score_version": collection["score_version"],
        "estimand": collection["estimand"],
        "functional_spec_id": application["functional_spec_id"],
        "functional_schema_version": application["functional_schema_version"],
        "calibration_policy": application["calibration_policy"],
        "scale_policy": application["scale_policy"],
        "sender_policy": application["sender_policy"],
        "softmin_power": application["softmin_power"],
        "epsilon": application["epsilon"],
        "receiver_gain_calibration_bindings": application[
            "receiver_gain_calibration_bindings"
        ],
        "all_receivers_gain_calibrated": application["all_receivers_gain_calibrated"],
        "cross_receiver_percentile_rank_eligible": application[
            "cross_receiver_percentile_rank_eligible"
        ],
        "lr_scores_digest": application["lr_scores_digest"],
        "sender_scores_digest": application["sender_scores_digest"],
        "n_lr_rows": application["n_lr_rows"],
        "n_sender_rows": application["n_sender_rows"],
        "common_functional_across_receivers": False,
        "receiver_balanced_descriptive_collection": True,
        "training_only_receiver_calibration": True,
        "receiver_scale_amplification": False,
        "formal_inference_allowed": False,
    }
    manifest: dict[str, object] = {
        "schema_version": CROSSFIT_RESULT_SCHEMA_VERSION,
        "source_crossfit_manifest": {
            "fold_artifacts": [
                {
                    "fold_id": "fold-1",
                    "cross_receiver_common_scoring_artifacts": [source_record],
                    "family_common_scoring_artifacts": [
                        {
                            "receiver": "Receiver",
                            "functional_id": "functional-1",
                            "application_id": "application-1",
                        }
                    ],
                }
            ]
        },
    }
    return manifest, applications


def _downgrade_common_record_to_v3(record: dict[str, object]) -> None:
    for field_name in (
        "scale_policy",
        "sender_policy",
        "softmin_power",
        "epsilon",
        "receiver_gain_calibration_bindings",
        "all_receivers_gain_calibrated",
        "cross_receiver_percentile_rank_eligible",
    ):
        record.pop(field_name)
    record.update(
        {
            "calibration_policy": (
                "positive_family_coefficient_quantile_median_shrink_only_v1"
            ),
            "calibration_quantile": 0.75,
            "global_training_anchor": 0.8,
            "receiver_training_anchors": [["Receiver", 1.0]],
            "receiver_scale_factors": [["Receiver", 0.8]],
        }
    )


def _scoring_registry(
    *,
    planned_only_registered: bool = False,
) -> ScoringCollectionDocument:
    primary = ReceiverScoringFunctionalManifest.registered(
        receiver="Receiver",
        receiver_family_model_id="receiver-family-1",
        receiver_incremental_model_id="receiver-incremental-1",
        filter_universe_id="filter-universe-1",
        scoring_functional_id="functional-1",
        score_version="score-v1",
        functional_status="observed",
        contract_version=SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    )
    if planned_only_registered:
        planned_only = ReceiverScoringFunctionalManifest.registered(
            receiver="PlannedOnly",
            receiver_family_model_id="receiver-family-2",
            receiver_incremental_model_id="receiver-incremental-2",
            filter_universe_id="filter-universe-1",
            scoring_functional_id="functional-2",
            score_version="score-v1",
            functional_status="observed",
            contract_version=SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
        )
    else:
        planned_only = ReceiverScoringFunctionalManifest.not_produced(
            receiver="PlannedOnly",
            receiver_family_model_id="receiver-family-2",
            receiver_incremental_model_id="receiver-incremental-2",
            filter_universe_id="filter-universe-1",
            reason_code="functional_not_requested",
            contract_version=SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
        )
    collection = ScoringCollectionManifest.planned_receiver_registry(
        contrast="treated_vs_control",
        repeat_id="repeat-1",
        fold_id="fold-1",
        planned_receivers=("PlannedOnly", "Receiver"),
        children=(primary, planned_only),
        contract_version=SCORING_COLLECTION_FOLD_LOCAL_AUTHORITATIVE_VERSION,
    )
    return ScoringCollectionDocument(
        collections=(collection,),
        planned_collections=(
            PlannedScoringCollectionManifest.from_collection(
                collection,
                filter_universe_id="filter-universe-1",
            ),
        ),
    )


def _registry_link_case() -> tuple[
    dict[str, object],
    dict[str, pd.DataFrame],
]:
    registry = _scoring_registry()
    assert registry.registry_id is not None
    application: dict[str, object] = {
        "fold_id": "fold-1",
        "contrast_id": "contrast-id-1",
        "contrast": "treated_vs_control",
        "receiver": "Receiver",
        "family_common_functional_id": "functional-1",
        "family_common_application_id": "application-1",
        "family_common_binding_id": "binding-1",
        "sender_functional_id": "sender-functional-1",
        "score_version": "score-v1",
        "certification_status": "uncertified",
        "is_oof_certified": False,
        "source_table_digests": {},
        "source_table_row_counts": {},
        "persisted_component_rows": 1,
        "persisted_differential_rows": 1,
    }
    row = {
        "crossfit_id": "crossfit-1",
        "fold_id": "fold-1",
        "contrast_id": "contrast-id-1",
        "contrast": "treated_vs_control",
        "receiver": "Receiver",
        "score_version": "score-v1",
        "family_common_functional_id": "functional-1",
        "family_common_application_id": "application-1",
        "family_common_binding_id": "binding-1",
        "sender_functional_id": "sender-functional-1",
        "certification_status": "uncertified",
        "is_oof_certified": False,
        "claim_scope": _UNCERTIFIED_CLAIM_SCOPE,
    }
    manifest: dict[str, object] = {
        "schema_version": "2.0.0",
        "crossfit_id": "crossfit-1",
        "repeat_id": "repeat-1",
        "certification_status": "uncertified",
        "complete_pipeline_oof_certified": False,
        "applications": [application],
        "source_crossfit_manifest": {
            "receiver_scoring_registry_id": registry.registry_id,
            "receiver_scoring_registry": registry.to_dict(),
        },
    }
    tables = {
        CROSSFIT_COMPONENT_TABLE: pd.DataFrame([row]),
        CROSSFIT_DIFFERENTIAL_TABLE: pd.DataFrame(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key != "sender_functional_id"
                }
            ]
        ),
    }
    return manifest, tables


def _v7_receiver_registry_case() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    pd.DataFrame,
]:
    config_digest = "config-digest-1"
    subject_ids = ["subject-1"]
    sample_ids = ["sample-1"]
    subject_content_digests = [["subject-1", "subject-digest-1"]]
    input_digest = stable_id(
        "sanitized_raw_fold_input",
        {
            "config_digest": config_digest,
            "subject_content_digests": subject_content_digests,
        },
        schema_version="2",
    )
    root_payload = {
        "config_digest": config_digest,
        "input_digest": input_digest,
        "sample_ids": sample_ids,
        "subject_content_digests": subject_content_digests,
        "subject_ids": subject_ids,
    }
    root_identity_id = stable_id(
        "sanitized_raw_input_identity", root_payload, schema_version="1"
    )
    root_identity = {
        "identity_id": root_identity_id,
        "config_digest": config_digest,
        "input_digest": input_digest,
        "subject_ids": subject_ids,
        "sample_ids": sample_ids,
        "subject_content_digests": [
            {"subject_id": "subject-1", "digest": "subject-digest-1"}
        ],
    }
    receiver_ids = ["PlannedOnly", "Receiver"]
    receiver_axis_id = stable_id(
        "receiver_axis",
        {"receiver_ids": receiver_ids},
        schema_version="1",
    )
    universe_payload = {
        "observed_cell_type_ids": ["Receiver"],
        "receiver_axis_id": receiver_axis_id,
        "receiver_ids": receiver_ids,
        "root_config_digest": config_digest,
        "root_input_digest": input_digest,
        "root_input_identity_id": root_identity_id,
        "root_subject_ids": subject_ids,
        "source_policy": "explicit_predeclared_receiver_ids_v1",
    }
    universe_id = stable_id(
        "frozen_receiver_universe", universe_payload, schema_version="1"
    )
    universe = {"universe_id": universe_id, **universe_payload}
    training_ids = ["Receiver"]
    support_records: list[dict[str, object]] = []
    for receiver in receiver_ids:
        status = "observed" if receiver in training_ids else "not_estimable"
        reason = None if status == "observed" else "receiver_absent_in_outer_training"
        support_payload = {
            "outer_fold_id": "fold-1",
            "reason_code": reason,
            "receiver_id": receiver,
            "receiver_universe_id": universe_id,
            "status": status,
            "training_cell_type_ids": training_ids,
        }
        support_records.append(
            {
                "support_record_id": stable_id(
                    "receiver_training_support",
                    support_payload,
                    schema_version="1",
                ),
                **support_payload,
            }
        )
    support_by_receiver = {
        str(record["receiver_id"]): record for record in support_records
    }
    primary_support = support_by_receiver["Receiver"]
    primary = ReceiverScoringFunctionalManifest.registered(
        receiver="Receiver",
        receiver_family_model_id="receiver-family-1",
        receiver_incremental_model_id="receiver-incremental-1",
        filter_universe_id="filter-universe-1",
        scoring_functional_id="functional-1",
        score_version="score-v1",
        functional_status="observed",
        receiver_training_support_id=str(primary_support["support_record_id"]),
        receiver_training_support_status="observed",
        receiver_training_support_reason_code=None,
        contract_version=SCORING_COLLECTION_EXTENSION_VERSION,
    )
    absent_support = support_by_receiver["PlannedOnly"]
    planned_only = ReceiverScoringFunctionalManifest.training_not_estimable(
        receiver="PlannedOnly",
        filter_universe_id="filter-universe-1",
        receiver_training_support_id=str(absent_support["support_record_id"]),
    )
    collection = ScoringCollectionManifest.planned_receiver_registry(
        contrast="treated_vs_control",
        repeat_id="repeat-1",
        fold_id="fold-1",
        planned_receivers=("PlannedOnly", "Receiver"),
        children=(primary, planned_only),
        contract_version=SCORING_COLLECTION_EXTENSION_VERSION,
    )
    registry = ScoringCollectionDocument(
        collections=(collection,),
        planned_collections=(
            PlannedScoringCollectionManifest.from_collection(
                collection,
                filter_universe_id="filter-universe-1",
            ),
        ),
    )
    assert registry.registry_id is not None
    application: dict[str, object] = {
        "fold_id": "fold-1",
        "contrast_id": "contrast-id-1",
        "contrast": "treated_vs_control",
        "receiver": "Receiver",
        "family_common_functional_id": "functional-1",
        "family_common_application_id": "application-1",
        "family_common_binding_id": "binding-1",
        "sender_functional_id": "sender-functional-1",
        "score_version": "score-v1",
        "certification_status": "uncertified",
        "is_oof_certified": False,
        "source_table_digests": {},
        "source_table_row_counts": {},
        "persisted_component_rows": 1,
        "persisted_differential_rows": 1,
    }
    manifest: dict[str, object] = {
        "schema_version": CROSSFIT_RESULT_SCHEMA_VERSION,
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "root_input_digest": input_digest,
        "receiver_universe_id": universe_id,
        "receiver_axis_id": receiver_axis_id,
        "source_crossfit_manifest": {
            "root_input_identity": root_identity,
            "receiver_universe": universe,
            "receiver_scoring_registry_id": registry.registry_id,
            "receiver_scoring_registry": registry.to_dict(),
            "fold_artifacts": [
                {
                    "fold_id": "fold-1",
                    "receiver_training_support": support_records,
                }
            ],
        },
    }
    _, support_table = _validate_source_receiver_universe(manifest)
    return manifest, {"application-1": application}, support_table


def _application(manifest: dict[str, object]) -> dict[str, object]:
    applications = cast(list[dict[str, object]], manifest["applications"])
    return applications[0]


def _source(manifest: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], manifest["source_crossfit_manifest"])


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def directional_lr_v7_manifest(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    from tests.integration.test_subject_crossfit import (
        _adata,
        _bundle,
        _config,
        _directional_spec,
        _prior,
    )

    artifacts = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_directional_spec(),
    )
    result = write_crossfit_result(
        artifacts,
        tmp_path_factory.mktemp("directional-lr-v7") / "result",
    )
    return result.manifest


def _rehash_crossfit_manifest(manifest: dict[str, object]) -> None:
    source = _source(manifest)
    manifest["source_crossfit_manifest_digest"] = canonical_digest(source)
    manifest["crossfit_result_id"] = stable_id(
        "crossfit_result",
        {key: value for key, value in manifest.items() if key != "crossfit_result_id"},
        schema_version="1",
    )


def _as_historical_v8_manifest(manifest: dict[str, object]) -> dict[str, object]:
    historical = copy.deepcopy(manifest)
    historical["schema_version"] = "8.0.0"
    for field_name in (
        "receiver_family_opportunity_universe_id",
        "family_axis_id",
        "receiver_family_opportunity_axis_id",
        "receiver_family_opportunity_universe",
    ):
        historical.pop(field_name)
        _source(historical).pop(field_name)
    _rehash_crossfit_manifest(historical)
    return historical


def _downgrade_directional_lr_universe_to_v1(
    manifest: dict[str, object],
) -> None:
    universe = cast(
        dict[str, object],
        _source(manifest)["directional_lr_hypothesis_universe"],
    )
    for field_name in (
        "molecular_lr_equivalence_universe_id",
        "molecular_lr_axis_id",
        "molecular_lr_equivalence_universe",
    ):
        universe.pop(field_name)
    mapping_policy = "resource_prior_unique_driver_match_v1"
    source_policy = "external_resource_target_prior_unique_driver_mapping_v1"
    universe["mapping_policy"] = mapping_policy
    universe["source_policy"] = source_policy
    universe["schema_version"] = "1.0.0"

    report = cast(dict[str, object], universe["mapping_report"])
    report["mapping_policy"] = mapping_policy
    report_payload = {
        "ambiguous_interactions": [
            {
                "candidate_driver_ids": entry["candidate_driver_ids"],
                "interaction_id": entry["interaction_id"],
            }
            for entry in cast(list[dict[str, object]], report["ambiguous_interactions"])
        ],
        "interaction_ids": report["interaction_ids"],
        "mapped_interaction_ids": report["mapped_interaction_ids"],
        "mapping_policy": mapping_policy,
        "resource_bundle_content_id": universe["resource_bundle_content_id"],
        "target_prior_content_id": universe["target_prior_content_id"],
        "unmapped_interaction_ids": report["unmapped_interaction_ids"],
    }
    report_id = stable_id(
        "receiver_family_lr_mapping_report",
        report_payload,
        schema_version="1",
    )
    report["report_id"] = report_id
    universe["mapping_report_id"] = report_id

    old_memberships = cast(list[dict[str, object]], universe["memberships"])
    triples = [
        [item["interaction_id"], item["driver_id"], item["family_id"]]
        for item in old_memberships
    ]
    membership_axis_id = stable_id(
        "receiver_family_lr_membership_axis",
        {
            "family_axis_id": universe["family_axis_id"],
            "mapped_memberships": triples,
            "mapping_policy": mapping_policy,
            "mapping_report_id": report_id,
            "resource_bundle_content_id": universe["resource_bundle_content_id"],
            "target_prior_content_id": universe["target_prior_content_id"],
        },
        schema_version="1",
    )
    memberships = [
        {
            "interaction_id": interaction_id,
            "driver_id": driver_id,
            "family_id": family_id,
            "membership_id": stable_id(
                "receiver_family_lr_membership",
                {
                    "driver_id": driver_id,
                    "family_id": family_id,
                    "interaction_id": interaction_id,
                    "membership_axis_id": membership_axis_id,
                },
                schema_version="1",
            ),
        }
        for interaction_id, driver_id, family_id in triples
    ]
    mode_axis_id = cast(str, universe["mode_axis_id"])
    hypothesis_axis_id = stable_id(
        "receiver_family_lr_hypothesis_axis",
        {
            "membership_axis_id": membership_axis_id,
            "mode_axis_id": mode_axis_id,
        },
        schema_version="1",
    )
    hypotheses = [
        {
            "interaction_id": membership["interaction_id"],
            "driver_id": membership["driver_id"],
            "family_id": membership["family_id"],
            "membership_id": membership["membership_id"],
            "mode": mode,
            "hypothesis_id": stable_id(
                "receiver_family_lr_hypothesis",
                {
                    "hypothesis_axis_id": hypothesis_axis_id,
                    "membership_id": membership["membership_id"],
                    "mode": mode,
                },
                schema_version="1",
            ),
        }
        for membership in memberships
        for mode in cast(list[str], universe["modes"])
    ]
    opportunity_axis_id = stable_id(
        "receiver_family_lr_opportunity_axis",
        {
            "hypothesis_axis_id": hypothesis_axis_id,
            "receiver_axis_id": universe["receiver_axis_id"],
        },
        schema_version="1",
    )
    opportunities = [
        {
            "receiver": receiver,
            "interaction_id": hypothesis["interaction_id"],
            "driver_id": hypothesis["driver_id"],
            "family_id": hypothesis["family_id"],
            "membership_id": hypothesis["membership_id"],
            "mode": hypothesis["mode"],
            "hypothesis_id": hypothesis["hypothesis_id"],
            "opportunity_id": stable_id(
                "receiver_family_lr_opportunity",
                {
                    "hypothesis_id": hypothesis["hypothesis_id"],
                    "opportunity_axis_id": opportunity_axis_id,
                    "receiver": receiver,
                },
                schema_version="1",
            ),
        }
        for receiver in cast(list[str], universe["receiver_ids"])
        for hypothesis in hypotheses
    ]
    universe.update(
        {
            "membership_axis_id": membership_axis_id,
            "memberships": memberships,
            "membership_count": len(memberships),
            "hypothesis_axis_id": hypothesis_axis_id,
            "hypotheses": hypotheses,
            "hypothesis_count": len(hypotheses),
            "opportunity_axis_id": opportunity_axis_id,
            "opportunities": opportunities,
            "opportunity_count": len(opportunities),
        }
    )
    child_payload = {
        field_name: universe[field_name]
        for field_name in (
            "family_axis_id",
            "feature_axis_id",
            "hypothesis_axis_id",
            "mapping_policy",
            "mapping_report_id",
            "membership_axis_id",
            "mode_axis_id",
            "modes",
            "opportunity_axis_id",
            "receiver_axis_id",
            "receiver_family_universe_id",
            "receiver_universe_id",
            "root_input_digest",
            "root_input_identity_id",
            "resource_bundle_content_id",
            "resource_id",
            "resource_manifest_digest",
            "resource_version",
            "schema_version",
            "source_policy",
            "target_prior_content_id",
            "target_prior_manifest_digest",
            "target_prior_resource_id",
            "target_prior_version",
        )
    }
    universe["universe_id"] = stable_id(
        "frozen_receiver_family_lr_hypothesis_universe",
        child_payload,
        schema_version="1",
    )
    _rehash_crossfit_manifest(manifest)


def _set_application_projection(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
    field_name: str,
    value: object,
) -> None:
    _application(manifest)[field_name] = value
    for table in tables.values():
        table.loc[:, field_name] = value


def test_crossfit_result_contract_exposes_required_grain_without_inference() -> None:
    required_component_grain = {
        "sample_id",
        "subject_id",
        "contrast",
        "receiver",
        "interaction_id",
        "component",
        "status",
        "reason_code",
        "family_common_functional_id",
        "family_common_application_id",
        "family_common_binding_id",
    }
    forbidden = {
        "p_value",
        "q_value",
        "comm_probability",
        "posterior",
        "confidence_interval",
        "standard_error",
    }

    assert required_component_grain.issubset(CROSSFIT_COMPONENT_COLUMNS)
    assert set(CROSSFIT_DIFFERENTIAL_COLUMNS).isdisjoint(forbidden)
    assert set(CROSSFIT_COMPONENT_COLUMNS).isdisjoint(forbidden)
    assert "differential_effect" in CROSSFIT_DIFFERENTIAL_COLUMNS
    assert "bounded_incremental_gain" not in CROSSFIT_DIFFERENTIAL_COLUMNS
    assert CROSSFIT_TABLE_NAMES == (
        "family_common_components",
        "descriptive_differential",
        "contrast_common_lr_scores",
        "contrast_common_sender_lr_scores",
        "directional_channel_registry",
        "receiver_training_support",
        "semantic_availability_scores",
        "semantic_receiver_program_scores",
        "semantic_integrated_lr_scores",
        "semantic_differential_effects",
    )
    assert "global_lr_score" in CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
    assert "global_sender_lr_score" in CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
    assert "calibrated_family_gain_percentile" in CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
    assert "training_receiver_scale_factor" not in CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
    assert "training_receiver_scale_factor" not in (
        CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
    )
    assert set(CROSSFIT_CONTRAST_COMMON_LR_COLUMNS).isdisjoint(forbidden)
    assert set(CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS).isdisjoint(forbidden)
    assert set(CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS).isdisjoint(forbidden)


def test_versioned_common_tables_remain_read_only_compatible() -> None:
    legacy_common_tables = (
        "family_common_components",
        "descriptive_differential",
        "contrast_common_lr_scores",
        "contrast_common_sender_lr_scores",
    )
    pre_receiver_persistence_tables = (
        *legacy_common_tables,
        "directional_channel_registry",
    )
    pre_semantic_persistence_tables = (
        *pre_receiver_persistence_tables,
        "receiver_training_support",
    )
    assert _bundle_table_names(CROSSFIT_RESULT_SCHEMA_VERSION) == CROSSFIT_TABLE_NAMES
    assert _bundle_table_names("7.0.0") == pre_semantic_persistence_tables
    assert _bundle_table_names("6.0.0") == pre_receiver_persistence_tables
    assert _bundle_table_names("5.0.0") == pre_receiver_persistence_tables
    assert _bundle_table_names("4.0.0") == legacy_common_tables
    assert _bundle_table_names("3.0.0") == legacy_common_tables
    assert _bundle_table_names("2.0.0") == CROSSFIT_RECEIVER_TABLE_NAMES
    assert _table_schema_version("3.0.0", CROSSFIT_CONTRAST_COMMON_LR_TABLE) == "1.0.0"
    assert (
        _table_schema_version(
            CROSSFIT_RESULT_SCHEMA_VERSION, CROSSFIT_CONTRAST_COMMON_LR_TABLE
        )
        == "2.0.0"
    )
    assert (
        _table_schema_version(
            CROSSFIT_RESULT_SCHEMA_VERSION,
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        )
        == "3.0.0"
    )
    for schema_version in ("4.0.0", "5.0.0"):
        assert (
            _table_schema_version(
                schema_version,
                CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            )
            == "2.0.0"
        )
        assert "assignment_weight" not in _table_columns(
            schema_version,
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        )
        assert "normalized_entropy" not in _table_columns(
            schema_version,
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        )
    assert (
        _table_schema_version(
            CROSSFIT_RESULT_SCHEMA_VERSION,
            CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
        )
        == "1.0.0"
    )
    assert (
        _table_schema_version(
            CROSSFIT_RESULT_SCHEMA_VERSION,
            CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE,
        )
        == "1.0.0"
    )
    with pytest.raises(KeyError):
        _table_schema_version("6.0.0", CROSSFIT_RECEIVER_TRAINING_SUPPORT_TABLE)

    current_lr = _contrast_common_lr_table().iloc[0].to_dict()
    current_sender = _contrast_common_sender_table().iloc[0].to_dict()
    current_lr["training_receiver_scale_factor"] = 0.8
    current_sender["training_receiver_scale_factor"] = 0.8
    current_sender["global_sender_lr_score"] = float(
        current_sender["global_lr_score"]
    ) * float(current_sender["raw_sender_evidence"])
    legacy_lr = pd.DataFrame(
        [current_lr],
        columns=_table_columns("3.0.0", CROSSFIT_CONTRAST_COMMON_LR_TABLE),
    )
    legacy_sender = pd.DataFrame(
        [current_sender],
        columns=_table_columns("3.0.0", CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE),
    )

    assert (
        len(
            _validate_table(
                CROSSFIT_CONTRAST_COMMON_LR_TABLE,
                legacy_lr,
                schema_version="3.0.0",
            )
        )
        == 1
    )
    assert (
        len(
            _validate_table(
                CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
                legacy_sender,
                schema_version="3.0.0",
            )
        )
        == 1
    )

    legacy_sender = _contrast_common_sender_table().copy(deep=True)
    legacy_sender.loc[:, "global_sender_lr_score"] = (
        legacy_sender["global_lr_score"] * legacy_sender["raw_sender_evidence"]
    )
    for schema_version in ("4.0.0", "5.0.0"):
        versioned_sender = legacy_sender.loc[
            :,
            list(
                _table_columns(
                    schema_version,
                    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
                )
            ),
        ].copy(deep=True)
        assert (
            len(
                _validate_table(
                    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
                    versioned_sender,
                    schema_version=schema_version,
                )
            )
            == 2
        )

        poisoned_raw = versioned_sender.copy(deep=True)
        poisoned_raw.loc[0, "raw_sender_evidence"] = 0.3
        with pytest.raises(ValueError, match="ligand-by-prevalence"):
            _validate_table(
                CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
                poisoned_raw,
                schema_version=schema_version,
            )

        poisoned_score = versioned_sender.copy(deep=True)
        poisoned_score.loc[0, "global_sender_lr_score"] = 0.9
        with pytest.raises(ValueError, match="multiplicative contract"):
            _validate_table(
                CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
                poisoned_score,
                schema_version=schema_version,
            )


def test_v7_receiver_support_grid_is_source_bound_and_registry_bound() -> None:
    manifest, applications, support = _v7_receiver_registry_case()

    receiver_ids, expected = _validate_source_receiver_universe(manifest)
    _validate_source_scoring_registry(manifest, applications)
    _validate_receiver_training_support_links(manifest, support)

    assert receiver_ids == ("PlannedOnly", "Receiver")
    assert tuple(expected.columns) == CROSSFIT_RECEIVER_TRAINING_SUPPORT_COLUMNS
    assert expected[["fold_id", "receiver"]].to_records(index=False).tolist() == [
        ("fold-1", "PlannedOnly"),
        ("fold-1", "Receiver"),
    ]
    assert expected["training_cell_type_ids"].tolist() == [
        canonical_json(["Receiver"]),
        canonical_json(["Receiver"]),
    ]


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field_name", "poisoned"),
    (
        ("receiver_training_support_id", "support-poisoned"),
        ("receiver_training_support_status", "observed"),
        ("receiver_training_support_reason_code", "reason-poisoned"),
        ("training_cell_type_ids", '["Receiver","Receiver"]'),
    ),
)
def test_v7_receiver_support_table_rejects_row_tamper(
    field_name: str,
    poisoned: object,
) -> None:
    _, _, support = _v7_receiver_registry_case()
    support.loc[0, field_name] = poisoned

    with pytest.raises(ValueError, match=r"receiver training|support"):
        _validate_receiver_training_support_table(support)


@pytest.mark.parametrize("mutation", ("missing", "extra"))  # type: ignore[untyped-decorator]
def test_v7_receiver_support_link_rejects_inexact_grid(mutation: str) -> None:
    manifest, _, support = _v7_receiver_registry_case()
    if mutation == "missing":
        poisoned = support.iloc[1:].reset_index(drop=True)
    else:
        extra = support.iloc[[0]].copy(deep=True)
        extra.loc[:, "fold_id"] = "fold-extra"
        payload = {
            "outer_fold_id": "fold-extra",
            "reason_code": "receiver_absent_in_outer_training",
            "receiver_id": "PlannedOnly",
            "receiver_universe_id": str(extra.iloc[0]["receiver_universe_id"]),
            "status": "not_estimable",
            "training_cell_type_ids": ["Receiver"],
        }
        extra.loc[:, "receiver_training_support_id"] = stable_id(
            "receiver_training_support", payload, schema_version="1"
        )
        poisoned = pd.concat([support, extra], ignore_index=True)

    with pytest.raises(ResultValidationError, match=r"support|Support"):
        _validate_receiver_training_support_links(manifest, poisoned)


def test_v9_receiver_family_universe_is_mandatory_and_strictly_replayed(
    directional_lr_v7_manifest: dict[str, object],
) -> None:
    manifest = copy.deepcopy(directional_lr_v7_manifest)

    contract = _validate_source_receiver_family_opportunity_universe(manifest)
    validated = _validate_manifest(cast(dict[str, Any], manifest))

    assert validated["schema_version"] == "9.0.0"
    assert (
        validated["receiver_family_opportunity_universe"]
        == _source(validated)["receiver_family_opportunity_universe"]
    )
    assert (
        validated["receiver_family_opportunity_universe_id"] == contract["universe_id"]
    )
    assert validated["family_axis_id"] == contract["family_axis_id"]
    assert (
        validated["receiver_family_opportunity_axis_id"]
        == contract["opportunity_axis_id"]
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "target",
    (
        "receiver",
        "root",
        "prior",
        "feature",
        "family",
        "opportunity",
        "universe_id",
        "source_family_axis",
        "result_opportunity_axis",
    ),
)
def test_v9_receiver_family_universe_rejects_rehashed_semantic_tamper(
    directional_lr_v7_manifest: dict[str, object],
    target: str,
) -> None:
    manifest = copy.deepcopy(directional_lr_v7_manifest)
    source = _source(manifest)
    universe = cast(dict[str, object], source["receiver_family_opportunity_universe"])
    if target == "receiver":
        receivers = cast(list[str], universe["receiver_ids"])
        universe["receiver_ids"] = receivers[:-1]
    elif target == "root":
        universe["root_input_digest"] = "poisoned-root-digest"
    elif target == "prior":
        universe["prior_manifest_digest"] = "poisoned-prior-digest"
    elif target == "feature":
        universe["feature_ids"] = list(
            reversed(cast(list[str], universe["feature_ids"]))
        )
    elif target == "family":
        definitions = cast(list[dict[str, object]], universe["family_definitions"])
        definitions[0]["family_id"] = "poisoned-family"
    elif target == "opportunity":
        opportunities = cast(list[dict[str, object]], universe["opportunities"])
        opportunities[0]["opportunity_id"] = "poisoned-opportunity"
    elif target == "universe_id":
        universe["universe_id"] = "poisoned-universe"
        source["receiver_family_opportunity_universe_id"] = "poisoned-universe"
        manifest["receiver_family_opportunity_universe_id"] = "poisoned-universe"
    elif target == "source_family_axis":
        source["family_axis_id"] = "poisoned-family-axis"
    else:
        manifest["receiver_family_opportunity_axis_id"] = "poisoned-opportunity-axis"
    manifest["receiver_family_opportunity_universe"] = copy.deepcopy(universe)
    _rehash_crossfit_manifest(manifest)

    assert manifest["source_crossfit_manifest_digest"] == canonical_digest(source)
    with pytest.raises(
        ValueError, match=r"receiver-family|receiver|family|opportunity"
    ):
        _validate_manifest(cast(dict[str, Any], manifest))


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "missing",
    (
        "source_universe",
        "source_universe_id",
        "result_universe",
        "result_universe_id",
    ),
)
def test_v9_receiver_family_universe_rejects_missing_mandatory_fields(
    directional_lr_v7_manifest: dict[str, object],
    missing: str,
) -> None:
    manifest = copy.deepcopy(directional_lr_v7_manifest)
    if missing == "source_universe":
        _source(manifest).pop("receiver_family_opportunity_universe")
    elif missing == "source_universe_id":
        _source(manifest).pop("receiver_family_opportunity_universe_id")
    elif missing == "result_universe":
        manifest.pop("receiver_family_opportunity_universe")
    else:
        manifest.pop("receiver_family_opportunity_universe_id")
    _rehash_crossfit_manifest(manifest)

    with pytest.raises(ValueError, match=r"manifest fields|receiver-family"):
        _validate_manifest(cast(dict[str, Any], manifest))


def test_v8_historical_manifest_remains_read_only_compatible_without_root_family(
    directional_lr_v7_manifest: dict[str, object],
) -> None:
    historical = _as_historical_v8_manifest(directional_lr_v7_manifest)

    validated = _validate_manifest(cast(dict[str, Any], historical))

    assert validated["schema_version"] == "8.0.0"
    assert "receiver_family_opportunity_universe" not in validated
    assert "receiver_family_opportunity_universe" not in _source(validated)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "target",
    ("axis", "universe", "root", "fold_support"),
)
def test_v7_source_receiver_lineage_rejects_tamper(target: str) -> None:
    manifest, _, _ = _v7_receiver_registry_case()
    source = _source(manifest)
    universe = cast(dict[str, object], source["receiver_universe"])
    if target == "axis":
        universe["receiver_axis_id"] = "axis-poisoned"
    elif target == "universe":
        universe["universe_id"] = "universe-poisoned"
    elif target == "root":
        root = cast(dict[str, object], source["root_input_identity"])
        root["identity_id"] = "root-poisoned"
    else:
        folds = cast(list[dict[str, object]], source["fold_artifacts"])
        records = cast(list[dict[str, object]], folds[0]["receiver_training_support"])
        records[0]["reason_code"] = "reason-poisoned"

    with pytest.raises(ValueError, match=r"receiver|root|support"):
        _validate_source_receiver_universe(manifest)


def test_v7_directional_lr_universe_accepts_historical_absence(
    directional_lr_v7_manifest: dict[str, object],
) -> None:
    historical = copy.deepcopy(directional_lr_v7_manifest)
    _source(historical).pop("directional_lr_hypothesis_universe")
    _rehash_crossfit_manifest(historical)

    validated = _validate_manifest(cast(dict[str, Any], historical))

    assert "directional_lr_hypothesis_universe" not in _source(validated)


def test_v7_directional_lr_universe_accepts_complete_producer_manifest(
    directional_lr_v7_manifest: dict[str, object],
) -> None:
    manifest = copy.deepcopy(directional_lr_v7_manifest)

    _validate_source_directional_lr_hypothesis_universe(manifest)
    _validate_manifest(cast(dict[str, Any], manifest))


def test_v7_directional_lr_universe_accepts_historical_v1_identity_chain(
    directional_lr_v7_manifest: dict[str, object],
) -> None:
    historical = copy.deepcopy(directional_lr_v7_manifest)
    _downgrade_directional_lr_universe_to_v1(historical)

    _validate_source_directional_lr_hypothesis_universe(historical)
    validated = _validate_manifest(cast(dict[str, Any], historical))

    universe = cast(
        dict[str, object],
        _source(validated)["directional_lr_hypothesis_universe"],
    )
    assert universe["schema_version"] == "1.0.0"
    assert "molecular_lr_equivalence_universe" not in universe
    assert all(
        "molecular_lr_equivalence_id" not in row
        for table_name in ("memberships", "hypotheses", "opportunities")
        for row in cast(list[dict[str, object]], universe[table_name])
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "target",
    (
        "molecular_extra_field",
        "source_binding",
        "source_mapping_report",
        "equivalence_class",
        "mechanistic_variant",
        "mapping_record",
        "molecular_axis",
        "molecular_universe_id",
        "outer_molecular_universe_id",
        "membership_molecular_id",
        "hypothesis_molecular_id",
        "opportunity_molecular_id",
    ),
)
def test_v7_directional_lr_universe_v2_rejects_molecular_semantic_tamper(
    directional_lr_v7_manifest: dict[str, object],
    target: str,
) -> None:
    manifest = copy.deepcopy(directional_lr_v7_manifest)
    source = _source(manifest)
    universe = cast(dict[str, object], source["directional_lr_hypothesis_universe"])
    molecular = cast(dict[str, object], universe["molecular_lr_equivalence_universe"])
    if target == "molecular_extra_field":
        molecular["poisoned_extra"] = True
    elif target == "source_binding":
        bindings = cast(list[dict[str, object]], molecular["source_bindings"])
        bindings[0]["source_binding_id"] = "poisoned-source-binding"
    elif target == "source_mapping_report":
        bindings = cast(list[dict[str, object]], molecular["source_bindings"])
        report = cast(dict[str, object], bindings[0]["mapping_report"])
        report["loaded_rows"] = cast(int, report["loaded_rows"]) + 1
    elif target == "equivalence_class":
        classes = cast(list[dict[str, object]], molecular["equivalence_classes"])
        classes[0]["molecular_lr_equivalence_id"] = "poisoned-core"
    elif target == "mechanistic_variant":
        variants = cast(list[dict[str, object]], molecular["mechanistic_variants"])
        variants[0]["mechanistic_variant_id"] = "poisoned-variant"
    elif target == "mapping_record":
        records = cast(list[dict[str, object]], molecular["mapping_records"])
        records[0]["mapping_record_id"] = "poisoned-mapping-record"
    elif target == "molecular_axis":
        molecular["molecular_lr_axis_id"] = "poisoned-molecular-axis"
    elif target == "molecular_universe_id":
        molecular["universe_id"] = "poisoned-molecular-universe"
    elif target == "outer_molecular_universe_id":
        universe["molecular_lr_equivalence_universe_id"] = "poisoned-parent"
    elif target == "membership_molecular_id":
        memberships = cast(list[dict[str, object]], universe["memberships"])
        memberships[0]["molecular_lr_equivalence_id"] = "poisoned-membership-core"
    elif target == "hypothesis_molecular_id":
        hypotheses = cast(list[dict[str, object]], universe["hypotheses"])
        hypotheses[0]["molecular_lr_equivalence_id"] = "poisoned-hypothesis-core"
    else:
        opportunities = cast(list[dict[str, object]], universe["opportunities"])
        opportunities[0]["molecular_lr_equivalence_id"] = "poisoned-opportunity-core"
    _rehash_crossfit_manifest(manifest)

    with pytest.raises(ValueError, match=r"molecular LR|directional LR"):
        _validate_manifest(cast(dict[str, Any], manifest))


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "target",
    (
        "null_universe",
        "source_spec",
        "extra_field",
        "parent_root",
        "parent_family",
        "parent_opportunity",
        "mapping_partition",
        "membership",
        "hypothesis",
        "opportunity",
        "child_universe_id",
    ),
)
def test_v7_directional_lr_universe_rejects_semantic_tamper_after_rehash(
    directional_lr_v7_manifest: dict[str, object],
    target: str,
) -> None:
    manifest = copy.deepcopy(directional_lr_v7_manifest)
    source = _source(manifest)
    if target == "null_universe":
        source["directional_lr_hypothesis_universe"] = None
    else:
        universe = cast(
            dict[str, object],
            source["directional_lr_hypothesis_universe"],
        )
        parent = cast(dict[str, object], universe["receiver_family_universe"])
        if target == "source_spec":
            spec = cast(dict[str, object], source["spec"])
            spec.pop("directional_pairs")
        elif target == "extra_field":
            universe["poisoned_extra"] = True
        elif target == "parent_root":
            parent["root_input_digest"] = "poisoned-root-digest"
        elif target == "parent_family":
            definitions = cast(list[dict[str, object]], parent["family_definitions"])
            definitions[0]["family_id"] = "poisoned-family"
        elif target == "parent_opportunity":
            opportunities = cast(list[dict[str, object]], parent["opportunities"])
            opportunities[0]["opportunity_id"] = "poisoned-parent-opportunity"
        elif target == "mapping_partition":
            report = cast(dict[str, object], universe["mapping_report"])
            mapped = cast(list[str], report["mapped_interaction_ids"])
            unmapped = cast(list[str], report["unmapped_interaction_ids"])
            unmapped.append(mapped[0])
            report["unmapped_count"] = len(unmapped)
        elif target == "membership":
            memberships = cast(list[dict[str, object]], universe["memberships"])
            memberships[0]["membership_id"] = "poisoned-membership"
        elif target == "hypothesis":
            hypotheses = cast(list[dict[str, object]], universe["hypotheses"])
            hypotheses[0]["hypothesis_id"] = "poisoned-hypothesis"
        elif target == "opportunity":
            opportunities = cast(list[dict[str, object]], universe["opportunities"])
            opportunities[0]["opportunity_id"] = "poisoned-opportunity"
        else:
            universe["universe_id"] = "poisoned-child-universe"
    _rehash_crossfit_manifest(manifest)

    assert manifest["source_crossfit_manifest_digest"] == canonical_digest(source)
    assert manifest["crossfit_result_id"] == stable_id(
        "crossfit_result",
        {key: value for key, value in manifest.items() if key != "crossfit_result_id"},
        schema_version="1",
    )
    with pytest.raises(ValueError, match=r"directional LR|receiver-family"):
        _validate_manifest(cast(dict[str, Any], manifest))


def test_result_schema_selects_exact_scoring_registry_authority() -> None:
    legacy_manifest, _ = _registry_link_case()
    legacy_applications = {
        "application-1": cast(list[dict[str, object]], legacy_manifest["applications"])[
            0
        ]
    }
    legacy_manifest["schema_version"] = "6.0.0"
    _validate_source_scoring_registry(
        legacy_manifest,
        legacy_applications,
    )

    current_manifest, current_applications, _ = _v7_receiver_registry_case()
    _validate_source_scoring_registry(
        current_manifest,
        current_applications,
    )

    poisoned_v6 = copy.deepcopy(current_manifest)
    poisoned_v6["schema_version"] = "6.0.0"
    with pytest.raises(ResultValidationError, match="version is incompatible"):
        _validate_source_scoring_registry(
            poisoned_v6,
            current_applications,
        )

    old_registry = _scoring_registry()
    assert old_registry.registry_id is not None
    poisoned_v7 = copy.deepcopy(current_manifest)
    poisoned_source = _source(poisoned_v7)
    poisoned_source["receiver_scoring_registry_id"] = old_registry.registry_id
    poisoned_source["receiver_scoring_registry"] = old_registry.to_dict()
    with pytest.raises(ResultValidationError, match="version is incompatible"):
        _validate_source_scoring_registry(
            poisoned_v7,
            current_applications,
        )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "status", ("observed", "not_estimable")
)
def test_directional_registry_has_exactly_two_nonadditive_source_bound_channels(
    status: str,
) -> None:
    manifest, table = _directional_registry_case(status=status)

    validated = _validate_directional_channel_registry_table(table)
    _validate_directional_registry_links(manifest, validated)

    assert validated["channel_role"].tolist() == ["forward", "reverse"]
    assert set(validated["channel"]) == {
        "increased_activation_compatible",
        "reduced_activation_compatible",
    }
    assert not validated["active_inhibition_allowed"].any()
    assert not validated["supports_active_inhibition_claim"].any()
    assert not validated["paired_score_comparison_allowed"].any()
    assert not validated["formal_inference_allowed"].any()
    if status == "observed":
        assert validated["response_pair_id"].notna().all()
        assert validated["response_channel_id"].notna().all()
    else:
        assert validated["reason_code"].notna().all()
        assert validated["response_pair_id"].isna().all()
        assert validated["response_channel_id"].isna().all()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field_name", "poisoned"),
    (
        ("n_directional_receiver_pair_opportunities", 3),
        ("n_directional_receiver_pair_absent_training", 0),
        ("directional_receiver_universe_id", "universe-poisoned"),
        ("directional_receiver_axis_id", "axis-poisoned"),
        ("directional_registry_complete", True),
        ("directional_supported_binding_registry_complete", False),
    ),
)
def test_v7_directional_registry_rejects_receiver_opportunity_tamper(
    field_name: str,
    poisoned: object,
) -> None:
    manifest, table = _directional_registry_case()
    _source(manifest)[field_name] = poisoned

    with pytest.raises(ResultValidationError, match="violates its contract"):
        _validate_directional_registry_links(manifest, table)


def test_v6_directional_registry_keeps_legacy_completion_semantics() -> None:
    manifest, table = _directional_registry_case()
    manifest["schema_version"] = "6.0.0"
    source = _source(manifest)
    for field_name in (
        "n_directional_receiver_pair_opportunities",
        "n_directional_receiver_pair_absent_training",
        "directional_receiver_universe_id",
        "directional_receiver_axis_id",
        "directional_supported_binding_registry_complete",
    ):
        source.pop(field_name)
    source["directional_registry_complete"] = True

    _validate_directional_registry_links(manifest, table)

    source["directional_supported_binding_registry_complete"] = True
    with pytest.raises(ResultValidationError, match="violates its contract"):
        _validate_directional_registry_links(manifest, table)


def test_directional_registry_rejects_missing_extra_and_swapped_lineage() -> None:
    manifest, table = _directional_registry_case()
    missing = table.iloc[:1].copy(deep=True)
    extra = pd.concat([table, table.iloc[[0]]], ignore_index=True)

    for poisoned in (missing, extra):
        with pytest.raises(
            ResultValidationError,
            match="do not exactly match source bindings",
        ):
            _validate_directional_registry_links(manifest, poisoned)

    jointly_removed_manifest = copy.deepcopy(manifest)
    source = cast(
        dict[str, object], jointly_removed_manifest["source_crossfit_manifest"]
    )
    folds = cast(list[dict[str, object]], source["fold_artifacts"])
    folds[0]["directional_response_bindings"] = []
    source["n_directional_response_bindings"] = 0
    source["directional_binding_status_counts"] = {}
    source["directional_diagnostic_complete"] = False
    with pytest.raises(ResultValidationError, match="violates its contract"):
        _validate_directional_registry_links(
            jointly_removed_manifest,
            table.iloc[0:0].copy(deep=True),
        )

    for field_name in (
        "contrast_id",
        "contrast",
        "response_id",
        "training_artifact_id",
        "application_id",
        "response_channel_id",
    ):
        swapped = table.copy(deep=True)
        swapped.loc[:, field_name] = swapped[field_name].iloc[::-1].to_numpy()
        _validate_directional_channel_registry_table(swapped)
        with pytest.raises(
            ResultValidationError,
            match="do not exactly match source bindings",
        ):
            _validate_directional_registry_links(manifest, swapped)


def test_directional_registry_rejects_binding_scope_claim_and_source_tamper() -> None:
    manifest, table = _directional_registry_case()
    for field_name, value in (
        ("pair_spec_id", "pair-poisoned"),
        ("binding_id", "binding-poisoned"),
        ("fold_id", "fold-poisoned"),
        ("receiver", "ReceiverPoisoned"),
    ):
        poisoned = table.copy(deep=True)
        poisoned.loc[0, field_name] = value
        with pytest.raises(ValueError):
            _validate_directional_channel_registry_table(poisoned)

    active_claim = table.copy(deep=True)
    active_claim.loc[:, "active_inhibition_allowed"] = True
    with pytest.raises(ValueError, match="claim semantics"):
        _validate_directional_channel_registry_table(active_claim)

    source = cast(dict[str, object], manifest["source_crossfit_manifest"])
    folds = cast(list[dict[str, object]], source["fold_artifacts"])
    bindings = cast(list[dict[str, object]], folds[0]["directional_response_bindings"])
    bindings[0]["forward_application_id"] = "application-poisoned"
    with pytest.raises(
        ResultValidationError,
        match="violates its contract",
    ):
        _validate_directional_registry_links(manifest, table)


def test_v3_common_manifest_retains_retired_scale_contract_for_reading() -> None:
    manifest = _contrast_common_manifest()
    manifest["schema_version"] = "3.0.0"
    collection = cast(list[dict[str, object]], manifest["contrast_common_collections"])[
        0
    ]
    application = cast(list[dict[str, object]], collection["fold_applications"])[0]
    _downgrade_common_record_to_v3(application)
    payload = {
        key: value
        for key, value in collection.items()
        if key != "contrast_common_collection_id"
    }
    collection["contrast_common_collection_id"] = stable_id(
        "contrast_common_oof_score_collection", payload, schema_version="1"
    )

    _validate_contrast_common_collections_manifest(manifest)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "schema_version", ("4.0.0", "5.0.0")
)
def test_gain_calibrated_common_manifest_remains_read_only_compatible(
    schema_version: str,
) -> None:
    manifest = _contrast_common_manifest()
    manifest["schema_version"] = schema_version

    _validate_contrast_common_collections_manifest(manifest)


def test_v3_source_and_table_lineage_remain_read_only_compatible() -> None:
    manifest, applications = _contrast_common_source_case()
    collection, application = applications["global-application-1"]
    source = cast(dict[str, object], manifest["source_crossfit_manifest"])
    folds = cast(list[dict[str, object]], source["fold_artifacts"])
    source_records = cast(
        list[dict[str, object]],
        folds[0]["cross_receiver_common_scoring_artifacts"],
    )
    source_record = source_records[0]
    _downgrade_common_record_to_v3(application)
    _downgrade_common_record_to_v3(source_record)

    _, current_tables = _contrast_common_lineage_case()
    legacy_tables: dict[str, pd.DataFrame] = {}
    for table_name in (
        CROSSFIT_CONTRAST_COMMON_LR_TABLE,
        CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
    ):
        frame = current_tables[table_name].copy(deep=True)
        frame.loc[:, "training_receiver_scale_factor"] = 0.8
        legacy_tables[table_name] = frame.loc[
            :, list(_table_columns("3.0.0", table_name))
        ].copy(deep=True)
    lr_source_columns = tuple(
        _table_columns("3.0.0", CROSSFIT_CONTRAST_COMMON_LR_TABLE)[8:-5]
    )
    sender_source_columns = tuple(
        _table_columns("3.0.0", CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE)[8:-5]
    )
    application["lr_scores_digest"] = _global_source_table_digest(
        "cross_receiver_common_lr_scores",
        legacy_tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE].loc[
            :, list(lr_source_columns)
        ],
        lr_source_columns,
    )
    application["sender_scores_digest"] = _global_source_table_digest(
        "cross_receiver_common_sender_scores",
        legacy_tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE].loc[
            :, list(sender_source_columns)
        ],
        sender_source_columns,
    )
    source_record["lr_scores_digest"] = application["lr_scores_digest"]
    source_record["sender_scores_digest"] = application["sender_scores_digest"]
    collection_payload = {
        key: value
        for key, value in collection.items()
        if key != "contrast_common_collection_id"
    }
    collection_id = stable_id(
        "contrast_common_oof_score_collection",
        collection_payload,
        schema_version="1",
    )
    collection["contrast_common_collection_id"] = collection_id
    for table in legacy_tables.values():
        table.loc[:, "contrast_common_collection_id"] = collection_id
    manifest["schema_version"] = "3.0.0"
    manifest["contrast_common_collections"] = [collection]

    _validate_contrast_common_registry_links(manifest, legacy_tables)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "schema_version", ("4.0.0", "5.0.0")
)
def test_v4_v5_sender_lineage_retains_raw_evidence_formula(
    schema_version: str,
) -> None:
    manifest, applications = _contrast_common_source_case()
    collection, application = applications["global-application-1"]
    source = cast(dict[str, object], manifest["source_crossfit_manifest"])
    folds = cast(list[dict[str, object]], source["fold_artifacts"])
    source_records = cast(
        list[dict[str, object]],
        folds[0]["cross_receiver_common_scoring_artifacts"],
    )
    source_record = source_records[0]

    lr = _contrast_common_lr_table()
    sender = _contrast_common_sender_table()
    sender.loc[:, "global_sender_lr_score"] = (
        sender["global_lr_score"] * sender["raw_sender_evidence"]
    )
    sender = sender.loc[
        :,
        list(
            _table_columns(
                schema_version,
                CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            )
        ),
    ].copy(deep=True)
    sender_source_columns = tuple(
        _table_columns(
            schema_version,
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        )[8:-5]
    )
    sender_digest = _global_source_table_digest(
        "cross_receiver_common_sender_scores",
        sender.loc[:, list(sender_source_columns)],
        sender_source_columns,
    )
    application["functional_schema_version"] = "3.0.0"
    application["sender_policy"] = (
        "absolute_sender_evidence_without_candidate_normalization_v1"
    )
    application["sender_scores_digest"] = sender_digest
    source_record["functional_schema_version"] = application[
        "functional_schema_version"
    ]
    source_record["sender_policy"] = application["sender_policy"]
    source_record["sender_scores_digest"] = sender_digest

    collection_payload = {
        key: value
        for key, value in collection.items()
        if key != "contrast_common_collection_id"
    }
    collection_id = stable_id(
        "contrast_common_oof_score_collection",
        collection_payload,
        schema_version="1",
    )
    collection["contrast_common_collection_id"] = collection_id
    lr.loc[:, "contrast_common_collection_id"] = collection_id
    sender.loc[:, "contrast_common_collection_id"] = collection_id
    manifest["schema_version"] = schema_version
    manifest["contrast_common_collections"] = [collection]
    tables = {
        CROSSFIT_CONTRAST_COMMON_LR_TABLE: _validate_table(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            lr,
            schema_version=schema_version,
        ),
        CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: _validate_table(
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            sender,
            schema_version=schema_version,
        ),
    }

    _validate_contrast_common_registry_links(manifest, tables)


def test_contrast_common_tables_enforce_score_and_source_row_semantics() -> None:
    lr = _contrast_common_lr_table()
    sender = _contrast_common_sender_table()

    assert len(_validate_contrast_common_lr_table(lr)) == 1
    assert len(_validate_contrast_common_sender_lr_table(sender)) == 2

    poisoned_score = sender.copy(deep=True)
    poisoned_score.loc[0, "global_sender_lr_score"] = 0.9
    with pytest.raises(ValueError, match="assignment precedence"):
        _validate_contrast_common_sender_lr_table(poisoned_score)

    poisoned_raw = sender.copy(deep=True)
    poisoned_raw.loc[0, "raw_sender_evidence"] = 0.3
    with pytest.raises(ValueError, match="ligand-by-prevalence"):
        _validate_contrast_common_sender_lr_table(poisoned_raw)

    poisoned_row_id = lr.copy(deep=True)
    poisoned_row_id.loc[0, "source_row_id"] = "row-poisoned"
    with pytest.raises(ValueError, match="source-row identity"):
        _validate_contrast_common_lr_table(poisoned_row_id)


def test_identifier_validation_handles_vectorized_and_object_columns() -> None:
    valid = pd.DataFrame(
        {
            "arrow": pd.Series(["alpha", "beta"], dtype="string[pyarrow]"),
            "category": pd.Series(["one", "two"], dtype="category"),
        }
    )
    _require_identifiers(valid, ("arrow", "category"), table_name="test")

    empty_arrow = valid.copy(deep=True)
    empty_arrow.loc[1, "arrow"] = ""
    with pytest.raises(ValueError, match="requires non-empty identifiers"):
        _require_identifiers(empty_arrow, ("arrow",), table_name="test")

    missing_category = valid.copy(deep=True)
    missing_category.loc[1, "category"] = pd.NA
    with pytest.raises(ValueError, match="requires non-empty identifiers"):
        _require_identifiers(missing_category, ("category",), table_name="test")

    class EmptyIdentifier:
        def __str__(self) -> str:
            return ""

    object_values = pd.DataFrame({"object": ["valid", EmptyIdentifier()]})
    with pytest.raises(ValueError, match="requires non-empty identifiers"):
        _require_identifiers(object_values, ("object",), table_name="test")


@pytest.mark.parametrize(
    "receiver",
    [
        "Cycling cells",
        'T \\"cell',
        "T cell beta",
        "T\tcell",
        "",
    ],
)
def test_source_row_id_fast_path_matches_canonical_stable_id(receiver: str) -> None:
    components = {
        "application_id": "application_123",
        "context_id": "context_123",
        "driver_id": "FGF2",
        "family_id": "family_123",
        "interaction_id": "interaction_123",
        "mode": "state",
        "receiver": receiver,
        "sample_id": "sample_123",
        "sender": None,
        "source_table": CROSSFIT_CONTRAST_COMMON_LR_TABLE,
        "subject_id": "subject_123",
    }

    observed = _source_row_id(
        source_table=cast(str, components["source_table"]),
        application_id=cast(str, components["application_id"]),
        sample_id=cast(str, components["sample_id"]),
        subject_id=cast(str, components["subject_id"]),
        context_id=cast(str, components["context_id"]),
        receiver=receiver,
        family_id=cast(str, components["family_id"]),
        driver_id=cast(str, components["driver_id"]),
        interaction_id=cast(str, components["interaction_id"]),
        mode=cast(str, components["mode"]),
    )

    assert observed == stable_id(
        "persisted_crossfit_source_row",
        components,
        schema_version="1",
    )


def test_global_source_digest_streaming_and_sort_fallback_match_legacy() -> None:
    columns = ("identifier", "value", "available")
    table = pd.DataFrame(
        [
            ("row_10", 0.25, True),
            ("row_2", math.nan, False),
            ("row_1", 1.0, True),
        ],
        columns=columns,
    )
    normalized = [
        [_canonical_table_scalar(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    sorted_indices = sorted(
        range(len(table)),
        key=lambda index: canonical_json(normalized[index]),
    )
    expected_rows = [normalized[index] for index in sorted_indices]
    expected = stable_id(
        "test_source_rows",
        {"columns": list(columns), "rows": expected_rows},
        schema_version="1",
        digest_length=64,
    )

    sorted_table = table.iloc[sorted_indices].reset_index(drop=True)
    reversed_table = sorted_table.iloc[::-1].reset_index(drop=True)
    assert (
        _global_source_table_digest("test_source_rows", sorted_table, columns)
        == expected
    )
    assert (
        _global_source_table_digest("test_source_rows", reversed_table, columns)
        == expected
    )


def test_v6_conserved_sender_groups_reject_incomplete_or_invalid_weights() -> None:
    sender = _contrast_common_sender_table()

    partial = sender.copy(deep=True)
    partial.loc[0, "assignment_weight"] = None
    with pytest.raises(ValueError, match="complete or entirely missing"):
        _validate_contrast_common_sender_lr_table(partial)

    nonnormalized = sender.copy(deep=True)
    nonnormalized.loc[:, "assignment_weight"] = [0.5, 0.25]
    with pytest.raises(ValueError, match="do not sum to one"):
        _validate_contrast_common_sender_lr_table(nonnormalized)

    wrong_entropy = sender.copy(deep=True)
    wrong_entropy.loc[:, "normalized_entropy"] = 0.0
    with pytest.raises(ValueError, match="entropy does not match"):
        _validate_contrast_common_sender_lr_table(wrong_entropy)


def test_v6_conserved_sender_groups_enforce_zero_and_ne_precedence() -> None:
    sender = _contrast_common_sender_table()

    zero_parent = sender.copy(deep=True)
    zero_parent.loc[:, "global_lr_score"] = 0.0
    zero_parent.loc[:, ["assignment_weight", "normalized_entropy"]] = math.nan
    zero_parent.loc[:, "global_sender_lr_score"] = 0.0
    zero_parent.loc[:, "status"] = "structural_zero"
    zero_parent.loc[:, "reason_code"] = "global_lr_score_zero"
    assert len(_validate_contrast_common_sender_lr_table(zero_parent)) == 2

    missing_parent = sender.copy(deep=True)
    missing_parent.loc[:, "global_lr_score"] = math.nan
    missing_parent.loc[:, "global_sender_lr_score"] = math.nan
    missing_parent.loc[:, "status"] = "not_estimable"
    missing_parent.loc[:, "reason_code"] = "global_lr_score_not_estimable"
    assert len(_validate_contrast_common_sender_lr_table(missing_parent)) == 2

    missing_assignment = sender.copy(deep=True)
    missing_assignment.loc[:, ["assignment_weight", "normalized_entropy"]] = math.nan
    missing_assignment.loc[:, "global_sender_lr_score"] = math.nan
    missing_assignment.loc[:, "status"] = "not_estimable"
    missing_assignment.loc[:, "reason_code"] = "sender_assignment_not_estimable"
    assert len(_validate_contrast_common_sender_lr_table(missing_assignment)) == 2

    zero_assignment = sender.copy(deep=True)
    parent = cast(float, zero_assignment.loc[0, "global_lr_score"])
    zero_assignment.loc[:, "assignment_weight"] = [1.0, 0.0]
    zero_assignment.loc[:, "normalized_entropy"] = 0.0
    zero_assignment.loc[:, "global_sender_lr_score"] = [parent, 0.0]
    zero_assignment.loc[1, "status"] = "structural_zero"
    zero_assignment.loc[1, "reason_code"] = "sender_assignment_weight_zero"
    assert len(_validate_contrast_common_sender_lr_table(zero_assignment)) == 2


def test_contrast_common_collection_allows_multiple_sender_applications() -> None:
    manifest = _contrast_common_manifest()

    _validate_contrast_common_collections_manifest(manifest)


def test_not_estimable_gain_binding_disables_cross_receiver_rank_eligibility() -> None:
    manifest = _contrast_common_manifest()
    collection = cast(list[dict[str, object]], manifest["contrast_common_collections"])[
        0
    ]
    application = cast(list[dict[str, object]], collection["fold_applications"])[0]
    binding = cast(
        list[dict[str, object]],
        application["receiver_gain_calibration_bindings"],
    )[0]
    binding["gain_calibration_status"] = "not_estimable"
    binding["gain_calibration_reason_code"] = "insufficient_inner_oof_subjects"
    binding["positive_gain_source_knots"] = []
    binding["positive_gain_percentile_knots"] = []
    binding_payload = {
        key: value
        for key, value in binding.items()
        if key != "gain_calibration_binding_id"
    }
    binding["gain_calibration_binding_id"] = stable_id(
        "receiver_gain_calibration_binding", binding_payload, schema_version="1"
    )
    application["all_receivers_gain_calibrated"] = False
    application["cross_receiver_percentile_rank_eligible"] = False
    collection_payload = {
        key: value
        for key, value in collection.items()
        if key != "contrast_common_collection_id"
    }
    collection["contrast_common_collection_id"] = stable_id(
        "contrast_common_oof_score_collection",
        collection_payload,
        schema_version="1",
    )

    _validate_contrast_common_collections_manifest(manifest)

    application["cross_receiver_percentile_rank_eligible"] = True
    with pytest.raises(ValueError, match="eligibility flags"):
        _validate_contrast_common_collections_manifest(manifest)


def test_contrast_common_collection_rejects_lineage_and_identity_tamper() -> None:
    manifest = _contrast_common_manifest()
    collections = cast(list[dict[str, object]], manifest["contrast_common_collections"])
    applications = cast(list[dict[str, object]], collections[0]["fold_applications"])
    lineages = cast(list[object], applications[0]["sender_lineages"])
    lineages.reverse()

    with pytest.raises(ValueError, match="noncanonical or incomplete"):
        _validate_contrast_common_collections_manifest(manifest)

    overlap = _contrast_common_manifest()
    overlap_collection = cast(
        list[dict[str, object]], overlap["contrast_common_collections"]
    )[0]
    overlap_application = cast(
        list[dict[str, object]], overlap_collection["fold_applications"]
    )[0]
    overlap_application["training_subject_ids"] = ["subject-1"]
    with pytest.raises(ValueError, match="subjects overlap"):
        _validate_contrast_common_collections_manifest(overlap)

    identity = _contrast_common_manifest()
    identity_collection = cast(
        list[dict[str, object]], identity["contrast_common_collections"]
    )[0]
    identity_collection["contrast_common_collection_id"] = "collection-poisoned"
    with pytest.raises(ValueError, match="identity is inconsistent"):
        _validate_contrast_common_collections_manifest(identity)

    calibration = _contrast_common_manifest()
    calibration_collection = cast(
        list[dict[str, object]], calibration["contrast_common_collections"]
    )[0]
    calibration_application = cast(
        list[dict[str, object]], calibration_collection["fold_applications"]
    )[0]
    bindings = cast(
        list[dict[str, object]],
        calibration_application["receiver_gain_calibration_bindings"],
    )
    bindings[0]["positive_gain_percentile_knots"] = [0.0, 0.7, 1.0]
    with pytest.raises(ValueError, match="binding identity"):
        _validate_contrast_common_collections_manifest(calibration)

    overclaim = _contrast_common_manifest()
    overclaim_collection = cast(
        list[dict[str, object]], overclaim["contrast_common_collections"]
    )[0]
    overclaim_collection["common_functional_across_receivers"] = True
    with pytest.raises(ValueError, match="collection scope is inconsistent"):
        _validate_contrast_common_collections_manifest(overclaim)

    application_overclaim = _contrast_common_manifest()
    descriptive_collection = cast(
        list[dict[str, object]],
        application_overclaim["contrast_common_collections"],
    )[0]
    descriptive_application = cast(
        list[dict[str, object]], descriptive_collection["fold_applications"]
    )[0]
    descriptive_application["common_functional_across_receivers"] = True
    with pytest.raises(ValueError, match="eligibility flags"):
        _validate_contrast_common_collections_manifest(application_overclaim)


def test_contrast_common_table_lineage_binds_source_digests_and_children() -> None:
    applications, tables = _contrast_common_lineage_case()

    _validate_contrast_common_table_lineage(
        CROSSFIT_CONTRAST_COMMON_LR_TABLE,
        tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE],
        applications,
    )
    _validate_contrast_common_table_lineage(
        CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
        tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE],
        applications,
    )
    _validate_sender_lr_cross_table_lineage(
        tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE],
        tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE],
    )

    missing_sender = (
        tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE].iloc[:1].copy(deep=True)
    )
    with pytest.raises(ResultValidationError, match="collection registry"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            missing_sender,
            applications,
        )

    child_tamper = tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE].copy(deep=True)
    child_tamper.loc[0, "source_family_common_application_id"] = "application-poisoned"
    with pytest.raises(ResultValidationError, match="receiver-child lineage"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            child_tamper,
            applications,
        )

    score_tamper = tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE].copy(deep=True)
    score_tamper.loc[0, "global_lr_score"] = 0.1
    with pytest.raises(ResultValidationError, match="frozen formula"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            score_tamper,
            applications,
        )

    sender_parent_tamper = tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE].copy(
        deep=True
    )
    sender_parent_tamper.loc[0, "global_lr_score"] = 0.7
    with pytest.raises(ResultValidationError, match="LR/calibration parent"):
        _validate_sender_lr_cross_table_lineage(
            tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE], sender_parent_tamper
        )

    calibration_tamper = tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE].copy(deep=True)
    calibration_tamper.loc[0, "calibrated_family_gain_percentile"] = 0.7
    with pytest.raises(ResultValidationError, match="frozen formula"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            calibration_tamper,
            applications,
        )

    raw_gain_tamper = tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE].copy(deep=True)
    raw_gain_tamper.loc[0, "receiver_relative_family_gain"] = 0.25
    with pytest.raises(ResultValidationError, match="frozen formula"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            raw_gain_tamper,
            applications,
        )

    prior_tamper = tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE].copy(deep=True)
    prior_tamper.loc[0, "prior_quality"] = 0.8
    with pytest.raises(ResultValidationError, match="frozen formula"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            prior_tamper,
            applications,
        )

    power_tamper = copy.deepcopy(applications)
    _, tampered_application = power_tamper["global-application-1"]
    tampered_application["softmin_power"] = 2.0
    with pytest.raises(ResultValidationError, match="frozen formula"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE],
            power_tamper,
        )

    binding_tamper = tables[CROSSFIT_CONTRAST_COMMON_LR_TABLE].copy(deep=True)
    binding_tamper.loc[0, "gain_calibration_artifact_id"] = "gain-artifact-2"
    with pytest.raises(ResultValidationError, match="calibration binding"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_LR_TABLE,
            binding_tamper,
            applications,
        )


def test_contrast_common_table_lineage_rejects_missing_sender_application() -> None:
    applications, tables = _contrast_common_lineage_case()
    missing = tables[CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE].iloc[:1].copy()

    with pytest.raises(ResultValidationError, match="collection registry"):
        _validate_contrast_common_table_lineage(
            CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
            missing,
            applications,
        )


def test_contrast_common_source_registry_binds_calibration_metadata() -> None:
    manifest, applications = _contrast_common_source_case()

    _validate_source_contrast_common_registry(manifest, applications)

    source = cast(dict[str, object], manifest["source_crossfit_manifest"])
    folds = cast(list[dict[str, object]], source["fold_artifacts"])
    records = cast(
        list[dict[str, object]],
        folds[0]["cross_receiver_common_scoring_artifacts"],
    )
    records[0]["softmin_power"] = 2.0
    with pytest.raises(ResultValidationError, match="source lineage"):
        _validate_source_contrast_common_registry(manifest, applications)


def test_crossfit_result_cannot_be_constructed_by_a_caller() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        CrossFitResult()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "table_name", CROSSFIT_RECEIVER_TABLE_NAMES
)
@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field_name", "poisoned"),
    (
        ("fold_id", "fold-poisoned"),
        ("contrast_id", "contrast-id-poisoned"),
        ("contrast", "poisoned_vs_control"),
        ("receiver", "ReceiverPoisoned"),
        ("family_common_functional_id", "functional-poisoned"),
        ("family_common_application_id", "application-unknown"),
        ("family_common_binding_id", "binding-poisoned"),
        ("score_version", "score-poisoned"),
        ("certification_status", "certification-poisoned"),
        ("is_oof_certified", True),
        ("claim_scope", "claim-poisoned"),
    ),
)
def test_application_lineage_projection_rejects_every_table_field_tamper(
    table_name: str,
    field_name: str,
    poisoned: object,
) -> None:
    manifest, tables = _registry_link_case()
    tables[table_name].loc[0, field_name] = poisoned

    with pytest.raises(ResultValidationError):
        _validate_registry_links(manifest, tables)


def test_component_projection_checks_sender_functional_lineage() -> None:
    manifest, tables = _registry_link_case()
    tables[CROSSFIT_COMPONENT_TABLE].loc[0, "sender_functional_id"] = (
        "sender-functional-poisoned"
    )

    with pytest.raises(
        ResultValidationError,
        match="mix or misstate application lineage",
    ):
        _validate_registry_links(manifest, tables)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "table_name", CROSSFIT_RECEIVER_TABLE_NAMES
)
def test_application_coverage_rejects_missing_rows(table_name: str) -> None:
    manifest, tables = _registry_link_case()
    tables[table_name] = tables[table_name].iloc[0:0].copy()

    with pytest.raises(
        ResultValidationError,
        match="rows do not match the application registry",
    ):
        _validate_registry_links(manifest, tables)


def test_application_projection_rejects_mixed_lineage_under_one_id() -> None:
    manifest, tables = _registry_link_case()
    duplicate = tables[CROSSFIT_COMPONENT_TABLE].copy(deep=True)
    tables[CROSSFIT_COMPONENT_TABLE] = pd.concat(
        [tables[CROSSFIT_COMPONENT_TABLE], duplicate],
        ignore_index=True,
    )
    tables[CROSSFIT_COMPONENT_TABLE].loc[1, "family_common_binding_id"] = (
        "binding-poisoned"
    )
    _application(manifest)["persisted_component_rows"] = 2

    with pytest.raises(
        ResultValidationError,
        match="mix or misstate application lineage",
    ):
        _validate_registry_links(manifest, tables)


def test_registry_allows_planned_not_produced_child_without_application() -> None:
    manifest, tables = _registry_link_case()

    _validate_registry_links(manifest, tables)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field_name", "poisoned"),
    (
        ("fold_id", "fold-poisoned"),
        ("contrast", "poisoned_vs_control"),
        ("receiver", "ReceiverPoisoned"),
        ("family_common_functional_id", "functional-poisoned"),
        ("score_version", "score-poisoned"),
    ),
)
def test_registry_rejects_application_scope_or_functional_lineage_tamper(
    field_name: str,
    poisoned: object,
) -> None:
    manifest, tables = _registry_link_case()
    _set_application_projection(manifest, tables, field_name, poisoned)

    with pytest.raises(ResultValidationError, match=r"registry|registered"):
        _validate_registry_links(manifest, tables)


def test_registry_rejects_registered_child_without_application() -> None:
    manifest, tables = _registry_link_case()
    registry = _scoring_registry(planned_only_registered=True)
    assert registry.registry_id is not None
    source = _source(manifest)
    source["receiver_scoring_registry_id"] = registry.registry_id
    source["receiver_scoring_registry"] = registry.to_dict()

    with pytest.raises(
        ResultValidationError,
        match="do not exactly cover registered planned receiver children",
    ):
        _validate_registry_links(manifest, tables)


def test_registry_rejects_two_applications_for_one_planned_child() -> None:
    manifest, tables = _registry_link_case()
    duplicate = copy.deepcopy(_application(manifest))
    duplicate["family_common_functional_id"] = "functional-duplicate"
    duplicate["family_common_application_id"] = "application-duplicate"
    duplicate["family_common_binding_id"] = "binding-duplicate"
    duplicate["persisted_component_rows"] = 0
    duplicate["persisted_differential_rows"] = 0
    applications = cast(list[dict[str, object]], manifest["applications"])
    applications.append(duplicate)

    with pytest.raises(
        ResultValidationError,
        match="Multiple applications claim one planned scoring registry child",
    ):
        _validate_registry_links(manifest, tables)


def test_registry_rejects_noncanonical_persisted_order() -> None:
    manifest, tables = _registry_link_case()
    raw_registry = cast(
        dict[str, object],
        _source(manifest)["receiver_scoring_registry"],
    )
    collections = cast(list[dict[str, object]], raw_registry["collections"])
    children = cast(list[object], collections[0]["children"])
    children.reverse()

    with pytest.raises(ResultValidationError, match="canonically serialized"):
        _validate_registry_links(manifest, tables)


def test_registry_rejects_poisoned_plan_filter_universe() -> None:
    manifest, tables = _registry_link_case()
    raw_registry = cast(
        dict[str, object],
        _source(manifest)["receiver_scoring_registry"],
    )
    plans = cast(list[dict[str, object]], raw_registry["planned_collections"])
    plans[0]["filter_universe_id"] = "filter-universe-poisoned"

    with pytest.raises(
        ResultValidationError,
        match="registry violates its contract",
    ):
        _validate_registry_links(manifest, tables)


def test_registry_rejects_source_registry_identity_mismatch() -> None:
    manifest, tables = _registry_link_case()
    _source(manifest)["receiver_scoring_registry_id"] = "registry-poisoned"

    with pytest.raises(ResultValidationError, match="identity is inconsistent"):
        _validate_registry_links(manifest, tables)


def test_current_schema_rejects_missing_source_registry() -> None:
    manifest, tables = _registry_link_case()
    _source(manifest).pop("receiver_scoring_registry")

    with pytest.raises(ResultValidationError, match="registry is missing or invalid"):
        _validate_registry_links(manifest, tables)


def test_legacy_schema_allows_source_manifest_without_receiver_registry() -> None:
    manifest, tables = _registry_link_case()
    manifest["schema_version"] = "1.0.0"
    source = _source(manifest)
    source.pop("receiver_scoring_registry")
    source.pop("receiver_scoring_registry_id")

    _validate_registry_links(manifest, tables)
