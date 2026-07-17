from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import LONG_TABLE_COLUMNS
from benchmarks.adapters.crychic.readback_crossfit import (
    _V3_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
    _V3_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
    _V6_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
    _V6_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
    _V45_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
    _V45_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
    BENCHMARK_SCOPE,
    convert_crossfit_result_to_long,
)

from crychic.attribution import GAIN_CALIBRATION_PERCENTILE_POLICY
from crychic.core import stable_id
from crychic.design import context_fields


@dataclass
class _FakeCrossFitResult:
    _manifest: dict[str, object]
    _sender: pd.DataFrame
    _lr: pd.DataFrame
    _directional: pd.DataFrame | None = None
    directional_read_count: int = 0

    @property
    def manifest(self) -> dict[str, object]:
        return copy.deepcopy(self._manifest)

    def read_contrast_common_sender_lr_scores(self) -> pd.DataFrame:
        return self._sender.copy(deep=True)

    def read_contrast_common_lr_scores(self) -> pd.DataFrame:
        return self._lr.copy(deep=True)

    def read_directional_channel_registry(self) -> pd.DataFrame:
        self.directional_read_count += 1
        if self._directional is None:
            raise KeyError("directional_channel_registry")
        return self._directional.copy(deep=True)


def _adata() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "sample_id": ["sample-1", "sample-1", "sample-2", "sample-2"],
            "subject_id": ["subject-1", "subject-1", "subject-2", "subject-2"],
            "condition": ["case", "case", "case", "case"],
            "cell_type": ["Sender", "Receiver", "Sender", "Receiver"],
        },
        index=[f"cell-{index}" for index in range(4)],
    )
    return ad.AnnData(X=np.ones((4, 2)), obs=obs)


def _application(
    *, fold_id: str, sample_number: int, context_id: str
) -> dict[str, object]:
    subject_id = f"subject-{sample_number}"
    other_subject = "subject-2" if sample_number == 1 else "subject-1"
    return {
        "fold_id": fold_id,
        "global_common_functional_id": f"global-functional-{sample_number}",
        "global_common_application_id": f"global-application-{sample_number}",
        "functional_spec_id": "functional-spec-1",
        "functional_schema_version": "2.0.0",
        "filter_universe_id": "filter-universe-1",
        "context_ids": [context_id],
        "receiver_ids": ["Receiver"],
        "training_subject_ids": [other_subject],
        "heldout_subject_ids": [subject_id],
        "receiver_children": [
            {
                "receiver": "Receiver",
                "family_common_functional_id": f"family-functional-{sample_number}",
                "family_common_application_id": f"family-application-{sample_number}",
            }
        ],
        "sender_functional_id": "sender-functional-1",
        "sender_lineages": [
            {
                "receiver": "Receiver",
                "sender_functional_id": "sender-functional-1",
                "sender_application_id": f"sender-application-{sample_number}",
            }
        ],
        "sender_application_digests": [
            {
                "receiver": "Receiver",
                "sender_application_digest": f"sender-digest-{sample_number}",
            }
        ],
        "calibration_policy": (
            "positive_family_coefficient_quantile_median_shrink_only_v1"
        ),
        "calibration_quantile": 0.75,
        "global_training_anchor": 0.8,
        "receiver_training_anchors": [["Receiver", 1.0]],
        "receiver_scale_factors": [["Receiver", 0.8]],
        "training_only_receiver_calibration": True,
        "receiver_scale_amplification": False,
        "common_functional_across_receivers": False,
        "receiver_balanced_descriptive_collection": True,
        "interaction_mapping_digest": "mapping-digest",
        "functional_certification_status": "heldout-diagnostic",
        "lr_scores_digest": f"lr-digest-{sample_number}",
        "sender_scores_digest": f"sender-score-digest-{sample_number}",
        "n_lr_rows": 3,
        "n_sender_rows": 3,
    }


def _collection() -> dict[str, object]:
    context_id, _ = context_fields({"condition": "case"}, ("condition",))
    payload: dict[str, object] = {
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "contrast_id": "contrast-1",
        "contrast": "case_vs_control",
        "score_version": "global-score-v2",
        "estimand": "receiver_balanced_descriptive_collection_v2",
        "common_functional_across_receivers": False,
        "receiver_balanced_descriptive_collection": True,
        "formal_inference_allowed": False,
        "certification_status": "uncertified",
        "is_oof_certified": False,
        "claim_scope": "heldout_cross_receiver_common_diagnostic",
        "fold_applications": [
            _application(fold_id="fold-1", sample_number=1, context_id=context_id),
            _application(fold_id="fold-2", sample_number=2, context_id=context_id),
        ],
    }
    return {
        "contrast_common_collection_id": stable_id(
            "contrast_common_oof_score_collection", payload, schema_version="1"
        ),
        **payload,
    }


def _source_tables(collection: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    collection_id = str(collection["contrast_common_collection_id"])
    applications = cast(list[dict[str, object]], collection["fold_applications"])
    sender_base = {
        column: None for column in _V3_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
    }
    lr_base = {column: None for column in _V3_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS}
    sender_rows: list[dict[str, object]] = []
    lr_rows: list[dict[str, object]] = []
    definitions = (
        ("interaction-observed", "observed", 0.8, 0.4, None),
        ("interaction-zero", "structural_zero", 0.0, 0.0, "structural"),
        ("interaction-ne", "not_estimable", None, None, "not-estimable"),
    )
    for sample_number, application in enumerate(applications, start=1):
        sample_id = f"sample-{sample_number}"
        subject_id = f"subject-{sample_number}"
        context_id = cast(list[str], application["context_ids"])[0]
        for row_number, (
            interaction,
            status,
            lr_score,
            sender_score,
            reason,
        ) in enumerate(definitions, start=1):
            common = {
                "crossfit_id": "crossfit-1",
                "spec_id": "spec-1",
                "repeat_id": "repeat-1",
                "fold_id": application["fold_id"],
                "contrast_id": "contrast-1",
                "contrast": "case_vs_control",
                "contrast_common_collection_id": collection_id,
                "global_common_application_id": application[
                    "global_common_application_id"
                ],
                "global_common_functional_id": application[
                    "global_common_functional_id"
                ],
                "source_family_common_functional_id": (
                    f"family-functional-{sample_number}"
                ),
                "source_family_common_application_id": (
                    f"family-application-{sample_number}"
                ),
                "sample_id": sample_id,
                "subject_id": subject_id,
                "context_id": context_id,
                "receiver": "Receiver",
                "family_id": "family-1",
                "driver_id": "driver-1",
                "interaction_id": interaction,
                "mode": "state",
                "training_receiver_scale_factor": 0.8,
                "status": status,
                "reason_code": reason,
                "score_version": "global-score-v2",
                "certification_status": "uncertified",
                "is_oof_certified": False,
                "formal_inference_status": "not_available_descriptive_only",
                "claim_scope": "heldout_cross_receiver_common_diagnostic",
            }
            lr_rows.append(
                {
                    **lr_base,
                    **common,
                    "receptor_eligible": True,
                    "ligand_contrast_gate_status": "eligible",
                    "availability": lr_score,
                    "receiver_relative_family_gain": lr_score,
                    "prior_quality": 1.0 if lr_score is not None else None,
                    "training_family_coefficient": 1.0,
                    "global_lr_core_strength": lr_score,
                    "global_lr_score": lr_score,
                    "source_row_id": f"lr-row-{sample_number}-{row_number}",
                }
            )
            sender_rows.append(
                {
                    **sender_base,
                    **common,
                    "source_sender_functional_id": "sender-functional-1",
                    "source_sender_application_id": (
                        f"sender-application-{sample_number}"
                    ),
                    "sender": "Sender",
                    "global_lr_score": lr_score,
                    "ligand_availability": 0.5 if lr_score is not None else None,
                    "training_prevalence_prior": (
                        1.0 if lr_score is not None else None
                    ),
                    "raw_sender_evidence": 0.5 if lr_score is not None else None,
                    "global_sender_lr_score": sender_score,
                    "source_row_id": f"sender-row-{sample_number}-{row_number}",
                }
            )
    return (
        pd.DataFrame(
            sender_rows,
            columns=_V3_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
        ),
        pd.DataFrame(lr_rows, columns=_V3_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS),
    )


def _result() -> _FakeCrossFitResult:
    collection = _collection()
    sender, lr = _source_tables(collection)
    manifest: dict[str, object] = {
        "schema_version": "3.0.0",
        "status": "complete",
        "crossfit_result_id": "crossfit-result-1",
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "contrast_common_stage_connected": True,
        "contrast_common_collections": [collection],
    }
    return _FakeCrossFitResult(manifest, sender, lr)


def _v4_binding(*, sample_number: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "receiver": "Receiver",
        "source_family_common_functional_id": (f"family-functional-{sample_number}"),
        "gain_calibration_artifact_id": f"gain-artifact-{sample_number}",
        "gain_calibration_spec_id": "gain-spec-1",
        "gain_calibration_status": "observed",
        "gain_calibration_reason_code": None,
        "percentile_policy": GAIN_CALIBRATION_PERCENTILE_POLICY,
        "positive_gain_source_knots": [0.0, 0.5, 1.0],
        "positive_gain_percentile_knots": [0.0, 0.5, 1.0],
        "n_supported_families": 1,
        "n_positive_observations": 4,
        "n_distinct_positive_gains": 2,
        "tuning_id": f"tuning-{sample_number}",
        "outer_incremental_functional_id": f"incremental-{sample_number}",
        "outer_selected_resolved_penalty_id": f"penalty-{sample_number}",
    }
    return {
        "gain_calibration_binding_id": stable_id(
            "receiver_gain_calibration_binding", payload, schema_version="1"
        ),
        **payload,
    }


def _v4_application(
    *, fold_id: str, sample_number: int, context_id: str
) -> dict[str, object]:
    application = _application(
        fold_id=fold_id,
        sample_number=sample_number,
        context_id=context_id,
    )
    for field in (
        "calibration_quantile",
        "global_training_anchor",
        "receiver_training_anchors",
        "receiver_scale_factors",
    ):
        del application[field]
    application.update(
        {
            "functional_schema_version": "3.0.0",
            "calibration_policy": ("selected_penalty_inner_oof_positive_gain_ecdf_v1"),
            "scale_policy": (
                "selected_penalty_inner_oof_gain_percentile_no_heldout_rescaling_v3"
            ),
            "sender_policy": (
                "absolute_sender_evidence_without_candidate_normalization_v1"
            ),
            "softmin_power": 4.0,
            "epsilon": 1e-12,
            "receiver_gain_calibration_bindings": [
                _v4_binding(sample_number=sample_number)
            ],
            "all_receivers_gain_calibrated": True,
            "cross_receiver_percentile_rank_eligible": True,
        }
    )
    return application


def _v4_collection() -> dict[str, object]:
    context_id, _ = context_fields({"condition": "case"}, ("condition",))
    payload: dict[str, object] = {
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "contrast_id": "contrast-1",
        "contrast": "case_vs_control",
        "score_version": "receiver_gain_percentile_mechanistic_strength_v3",
        "estimand": "receiver_balanced_descriptive_collection_v3",
        "common_functional_across_receivers": False,
        "receiver_balanced_descriptive_collection": True,
        "formal_inference_allowed": False,
        "certification_status": "uncertified",
        "is_oof_certified": False,
        "claim_scope": "heldout_cross_receiver_common_diagnostic",
        "fold_applications": [
            _v4_application(fold_id="fold-1", sample_number=1, context_id=context_id),
            _v4_application(fold_id="fold-2", sample_number=2, context_id=context_id),
        ],
    }
    return {
        "contrast_common_collection_id": stable_id(
            "contrast_common_oof_score_collection", payload, schema_version="1"
        ),
        **payload,
    }


def _v4_source_tables(
    collection: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_sender, old_lr = _source_tables(collection)
    applications = cast(list[dict[str, object]], collection["fold_applications"])
    bindings = {
        str(application["global_common_application_id"]): cast(
            list[dict[str, object]],
            application["receiver_gain_calibration_bindings"],
        )[0]
        for application in applications
    }
    for table in (old_sender, old_lr):
        table["score_version"] = "receiver_gain_percentile_mechanistic_strength_v3"
        table["gain_calibration_binding_id"] = table[
            "global_common_application_id"
        ].map(
            {
                application_id: binding["gain_calibration_binding_id"]
                for application_id, binding in bindings.items()
            }
        )
        table["gain_calibration_artifact_id"] = table[
            "global_common_application_id"
        ].map(
            {
                application_id: binding["gain_calibration_artifact_id"]
                for application_id, binding in bindings.items()
            }
        )
        table["gain_calibration_status"] = "observed"
        table["gain_calibration_reason_code"] = None
    old_lr["calibrated_family_gain_percentile"] = old_lr[
        "receiver_relative_family_gain"
    ]
    sender = old_sender.reindex(columns=_V45_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS)
    lr = old_lr.reindex(columns=_V45_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS)
    return sender, lr


def _v4_result() -> _FakeCrossFitResult:
    collection = _v4_collection()
    sender, lr = _v4_source_tables(collection)
    manifest: dict[str, object] = {
        "schema_version": "4.0.0",
        "status": "complete",
        "crossfit_result_id": "crossfit-result-v4",
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "contrast_common_stage_connected": True,
        "contrast_common_collections": [collection],
    }
    return _FakeCrossFitResult(manifest, sender, lr)


def _v5_result() -> _FakeCrossFitResult:
    result = _v4_result()
    result._manifest["schema_version"] = "5.0.0"
    result._directional = pd.DataFrame(
        [
            {
                "pair_spec_id": "pair-1",
                "binding_id": "binding-1",
                "fold_id": "fold-1",
                "receiver": "Receiver",
                "channel": "increased_activation_compatible",
                "status": "observed",
            },
            {
                "pair_spec_id": "pair-1",
                "binding_id": "binding-1",
                "fold_id": "fold-1",
                "receiver": "Receiver",
                "channel": "reduced_activation_compatible",
                "status": "observed",
            },
        ]
    )
    return result


def _v6_application(
    *, fold_id: str, sample_number: int, context_id: str
) -> dict[str, object]:
    application = _v4_application(
        fold_id=fold_id,
        sample_number=sample_number,
        context_id=context_id,
    )
    application.update(
        {
            "functional_schema_version": "4.0.0",
            "sender_policy": ("contrast_common_frozen_softmax_conserved_allocation_v1"),
            "n_sender_rows": 6,
        }
    )
    return application


def _v6_collection() -> dict[str, object]:
    context_id, _ = context_fields({"condition": "case"}, ("condition",))
    payload: dict[str, object] = {
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "contrast_id": "contrast-1",
        "contrast": "case_vs_control",
        "score_version": "receiver_gain_percentile_mechanistic_conserved_sender_v4",
        "estimand": "receiver_balanced_descriptive_mechanistic_evidence_collection_v4",
        "common_functional_across_receivers": False,
        "receiver_balanced_descriptive_collection": True,
        "formal_inference_allowed": False,
        "certification_status": "uncertified",
        "is_oof_certified": False,
        "claim_scope": "heldout_cross_receiver_common_diagnostic",
        "fold_applications": [
            _v6_application(fold_id="fold-1", sample_number=1, context_id=context_id),
            _v6_application(fold_id="fold-2", sample_number=2, context_id=context_id),
        ],
    }
    return {
        "contrast_common_collection_id": stable_id(
            "contrast_common_oof_score_collection", payload, schema_version="1"
        ),
        **payload,
    }


def _v6_source_tables(
    collection: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy_sender, legacy_lr = _v4_source_tables(collection)
    score_version = "receiver_gain_percentile_mechanistic_conserved_sender_v4"
    legacy_lr["score_version"] = score_version
    lr = legacy_lr.reindex(columns=_V6_CROSSFIT_CONTRAST_COMMON_LR_COLUMNS)

    entropy = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)) / np.log(2.0)
    rows: list[dict[str, object]] = []
    for source in legacy_sender.to_dict(orient="records"):
        parent_status = str(source["status"])
        parent_score = source["global_lr_score"]
        for sender, weight in (("Sender-A", 0.25), ("Sender-B", 0.75)):
            row = dict(source)
            row["sender"] = sender
            row["score_version"] = score_version
            row["source_row_id"] = f"{source['source_row_id']}-{sender}"
            if parent_status == "not_estimable":
                row.update(
                    {
                        "ligand_availability": None,
                        "training_prevalence_prior": None,
                        "raw_sender_evidence": None,
                        "assignment_weight": None,
                        "normalized_entropy": None,
                        "global_sender_lr_score": None,
                    }
                )
            else:
                row.update(
                    {
                        "ligand_availability": weight,
                        "training_prevalence_prior": 1.0,
                        "raw_sender_evidence": weight,
                        "assignment_weight": weight,
                        "normalized_entropy": entropy,
                        "global_sender_lr_score": float(parent_score) * weight,
                    }
                )
            rows.append(row)
    sender = pd.DataFrame(
        rows,
        columns=_V6_CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
    )
    return sender, lr


def _v6_result() -> _FakeCrossFitResult:
    collection = _v6_collection()
    sender, lr = _v6_source_tables(collection)
    manifest: dict[str, object] = {
        "schema_version": "6.0.0",
        "status": "complete",
        "crossfit_result_id": "crossfit-result-v6",
        "crossfit_id": "crossfit-1",
        "spec_id": "spec-1",
        "repeat_id": "repeat-1",
        "contrast_common_stage_connected": True,
        "contrast_common_collections": [collection],
    }
    return _FakeCrossFitResult(manifest, sender, lr)


def _reidentify_collection(result: _FakeCrossFitResult) -> None:
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    payload = {
        key: value
        for key, value in collection.items()
        if key != "contrast_common_collection_id"
    }
    collection_id = stable_id(
        "contrast_common_oof_score_collection", payload, schema_version="1"
    )
    old_id = str(collection["contrast_common_collection_id"])
    collection["contrast_common_collection_id"] = collection_id
    result._sender.loc[
        result._sender["contrast_common_collection_id"].eq(old_id),
        "contrast_common_collection_id",
    ] = collection_id
    result._lr.loc[
        result._lr["contrast_common_collection_id"].eq(old_id),
        "contrast_common_collection_id",
    ] = collection_id


def test_crossfit_readback_maps_status_without_inferential_claims() -> None:
    result = _result()
    collection_id = str(
        cast(
            list[dict[str, object]],
            result._manifest["contrast_common_collections"],
        )[0]["contrast_common_collection_id"]
    )

    table, views = convert_crossfit_result_to_long(
        result,
        _adata(),
        dataset_id="toy",
        collection_id=collection_id,
        contrast="case_vs_control",
        method_version="test-version",
    )

    assert tuple(table.columns) == LONG_TABLE_COLUMNS
    observed = table.loc[table["interaction_id"].eq("interaction-observed")]
    structural = table.loc[table["interaction_id"].eq("interaction-zero")]
    missing = table.loc[table["interaction_id"].eq("interaction-ne")]
    assert set(observed["status"]) == {"ok"}
    assert set(observed["score"]) == {0.4}
    assert set(structural["status"]) == {"ok"}
    assert set(structural["score"]) == {0.0}
    assert set(missing["status"]) == {"missing"}
    assert missing["score"].isna().all()
    assert (
        table[
            [
                "within_dataset_p_value",
                "differential_effect",
                "differential_p_value",
                "differential_q_value",
            ]
        ]
        .isna()
        .all()
        .all()
    )
    assert len(views) == 1
    assert views[0]["common_functional_claim"] is False
    assert views[0]["receiver_balanced_descriptive_collection"] is True
    assert views[0]["global_cross_receiver_endpoint_eligible"] is False
    assert views[0]["benchmark_scope"] == BENCHMARK_SCOPE
    assert views[0]["native_only_no_candidate_expansion"] is True


def test_crossfit_readback_rejects_manifest_scope_tamper() -> None:
    result = _result()
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    collection["common_functional_across_receivers"] = True

    with pytest.raises(ValueError, match="scope or lineage"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy")


def test_crossfit_readback_rejects_receiver_factor_mismatch() -> None:
    result = _result()
    result._sender.loc[0, "training_receiver_scale_factor"] = 0.7

    with pytest.raises(ValueError, match="scale_factor disagrees"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy")


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("status", "score", "message"),
    [
        ("unsupported", 0.4, "unsupported contrast-common statuses"),
        ("structural_zero", 0.2, "exact zero"),
        ("not_estimable", 0.2, "must not carry a score"),
    ],
)
def test_crossfit_readback_rejects_invalid_status_score_pairs(
    status: str, score: float, message: str
) -> None:
    result = _result()
    result._sender.loc[0, "status"] = status
    result._sender.loc[0, "global_sender_lr_score"] = score

    with pytest.raises(ValueError, match=message):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy")


def test_crossfit_readback_rejects_duplicate_native_grid_row() -> None:
    result = _result()
    result._sender = pd.concat(
        [result._sender, result._sender.iloc[[0]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="duplicate sample/receiver/fold grid"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy")


def test_crossfit_readback_fails_closed_on_different_sample_universes() -> None:
    result = _result()
    removed = result._sender["source_row_id"].eq("sender-row-2-3")
    result._sender = result._sender.loc[~removed].reset_index(drop=True)
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    applications = cast(list[dict[str, object]], collection["fold_applications"])
    applications[1]["n_sender_rows"] = 2
    _reidentify_collection(result)

    with pytest.raises(ValueError, match="universe differs between samples"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy")


def test_crossfit_readback_rejects_sender_application_lineage_tamper() -> None:
    result = _result()
    result._sender.loc[0, "source_sender_application_id"] = "forged-application"

    with pytest.raises(ValueError, match="sender application lineage"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy")


def test_v4_readback_validates_calibrated_gain_without_pooled_claim() -> None:
    result = _v4_result()

    table, views = convert_crossfit_result_to_long(
        result,
        _adata(),
        dataset_id="toy-v4",
        method_version="test-version-v4",
    )

    assert tuple(table.columns) == LONG_TABLE_COLUMNS
    assert set(
        table.loc[table["interaction_id"].eq("interaction-observed"), "score"]
    ) == {0.4}
    assert len(views) == 1
    assert views[0]["source_manifest_schema_version"] == "4.0.0"
    assert views[0]["source_cross_receiver_percentile_rank_eligible"] is True
    assert views[0]["common_functional_claim"] is False
    assert views[0]["global_cross_receiver_endpoint_eligible"] is False
    assert views[0]["benchmark_scope"] == BENCHMARK_SCOPE
    assert views[0]["rank_scope"] == "within_sample_receiver_only"


def test_v5_readback_preserves_v4_lr_output_and_ignores_directional_rows() -> None:
    v4 = _v4_result()
    v5 = _v5_result()

    v4_table, v4_views = convert_crossfit_result_to_long(
        v4,
        _adata(),
        dataset_id="toy-calibrated",
        method_version="test-version-calibrated",
    )
    v5_table, v5_views = convert_crossfit_result_to_long(
        v5,
        _adata(),
        dataset_id="toy-calibrated",
        method_version="test-version-calibrated",
    )

    pd.testing.assert_frame_equal(v5_table, v4_table)
    assert v5.directional_read_count == 0
    assert v4_views[0]["source_manifest_schema_version"] == "4.0.0"
    assert v5_views[0]["source_manifest_schema_version"] == "5.0.0"
    v4_view = {**v4_views[0], "source_manifest_schema_version": "current"}
    v5_view = {**v5_views[0], "source_manifest_schema_version": "current"}
    assert v5_view == v4_view


def test_v4_readback_retains_legacy_raw_sender_multiplier_validation() -> None:
    result = _v4_result()
    result._sender.loc[0, "global_sender_lr_score"] = 0.41

    with pytest.raises(ValueError, match="raw-evidence multiplication"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")


def test_v6_readback_exports_conserved_global_sender_scores() -> None:
    result = _v6_result()

    table, views = convert_crossfit_result_to_long(
        result,
        _adata(),
        dataset_id="toy-v6",
        method_version="test-version-v6",
    )

    observed = table.loc[table["interaction_id"].eq("interaction-observed")]
    structural = table.loc[table["interaction_id"].eq("interaction-zero")]
    missing = table.loc[table["interaction_id"].eq("interaction-ne")]
    assert sorted(observed["score"].unique()) == pytest.approx([0.2, 0.6])
    assert structural["score"].eq(0.0).all()
    assert missing["score"].isna().all()
    assert set(table["score_name"]) == {"global_sender_lr_score"}
    assert views[0]["source_manifest_schema_version"] == "6.0.0"
    assert views[0]["score_version"] == (
        "receiver_gain_percentile_mechanistic_conserved_sender_v4"
    )
    assert views[0]["global_cross_receiver_endpoint_eligible"] is False


def test_v6_readback_preserves_lr_zero_and_ne_precedence() -> None:
    result = _v6_result()
    observed = result._sender["interaction_id"].eq("interaction-observed")
    structural = result._sender["interaction_id"].eq("interaction-zero")
    not_estimable = result._sender["interaction_id"].eq("interaction-ne")

    for mask in (observed, structural):
        result._sender.loc[
            mask,
            [
                "ligand_availability",
                "training_prevalence_prior",
                "raw_sender_evidence",
                "assignment_weight",
                "normalized_entropy",
            ],
        ] = None
    result._sender.loc[observed, "global_sender_lr_score"] = None
    result._sender.loc[observed, "status"] = "not_estimable"
    result._sender.loc[observed, "reason_code"] = "all_candidate_evidence_missing"

    entropy = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75)) / np.log(2.0)
    for sender, weight in (("Sender-A", 0.25), ("Sender-B", 0.75)):
        selected = not_estimable & result._sender["sender"].eq(sender)
        result._sender.loc[selected, "ligand_availability"] = weight
        result._sender.loc[selected, "training_prevalence_prior"] = 1.0
        result._sender.loc[selected, "raw_sender_evidence"] = weight
        result._sender.loc[selected, "assignment_weight"] = weight
        result._sender.loc[selected, "normalized_entropy"] = entropy

    table, _ = convert_crossfit_result_to_long(
        result,
        _adata(),
        dataset_id="toy-v6-precedence",
    )

    assert (
        table.loc[table["interaction_id"].eq("interaction-observed"), "status"]
        .eq("missing")
        .all()
    )
    assert (
        table.loc[table["interaction_id"].eq("interaction-zero"), "score"].eq(0.0).all()
    )
    assert (
        table.loc[table["interaction_id"].eq("interaction-ne"), "status"]
        .eq("missing")
        .all()
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("mutation", "message"),
    [
        ("weight_sum", "assignment weights do not sum to one"),
        ("partial_weight", "entirely numeric or entirely missing"),
        ("entropy", "normalized entropy disagrees"),
        ("score", "conserved LR allocation"),
        ("structural_status", "conserved LR allocation"),
    ],
)
def test_v6_readback_rejects_nonconserving_or_partial_sender_groups(
    mutation: str,
    message: str,
) -> None:
    result = _v6_result()
    observed = result._sender["interaction_id"].eq(
        "interaction-observed"
    ) & result._sender["sample_id"].eq("sample-1")
    first = result._sender.index[observed][0]
    if mutation == "weight_sum":
        result._sender.loc[first, "assignment_weight"] = 0.35
    elif mutation == "partial_weight":
        result._sender.loc[first, "assignment_weight"] = None
    elif mutation == "entropy":
        result._sender.loc[observed, "normalized_entropy"] = 0.1
    elif mutation == "score":
        result._sender.loc[first, "global_sender_lr_score"] = 0.3
    else:
        structural = result._sender["interaction_id"].eq("interaction-zero")
        result._sender.loc[result._sender.index[structural][0], "status"] = "observed"
        result._sender.loc[result._sender.index[structural][0], "reason_code"] = None

    with pytest.raises(ValueError, match=message):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v6")


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field", "value", "message"),
    [
        ("sender_policy", "forged-policy", "calibration policies"),
        (
            "functional_schema_version",
            "3.0.0",
            "score version or functional schema",
        ),
    ],
)
def test_v6_readback_rejects_old_or_forged_sender_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    result = _v6_result()
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    application = cast(list[dict[str, object]], collection["fold_applications"])[0]
    application[field] = value
    _reidentify_collection(result)

    with pytest.raises(ValueError, match=message):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v6")


def test_v6_readback_rejects_legacy_score_version() -> None:
    result = _v6_result()
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    collection["score_version"] = "receiver_gain_percentile_mechanistic_strength_v3"
    _reidentify_collection(result)

    with pytest.raises(ValueError, match="score version or functional schema"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v6")


def test_v4_readback_rejects_incomplete_binding_coverage() -> None:
    result = _v4_result()
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    application = cast(list[dict[str, object]], collection["fold_applications"])[0]
    application["receiver_gain_calibration_bindings"] = []
    _reidentify_collection(result)

    with pytest.raises(ValueError, match="exact receiver coverage"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")


def test_v4_readback_rejects_binding_identity_tamper() -> None:
    result = _v4_result()
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    application = cast(list[dict[str, object]], collection["fold_applications"])[0]
    binding = cast(
        list[dict[str, object]],
        application["receiver_gain_calibration_bindings"],
    )[0]
    binding["gain_calibration_artifact_id"] = "forged-artifact"
    _reidentify_collection(result)

    with pytest.raises(ValueError, match="binding identity"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("field", "value", "message"),
    [
        ("calibration_policy", "forged-policy", "calibration policies"),
        (
            "cross_receiver_percentile_rank_eligible",
            False,
            "eligibility flags",
        ),
    ],
)
def test_v4_readback_rejects_policy_or_eligibility_tamper(
    field: str, value: object, message: str
) -> None:
    result = _v4_result()
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    application = cast(list[dict[str, object]], collection["fold_applications"])[0]
    application[field] = value
    _reidentify_collection(result)

    with pytest.raises(ValueError, match=message):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")


def test_v4_readback_rejects_sender_binding_tamper() -> None:
    result = _v4_result()
    result._sender.loc[0, "gain_calibration_binding_id"] = "forged-binding"

    with pytest.raises(ValueError, match="binding_id disagrees"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")


def test_v4_readback_rejects_calibrated_percentile_tamper() -> None:
    result = _v4_result()
    result._lr.loc[0, "calibrated_family_gain_percentile"] = 0.7

    with pytest.raises(ValueError, match="calibrated_family_gain_percentile"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")


def test_v4_readback_forbids_retired_receiver_scale_factor_column() -> None:
    result = _v4_result()
    result._sender.insert(24, "training_receiver_scale_factor", 0.8)

    with pytest.raises(ValueError, match=r"4\.0\.0 table schema"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")


def test_v4_readback_forbids_retired_manifest_scale_factors() -> None:
    result = _v4_result()
    collection = cast(
        list[dict[str, object]], result._manifest["contrast_common_collections"]
    )[0]
    application = cast(list[dict[str, object]], collection["fold_applications"])[0]
    application["receiver_scale_factors"] = [["Receiver", 0.8]]
    _reidentify_collection(result)

    with pytest.raises(ValueError, match="retired scale-factor fields"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")


def test_v4_readback_distinguishes_missing_parent_score_from_zero() -> None:
    result = _v4_result()
    row = result._sender["interaction_id"].eq("interaction-ne")
    result._sender.loc[row.idxmax(), "global_lr_score"] = 0.0

    with pytest.raises(ValueError, match="global_lr_score disagrees with LR parent"):
        convert_crossfit_result_to_long(result, _adata(), dataset_id="toy-v4")
