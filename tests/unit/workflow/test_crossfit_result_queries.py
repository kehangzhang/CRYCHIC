from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crychic.workflow import CrossFitResult
from crychic.workflow.crossfit_persistence import (
    CROSSFIT_COMPONENT_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_LR_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_LR_TABLE,
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
    CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE,
    CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS,
    CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE,
)


def _component_table() -> pd.DataFrame:
    base: dict[str, object] = {column: None for column in CROSSFIT_COMPONENT_COLUMNS}
    rows: list[dict[str, object]] = []
    for scope, component, interaction, sender, value, status in (
        ("family", "integrated_lr_score", None, None, 0.4, "observed"),
        ("lr_member", "sender_unresolved_strength", "L1_R1", None, 0.4, "observed"),
        (
            "sender_lr_member",
            "sender_resolved_strength",
            "L1_R1",
            "T_cell",
            0.25,
            "observed",
        ),
        (
            "sender_lr_member",
            "sender_resolved_strength",
            "L1_R1",
            "B_cell",
            None,
            "not_estimable",
        ),
    ):
        rows.append(
            {
                **base,
                "crossfit_id": "crossfit-1",
                "spec_id": "spec-1",
                "repeat_id": "repeat-1",
                "fold_id": "fold-1",
                "contrast_id": "contrast-1",
                "contrast": "stim_vs_control",
                "sample_id": "sample-1",
                "subject_id": "subject-1",
                "context_id": "stim",
                "receiver": "Receiver",
                "family_id": "family-1",
                "interaction_id": interaction,
                "mode": "state",
                "sender": sender,
                "component_scope": scope,
                "component": component,
                "component_value": value,
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=CROSSFIT_COMPONENT_COLUMNS)


def _contrast_common_tables() -> dict[str, pd.DataFrame]:
    lr_base: dict[str, object] = {
        column: None for column in CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
    }
    sender_base: dict[str, object] = {
        column: None for column in CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS
    }
    lr_rows: list[dict[str, object]] = []
    sender_rows: list[dict[str, object]] = []
    for sample, context, receiver, value, status in (
        ("sample-1", "stim", "ReceiverA", 0.8, "observed"),
        ("sample-2", "control", "ReceiverB", None, "not_estimable"),
    ):
        common = {
            "crossfit_id": "crossfit-1",
            "spec_id": "spec-1",
            "repeat_id": "repeat-1",
            "fold_id": "fold-1",
            "contrast_id": "contrast-1",
            "contrast": "stim_vs_control",
            "contrast_common_collection_id": "collection-1",
            "global_common_application_id": "global-application-1",
            "global_common_functional_id": "global-functional-1",
            "source_family_common_functional_id": f"functional-{receiver}",
            "source_family_common_application_id": f"application-{receiver}",
            "sample_id": sample,
            "subject_id": f"subject-{sample}",
            "context_id": context,
            "receiver": receiver,
            "family_id": "family-1",
            "driver_id": "R1",
            "interaction_id": "L1_R1",
            "mode": "state",
            "gain_calibration_binding_id": f"gain-binding-{receiver}",
            "gain_calibration_artifact_id": (
                "gain-artifact-1" if receiver == "ReceiverA" else None
            ),
            "gain_calibration_status": (
                "observed" if receiver == "ReceiverA" else "not_estimable"
            ),
            "gain_calibration_reason_code": (
                None if receiver == "ReceiverA" else "insufficient_inner_oof_subjects"
            ),
            "status": status,
            "score_version": "global-score-v1",
        }
        lr_rows.append(
            {
                **lr_base,
                **common,
                "receiver_relative_family_gain": value,
                "calibrated_family_gain_percentile": value,
                "global_lr_score": value,
            }
        )
        sender_rows.append(
            {
                **sender_base,
                **common,
                "source_sender_functional_id": "sender-functional-1",
                "source_sender_application_id": "sender-application-1",
                "sender": "T_cell" if receiver == "ReceiverA" else "B_cell",
                "global_lr_score": value,
                "ligand_availability": 0.5 if value is not None else None,
                "training_prevalence_prior": 0.5 if value is not None else None,
                "raw_sender_evidence": 0.25 if value is not None else None,
                "assignment_weight": 1.0 if value is not None else None,
                "normalized_entropy": 0.0 if value is not None else None,
                "global_sender_lr_score": value,
            }
        )
    return {
        CROSSFIT_CONTRAST_COMMON_LR_TABLE: pd.DataFrame(
            lr_rows, columns=CROSSFIT_CONTRAST_COMMON_LR_COLUMNS
        ),
        CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE: pd.DataFrame(
            sender_rows,
            columns=CROSSFIT_CONTRAST_COMMON_SENDER_LR_COLUMNS,
        ),
    }


def _directional_registry_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (
        pair_spec_id,
        binding_id,
        fold_id,
        receiver,
        status,
        reason_code,
    ) in (
        ("pair-1", "binding-1", "fold-1", "ReceiverA", "observed", None),
        (
            "pair-2",
            "binding-2",
            "fold-2",
            "ReceiverB",
            "not_estimable",
            "directional_components_not_estimable",
        ),
    ):
        for role, channel in (
            ("forward", "increased_activation_compatible"),
            ("reverse", "reduced_activation_compatible"),
        ):
            row: dict[str, object] = {
                column: None for column in CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS
            }
            row.update(
                {
                    "crossfit_id": "crossfit-1",
                    "spec_id": "spec-1",
                    "repeat_id": "repeat-1",
                    "fold_id": fold_id,
                    "receiver": receiver,
                    "pair_spec_id": pair_spec_id,
                    "binding_id": binding_id,
                    "channel_role": role,
                    "channel": channel,
                    "contrast_id": f"contrast-{role}-{pair_spec_id}",
                    "contrast": f"contrast_{role}_{pair_spec_id}",
                    "response_id": f"response-{role}-{binding_id}",
                    "training_artifact_id": f"model-{role}-{binding_id}",
                    "application_id": f"application-{role}-{binding_id}",
                    "response_pair_id": (
                        "response-pair-1" if status == "observed" else None
                    ),
                    "response_channel_id": (
                        f"response-channel-{role}-1" if status == "observed" else None
                    ),
                    "status": status,
                    "reason_code": reason_code,
                    "combination_rule": ("independent_contrast_views_not_additive_v1"),
                    "active_inhibition_allowed": False,
                    "supports_active_inhibition_claim": False,
                    "paired_score_comparison_allowed": False,
                    "formal_inference_allowed": False,
                    "certification_status": "uncertified",
                    "is_oof_certified": False,
                    "formal_inference_status": "not_available_descriptive_only",
                    "claim_scope": (
                        "heldout_family_common_diagnostic_not_complete_oof_certified"
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=CROSSFIT_DIRECTIONAL_CHANNEL_COLUMNS)


@pytest.fixture  # type: ignore[untyped-decorator]
def result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CrossFitResult:
    value = CrossFitResult._from_validated(tmp_path, {})
    table = _component_table()
    monkeypatch.setattr(
        CrossFitResult,
        "read_components",
        lambda self: table.copy(deep=True),
    )
    return value


@pytest.fixture  # type: ignore[untyped-decorator]
def contrast_common_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> CrossFitResult:
    value = CrossFitResult._from_validated(tmp_path, {})
    tables = _contrast_common_tables()
    monkeypatch.setattr(
        CrossFitResult,
        "read_table",
        lambda self, name: tables[name].copy(deep=True),
    )
    return value


@pytest.fixture  # type: ignore[untyped-decorator]
def directional_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> CrossFitResult:
    value = CrossFitResult._from_validated(tmp_path, {})
    table = _directional_registry_table()
    monkeypatch.setattr(
        CrossFitResult,
        "read_directional_channel_registry",
        lambda self: table.copy(deep=True),
    )
    return value


@pytest.fixture  # type: ignore[untyped-decorator]
def integrated_lr_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> CrossFitResult:
    components = _component_table()
    directional_member = components.loc[
        components["component_scope"].eq("lr_member")
    ].copy(deep=True)
    directional_member["fold_id"] = "fold-1"
    directional_member["contrast_id"] = "contrast-forward-pair-1"
    directional_member["contrast"] = "contrast_forward_pair-1"
    directional_member["component_value"] = 0.9
    directional_member["source_row_id"] = "directional-source-row"
    components = pd.concat([components, directional_member], ignore_index=True)
    registry = _directional_registry_table()
    value = CrossFitResult._from_validated(
        tmp_path,
        {"tables": {CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE: {}}},
    )
    monkeypatch.setattr(
        CrossFitResult,
        "read_components",
        lambda self: components.copy(deep=True),
    )
    monkeypatch.setattr(
        CrossFitResult,
        "read_directional_channel_registry",
        lambda self: registry.copy(deep=True),
    )
    return value


def test_grain_queries_use_noncompensating_terminal_components(
    result: CrossFitResult,
) -> None:
    family = result.query_family_scores(context_id="stim", receiver="Receiver")
    lr = result.query_lr_pairs(interaction_id="L1_R1", mode="state")
    sender = result.query_sender_lr_pairs(sender="T_cell")

    assert family["component"].tolist() == ["integrated_lr_score"]
    assert lr["component"].tolist() == ["sender_unresolved_strength"]
    assert sender["component"].tolist() == ["sender_resolved_strength"]
    assert sender["component_value"].tolist() == [0.25]


def test_canonical_integrated_lr_query_fixes_component_and_excludes_directional(
    integrated_lr_result: CrossFitResult,
) -> None:
    low_level = integrated_lr_result.query_lr_pairs(mode="state")
    canonical = integrated_lr_result.query_integrated_lr_scores(mode="state")

    assert set(low_level["contrast"]) == {
        "stim_vs_control",
        "contrast_forward_pair-1",
    }
    assert canonical["contrast"].tolist() == ["stim_vs_control"]
    assert canonical["integrated_lr_score"].tolist() == [0.4]
    assert canonical["source_component"].tolist() == [
        "sender_unresolved_strength"
    ]
    assert canonical["semantic_output"].tolist() == ["integrated_lr_score"]
    assert canonical["crossfit_spec_id"].tolist() == canonical["spec_id"].tolist()
    assert canonical["directional_contrasts_excluded"].eq(True).all()
    assert canonical["formal_inference_allowed"].eq(False).all()


def test_canonical_integrated_lr_query_keeps_legacy_non_directional_rows(
    result: CrossFitResult,
) -> None:
    canonical = result.query_integrated_lr_scores(
        interaction_id="L1_R1",
        status="observed",
    )

    assert canonical["integrated_lr_score"].tolist() == [0.4]
    first = result.query_integrated_lr_scores()
    first.loc[:, "integrated_lr_score"] = 999.0
    assert 999.0 not in result.query_integrated_lr_scores()[
        "integrated_lr_score"
    ].tolist()


def test_queries_preserve_not_estimable_rows_unless_status_is_requested(
    result: CrossFitResult,
) -> None:
    all_rows = result.query_sender_lr_pairs(interaction_id="L1_R1")
    observed = result.query_sender_lr_pairs(interaction_id="L1_R1", status="observed")

    assert set(all_rows["status"]) == {"observed", "not_estimable"}
    assert observed["sender"].tolist() == ["T_cell"]


def test_query_rejects_wrong_grain_or_invalid_filters(
    result: CrossFitResult,
) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        result.query_lr_pairs(component="integrated_lr_score")
    with pytest.raises(ValueError, match="mode must"):
        result.query_family_scores(mode="invalid")
    with pytest.raises(ValueError, match="canonical"):
        result.query_sender_lr_pairs(receiver=" Receiver")


def test_contrast_common_queries_filter_without_dropping_ne_rows(
    contrast_common_result: CrossFitResult,
) -> None:
    all_lr = contrast_common_result.query_contrast_common_lr_scores(
        interaction_id="L1_R1"
    )
    observed = contrast_common_result.query_contrast_common_lr_pairs(status="observed")
    sender = contrast_common_result.query_contrast_common_sender_lr_pairs(
        sender="T_cell",
        contrast_common_collection_id="collection-1",
    )

    assert set(all_lr["status"]) == {"observed", "not_estimable"}
    assert observed["receiver"].tolist() == ["ReceiverA"]
    assert observed["calibrated_family_gain_percentile"].tolist() == [0.8]
    assert observed["gain_calibration_artifact_id"].tolist() == ["gain-artifact-1"]
    assert sender["global_sender_lr_score"].tolist() == [0.8]


def test_contrast_common_queries_are_defensive_and_validate_filters(
    contrast_common_result: CrossFitResult,
) -> None:
    first = contrast_common_result.query_contrast_common_sender_lr_scores()
    first.loc[:, "global_sender_lr_score"] = 999.0
    repeated = contrast_common_result.query_contrast_common_sender_lr_scores()

    assert 999.0 not in repeated["global_sender_lr_score"].tolist()
    with pytest.raises(ValueError, match="canonical"):
        contrast_common_result.query_contrast_common_lr_scores(receiver=" ReceiverA")
    with pytest.raises(ValueError, match="mode must"):
        contrast_common_result.query_contrast_common_sender_lr_scores(mode="invalid")


def test_directional_registry_queries_pair_fold_receiver_channel_and_status(
    directional_result: CrossFitResult,
) -> None:
    pair = directional_result.query_directional_channels(pair_spec_id="pair-1")
    fold = directional_result.query_directional_channels(fold_id="fold-2")
    receiver = directional_result.query_directional_channels(receiver="ReceiverA")
    forward = directional_result.query_directional_channels(
        channel="increased_activation_compatible"
    )
    not_estimable = directional_result.query_directional_channel_registry(
        status="not_estimable"
    )

    assert pair["channel_role"].tolist() == ["forward", "reverse"]
    assert set(fold["binding_id"]) == {"binding-2"}
    assert set(receiver["binding_id"]) == {"binding-1"}
    assert forward["channel_role"].tolist() == ["forward", "forward"]
    assert set(not_estimable["receiver"]) == {"ReceiverB"}
    assert not_estimable["reason_code"].notna().all()


def test_directional_registry_query_is_defensive_and_validates_filters(
    directional_result: CrossFitResult,
) -> None:
    first = directional_result.query_directional_channels()
    first.loc[:, "receiver"] = "poisoned"
    repeated = directional_result.query_directional_channels()

    assert "poisoned" not in repeated["receiver"].tolist()
    with pytest.raises(ValueError, match="canonical"):
        directional_result.query_directional_channels(receiver=" ReceiverA")
    with pytest.raises(ValueError, match="channel must"):
        directional_result.query_directional_channels(channel="activation")
    with pytest.raises(ValueError, match="status must"):
        directional_result.query_directional_channels(status="structural_zero")


def test_legacy_result_does_not_invent_contrast_common_tables(
    tmp_path: Path,
) -> None:
    legacy = CrossFitResult._from_validated(
        tmp_path,
        {"schema_version": "2.0.0", "tables": {}},
    )

    with pytest.raises(KeyError, match=CROSSFIT_CONTRAST_COMMON_LR_TABLE):
        legacy.read_contrast_common_lr_scores()
    with pytest.raises(KeyError, match=CROSSFIT_CONTRAST_COMMON_SENDER_LR_TABLE):
        legacy.read_contrast_common_sender_lr_scores()

    v4 = CrossFitResult._from_validated(
        tmp_path,
        {"schema_version": "4.0.0", "tables": {}},
    )
    with pytest.raises(KeyError, match=CROSSFIT_DIRECTIONAL_CHANNEL_REGISTRY_TABLE):
        v4.read_directional_channel_registry()
