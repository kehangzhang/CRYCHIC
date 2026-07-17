from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from crychic.core import ContractError, canonical_json
from crychic.signatures import (
    COMPONENT_RECORD_COLUMNS,
    FittedSignatureComponentRecord,
    LayeredSignatureStatus,
    LayeredSignatureTable,
    SignatureComponentLevel,
    SignatureFeatureKind,
    build_layered_signature_table,
)


def _component(
    level: SignatureComponentLevel | str,
    source_record_id: str,
    contribution: float | None,
    *,
    feature_id: str = "G1",
    family_id: str = "F1",
    interaction_id: str | None = None,
    sender: str | None = None,
    observed: float | None = 3.0,
    predicted: float | None = 2.0,
    residual: float | None = 1.0,
    entropy: float = 0.2,
    response_status: LayeredSignatureStatus | str = LayeredSignatureStatus.OK,
    component_status: LayeredSignatureStatus | str = LayeredSignatureStatus.OK,
    reason_code: str | None = None,
) -> FittedSignatureComponentRecord:
    return FittedSignatureComponentRecord(
        component_level=level,  # type: ignore[arg-type]
        source_record_id=source_record_id,
        context_id="treated",
        context_json=canonical_json({"condition": "treated", "tissue": "blood"}),
        receiver="NK_cell",
        contrast="treated-v-control",
        feature_id=feature_id,
        feature_kind=SignatureFeatureKind.GENE,
        feature_namespace="HGNC_SYMBOL",
        family_id=family_id,
        interaction_id=interaction_id,
        sender=sender,
        observed=observed,
        predicted=predicted,
        residual=residual,
        attributed_contribution=contribution,
        response_status=response_status,  # type: ignore[arg-type]
        component_status=component_status,  # type: ignore[arg-type]
        reason_code=reason_code,
        within_family_entropy=entropy,
        uncertainty=0.1,
        uncertainty_kind="standard_error",
        source_artifact_id="fitted_components_v1",
        scoring_functional_id="common_functional_v2",
        fold_id="fold_1",
        repeat_id="repeat_1",
        provenance_json=canonical_json(
            {"config_digest": "cfg_123", "run_id": "run_123"}
        ),
    )


def _complete_components() -> tuple[
    list[FittedSignatureComponentRecord],
    list[FittedSignatureComponentRecord],
    list[FittedSignatureComponentRecord],
]:
    families = [
        _component("family", "family_g1_f1", 1.2),
        _component("family", "family_g1_f2", 0.8, family_id="F2"),
        _component(
            "family",
            "family_g2_f1",
            -1.0,
            feature_id="G2",
            observed=-2.0,
            predicted=-1.0,
            residual=-1.0,
        ),
    ]
    lrs = [
        _component("lr", "lr_g1_1", 0.7, interaction_id="L1_R1"),
        _component("lr", "lr_g1_2", 0.5, interaction_id="L2_R2"),
        _component(
            "lr",
            "lr_g1_3",
            0.8,
            family_id="F2",
            interaction_id="L3_R3",
        ),
        _component(
            "lr",
            "lr_g2_1",
            -1.0,
            feature_id="G2",
            interaction_id="L1_R1",
            observed=-2.0,
            predicted=-1.0,
            residual=-1.0,
        ),
    ]
    senders = [
        _component(
            "sender_lr",
            "sender_g1_1a",
            0.4,
            interaction_id="L1_R1",
            sender="T_cell",
        ),
        _component(
            "sender_lr",
            "sender_g1_1b",
            0.3,
            interaction_id="L1_R1",
            sender="B_cell",
        ),
        _component(
            "sender_lr",
            "sender_g1_2a",
            0.5,
            interaction_id="L2_R2",
            sender="T_cell",
        ),
        _component(
            "sender_lr",
            "sender_g1_3a",
            0.8,
            family_id="F2",
            interaction_id="L3_R3",
            sender="Myeloid",
        ),
        _component(
            "sender_lr",
            "sender_g2_1a",
            -1.0,
            feature_id="G2",
            interaction_id="L1_R1",
            sender="T_cell",
            observed=-2.0,
            predicted=-1.0,
            residual=-1.0,
        ),
    ]
    return families, lrs, senders


def test_three_signature_grains_preserve_full_signed_response_and_direction() -> None:
    families, lrs, senders = _complete_components()
    table = build_layered_signature_table(families, lrs, senders)

    receiver = table.receiver_context.set_index("feature_id")
    assert receiver.loc["G1", "observed"] == 3.0
    assert receiver.loc["G1", "predicted"] == 2.0
    assert receiver.loc["G1", "residual"] == 1.0
    assert receiver.loc["G2", "observed"] == -2.0
    assert receiver.loc["G2", "predicted"] == -1.0
    assert receiver.loc["G2", "residual"] == -1.0
    assert receiver.loc["G2", "direction_agreement"] == "agrees"

    negative_lr = table.lr_attributed.loc[
        table.lr_attributed["feature_id"].eq("G2")
    ].iloc[0]
    assert negative_lr["attributed_contribution"] == -1.0
    assert negative_lr["direction_consistent_attributed"] == -1.0
    assert negative_lr["direction_agreement"] == "agrees"
    assert set(table.sender_lr_receiver["sender"]) == {
        "B_cell",
        "Myeloid",
        "T_cell",
    }
    assert not table.inference_eligible
    assert table.post_fit_only
    assert len(table.query_lr(interaction_id="L1_R1")) == 2
    assert len(table.query_sender_lr(sender="T_cell")) == 3


def test_stable_ids_and_ranks_do_not_depend_on_input_order() -> None:
    families, lrs, senders = _complete_components()
    forward = build_layered_signature_table(families, lrs, senders)
    reverse = build_layered_signature_table(
        list(reversed(families)), list(reversed(lrs)), list(reversed(senders))
    )

    for left, right in (
        (forward.receiver_context, reverse.receiver_context),
        (forward.lr_attributed, reverse.lr_attributed),
        (forward.sender_lr_receiver, reverse.sender_lr_receiver),
    ):
        assert left["signature_id"].tolist() == right["signature_id"].tolist()
        assert left["signature_rank"].tolist() == right["signature_rank"].tolist()

    lr1 = forward.lr_attributed.loc[
        forward.lr_attributed["interaction_id"].eq("L1_R1")
    ].set_index("feature_id")
    assert lr1.loc["G2", "signature_rank"] == 1
    assert lr1.loc["G1", "signature_rank"] == 2


def test_high_family_entropy_fail_closes_lr_and_sender_but_not_receiver() -> None:
    family = _component("family", "family_high", 2.0, entropy=0.95)
    lrs = [
        _component("lr", "lr_high_1", 1.0, interaction_id="L1_R1", entropy=0.95),
        _component("lr", "lr_high_2", 1.0, interaction_id="L2_R2", entropy=0.95),
    ]
    senders = [
        _component(
            "sender_lr",
            "sender_high_1",
            1.0,
            interaction_id="L1_R1",
            sender="T_cell",
            entropy=0.95,
        ),
        _component(
            "sender_lr",
            "sender_high_2",
            1.0,
            interaction_id="L2_R2",
            sender="B_cell",
            entropy=0.95,
        ),
    ]
    table = build_layered_signature_table([family], lrs, senders, entropy_threshold=0.8)

    receiver = table.receiver_context.iloc[0]
    assert receiver["predicted"] == 2.0
    assert receiver["residual"] == 1.0
    assert receiver["status"] == LayeredSignatureStatus.OK.value
    assert receiver["signature_rank"] == 1
    for frame in (table.lr_attributed, table.sender_lr_receiver):
        assert set(frame["status"]) == {
            LayeredSignatureStatus.FAMILY_HIGH_ENTROPY.value
        }
        assert frame["attributed_contribution"].isna().all()
        assert frame["signature_rank"].isna().all()
        assert frame["uncertainty"].eq(0.1).all()
        assert set(frame["reason_code"]) == {
            "within_family_entropy_at_or_above_threshold"
        }


def test_not_estimable_component_is_missing_and_never_ranked() -> None:
    family = _component("family", "family_ok", 2.0)
    lr_ne = _component(
        "lr",
        "lr_ne",
        None,
        interaction_id="L1_R1",
        component_status=LayeredSignatureStatus.NOT_ESTIMABLE,
        reason_code="member_allocation_unresolved",
    )
    sender_ne = _component(
        "sender_lr",
        "sender_ne",
        None,
        interaction_id="L1_R1",
        sender="T_cell",
        component_status=LayeredSignatureStatus.NOT_ESTIMABLE,
        reason_code="member_allocation_unresolved",
    )
    table = build_layered_signature_table([family], [lr_ne], [sender_ne])

    assert pd.isna(table.lr_attributed.iloc[0]["attributed_contribution"])
    assert pd.isna(table.lr_attributed.iloc[0]["signature_rank"])
    assert pd.isna(table.sender_lr_receiver.iloc[0]["signature_rank"])
    assert table.receiver_context.iloc[0]["component_coverage_status"] == "complete"

    with pytest.raises(ContractError, match="Do not encode"):
        replace(lr_ne, attributed_contribution=0.0)


def test_parent_child_conservation_and_parent_status_are_enforced() -> None:
    family = _component("family", "family_parent", 2.0)
    incomplete_lr = _component("lr", "lr_incomplete", 1.0, interaction_id="L1_R1")
    with pytest.raises(ContractError) as conservation:
        build_layered_signature_table([family], [incomplete_lr], [])
    assert conservation.value.details.code == "invalid_layered_signature_conservation"

    unavailable_family = _component(
        "family",
        "family_ne",
        None,
        component_status=LayeredSignatureStatus.NOT_ESTIMABLE,
        reason_code="family_not_identifiable",
    )
    estimable_lr = _component("lr", "lr_child", 2.0, interaction_id="L1_R1")
    with pytest.raises(ContractError) as status_error:
        build_layered_signature_table([unavailable_family], [estimable_lr], [])
    assert status_error.value.details.code == "invalid_layered_signature_status"


def test_dataframe_contract_preserves_uncertainty_and_provenance() -> None:
    family = _component("family", "family_df", 2.0)
    lr = _component("lr", "lr_df", 2.0, interaction_id="L1_R1")
    sender = _component(
        "sender_lr",
        "sender_df",
        2.0,
        interaction_id="L1_R1",
        sender="T_cell",
    )

    def frame(record: FittedSignatureComponentRecord) -> pd.DataFrame:
        row = {column: getattr(record, column) for column in COMPONENT_RECORD_COLUMNS}
        row["context_json"] = {"tissue": "blood", "condition": "treated"}
        row["provenance_json"] = {
            "run_id": "run_123",
            "config_digest": "cfg_123",
        }
        return pd.DataFrame([row])

    table = build_layered_signature_table(frame(family), frame(lr), frame(sender))
    assert table.lr_attributed.iloc[0]["uncertainty"] == pytest.approx(0.1)
    assert table.lr_attributed.iloc[0]["provenance_json"] == lr.provenance_json

    forbidden = frame(lr)
    forbidden["comm_probability"] = 0.9
    with pytest.raises(ContractError) as error:
        build_layered_signature_table(frame(family), forbidden, frame(sender))
    assert error.value.details.code == "forbidden_layered_signature_inference"

    with pytest.raises(ContractError) as nested_error:
        replace(
            sender,
            provenance_json=canonical_json({"posterior_probability": 0.8}),
        )
    assert nested_error.value.details.code == "forbidden_layered_signature_inference"


def test_parquet_tables_roundtrip_without_inferential_fields(tmp_path: Path) -> None:
    families, lrs, senders = _complete_components()
    table = build_layered_signature_table(families, lrs, senders)
    receiver_path = tmp_path / "receiver.parquet"
    lr_path = tmp_path / "lr.parquet"
    sender_path = tmp_path / "sender.parquet"
    table.receiver_context.to_parquet(receiver_path, index=False)
    table.lr_attributed.to_parquet(lr_path, index=False)
    table.sender_lr_receiver.to_parquet(sender_path, index=False)

    restored = LayeredSignatureTable(
        receiver_context=pd.read_parquet(receiver_path),
        lr_attributed=pd.read_parquet(lr_path),
        sender_lr_receiver=pd.read_parquet(sender_path),
        entropy_threshold=table.entropy_threshold,
    )
    pdt.assert_frame_equal(
        restored.receiver_context,
        table.receiver_context,
        check_dtype=False,
    )
    pdt.assert_frame_equal(
        restored.lr_attributed, table.lr_attributed, check_dtype=False
    )
    pdt.assert_frame_equal(
        restored.sender_lr_receiver,
        table.sender_lr_receiver,
        check_dtype=False,
    )
    forbidden_names = {"p", "p_value", "q", "q_value", "comm_probability"}
    for output in (
        restored.receiver_context,
        restored.lr_attributed,
        restored.sender_lr_receiver,
    ):
        assert not forbidden_names.intersection(output.columns)
