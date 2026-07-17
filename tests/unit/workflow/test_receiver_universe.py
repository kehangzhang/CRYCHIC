from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import crychic.workflow.training as training_module
from crychic.core import ContractError, CrychicConfig
from crychic.workflow import (
    FrozenReceiverUniverse,
    ReceiverTrainingSupportRecord,
    ReceiverTrainingSupportStatus,
    ReceiverUniverseReuseBinding,
    ReceiverUniverseSourcePolicy,
    assess_receiver_training_support,
    bind_reused_receiver_universe,
    freeze_receiver_universe,
)
from crychic.workflow.training import SanitizedRawInputIdentity


def _root_identity(
    *,
    receiver_ids: tuple[str, ...] = ("ReceiverB", "ReceiverA"),
    random_seed: int = 0,
) -> SanitizedRawInputIdentity:
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    row_names: list[str] = []
    for subject_index, subject_id in enumerate(("subject-1", "subject-2")):
        for condition in ("control", "treated"):
            sample_id = f"{subject_id}:{condition}"
            values = {
                "ReceiverB": [2 + subject_index, 7],
                "ReceiverA": [8, 1 + subject_index],
            }
            for cell_type in receiver_ids:
                counts = values[cell_type]
                rows.append(counts)
                metadata.append(
                    {
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "cell_type": cell_type,
                        "condition": condition,
                    }
                )
                row_names.append(f"{sample_id}:{cell_type}")
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int64))
    adata = AnnData(
        X=sparse.csr_matrix(counts.shape, dtype=float),
        obs=pd.DataFrame(metadata, index=row_names),
        var=pd.DataFrame(index=("GeneA", "GeneB")),
    )
    adata.layers["counts"] = counts
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=random_seed,
    )
    return training_module._sanitized_raw_input_identity(adata, config)


def test_default_universe_is_producer_owned_sorted_and_order_invariant() -> None:
    root = _root_identity()
    first = freeze_receiver_universe(
        root,
        ("ReceiverB", "ReceiverA"),
    )
    second = freeze_receiver_universe(
        root,
        ("ReceiverA", "ReceiverB"),
    )

    assert first.receiver_ids == ("ReceiverA", "ReceiverB")
    assert first.receiver_axis_id == second.receiver_axis_id
    assert first.observed_cell_type_ids == ("ReceiverA", "ReceiverB")
    assert first.universe_id == second.universe_id
    assert first.source_policy is (
        ReceiverUniverseSourcePolicy.ROOT_OBSERVED_CONTEXT_LABEL_BLIND_V1
    )
    assert first.to_dict()["source_policy"] == (
        "root_observed_cell_type_ids_context_label_blind_v1"
    )
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenReceiverUniverse()
    with pytest.raises(ValueError, match="unique"):
        freeze_receiver_universe(root, ("ReceiverA", "ReceiverA"))


def test_explicit_universe_preserves_absent_and_rejects_unknown_observed() -> None:
    root = _root_identity()
    universe = freeze_receiver_universe(
        root,
        ("ReceiverB", "ReceiverA"),
        predeclared_receiver_ids=("ReceiverC", "ReceiverA", "ReceiverB"),
    )

    assert universe.receiver_ids == ("ReceiverA", "ReceiverB", "ReceiverC")
    assert universe.observed_cell_type_ids == ("ReceiverA", "ReceiverB")
    assert universe.source_policy is (
        ReceiverUniverseSourcePolicy.EXPLICIT_PREDECLARED_V1
    )
    with pytest.raises(ContractError) as caught:
        freeze_receiver_universe(
            root,
            ("ReceiverA", "UnknownReceiver"),
            predeclared_receiver_ids=("ReceiverA", "ReceiverB"),
        )
    assert caught.value.details.code == (
        "observed_cell_type_outside_predeclared_receiver_universe"
    )


def test_universe_binds_exact_root_lineage_and_fails_closed_on_tampering() -> None:
    root = _root_identity()
    universe = freeze_receiver_universe(root, ("ReceiverA", "ReceiverB"))

    assert universe.root_input_identity_id == root.identity_id
    assert universe.root_input_digest == root.input_digest
    assert universe.root_config_digest == root.config_digest
    assert universe.root_subject_ids == root.subject_ids

    object.__setattr__(universe, "root_input_digest", "0" * 64)
    with pytest.raises(ContractError) as caught:
        universe.to_dict()
    assert caught.value.details.code == (
        "frozen_receiver_universe_integrity_violation"
    )


def test_resample_reuse_binding_preserves_axis_across_distinct_roots() -> None:
    source = freeze_receiver_universe(
        _root_identity(),
        ("ReceiverA", "ReceiverB"),
    )
    child = freeze_receiver_universe(
        _root_identity(receiver_ids=("ReceiverA",), random_seed=7),
        ("ReceiverA",),
        predeclared_receiver_ids=source.receiver_ids,
    )

    binding = bind_reused_receiver_universe(
        source,
        child,
        operation="subject_bootstrap",
        plan_id="bootstrap-1",
        resample_index=1,
        materialized_input_id="materialized-bootstrap-1",
    )

    assert source.universe_id != child.universe_id
    assert source.receiver_axis_id == child.receiver_axis_id
    assert binding.receiver_ids == ("ReceiverA", "ReceiverB")
    assert binding.source_receiver_universe_id == source.universe_id
    assert binding.child_receiver_universe_id == child.universe_id
    assert binding.to_dict()["receiver_axis_id"] == source.receiver_axis_id
    with pytest.raises(TypeError, match="producer-owned"):
        ReceiverUniverseReuseBinding()

    shrunken = freeze_receiver_universe(
        _root_identity(receiver_ids=("ReceiverA",), random_seed=11),
        ("ReceiverA",),
    )
    with pytest.raises(ContractError) as mismatch:
        bind_reused_receiver_universe(
            source,
            shrunken,
            operation="subject_bootstrap",
            plan_id="bootstrap-2",
            resample_index=2,
            materialized_input_id="materialized-bootstrap-2",
        )
    assert mismatch.value.details.code == "receiver_universe_reuse_axis_mismatch"

    object.__setattr__(binding, "materialized_input_id", "forged-input")
    with pytest.raises(ContractError) as tampered:
        binding.to_dict()
    assert tampered.value.details.code == (
        "receiver_universe_reuse_binding_integrity_violation"
    )


def test_training_support_keeps_missing_receiver_as_typed_not_estimable() -> None:
    root = _root_identity()
    universe = freeze_receiver_universe(
        root,
        ("ReceiverA", "ReceiverB"),
        predeclared_receiver_ids=("ReceiverC", "ReceiverB", "ReceiverA"),
    )
    first = assess_receiver_training_support(
        universe,
        outer_fold_id="outer-fold-1",
        training_cell_type_ids=("ReceiverB", "ReceiverA"),
    )
    second = assess_receiver_training_support(
        universe,
        outer_fold_id="outer-fold-1",
        training_cell_type_ids=("ReceiverA", "ReceiverB"),
    )

    assert tuple(record.receiver_id for record in first) == universe.receiver_ids
    assert tuple(record.support_record_id for record in first) == tuple(
        record.support_record_id for record in second
    )
    assert all(
        record.status is ReceiverTrainingSupportStatus.OBSERVED
        and record.reason_code is None
        for record in first[:2]
    )
    missing = first[2]
    assert missing.status == "not_estimable"
    assert missing.reason_code == "receiver_absent_in_outer_training"
    assert not hasattr(missing, "gate")
    assert not hasattr(missing, "score")
    assert "gate" not in missing.to_dict()
    assert "score" not in missing.to_dict()
    with pytest.raises(TypeError, match="producer-owned"):
        ReceiverTrainingSupportRecord()


def test_training_support_rejects_unknown_axis_and_tampered_status() -> None:
    universe = freeze_receiver_universe(
        _root_identity(),
        ("ReceiverA", "ReceiverB"),
    )
    with pytest.raises(ContractError) as caught:
        assess_receiver_training_support(
            universe,
            outer_fold_id="outer-fold-1",
            training_cell_type_ids=("ReceiverA", "UnknownReceiver"),
        )
    assert caught.value.details.code == (
        "outer_training_cell_type_outside_receiver_universe"
    )

    missing = assess_receiver_training_support(
        universe,
        outer_fold_id="outer-fold-2",
        training_cell_type_ids=("ReceiverA",),
    )[1]
    object.__setattr__(missing, "status", ReceiverTrainingSupportStatus.OBSERVED)
    with pytest.raises(ContractError) as tampered:
        missing.to_dict()
    assert tampered.value.details.code == (
        "receiver_training_support_integrity_violation"
    )
