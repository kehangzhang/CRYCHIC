from __future__ import annotations

import pandas as pd
import pytest

from crychic.workflow.contracts import EDGE_EVIDENCE_COLUMNS, EdgeEvidenceLedger


def _valid_table() -> pd.DataFrame:
    row = {
        "sample_id": "s1",
        "subject_id": "p1",
        "context": "treated",
        "context_id": "context-1",
        "sender": "Sender",
        "receiver": "Receiver",
        "interaction_id": "L_R",
        "driver_id": "L",
        "mode": "state",
        "contrast": "treated_vs_control",
        "fold_id": "in_sample",
        "state_availability": 0.8,
        "state_availability_status": "observed",
        "state_availability_reason_code": None,
        "ecosystem_availability": 0.4,
        "ecosystem_availability_status": "observed",
        "ecosystem_availability_reason_code": None,
        "receptor_gate": 0.7,
        "receptor_gate_status": "observed",
        "receptor_gate_reason_code": None,
        "receiver_program_score": 0.6,
        "receiver_program_status": "observed",
        "receiver_program_reason_code": None,
        "incremental_downstream": None,
        "incremental_downstream_status": "not_estimable",
        "incremental_downstream_reason_code": (
            "cross_fitted_receiver_null_not_implemented"
        ),
        "attribution_support": 0.5,
        "attribution_support_method": "gated_prior_attribution_v1",
        "attribution_support_status": "observed",
        "attribution_support_reason_code": None,
        "sender_weight": 0.9,
        "sender_status": "ok",
        "sender_reason_code": None,
        "prior_quality": 1.0,
        "prior_quality_source": "constant=1.0;resource=fixture",
        "prior_quality_status": "observed",
        "prior_quality_reason_code": None,
        "legacy_downstream_activity": 0.3,
        "legacy_downstream_status": "observed",
        "legacy_downstream_reason_code": None,
        "legacy_integrated_strength": 0.4,
        "legacy_integrated_status": "ok",
        "legacy_integrated_reason_code": None,
        "scoring_function_status": "exploratory_in_sample",
        "scoring_function_reason_code": "exploratory_not_cross_fitted",
        "score_version": "geometric_v1_tracked",
        "model_manifest_id": "model-1",
        "scoring_function_id": "functional-1",
        "sample_score_status": "linked",
        "sample_score_reason_code": None,
    }
    return pd.DataFrame([row], columns=EDGE_EVIDENCE_COLUMNS)


def test_edge_evidence_contract_copies_valid_component_ledger() -> None:
    source = _valid_table()

    ledger = EdgeEvidenceLedger(source)
    source.loc[0, "state_availability"] = 0.1

    assert ledger.table.loc[0, "state_availability"] == pytest.approx(0.8)


def test_edge_evidence_contract_rejects_duplicate_primary_key() -> None:
    duplicated = pd.concat([_valid_table(), _valid_table()], ignore_index=True)

    with pytest.raises(ValueError, match="primary keys must be unique"):
        EdgeEvidenceLedger(duplicated)


def test_edge_evidence_contract_requires_reason_for_missing_component() -> None:
    invalid = _valid_table()
    invalid.loc[0, "receiver_program_score"] = None
    invalid.loc[0, "receiver_program_status"] = "not_estimable"
    invalid.loc[0, "receiver_program_reason_code"] = None

    with pytest.raises(ValueError, match="null receiver_program_score requires"):
        EdgeEvidenceLedger(invalid)


def test_edge_evidence_contract_rejects_partial_score_provenance() -> None:
    invalid = _valid_table()
    invalid.loc[0, "model_manifest_id"] = None

    with pytest.raises(ValueError, match="entirely present or entirely NA"):
        EdgeEvidenceLedger(invalid)


def test_edge_evidence_contract_rejects_zero_filled_missing_sender() -> None:
    invalid = _valid_table()
    invalid.loc[0, "sender_weight"] = 0.0
    invalid.loc[0, "sender_status"] = "missing_evidence"
    invalid.loc[0, "sender_reason_code"] = "missing_ligand_availability"

    with pytest.raises(ValueError, match="requires an NA value"):
        EdgeEvidenceLedger(invalid)
