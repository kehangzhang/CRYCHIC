from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from crychic.core import (
    CrychicConfig,
    RunProvenance,
    SeedLineage,
    canonical_json,
    stable_id,
)
from crychic.results import RESULT_SCHEMA_VERSION, empty_table
from crychic.workflow.contracts import EDGE_EVIDENCE_COLUMNS


@pytest.fixture  # type: ignore[untyped-decorator]
def table_factory() -> Callable[[str, list[dict[str, Any]]], pd.DataFrame]:
    def make(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
        template = empty_table(name)
        if not rows:
            return template
        frame = pd.DataFrame(rows, columns=template.columns)
        for column in template:
            frame[column] = frame[column].astype(template[column].dtype)
        return frame

    return make


@pytest.fixture  # type: ignore[untyped-decorator]
def edge_evidence_frame() -> pd.DataFrame:
    row: dict[str, object] = {
        "sample_id": "donor-1-stim",
        "subject_id": "donor-1",
        "context": "stim",
        "context_id": stable_id("context", {"condition": "stim"}),
        "sender": "Monocyte",
        "receiver": "B cell",
        "interaction_id": "CXCL10_CXCR3",
        "driver_id": "CXCR3",
        "mode": "state",
        "contrast": "stim_vs_ctrl",
        "fold_id": "fold-0",
        "state_availability": 0.8,
        "state_availability_status": "observed",
        "state_availability_reason_code": None,
        "ecosystem_availability": 0.7,
        "ecosystem_availability_status": "observed",
        "ecosystem_availability_reason_code": None,
        "receptor_gate": 0.9,
        "receptor_gate_status": "observed",
        "receptor_gate_reason_code": None,
        "receiver_program_score": 0.75,
        "receiver_program_status": "observed",
        "receiver_program_reason_code": None,
        "incremental_downstream": None,
        "incremental_downstream_status": "not_estimable",
        "incremental_downstream_reason_code": "crossfit_not_available",
        "attribution_support": 0.85,
        "attribution_support_method": "signed_attribution",
        "attribution_support_status": "observed",
        "attribution_support_reason_code": None,
        "sender_weight": 0.7,
        "sender_status": "ok",
        "sender_reason_code": None,
        "prior_quality": 0.9,
        "prior_quality_source": "fixture-prior",
        "prior_quality_status": "observed",
        "prior_quality_reason_code": None,
        "legacy_downstream_activity": 0.9,
        "legacy_downstream_status": "observed",
        "legacy_downstream_reason_code": None,
        "legacy_integrated_strength": 0.82,
        "legacy_integrated_status": "ok",
        "legacy_integrated_reason_code": None,
        "scoring_function_status": "ok",
        "scoring_function_reason_code": None,
        "score_version": "0.1.0",
        "model_manifest_id": "model-1",
        "scoring_function_id": "functional-1",
        "sample_score_status": "linked",
        "sample_score_reason_code": None,
    }
    return pd.DataFrame([row], columns=EDGE_EVIDENCE_COLUMNS)


@pytest.fixture  # type: ignore[untyped-decorator]
def result_payload(
    table_factory: Callable[[str, list[dict[str, Any]]], pd.DataFrame],
) -> dict[str, Any]:
    context_json = canonical_json({"condition": "stim"})
    context_id = stable_id("context", {"condition": "stim"})
    config = CrychicConfig(context_keys=["condition"], random_seed=17)
    provenance = RunProvenance(
        package_version="0.1.0",
        git_commit="ccc6795",
        git_dirty=False,
        config_digest=config.digest,
        input_digest="a" * 64,
        resource_digests={"fixture": "b" * 64},
        result_schema_version=RESULT_SCHEMA_VERSION,
        seed_lineage=SeedLineage(config.random_seed),
        created_at="2026-07-12T00:00:00Z",
    )
    tables = {
        "interactions": table_factory(
            "interactions",
            [
                {
                    "context_id": context_id,
                    "context_json": context_json,
                    "sender": "Monocyte",
                    "receiver": "B cell",
                    "interaction_id": "CXCL10_CXCR3",
                    "mode": "state",
                    "contrast": "stim_vs_ctrl",
                    "availability": 0.8,
                    "assignment_weight": 0.7,
                    "comm_strength": 0.9,
                    "comm_probability": None,
                    "prior_quality": 0.9,
                    "n_subjects": 8,
                    "status": "ok",
                    "reason_code": "v0_1_inferential_disabled",
                },
                {
                    "context_id": context_id,
                    "context_json": context_json,
                    "sender": "T cell",
                    "receiver": "B cell",
                    "interaction_id": "CD40LG_CD40",
                    "mode": "state",
                    "contrast": "stim_vs_ctrl",
                    "availability": 0.6,
                    "assignment_weight": 0.5,
                    "comm_strength": 0.7,
                    "comm_probability": None,
                    "prior_quality": 0.8,
                    "n_subjects": 8,
                    "status": "ok",
                    "reason_code": "v0_1_inferential_disabled",
                },
            ],
        ),
        "differential": table_factory(
            "differential",
            [
                {
                    "hypothesis_level": "driver_family_receiver",
                    "hypothesis_id": "ifn_family_b_cell",
                    "contrast": "stim_vs_ctrl",
                    "mode": "state",
                    "view": "gene",
                    "effect_size": 1.2,
                    "standard_error": 0.3,
                    "statistic": 4.0,
                    "p_value": None,
                    "q_value": None,
                    "specificity_support": None,
                    "change_class": None,
                    "status": "ok",
                    "reason_code": "v0_1_inferential_disabled",
                }
            ],
        ),
        "responses": table_factory(
            "responses",
            [
                {
                    "context_id": context_id,
                    "context_json": context_json,
                    "receiver": "B cell",
                    "gene": "ISG15",
                    "contrast": "stim_vs_ctrl",
                    "effect_size": 2.0,
                    "standard_error": 0.4,
                    "z_score": 5.0,
                    "precision": 6.25,
                    "p_value": None,
                    "n_subjects": 8,
                    "status": "ok",
                    "reason_code": "v0_1_inferential_disabled",
                }
            ],
        ),
        "sample_scores": table_factory(
            "sample_scores",
            [
                {
                    "subject_id": "donor-1",
                    "sample_id": "donor-1-stim",
                    "context_id": context_id,
                    "context_json": context_json,
                    "design_row_id": stable_id(
                        "design_row",
                        {"context_id": context_id, "sample_id": "donor-1-stim"},
                    ),
                    "edge_id": stable_id(
                        "communication_edge",
                        {
                            "interaction_id": "CXCL10_CXCR3",
                            "receiver": "B cell",
                            "sender": "Monocyte",
                        },
                    ),
                    "scoring_functional_id": "functional-1",
                    "repeat_id": "repeat-0",
                    "fold_id": "fold-0",
                    "mode": "state",
                    "availability": 0.8,
                    "downstream": 0.9,
                    "sender_component": 0.7,
                    "prior_quality": 0.9,
                    "abundance_component": 1.0,
                    "comm_strength": 0.82,
                    "eligible": True,
                    "status": "ok",
                    "reason_code": None,
                }
            ],
        ),
        "signatures": table_factory(
            "signatures",
            [
                {
                    "context_id": context_id,
                    "context_json": context_json,
                    "sender_id": "Monocyte",
                    "receiver": "B cell",
                    "interaction_id": "CXCL10_CXCR3",
                    "gene": "ISG15",
                    "view": "gene",
                    "direction": "positive",
                    "observed": 2.0,
                    "predicted": 1.4,
                    "residual": 0.6,
                    "weight": 0.8,
                    "stability": 0.9,
                    "status": "ok",
                    "reason_code": None,
                },
                {
                    "context_id": context_id,
                    "context_json": context_json,
                    "sender_id": "Monocyte",
                    "receiver": "B cell",
                    "interaction_id": "CXCL10_CXCR3",
                    "gene": "STAT1",
                    "view": "gene",
                    "direction": "positive",
                    "observed": 1.6,
                    "predicted": 1.0,
                    "residual": 0.6,
                    "weight": 0.7,
                    "stability": 0.8,
                    "status": "ok",
                    "reason_code": None,
                },
            ],
        ),
    }
    return {
        "config": config,
        "provenance": provenance,
        "run_manifest": {
            "run_id": "run-fixture",
            "mode": "exploratory",
            "stages": [{"name": "results", "status": "complete", "reason_code": None}],
            "warnings": [],
        },
        "tables": tables,
    }
