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
                    "design_row_id": "design-1",
                    "edge_id": "edge-1",
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
