from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from benchmarks.comprehensive.evaluate_m1_mechanism_concordance import (
    M0_METHOD,
    M1_METHOD,
    _gate,
    _method_summary,
    _program_sample_table,
    _scenario_summary,
    _seed_metrics,
    signed_geometric_program_concordance,
)
from benchmarks.comprehensive.generate_m1_mechanism_fixture import (
    EXPECTED_COMPONENTS,
    SCHEMA_VERSION,
    _scenario_seed,
    generate,
)
from crychic.pseudobulk import PseudobulkDataset


def test_signed_concordance_retains_alignment_and_rejects_discordance() -> None:
    assert signed_geometric_program_concordance(
        0.16, 0.09, expected_program_direction=1
    ) == pytest.approx(0.12)
    assert signed_geometric_program_concordance(
        -0.16, -0.09, expected_program_direction=1
    ) == pytest.approx(-0.12)
    assert (
        signed_geometric_program_concordance(-0.16, 0.09, expected_program_direction=1)
        == 0.0
    )
    assert signed_geometric_program_concordance(
        0.16, -0.09, expected_program_direction=-1
    ) == pytest.approx(0.12)
    with pytest.raises(ValueError, match="-1 or 1"):
        signed_geometric_program_concordance(0.1, 0.1, expected_program_direction=0)


def _aggregate() -> PseudobulkDataset:
    unit_ids = ("s1:r", "s2:r", "s1:s", "s2:s")
    metadata = pd.DataFrame.from_records(
        [
            {
                "unit_id": unit_id,
                "sample_id": sample,
                "subject_id": subject,
                "cell_type": cell_type,
                "context": (("condition", condition),),
                "condition": condition,
                "matrix_row": index,
                "n_cells": 20,
                "cell_proportion": 0.5,
                "state_eligible": True,
                "abundance_eligible": True,
                "missingness_reason": "observed",
            }
            for index, (unit_id, sample, subject, condition, cell_type) in enumerate(
                (
                    ("s1:r", "s1", "p1", "ctrl", "Receiver"),
                    ("s2:r", "s2", "p1", "stim", "Receiver"),
                    ("s1:s", "s1", "p1", "ctrl", "Sender"),
                    ("s2:s", "s2", "p1", "stim", "Sender"),
                )
            )
        ]
    )
    counts = sparse.csr_matrix(
        np.asarray(
            [
                [10, 30, 60],
                [40, 40, 20],
                [1, 1, 98],
                [1, 1, 98],
            ],
            dtype=int,
        )
    )
    return PseudobulkDataset(
        counts=counts,
        detection_fraction=sparse.csr_matrix(np.ones(counts.shape)),
        unit_metadata=metadata,
        feature_ids=("T1", "T2", "H"),
        matrix_unit_ids=unit_ids,
        source_location="fixture",
    )


def test_program_sample_table_uses_fixed_log_cpm_scale_and_receiver_only() -> None:
    definition = pd.DataFrame.from_records(
        [
            {
                "program_id": "p",
                "interaction_id": "lr",
                "receiver": "Receiver",
                "target_gene": target,
                "weight": 0.5,
                "expected_program_direction": 1,
            }
            for target in ("T1", "T2")
        ]
    )
    result = _program_sample_table(
        _aggregate(), definition, expression_reference=1_000_000.0
    )

    assert len(result) == 2
    assert set(result["receiver"]) == {"Receiver"}
    expected_ctrl = np.mean(
        np.log1p(np.asarray([100_000.0, 300_000.0])) / np.log1p(1_000_000.0)
    )
    assert result.set_index("condition").loc[
        "ctrl", "receiver_program_raw"
    ] == pytest.approx(expected_ctrl)


def _effects() -> pd.DataFrame:
    rows = []
    scenarios = tuple(EXPECTED_COMPONENTS)
    for seed in (11, 12, 13):
        for scenario in scenarios:
            integrated = scenario == "active"
            rows.append(
                {
                    "root_seed": seed,
                    "scenario": scenario,
                    "expected_integrated_edge": integrated,
                    "m0_ranking_score": (
                        0.8
                        if scenario == "receptor_knockout"
                        else 0.6
                        if scenario in {"active", "ligand_only"}
                        else 0.05
                    ),
                    "m1_ranking_score": 0.5 if integrated else 0.01,
                    "m1_same_direction": integrated,
                    "paired_subjects": 10,
                    "receiver_program_effect": 0.3 if integrated else 0.01,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_seed_metrics_separate_mechanism_family_from_m0_activity() -> None:
    metrics = _seed_metrics(_effects())
    summary = _method_summary(metrics).set_index("method")

    assert summary.loc[M1_METHOD, "average_precision"] == pytest.approx(1.0)
    assert summary.loc[M1_METHOD, "auroc"] == pytest.approx(1.0)
    assert summary.loc[M1_METHOD, "median_active_rank"] == pytest.approx(1.0)
    assert summary.loc[M0_METHOD, "average_precision"] < 1.0
    assert summary.loc[M0_METHOD, "auroc"] < 1.0


def test_m1_gate_requires_seed_level_superiority_and_partial_suppression() -> None:
    metrics = _seed_metrics(_effects())
    summary = _method_summary(metrics)
    scenario = _scenario_summary(_effects())
    paired = pd.DataFrame.from_records(
        [
            {"metric": "average_precision", "paired_seeds": 3, "ci_low": 0.1},
            {"metric": "auroc", "paired_seeds": 3, "ci_low": 0.1},
        ]
    )
    config = {
        "development": {"seeds": [11, 12, 13], "minimum_paired_seeds": 3},
        "gates": {
            "average_precision_delta_ci_lower_minimum_exclusive": 0.0,
            "auroc_delta_ci_lower_minimum_exclusive": 0.0,
            "minimum_active_retention_fraction": 0.9,
            "maximum_median_active_rank": 1.0,
            "maximum_partial_to_active_ratio": 0.25,
            "minimum_event_coverage": 1.0,
        },
    }
    result = _gate(summary, paired, scenario, config=config, role="development")

    assert result["status"] == "DEVELOPMENT_PASS"
    assert all(result["checks"].values())


def _resource(path: Path) -> Path:
    table = pd.DataFrame.from_records(
        [
            {
                "harmonized_interaction_id": f"lr-{index}",
                "ligand": ligand,
                "receptor": receptor,
                "scseqcommdiff_source_interaction_id": f"source-{index}",
                "scseqcommdiff_covered": True,
            }
            for index, (ligand, receptor) in enumerate(
                (
                    ("CXCL10", "CXCR3"),
                    ("CCL5", "CCR5"),
                    ("VEGFA", "FLT1"),
                    ("CXCL12", "CXCR4"),
                    ("EGF", "EGFR"),
                ),
                start=1,
            )
        ]
    )
    table.to_csv(path, sep="\t", index=False)
    return path


def test_generator_masks_method_ids_and_freezes_program_truth(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    manifest = generate(
        output,
        _resource(tmp_path / "resource.tsv"),
        seeds=(20270101,),
        n_subjects=4,
        mean_cells_per_sample=60,
        n_jobs=2,
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["generation_workers"] == 2
    assert len(manifest["records"]) == 7
    assert all("active" not in record["dataset_id"] for record in manifest["records"])
    truth = pd.read_csv(output / "mechanism_truth.tsv", sep="\t")
    assert truth["expected_integrated_edge"].sum() == 1
    assert set(truth["scenario"]) == set(EXPECTED_COMPONENTS)
    program = pd.read_csv(output / "program_definitions.tsv", sep="\t")
    assert program["weight"].sum() == pytest.approx(1.0)
    persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["program_definition"]["rows"] == len(program)


def test_scenario_seed_is_stable_and_scenario_specific() -> None:
    first = _scenario_seed(11, "active")
    assert first == _scenario_seed(11, "active")
    assert first != _scenario_seed(11, "global_null")
    assert first != _scenario_seed(12, "active")
