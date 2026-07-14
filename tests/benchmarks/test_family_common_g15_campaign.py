from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.simulation.run_family_common_g15_campaign import (
    ALL_SCENARIOS,
    AUDIT_CLAIMS,
    CAMPAIGN_SOURCE_PATHS,
    DEFAULT_DEVELOPMENT_SCENARIOS,
    FROZEN_LAMBDA1_FRACTIONS,
    FROZEN_LAMBDA2_FRACTIONS,
    RELEASED_MODES,
    SCORE_KINDS,
    _aggregate_component_semantics,
    _aggregate_control_families,
    _aggregate_score_campaign,
    _component_summaries,
    _paired_effect_summary,
    _select_seed_records,
    build_campaign_spec,
    build_parser,
    campaign_source_sha256,
    compact_campaign_summary,
    generate_campaign_input,
    load_campaign_registry,
)
from scipy import sparse

from crychic.response import load_receiver_autonomous_program_resource

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    REPO_ROOT / "benchmarks/fixtures/family_common_g15_campaign_v1.json"
)
SUMMARY_PATH = (
    REPO_ROOT
    / "benchmarks/results/public_family_common_g15_campaign_v1_summary.json"
)
FULL_ARTIFACT_PATH = (
    REPO_ROOT.parent
    / "benchmark_work/algorithm_smoke/public_family_common_g15_campaign_v1.json"
)
AUTONOMOUS_ROOT = (
    REPO_ROOT / "benchmarks/fixtures/synthetic_receiver_autonomous_program"
)
AUTONOMOUS_REGISTRATION_ID = "crychic.synthetic_receiver_autonomous_program.v1"


def _registry() -> dict[str, object]:
    return load_campaign_registry(REGISTRY_PATH)


def _seed_record(
    registry: Mapping[str, object], index: int = 0
) -> Mapping[str, object]:
    records = cast(list[Mapping[str, object]], registry["seed_sets"])
    return records[index]


def _raw_counts(adata: ad.AnnData) -> sparse.csr_matrix:
    return sparse.csr_matrix(adata.layers["counts"])


def _mean_count(
    adata: ad.AnnData,
    *,
    genes: tuple[str, ...],
    cell_type: str,
    condition: str,
) -> float:
    obs = adata.obs
    var_names = adata.var_names
    mask = obs["cell_type"].astype(str).eq(cell_type) & obs["condition"].astype(
        str
    ).eq(condition)
    indices = [var_names.get_loc(gene) for gene in genes]
    values = _raw_counts(adata)[np.flatnonzero(mask.to_numpy()), :][:, indices]
    return float(values.mean())


def test_registry_freezes_seven_scenarios_multiple_edges_seeds_and_grid() -> None:
    registry = _registry()
    tuning = cast(Mapping[str, object], registry["penalty_tuning_policy"])

    assert tuple(cast(Sequence[object], registry["scenarios"])) == ALL_SCENARIOS
    assert tuple(
        cast(Sequence[object], registry["default_development_scenarios"])
    ) == (
        DEFAULT_DEVELOPMENT_SCENARIOS
    )
    assert len(cast(list[object], registry["known_edges"])) == 3
    assert len(cast(list[object], registry["seed_sets"])) == 3
    assert registry["lock_scope"] == "before_first_multiseed_execution"
    debug_policy = cast(Mapping[str, object], registry["prelock_debug_policy"])
    assert debug_policy["debug_result_is_excluded_from_all_campaign_metrics"] is True
    assert debug_policy["first_eligible_campaign_requires_at_least_two_frozen_seeds"]
    assert tuple(
        cast(Sequence[object], tuning["lambda1_fractions"])
    ) == FROZEN_LAMBDA1_FRACTIONS
    assert tuple(
        cast(Sequence[object], tuning["lambda2_fractions"])
    ) == FROZEN_LAMBDA2_FRACTIONS
    assert tuning["selection_rule"] == (
        "subject_equal_paired_delta_one_se_l1_then_l2_sparsity_priority_v2"
    )
    assert tuning["candidate_priority"] == (
        "l1_fraction_desc_then_l2_fraction_desc_v1"
    )
    truth = cast(Mapping[str, Mapping[str, object]], registry["component_truth"])
    assert truth["ligand_only"]["receiver_program"] == "zero"
    assert truth["receiver_autonomous"]["receiver_program"] == "allowed_positive"
    assert truth["target_only"]["receiver_program"] == "positive"
    assert truth["receptor_knockout"]["receiver_program"] == "positive"
    estimands = cast(Mapping[str, object], registry["component_estimands"])
    assert estimands["receiver_program"] == "subject_level_stim_minus_ctrl"
    assert estimands["incremental_downstream"] == "raw_heldout_family_gain"
    assert estimands["integrated"] == "subject_level_stim_minus_ctrl"


def test_source_closure_hashes_public_workflow_and_frozen_registry() -> None:
    required = {
        "benchmarks/fixtures/family_common_g15_campaign_v1.json",
        "benchmarks/simulation/run_family_common_g15_campaign.py",
        "src/crychic/scoring/receiver_program.py",
        "src/crychic/workflow/crossfit.py",
        "src/crychic/scoring/family_common.py",
    }
    assert required.issubset(CAMPAIGN_SOURCE_PATHS)
    observed = campaign_source_sha256()
    assert set(observed) == set(CAMPAIGN_SOURCE_PATHS)
    assert all(
        digest == hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path, digest in observed.items()
    )


def test_campaign_spec_uses_three_candidate_public_tuned_policy() -> None:
    autonomous = load_receiver_autonomous_program_resource(
        AUTONOMOUS_ROOT,
        manifest_path=AUTONOMOUS_ROOT / "manifest.json",
        registration_id=AUTONOMOUS_REGISTRATION_ID,
    )
    spec = build_campaign_spec(autonomous, crossfit_seed=2578925092)
    tuning = spec.penalty_tuning_spec

    assert autonomous.is_manifest_verified_trusted
    assert tuning is not None
    assert tuning.lambda1_fractions == FROZEN_LAMBDA1_FRACTIONS
    assert tuning.lambda2_fractions == FROZEN_LAMBDA2_FRACTIONS
    assert tuning.root_seed == 2578925092
    assert tuning.selection_rule == (
        "subject_equal_paired_delta_one_se_l1_then_l2_sparsity_priority_v2"
    )


def test_multiedge_inputs_are_deterministic_paired_and_mechanism_specific() -> None:
    registry = _registry()
    seed = _seed_record(registry)
    registry_sha = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    generated = {
        scenario: generate_campaign_input(
            registry,
            profile_name="tiny",
            seed_record=seed,
            scenario=scenario,
            registry_sha256=registry_sha,
        )
        for scenario in ALL_SCENARIOS
    }
    active, active_audit = generated["active"]
    active_again, repeat_audit = generate_campaign_input(
        registry,
        profile_name="tiny",
        seed_record=seed,
        scenario="active",
        registry_sha256=registry_sha,
    )
    ligand_only = generated["ligand_only"][0]
    target_only = generated["target_only"][0]
    receptor_knockout = generated["receptor_knockout"][0]
    receiver_autonomous = generated["receiver_autonomous"][0]
    global_null = generated["global_null"][0]

    assert active_audit["input_identity"] == repeat_audit["input_identity"]
    assert active_audit["content_sha256"] == repeat_audit["content_sha256"]
    assert (_raw_counts(active) != _raw_counts(active_again)).nnz == 0
    assert all(audit[1]["paired_contexts_complete"] for audit in generated.values())
    assert all(adata.n_vars == 41 for adata, _ in generated.values())

    edge_records = cast(list[Mapping[str, object]], registry["known_edges"])
    ligands = tuple(str(edge["ligand"]) for edge in edge_records)
    receptors = tuple(str(edge["receptor"]) for edge in edge_records)
    targets = tuple(
        str(target)
        for edge in edge_records
        for target in cast(list[object], edge["target_genes"])
    )
    autonomous_targets = tuple(
        str(target)
        for target in cast(
            Sequence[object],
            cast(Mapping[str, object], registry["signal_policy"])[
                "autonomous_target_genes"
            ],
        )
    )

    assert _mean_count(
        active, genes=ligands, cell_type="Sender", condition="stim"
    ) == _mean_count(
        ligand_only, genes=ligands, cell_type="Sender", condition="stim"
    )
    assert _mean_count(
        active, genes=targets, cell_type="Receiver", condition="stim"
    ) > _mean_count(
        ligand_only, genes=targets, cell_type="Receiver", condition="stim"
    )
    assert _mean_count(
        active, genes=targets, cell_type="Receiver", condition="stim"
    ) == _mean_count(
        target_only, genes=targets, cell_type="Receiver", condition="stim"
    )
    assert _mean_count(
        receptor_knockout,
        genes=receptors,
        cell_type="Receiver",
        condition="ctrl",
    ) == 0.0
    assert _mean_count(
        receiver_autonomous,
        genes=autonomous_targets,
        cell_type="Receiver",
        condition="stim",
    ) > _mean_count(
        global_null,
        genes=autonomous_targets,
        cell_type="Receiver",
        condition="stim",
    )


def _score_rows(
    edge_ids: tuple[str, ...], *, active_mean: float, coverage: float
) -> list[dict[str, object]]:
    return [
        {
            "known_edge_id": edge_id,
            "mode": mode,
            "score_kind": score_kind,
            "mean": active_mean,
            "coverage": coverage,
            "raw_strength_mean": active_mean + 0.2,
            "raw_strength_maximum": active_mean + 0.4,
            "dense_rank": 1,
            "recovered": active_mean > 0.0,
        }
        for edge_id in edge_ids
        for mode in RELEASED_MODES
        for score_kind in SCORE_KINDS
    ]


def test_paired_margin_aggregation_retains_coverage_and_equal_edge_grain() -> None:
    edges = (
        {"known_edge_id": "edge-a"},
        {"known_edge_id": "edge-b"},
    )
    records: list[dict[str, object]] = []
    for seed_id in ("seed-1", "seed-2"):
        records.extend(
            [
                {
                    "seed_id": seed_id,
                    "scenario": "active",
                    "known_edge_scores": _score_rows(
                        ("edge-a", "edge-b"), active_mean=0.4, coverage=0.95
                    ),
                },
                {
                    "seed_id": seed_id,
                    "scenario": "ligand_only",
                    "known_edge_scores": _score_rows(
                        ("edge-a", "edge-b"), active_mean=0.1, coverage=1.0
                    ),
                },
            ]
        )
    result = _aggregate_score_campaign(
        records,
        edge_catalog=edges,
        seed_ids=("seed-1", "seed-2"),
        positive_scenario="active",
        reference_scenario="ligand_only",
        maximum_allowed_coverage_loss=0.05,
    )
    summaries = cast(list[Mapping[str, object]], result["macro_summaries"])

    assert len(cast(list[object], result["paired_records"])) == 16
    assert all(summary["n_expected_pairs"] == 4 for summary in summaries)
    assert all(
        np.isclose(
            cast(float, summary["mean_active_minus_ligand_only_margin"]), 0.3
        )
        for summary in summaries
    )
    assert all(summary["active_recovery_fraction"] == 1.0 for summary in summaries)
    assert all(summary["coverage_noninferior"] is True for summary in summaries)
    assert all(
        summary["primary_estimand"] == "subject_level_stim_minus_ctrl"
        for summary in summaries
    )
    paired_records = cast(list[Mapping[str, object]], result["paired_records"])
    assert all(
        record["active_raw_strength_maximum"] == 0.8
        for record in paired_records
    )
    assert all(
        record["raw_strength_estimand"]
        == "raw_heldout_score_over_context_rows"
        for record in paired_records
    )


def test_paired_effect_uses_subject_contrasts_not_hashed_context_labels() -> None:
    table = pd.DataFrame(
        {
            "sample_id": ["S01:ctrl", "S01:stim", "S02:ctrl"],
            "subject_id": ["S01", "S01", "S02"],
            "context_id": ["context_hash_a", "context_hash_b", "context_hash_a"],
            "score": [0.2, 0.7, 0.4],
        }
    )

    summary = _paired_effect_summary(table, score_column="score")

    assert summary["estimand"] == "subject_level_stim_minus_ctrl"
    assert summary["n_expected_subjects"] == 2
    assert summary["n_finite"] == 1
    assert summary["coverage"] == 0.5
    assert np.isclose(cast(float, summary["mean"]), 0.5)


def test_component_primary_estimands_separate_gain_from_differential() -> None:
    rows: list[dict[str, object]] = []
    for mode in RELEASED_MODES:
        for subject_id in ("S01", "S02"):
            for condition in ("ctrl", "stim"):
                rows.append(
                    {
                        "sample_id": f"{subject_id}:{condition}",
                        "subject_id": subject_id,
                        "context_id": f"hash-{condition}",
                        "receiver": "Receiver",
                        "family_id": "family-a",
                        "mode": mode,
                        "receiver_program_score": (
                            0.1 if condition == "ctrl" else 0.5
                        ),
                        "receiver_program_status": "observed",
                        "receiver_program_reason_code": None,
                        "incremental_downstream_gain": 0.2,
                        "integrated_lr_score": (
                            0.1 if condition == "ctrl" else 0.3
                        ),
                        "status": "ok",
                        "reason_code": None,
                    }
                )
    summaries = _component_summaries(
        pd.DataFrame(rows),
        memberships={"edge-a": ("family-a",)},
        positive_tolerance=1e-12,
    )
    state = {
        str(row["component"]): row
        for row in summaries
        if row["mode"] == "state"
    }

    assert state["receiver_program"]["estimand"] == (
        "subject_level_stim_minus_ctrl"
    )
    assert np.isclose(cast(float, state["receiver_program"]["mean"]), 0.4)
    assert state["incremental_downstream"]["estimand"] == (
        "raw_heldout_family_gain"
    )
    assert np.isclose(cast(float, state["incremental_downstream"]["mean"]), 0.2)
    assert state["incremental_downstream"]["paired_effect_mean"] == 0.0
    assert state["integrated"]["estimand"] == "subject_level_stim_minus_ctrl"
    assert np.isclose(cast(float, state["integrated"]["mean"]), 0.2)


def test_not_estimable_receiver_program_never_counts_as_semantic_pass() -> None:
    registry = _registry()
    edge = cast(Mapping[str, object], cast(list[object], registry["known_edges"])[0])
    edge_id = str(edge["known_edge_id"])
    component_scores = [
        {
            "known_edge_id": edge_id,
            "mode": mode,
            "component": component,
            "mean": None if component == "receiver_program" else 0.0,
            "coverage": 0.0 if component == "receiver_program" else 1.0,
            "raw_strength_mean": 0.25,
            "raw_strength_minimum": 0.0,
            "raw_strength_maximum": 0.5,
            "raw_strength_coverage": 1.0,
            "raw_strength_estimand": "raw_heldout_score_over_context_rows",
            "paired_effect_mean": (
                None if component == "receiver_program" else 0.0
            ),
            "paired_effect_coverage": (
                0.0 if component == "receiver_program" else 1.0
            ),
            "estimand": (
                "raw_heldout_family_gain"
                if component == "incremental_downstream"
                else "subject_level_stim_minus_ctrl"
            ),
        }
        for mode in RELEASED_MODES
        for component in (
            "receiver_program",
            "incremental_downstream",
            "integrated",
        )
    ]
    result = _aggregate_component_semantics(
        [
            {
                "seed_id": "seed-1",
                "scenario": "target_only",
                "component_scores": component_scores,
            }
        ],
        registry=registry,
        edge_catalog=({"known_edge_id": edge_id},),
        seed_ids=("seed-1",),
        positive_tolerance=1e-12,
    )
    records = cast(list[Mapping[str, object]], result["records"])
    receiver_program = [
        record
        for record in records
        if record["scenario"] == "target_only"
        and record["component"] == "receiver_program"
    ]

    assert len(receiver_program) == 2
    assert all(record["truth_code"] == "positive" for record in receiver_program)
    assert all(record["status"] == "not_estimable" for record in receiver_program)
    assert all(record["conforms"] is None for record in receiver_program)
    assert all(record["not_estimable_never_counts_as_pass"] for record in records)


def test_control_family_summary_keeps_raw_false_positive_diagnostic() -> None:
    family_mode = {
        "n_positive_integrated_families": 0,
        "n_positive_integrated_nontruth_families": 0,
        "n_positive_raw_integrated_families": 1,
        "n_positive_raw_integrated_nontruth_families": 1,
        "maximum_raw_integrated_family_mean": 0.2,
    }
    records = [
        {
            "seed_id": "seed-1",
            "scenario": "receiver_autonomous",
            "family_diagnostics": {
                "n_training_selected_families": 1,
                "n_heldout_selected_rows": 1,
                "modes": [family_mode, family_mode],
            },
        }
    ]

    summary = _aggregate_control_families(
        records, scenarios=("receiver_autonomous",)
    )[0]

    assert summary["any_positive_integrated_family_rate"] == 0.0
    assert summary["any_positive_raw_integrated_family_rate"] == 1.0
    assert summary["mean_positive_raw_integrated_nontruth_families"] == 1.0
    assert summary["maximum_raw_integrated_family_mean"] == 0.2
    assert summary["paired_integrated_estimand"] == (
        "subject_level_stim_minus_ctrl"
    )
    assert summary["raw_integrated_estimand"] == (
        "raw_heldout_score_over_context_rows"
    )


def test_cli_defaults_are_a_multiseed_development_subset_not_full_g15() -> None:
    args = build_parser().parse_args([])

    assert args.seed_count == 2
    assert args.seed_ids is None
    assert tuple(args.scenarios) == DEFAULT_DEVELOPMENT_SCENARIOS
    assert tuple(args.scenarios) != ALL_SCENARIOS
    assert args.profile == "quick"
    assert AUDIT_CLAIMS["default_switch_allowed"] is False
    assert AUDIT_CLAIMS["complete_pipeline_oof_certification"] is False


def test_seed_id_selection_can_run_only_the_frozen_unseen_seed() -> None:
    registry_path = REGISTRY_PATH
    registry_sha256 = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    selected = _select_seed_records(
        _registry(),
        seed_count=None,
        seed_ids=("public-g15-003",),
    )

    assert [record["seed_id"] for record in selected] == ["public-g15-003"]
    assert selected[0]["input_seed"] == 3364261018
    assert selected[0]["crossfit_seed"] == 4109871368
    assert hashlib.sha256(registry_path.read_bytes()).hexdigest() == registry_sha256


@pytest.mark.parametrize(
    ("seed_ids", "message"),
    [
        (("public-g15-004",), "unknown campaign seed_ids"),
        (
            ("public-g15-003", "public-g15-003"),
            "seed_ids must be unique",
        ),
    ],
)
def test_seed_id_selection_rejects_unknown_or_duplicate_ids(
    seed_ids: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _select_seed_records(
            _registry(),
            seed_count=None,
            seed_ids=seed_ids,
        )


def test_seed_count_and_seed_ids_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _select_seed_records(
            _registry(),
            seed_count=1,
            seed_ids=("public-g15-003",),
        )

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--seed-count",
                "1",
                "--seed-ids",
                "public-g15-003",
            ]
        )


def test_seed_count_selection_retains_the_frozen_prefix_behavior() -> None:
    default_selected = _select_seed_records(
        _registry(),
        seed_count=None,
        seed_ids=None,
    )
    explicit_selected = _select_seed_records(
        _registry(),
        seed_count=2,
        seed_ids=None,
    )

    expected = ["public-g15-001", "public-g15-002"]
    assert [record["seed_id"] for record in default_selected] == expected
    assert [record["seed_id"] for record in explicit_selected] == expected


def test_cli_accepts_seed_ids_without_changing_the_seed_count_default() -> None:
    args = build_parser().parse_args(["--seed-ids", "public-g15-003"])

    assert args.seed_count == 2
    assert args.seed_ids == ["public-g15-003"]


def test_tracked_multiseed_summary_is_current_and_semantically_pinned() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert summary["source_sha256"] == campaign_source_sha256()
    assert summary["scope"] == "development_full_seven_scenario_diagnostic"
    assert summary["checks"]["eligible_for_campaign_metric_interpretation"]
    assert summary["checks"]["executed_multiple_seeds"]
    assert summary["checks"]["executed_full_seven_scenario_contract"]
    assert summary["registry"]["executed_seed_ids"] == [
        "public-g15-001",
        "public-g15-002",
    ]
    assert tuple(summary["registry"]["executed_scenarios"]) == ALL_SCENARIOS
    macro = summary["aggregate_metrics"]["active_vs_ligand_only"][
        "macro_summaries"
    ]
    assert len(macro) == 4
    assert all(record["active_recovery_fraction"] == 1.0 for record in macro)
    assert all(record["positive_margin_fraction"] == 1.0 for record in macro)
    assert all(record["maximum_coverage_loss"] == 0.0 for record in macro)
    controls = summary["aggregate_metrics"][
        "control_family_false_positive_and_selection"
    ]
    assert {record["scenario"] for record in controls} == set(
        ALL_SCENARIOS
    ).difference({"active"})
    assert all(
        record["any_positive_integrated_family_rate"] == 0.0
        for record in controls
    )
    assert all(
        record["mean_positive_integrated_families"] == 0.0
        for record in controls
    )

    if FULL_ARTIFACT_PATH.exists():
        full = json.loads(FULL_ARTIFACT_PATH.read_text(encoding="utf-8"))
        assert compact_campaign_summary(full) == summary
