from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from benchmarks.simulation.run_family_common_crossfit_smoke import (
    ACTIVE_SOURCE_INTERACTION_ID,
    ALLOWED_SCENARIOS,
    AUDIT_CLAIMS,
    AUTONOMOUS_REGISTRATION_ID,
    DEFAULT_SCENARIOS,
    FAMILY_COMMON_SMOKE_SOURCE_PATHS,
    INPUT_REGISTRY_RELATIVE_PATH,
    _rank_interaction_table,
    _tuning_candidate_audit,
    build_artifact_metadata,
    build_compact_summary,
    build_family_common_smoke_spec,
    build_parser,
    family_common_smoke_source_sha256,
)

from crychic.attribution import (
    PenaltyTuningSpec,
    record_penalty_fold_evaluation,
    select_penalty_candidate,
)
from crychic.response import load_receiver_autonomous_program_resource

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "benchmarks/fixtures/synthetic_receiver_autonomous_program"
SUMMARY_PATH = (
    REPO_ROOT / "benchmarks/results/family_common_crossfit_smoke_v1_summary.json"
)
FULL_ARTIFACT_PATH = (
    REPO_ROOT.parent
    / "benchmark_work/algorithm_smoke/family_common_crossfit_smoke_v1.json"
)
INPUT_REGISTRY_PATH = REPO_ROOT / INPUT_REGISTRY_RELATIVE_PATH
REQUIRED_SOURCE_CLOSURE = {
    "benchmarks/fixtures/family_common_crossfit_smoke_inputs_v1.json",
    "benchmarks/fixtures/synthetic_receiver_autonomous_program/manifest.json",
    "benchmarks/fixtures/synthetic_receiver_autonomous_program/programs.tsv",
    "benchmarks/simulation/run_family_common_crossfit_smoke.py",
    "src/crychic/attribution/tuning.py",
    "src/crychic/resources/autonomous_registry.py",
    "src/crychic/response/autonomous.py",
    "src/crychic/scoring/downstream.py",
    "src/crychic/scoring/family_common.py",
    "src/crychic/scoring/receiver_program.py",
    "src/crychic/workflow/crossfit.py",
    "src/crychic/workflow/receiver_incremental.py",
}


def test_source_closure_binds_new_algorithm_registry_and_fixture() -> None:
    assert REQUIRED_SOURCE_CLOSURE.issubset(FAMILY_COMMON_SMOKE_SOURCE_PATHS)
    observed = family_common_smoke_source_sha256()

    assert set(observed) == set(FAMILY_COMMON_SMOKE_SOURCE_PATHS)
    assert all(
        digest == hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path, digest in observed.items()
    )


def test_smoke_spec_uses_registry_trusted_resource_and_small_inner_grid() -> None:
    resource = load_receiver_autonomous_program_resource(
        FIXTURE_ROOT,
        manifest_path=FIXTURE_ROOT / "manifest.json",
        registration_id=AUTONOMOUS_REGISTRATION_ID,
    )
    spec = build_family_common_smoke_spec(resource)
    tuning = spec.penalty_tuning_spec

    assert resource.is_manifest_verified_trusted
    assert resource.review_scope == "synthetic_benchmark_only"
    assert spec.autonomous_program_resource is resource
    assert tuning is not None
    assert tuning.inner_allowed_n_splits == (2,)
    assert tuning.min_inner_train_subjects_per_context == 2
    assert tuning.selection_rule == (
        "subject_equal_paired_delta_one_se_l1_then_l2_sparsity_priority_v2"
    )
    assert {
        (candidate.lambda1_fraction, candidate.lambda2_fraction)
        for candidate in tuning.candidates
    } == {(1.0, 0.0), (0.1, 0.0)}
    assert spec.to_dict()["penalty_tuning_spec"] is not None


def test_candidate_audit_expands_paired_thresholds_and_inner_losses() -> None:
    spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(0.0,),
        inner_allowed_n_splits=(2,),
    )
    losses = {
        1.0: (12.0, 102.0, 1_002.0, 10_002.0),
        0.1: (10.0, 100.0, 1_000.0, 10_000.0),
    }
    evaluations = []
    for candidate in spec.candidates:
        values = losses[candidate.lambda1_fraction]
        evaluations.extend(
            [
                record_penalty_fold_evaluation(
                    candidate,
                    inner_fold_id="inner-1",
                    validation_subject_ids=("s1", "s2"),
                    subject_losses=np.asarray(values[:2]),
                ),
                record_penalty_fold_evaluation(
                    candidate,
                    inner_fold_id="inner-2",
                    validation_subject_ids=("s3", "s4"),
                    subject_losses=np.asarray(values[2:]),
                ),
            ]
        )
    tuning = select_penalty_candidate(
        spec,
        evaluations,
        tuning_scope_id="audit-test",
        training_subject_ids=("s1", "s2", "s3", "s4"),
        inner_fold_ids=("inner-1", "inner-2"),
    )

    records = _tuning_candidate_audit(tuning)
    by_fraction = {record["lambda1_fraction"]: record for record in records}
    strongest = by_fraction[1.0]
    selected = by_fraction[0.1]

    assert strongest["subject_losses"] == list(losses[1.0])
    assert strongest["mean_loss_difference_to_best"] == 2.0
    assert strongest["paired_delta_one_se_threshold"] == 0.0
    assert strongest["within_paired_delta_one_se"] is False
    assert selected["is_selected"] is True
    inner_evaluations = cast(
        list[dict[str, object]], strongest["inner_fold_evaluations"]
    )
    assert len(inner_evaluations) == 2
    assert inner_evaluations[0]["resolved_lambda1"] is None


def test_active_ranking_retains_all_zero_scores_and_dense_ties() -> None:
    table = pd.DataFrame(
        {
            "receiver": ["Receiver"] * 4,
            "sender": ["Sender"] * 4,
            "interaction_id": [
                "active",
                "decoy",
                "active",
                "decoy",
            ],
            "mode": ["state", "state", "ecosystem", "ecosystem"],
            "sender_resolved_strength": [0.0, 0.0, 0.0, 0.5],
            "status": ["structural_zero", "structural_zero", "ok", "ok"],
            "reason_code": ["zero", "zero", None, None],
        }
    )

    ranked = _rank_interaction_table(
        table,
        score_column="sender_resolved_strength",
        active_interaction_id="active",
        receiver="Receiver",
        sender="Sender",
    )
    modes = cast(list[dict[str, object]], ranked["modes"])
    by_mode = {row["mode"]: row for row in modes}
    state = by_mode["state"]
    ecosystem = by_mode["ecosystem"]
    state_active = cast(dict[str, object], state["active_interaction"])
    ecosystem_active = cast(dict[str, object], ecosystem["active_interaction"])

    assert state["all_finite_interaction_means_exactly_zero"] is True
    assert state["n_interactions_with_finite_scores"] == 2
    assert state["n_distinct_finite_mean_values"] == 1
    assert state_active["mean"] == 0.0
    assert state_active["dense_rank"] == 1
    assert state_active["n_interactions_tied_at_rank"] == 2
    assert ecosystem_active["mean"] == 0.0
    assert ecosystem_active["dense_rank"] == 2


def test_cli_defaults_and_claim_boundary_are_explicit() -> None:
    args = build_parser().parse_args([])

    assert tuple(args.scenarios) == DEFAULT_SCENARIOS
    assert ACTIVE_SOURCE_INTERACTION_ID == "CXCL10_CXCR3"
    assert AUDIT_CLAIMS == {
        "development_only": True,
        "biological_validation": False,
        "method_superiority": False,
        "complete_pipeline_oof_certification": False,
        "family_common_full_oof_certification": False,
    }


def test_tracked_family_common_summary_is_semantically_pinned() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    registry = json.loads(INPUT_REGISTRY_PATH.read_text(encoding="utf-8"))

    assert registry["schema_version"] == (
        "crychic-family-common-smoke-input-registry-v1"
    )
    assert {record["scenario"] for record in registry["records"]} == set(
        ALLOWED_SCENARIOS
    )
    assert summary["source_sha256"] == family_common_smoke_source_sha256()
    assert all(summary["checks"].values())
    assert summary["claims"] == AUDIT_CLAIMS
    assert summary["selection_rule"] == (
        "subject_equal_paired_delta_one_se_l1_then_l2_sparsity_priority_v2"
    )
    assert summary["candidate_priority"] == (
        "l1_fraction_desc_then_l2_fraction_desc_v1"
    )
    assert summary["resource_boundary"]["review_scope"] == ("synthetic_benchmark_only")
    assert summary["resource_boundary"]["biological_reference_trusted"] is False
    assert summary["synthetic_input_registry"] == {
        "registry_relative_path": INPUT_REGISTRY_RELATIVE_PATH,
        "registry_sha256": hashlib.sha256(INPUT_REGISTRY_PATH.read_bytes()).hexdigest(),
        "source_manifest": registry["source_manifest"],
    }

    by_scenario = {record["scenario"]: record for record in summary["records"]}
    registry_by_scenario = {
        record["scenario"]: record for record in registry["records"]
    }
    assert set(by_scenario) == set(DEFAULT_SCENARIOS)
    for scenario, record in by_scenario.items():
        pinned = registry_by_scenario[scenario]
        assert record["dataset"] == pinned["dataset_id"]
        assert record["scenario_seed"] == pinned["scenario_seed"]
        assert record["input_sha256"] == pinned["sha256"]
        assert record["n_subjects"] == pinned["n_subjects"]
        assert record["n_samples"] == pinned["n_samples"]
        assert record["simulation_truth"] == {
            "active_interaction": pinned["active_interaction"],
            "expected_state_change": pinned["expected_state_change"],
            "expected_ecosystem_change": pinned["expected_ecosystem_change"],
            "expected_receiver_response": pinned["expected_receiver_response"],
            "expected_integrated_edge": pinned["expected_integrated_edge"],
            "tracked_active_source_interaction_is_positive_truth": (
                pinned["active_interaction"] == ACTIVE_SOURCE_INTERACTION_ID
            ),
        }
    active = by_scenario["active"]
    assert active["selected_penalty_fractions"] == [
        {"count": 2, "lambda1_fraction": 0.1, "lambda2_fraction": 0.0}
    ]
    assert all(value > 0 for value in active["receiver_final_nonzero_family_counts"])
    for key in (
        "active_interaction_member_modes",
        "active_interaction_sender_modes",
    ):
        assert all(
            mode["mean"] > 0
            and mode["dense_rank"] == 1
            and mode["n_interactions_tied_at_rank"] == 1
            for mode in active[key]
        )
    for scenario in ("ligand_only", "receiver_autonomous", "global_null"):
        control = by_scenario[scenario]
        assert control["selected_penalty_fractions"] == [
            {"count": 2, "lambda1_fraction": 1.0, "lambda2_fraction": 0.0}
        ]
        assert control["receiver_final_nonzero_family_counts"] == [0, 0]
        assert control["family_score_status_counts"] == {"structural_zero": 480}
        for key in (
            "active_interaction_member_modes",
            "active_interaction_sender_modes",
        ):
            assert all(mode["mean"] == 0 for mode in control[key])

    assert summary["limitations"] == {
        "control_family_score_ok_rows": {
            "global_null": 0,
            "ligand_only": 0,
            "receiver_autonomous": 0,
        },
        "control_final_nonzero_family_counts": {
            "global_null": [0, 0],
            "ligand_only": [0, 0],
            "receiver_autonomous": [0, 0],
        },
        "single_seed_two_candidate_grid": True,
    }


def test_workspace_family_common_artifact_matches_tracked_summary() -> None:
    if not FULL_ARTIFACT_PATH.exists():
        pytest.skip("workspace family-common smoke artifact is not tracked")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    raw = FULL_ARTIFACT_PATH.read_bytes()
    full = json.loads(raw)
    artifact = summary["full_artifact"]
    expected = build_compact_summary(
        full,
        build_artifact_metadata(
            full,
            serialized=raw,
            relative_workspace_path=artifact["relative_workspace_path"],
        ),
        base_revision=summary["base_revision"],
        generated_on=summary["generated_on"],
    )

    assert summary == expected
