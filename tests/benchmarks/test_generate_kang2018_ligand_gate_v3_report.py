from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from benchmarks.report.generate_kang2018_ligand_gate_v3_report import (
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_SCOPE,
    CLAIMS,
    DIAGNOSTIC_META,
    DIAGNOSTICS,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INPUT_SHA256,
    EXPECTED_PARTITION_MANIFEST_ID,
    EXPECTED_RESOURCE_SIGNATURE,
    GATE_POLICY,
    PARTITION_IDS,
    RECEIVERS,
    ROLE_POLICY,
    SUBJECT_PRIVACY_SEMANTICS,
    generate_report,
)

GENERATED_AT = "2026-07-15T00:00:00+00:00"


def _subject_manifest(digest: str, n_subjects: int) -> dict[str, object]:
    return {
        "digest_domain": "crychic-kang2018-subject-set-manifest-v1",
        "n_subjects": n_subjects,
        "privacy_semantics": SUBJECT_PRIVACY_SEMANTICS,
        "subject_set_digest": digest * 64,
    }


def _partitions(role: str) -> list[dict[str, object]]:
    fold_ids = {
        "default": ("default-fold-a", "default-fold-b"),
        "minimum_effect_0p02": ("effect-fold-a", "effect-fold-b"),
        "receptor_threshold_0p01": ("receptor-fold-a", "receptor-fold-b"),
    }[role]
    training = _subject_manifest("a", 4)
    heldout = _subject_manifest("b", 4)
    return [
        {
            "fold_id": fold_ids[0],
            "fold_partition_id": PARTITION_IDS[0],
            "training_subject_manifest": training,
            "heldout_subject_manifest": heldout,
        },
        {
            "fold_id": fold_ids[1],
            "fold_partition_id": PARTITION_IDS[1],
            "training_subject_manifest": heldout,
            "heldout_subject_manifest": training,
        },
    ]


def _diagnostic_manifest() -> list[dict[str, object]]:
    return [
        {
            "diagnostic_id": diagnostic_id,
            "harmonized_interaction_id": meta["interaction_id"],
            "ligand": meta["ligand"],
            "ligand_subunits": [meta["ligand"]],
            "receptor": meta["artifact_receptor"],
            "receptor_subunits": meta["artifact_receptor"].split("_"),
            "pre_specified_role": meta["role"],
            "source_interaction_id": diagnostic_id,
        }
        for diagnostic_id, meta in (
            (diagnostic_id, DIAGNOSTIC_META[diagnostic_id])
            for diagnostic_id in DIAGNOSTICS
        )
    ]


def _support_values(
    role: str, partition_id: str, diagnostic_id: str
) -> tuple[float, float, float, int, str]:
    partition_index = PARTITION_IDS.index(partition_id)
    effects = {
        "CXCL10_CXCR3": (0.95, 0.92),
        "IFNB1_IFNAR1_IFNAR2": (0.006, 0.004),
        "CCL5_CCR5": (-0.038, 0.061),
    }
    default_p = {
        "CXCL10_CXCR3": (1e-6, 3e-5, 1),
        "IFNB1_IFNAR1_IFNAR2": (0.06, 0.12, 2),
        "CCL5_CCR5": (0.08, 0.20, 3),
    }
    sensitivity_p = {
        "CXCL10_CXCR3": (2e-6, 4e-5, 1),
        "IFNB1_IFNAR1_IFNAR2": (0.99, 1.0, 3),
        "CCL5_CCR5": (0.20, 0.40, 2),
    }
    p_values = sensitivity_p if role == "minimum_effect_0p02" else default_p
    raw_p, holm_p, rank = p_values[diagnostic_id]
    status = "supported" if diagnostic_id == "CXCL10_CXCR3" else "unsupported"
    return effects[diagnostic_id][partition_index], raw_p, holm_p, rank, status


def _support_rows(
    role: str, partitions: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for partition in partitions:
        partition_id = str(partition["fold_partition_id"])
        for diagnostic_id in DIAGNOSTICS:
            effect, raw_p, holm_p, rank, status = _support_values(
                role, partition_id, diagnostic_id
            )
            reason = (
                None
                if status == "supported"
                else "ligand_contrast_holm_adjusted_p_not_below_familywise_alpha"
            )
            for receiver in RECEIVERS:
                rows.append(
                    {
                        "fold_id": partition["fold_id"],
                        "fold_partition_id": partition_id,
                        "training_subject_manifest": partition[
                            "training_subject_manifest"
                        ],
                        "heldout_subject_manifest": partition[
                            "heldout_subject_manifest"
                        ],
                        "receiver": receiver,
                        "diagnostic_id": diagnostic_id,
                        "interaction_id": DIAGNOSTIC_META[diagnostic_id][
                            "interaction_id"
                        ],
                        "sender_functional_id": f"functional-{role}-{partition_id}",
                        "support_id": (
                            f"support-{role}-{partition_id}-{diagnostic_id}-{receiver}"
                        ),
                        "n_complete": 4,
                        "minimum_complete_subjects": 4,
                        "mean_effect": effect,
                        "raw_one_sided_p_value": raw_p,
                        "holm_adjusted_p_value": holm_p,
                        "holm_rank": rank,
                        "multiplicity_family_size": 3,
                        "multiplicity_family_id": (
                            f"family-{role}-{partition_id}-{receiver}"
                        ),
                        "status": status,
                        "reason_code": reason,
                        "ligand_contrast_gate": 1.0 if status == "supported" else 0.0,
                        "ligand_contrast_gate_id": (
                            f"gate-{role}-{partition_id}-{diagnostic_id}-{receiver}"
                        ),
                        "ligand_contrast_gate_status": status,
                        "ligand_contrast_gate_reason_code": reason,
                    }
                )
    return rows


def _receptor_gate(diagnostic_id: str, receiver: str) -> float:
    return {
        ("CXCL10_CXCR3", "CD14+ Monocytes"): 0.005,
        ("CXCL10_CXCR3", "CD8 T cells"): 0.25,
        ("IFNB1_IFNAR1_IFNAR2", "CD14+ Monocytes"): 0.05,
        ("IFNB1_IFNAR1_IFNAR2", "CD8 T cells"): 0.03,
        ("CCL5_CCR5", "CD14+ Monocytes"): 0.20,
        ("CCL5_CCR5", "CD8 T cells"): 0.03,
    }[(diagnostic_id, receiver)]


def _receptor_rows(
    role: str,
    partitions: list[dict[str, object]],
    threshold: float,
) -> list[dict[str, object]]:
    rows = []
    for partition in partitions:
        partition_id = str(partition["fold_partition_id"])
        for diagnostic_id in DIAGNOSTICS:
            for receiver in RECEIVERS:
                gate = _receptor_gate(diagnostic_id, receiver)
                rows.append(
                    {
                        "fold_id": partition["fold_id"],
                        "fold_partition_id": partition_id,
                        "training_subject_manifest": partition[
                            "training_subject_manifest"
                        ],
                        "heldout_subject_manifest": partition[
                            "heldout_subject_manifest"
                        ],
                        "receiver": receiver,
                        "diagnostic_id": diagnostic_id,
                        "interaction_id": DIAGNOSTIC_META[diagnostic_id][
                            "interaction_id"
                        ],
                        "driver_id": DIAGNOSTIC_META[diagnostic_id]["ligand"],
                        "receptor_gate": gate,
                        "receptor_gate_threshold": threshold,
                        "receptor_eligible": gate >= threshold,
                        "receptor_gate_policy": dict(GATE_POLICY),
                        "receptor_gate_manifest_id": (
                            f"receptor-manifest-{role}-{partition_id}-{receiver}"
                        ),
                        "receptor_evidence_digest": (
                            f"evidence-{partition_id}-{diagnostic_id}-{receiver}"
                        ),
                    }
                )
    return rows


def _resources() -> dict[str, object]:
    return {
        "cellchat": {
            "resource_id": "cellchatdb_v2",
            "manifest_digest": EXPECTED_RESOURCE_SIGNATURE["cellchat_manifest_digest"],
            "selected_interaction_id_digest": EXPECTED_RESOURCE_SIGNATURE[
                "cellchat_selected_digest"
            ],
            "n_selected_interactions": 3,
        },
        "nichenet": {
            "resource_id": "nichenet_v2_ligand_target_human",
            "manifest_digest": EXPECTED_RESOURCE_SIGNATURE["nichenet_manifest_digest"],
            "diagnostic_target_prior": {
                "selected_target_link_digest": EXPECTED_RESOURCE_SIGNATURE[
                    "nichenet_selected_digest"
                ],
                "n_drivers": 3,
                "n_links": 60,
                "n_targets": 57,
            },
        },
    }


def _payload(role: str) -> dict[str, object]:
    minimum_effect, receptor_threshold = ROLE_POLICY[role]
    partitions = _partitions(role)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "scope": ARTIFACT_SCOPE,
        "claims": dict(CLAIMS),
        "sensitivity_comparison_policy": {
            "aligned_by": "subject_partition_manifest_id",
            "contrast_support_comparison_allowed": True,
            "family_member_sender_score_comparison_allowed": False,
            "scope": "contrast_support_only",
            "reason": "downstream fits are threshold-sensitive",
        },
        "configuration": {
            "sha256": EXPECTED_CONFIG_SHA256,
            "ligand_contrast_minimum_effect": minimum_effect,
            "receptor_gate_threshold": receptor_threshold,
            "crychic_config": {
                "counts_layer": "counts",
                "random_seed": 31072018,
            },
            "crossfit_spec": {
                "allowed_n_splits": [2],
                "autonomous_program_use_scope": "biological_analysis",
                "outer_fold_partition_seed": 18021988,
                "receptor_gate_threshold": receptor_threshold,
                "repeat_id": f"repeat-{role}",
                "spec_id": f"spec-{role}",
                "training_spec_id": f"training-{role}",
            },
        },
        "input": {
            "expected_sha256": EXPECTED_INPUT_SHA256,
            "observed_sha256": EXPECTED_INPUT_SHA256,
            "checksum_verified": True,
            "conversion_lineage_verified": True,
            "analysis_shape": [7427, 32938],
            "subject_manifest": _subject_manifest("c", 8),
        },
        "resources": _resources(),
        "source_provenance": {
            "git_commit": "1" * 40,
            "git_worktree_dirty": False,
            "git_status_porcelain": [],
            "source_sha256": {"runner.py": "2" * 64},
        },
        "frozen_manifest": {
            "diagnostic_interactions": _diagnostic_manifest(),
            "subject_partition": {
                "subject_partition_manifest_id": EXPECTED_PARTITION_MANIFEST_ID,
                "outer_fold_partition_seed": 18021988,
                "partition_seed_lineage": {
                    "root_seed": 18021988,
                    "derived_seed": 42,
                    "path": ["subject_crossfit_outer_partition_v1", "repeat=0"],
                },
                "partitions": partitions,
                "sensitivity_alignment_scope": "contrast_support_only",
                (
                    "downstream_inner_tuned_scores_comparable_across_sensitivity_runs"
                ): False,
            },
        },
        "paired_summaries": {
            "within_run_descriptive_only": True,
            "comparable_across_threshold_sensitivity_runs": False,
            "family": {"must_not_be_read": 99},
            "member": {"must_not_be_read": 99},
            "sender": {"must_not_be_read": 99},
        },
        "crossfit_audit": {
            "complete_pipeline_oof_certified": False,
            "receiver_autonomous_nuisance_intentionally_unresolved": True,
        },
        "fold_receiver_interaction_supports": _support_rows(role, partitions),
        "fold_receiver_interaction_receptor_gates": _receptor_rows(
            role, partitions, receptor_threshold
        ),
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = []
    for role, filename in (
        ("default", "default.json"),
        ("minimum_effect_0p02", "minimum_effect.json"),
        ("receptor_threshold_0p01", "receptor.json"),
    ):
        path = tmp_path / filename
        path.write_text(
            json.dumps(_payload(role), sort_keys=True) + "\n", encoding="utf-8"
        )
        paths.append(path)
    return paths[0], paths[1], paths[2]


def _tree_sha256(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run_report(
    inputs: tuple[Path, Path, Path], output: Path, summary: Path
) -> dict[str, object]:
    return generate_report(
        default_artifact=inputs[0],
        minimum_effect_artifact=inputs[1],
        receptor_artifact=inputs[2],
        output_dir=output,
        generated_at=GENERATED_AT,
        summary_output=summary,
    )


def test_report_is_byte_deterministic_private_and_fold_deduplicated(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    first = tmp_path / "report-first"
    second = tmp_path / "report-second"
    first_summary = tmp_path / "summary-first.json"
    second_summary = tmp_path / "summary-second.json"

    result = _run_report(inputs, first, first_summary)
    _run_report(inputs, second, second_summary)

    assert result["artifact_count"] == 13
    assert _tree_sha256(first) == _tree_sha256(second)
    assert first_summary.read_bytes() == second_summary.read_bytes()

    metrics = json.loads((first / "metrics_summary.json").read_text())
    support = {
        (row["artifact_role"], row["diagnostic_id"]): row
        for row in metrics["ligand_support_by_fold"]
    }
    assert support[("default", "CXCL10_CXCR3")]["supported_fold_partitions"] == 2
    assert support[("default", "CXCL10_CXCR3")]["n_fold_partitions"] == 2
    assert support[("default", "IFNB1_IFNAR1_IFNAR2")]["supported_fold_partitions"] == 0
    assert metrics["comparison_contract"]["alignment_key"] == "fold_partition_id"
    assert (
        metrics["comparison_contract"][
            "receiver_duplicated_ligand_rows_are_independent"
        ]
        is False
    )
    assert (
        metrics["comparison_contract"]["family_member_sender_score_comparison_allowed"]
        is False
    )

    main_rows = pd.read_csv(first / "source_data/figure01_fold_ligand_contrast.csv")
    support_audit = pd.read_csv(first / "source_data/support_audit_rows.csv")
    receptor_audit = pd.read_csv(first / "source_data/receptor_gate_audit_rows.csv")
    assert len(main_rows) == 12
    assert len(support_audit) == 36
    assert len(receptor_audit) == 24
    assert set(main_rows["evidence_unit"]) == {"training_fold_partition"}

    matrix = metrics["support_receptor_matrix"]
    cxcl10_cd8 = next(
        row
        for row in matrix
        if row["artifact_role"] == "default"
        and row["diagnostic_id"] == "CXCL10_CXCR3"
        and row["receiver"] == "CD8 T cells"
    )
    cxcl10_mono = next(
        row
        for row in matrix
        if row["artifact_role"] == "default"
        and row["diagnostic_id"] == "CXCL10_CXCR3"
        and row["receiver"] == "CD14+ Monocytes"
    )
    assert cxcl10_cd8["joint_gate_eligible_folds"] == 2
    assert cxcl10_mono["joint_gate_eligible_folds"] == 0

    report = (first / "REPORT.md").read_text(encoding="utf-8")
    assert "2/2 training partitions, not 4/4 independent observations" in report
    assert "not result-layer inferential p-values or q-values" in report
    assert "exogenous IFN-beta attribution guard" in report
    assert "not evidence of causal communication" in report
    assert "method superiority" in report
    for stem in (
        "figure01_fold_ligand_contrast",
        "figure02_support_receptor_matrix",
    ):
        for suffix in ("png", "pdf", "svg"):
            assert (first / f"figures/{stem}.{suffix}").stat().st_size > 0
    svg = (first / "figures/figure01_fold_ligand_contrast.svg").read_text()
    assert GENERATED_AT in svg
    assert (
        b"D:20260715000000Z"
        in (first / "figures/figure01_fold_ligand_contrast.pdf").read_bytes()
    )

    summary_text = first_summary.read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["generated_on"] == GENERATED_AT
    assert summary["claims"] == CLAIMS
    assert summary["source_commit"] == "1" * 40
    assert summary["frozen_identities"]["input_h5ad_sha256"] == (EXPECTED_INPUT_SHA256)
    assert "subject_set_digest" not in summary_text
    assert "paired_summaries" not in summary_text
    assert "member_scores" not in summary_text
    assert str(tmp_path) not in summary_text
    assert all(
        item["relative_path"].startswith("benchmark_work/")
        for item in summary["input_artifacts"]
    )


def _mutate_json(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("mutation", "match"),
    (
        (lambda payload: payload.update(schema_version="wrong"), "schema"),
        (
            lambda payload: payload["claims"].update(biological_validation=True),
            "claim boundary",
        ),
        (
            lambda payload: payload["input"].update(observed_sha256="f" * 64),
            "input identity",
        ),
        (
            lambda payload: payload["configuration"].update(sha256="f" * 64),
            "config checksum",
        ),
        (
            lambda payload: payload["configuration"]["crossfit_spec"].update(
                autonomous_program_use_scope="algorithm_diagnostic"
            ),
            "biological autonomous programs",
        ),
        (
            lambda payload: payload["resources"]["cellchat"].update(
                manifest_digest="f" * 64
            ),
            "resource identity",
        ),
        (
            lambda payload: payload["source_provenance"].update(git_commit="3" * 40),
            "source provenance",
        ),
        (
            lambda payload: payload["sensitivity_comparison_policy"].update(
                family_member_sender_score_comparison_allowed=True
            ),
            "sensitivity policy",
        ),
    ),
)
def test_shared_contract_mutations_fail_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    _mutate_json(inputs[1], mutation)
    with pytest.raises(ValueError, match=match):
        _run_report(inputs, tmp_path / "report", tmp_path / "summary.json")


def test_missing_exact_support_or_receptor_key_fails_closed(tmp_path: Path) -> None:
    for field in (
        "fold_receiver_interaction_supports",
        "fold_receiver_interaction_receptor_gates",
    ):
        case = tmp_path / field
        case.mkdir()
        inputs = _write_inputs(case)

        def remove_row(payload: dict[str, Any], *, target: str = field) -> None:
            payload[target].pop()

        _mutate_json(inputs[1], remove_row)
        with pytest.raises(ValueError, match="exactly 12 rows"):
            _run_report(inputs, case / "report", case / "summary.json")


def test_receiver_duplicate_mismatch_fails_closed(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)

    def change_duplicate(payload: dict[str, Any]) -> None:
        rows = payload["fold_receiver_interaction_supports"]
        changed = copy.deepcopy(rows[1])
        changed["mean_effect"] = float(changed["mean_effect"]) + 0.01
        rows[1] = changed

    _mutate_json(inputs[0], change_duplicate)
    with pytest.raises(ValueError, match="receiver-duplicated support rows"):
        _run_report(inputs, tmp_path / "report", tmp_path / "summary.json")


def test_partition_id_not_fold_id_is_the_only_cross_artifact_key(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    # Fixture fold IDs differ by role; the unmodified report succeeds.
    _run_report(inputs, tmp_path / "valid-report", tmp_path / "valid-summary.json")

    def change_partition(payload: dict[str, Any]) -> None:
        payload["fold_receiver_interaction_supports"][0]["fold_partition_id"] = "f" * 64

    _mutate_json(inputs[1], change_partition)
    with pytest.raises(ValueError, match="unexpected support key"):
        _run_report(inputs, tmp_path / "bad-report", tmp_path / "bad-summary.json")
