from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from benchmarks.metrics import finalize_multicondition as module
from benchmarks.report import generate_multicondition_report as report_module

EDGES = (
    ("S", "R", "I1", "L1", "R1"),
    ("S", "R", "I2", "L2", "R2"),
    ("S", "R", "I3", "L3", "R3"),
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adapter_table(
    dataset: str,
    method: str,
    subjects: list[tuple[str, str]],
    scores: dict[tuple[str, str], tuple[float, float, float]],
    *,
    track: str = "lr_stlr",
    universe_id: str | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    universe = universe_id or f"{dataset}-common-v1"
    for subject, context in subjects:
        sample = f"{subject}_{context}"
        for edge, score in zip(EDGES, scores[(subject, context)], strict=True):
            sender, receiver, interaction, ligand, receptor = edge
            rows.append(
                {
                    "schema_version": "crychic-external-interactions-long-v1",
                    "run_id": run_id or f"run-{dataset}-{method}",
                    "dataset_id": dataset,
                    "method_id": method,
                    "method_version": "1.0",
                    "analysis_track": track,
                    "resource_mode": "H-common",
                    "resource_id": "fixture-common",
                    "resource_version": "2026-07-12",
                    "universe_id": universe,
                    "universe_member": True,
                    "universe_size": len(EDGES),
                    "sample_id": sample,
                    "subject_id": subject,
                    "context_json": json.dumps({"condition": context}),
                    "sender": sender,
                    "receiver": receiver,
                    "interaction_id": interaction,
                    "native_interaction_id": interaction,
                    "interaction_direction": "ligand_to_receptor",
                    "ligand": ligand,
                    "receptor": receptor,
                    "target": "T1" if track != "lr_stlr" else "",
                    "score": score,
                    "score_name": "fixture_strength",
                    "score_direction": "higher",
                    "rank": np.nan,
                    "specificity_score": np.nan,
                    "specificity_score_name": "not_emitted",
                    "within_dataset_p_value": np.nan,
                    "within_dataset_p_value_semantics": "not_emitted",
                    "differential_effect": np.nan,
                    "differential_p_value": np.nan,
                    "differential_q_value": np.nan,
                    "status": "ok",
                    "reason_code": "external_adapter_no_between_condition_inference",
                }
            )
    return pd.DataFrame(rows)


def _write_adapter(
    root: Path,
    table: pd.DataFrame,
    name: str,
    *,
    manifest_status: str = "complete",
    score_views: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    table_path = run_dir / "interactions_long.parquet"
    table.to_parquet(table_path, index=False)
    first = table.iloc[0]
    manifest: dict[str, object] = {
        "schema_version": "crychic-external-adapter-manifest-v1",
        "dataset_id": first["dataset_id"],
        "run_id": first["run_id"],
        "status": manifest_status,
        "elapsed_seconds": 1.5,
        "method": {
            "id": first["method_id"],
            "version": first["method_version"],
        },
        "resource": {
            "resource_id": first["resource_id"],
            "version": first["resource_version"],
            "mode": first["resource_mode"],
        },
        "score_semantics": {"primary_score": first["score_name"]},
        "environment": {"threads": 1},
        "output": {
            "table": table_path.name,
            "rows": len(table),
            "sha256": _sha256(table_path),
        },
    }
    if score_views is not None:
        manifest["source_result"] = {"score_views": score_views}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "manifest": manifest_path.relative_to(root).as_posix(),
        "long_table": table_path.relative_to(root).as_posix(),
    }


def _failed_adapter(root: Path, dataset: str) -> dict[str, object]:
    run_dir = root / "runs" / "failed"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "crychic-external-adapter-manifest-v1",
                "dataset_id": dataset,
                "run_id": "failed-run",
                "status": "failed",
                "method": {"id": "failed_method", "version": "1"},
                "resource": {
                    "resource_id": "fixture-common",
                    "version": "2026-07-12",
                    "mode": "H-common",
                },
                "score_semantics": {"primary_score": "fixture_strength"},
                "environment": {"threads": 1},
            }
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path.relative_to(root).as_posix(),
        "long_table": "runs/failed/interactions_long.parquet",
        "identity": {"analysis_track": "lr_stlr"},
    }


def _fixture(root: Path) -> Path:
    paired_subjects = [
        (subject, context)
        for subject in ("p1", "p2", "p3")
        for context in ("Normal", "Tumor")
    ]
    normal = {(subject, "Normal"): (0.1, 0.5, 0.9) for subject in ("p1", "p2", "p3")}
    tumor = {
        ("p1", "Tumor"): (0.9, 0.2, 0.5),
        ("p2", "Tumor"): (0.8, 0.1, 0.6),
        ("p3", "Tumor"): (0.7, 0.3, 0.4),
    }
    paired_scores = normal | tumor
    paired_m1 = _write_adapter(
        root,
        _adapter_table("synthetic_paired", "m1", paired_subjects, paired_scores),
        "paired_m1",
    )
    paired_m2 = _write_adapter(
        root,
        _adapter_table(
            "synthetic_paired",
            "m2",
            paired_subjects,
            {
                key: (value[0], value[1] + 0.01, value[2] + 0.02)
                for key, value in paired_scores.items()
            },
        ),
        "paired_m2",
    )
    crychic_primary = _adapter_table(
        "synthetic_paired",
        "crychic",
        paired_subjects,
        paired_scores,
        run_id="crychic-primary-view",
    )
    crychic_sensitivity = _adapter_table(
        "synthetic_paired",
        "crychic",
        paired_subjects,
        {key: (value[2], value[1], value[0]) for key, value in paired_scores.items()},
        run_id="crychic-sensitivity-view",
    )
    crychic = _write_adapter(
        root,
        pd.concat([crychic_primary, crychic_sensitivity], ignore_index=True),
        "paired_crychic",
        score_views=[
            {
                "run_id": "crychic-primary-view",
                "contrast_candidates": ["global:'Tumor'"],
            },
            {
                "run_id": "crychic-sensitivity-view",
                "contrast_candidates": ["global:'Normal'"],
            },
        ],
    )
    crychic["score_views"] = [
        {
            "run_id": "crychic-primary-view",
            "contrast_candidate": "global:'Tumor'",
            "label": "tumor_functional",
            "role": "primary",
            "contrast": "Tumor_vs_Normal",
        },
        {
            "run_id": "crychic-sensitivity-view",
            "contrast_candidate": "global:'Normal'",
            "label": "normal_functional",
            "role": "sensitivity",
            "contrast": "Tumor_vs_Normal",
        },
    ]

    unpaired_subjects = [
        *((f"c{index}", "Ctrl") for index in range(1, 5)),
        *((f"t{index}", "Case") for index in range(1, 5)),
    ]
    unpaired_scores: dict[tuple[str, str], tuple[float, float, float]] = {}
    for index in range(1, 5):
        unpaired_scores[(f"c{index}", "Ctrl")] = (
            0.1 + index * 0.01,
            0.5 - index * 0.01,
            0.9,
        )
        unpaired_scores[(f"t{index}", "Case")] = (
            0.9 - index * 0.01,
            0.2 + index * 0.01,
            0.5,
        )
    unpaired = _write_adapter(
        root,
        _adapter_table("real_unpaired", "m1", unpaired_subjects, unpaired_scores),
        "unpaired_m1",
    )
    track_b = _write_adapter(
        root,
        _adapter_table(
            "real_unpaired",
            "nichenet",
            unpaired_subjects,
            unpaired_scores,
            track="ligand_target_program",
            universe_id="real-unpaired-track-b-v1",
        ),
        "nichenet_track_b",
    )
    failed = _failed_adapter(root, "real_unpaired")

    truth = pd.DataFrame(
        [
            {
                "dataset": "synthetic_paired",
                "contrast": "Tumor_vs_Normal",
                "universe_id": "synthetic_paired-common-v1",
                "sender": sender,
                "receiver": receiver,
                "interaction_id": interaction,
                "ligand": ligand,
                "receptor": receptor,
                "is_positive": int(interaction == "I1"),
                "truth_scope": "simulation",
            }
            for sender, receiver, interaction, ligand, receptor in EDGES
        ]
    )
    truth_path = root / "simulation_truth.tsv"
    truth.to_csv(truth_path, sep="\t", index=False)
    biology_path = root / "supportive_biology.yaml"
    biology_path.write_text(
        "truth_set_id: fixture\n"
        "datasets:\n"
        "  real_unpaired:\n"
        "    expected_observations:\n"
        "      - id: known_program\n"
        "        direction: Case_up\n"
        "        metric: supportive_rank\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": module.SPEC_SCHEMA_VERSION,
        "generated_at": "2026-07-12T12:00:00+08:00",
        "supportive_biology_truth": biology_path.name,
        "parameters": {
            "top_k": 2,
            "minimum_shared_edges": 3,
            "n_bootstrap": 20,
            "n_split_repeats": 10,
            "min_subjects_per_half": 2,
            "random_seed": 7,
        },
        "datasets": [
            {
                "dataset_id": "synthetic_paired",
                "context_key": "condition",
                "truth_scope": "simulation",
                "simulation_truth": truth_path.name,
                "design": {
                    "n_cells": 100,
                    "n_samples": 6,
                    "n_subjects": 3,
                    "n_contexts": 2,
                    "n_cell_types": 2,
                },
                "comparison": {
                    "design": "paired",
                    "reference": "Normal",
                    "target": "Tumor",
                    "contrast": "Tumor_vs_Normal",
                    "min_subjects": 3,
                    "primary_score_view": "global:'Tumor'",
                },
                "adapter_runs": [paired_m1, paired_m2, crychic],
            },
            {
                "dataset_id": "real_unpaired",
                "context_key": "condition",
                "truth_scope": "real_data",
                "design": {
                    "n_cells": 120,
                    "n_samples": 8,
                    "n_subjects": 8,
                    "n_contexts": 2,
                    "n_cell_types": 2,
                },
                "comparison": {
                    "design": "unpaired",
                    "reference": "Ctrl",
                    "target": "Case",
                    "contrast": "Case_vs_Ctrl",
                    "min_subjects": 3,
                },
                "adapter_runs": [unpaired, track_b, failed],
            },
            {
                "dataset_id": "unsupported_design",
                "context_key": "condition",
                "truth_scope": "real_data",
                "design": {
                    "n_cells": 20,
                    "n_samples": 3,
                    "n_subjects": 2,
                    "n_contexts": 3,
                    "n_cell_types": 2,
                },
                "comparison": {"design": "unsupported"},
                "adapter_runs": [],
            },
        ],
    }
    spec = root / "finalize.json"
    spec.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return spec


def _molecular_lr_crosswalk() -> pd.DataFrame:
    molecular_ids = {"I1": "molecular-a", "I2": "molecular-a", "I3": "molecular-b"}
    return pd.DataFrame(
        [
            {
                "resource_id": "fixture-common",
                "resource_version": "2026-07-12",
                "resource_manifest_digest": "d" * 64,
                "resource_bundle_content_id": "fixture-bundle-content",
                "interaction_id": interaction,
                "source_interaction_id": f"source-{interaction}",
                "mapping_status": "mapped",
                "reason_code": None,
                "molecular_lr_equivalence_id": molecular_ids[interaction],
                "mechanistic_variant_id": f"variant-{interaction}",
                "molecular_lr_equivalence_universe_id": "fixture-molecular-universe",
                "molecular_lr_axis_id": "fixture-molecular-axis",
                "mechanistic_variant_axis_id": "fixture-variant-axis",
                "mapping_axis_id": "fixture-mapping-axis",
            }
            for _, _, interaction, _, _ in EDGES
        ]
    )


def _bind_molecular_lr_crosswalk(
    root: Path,
    run: dict[str, Any],
    *,
    table: pd.DataFrame | None = None,
    name: str = "molecular_lr_crosswalk",
    manifest_output_sha256: str | None = None,
) -> tuple[Path, Path]:
    crosswalk = _molecular_lr_crosswalk() if table is None else table
    crosswalk_path = root / f"{name}.tsv"
    crosswalk.to_csv(crosswalk_path, sep="\t", index=False)
    crosswalk_sha256 = _sha256(crosswalk_path)
    first = crosswalk.iloc[0]
    manifest = {
        "schema_version": module.MOLECULAR_LR_CROSSWALK_MANIFEST_SCHEMA_VERSION,
        "output": {
            "sha256": manifest_output_sha256 or crosswalk_sha256,
            "rows": len(crosswalk),
        },
        "resource": {
            "resource_id": first["resource_id"],
            "resource_version": first["resource_version"],
            "resource_manifest_digest": first["resource_manifest_digest"],
            "resource_bundle_content_id": first["resource_bundle_content_id"],
        },
        "axes": {
            "molecular_lr_equivalence_universe_id": first[
                "molecular_lr_equivalence_universe_id"
            ],
            "molecular_lr_axis_id": first["molecular_lr_axis_id"],
            "mechanistic_variant_axis_id": first["mechanistic_variant_axis_id"],
            "mapping_axis_id": first["mapping_axis_id"],
        },
    }
    manifest_path = root / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    run["molecular_lr_crosswalk"] = {
        "path": crosswalk_path.name,
        "sha256": crosswalk_sha256,
        "manifest": {
            "path": manifest_path.name,
            "sha256": _sha256(manifest_path),
        },
    }
    return crosswalk_path, manifest_path


def _bind_effect_contract(
    root: Path,
    dataset: dict[str, Any],
    sample_design: pd.DataFrame,
    effect_models: list[dict[str, object]],
    *,
    name: str,
) -> Path:
    path = root / name
    sample_design.to_csv(path, sep="\t", index=False)
    dataset["sample_design"] = {
        "path": path.name,
        "sha256": _sha256(path),
    }
    dataset["effect_models"] = effect_models
    return path


def _paired_effect_sample_design() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"{subject}_{condition}",
                "subject_id": subject,
                "condition": condition,
            }
            for subject in ("p1", "p2", "p3")
            for condition in ("Normal", "Tumor")
        ]
    )


def _unpaired_effect_sample_design(*, confounded: bool) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for condition, prefix in (("Ctrl", "c"), ("Case", "t")):
        for index in range(1, 5):
            rows.append(
                {
                    "sample_id": f"{prefix}{index}_{condition}",
                    "subject_id": f"{prefix}{index}",
                    "condition": condition,
                    "batch": (
                        "b1"
                        if confounded and condition == "Ctrl"
                        else "b2"
                        if confounded
                        else f"b{1 + index % 2}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _mark_adapter_edge_missing(
    root: Path,
    run: dict[str, Any],
    *,
    interaction_id: str,
) -> None:
    table_path = root / str(run["long_table"])
    table = pd.read_parquet(table_path)
    selected = table["interaction_id"].astype(str).eq(interaction_id)
    table.loc[selected, "status"] = "missing"
    table.loc[selected, "score"] = np.nan
    table.to_parquet(table_path, index=False)
    manifest_path = root / str(run["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = _sha256(table_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_finalize_emits_paired_unpaired_track_b_and_ne_outputs(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    output = tmp_path / "final"

    manifest = module.finalize(spec, output)

    assert manifest["guardrails"]["real_data_edge_auroc_reported"] is False
    assert manifest["guardrails"]["preregistered_track_b_primary_reported"] is False
    assert (
        manifest["guardrails"]["track_b_proxy_mislabelled_as_native_nichenet"] is False
    )
    expected = set(module.METRIC_FILES.values())
    assert expected == {path.name for path in (output / "metrics").glob("*.tsv")}
    report_inputs = json.loads((output / "report_inputs.json").read_text())
    assert report_inputs["schema_version"] == module.REPORT_INPUT_SCHEMA_VERSION
    assert "molecular_lr_crosswalks" not in report_inputs
    assert manifest["ranking_parameters"]["lr_family_mapping_status"] == (
        "not_available_in_score_contract"
    )
    assert manifest["molecular_lr_crosswalk_bindings"] == []
    assert "effect_model_sample_designs" not in report_inputs
    assert "exploratory_effect_models" not in report_inputs
    assert not (output / "derived/d_common_edge_effects.parquet").exists()
    assert not (output / "derived/repeated_measures_edge_effects.parquet").exists()
    assert report_inputs["supportive_biology_dataset_aliases"] == {}
    assert report_inputs["dataset_truth_scopes"] == {
        "real_unpaired": "real_data",
        "synthetic_paired": "simulation",
        "unsupported_design": "real_data",
    }
    assert len(report_inputs["score_tables"]) == 6

    primary = pd.read_csv(output / "metrics/loso_primary_endpoint.tsv", sep="\t")
    paired = primary[
        primary["dataset"].eq("synthetic_paired") & primary["method"].eq("m1")
    ]
    unpaired = primary[
        primary["dataset"].eq("real_unpaired") & primary["method"].eq("m1")
    ]
    track_b = primary[primary["method"].eq("nichenet")]
    failed = primary[primary["method"].eq("failed_method")]
    assert set(paired["status"]) == {"observed"}
    assert set(paired["truth_scope"]) == {"simulation"}
    assert set(unpaired["status"]) == {"observed"}
    assert set(unpaired["truth_scope"]) == {"real_data"}
    assert set(unpaired["interval_type"]) == {"empirical_repeated_split_quantiles"}
    assert set(track_b["reason_code"]) == {
        "track_b_ligand_target_program_not_lr_stlr_comparable"
    }
    assert set(track_b["truth_scope"]) == {"real_data"}
    assert set(failed["reason_code"]) == {"adapter_long_table_missing_after_failed_run"}
    assert set(failed["truth_scope"]) == {"real_data"}
    unsupported = primary[primary["dataset"].eq("unsupported_design")]
    assert set(unsupported["reason_code"]) == {"unsupported_comparison_design"}
    assert set(unsupported["truth_scope"]) == {"real_data"}
    assert not primary["dataset"].eq("__cross_dataset__").any()

    stability = pd.read_csv(output / "metrics/stability_summary.tsv", sep="\t")
    assert set(stability["truth_scope"]) == {"real_data", "simulation"}
    ranking = pd.read_csv(output / "metrics/ranking_agreement_summary.tsv", sep="\t")
    unpaired_ranking = ranking[
        ranking["dataset"].eq("real_unpaired") & ranking["method"].eq("m1")
    ]
    assert "observed" in set(unpaired_ranking["status"])
    assert set(unpaired_ranking["n_bootstrap"]) == {2000}
    assert set(unpaired_ranking["n_split_repeats"]) == {200}
    assert set(unpaired_ranking["random_seed"]) == {20260712}
    assert set(
        unpaired_ranking.loc[
            unpaired_ranking["ranking_level"].eq("lr_family"), "reason_code"
        ]
    ) == {"lr_family_mapping_not_available_in_score_contract"}
    track_b_ranking = ranking[ranking["method"].eq("nichenet")]
    assert set(track_b_ranking["status"]) == {"not_estimable"}
    assert set(track_b_ranking["reason_code"]) == {
        "track_b_ligand_target_program_not_lr_stlr_comparable"
    }
    unsupported_ranking = ranking[ranking["dataset"].eq("unsupported_design")]
    assert set(unsupported_ranking["status"]) == {"not_estimable"}
    assert set(unsupported_ranking["reason_code"]) == {"unsupported_comparison_design"}

    concordance = pd.read_csv(output / "metrics/concordance_summary.tsv", sep="\t")
    assert set(concordance["truth_scope"]) == {"real_data", "simulation"}
    real_concordance = concordance[concordance["truth_scope"].eq("real_data")]
    assert set(real_concordance["reason_code"]) == {
        "no_comparable_real_data_lr_method_pair"
    }
    simulation_concordance = concordance[concordance["truth_scope"].eq("simulation")]
    method_pairs = {
        tuple(sorted(pair))
        for pair in simulation_concordance[["method_left", "method_right"]].itertuples(
            index=False, name=None
        )
    }
    assert method_pairs == {("crychic", "m1"), ("crychic", "m2"), ("m1", "m2")}
    semantics = pd.concat(
        [concordance["score_semantics_left"], concordance["score_semantics_right"]]
    ).astype(str)
    assert not semantics.str.contains("normal_functional", regex=False).any()
    simulation = pd.read_csv(output / "metrics/simulation_truth_metrics.tsv", sep="\t")
    assert set(simulation["truth_scope"]) == {"simulation"}
    assert simulation.loc[simulation["status"].eq("observed"), "auroc"].notna().all()
    biology = pd.read_csv(output / "metrics/biology_support.tsv", sep="\t")
    assert set(biology["support_status"]) == {"not_evaluated"}
    assert set(biology["reason_code"]) == {"supportive_evidence_not_supplied"}
    coverage = pd.read_csv(output / "metrics/coverage_summary.tsv", sep="\t")
    assert set(coverage["truth_scope"]) == {"real_data", "simulation"}

    score_index = pd.read_csv(output / "score_tables/score_table_index.tsv", sep="\t")
    assert set(score_index["analysis_track"]) == {
        "lr_stlr",
        "ligand_target_program",
    }
    crychic_views = score_index[score_index["method"].eq("crychic")]
    assert set(crychic_views["score_view_role"]) == {"primary", "sensitivity"}
    assert crychic_views["score_semantics"].nunique() == 2
    sensitivity = pd.read_csv(
        output / "derived/sensitivity_primary_endpoint.tsv", sep="\t"
    )
    assert set(sensitivity["score_view_role"]) == {"sensitivity"}
    assert set(sensitivity["truth_scope"]) == {"simulation"}
    loso_folds = pd.read_parquet(output / "derived/paired_loso_folds.parquet")
    assert set(loso_folds["truth_scope"]) == {"simulation"}
    performance = pd.read_csv(output / "metrics/performance_summary.tsv", sep="\t")
    assert set(performance["truth_scope"]) == {"real_data", "simulation"}

    resolved_report_inputs = report_module.resolve_inputs(output)
    primary_report_input = resolved_report_inputs.metric_tables["loso_primary"]
    assert primary_report_input is not None
    report_primary = pd.read_csv(primary_report_input, sep="\t")
    assert set(report_primary["truth_scope"]) == {"real_data", "simulation"}
    checksums = pd.read_csv(output / "SHA256SUMS.tsv", sep="\t")
    present = checksums[checksums["status"].ne("missing")]
    assert present["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    report_output = tmp_path / "report"
    report_module.generate(
        output,
        report_output,
        "2026-07-12T12:00:00+08:00",
        render_pdf=False,
    )
    report_manifest = json.loads((report_output / "report_manifest.json").read_text())
    assert report_manifest["guardrails"]["real_data_edge_auroc_reported"] is False


def test_finalize_applies_checksum_bound_molecular_lr_crosswalk(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    datasets = cast(list[dict[str, Any]], payload["datasets"])
    runs = cast(list[dict[str, Any]], datasets[1]["adapter_runs"])
    crosswalk_path, crosswalk_manifest_path = _bind_molecular_lr_crosswalk(
        tmp_path, runs[0]
    )
    spec.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "molecular_crosswalk_final"
    manifest = module.finalize(spec, output)

    agreement = pd.read_csv(
        output / "metrics/ranking_agreement_summary.tsv", sep="\t"
    )
    family = agreement[
        agreement["dataset"].eq("real_unpaired")
        & agreement["method"].eq("m1")
        & agreement["ranking_level"].eq("lr_family")
    ]
    assert set(family["status"]) == {"observed"}
    assert family["reason_code"].isna().all()
    assert set(family["ranking_universe_size"]) == {2}

    score_index = pd.read_csv(
        output / "score_tables/score_table_index.tsv", sep="\t"
    )
    score_row = score_index[
        score_index["dataset"].eq("real_unpaired")
        & score_index["method"].eq("m1")
    ].iloc[0]
    assert score_row["molecular_lr_crosswalk_sha256"] == _sha256(crosswalk_path)
    assert score_row["molecular_lr_crosswalk_manifest_sha256"] == _sha256(
        crosswalk_manifest_path
    )
    score_table = pd.read_parquet(output / str(score_row["output_path"]))
    assert score_table["molecular_lr_equivalence_id"].notna().all()
    assert set(score_table["molecular_lr_equivalence_id"]) == {
        "molecular-a",
        "molecular-b",
    }

    assert manifest["ranking_parameters"]["lr_family_mapping_status"] == (
        "available_for_checksum_bound_score_tables"
    )
    assert manifest["ranking_parameters"]["molecular_lr_crosswalk_score_tables"] == 1
    binding = manifest["molecular_lr_crosswalk_bindings"][0]
    assert binding["status"] == "applied"
    assert binding["score_tables_attached"] == 1
    assert binding["molecular_lr_axis_id"] == "fixture-molecular-axis"
    report_inputs = json.loads((output / "report_inputs.json").read_text())
    assert len(report_inputs["molecular_lr_crosswalks"]) == 1
    assert len(report_inputs["molecular_lr_crosswalk_manifests"]) == 1


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("partial", "does not completely map the frozen score universe"),
        ("unsupported", "does not completely map the frozen score universe"),
        ("duplicate", "duplicate resource-edge keys"),
        ("checksum_tamper", "molecular_lr_crosswalk SHA256 mismatch"),
        ("manifest_tamper", "output.sha256 disagrees with the bound crosswalk"),
    ],
)
def test_finalize_rejects_invalid_molecular_lr_crosswalk_bindings(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    datasets = cast(list[dict[str, Any]], payload["datasets"])
    runs = cast(list[dict[str, Any]], datasets[0]["adapter_runs"])
    crosswalk = _molecular_lr_crosswalk()
    if failure == "partial":
        crosswalk = crosswalk.iloc[:-1].copy()
    elif failure == "unsupported":
        crosswalk.loc[2, "mapping_status"] = "unsupported_direction"
        crosswalk.loc[2, "reason_code"] = "unsupported_interaction_direction"
        crosswalk.loc[
            2, ["molecular_lr_equivalence_id", "mechanistic_variant_id"]
        ] = None
    elif failure == "duplicate":
        crosswalk = pd.concat([crosswalk, crosswalk.iloc[[0]]], ignore_index=True)
    _bind_molecular_lr_crosswalk(
        tmp_path,
        runs[0],
        table=crosswalk,
        name=f"crosswalk_{failure}",
        manifest_output_sha256=("0" * 64 if failure == "manifest_tamper" else None),
    )
    if failure == "checksum_tamper":
        binding = cast(dict[str, Any], runs[0]["molecular_lr_crosswalk"])
        binding["sha256"] = "0" * 64
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.finalize(spec, tmp_path / f"invalid_{failure}")


def test_finalize_rejects_crosswalk_without_an_applicable_long_table(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    datasets = cast(list[dict[str, Any]], payload["datasets"])
    runs = cast(list[dict[str, Any]], datasets[1]["adapter_runs"])
    failed_run = runs.pop(2)
    runs.insert(0, failed_run)
    _bind_molecular_lr_crosswalk(tmp_path, failed_run, name="crosswalk_without_scores")
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="requires an existing adapter long_table"):
        module.finalize(spec, tmp_path / "invalid_crosswalk_without_scores")


def test_finalize_emits_shared_exploratory_effect_models_without_zero_fill(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    datasets = cast(list[dict[str, Any]], payload["datasets"])
    paired = datasets[0]
    runs = cast(list[dict[str, Any]], paired["adapter_runs"])
    _mark_adapter_edge_missing(tmp_path, runs[1], interaction_id="I3")
    _bind_effect_contract(
        tmp_path,
        paired,
        _paired_effect_sample_design(),
        [
            {
                "id": "paired-d-common",
                "backend": "d_common",
                "contrast": "Tumor_vs_Normal",
                "reference": "Normal",
                "target": "Tumor",
                "min_subjects_per_group": 3,
            },
            {
                "id": "paired-repeated",
                "backend": "repeated_measures",
                "contrast": "Tumor_vs_Normal",
                "reference": "Normal",
                "target": "Tumor",
                "min_subjects_per_context": 3,
                "min_subject_clusters": 3,
            },
        ],
        name="paired_sample_design.tsv",
    )
    spec.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "effect_final"
    manifest = module.finalize(spec, output)

    d_common = pd.read_parquet(output / "derived/d_common_edge_effects.parquet")
    repeated = pd.read_parquet(
        output / "derived/repeated_measures_edge_effects.parquet"
    )
    expected_methods = {"crychic", "m1", "m2"}
    assert set(d_common["method"]) == expected_methods
    assert set(repeated["method"]) == expected_methods
    assert d_common["d_common_effect_model_id"].nunique() == 1
    assert repeated["repeated_measures_design_id"].nunique() == 1
    missing_common = d_common.loc[
        d_common["method"].eq("m2") & d_common["interaction_id"].eq("I3")
    ]
    assert len(missing_common) == 1
    assert missing_common.iloc[0]["status"] == "not_estimable"
    assert pd.isna(missing_common.iloc[0]["effect"])
    missing_edge = repeated.loc[
        repeated["method"].eq("m2") & repeated["interaction_id"].eq("I3")
    ]
    assert len(missing_edge) == 1
    assert missing_edge.iloc[0]["status"] == "not_estimable"
    assert missing_edge.iloc[0]["reason_code"] == ("insufficient_subjects_per_context")
    assert pd.isna(missing_edge.iloc[0]["effect"])
    for result in (d_common, repeated):
        assert result["formal_inference_allowed"].eq(False).all()
        assert not {"p", "p_value", "q", "q_value"}.intersection(result.columns)
        assert set(result["score_view_role"]) == {"primary"}
    legacy = pd.read_parquet(output / "derived/edge_effects.parquet")
    assert "effect_model_backend" not in legacy
    assert (
        manifest["guardrails"]["exploratory_effects_replaced_legacy_edge_effects"]
        is False
    )
    report_inputs = json.loads(
        (output / "report_inputs.json").read_text(encoding="utf-8")
    )
    assert set(report_inputs["exploratory_effect_models"]) == {
        "d_common",
        "repeated_measures",
    }


def test_finalize_explicit_cr2_backend_emits_diagnostic_only_effects(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    dataset = cast(list[dict[str, Any]], payload["datasets"])[0]
    _bind_effect_contract(
        tmp_path,
        dataset,
        _paired_effect_sample_design(),
        [
            {
                "id": "paired-cr2",
                "backend": "repeated_measures_cr2",
                "contrast": "Tumor_vs_Normal",
                "reference": "Normal",
                "target": "Tumor",
                "min_subjects_per_context": 3,
                "min_subject_clusters": 3,
                "subject_fixed_effects": False,
            }
        ],
        name="paired_cr2_sample_design.tsv",
    )
    spec.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "cr2_effect_final"
    manifest = module.finalize(spec, output)

    path = output / "derived/repeated_measures_cr2_edge_effects.parquet"
    result = pd.read_parquet(path)
    assert set(result["effect_model_backend"]) == {"repeated_measures_cr2"}
    assert set(result["status"]) == {"not_estimable"}
    assert set(result["reason_code"]) == {"insufficient_subject_clusters"}
    assert result["effect"].isna().all()
    assert result["standard_error"].isna().all()
    assert result["n_subject_clusters"].eq(3).all()
    assert result["formal_backend_eligible"].eq(False).all()
    assert result["formal_inference_allowed"].eq(False).all()
    assert not {"p", "p_value", "q", "q_value"}.intersection(result.columns)
    assert manifest["exploratory_effect_model_outputs"][
        "repeated_measures_cr2"
    ] == "derived/repeated_measures_cr2_edge_effects.parquet"


def test_cr2_effect_model_config_rejects_non_boolean_subject_fixed_effects() -> None:
    with pytest.raises(ValueError, match="subject_fixed_effects must be boolean"):
        module._effect_model_specs(
            [
                {
                    "id": "invalid-cr2",
                    "backend": "repeated_measures_cr2",
                    "reference": "control",
                    "target": "case",
                    "subject_fixed_effects": "yes",
                }
            ],
            field="datasets[0].effect_models",
            context_key="condition",
            comparison={"contrast": "case_vs_control"},
        )


def test_finalize_preserves_typed_ne_for_confounding_and_rank_deficiency(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    dataset = cast(list[dict[str, Any]], payload["datasets"])[1]
    _bind_effect_contract(
        tmp_path,
        dataset,
        _unpaired_effect_sample_design(confounded=True),
        [
            {
                "id": "confounded-d-common",
                "backend": "d_common",
                "contrast": "Case_vs_Ctrl",
                "reference": "Ctrl",
                "target": "Case",
                "batch_keys": ["batch"],
                "min_subjects_per_group": 3,
            },
            {
                "id": "confounded-repeated",
                "backend": "repeated_measures",
                "contrast": "Case_vs_Ctrl",
                "reference": "Ctrl",
                "target": "Case",
                "batch_keys": ["batch"],
                "min_subjects_per_context": 3,
                "min_subject_clusters": 6,
            },
        ],
        name="confounded_sample_design.tsv",
    )
    spec.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "confounded_final"
    module.finalize(spec, output)

    d_common = pd.read_parquet(output / "derived/d_common_edge_effects.parquet")
    repeated = pd.read_parquet(
        output / "derived/repeated_measures_edge_effects.parquet"
    )
    assert set(d_common["status"]) == {"not_estimable"}
    assert set(d_common["reason_code"]) == {
        "target_not_estimable_after_batch_adjustment"
    }
    assert d_common["effect"].isna().all()
    assert set(repeated["status"]) == {"not_estimable"}
    assert set(repeated["reason_code"]) == {"rank_deficient_declared_design"}
    assert repeated["effect"].isna().all()


def test_effect_model_role_cannot_claim_primary_release() -> None:
    with pytest.raises(ValueError, match="role must be exploratory"):
        module._effect_model_specs(
            [
                {
                    "id": "forged-primary-effect",
                    "backend": "d_common",
                    "reference": "control",
                    "target": "case",
                    "role": "primary",
                }
            ],
            field="datasets[0].effect_models",
            context_key="condition",
            comparison={"contrast": "case_vs_control"},
        )


def test_finalize_rejects_sample_design_checksum_mismatch(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    dataset = cast(list[dict[str, Any]], payload["datasets"])[0]
    _bind_effect_contract(
        tmp_path,
        dataset,
        _paired_effect_sample_design(),
        [
            {
                "id": "paired-d-common",
                "backend": "d_common",
                "reference": "Normal",
                "target": "Tumor",
            }
        ],
        name="checksum_sample_design.tsv",
    )
    binding = cast(dict[str, str], dataset["sample_design"])
    binding["sha256"] = "0" * 64
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_design SHA256 mismatch"):
        module.finalize(spec, tmp_path / "checksum_invalid")


def test_finalize_rejects_inexact_sample_design_coverage(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    dataset = cast(list[dict[str, Any]], payload["datasets"])[0]
    incomplete = _paired_effect_sample_design().iloc[:-1].copy()
    _bind_effect_contract(
        tmp_path,
        dataset,
        incomplete,
        [
            {
                "id": "paired-d-common",
                "backend": "d_common",
                "reference": "Normal",
                "target": "Tumor",
            }
        ],
        name="incomplete_sample_design.tsv",
    )
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sample coverage mismatch"):
        module.finalize(spec, tmp_path / "coverage_invalid")


def test_cross_dataset_primary_excludes_synthetic_loso() -> None:
    rows: list[dict[str, object]] = []
    datasets = {
        "real_a": ("real_data", (0.1, 0.3)),
        "real_b": ("real_data", (0.3, 0.5)),
        "synthetic": ("simulation", (0.9, 0.9)),
    }
    for dataset, (truth_scope, values) in datasets.items():
        for index, value in enumerate(values, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "method": "m1",
                    "method_version": "1",
                    "analysis_track": "lr_stlr",
                    "resource": "common",
                    "resource_version": "1",
                    "resource_mode": "H-common",
                    "score_semantics": "rank_strength",
                    "universe_id": f"{dataset}-universe",
                    "contrast": "target_vs_reference",
                    "held_out_subject": f"subject_{index}",
                    "effect_spearman": value,
                    "status": "observed",
                    "truth_scope": truth_scope,
                }
            )

    aggregate = module._cross_dataset_primary(
        pd.DataFrame(rows),
        {"n_bootstrap": 20, "random_seed": 7},
    )

    assert len(aggregate) == 1
    row = aggregate.iloc[0]
    assert row["truth_scope"] == "real_data"
    assert row["n_datasets"] == 2
    assert row["n_subjects_estimable"] == 4
    assert row["estimate"] == pytest.approx(0.3)
    contracts = json.loads(str(row["dataset_contract_json"]))
    assert {contract["dataset"] for contract in contracts} == {"real_a", "real_b"}

    synthetic_only = pd.DataFrame(rows).loc[
        pd.DataFrame(rows)["truth_scope"].eq("simulation")
    ]
    assert module._cross_dataset_primary(
        synthetic_only,
        {"n_bootstrap": 20, "random_seed": 7},
    ).empty


def test_finalize_forbids_real_data_edge_truth(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text())
    payload["datasets"][1]["simulation_truth"] = "simulation_truth.tsv"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden for real datasets"):
        module.finalize(spec, tmp_path / "invalid")


@pytest.mark.parametrize(
    "replacement",
    [
        "True",
        1,
    ],
)
def test_finalize_rejects_non_boolean_universe_member_schema(
    tmp_path: Path,
    replacement: object,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    run = payload["datasets"][0]["adapter_runs"][0]
    table_path = tmp_path / run["long_table"]
    table = pd.read_parquet(table_path)
    table["universe_member"] = replacement
    table.to_parquet(table_path, index=False)
    manifest_path = tmp_path / run["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = _sha256(table_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="universe_member must use an Arrow boolean type",
    ):
        module.finalize(spec, tmp_path / "invalid_membership")


def test_common_functional_false_fails_closed_across_rank_endpoints(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text())
    payload["datasets"][1]["comparison"]["min_subjects"] = 4
    run = payload["datasets"][1]["adapter_runs"][0]
    manifest_path = tmp_path / run["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["source_result"] = {
        "score_views": [
            {
                "run_id": "run-real_unpaired-m1",
                "contrast_candidates": [],
                "common_functional_claim": False,
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    biology_path = tmp_path / "receiver_scoped_biology.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "real_unpaired",
                "observation_id": "known_program",
                "method": "m1",
                "resource_mode": "H-common",
                "rank_scope": "within_receiver_macro",
                "support_status": "partial",
                "observed_direction": "Case_up",
                "evidence_note": "receiver-scoped diagnostic",
                "status": "observed",
                "reason_code": "supportive_silver_standard",
                "source_biology_file": "receiver_scoped_biology.tsv",
            }
        ]
    ).to_csv(biology_path, sep="\t", index=False)
    payload["biology_support"] = biology_path.name
    spec.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "rank_scope_ne"
    module.finalize(spec, output)

    reason = "receiver_child_functionals_not_globally_comparable"
    primary = pd.read_csv(output / "metrics/loso_primary_endpoint.tsv", sep="\t")
    selected_primary = primary[
        primary["dataset"].eq("real_unpaired") & primary["method"].eq("m1")
    ]
    assert set(selected_primary["status"]) == {"not_estimable"}
    assert set(selected_primary["reason_code"]) == {reason}
    assert set(selected_primary["rank_scope"]) == {
        "not_estimable_receiver_child_functionals"
    }

    stability = pd.read_csv(output / "metrics/stability_summary.tsv", sep="\t")
    selected_stability = stability[
        stability["dataset"].eq("real_unpaired") & stability["method"].eq("m1")
    ]
    assert set(selected_stability["status"]) == {"not_estimable"}
    assert set(selected_stability["reason_code"]) == {reason}

    ranking = pd.read_csv(output / "metrics/ranking_agreement_summary.tsv", sep="\t")
    selected_ranking = ranking[
        ranking["dataset"].eq("real_unpaired") & ranking["method"].eq("m1")
    ]
    assert set(selected_ranking["status"]) == {"not_estimable"}
    assert set(selected_ranking["reason_code"]) == {reason}
    assert set(selected_ranking["minimum_subjects"]) == {4}

    biology = pd.read_csv(output / "metrics/biology_support.tsv", sep="\t")
    selected_biology = biology[
        biology["dataset"].eq("real_unpaired") & biology["method"].eq("m1")
    ]
    assert set(selected_biology["support_status"]) == {"not_estimable"}
    assert set(selected_biology["reason_code"]) == {reason}
    assert set(selected_biology["evidence_rank_scope"]) == {"within_receiver_macro"}
    assert set(selected_biology["source_biology_file"]) == {
        "receiver_scoped_biology.tsv"
    }

    score_index = pd.read_csv(output / "score_tables/score_table_index.tsv", sep="\t")
    selected_index = score_index[
        score_index["dataset"].eq("real_unpaired") & score_index["method"].eq("m1")
    ]
    assert set(selected_index["rank_scope"]) == {
        "not_estimable_receiver_child_functionals"
    }


def test_common_functional_false_rejects_explicit_global_scope(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text())
    run = payload["datasets"][1]["adapter_runs"][0]
    run["rank_scope"] = "global_common_functional"
    manifest_path = tmp_path / run["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["source_result"] = {
        "score_views": [
            {
                "run_id": "run-real_unpaired-m1",
                "contrast_candidates": [],
                "common_functional_claim": False,
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbids global rank_scope"):
        module.finalize(spec, tmp_path / "invalid_global_scope")


def test_cli_contract_requires_locked_supportive_truth(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text())
    del payload["supportive_biology_truth"]
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="supportive_biology_truth is required"):
        module.finalize(spec, tmp_path / "invalid")


def test_supportive_biology_dataset_alias_preserves_locked_observations(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text())
    truth = tmp_path / payload["supportive_biology_truth"]
    truth.write_text(
        truth.read_text().replace("  real_unpaired:\n", "  locked_ms_atlas:\n"),
        encoding="utf-8",
    )
    payload["supportive_biology_dataset_aliases"] = {"locked_ms_atlas": "real_unpaired"}
    spec.write_text(json.dumps(payload), encoding="utf-8")

    module.finalize(spec, tmp_path / "aliased")

    biology = pd.read_csv(tmp_path / "aliased/metrics/biology_support.tsv", sep="\t")
    selected = biology[biology["dataset"].eq("real_unpaired")]
    assert set(selected["truth_dataset"]) == {"locked_ms_atlas"}
    assert "m1" in set(selected["method"])
    report_inputs = json.loads(
        (tmp_path / "aliased/report_inputs.json").read_text(encoding="utf-8")
    )
    assert report_inputs["supportive_biology_dataset_aliases"] == {
        "locked_ms_atlas": "real_unpaired"
    }

    report_output = tmp_path / "aliased_report"
    report_module.generate(
        tmp_path / "aliased",
        report_output,
        "2026-07-12T12:00:00+08:00",
        render_pdf=False,
    )
    biology_source = pd.read_csv(
        report_output / "source_data/figure05_supportive_biology.csv"
    )
    assert "real_unpaired" in set(biology_source["dataset"])


def test_biology_evidence_can_add_native_sensitivity_variant(
    tmp_path: Path,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text())
    evidence = pd.DataFrame(
        [
            {
                "dataset": "real_unpaired",
                "observation_id": "known_program",
                "method": "native_only_method",
                "resource_mode": "native",
                "support_status": "partial",
                "observed_direction": "Case_up",
                "evidence_note": "native sensitivity evidence",
                "status": "observed",
                "reason_code": "supportive_silver_standard",
            }
        ]
    )
    evidence_path = tmp_path / "biology_support.tsv"
    evidence.to_csv(evidence_path, sep="\t", index=False)
    payload["biology_support"] = evidence_path.name
    coverage_path = tmp_path / "native_coverage.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "real_unpaired",
                "method": "native_only_method",
                "resource_mode": "native",
                "frozen_universe_edges": 500,
                "resource_coverage_fraction": 1.0,
                "comparison_coverage_fraction": 0.75,
                "status": "observed",
                "reason_code": "native_coverage_only_sensitivity",
            }
        ]
    ).to_csv(coverage_path, sep="\t", index=False)
    payload["coverage_records"] = coverage_path.name
    simulation_path = tmp_path / "track_b_simulation.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "synthetic_paired",
                "method": "track_b_proxy",
                "resource_mode": "native",
                "truth_scope": "simulation",
                "metric": "target_program_recovery",
                "estimate": 1.0,
                "status": "observed",
                "reason_code": "track_b_scope",
            }
        ]
    ).to_csv(simulation_path, sep="\t", index=False)
    payload["simulation_records"] = simulation_path.name
    spec.write_text(json.dumps(payload), encoding="utf-8")

    manifest = module.finalize(spec, tmp_path / "native_biology")

    assert manifest["guardrails"]["preregistered_track_b_primary_reported"] is False

    biology = pd.read_csv(
        tmp_path / "native_biology/metrics/biology_support.tsv", sep="\t"
    )
    selected = biology[
        biology["method"].eq("native_only_method")
        & biology["resource_mode"].eq("native")
    ]
    assert len(selected) == 1
    assert selected.iloc[0]["support_status"] == "partial"
    coverage = pd.read_csv(
        tmp_path / "native_biology/metrics/coverage_summary.tsv", sep="\t"
    )
    native_coverage = coverage[coverage["method"].eq("native_only_method")]
    assert len(native_coverage) == 1
    assert native_coverage.iloc[0]["frozen_universe_edges"] == 500
    simulation = pd.read_csv(
        tmp_path / "native_biology/metrics/simulation_truth_metrics.tsv", sep="\t"
    )
    assert "track_b_proxy" in set(simulation["method"])
    primary = pd.read_csv(
        tmp_path / "native_biology/metrics/loso_primary_endpoint.tsv", sep="\t"
    )
    assert "track_b_proxy" not in set(primary["method"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("remove_mapping", "must map every view explicitly"),
        ("reverse_primary", "does not match the preregistered"),
    ],
)  # type: ignore[untyped-decorator]
def test_multiview_score_selection_is_explicit_and_preregistered(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    spec = _fixture(tmp_path)
    payload = json.loads(spec.read_text())
    crychic = payload["datasets"][0]["adapter_runs"][2]
    if mutation == "remove_mapping":
        del crychic["score_views"]
    else:
        crychic["score_views"][0]["role"] = "sensitivity"
        crychic["score_views"][1]["role"] = "primary"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        module.finalize(spec, tmp_path / "invalid")


def test_real_data_iteration_auroc_aliases_are_forbidden(tmp_path: Path) -> None:
    path = tmp_path / "iteration.tsv"
    pd.DataFrame(
        [
            {
                "dataset": "real_unpaired",
                "method": "m1",
                "metric": "edge-AUROC-primary",
                "iteration_from": "v0",
                "iteration_to": "v1",
                "before": 0.5,
                "after": 0.8,
                "metric_direction": "higher",
                "status": "observed",
                "reason_code": "",
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="real-data AUROC/AUPRC"):
        module._iteration_table(path, frozenset({"real_unpaired"}))


def test_readback_only_elapsed_is_excluded_from_method_runtime(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "run_id": "source_pipeline",
                "dataset_id": "cscc",
                "method": {"id": "crychic"},
                "resource": {"id": "common", "mode": "H-common"},
                "status": "complete",
                "elapsed_seconds": 7023.0,
                "environment": {"threads": 8},
                "output": {"sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    current_manifest_path = tmp_path / "compact_manifest.json"
    current_manifest = {
        "run_id": "compact_readback",
        "status": "complete",
        "elapsed_seconds": 212.0,
        "environment": {"threads": 1},
        "output": {"sha256": "b" * 64},
    }
    current_manifest_path.write_text(json.dumps(current_manifest), encoding="utf-8")
    long_path = tmp_path / "scores.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(long_path, index=False)
    identity = module.RunIdentity(
        dataset="cscc",
        method="crychic",
        method_version="1",
        analysis_track="lr_stlr",
        resource="common",
        resource_version="1",
        resource_mode="H-common",
        score_semantics="comm_strength",
        universe_id="u1",
        contrast="Tumor_vs_Normal",
    )

    records, sources = module._performance_records_for_run(
        spec_root=tmp_path,
        identity=identity,
        run={
            "performance_role": "adapter_readback",
            "performance_override": {
                "source_manifest": source_manifest.name,
                "source_role": "source_pipeline_total",
                "expected_sha256": _sha256(source_manifest),
            },
        },
        manifest=current_manifest,
        manifest_path=current_manifest_path,
        long_path=long_path,
    )

    assert sources == (source_manifest,)
    records_table = pd.DataFrame(records)
    records_table.loc[
        records_table["performance_role"].eq("adapter_readback"),
        "include_in_method_runtime",
    ] = True
    records_table.loc[
        records_table["performance_role"].eq("source_pipeline_total"),
        "include_in_method_runtime",
    ] = False
    runtime_input = module._prepare_performance_runtime_input(records_table)
    summary = module.summarize_run_performance(runtime_input)
    components = module._performance_component_summary(records_table)
    assert summary.iloc[0]["median_wall_time_seconds"] == 7023.0
    assert summary.iloc[0]["median_wall_time_seconds"] != 212.0
    assert components.iloc[0]["method_runtime_role"] == "source_pipeline_total"
    assert components.iloc[0]["median_adapter_readback_wall_time_seconds"] == 212.0
    assert (
        components.iloc[0]["median_source_pipeline_total_wall_time_seconds"] == 7023.0
    )


def test_performance_override_source_manifest_is_identity_bound(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "run_id": "source_pipeline",
                "dataset_id": "wrong_dataset",
                "method": {"id": "crychic"},
                "resource": {"id": "common", "mode": "H-common"},
                "status": "complete",
                "elapsed_seconds": 10.0,
            }
        ),
        encoding="utf-8",
    )
    current_path = tmp_path / "current.json"
    current_path.write_text(
        json.dumps({"run_id": "readback", "status": "complete"}),
        encoding="utf-8",
    )
    identity = module.RunIdentity(
        dataset="cscc",
        method="crychic",
        method_version="1",
        analysis_track="lr_stlr",
        resource="common",
        resource_version="1",
        resource_mode="H-common",
        score_semantics="comm_strength",
        universe_id="u1",
        contrast="Tumor_vs_Normal",
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        module._performance_records_for_run(
            spec_root=tmp_path,
            identity=identity,
            run={
                "performance_role": "adapter_readback",
                "performance_override": {
                    "source_manifest": source_manifest.name,
                    "expected_sha256": _sha256(source_manifest),
                },
            },
            manifest={"run_id": "readback", "status": "complete"},
            manifest_path=current_path,
            long_path=None,
        )


def test_performance_override_requires_adapter_only_role(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "run_id": "source",
                "status": "complete",
                "elapsed_seconds": 10.0,
            }
        ),
        encoding="utf-8",
    )
    current_path = tmp_path / "current.json"
    current = {
        "run_id": "current",
        "status": "complete",
        "elapsed_seconds": 2.0,
    }
    current_path.write_text(json.dumps(current), encoding="utf-8")
    identity = module.RunIdentity(
        dataset="d",
        method="m",
        method_version="1",
        analysis_track="lr_stlr",
        resource="r",
        resource_version="1",
        resource_mode="H-common",
        score_semantics="s",
        universe_id="u",
        contrast="c",
    )

    with pytest.raises(ValueError, match="adapter_readback"):
        module._performance_records_for_run(
            spec_root=tmp_path,
            identity=identity,
            run={
                "performance_role": "method_total",
                "performance_override": {
                    "source_manifest": source_manifest.name,
                },
            },
            manifest=current,
            manifest_path=current_path,
            long_path=None,
        )
