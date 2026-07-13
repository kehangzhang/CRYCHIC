from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.report import generate_multicondition_report as module


def _write_tsv(root: Path, name: str, rows: list[dict[str, object]]) -> str:
    path = root / "metrics" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path.relative_to(root).as_posix()


def _fixture(root: Path) -> Path:
    adapter = root / "adapters" / "cellchat"
    adapter.mkdir(parents=True)
    (adapter / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "cscc",
                "method_id": "CellChat",
                "method_version": "2.2",
                "resource_id": "common",
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "dataset_id": ["cscc", "cscc", "cscc"],
            "method_id": ["CellChat"] * 3,
            "resource_mode": ["H-common"] * 3,
            "status": ["ok", "not_returned", "resource_unavailable"],
        }
    ).to_parquet(adapter / "scores_long.parquet", index=False)
    datasets = root / "datasets" / "cscc"
    datasets.mkdir(parents=True)
    (datasets / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "cscc",
                "input": {
                    "n_cells": 47068,
                    "n_samples": 20,
                    "n_subjects": 10,
                    "n_contexts": 2,
                    "n_cell_types": 7,
                },
                "design_audit": {"status": "observed"},
            }
        ),
        encoding="utf-8",
    )
    truth = root / "truth" / "multicondition_supportive_biology.yaml"
    truth.parent.mkdir(parents=True)
    truth.write_text(
        "truth_set_id: fixture\n"
        "datasets:\n"
        "  cscc:\n"
        "    expected_observations:\n"
        "      - id: epithelial_stromal_hub\n"
        "        direction: Tumor_up\n"
        "        metric: paired_differential_rank\n",
        encoding="utf-8",
    )
    ranking_parameters = {
        "n_bootstrap": 2000,
        "n_split_repeats": 200,
        "rank_interval_quantile_level": 0.95,
        "minimum_top_k_frequency": 0.8,
        "minimum_estimable_replicate_fraction": 0.8,
        "minimum_subjects": 3,
        "minimum_observed_ranks": 2,
        "rbo_persistence": 0.9,
        "weighted_kendall_power": 1.0,
        "random_seed": 20260712,
    }
    ranking_identity = {
        "dataset": "cscc",
        "method": "CellChat",
        "resource_mode": "H-common",
        "analysis_track": "lr_stlr",
        "truth_scope": "real_data",
        "contrast": "Tumor_vs_Normal",
        "rank_scope": "global_common_functional",
        "design": "paired",
        "reference": "Normal",
        "target": "Tumor",
        "n_reference_subjects": 10,
        "n_target_subjects": 10,
        "n_paired_subjects": 10,
        "ranking_level": "sender_receiver_pair",
        "ranking_universe_size": 3,
        **ranking_parameters,
    }

    metrics = {
        "dataset_design": _write_tsv(
            root,
            "dataset_design.tsv",
            [
                {
                    "dataset": "cscc",
                    "n_cells": 47068,
                    "n_samples": 20,
                    "n_subjects": 10,
                    "n_contexts": 2,
                    "n_cell_types": 7,
                    "design_type": "paired",
                    "status": "observed",
                    "reason_code": "",
                },
                {
                    "dataset": "kuppe",
                    "n_cells": 191795,
                    "n_samples": 29,
                    "n_subjects": 20,
                    "n_contexts": 5,
                    "n_cell_types": 9,
                    "design_type": "partial_repeated_regions",
                    "status": "not_estimable",
                    "reason_code": "mixed_paired_unpaired_design",
                },
            ],
        ),
        "coverage": _write_tsv(
            root,
            "coverage_summary.tsv",
            [
                {
                    "dataset": "cscc",
                    "method": "CellChat",
                    "resource_mode": "H-common",
                    "truth_scope": "real_data",
                    "resource_coverage_fraction": 1.0,
                    "comparison_coverage_fraction": 0.8,
                    "observed_rows": 80,
                    "not_predicted_rows": 10,
                    "missing_rows": 0,
                    "cell_type_missing_rows": 5,
                    "not_estimable_rows": 0,
                    "failed_rows": 0,
                    "not_supported_rows": 0,
                    "status": "observed",
                    "reason_code": "",
                },
                {
                    "dataset": "cscc",
                    "method": "NicheNet",
                    "resource_mode": "H-common",
                    "truth_scope": "real_data",
                    "resource_coverage_fraction": 0.0,
                    "comparison_coverage_fraction": None,
                    "observed_rows": 0,
                    "not_predicted_rows": 0,
                    "missing_rows": 0,
                    "cell_type_missing_rows": 0,
                    "not_estimable_rows": 0,
                    "failed_rows": 0,
                    "not_supported_rows": 100,
                    "status": "not_supported",
                    "reason_code": "track_b_only",
                },
            ],
        ),
        "loso_primary": _write_tsv(
            root,
            "loso_primary_endpoint.tsv",
            [
                {
                    "dataset": "cscc",
                    "method": "CellChat",
                    "resource_mode": "H-common",
                    "contrast": "Tumor_vs_Normal",
                    "truth_scope": "real_data",
                    "estimate": 0.42,
                    "ci_lower": 0.21,
                    "ci_upper": 0.61,
                    "n_subjects_estimable": 10,
                    "status": "observed",
                    "reason_code": "",
                },
                {
                    "dataset": "cscc",
                    "method": "NicheNet",
                    "resource_mode": "H-common",
                    "contrast": "Tumor_vs_Normal",
                    "truth_scope": "real_data",
                    "estimate": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "n_subjects_estimable": 0,
                    "status": "not_estimable",
                    "reason_code": "track_b_only",
                },
            ],
        ),
        "stability": _write_tsv(
            root,
            "stability_summary.tsv",
            [
                {
                    "dataset": "cscc",
                    "method": "CellChat",
                    "resource_mode": "H-common",
                    "metric": "median_top_k_jaccard",
                    "truth_scope": "real_data",
                    "estimate": 0.55,
                    "status": "observed",
                    "reason_code": "",
                },
                {
                    "dataset": "cscc",
                    "method": "CRYCHIC",
                    "resource_mode": "H-common",
                    "metric": "median_top_k_jaccard",
                    "truth_scope": "real_data",
                    "estimate": 0.63,
                    "status": "observed",
                    "reason_code": "",
                },
            ],
        ),
        "concordance": _write_tsv(
            root,
            "concordance_summary.tsv",
            [
                {
                    "dataset": "cscc",
                    "method_left": "CellChat",
                    "method_right": "CRYCHIC",
                    "resource_mode": "H-common",
                    "effect_spearman": 0.38,
                    "truth_scope": "real_data",
                    "direction_agreement": 0.62,
                    "shared_edges": 500,
                    "status": "observed",
                    "reason_code": "",
                }
            ],
        ),
        "performance": _write_tsv(
            root,
            "performance_summary.tsv",
            [
                {
                    "dataset": "cscc",
                    "method": "CellChat",
                    "resource_mode": "H-common",
                    "median_wall_time_seconds": 90.0,
                    "truth_scope": "real_data",
                    "median_peak_rss_mb": 2500.0,
                    "success_rate": 1.0,
                    "n_failed": 0,
                    "status": "observed",
                    "reason_code": "",
                },
                {
                    "dataset": "cscc",
                    "method": "CRYCHIC",
                    "resource_mode": "H-common",
                    "median_wall_time_seconds": 45.0,
                    "truth_scope": "real_data",
                    "median_peak_rss_mb": 1800.0,
                    "success_rate": 1.0,
                    "n_failed": 0,
                    "status": "observed",
                    "reason_code": "",
                },
            ],
        ),
        "biology_support": _write_tsv(
            root,
            "biology_support.tsv",
            [
                {
                    "dataset": "cscc",
                    "observation_id": "epithelial_stromal_hub",
                    "method": "CellChat",
                    "resource_mode": "H-common",
                    "support_status": "supported",
                    "observed_direction": "Tumor_up",
                    "evidence_note": "paired rank",
                    "status": "observed",
                    "reason_code": "",
                },
                {
                    "dataset": "cscc",
                    "observation_id": "epithelial_stromal_hub",
                    "method": "CRYCHIC",
                    "resource_mode": "H-common",
                    "support_status": "partial",
                    "observed_direction": "Tumor_up",
                    "evidence_note": "aggregate support",
                    "status": "observed",
                    "reason_code": "",
                },
            ],
        ),
        "simulation_truth": _write_tsv(
            root,
            "simulation_truth_metrics.tsv",
            [
                {
                    "dataset": "active",
                    "method": "CRYCHIC",
                    "resource_mode": "H-common",
                    "truth_scope": "simulation",
                    "metric": "average_precision",
                    "estimate": 0.72,
                    "n_positive": 1,
                    "n_negative": 44,
                    "status": "observed",
                    "reason_code": "",
                },
                {
                    "dataset": "global_null",
                    "method": "CRYCHIC",
                    "resource_mode": "H-common",
                    "truth_scope": "simulation",
                    "metric": "type_i_error",
                    "estimate": 0.04,
                    "n_null_replicates": 1000,
                    "status": "observed",
                    "reason_code": "",
                },
            ],
        ),
        "iteration_comparison": _write_tsv(
            root,
            "iteration_comparison.tsv",
            [
                {
                    "dataset": "cscc",
                    "method": "CRYCHIC",
                    "metric": "loso_spearman",
                    "iteration_from": "v0",
                    "iteration_to": "v1",
                    "before": 0.31,
                    "after": 0.44,
                    "metric_direction": "higher",
                    "status": "observed",
                    "reason_code": "",
                },
                {
                    "dataset": "simulation",
                    "method": "CRYCHIC",
                    "metric": "type_i_error",
                    "iteration_from": "v0",
                    "iteration_to": "v1",
                    "before": 0.05,
                    "after": 0.07,
                    "metric_direction": "lower",
                    "status": "observed",
                    "reason_code": "",
                },
            ],
        ),
        "ranking_agreement": _write_tsv(
            root,
            "ranking_agreement_summary.tsv",
            [
                {
                    **ranking_identity,
                    "metric": metric,
                    "estimate": estimate,
                    "envelope_lower": estimate - 0.1,
                    "envelope_upper": estimate + 0.1,
                    "interval_type": "split_repeat_quantile_envelope",
                    "n_repeats_requested": 200,
                    "n_repeats_estimable": 200,
                    "status": "observed",
                    "reason_code": "",
                }
                for metric, estimate in (
                    ("rank_biased_overlap", 0.72),
                    ("weighted_kendall_tau", 0.61),
                )
            ],
        ),
        "ranking_top_k_curve": _write_tsv(
            root,
            "ranking_top_k_stability_curve.tsv",
            [
                {
                    **ranking_identity,
                    "metric": "tie_inclusive_top_k_jaccard",
                    "k": k,
                    "estimate": estimate,
                    "envelope_lower": max(0.0, estimate - 0.1),
                    "envelope_upper": min(1.0, estimate + 0.1),
                    "interval_type": "split_repeat_quantile_envelope",
                    "n_repeats_requested": 200,
                    "n_repeats_estimable": 200,
                    "status": "observed",
                    "reason_code": "",
                }
                for k, estimate in ((1, 0.8), (2, 0.65), (3, 0.55))
            ],
        ),
        "ranking_intervals": _write_tsv(
            root,
            "bootstrap_rank_intervals.tsv",
            [
                {
                    **ranking_identity,
                    "item_id": f"family_{index}",
                    "item_label": label,
                    "lower_rank": lower,
                    "median_rank": median,
                    "upper_rank": upper,
                    "rank_availability_frequency": 1.0,
                    "top_k_frequency": frequency,
                    "n_replicates_requested": 2000,
                    "status": "observed",
                    "reason_code": "",
                }
                for index, (label, lower, median, upper, frequency) in enumerate(
                    (
                        ("Tumor -> Fibroblast", 1.0, 1.0, 2.0, 0.96),
                        ("Fibroblast -> Tumor", 1.0, 2.0, 3.0, 0.82),
                    ),
                    start=1,
                )
            ],
        ),
        "ranking_tiers": _write_tsv(
            root,
            "stable_ranking_tiers.tsv",
            [
                {
                    **ranking_identity,
                    "item_id": f"family_{index}",
                    "item_label": label,
                    "tier": tier,
                    "lower_rank": lower,
                    "upper_rank": upper,
                    "rank_availability_frequency": 1.0,
                    "top_k_frequency": frequency,
                    "status": "observed",
                    "reason_code": "",
                }
                for index, (label, tier, lower, upper, frequency) in enumerate(
                    (
                        ("Tumor -> Fibroblast", "stable_top_k", 1.0, 2.0, 0.96),
                        ("Fibroblast -> Tumor", "possible_top_k", 1.0, 3.0, 0.82),
                    ),
                    start=1,
                )
            ],
        ),
    }
    spec = {
        "schema_version": module.INPUT_SCHEMA_VERSION,
        "adapter_manifests": ["adapters/cellchat/adapter_manifest.json"],
        "score_tables": ["adapters/cellchat/scores_long.parquet"],
        "dataset_manifests": ["datasets/cscc/dataset_manifest.json"],
        "truth_yaml": "truth/multicondition_supportive_biology.yaml",
        "metrics": metrics,
    }
    path = root / "report_inputs.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def test_generate_multicondition_report_from_frozen_fixture(tmp_path: Path) -> None:
    final_dir = tmp_path / "frozen"
    final_dir.mkdir()
    _fixture(final_dir)
    for filename in (
        "coverage_summary.tsv",
        "loso_primary_endpoint.tsv",
        "stability_summary.tsv",
        "concordance_summary.tsv",
        "performance_summary.tsv",
    ):
        path = final_dir / "metrics" / filename
        table = pd.read_csv(path, sep="\t")
        synthetic = table.iloc[[0]].copy()
        synthetic["dataset"] = "synthetic_active"
        synthetic["truth_scope"] = "simulation"
        pd.concat([table, synthetic], ignore_index=True).to_csv(
            path, sep="\t", index=False
        )
    output = tmp_path / "report"

    module.generate(
        final_dir,
        output,
        "2026-07-12T12:00:00+08:00",
        render_pdf=False,
    )

    stems = (
        "figure01_design_estimability",
        "figure02_coverage_status",
        "figure03_loso_primary",
        "figure04_robustness_performance",
        "figure05_supportive_biology",
        "figure06_simulation_truth",
        "figure07_iteration_comparison",
        "figure08_rank_stability",
    )
    for stem in stems:
        for suffix in ("png", "svg", "pdf"):
            artifact = output / "figures" / f"{stem}.{suffix}"
            assert artifact.stat().st_size > 100
    report_html = (output / "REPORT.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in report_html
    assert "Real-data AUROC/AUPRC are not calculated" in report_html
    manifest = json.loads((output / "report_manifest.json").read_text())
    assert manifest["report_id"] == "report"
    assert not manifest["guardrails"]["real_data_edge_auroc_reported"]
    assert not manifest["guardrails"]["preregistered_track_b_primary_reported"]
    assert manifest["endpoint_status_counts"]["loso_not_estimable"] == 1
    assert manifest["endpoint_status_counts"]["iteration_regressions"] == 1
    assert len(manifest["artifacts"]) == 35
    assert manifest["ranking_parameters"]["n_bootstrap"] == 2000
    assert not manifest["guardrails"]["nichenet_lr_or_sender_ranking_reported"]
    assert manifest["endpoint_status_counts"]["ranking_agreement_observed"] == 2
    assert all(record["bytes"] > 0 for record in manifest["artifacts"])
    assert all(
        len(record["sha256"]) == 64 for record in manifest["artifacts"]
    )
    inputs = pd.read_csv(output / "input_manifest.tsv", sep="\t")
    assert inputs["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    artifacts = pd.read_csv(output / "artifact_manifest.tsv", sep="\t")
    assert "report_manifest.json" in set(artifacts["path"])
    assert artifacts["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    loso_source = pd.read_csv(output / "source_data/figure03_loso_primary.csv")
    assert set(loso_source["status"]) == {"observed", "not_estimable"}
    for stem in (
        "figure02_coverage_status",
        "figure03_loso_primary",
        "figure04_robustness_performance",
    ):
        source = pd.read_csv(output / f"source_data/{stem}.csv")
        assert "synthetic_active" not in set(source["dataset"])


def test_simulation_metrics_reject_real_data_truth_scope() -> None:
    table = pd.DataFrame(
        {
            "dataset": ["cscc"],
            "method": ["m"],
            "truth_scope": ["real_data"],
            "metric": ["auroc"],
            "estimate": [0.9],
            "status": ["observed"],
        }
    )
    with pytest.raises(ValueError, match="explicit synthetic truth scopes"):
        module._simulation_source(table)


def test_simulation_metrics_reject_single_class_auc_and_underpowered_nulls() -> None:
    single_class = pd.DataFrame(
        {
            "dataset": ["global_null"],
            "method": ["m"],
            "truth_scope": ["simulation"],
            "metric": ["auroc"],
            "estimate": [0.5],
            "n_positive": [0],
            "n_negative": [45],
            "status": ["observed"],
        }
    )
    with pytest.raises(ValueError, match="single-class"):
        module._simulation_source(single_class)

    underpowered = pd.DataFrame(
        {
            "dataset": ["global_null"],
            "method": ["m"],
            "truth_scope": ["simulation"],
            "metric": ["type_i_error"],
            "estimate": [0.05],
            "n_null_replicates": [999],
            "status": ["observed"],
        }
    )
    with pytest.raises(ValueError, match="at least 1000"):
        module._simulation_source(underpowered)


def test_simulation_source_handles_mixed_wide_and_long_records() -> None:
    table = pd.DataFrame(
        [
            {
                "dataset": "synthetic_active",
                "truth_scope": "simulation",
                "method": "m",
                "n_positive": 1,
                "n_negative": 44,
                "auroc": 0.9,
                "average_precision": 0.5,
                "status": "observed",
            },
            {
                "dataset": "synthetic_active",
                "scenario": "active",
                "truth_scope": "simulation",
                "method": "m",
                "record_source": "track_a_differential_truth",
                "metric": "differential_auroc",
                "estimate": 0.95,
                "status": "observed",
            },
        ]
    )

    source = module._simulation_source(table)

    assert set(source["metric"]) == {
        "auroc",
        "average_precision",
        "differential_auroc",
    }
    assert set(source["scenario"]) == {"active"}
    assert source.loc[source["status"].eq("observed"), "estimate"].notna().all()


def test_loso_source_excludes_synthetic_and_fails_closed_without_scope() -> None:
    rows = pd.DataFrame(
        [
            {
                "dataset": "real",
                "method": "m",
                "truth_scope": "real_data",
                "estimate": 0.4,
                "ci_lower": 0.2,
                "ci_upper": 0.6,
                "status": "observed",
            },
            {
                "dataset": "synthetic_active",
                "method": "m",
                "truth_scope": "simulation",
                "estimate": 0.9,
                "ci_lower": 0.8,
                "ci_upper": 1.0,
                "status": "observed",
            },
        ]
    )

    source = module._loso_source(rows)

    assert set(source["dataset"]) == {"real"}
    unknown = module._loso_source(rows.drop(columns="truth_scope"))
    assert set(unknown["status"]) == {"not_estimable"}
    assert set(unknown["reason_code"]) == {"real_data_primary_endpoint_missing"}


def test_iteration_source_does_not_score_not_estimable_rows() -> None:
    source = module._iteration_source(
        pd.DataFrame(
            [
                {
                    "dataset": "d",
                    "method": "CRYCHIC",
                    "metric": "rows",
                    "before": 100.0,
                    "after": 50.0,
                    "metric_direction": "lower",
                    "status": "observed",
                },
                {
                    "dataset": "d",
                    "method": "CRYCHIC",
                    "metric": "wall_time",
                    "before": 100.0,
                    "after": 80.0,
                    "metric_direction": "lower",
                    "status": "not_estimable",
                    "reason_code": "concurrent_run_not_controlled",
                },
            ]
        )
    )

    observed = source.loc[source["metric"].eq("rows")].iloc[0]
    unavailable = source.loc[source["metric"].eq("wall_time")].iloc[0]
    assert observed["relative_improvement"] == pytest.approx(0.5)
    assert observed["change_class"] == "improved"
    assert pd.isna(unavailable["signed_improvement"])
    assert pd.isna(unavailable["relative_improvement"])
    assert unavailable["change_class"] == "not_estimable"


def test_biology_source_applies_locked_to_benchmark_dataset_alias(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "truth.yaml"
    truth.write_text(
        "datasets:\n"
        "  locked_ms_atlas:\n"
        "    expected_observations:\n"
        "      - id: glial_network\n"
        "        direction: CA_up\n"
        "        metric: supportive_only\n",
        encoding="utf-8",
    )
    table = pd.DataFrame(
        {
            "dataset": ["ms_ca_vs_ctrl"],
            "observation_id": ["glial_network"],
            "method": ["CRYCHIC"],
            "resource_mode": ["H-common"],
            "support_status": ["supported"],
            "observed_direction": ["CA_up"],
            "status": ["observed"],
            "reason_code": [""],
        }
    )

    source = module._biology_source(
        table,
        truth,
        {"locked_ms_atlas": "ms_ca_vs_ctrl"},
    )

    assert source.loc[0, "dataset"] == "ms_ca_vs_ctrl"
    assert source.loc[0, "support_status"] == "supported"


def test_simulation_track_plot_preserves_estimand_and_proxy_scope(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for estimand in (
        "paired_oriented_native_score_difference",
        "paired_rank_strength_difference",
    ):
        for metric, estimate in (
            ("differential_auroc", 0.9),
            ("differential_average_precision", 0.5),
            ("estimable_edge_coverage_fraction", 1.0),
        ):
            rows.append(
                {
                    "dataset": "synthetic_active",
                    "scenario": "active",
                    "method": "crychic",
                    "resource_mode": "H-common",
                    "truth_scope": "simulation",
                    "estimand": estimand,
                    "record_source": "track_a_differential_truth",
                    "metric": metric,
                    "estimate": estimate,
                    "status": "observed",
                }
            )
    for metric, estimate in (
        ("cxcl10_receiver_program_effect", 0.125),
        ("cxcl10_receiver_program_percentile", 1.0),
    ):
        rows.append(
            {
                "dataset": "synthetic_active",
                "scenario": "active",
                "method": "nichenet_prior_activity",
                "resource_mode": "native",
                "truth_scope": "simulation",
                "record_source": "track_b_receiver_program_truth",
                "proxy_semantics": (
                    "frozen_prior_activity_proxy_not_native_"
                    "predict_ligand_activities"
                ),
                "metric": metric,
                "estimate": estimate,
                "status": "observed",
            }
        )

    source = module._simulation_source(pd.DataFrame(rows))
    module._plot_simulation(source, tmp_path)

    assert set(source["estimand"]) == {
        "",
        "paired_oriented_native_score_difference",
        "paired_rank_strength_difference",
    }
    assert set(source["record_source"]) == {
        "track_a_differential_truth",
        "track_b_receiver_program_truth",
    }
    for suffix in ("png", "svg", "pdf"):
        assert (tmp_path / f"figure06_simulation_truth.{suffix}").stat().st_size > 100


def test_pdf_renderer_produces_pdf(tmp_path: Path) -> None:
    if shutil.which("weasyprint") is None:
        pytest.skip("weasyprint absent")
    html_path = tmp_path / "report.html"
    pdf_path = tmp_path / "report.pdf"
    html_path.write_text("<html><body><h1>fixture</h1></body></html>", encoding="utf-8")

    module._render_pdf(html_path, pdf_path)

    assert pdf_path.read_bytes().startswith(b"%PDF")
