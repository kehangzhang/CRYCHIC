from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.common import sha256_file
from benchmarks.comprehensive.evaluate_olink import (
    OUTPUT_SCHEMA,
    TRUTH_SCHEMA,
    _average_precision,
    _directional_ndcg,
    ligand_universe_id,
    run,
)


def _truth_bundle(
    tmp_path: Path,
    *,
    effects: list[float] | None = None,
    q_values: list[float] | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    effects = effects or [4.0, 2.0, -3.0, -1.0, 0.0]
    q_values = q_values or [0.01, 0.02, 0.01, 0.20, 0.80]
    analytes = [f"L{index}" for index in range(1, len(effects) + 1)]
    truth = pd.DataFrame(
        {
            "schema_version": [TRUTH_SCHEMA] * len(effects),
            "dataset_id": ["misc_olink"] * len(effects),
            "source_row": range(1, len(effects) + 1),
            "contrast_id": ["misc_m_vs_s"] * len(effects),
            "contrast_label": ["M-vs-S"] * len(effects),
            "contrast_numerator": ["MIS-C"] * len(effects),
            "contrast_denominator": ["Diorio_healthy_control"] * len(effects),
            "effect_semantics": [
                "mean_diff_d0 = MIS-C day 0 minus Diorio healthy controls"
            ]
            * len(effects),
            "q_value_semantics": [
                "adj_p_values is the source multiple-testing-adjusted p-value"
            ]
            * len(effects),
            "analyte_original": analytes,
            "analyte_make_names": analytes,
            "variables": analytes,
            "mean_d0": effects,
            "mean_hc": [0.0] * len(effects),
            "mean_diff_d0": effects,
            "unadj_p_values": [value / 2 for value in q_values],
            "adj_p_values": q_values,
        }
    )
    truth_path = tmp_path / "olink_truth.tsv"
    truth.to_csv(truth_path, sep="\t", index=False, lineterminator="\n")
    manifest = {
        "schema_version": TRUTH_SCHEMA,
        "status": "complete",
        "dataset_id": "misc_olink",
        "truth_contract": {
            "canonical_contrast_id": "misc_m_vs_s",
            "contrast_label": "M-vs-S",
            "effect_column": "mean_diff_d0",
            "reverse_contrast_id": "misc_s_vs_m",
            "reverse_contrast_label": "S-vs-M",
            "reverse_is_evaluator_derived": True,
            "reverse_effect": "-mean_diff_d0",
            "q_value_column": "adj_p_values",
            "olink_is_held_out_evaluation_only": True,
            "olink_must_not_enter_method_scores": True,
            "missing_analytes_are_not_zero": True,
        },
        "output": {
            "truth_tsv": {
                "filename": truth_path.name,
                "rows": len(truth),
                "sha256": sha256_file(truth_path),
            }
        },
    }
    manifest_path = tmp_path / "truth_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return truth_path, manifest_path, truth


def _prediction_frame(
    effects: list[float],
    *,
    contrasts: tuple[str, ...] = ("misc_m_vs_s", "misc_s_vs_m"),
) -> pd.DataFrame:
    records = []
    for contrast in contrasts:
        sign = 1.0 if contrast == "misc_m_vs_s" else -1.0
        for index, effect in enumerate(effects, start=1):
            records.append(
                {
                    "dataset_id": "misc_olink",
                    "method_id": "fixture_method",
                    "resource_mode": "H-common",
                    "universe_id": "pending",
                    "contrast": contrast,
                    "ligand": f"L{index}",
                    "score": sign * effect,
                    "status": "observed",
                }
            )
    result = pd.DataFrame(records)
    result["universe_id"] = ligand_universe_id(result["ligand"].tolist())
    return result


def _rebind_universe_id(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["universe_id"] = ligand_universe_id(result["ligand"].tolist())
    return result


def _write_predictions(path: Path, predictions: pd.DataFrame) -> str:
    predictions.to_csv(path, sep="\t", index=False, lineterminator="\n")
    return sha256_file(path)


def test_exact_metrics_handle_ranking_and_ties() -> None:
    labels = np.array([1, 0, 1, 0], dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert _average_precision(labels, scores) == pytest.approx(5 / 6)

    relevance = np.array([3.0, 0.0, 1.0, 0.0])
    expected = 3.5 / (3.0 + 1.0 / np.log2(3.0))
    assert _directional_ndcg(relevance, scores) == pytest.approx(expected)

    tied_scores = np.array([1.0, 1.0, 0.0])
    tied_relevance = np.array([3.0, 1.0, 0.0])
    tied_dcg = 4.0 * (1.0 + 1.0 / np.log2(3.0)) / 2.0
    tied_ideal = 3.0 + 1.0 / np.log2(3.0)
    assert _directional_ndcg(tied_relevance, tied_scores) == pytest.approx(
        tied_dcg / tied_ideal
    )


def test_forward_and_reverse_contrasts_are_exact_and_checksum_bound(
    tmp_path: Path,
) -> None:
    effects = [4.0, 2.0, -3.0, -1.0, 0.0]
    truth_path, truth_manifest, _ = _truth_bundle(tmp_path, effects=effects)
    predictions_path = tmp_path / "predictions.tsv"
    prediction_sha = _write_predictions(predictions_path, _prediction_frame(effects))
    output_dir = tmp_path / "evaluation"

    manifest = run(
        predictions_path,
        truth_path,
        truth_manifest,
        output_dir,
        predictions_sha256=prediction_sha,
    )

    metrics = pd.read_csv(output_dir / "metrics.tsv", sep="\t")
    assert set(metrics["contrast_id"]) == {"misc_m_vs_s", "misc_s_vs_m"}
    assert metrics["olink_ligand_ap"].eq(1.0).all()
    assert metrics["olink_logfc_spearman"].eq(1.0).all()
    assert metrics["protein_ndcg"].eq(1.0).all()
    assert metrics["truth_representable_fraction"].eq(1.0).all()
    assert metrics["representable_scored_fraction"].eq(1.0).all()
    forward = metrics.loc[metrics["contrast_id"].eq("misc_m_vs_s")].iloc[0]
    reverse = metrics.loc[metrics["contrast_id"].eq("misc_s_vs_m")].iloc[0]
    assert bool(forward["rank_eligible"])
    assert not bool(reverse["rank_eligible"])
    assert reverse["ranking_exclusion_reason"] == (
        "derived_reverse_contrast_diagnostic_only"
    )
    assert manifest["inputs"]["predictions"]["expected_sha256"] == prediction_sha
    assert manifest["evaluation_contract"][
        "prediction_score_is_never_recalculated_from_olink"
    ]
    assert manifest["output"]["metrics_tsv"]["sha256"] == sha256_file(
        output_dir / "metrics.tsv"
    )
    assert (
        json.loads((output_dir / "manifest.json").read_text())["schema_version"]
        == OUTPUT_SCHEMA
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        run(
            predictions_path,
            truth_path,
            truth_manifest,
            output_dir,
            predictions_sha256=prediction_sha,
        )


def test_low_scored_coverage_keeps_metrics_as_diagnostics_without_imputation(
    tmp_path: Path,
) -> None:
    effects = [4.0, 2.0, -3.0, -1.0, 0.0]
    truth_path, truth_manifest, _ = _truth_bundle(tmp_path, effects=effects)
    predictions = _rebind_universe_id(
        _prediction_frame(effects, contrasts=("misc_m_vs_s",)).iloc[:4]
    )
    predictions.loc[predictions["ligand"].eq("L4"), ["score", "status"]] = [
        np.nan,
        "not_estimable",
    ]
    predictions_path = tmp_path / "predictions.tsv"
    prediction_sha = _write_predictions(predictions_path, predictions)

    run(
        predictions_path,
        truth_path,
        truth_manifest,
        tmp_path / "evaluation",
        predictions_sha256=prediction_sha,
    )
    row = pd.read_csv(tmp_path / "evaluation/metrics.tsv", sep="\t").iloc[0]

    assert row["truth_representable_fraction"] == pytest.approx(0.8)
    assert row["representable_scored_fraction"] == pytest.approx(0.75)
    assert row["n_measured_ligands_scored"] == 3
    assert pd.notna(row["olink_ligand_ap"])
    assert pd.notna(row["olink_logfc_spearman"])
    assert pd.notna(row["protein_ndcg"])
    assert not bool(row["rank_eligible"])
    assert row["ranking_exclusion_reason"] == (
        "representable_scored_coverage_below_0_80_diagnostic_only"
    )


def test_frozen_representable_universe_not_all_olink_analytes_is_denominator(
    tmp_path: Path,
) -> None:
    effects = [4.0, 2.0, -3.0, -1.0, 0.0]
    truth_path, truth_manifest, _ = _truth_bundle(tmp_path, effects=effects)
    predictions = _rebind_universe_id(
        _prediction_frame(effects, contrasts=("misc_m_vs_s",)).iloc[:4]
    )
    predictions_path = tmp_path / "predictions.tsv"
    prediction_sha = _write_predictions(predictions_path, predictions)

    run(
        predictions_path,
        truth_path,
        truth_manifest,
        tmp_path / "evaluation",
        predictions_sha256=prediction_sha,
    )
    row = pd.read_csv(tmp_path / "evaluation/metrics.tsv", sep="\t").iloc[0]

    assert row["n_truth_olink_analytes"] == 5
    assert row["n_representable_measured_ligands"] == 4
    assert row["truth_representable_fraction"] == pytest.approx(0.8)
    assert row["representable_scored_fraction"] == pytest.approx(1.0)
    assert bool(row["rank_eligible"])


def test_native_resource_arm_is_diagnostic_only(tmp_path: Path) -> None:
    effects = [4.0, 2.0, -3.0, -1.0, 0.0]
    truth_path, truth_manifest, _ = _truth_bundle(tmp_path, effects=effects)
    predictions = _prediction_frame(effects, contrasts=("misc_m_vs_s",))
    predictions["resource_mode"] = "native"
    predictions_path = tmp_path / "predictions.tsv"
    prediction_sha = _write_predictions(predictions_path, predictions)

    run(
        predictions_path,
        truth_path,
        truth_manifest,
        tmp_path / "evaluation",
        predictions_sha256=prediction_sha,
    )
    row = pd.read_csv(tmp_path / "evaluation/metrics.tsv", sep="\t").iloc[0]

    assert row["olink_ligand_ap"] == pytest.approx(1.0)
    assert not bool(row["rank_eligible"])
    assert row["ranking_exclusion_reason"] == ("non_h_common_resource_diagnostic_only")


def test_single_class_primary_metric_is_ne_not_numeric_zero(tmp_path: Path) -> None:
    effects = [5.0, 4.0, 3.0, 2.0, 1.0]
    q_values = [0.01] * 5
    truth_path, truth_manifest, _ = _truth_bundle(
        tmp_path, effects=effects, q_values=q_values
    )
    predictions_path = tmp_path / "predictions.tsv"
    prediction_sha = _write_predictions(
        predictions_path,
        _prediction_frame(effects, contrasts=("misc_m_vs_s",)),
    )

    manifest = run(
        predictions_path,
        truth_path,
        truth_manifest,
        tmp_path / "evaluation",
        predictions_sha256=prediction_sha,
    )
    row = pd.read_csv(tmp_path / "evaluation/metrics.tsv", sep="\t").iloc[0]

    assert pd.isna(row["olink_ligand_ap"])
    assert row["olink_ligand_ap_status"] == "not_estimable"
    assert row["olink_ligand_ap_reason"] == (
        "single_class_direction_matched_q_0_05_truth"
    )
    assert row["olink_logfc_spearman"] == pytest.approx(1.0)
    assert row["protein_ndcg"] == pytest.approx(1.0)
    assert not bool(row["rank_eligible"])
    record = manifest["records"][0]
    assert record["olink_ligand_ap"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("olink_column", "Olink-free schema"),
        ("unsupported_contrast", "only canonical"),
        ("score_on_missing", "non-observed.*missing scores"),
    ],
)
def test_prediction_contract_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    effects = [4.0, 2.0, -3.0, -1.0, 0.0]
    truth_path, truth_manifest, _ = _truth_bundle(tmp_path, effects=effects)
    predictions = _prediction_frame(effects, contrasts=("misc_m_vs_s",))
    if mutation == "olink_column":
        predictions["olink_logfc"] = effects
    elif mutation == "unsupported_contrast":
        predictions["contrast"] = "misc_m_vs_c"
    else:
        predictions.loc[0, "status"] = "not_estimable"
    predictions_path = tmp_path / "predictions.tsv"
    prediction_sha = _write_predictions(predictions_path, predictions)
    output_dir = tmp_path / "evaluation"

    with pytest.raises(ValueError, match=message):
        run(
            predictions_path,
            truth_path,
            truth_manifest,
            output_dir,
            predictions_sha256=prediction_sha,
        )

    assert not output_dir.exists()


def test_checksum_mismatch_is_rejected_before_truth_evaluation(tmp_path: Path) -> None:
    truth_path, truth_manifest, _ = _truth_bundle(tmp_path)
    predictions_path = tmp_path / "predictions.tsv"
    _write_predictions(
        predictions_path,
        _prediction_frame([4.0, 2.0, -3.0, -1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="frozen checksum"):
        run(
            predictions_path,
            truth_path,
            truth_manifest,
            tmp_path / "evaluation",
            predictions_sha256="0" * 64,
        )


def test_universe_id_must_match_the_complete_frozen_ligand_set(
    tmp_path: Path,
) -> None:
    effects = [4.0, 2.0, -3.0, -1.0, 0.0]
    truth_path, truth_manifest, _ = _truth_bundle(tmp_path, effects=effects)
    first = _prediction_frame(effects, contrasts=("misc_m_vs_s",))
    second = first.copy()
    second["method_id"] = "second_method"
    second = second.loc[~second["ligand"].eq("L5")]
    predictions = pd.concat([first, second], ignore_index=True)
    predictions_path = tmp_path / "predictions.tsv"
    prediction_sha = _write_predictions(predictions_path, predictions)

    with pytest.raises(ValueError, match="universe_id does not match"):
        run(
            predictions_path,
            truth_path,
            truth_manifest,
            tmp_path / "evaluation",
            predictions_sha256=prediction_sha,
        )


def test_r_truth_export_preserves_raw_and_make_names_mapping(tmp_path: Path) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is unavailable")
    packages = subprocess.run(
        [
            rscript,
            "-e",
            (
                "quit(status=ifelse(all(vapply(c('readxl','digest','jsonlite'), "
                "requireNamespace, logical(1), quietly=TRUE)),0,1))"
            ),
        ],
        check=False,
    )
    if packages.returncode != 0:
        pytest.skip("required R truth-export packages are unavailable")
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(
        [
            "variables",
            "mean_d0",
            "mean_hc",
            "mean_diff_d0",
            "unadj_p_values",
            "adj_p_values",
        ]
    )
    sheet.append(["IL-6", 3.0, 1.0, 2.0, 0.01, 0.02])
    sheet.append(["CXCL10", 0.5, 1.0, -0.5, 0.20, 0.40])
    xlsx_path = tmp_path / "fixture.xlsx"
    workbook.save(xlsx_path)
    output_dir = tmp_path / "truth"
    script = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/comprehensive/prepare_olink_truth.R"
    )

    completed = subprocess.run(
        [
            rscript,
            str(script),
            "--input-xlsx",
            str(xlsx_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"analytes":2' in completed.stdout
    truth = pd.read_csv(output_dir / "olink_truth.tsv", sep="\t")
    assert truth["analyte_original"].tolist() == ["IL-6", "CXCL10"]
    assert truth["analyte_make_names"].tolist() == ["IL.6", "CXCL10"]
    assert truth["mean_diff_d0"].tolist() == [2.0, -0.5]
    assert set(truth["effect_semantics"]) == {
        "mean_diff_d0 = MIS-C day 0 minus Diorio healthy controls"
    }
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["truth_contract"]["q_value_column"] == "adj_p_values"
    assert manifest["truth_contract"]["reverse_effect"] == "-mean_diff_d0"
    assert manifest["truth_contract"]["canonical_contrast_id"] == "misc_m_vs_s"
    assert manifest["source"]["specimen_wording"] == {
        "vignette_and_zenodo": "serum",
        "Diorio_publication_methods": "plasma",
        "policy": "retain the source discrepancy; do not relabel as one specimen type",
    }
    assert manifest["dimensions"]["make_names_changed"] == 1
    assert manifest["dimensions"]["make_names_unchanged"] == 1
    assert (
        manifest["source"]["workbook"]["sha256"]
        == hashlib.sha256(xlsx_path.read_bytes()).hexdigest()
    )
    assert manifest["output"]["truth_tsv"]["sha256"] == sha256_file(
        output_dir / "olink_truth.tsv"
    )
