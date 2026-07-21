"""Run frozen RC3 receiver-program soft evidence on Kuppe or MS."""

from __future__ import annotations

import argparse
import json
import resource as process_resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)
from benchmarks.literature.component_swap_benchmark import _evaluate
from benchmarks.literature.receiver_program_soft import (
    fit_receiver_program_reliability,
    receiver_program_weights,
)
from benchmarks.literature.run_liana_hypergraph_residual_real import (
    METHOD_STRICT as RC2_METHOD_STRICT,
)
from benchmarks.literature.run_liana_hypergraph_residual_real import (
    SCHEMA_VERSION as RC2_SCHEMA_VERSION,
)

SCHEMA_VERSION = "crychic-receiver-program-soft-real-v1"
METHOD_STRICT = "CRYCHIC_RC3_program_soft_strict"
METHOD_FALLBACK = "CRYCHIC_RC3_program_soft_fallback"
METHOD_VERSION = "receiver-program-soft-rc3-v1"
FROZEN_SHA256 = "9bd7c0c0d8b3d581c5a2e08fd7f9bdd2cfffdc1588f569582de1f274b817545b"
RC2_SHA256 = {
    "kuppe": {
        "directed_edge_residuals.parquet": (
            "887f12fb5ec8ae727e1591cb25b58d5d5fa4d9c3434f9b0e06435463198c3e9c"
        ),
        "condition_cell_pair_rankings.tsv": (
            "7888d5e8e5e343125d7ca02a4630db4661b815d4ff75dae4797daf16c60996d6"
        ),
    },
    "ms": {
        "directed_edge_residuals.parquet": (
            "b126ccf9c847ef0a6214555bba99844ff0bc0ed7ae42913a12454db13f179856"
        ),
        "condition_cell_pair_rankings.tsv": (
            "1362c3ce01b852ea7d076cd7ea4a0350e76b2cea13f49b83ed0f48b72e167e7c"
        ),
    },
}
PROGRAM_SHA256 = {
    "kuppe": "190f3ac392186929b69e58b9f86dacdb1557e842c5a499973fbdb66d8d015517",
    "ms": "3245ca7d42500339e48957fbcd7b1292b663315acf443b5b2eea77f09f550a7c",
}
CONDITION_COLUMN = {"kuppe": "condition", "ms": "lesion_type"}


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return cast(dict[str, Any], value)


def _input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "columns": list(table.columns),
        "sha256": sha256_file(path),
    }


def _bound_output(
    root: Path,
    manifest: Mapping[str, Any],
    filename: str,
    *,
    expected_sha256: str | None = None,
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(
        outputs.get(filename), Mapping
    ):
        raise ValueError(f"manifest lacks bound output: {filename}")
    record = cast(Mapping[str, Any], outputs[filename])
    path = root / filename
    observed = sha256_file(path)
    if record.get("filename") != filename or record.get("sha256") != observed:
        raise ValueError(f"manifest output checksum mismatch: {filename}")
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"frozen input checksum mismatch: {filename}")
    return path


def derive_receiver_program_effects(
    program: pd.DataFrame,
    sample_conditions: pd.DataFrame,
    *,
    condition_column: str,
    target: str,
    reference: str,
) -> pd.DataFrame:
    """Compute sample-level target-minus-reference Welch statistics."""

    keys = ["sample_id", "receiver", "family_id"]
    required = {
        *keys,
        "receiver_program_score",
        "status",
        "formal_inference_allowed",
    }
    missing = required.difference(program.columns)
    if missing:
        raise ValueError(f"receiver-program columns are missing: {sorted(missing)}")
    if program.duplicated(keys).any():
        raise ValueError(
            "receiver-program rows are not unique per sample/receiver/family"
        )
    if program["formal_inference_allowed"].fillna(True).astype(bool).any():
        raise ValueError("benchmark-only receiver-program contract was altered")
    if sample_conditions.duplicated("sample_id").any():
        raise ValueError("sample condition mapping is not unique")
    working = program.merge(
        sample_conditions.loc[:, ["sample_id", condition_column]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if working[condition_column].isna().any():
        raise ValueError("receiver-program samples lack condition metadata")
    unexpected = set(working[condition_column].astype(str)).difference(
        {target, reference}
    )
    if unexpected:
        raise ValueError(
            f"unexpected receiver-program conditions: {sorted(unexpected)}"
        )

    universe = working.loc[:, ["receiver", "family_id"]].drop_duplicates()
    observed = working.loc[
        working["status"].eq("observed")
        & pd.to_numeric(working["receiver_program_score"], errors="coerce").notna()
    ].copy()
    observed["receiver_program_score"] = pd.to_numeric(
        observed["receiver_program_score"], errors="coerce"
    )

    def summarize(condition: str, suffix: str) -> pd.DataFrame:
        return (
            observed.loc[observed[condition_column].eq(condition)]
            .groupby(["receiver", "family_id"], observed=True, sort=True)[
                "receiver_program_score"
            ]
            .agg(["count", "mean", "var"])
            .rename(
                columns={
                    "count": f"n_samples_{suffix}",
                    "mean": f"mean_{suffix}",
                    "var": f"variance_{suffix}",
                }
            )
            .reset_index()
        )

    result = universe.merge(
        summarize(target, "target"),
        on=["receiver", "family_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        summarize(reference, "reference"),
        on=["receiver", "family_id"],
        how="left",
        validate="one_to_one",
    )
    for column in ("n_samples_target", "n_samples_reference"):
        result[column] = result[column].fillna(0).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.sqrt(
            result["variance_target"] / result["n_samples_target"]
            + result["variance_reference"] / result["n_samples_reference"]
        )
    result["program_effect_target_minus_reference"] = (
        result["mean_target"] - result["mean_reference"]
    )
    result["program_standard_error"] = standard_error
    estimable = (
        result["n_samples_target"].ge(2)
        & result["n_samples_reference"].ge(2)
        & np.isfinite(standard_error)
        & standard_error.gt(0.0)
    )
    result["program_z"] = np.nan
    result.loc[estimable, "program_z"] = (
        result.loc[estimable, "program_effect_target_minus_reference"]
        / standard_error.loc[estimable]
    )
    result["status"] = np.where(estimable, "observed", "not_estimable")
    enough_samples = result["n_samples_target"].ge(2) & result[
        "n_samples_reference"
    ].ge(2)
    result["reason_code"] = np.where(
        estimable,
        None,
        np.where(
            enough_samples,
            "nonpositive_or_nonfinite_welch_standard_error",
            "fewer_than_two_observed_samples_in_a_condition",
        ),
    )
    result["target_condition"] = target
    result["reference_condition"] = reference
    result["effect_semantics"] = (
        "sample_level_oof_receiver_program_target_minus_reference_welch_z"
    )
    result["formal_inference_allowed"] = False
    return result.sort_values(
        ["receiver", "family_id"], kind="stable", ignore_index=True
    )


def _weighted_strict_rankings(
    edges: pd.DataFrame,
    template: pd.DataFrame,
    weights: np.ndarray,
    *,
    target: str,
    reference: str,
    method: str,
    semantics: str,
) -> tuple[pd.DataFrame, int]:
    if len(edges) != len(weights):
        raise ValueError("edge and receiver-program weight lengths differ")
    p_value = pd.to_numeric(edges["interaction_pvalue"], errors="coerce").to_numpy()
    sign = pd.to_numeric(edges["final_sign_statistic"], errors="coerce").to_numpy()
    selected = (
        np.isfinite(p_value) & np.isfinite(sign) & (p_value < 0.05) & (sign != 0.0)
    )
    sender = edges["sender"].astype(str).to_numpy()
    receiver = edges["receiver"].astype(str).to_numpy()
    calls = pd.DataFrame(
        {
            "condition": np.where(sign[selected] > 0.0, target, reference),
            "sender": np.minimum(sender[selected], receiver[selected]),
            "receiver": np.maximum(sender[selected], receiver[selected]),
            "weighted_strength": weights[selected],
        }
    )
    scores = (
        calls.groupby(["condition", "sender", "receiver"], observed=True, sort=True)[
            "weighted_strength"
        ]
        .sum()
        .reset_index()
    )
    result = template.merge(
        scores,
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    )
    observed = result["status"].eq("observed")
    result.loc[observed & result["weighted_strength"].isna(), "weighted_strength"] = 0.0
    result.loc[~observed, "weighted_strength"] = np.nan
    result["ranked_strength"] = result["weighted_strength"].astype("Float64")
    result["condition_specific_directed_lr"] = result["ranked_strength"]
    result["method"] = method
    result["method_version"] = METHOD_VERSION
    result["ranking_semantics"] = semantics
    result = result.drop(columns="weighted_strength")
    return (
        result.sort_values(
            ["dataset", "method", "condition", "sender", "receiver"],
            kind="stable",
            ignore_index=True,
        ),
        int(selected.sum()),
    )


def _assert_alpha_zero_parity(
    rebuilt: pd.DataFrame, original: pd.DataFrame
) -> dict[str, object]:
    compare = [
        "condition",
        "sender",
        "receiver",
        "ranked_strength",
        "condition_specific_directed_lr",
        "estimable_directed_lr",
        "status",
        "reason_code",
    ]
    order = ["condition", "sender", "receiver"]
    left = (
        rebuilt.sort_values(order, kind="stable").loc[:, compare].reset_index(drop=True)
    )
    right = (
        original.sort_values(order, kind="stable")
        .loc[:, compare]
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    return {"rankings_exact": True, "ranking_rows": len(left)}


def _coverage_fallback(
    strict: pd.DataFrame, all_rc2_rankings: pd.DataFrame
) -> pd.DataFrame:
    result = strict.copy()
    observed = result["status"].eq("observed")
    result["_strict_rank"] = np.nan
    result.loc[observed, "_strict_rank"] = (
        result.loc[observed]
        .groupby("condition", observed=True)["ranked_strength"]
        .rank(pct=True, method="average")
    )
    fallback = all_rc2_rankings.loc[
        all_rc2_rankings["method"].eq("C"),
        [
            "condition",
            "sender",
            "receiver",
            "ranked_strength",
            "estimable_directed_lr",
            "status",
        ],
    ].copy()
    fallback_observed = fallback["status"].eq("observed")
    fallback["_fallback_rank"] = np.nan
    fallback.loc[fallback_observed, "_fallback_rank"] = (
        fallback.loc[fallback_observed]
        .groupby("condition", observed=True)["ranked_strength"]
        .rank(pct=True, method="average")
    )
    fallback = fallback.rename(columns={"estimable_directed_lr": "_fallback_estimable"})
    result = result.merge(
        fallback.loc[
            :,
            [
                "condition",
                "sender",
                "receiver",
                "_fallback_rank",
                "_fallback_estimable",
            ],
        ],
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    )
    observed = result["status"].eq("observed")
    use_fallback = ~observed & result["_fallback_rank"].notna()
    result["ranked_strength"] = result["_strict_rank"].where(
        observed, result["_fallback_rank"]
    )
    result.loc[use_fallback, "estimable_directed_lr"] = result.loc[
        use_fallback, "_fallback_estimable"
    ]
    result.loc[use_fallback, "condition_specific_directed_lr"] = np.nan
    result["status"] = np.where(
        result["ranked_strength"].notna(), "observed", "not_estimable"
    )
    result["reason_code"] = np.where(
        observed,
        None,
        np.where(
            use_fallback,
            "program_soft_not_estimable_used_crychic_wald_rank_fallback",
            "program_soft_and_crychic_fallback_not_estimable",
        ),
    )
    result["method"] = METHOD_FALLBACK
    result["method_version"] = METHOD_VERSION
    result["ranking_semantics"] = (
        "receiver_program_soft_cardinality_percentile_with_crychic_wald_"
        "percentile_only_when_liana_pair_not_estimable"
    )
    return result.drop(
        columns=["_strict_rank", "_fallback_rank", "_fallback_estimable"]
    )


def _load_rc2(
    dataset: str, rc2_run: Path, expected_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest_path = rc2_run / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != RC2_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != dataset
    ):
        raise ValueError("RC2 run schema, status, or dataset mismatch")
    edge_path = _bound_output(
        rc2_run,
        manifest,
        "directed_edge_residuals.parquet",
        expected_sha256=RC2_SHA256[dataset]["directed_edge_residuals.parquet"],
    )
    ranking_path = _bound_output(
        rc2_run,
        manifest,
        "condition_cell_pair_rankings.tsv",
        expected_sha256=RC2_SHA256[dataset]["condition_cell_pair_rankings.tsv"],
    )
    expected_record = (
        manifest.get("inputs", {}).get("component", {}).get("expected_sets")
    )
    if not isinstance(expected_record, Mapping) or expected_record.get(
        "sha256"
    ) != sha256_file(expected_path):
        raise ValueError("expected spatial sets do not match the frozen RC2 run")
    rankings = pd.read_csv(ranking_path, sep="\t")
    strict = rankings.loc[rankings["method"].eq(RC2_METHOD_STRICT)].copy()
    if strict.empty:
        raise ValueError("RC2 strict ranking template is missing")
    return (
        pd.read_parquet(edge_path),
        rankings,
        pd.read_csv(expected_path, sep="\t"),
        {
            "manifest": _input_record(manifest_path),
            "edges": _input_record(edge_path),
            "rankings": _input_record(ranking_path),
            "expected_sets": _input_record(expected_path),
        },
    )


def _load_program_run(
    dataset: str, program_run: Path
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    str,
    dict[str, object],
]:
    manifest_path = program_run / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("CRYCHIC program source run is incomplete")
    contrast = manifest.get("contrast")
    if not isinstance(contrast, Mapping):
        raise ValueError("program source contrast is missing")
    target = str(contrast["target"])
    reference = str(contrast["reference"])
    layer_path = _bound_output(program_run, manifest, "sender_lr_score_layers.parquet")
    family_path = _bound_output(program_run, manifest, "directed_lr_effects.parquet")
    crossfit = manifest.get("crossfit_result")
    if not isinstance(crossfit, Mapping):
        raise ValueError("program source crossfit result is missing")
    crossfit_root = program_run / str(crossfit["directory"])
    crossfit_manifest_path = crossfit_root / "crossfit_manifest.json"
    if crossfit.get("manifest_sha256") != sha256_file(crossfit_manifest_path):
        raise ValueError("program source crossfit manifest checksum mismatch")
    program_path = crossfit_root / "semantic_receiver_program_scores.parquet"
    if sha256_file(program_path) != PROGRAM_SHA256[dataset]:
        raise ValueError("frozen receiver-program table checksum mismatch")

    condition_column = CONDITION_COLUMN[dataset]
    sample_conditions = pd.read_parquet(
        layer_path, columns=["sample_id", condition_column]
    ).drop_duplicates()
    if sample_conditions.duplicated("sample_id").any():
        raise ValueError("score layer maps a sample to multiple conditions")
    family_keys = ["sender", "receiver", "ligand", "receptor"]
    family_map = pd.read_parquet(
        family_path, columns=[*family_keys, "family_id"]
    ).drop_duplicates()
    if family_map.duplicated(family_keys).any():
        raise ValueError("directed LR keys map to multiple program families")
    program = pd.read_parquet(program_path)
    provenance = {
        "manifest": _input_record(manifest_path),
        "crossfit_manifest": _input_record(crossfit_manifest_path),
        "receiver_program_scores": _input_record(program_path),
        "sample_condition_layer": _input_record(layer_path),
        "family_mapping": _input_record(family_path),
        "crossfit_schema_version": crossfit.get("schema_version"),
        "heldout_fold_audit": crossfit.get("heldout_fold_audit"),
    }
    return program, sample_conditions, family_map, target, reference, provenance


def run(
    *,
    dataset: str,
    rc2_run: Path,
    program_run: Path,
    expected_path: Path,
    frozen_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if sha256_file(frozen_path) != FROZEN_SHA256:
        raise ValueError("RC3 frozen simulation candidate checksum mismatch")
    frozen = _read_json(frozen_path)
    candidate = cast(Mapping[str, Any], frozen["candidate"])
    policy = cast(Mapping[str, Any], frozen["program_policy"])
    if candidate.get("name") != "program_max_alpha_4":
        raise ValueError("unexpected RC3 frozen candidate")
    output = prepare_output(output_dir, overwrite=overwrite)
    edges, rc2_rankings, expected, rc2_provenance = _load_rc2(
        dataset, rc2_run.resolve(), expected_path.resolve()
    )
    (
        program,
        sample_conditions,
        family_map,
        target,
        reference,
        program_provenance,
    ) = _load_program_run(dataset, program_run.resolve())
    strict_template = rc2_rankings.loc[
        rc2_rankings["method"].eq(RC2_METHOD_STRICT)
    ].copy()
    if set(strict_template["condition"].astype(str)) != {target, reference}:
        raise ValueError("RC2 and receiver-program contrasts differ")
    effects = derive_receiver_program_effects(
        program,
        sample_conditions,
        condition_column=CONDITION_COLUMN[dataset],
        target=target,
        reference=reference,
    )
    family_keys = ["sender", "receiver", "ligand", "receptor"]
    aligned = edges.merge(
        family_map,
        on=family_keys,
        how="left",
        validate="one_to_one",
    ).merge(
        effects.loc[:, ["receiver", "family_id", "program_z", "status"]].rename(
            columns={"status": "program_status"}
        ),
        on=["receiver", "family_id"],
        how="left",
        validate="many_to_one",
    )
    fit = fit_receiver_program_reliability(
        aligned["interaction_stat"].to_numpy(dtype=float),
        aligned["program_z"].to_numpy(dtype=float),
        maximum_alpha=float(candidate["maximum_alpha"]),
        alpha_cap=float(policy["alpha_cap"]),
        minimum_edges=int(policy["minimum_concordance_edges"]),
    )
    final_sign = pd.to_numeric(
        aligned["final_sign_statistic"], errors="coerce"
    ).to_numpy(dtype=float)
    direction = np.where(final_sign >= 0.0, 1.0, -1.0)
    weights, evidence = receiver_program_weights(
        direction, aligned["program_z"].to_numpy(dtype=float), fit
    )
    alpha_zero, alpha_zero_calls = _weighted_strict_rankings(
        aligned,
        strict_template,
        np.ones(len(aligned), dtype=float),
        target=target,
        reference=reference,
        method="RC3_alpha_zero_parity_check",
        semantics="alpha_zero_parity_check",
    )
    parity = _assert_alpha_zero_parity(alpha_zero, strict_template)
    parity["selected_calls"] = alpha_zero_calls
    strict, selected_calls = _weighted_strict_rankings(
        aligned,
        strict_template,
        weights,
        target=target,
        reference=reference,
        method=METHOD_STRICT,
        semantics=(
            "exact_rc2_calls_weighted_by_reliability_shrunk_oof_receiver_program_"
            "evidence;missing_program_evidence_neutral"
        ),
    )
    if selected_calls != alpha_zero_calls:
        raise AssertionError("RC3 changed the frozen RC2 call set")
    fallback = _coverage_fallback(strict, rc2_rankings)
    all_rankings = pd.concat([rc2_rankings, strict, fallback], ignore_index=True)
    scores, coverage, summary = _evaluate(all_rankings, expected, dataset=dataset)

    aligned["program_evidence"] = evidence
    aligned["program_weight"] = weights
    aligned["program_available"] = aligned["program_z"].notna()
    aligned["selected_rc2_call"] = pd.to_numeric(
        aligned["interaction_pvalue"], errors="coerce"
    ).lt(0.05) & pd.to_numeric(aligned["final_sign_statistic"], errors="coerce").ne(0.0)
    aligned["formal_inference_allowed"] = False
    paths: dict[str, tuple[Path, pd.DataFrame]] = {
        "directed_edge_program_weights.parquet": (
            output / "directed_edge_program_weights.parquet",
            aligned,
        ),
        "receiver_program_effects.parquet": (
            output / "receiver_program_effects.parquet",
            effects,
        ),
        "condition_cell_pair_rankings.tsv": (
            output / "condition_cell_pair_rankings.tsv",
            all_rankings,
        ),
        "spatial_des_scores.tsv": (output / "spatial_des_scores.tsv", scores),
        "spatial_des_coverage.tsv": (
            output / "spatial_des_coverage.tsv",
            coverage,
        ),
        "spatial_des_summary.tsv": (
            output / "spatial_des_summary.tsv",
            summary,
        ),
    }
    for filename, (path, table) in paths.items():
        if filename.endswith(".parquet"):
            table.to_parquet(path, index=False, compression="zstd")
        else:
            table.to_csv(path, sep="\t", index=False, lineterminator="\n")

    finite_program = aligned["program_z"].notna()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "contrast": {"target": target, "reference": reference},
        "frozen_candidate": frozen,
        "receiver_program_fit": fit.to_dict(),
        "alignment": {
            "rc2_edges": len(aligned),
            "family_mapped_edges": int(aligned["family_id"].notna().sum()),
            "family_mapping_coverage": float(aligned["family_id"].notna().mean()),
            "program_observed_edges": int(finite_program.sum()),
            "program_coverage": float(finite_program.mean()),
            "selected_rc2_calls": selected_calls,
            "missing_evidence_weight_is_neutral": True,
        },
        "alpha_zero_rc2_parity": parity,
        "backbone_modified": False,
        "rc2_call_set_modified": False,
        "benchmark_head_modified": True,
        "formal_release_allowed": False,
        "limitations": [
            "receiver-program source is partial-pipeline cross-fit evidence",
            "source rows and RC3 outputs do not permit formal inference",
            "sample-level Welch effects are descriptive benchmark weights",
        ],
        "inputs": {
            "rc2": rc2_provenance,
            "program": program_provenance,
            "frozen_candidate": _input_record(frozen_path),
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(
                process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "code": git_metadata(Path(__file__).resolve().parents[2]),
        "outputs": {
            filename: _output_record(path, table)
            for filename, (path, table) in paths.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kuppe", "ms"), required=True)
    parser.add_argument("--rc2-run", type=Path, required=True)
    parser.add_argument("--program-run", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        dataset=args.dataset,
        rc2_run=args.rc2_run,
        program_run=args.program_run,
        expected_path=args.expected,
        frozen_path=args.frozen,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "fit": result["receiver_program_fit"],
                "alignment": result["alignment"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
