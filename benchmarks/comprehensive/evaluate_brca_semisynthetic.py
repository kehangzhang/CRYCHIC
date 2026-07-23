"""Evaluate known-truth paired 2x2 effects from per-sample CCC scores."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-brca-semisynthetic-evaluation-v1"
EVENT_KEYS = ("sender", "receiver", "interaction_id", "ligand", "receptor")
REQUIRED_SCORE_COLUMNS = {
    "run_id",
    "dataset_id",
    "method_id",
    "resource_mode",
    "sample_id",
    "score",
    "score_direction",
    "status",
    *EVENT_KEYS,
}
ENGINES = ("native_raw_mean", "within_sample_rank_mean")
_DIRECTION_MULTIPLIER = {
    "higher": 1.0,
    "higher_is_stronger": 1.0,
    "lower": -1.0,
    "lower_is_stronger": -1.0,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _publish(staged: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    if not output.exists():
        os.replace(staged, output)
        return
    backup = output.with_name(f".{output.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(output, backup)
    try:
        os.replace(staged, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def _bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.notna() & np.isfinite(values) & values.between(0.0, 1.0)
    observed = values.loc[finite].to_numpy(dtype=float)
    if not len(observed):
        return result
    order = np.argsort(observed, kind="stable")
    ranked = observed[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result.loc[finite] = restored
    return result


def _design_table(path: Path) -> pd.DataFrame:
    source = ad.read_h5ad(path, backed="r")
    try:
        required = {"sample_id", "subject_id", "timepoint", "expansion"}
        missing = required.difference(source.obs.columns)
        if missing:
            raise ValueError(f"factorial metadata is missing: {sorted(missing)}")
        raw = source.obs.loc[:, sorted(required)]
        if raw.isna().any().any():
            raise ValueError("factorial metadata contains missing values")
        design = raw.astype(str).drop_duplicates()
    finally:
        source.file.close()
    if design.duplicated("sample_id", keep=False).any():
        raise ValueError("one sample maps to multiple factorial design rows")
    if set(design["timepoint"]) != {"Pre", "On"}:
        raise ValueError("timepoint must contain Pre and On")
    if set(design["expansion"]) != {"E", "NE"}:
        raise ValueError("expansion must contain E and NE")
    subject_expansion = design[["subject_id", "expansion"]].drop_duplicates()
    if subject_expansion.duplicated("subject_id", keep=False).any():
        raise ValueError("one subject maps to multiple expansion groups")
    support = design.groupby(
        ["subject_id", "timepoint"], observed=True, sort=False
    ).size()
    if len(support) != 2 * design["subject_id"].nunique() or not support.eq(1).all():
        raise ValueError("every subject must have one Pre and one On sample")
    return design.sort_values("sample_id", kind="stable", ignore_index=True)


def _view_labels(manifest_path: Path | None) -> dict[str, str]:
    if manifest_path is None or not manifest_path.is_file():
        return {}
    value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"method manifest must be an object: {manifest_path}")
    source_result = value.get("source_result")
    if not isinstance(source_result, dict):
        return {}
    views = source_result.get("score_views")
    if not isinstance(views, list):
        return {}
    labels: dict[str, str] = {}
    for index, item in enumerate(views, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str):
            continue
        candidates = item.get("contrast_candidates")
        labels[str(item["run_id"])] = (
            ";".join(map(str, candidates))
            if isinstance(candidates, list) and candidates
            else f"score_view_{index}"
        )
    return labels


def _runtime_metadata(
    manifest_path: Path | None,
    time_path: Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "adapter_elapsed_seconds": math.nan,
        "process_wall_seconds": math.nan,
        "peak_rss_kib": math.nan,
    }
    if manifest_path is not None and manifest_path.is_file():
        value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            elapsed = value.get("elapsed_seconds")
            if isinstance(elapsed, int | float) and not isinstance(elapsed, bool):
                result["adapter_elapsed_seconds"] = float(elapsed)
    if time_path is None or not time_path.is_file():
        return result
    text = time_path.read_text(encoding="utf-8")
    wall_match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)",
        text,
    )
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if wall_match:
        parts = [float(item) for item in wall_match.group(1).strip().split(":")]
        if len(parts) == 2:
            result["process_wall_seconds"] = 60.0 * parts[0] + parts[1]
        elif len(parts) == 3:
            result["process_wall_seconds"] = (
                3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
            )
    if rss_match:
        result["peak_rss_kib"] = float(rss_match.group(1))
    return result


def _score_variants(
    score_path: Path,
    design: pd.DataFrame,
    *,
    manifest_path: Path | None,
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    table = pd.read_parquet(score_path)
    missing = REQUIRED_SCORE_COLUMNS.difference(table.columns)
    if table.empty or missing:
        raise ValueError(f"invalid score table {score_path}: missing={sorted(missing)}")
    dataset_ids = tuple(sorted(table["dataset_id"].astype(str).unique()))
    if len(dataset_ids) != 1:
        raise ValueError("one method table must contain exactly one dataset")
    if set(table["resource_mode"].astype(str)) != {"H-common"}:
        raise ValueError("semi-synthetic comparison requires H-common scores")
    unknown_samples = set(table["sample_id"].astype(str)).difference(
        design["sample_id"]
    )
    if unknown_samples:
        raise ValueError(
            f"score table has unknown samples: {sorted(unknown_samples)[:5]}"
        )
    labels = _view_labels(manifest_path)
    variants: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for run_id, group in table.groupby("run_id", observed=True, sort=True):
        methods = tuple(sorted(group["method_id"].astype(str).unique()))
        directions = tuple(sorted(group["score_direction"].astype(str).unique()))
        if len(methods) != 1 or len(directions) != 1:
            raise ValueError("one score view must have one method and direction")
        if directions[0] not in _DIRECTION_MULTIPLIER:
            raise ValueError(f"unsupported score direction: {directions[0]}")
        keys = ["sample_id", *EVENT_KEYS]
        if group.duplicated(keys).any():
            raise ValueError(f"score view {run_id} has duplicate sample-event rows")
        selected = group.loc[:, [*keys, "score", "status"]].copy()
        numeric = pd.to_numeric(selected["score"], errors="coerce")
        observed = selected["status"].astype(str).eq("ok") & np.isfinite(numeric)
        selected["oriented_score"] = np.where(
            observed,
            numeric * _DIRECTION_MULTIPLIER[directions[0]],
            np.nan,
        )
        selected = selected.merge(
            design,
            on="sample_id",
            how="left",
            validate="many_to_one",
        )
        variants.append(
            (
                {
                    "dataset_id": dataset_ids[0],
                    "base_method_id": methods[0],
                    "run_id": str(run_id),
                    "view_label": labels.get(str(run_id), "canonical_sample_score"),
                    "score_direction": directions[0],
                    "score_path": str(score_path.resolve()),
                    "score_sha256": sha256_file(score_path),
                    "manifest_path": (
                        None if manifest_path is None else str(manifest_path.resolve())
                    ),
                },
                selected,
            )
        )
    return variants


def _engine_scores(table: pd.DataFrame, engine: str) -> pd.DataFrame:
    result = table.copy()
    if engine == "native_raw_mean":
        result["engine_score"] = result["oriented_score"]
    elif engine == "within_sample_rank_mean":
        result["engine_score"] = result.groupby(
            "sample_id", observed=True, sort=False
        )["oriented_score"].rank(method="average", pct=True, na_option="keep")
    else:
        raise ValueError(f"unknown differential engine: {engine}")
    return result


def _subject_deltas(table: pd.DataFrame) -> pd.DataFrame:
    index = ["subject_id", "expansion", *EVENT_KEYS]
    scores = (
        table.groupby([*index, "timepoint"], observed=True, sort=False)[
            "engine_score"
        ]
        .mean()
        .unstack("timepoint")
    )
    if not {"Pre", "On"}.issubset(scores.columns):
        return pd.DataFrame(columns=[*index, "paired_delta"])
    paired = scores.loc[:, ["Pre", "On"]].dropna()
    paired["paired_delta"] = paired["On"] - paired["Pre"]
    return paired[["paired_delta"]].reset_index()


def _paired_effects(
    deltas: pd.DataFrame,
    *,
    min_subjects_per_group: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for event, group in deltas.groupby(list(EVENT_KEYS), observed=True, sort=False):
        values = {
            str(expansion): frame["paired_delta"].to_numpy(dtype=float)
            for expansion, frame in group.groupby("expansion", observed=True)
        }
        left = values.get("E", np.empty(0, dtype=float))
        right = values.get("NE", np.empty(0, dtype=float))
        estimable = (
            len(left) >= min_subjects_per_group
            and len(right) >= min_subjects_per_group
            and np.isfinite(left).all()
            and np.isfinite(right).all()
        )
        effect = float(left.mean() - right.mean()) if estimable else math.nan
        p_value = math.nan
        if estimable and (np.var(left) > 0.0 or np.var(right) > 0.0):
            p_value = float(
                ttest_ind(left, right, equal_var=False, nan_policy="raise").pvalue
            )
        records.append(
            {
                **dict(zip(EVENT_KEYS, event, strict=True)),
                "n_subjects_E": len(left),
                "n_subjects_NE": len(right),
                "mean_delta_E": float(left.mean()) if len(left) else math.nan,
                "mean_delta_NE": float(right.mean()) if len(right) else math.nan,
                "difference_in_differences": effect,
                "p_value": p_value,
                "status": "observed" if estimable else "not_estimable",
                "reason_code": (
                    ""
                    if estimable
                    else "insufficient_complete_paired_subjects_per_expansion"
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return pd.DataFrame(
            columns=[
                *EVENT_KEYS,
                "n_subjects_E",
                "n_subjects_NE",
                "mean_delta_E",
                "mean_delta_NE",
                "difference_in_differences",
                "p_value",
                "q_value",
                "status",
                "reason_code",
            ]
        )
    result["q_value"] = _bh_adjust(result["p_value"])
    return result


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or not len(labels):
        return math.nan
    return float(average_precision_score(labels, scores))


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or not len(labels):
        return math.nan
    return float(roc_auc_score(labels, scores))


def _metrics(effects: pd.DataFrame) -> dict[str, Any]:
    observed = effects["status"].eq("observed") & np.isfinite(
        pd.to_numeric(effects["difference_in_differences"], errors="coerce")
    )
    scored = effects.loc[observed].copy()
    labels = scored["truth_label"].to_numpy(dtype=int)
    directions = scored["truth_direction"].to_numpy(dtype=int)
    values = scored["difference_in_differences"].to_numpy(dtype=float)
    active = directions != 0
    positive = directions > 0
    negative = directions < 0
    main_control = scored["planted_main_effect_control"].astype(bool).to_numpy()
    no_effect = ~(active | main_control)
    discovered = pd.to_numeric(scored["q_value"], errors="coerce").lt(0.05).to_numpy()
    true_positives = int(np.sum(discovered & active))
    false_positives = int(np.sum(discovered & ~active))
    active_median = (
        float(np.median(np.abs(values[active]))) if active.any() else math.nan
    )
    main_median = (
        float(np.median(np.abs(values[main_control])))
        if main_control.any()
        else math.nan
    )
    return {
        "n_truth_events": len(effects),
        "n_truth_active": int(effects["truth_label"].sum()),
        "n_estimable_events": int(observed.sum()),
        "coverage": float(observed.mean()),
        "active_coverage": float(
            observed.loc[effects["truth_label"].eq(1)].mean()
        ),
        "omnibus_auprc": _safe_ap(labels, np.abs(values)),
        "omnibus_auroc": _safe_auc(labels, np.abs(values)),
        "positive_direction_ap": _safe_ap(positive.astype(int), values),
        "negative_direction_ap": _safe_ap(negative.astype(int), -values),
        "direction_accuracy_active": (
            float(np.mean(np.sign(values[active]) == directions[active]))
            if active.any()
            else math.nan
        ),
        "effect_all_zero_fraction": float(np.mean(values == 0.0)),
        "effect_dynamic_range": float(np.max(values) - np.min(values)),
        "active_median_abs_effect": active_median,
        "main_only_median_abs_leakage": main_median,
        "main_to_active_abs_ratio": (
            main_median / active_median
            if np.isfinite(main_median) and active_median > 0.0
            else math.nan
        ),
        "no_effect_standard_deviation": (
            float(np.std(values[no_effect], ddof=1))
            if np.sum(no_effect) > 1
            else math.nan
        ),
        "discoveries_q_lt_005": int(discovered.sum()),
        "true_positive_discoveries": true_positives,
        "false_positive_discoveries": false_positives,
        "empirical_fdr": (
            false_positives / int(discovered.sum()) if discovered.any() else 0.0
        ),
        "power": true_positives / int(np.sum(active)) if active.any() else math.nan,
        "main_only_false_discoveries": int(np.sum(discovered & main_control)),
    }


def _truth_for_dataset(truth: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    selected = truth.loc[truth["dataset_id"].astype(str).eq(dataset_id)].copy()
    required = {
        *EVENT_KEYS,
        "event_class",
        "truth_label",
        "truth_direction",
        "planted_main_effect_control",
    }
    missing = required.difference(selected.columns)
    if selected.empty or missing:
        raise ValueError(
            f"truth is missing dataset or columns for {dataset_id}: {sorted(missing)}"
        )
    if selected.duplicated(list(EVENT_KEYS)).any():
        raise ValueError("truth contains duplicate event keys")
    return selected


def evaluate(
    truth_path: Path,
    dataset_inputs: Mapping[str, Path],
    method_specs: Sequence[tuple[Path, Path | None, Path | None]],
    output_dir: Path,
    *,
    min_subjects_per_group: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    if min_subjects_per_group < 2:
        raise ValueError("minimum subjects per expansion must be at least two")
    truth_path = truth_path.expanduser().resolve()
    truth = pd.read_csv(truth_path, sep="\t", keep_default_na=False)
    designs = {
        dataset_id: _design_table(path.expanduser().resolve())
        for dataset_id, path in dataset_inputs.items()
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    effects_parts: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    published = False
    try:
        for score_path, manifest_path, time_path in method_specs:
            resolved_score = score_path.expanduser().resolve()
            method_table = pd.read_parquet(
                resolved_score, columns=["dataset_id"]
            )
            dataset_ids = tuple(
                sorted(method_table["dataset_id"].astype(str).unique())
            )
            if len(dataset_ids) != 1 or dataset_ids[0] not in designs:
                raise ValueError(
                    f"method dataset is not registered: {dataset_ids}"
                )
            dataset_id = dataset_ids[0]
            runtime = _runtime_metadata(manifest_path, time_path)
            for metadata, table in _score_variants(
                resolved_score,
                designs[dataset_id],
                manifest_path=manifest_path,
            ):
                truth_dataset = _truth_for_dataset(truth, dataset_id)
                for engine in ENGINES:
                    variant_id = (
                        f"{metadata['base_method_id']}::{metadata['run_id']}::{engine}"
                    )
                    deltas = _subject_deltas(_engine_scores(table, engine))
                    estimated = _paired_effects(
                        deltas,
                        min_subjects_per_group=min_subjects_per_group,
                    )
                    merged = truth_dataset.merge(
                        estimated,
                        on=list(EVENT_KEYS),
                        how="left",
                        validate="one_to_one",
                    )
                    merged["status"] = merged["status"].fillna("not_estimable")
                    merged["reason_code"] = merged["reason_code"].fillna(
                        "event_not_emitted_or_incomplete_pairs"
                    )
                    merged.insert(0, "method_variant_id", variant_id)
                    merged.insert(1, "base_method_id", metadata["base_method_id"])
                    merged.insert(2, "run_id", metadata["run_id"])
                    merged.insert(3, "view_label", metadata["view_label"])
                    merged.insert(4, "differential_engine", engine)
                    effects_parts.append(merged)
                    metric_records.append(
                        {
                            "method_variant_id": variant_id,
                            "dataset_id": dataset_id,
                            "base_method_id": metadata["base_method_id"],
                            "run_id": metadata["run_id"],
                            "view_label": metadata["view_label"],
                            "differential_engine": engine,
                            **_metrics(merged),
                            **runtime,
                        }
                    )
                variants.append(
                    {
                        **metadata,
                        "time_path": (
                            None if time_path is None else str(time_path.resolve())
                        ),
                    }
                )

        all_effects = pd.concat(effects_parts, ignore_index=True, sort=False)
        metrics = pd.DataFrame.from_records(metric_records).sort_values(
            ["dataset_id", "omnibus_auprc", "method_variant_id"],
            ascending=[True, False, True],
            kind="stable",
            ignore_index=True,
        )
        metrics["omnibus_auprc_rank"] = metrics.groupby(
            "dataset_id", observed=True
        )["omnibus_auprc"].rank(method="min", ascending=False)
        effects_path = staged / "event_effects.tsv.gz"
        metrics_path = staged / "metrics.tsv"
        all_effects.to_csv(effects_path, sep="\t", index=False, compression="gzip")
        metrics.to_csv(metrics_path, sep="\t", index=False)
        report_lines = [
            "# BRCA-shaped 2x2 semi-synthetic benchmark",
            "",
            "Known interaction truth is evaluated with missing events left missing.",
            "No best CRYCHIC context-trained view is selected post hoc.",
            "",
            "| Rank | Method | View | Engine | AUPRC | AUROC | Direction | "
            "Main/active leakage | Power | FDR |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in metrics.itertuples(index=False):
            report_lines.append(
                "| {rank:.0f} | {method} | {view} | {engine} | {ap:.4f} | "
                "{auc:.4f} | {direction:.4f} | {leakage:.4f} | {power:.4f} | "
                "{fdr:.4f} |".format(
                    rank=row.omnibus_auprc_rank,
                    method=row.base_method_id,
                    view=str(row.view_label).replace("|", "/"),
                    engine=row.differential_engine,
                    ap=row.omnibus_auprc,
                    auc=row.omnibus_auroc,
                    direction=row.direction_accuracy_active,
                    leakage=row.main_to_active_abs_ratio,
                    power=row.power,
                    fdr=row.empirical_fdr,
                )
            )
        report_path = staged / "REPORT.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "estimand": "(On-Pre in E) - (On-Pre in NE)",
            "engines": list(ENGINES),
            "minimum_subjects_per_expansion": min_subjects_per_group,
            "missing_policy": "complete_paired_subjects_per_event;never_zero_imputed",
            "truth": {
                "filename": truth_path.name,
                "sha256": sha256_file(truth_path),
                "rows": len(truth),
            },
            "datasets": {
                dataset_id: {
                    "input": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for dataset_id, path in dataset_inputs.items()
            },
            "variants": variants,
            "outputs": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (effects_path, metrics_path, report_path)
            },
            "code": git_metadata(Path(__file__).resolve().parents[2]),
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _method_spec(value: str) -> tuple[Path, Path | None, Path | None]:
    parts = value.split("::")
    if not 1 <= len(parts) <= 3 or not parts[0]:
        raise argparse.ArgumentTypeError(
            "--method expects SCORE[::MANIFEST[::TIME_FILE]]"
        )
    paths = [Path(part) if part else None for part in parts]
    return (
        paths[0],
        paths[1] if len(paths) > 1 else None,
        paths[2] if len(paths) > 2 else None,
    )  # type: ignore[return-value]


def _dataset_spec(value: str) -> tuple[str, Path]:
    dataset, separator, path = value.partition("=")
    if not separator or not dataset or not path:
        raise argparse.ArgumentTypeError("--dataset expects DATASET_ID=H5AD")
    return dataset, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--dataset", required=True, action="append", type=_dataset_spec)
    parser.add_argument("--method", required=True, action="append", type=_method_spec)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-subjects-per-group", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    datasets = dict(args.dataset)
    if len(datasets) != len(args.dataset):
        raise ValueError("dataset identifiers must be unique")
    manifest = evaluate(
        args.truth,
        datasets,
        args.method,
        args.output_dir,
        min_subjects_per_group=args.min_subjects_per_group,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest["outputs"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
