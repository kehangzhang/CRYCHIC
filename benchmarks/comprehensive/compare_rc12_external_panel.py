"""Compare the frozen RC12 detection head with the same-seed external panel."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-rc12-external-comparison-v1"
CANDIDATE_SCHEMA = "crychic-bounded-detection-evidence-evaluation-v1"
EXTERNAL_SCHEMA = "crychic-three-group-evaluation-v2"
CANDIDATE_SOURCE_METHOD = "sender_downstream_w010__native_raw_mean"
CANDIDATE_METHOD = "crychic_rc12_detection"
COMPARATORS = (
    "scseqcommdiff",
    "cellchat",
    "crychic",
    "liana_rank_aggregate",
)
METHOD_LABELS = {
    CANDIDATE_METHOD: "CRYCHIC RC12 detection head",
    "scseqcommdiff": "scSeqCommDiff",
    "cellchat": "CellChat",
    "crychic": "CRYCHIC generic baseline",
    "liana_rank_aggregate": "LIANA",
}
METRICS = (
    ("active", "omnibus_prevalence_adjusted_ap", "higher"),
    ("active", "omnibus_auprc", "higher"),
    ("active", "omnibus_auroc", "higher"),
    ("active", "localization_macro_auprc", "higher"),
    ("global_null", "effect_standard_deviation", "lower"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _bound_table(
    root: Path, manifest: Mapping[str, Any], filename: str
) -> pd.DataFrame:
    outputs = manifest.get("outputs")
    record = outputs.get(filename) if isinstance(outputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"manifest does not bind {filename}: {root}")
    path = root / filename
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"checksum mismatch for {path}")
    return pd.read_csv(path, sep="\t")


def _validate_alignment(
    candidate_root: Path,
    external_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_manifest_path = candidate_root / "manifest.json"
    external_manifest_path = external_root / "manifest.json"
    candidate = _read_json(candidate_manifest_path)
    external = _read_json(external_manifest_path)
    if candidate.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError("unsupported RC12 candidate manifest")
    if external.get("schema_version") != EXTERNAL_SCHEMA:
        raise ValueError("unsupported external evaluation manifest")
    if candidate.get("status") != "complete" or external.get("status") != "complete":
        raise ValueError("both source evaluations must be complete")
    gate = candidate.get("gate")
    if not isinstance(gate, Mapping) or gate.get("status") != "ACCEPT":
        raise ValueError("RC12 candidate did not pass its frozen validation gate")
    if gate.get("primary_detection_head") != CANDIDATE_SOURCE_METHOD:
        raise ValueError("unexpected RC12 detection head")
    source = candidate.get("source_evaluation")
    if not isinstance(source, Mapping):
        raise ValueError("RC12 source evaluation binding is absent")
    source_dir = Path(str(source.get("directory", ""))).resolve()
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.is_file() or sha256_file(
        source_manifest_path
    ) != source.get("manifest_sha256"):
        raise ValueError("RC12 source evaluation checksum mismatch")
    source_manifest = _read_json(source_manifest_path)
    alignment_fields = (
        "fixture_manifest_sha256",
        "truth_sha256",
        "frozen_code_commit",
    )
    for field in alignment_fields:
        if source_manifest.get(field) != external.get(field):
            raise ValueError(f"source evaluations disagree on {field}")
    if external.get("missing_runs") or external.get("excluded_runs"):
        raise ValueError("external panel is incomplete")
    return candidate, external


def _combined_metrics(
    candidate_metrics: pd.DataFrame,
    external_metrics: pd.DataFrame,
) -> pd.DataFrame:
    candidate = candidate_metrics.loc[
        candidate_metrics["method"].astype(str).eq(CANDIDATE_SOURCE_METHOD)
    ].copy()
    candidate["method"] = CANDIDATE_METHOD
    external = external_metrics.loc[
        external_metrics["method"].astype(str).isin(COMPARATORS)
    ].copy()
    required = {CANDIDATE_METHOD, *COMPARATORS}
    combined = pd.concat([candidate, external], ignore_index=True, sort=False)
    if set(combined["method"].astype(str)) != required:
        raise ValueError("one or more comparison methods are absent")
    keys = ["scenario", "seed", "dataset_id", "method"]
    if combined.duplicated(keys).any():
        raise ValueError("comparison metrics contain duplicate method/seed rows")
    for method in required:
        selected = combined.loc[combined["method"].astype(str).eq(method)]
        counts = selected.groupby("scenario", observed=True)["seed"].nunique()
        if counts.to_dict() != {"active": 20, "global_null": 20}:
            raise ValueError(f"method has incomplete paired seeds: {method}")
    return combined


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    active = metrics.loc[metrics["scenario"].astype(str).eq("active")]
    null = metrics.loc[metrics["scenario"].astype(str).eq("global_null")]
    summary = (
        active.groupby("method", observed=True, sort=False)
        .agg(
            active_seeds=("seed", "nunique"),
            omnibus_prevalence_adjusted_ap=(
                "omnibus_prevalence_adjusted_ap",
                "mean",
            ),
            omnibus_auprc=("omnibus_auprc", "mean"),
            omnibus_auroc=("omnibus_auroc", "mean"),
            localization_macro_auprc=("localization_macro_auprc", "mean"),
            positive_direction_ap=("positive_direction_ap", "mean"),
            negative_direction_ap=("negative_direction_ap", "mean"),
            direction_accuracy=("direction_accuracy_all_active", "mean"),
            minimum_event_coverage=("event_coverage", "min"),
        )
        .reset_index()
    )
    null_summary = (
        null.groupby("method", observed=True, sort=False)
        .agg(
            null_seeds=("seed", "nunique"),
            global_null_effect_sd=("effect_standard_deviation", "mean"),
            global_null_effect_range=("effect_dynamic_range", "mean"),
            global_null_zero_fraction=("effect_all_zero_fraction", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(null_summary, on="method", validate="one_to_one")
    summary.insert(1, "method_label", summary["method"].map(METHOD_LABELS))
    higher = (
        "omnibus_prevalence_adjusted_ap",
        "omnibus_auprc",
        "omnibus_auroc",
        "localization_macro_auprc",
    )
    for metric in higher:
        summary[f"{metric}_rank"] = summary[metric].rank(
            ascending=False, method="average"
        )
    summary["global_null_effect_sd_rank"] = summary["global_null_effect_sd"].rank(
        ascending=True, method="average"
    )
    return summary.sort_values(
        ["omnibus_prevalence_adjusted_ap_rank", "method_label"],
        ignore_index=True,
    )


def _paired_comparison(
    metrics: pd.DataFrame,
    *,
    comparator: str,
    scenario: str,
    metric: str,
    direction: str,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = metrics.loc[metrics["scenario"].astype(str).eq(scenario)]
    pivot = selected.pivot(index="seed", columns="method", values=metric)
    required = {CANDIDATE_METHOD, comparator}
    if missing := required.difference(pivot.columns):
        raise ValueError(f"paired methods are absent: {sorted(missing)}")
    pair = pivot.loc[:, [CANDIDATE_METHOD, comparator]].dropna()
    if len(pair) != 20:
        raise ValueError(f"paired comparison requires 20 seeds: {comparator}/{metric}")
    raw = pair[CANDIDATE_METHOD].to_numpy(dtype=float) - pair[comparator].to_numpy(
        dtype=float
    )
    if direction == "higher":
        oriented = raw
    elif direction == "lower":
        oriented = -raw
    else:
        raise ValueError("direction must be higher or lower")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(oriented, size=(replicates, len(oriented)), replace=True)
    low, high = np.quantile(sampled.mean(axis=1), (0.025, 0.975))
    detail = pd.DataFrame(
        {
            "scenario": scenario,
            "metric": metric,
            "direction": direction,
            "comparator": comparator,
            "seed": pair.index.astype(int),
            "candidate": pair[CANDIDATE_METHOD].to_numpy(dtype=float),
            "comparator_value": pair[comparator].to_numpy(dtype=float),
            "raw_candidate_minus_comparator": raw,
            "oriented_improvement": oriented,
        }
    )
    return (
        {
            "scenario": scenario,
            "metric": metric,
            "direction": direction,
            "comparator": comparator,
            "paired_seeds": len(pair),
            "candidate_mean": float(pair[CANDIDATE_METHOD].mean()),
            "comparator_mean": float(pair[comparator].mean()),
            "raw_candidate_minus_comparator": float(raw.mean()),
            "oriented_mean_improvement": float(oriented.mean()),
            "oriented_ci_low": float(low),
            "oriented_ci_high": float(high),
            "wins": int((oriented > 0.0).sum()),
            "ties": int((oriented == 0.0).sum()),
            "losses": int((oriented < 0.0).sum()),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
        },
        detail,
    )


def _comparisons(
    metrics: pd.DataFrame, *, replicates: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    index = 0
    for comparator in COMPARATORS:
        for scenario, metric, direction in METRICS:
            record, detail = _paired_comparison(
                metrics,
                comparator=comparator,
                scenario=scenario,
                metric=metric,
                direction=direction,
                replicates=replicates,
                seed=seed + index,
            )
            records.append(record)
            details.append(detail)
            index += 1
    return pd.DataFrame.from_records(records), pd.concat(details, ignore_index=True)


def _display(value: object) -> str:
    number = float(value)
    return "NE" if not math.isfinite(number) else f"{number:.4f}"


def _report(summary: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    lines = [
        "# RC12 same-seed external comparison",
        "",
        "The frozen RC12 unsigned detection head is evaluated on the same 20 active",
        "and 20 matched-global-null seeds as the external methods. All rows use the",
        "same H-common resource, event truth, and fixture commit. The RC12 head is a",
        "benchmark-only ranking head; it is not the public communication-strength",
        "default and it does not emit formal p/q values.",
        "",
        "| AP rank | Method | Adj. AP | AUPRC | AUROC | Localization AP | Null SD |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                (
                    _display(row.omnibus_prevalence_adjusted_ap_rank),
                    str(row.method_label),
                    _display(row.omnibus_prevalence_adjusted_ap),
                    _display(row.omnibus_auprc),
                    _display(row.omnibus_auroc),
                    _display(row.localization_macro_auprc),
                    _display(row.global_null_effect_sd),
                )
            )
            + " |"
        )
    scseq = comparisons.loc[comparisons["comparator"].eq("scseqcommdiff")]
    lines.extend(
        (
            "",
            "## RC12 versus scSeqCommDiff",
            "",
            "Positive oriented differences favor RC12. Intervals are paired-seed",
            "percentile bootstrap intervals and are descriptive, not a release gate.",
            "",
            "| Metric | RC12 | scSeqCommDiff | Raw delta | 95% oriented CI | W/T/L |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for row in scseq.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {_display(row.candidate_mean)} | "
            f"{_display(row.comparator_mean)} | "
            f"{_display(row.raw_candidate_minus_comparator)} | "
            f"[{_display(row.oriented_ci_low)}, {_display(row.oriented_ci_high)}] | "
            f"{row.wins}/{row.ties}/{row.losses} |"
        )
    lines.extend(
        (
            "",
            "The unsigned RC12 score is used only for detection and localization.",
            "Its direction columns are diagnostics; the canonical signed head remains",
            "the direction estimand. Native-scale effect RMSE is not ranked across",
            "methods because the score scales differ.",
        )
    )
    return "\n".join(lines) + "\n"


def compare(
    *,
    candidate_root: Path,
    external_root: Path,
    output_dir: Path,
    repo_root: Path,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 20_260_730,
) -> dict[str, Any]:
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    started = time.perf_counter()
    candidate_root = candidate_root.resolve()
    external_root = external_root.resolve()
    candidate_manifest, external_manifest = _validate_alignment(
        candidate_root, external_root
    )
    candidate_metrics = _bound_table(
        candidate_root, candidate_manifest, "multigroup_metrics.tsv"
    )
    external_metrics = _bound_table(
        external_root, external_manifest, "multigroup_metrics.tsv"
    )
    metrics = _combined_metrics(candidate_metrics, external_metrics)
    summary = _summarize(metrics)
    comparisons, paired = _comparisons(
        metrics, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    code = git_metadata(repo_root)
    if code.get("dirty") is not False:
        raise ValueError("comparison must be published from a clean worktree")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    published = False
    try:
        tables = {
            "method_summary.tsv": summary,
            "paired_comparisons.tsv": comparisons,
            "paired_seed_metrics.tsv": paired,
        }
        outputs: dict[str, dict[str, Any]] = {}
        for filename, table in tables.items():
            path = staged / filename
            table.to_csv(path, sep="\t", index=False)
            outputs[filename] = {
                "rows": len(table),
                "sha256": sha256_file(path),
            }
        report_path = staged / "REPORT.md"
        report_path.write_text(_report(summary, comparisons), encoding="utf-8")
        outputs[report_path.name] = {
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "code": code,
            "candidate_evaluation": {
                "directory": str(candidate_root),
                "manifest_sha256": sha256_file(candidate_root / "manifest.json"),
            },
            "external_evaluation": {
                "directory": str(external_root),
                "manifest_sha256": sha256_file(external_root / "manifest.json"),
            },
            "alignment": {
                field: external_manifest[field]
                for field in (
                    "fixture_manifest_sha256",
                    "truth_sha256",
                    "frozen_code_commit",
                )
            },
            "candidate_method": CANDIDATE_METHOD,
            "candidate_status": "benchmark_only_unreleased",
            "comparators": list(COMPARATORS),
            "paired_seeds_per_scenario": 20,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "outputs": outputs,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (staged / "manifest.json").write_text(
            json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        staged.replace(output_dir)
        published = True
        return manifest
    finally:
        if not published:
            shutil.rmtree(staged, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--external-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_730)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = compare(
        candidate_root=args.candidate_evaluation,
        external_root=args.external_evaluation,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
