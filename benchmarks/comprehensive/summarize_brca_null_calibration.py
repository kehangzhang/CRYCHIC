"""Summarize subject-level calibration on randomized BRCA global-null fixtures."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, t

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-brca-randomized-null-calibration-summary-v1"


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


def _json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _bound_output(evaluation_dir: Path, name: str) -> Path:
    manifest_path = evaluation_dir / "manifest.json"
    manifest = _json_object(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"evaluation is not complete: {manifest_path}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(name), dict):
        raise ValueError(f"evaluation manifest does not bind {name}: {manifest_path}")
    path = evaluation_dir / name
    if not path.is_file() or outputs[name].get("sha256") != sha256_file(path):
        raise ValueError(f"evaluation output checksum mismatch: {path}")
    return path


def _replicate_rows(
    evaluation_dirs: Sequence[Path],
    *,
    method: str,
    view_label: str,
    engine: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    parts: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []
    for directory in evaluation_dirs:
        resolved = directory.expanduser().resolve()
        manifest_path = resolved / "manifest.json"
        effects_path = _bound_output(resolved, "event_effects.tsv.gz")
        metrics_path = _bound_output(resolved, "metrics.tsv")
        effects = pd.read_csv(effects_path, sep="\t")
        selected = effects.loc[
            effects["base_method_id"].astype(str).eq(method)
            & effects["view_label"].astype(str).eq(view_label)
            & effects["differential_engine"].astype(str).eq(engine)
        ].copy()
        if selected.empty:
            raise ValueError(f"requested null-calibration view is absent: {resolved}")
        if not selected["formal_did_p_value"].astype(bool).all():
            raise ValueError("null calibration requires a formal DID inference track")
        if not selected["truth_label"].eq(0).all() or not selected[
            "truth_direction"
        ].eq(0).all():
            raise ValueError("randomized-null calibration contains non-null truth")
        metrics = pd.read_csv(metrics_path, sep="\t")
        metric_selected = metrics.loc[
            metrics["base_method_id"].astype(str).eq(method)
            & metrics["view_label"].astype(str).eq(view_label)
            & metrics["differential_engine"].astype(str).eq(engine)
        ]
        expected_datasets = set(selected["dataset_id"].astype(str))
        if (
            set(metric_selected["dataset_id"].astype(str)) != expected_datasets
            or metric_selected.duplicated("dataset_id").any()
        ):
            raise ValueError("metrics and event effects disagree on null replicates")

        for dataset_id, group in selected.groupby(
            "dataset_id", observed=True, sort=True
        ):
            if group["method_variant_id"].nunique() != 1:
                raise ValueError("one null replicate maps to multiple method variants")
            observed = group["status"].astype(str).eq("observed")
            p_value = pd.to_numeric(group["p_value"], errors="coerce")
            q_value = pd.to_numeric(group["q_value"], errors="coerce")
            ci_low = pd.to_numeric(group["ci_low_95"], errors="coerce")
            ci_high = pd.to_numeric(group["ci_high_95"], errors="coerce")
            formal = observed & p_value.notna() & np.isfinite(p_value)
            interval = observed & ci_low.notna() & ci_high.notna()
            if not formal.any() or not interval.any():
                raise ValueError(
                    f"null replicate has no formal inference: {dataset_id}"
                )
            discoveries = q_value.notna() & q_value.lt(0.05)
            seed_values = pd.to_numeric(group["seed"], errors="raise").unique()
            if len(seed_values) != 1:
                raise ValueError("one null dataset must contain exactly one seed")
            parts.append(
                pd.DataFrame.from_records(
                    [
                        {
                            "campaign_id": resolved.name,
                            "dataset_id": str(dataset_id),
                            "seed": int(seed_values[0]),
                            "n_events": len(group),
                            "n_observed": int(observed.sum()),
                            "n_formal_tests": int(formal.sum()),
                            "formal_test_fraction": float(formal.mean()),
                            "unadjusted_rejections_p_lt_005": int(
                                (formal & p_value.lt(0.05)).sum()
                            ),
                            "type_i_005": float(p_value.loc[formal].lt(0.05).mean()),
                            "ci_95_coverage": float(
                                (
                                    ci_low.loc[interval].le(0.0)
                                    & ci_high.loc[interval].ge(0.0)
                                ).mean()
                            ),
                            "bh_discoveries_q_lt_005": int(discoveries.sum()),
                            "bh_any_false_discovery": int(discoveries.any()),
                        }
                    ]
                )
            )
        sources.append(
            {
                "evaluation_dir": resolved.name,
                "manifest_sha256": sha256_file(manifest_path),
                "effects_sha256": sha256_file(effects_path),
                "metrics_sha256": sha256_file(metrics_path),
            }
        )
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["campaign_id", "dataset_id"]).any():
        raise ValueError("duplicate null replicate identifiers")
    if result["seed"].duplicated().any():
        raise ValueError("null calibration seeds must be unique across campaigns")
    return result.sort_values("seed", kind="stable", ignore_index=True), sources


def _mean_interval(values: pd.Series) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("calibration intervals require at least two finite replicates")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    two_sided = float(t.ppf(0.975, df=len(array) - 1)) * standard_error
    one_sided = float(t.ppf(0.95, df=len(array) - 1)) * standard_error
    return {
        "n": len(array),
        "mean": mean,
        "ci_low_95": mean - two_sided,
        "ci_high_95": mean + two_sided,
        "one_sided_low_95": mean - one_sided,
        "one_sided_high_95": mean + one_sided,
    }


def _binomial_upper(successes: int, trials: int, *, confidence: float = 0.95) -> float:
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("invalid binomial counts")
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def summarize(
    evaluation_dirs: Sequence[Path],
    output_dir: Path,
    *,
    method: str = "crychic",
    view_label: str = "mechanistic_sender_lr_score",
    engine: str = "native_raw_mean",
    minimum_replicates: int = 30,
    maximum_type_i_upper: float = 0.075,
    minimum_ci_coverage_lower: float = 0.925,
    maximum_bh_fwer_upper: float = 0.10,
    minimum_formal_test_fraction: float = 0.95,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a checksum-bound calibration report with replicate-level gates."""

    if not evaluation_dirs:
        raise ValueError("at least one evaluation directory is required")
    if minimum_replicates < 2:
        raise ValueError("minimum_replicates must be at least two")
    for value in (
        maximum_type_i_upper,
        minimum_ci_coverage_lower,
        maximum_bh_fwer_upper,
        minimum_formal_test_fraction,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("calibration thresholds must lie in [0, 1]")

    replicates, sources = _replicate_rows(
        evaluation_dirs,
        method=method,
        view_label=view_label,
        engine=engine,
    )
    type_i = _mean_interval(replicates["type_i_005"])
    coverage = _mean_interval(replicates["ci_95_coverage"])
    formal = _mean_interval(replicates["formal_test_fraction"])
    fwer_events = int(replicates["bh_any_false_discovery"].sum())
    fwer_upper = _binomial_upper(fwer_events, len(replicates))
    checks = {
        "minimum_replicates": len(replicates) >= minimum_replicates,
        "formal_test_coverage": formal["one_sided_low_95"]
        >= minimum_formal_test_fraction,
        "type_i_not_inflated": type_i["one_sided_high_95"]
        <= maximum_type_i_upper,
        "ci_coverage_adequate": coverage["one_sided_low_95"]
        >= minimum_ci_coverage_lower,
        "bh_global_null_fwer": fwer_upper <= maximum_bh_fwer_upper,
    }
    gate = {
        "name": "randomized_global_null_subject_level_calibration",
        "status": "PASS" if all(checks.values()) else "REJECT",
        "checks": checks,
        "thresholds": {
            "minimum_replicates": minimum_replicates,
            "minimum_formal_test_fraction": minimum_formal_test_fraction,
            "maximum_type_i_one_sided_upper_95": maximum_type_i_upper,
            "minimum_ci_coverage_one_sided_lower_95": minimum_ci_coverage_lower,
            "maximum_bh_fwer_clopper_pearson_upper_95": maximum_bh_fwer_upper,
        },
        "estimates": {
            "formal_test_fraction": formal,
            "type_i_005": type_i,
            "ci_95_coverage": coverage,
            "bh_fwer_events": fwer_events,
            "bh_fwer_rate": fwer_events / len(replicates),
            "bh_fwer_one_sided_upper_95": fwer_upper,
        },
    }

    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        replicate_path = staged / "replicate_calibration.tsv"
        replicates.to_csv(replicate_path, sep="\t", index=False)
        gate_path = staged / "gate.json"
        _write_json(gate_path, gate)
        report_path = staged / "REPORT.md"
        report_path.write_text(
            "\n".join(
                [
                    "# BRCA randomized global-null calibration",
                    "",
                    f"Gate: **{gate['status']}**",
                    "",
                    f"- Replicates: {len(replicates)}",
                    f"- Formal-test fraction: {formal['mean']:.4f} "
                    f"(one-sided lower 95% {formal['one_sided_low_95']:.4f})",
                    f"- Unadjusted type-I at 0.05: {type_i['mean']:.4f} "
                    f"(one-sided upper 95% {type_i['one_sided_high_95']:.4f})",
                    f"- Welch 95% CI coverage: {coverage['mean']:.4f} "
                    f"(one-sided lower 95% {coverage['one_sided_low_95']:.4f})",
                    f"- BH any-false-discovery replicates: {fwer_events}/"
                    f"{len(replicates)} (upper 95% {fwer_upper:.4f})",
                    "",
                    "Inference unit is the biological subject. Each subject "
                    "contributes "
                    "one paired On-minus-Pre delta; E/NE labels are randomized across "
                    "subjects while group sizes are preserved. Missing events are "
                    "never "
                    "zero-imputed.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "method": method,
            "view_label": view_label,
            "differential_engine": engine,
            "inference_unit": "biological_subject",
            "null_design": (
                "between_subject_expansion_label_randomization_preserve_sizes"
            ),
            "sources": sources,
            "gate": gate,
            "outputs": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (replicate_path, gate_path, report_path)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", default="crychic")
    parser.add_argument("--view-label", default="mechanistic_sender_lr_score")
    parser.add_argument("--engine", default="native_raw_mean")
    parser.add_argument("--minimum-replicates", type=int, default=30)
    parser.add_argument("--maximum-type-i-upper", type=float, default=0.075)
    parser.add_argument("--minimum-ci-coverage-lower", type=float, default=0.925)
    parser.add_argument("--maximum-bh-fwer-upper", type=float, default=0.10)
    parser.add_argument("--minimum-formal-test-fraction", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = summarize(
        args.evaluation_dirs,
        args.output_dir,
        method=args.method,
        view_label=args.view_label,
        engine=args.engine,
        minimum_replicates=args.minimum_replicates,
        maximum_type_i_upper=args.maximum_type_i_upper,
        minimum_ci_coverage_lower=args.minimum_ci_coverage_lower,
        maximum_bh_fwer_upper=args.maximum_bh_fwer_upper,
        minimum_formal_test_fraction=args.minimum_formal_test_fraction,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": manifest["status"], "gate": manifest["gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
