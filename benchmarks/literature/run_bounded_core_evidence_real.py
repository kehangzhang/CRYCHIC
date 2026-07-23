"""Run the frozen RC11 RC9-plus-current evidence head on Kuppe or MS."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.literature.bounded_core_evidence import (
    BoundedCoreEvidencePolicy,
    bounded_core_evidence_rankings,
)
from benchmarks.literature.component_swap_benchmark import _evaluate
from benchmarks.literature.run_bounded_program_blend_real import (
    METHOD_FALLBACK as RC9_METHOD,
)

SCHEMA_VERSION = "crychic-bounded-core-evidence-real-v1"
CONFIG_SCHEMA_VERSION = "crychic-bounded-core-evidence-config-v1"
RC9_SCHEMA_VERSION = "crychic-bounded-program-blend-real-v1"
CURRENT_SCHEMAS = frozenset(
    {
        "crychic-kuppe-ctrl-iz-crossfit-run-v3",
        "crychic-ms-ctrl-ca-crossfit-run-v2",
        "crychic-ms-ctrl-ca-crossfit-run-v3",
    }
)
CURRENT_RANKING_FILENAME = "mechanistic_condition_cell_pair_rankings.tsv"


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


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
    run_dir: Path,
    manifest: Mapping[str, Any],
    filename: str,
) -> Path:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"run manifest has no outputs: {run_dir}")
    record = outputs.get(filename)
    if not isinstance(record, Mapping):
        raise ValueError(f"run manifest does not bind {filename}: {run_dir}")
    path = run_dir / str(record.get("filename", filename))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"bound output checksum mismatch: {path}")
    return path


def _load_config(path: Path) -> tuple[dict[str, Any], BoundedCoreEvidencePolicy]:
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("RC11 configuration schema is unsupported")
    release = config.get("release")
    if not isinstance(release, Mapping) or any(
        (
            release.get("formal_inference_allowed") is not False,
            release.get("canonical_score_replaced") is not False,
            release.get("algorithm_backbone_modified") is not False,
            release.get("ranking_head_only") is not True,
        )
    ):
        raise ValueError("RC11 release boundary was altered")
    values = config.get("policy")
    if not isinstance(values, Mapping) or values.get("base") != RC9_METHOD:
        raise ValueError("RC11 policy does not bind the frozen RC9 base")
    policy = BoundedCoreEvidencePolicy(
        expert_fraction=float(values["expert_fraction"]),
        missing_base_penalty=float(values["missing_base_penalty"]),
    )
    return config, policy


def _load_rc9(
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, object]]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != RC9_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("formal_release_allowed") is not False
    ):
        raise ValueError("RC9 source is not a complete unreleased ranking run")
    path = _bound_output(run_dir, manifest, "condition_cell_pair_rankings.tsv")
    rankings = pd.read_csv(path, sep="\t")
    base = rankings.loc[rankings["method"].astype(str).eq(RC9_METHOD)].copy()
    if base.empty:
        raise ValueError("RC9 fallback ranking is absent")
    return (
        base,
        rankings,
        manifest,
        {
            "manifest": _input_record(manifest_path),
            "rankings": _input_record(path),
        },
    )


def _load_current(
    run_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") not in CURRENT_SCHEMAS
        or manifest.get("status") != "complete"
    ):
        raise ValueError("current-core source is not a complete supported run")
    path = _bound_output(run_dir, manifest, CURRENT_RANKING_FILENAME)
    ranking = pd.read_csv(path, sep="\t")
    if ranking.empty:
        raise ValueError("current mechanistic ranking is empty")
    if (
        "formal_inference_allowed" in ranking
        and ranking["formal_inference_allowed"].fillna(True).astype(bool).any()
    ):
        raise ValueError("current exploratory ranking emitted formal inference")
    return (
        ranking,
        manifest,
        {
            "manifest": _input_record(manifest_path),
            "rankings": _input_record(path),
        },
    )


def _source_expected_record(rc9_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = rc9_manifest.get("inputs")
    rc3 = inputs.get("rc3") if isinstance(inputs, Mapping) else None
    expected = rc3.get("expected_sets") if isinstance(rc3, Mapping) else None
    if not isinstance(expected, Mapping):
        raise ValueError("RC9 source does not bind spatial expected sets")
    return expected


def _summary_row(summary: pd.DataFrame, method: str) -> pd.Series:
    selected = summary.loc[summary["method"].astype(str).eq(method)]
    if len(selected) != 1:
        raise ValueError(f"expected one DES summary row for {method}: {len(selected)}")
    return selected.iloc[0]


def _acceptance(
    summary: pd.DataFrame,
    *,
    dataset: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    method = str(config["method"])
    candidate = _summary_row(summary, method)
    rc9 = _summary_row(summary, RC9_METHOD)
    scseq = _summary_row(summary, "R_scseq_native")
    selection = cast(Mapping[str, Any], config["selection"])
    gate = cast(Mapping[str, Any], config["validation_gate"])
    role = (
        "development"
        if dataset == str(selection["development_dataset"])
        else "validation"
        if dataset == str(selection["validation_dataset"])
        else "unregistered"
    )
    result: dict[str, Any] = {
        "dataset_role": role,
        "candidate_count": int(candidate["count"]),
        "candidate_median": float(candidate["median"]),
        "candidate_mean": float(candidate["mean"]),
        "rc9_median": float(rc9["median"]),
        "rc9_mean": float(rc9["mean"]),
        "scseqcommdiff_median": float(scseq["median"]),
        "scseqcommdiff_mean": float(scseq["mean"]),
        "median_delta_vs_rc9": float(candidate["median"] - rc9["median"]),
        "mean_delta_vs_rc9": float(candidate["mean"] - rc9["mean"]),
        "median_delta_vs_scseqcommdiff": float(candidate["median"] - scseq["median"]),
        "independent_validation": role == "validation",
    }
    if role == "validation":
        result["coverage_pass"] = int(candidate["count"]) == int(
            gate["required_DES_strata"]
        )
        result["median_vs_rc9_pass"] = result["median_delta_vs_rc9"] > float(
            gate["minimum_median_delta_vs_rc9"]
        )
        result["mean_vs_rc9_pass"] = result["mean_delta_vs_rc9"] > float(
            gate["minimum_mean_delta_vs_rc9"]
        )
        result["median_vs_scseqcommdiff_pass"] = result[
            "median_delta_vs_scseqcommdiff"
        ] > float(gate["minimum_median_delta_vs_scseqcommdiff"])
        result["status"] = (
            "ACCEPT"
            if all(
                bool(result[key])
                for key in (
                    "coverage_pass",
                    "median_vs_rc9_pass",
                    "mean_vs_rc9_pass",
                    "median_vs_scseqcommdiff_pass",
                )
            )
            else "REJECT"
        )
    else:
        result["status"] = "DEVELOPMENT_ONLY_NOT_A_VALIDATION_GATE"
    return result


def run(
    *,
    dataset: str,
    rc9_run: Path,
    current_run: Path,
    expected_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create and evaluate one checksum-bound RC11 ranking result."""

    started = time.perf_counter()
    if dataset not in {"kuppe", "ms"}:
        raise ValueError("dataset must be kuppe or ms")
    config, policy = _load_config(config_path.resolve())
    base, rc9_rankings, rc9_manifest, rc9_provenance = _load_rc9(rc9_run.resolve())
    current, current_manifest, current_provenance = _load_current(current_run.resolve())
    if rc9_manifest.get("dataset") != dataset:
        raise ValueError("RC9 source dataset does not match requested dataset")
    expected_record = _source_expected_record(rc9_manifest)
    expected_path = expected_path.resolve()
    if not expected_path.is_file() or sha256_file(expected_path) != expected_record.get(
        "sha256"
    ):
        raise ValueError("spatial expected sets differ from the RC9 source")
    if set(base["condition"].astype(str)) != set(current["condition"].astype(str)):
        raise ValueError("RC9 and current-core conditions disagree")

    candidate, diagnostics = bounded_core_evidence_rankings(
        base, current, policy=policy
    )
    base_dataset_id = str(base["dataset"].iloc[0])
    current = current.copy()
    current["dataset"] = base_dataset_id
    candidate.insert(0, "dataset", base_dataset_id)
    candidate.insert(1, "method", str(config["method"]))
    candidate.insert(2, "method_version", str(config["method_version"]))
    candidate.insert(3, "resource", str(base["resource"].iloc[0]))
    candidate.insert(
        4,
        "ranking_semantics",
        (
            "rc9_base_with_within_condition_quantile_matched_current_"
            "mechanistic_bounded_evidence;benchmark_only"
        ),
    )
    candidate["formal_inference_allowed"] = False
    combined = pd.concat(
        [rc9_rankings, current, candidate], ignore_index=True, sort=False
    )
    expected = pd.read_csv(expected_path, sep="\t")
    scores, coverage, summary = _evaluate(combined, expected, dataset=dataset)
    acceptance = _acceptance(summary, dataset=dataset, config=config)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    published = False
    try:
        tables = {
            "candidate_condition_cell_pair_rankings.tsv": candidate,
            "blend_diagnostics.tsv": diagnostics,
            "spatial_des_scores.tsv": scores,
            "spatial_des_coverage.tsv": coverage,
            "spatial_des_summary.tsv": summary,
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "dataset": dataset,
            "method": str(config["method"]),
            "method_version": str(config["method_version"]),
            "policy": policy.to_dict(),
            "acceptance": acceptance,
            "formal_inference_allowed": False,
            "canonical_score_replaced": False,
            "algorithm_backbone_modified": False,
            "ranking_head_modified": True,
            "selection_bias": cast(Mapping[str, Any], config["selection"])[
                "selection_bias"
            ],
            "inputs": {
                "config": _input_record(config_path.resolve()),
                "rc9": rc9_provenance,
                "current_core": current_provenance,
                "expected_sets": _input_record(expected_path),
            },
            "source_code": {
                "rc9": rc9_manifest.get("code"),
                "current_core": current_manifest.get("code"),
                "runner": git_metadata(Path(__file__).resolve().parents[2]),
            },
            "performance": {"elapsed_seconds": time.perf_counter() - started},
            "limitations": [
                "kuppe_was_used_for_candidate_selection",
                "ms_is_the_only_independent_real_cohort_validation",
                "spatial_colocalization_is_an_indirect_proxy",
                "ranking_head_has_no_formal_p_or_q_values",
                "current_mechanistic_expert_is_exploratory",
            ],
            "outputs": {
                filename: _output_record(staged / filename, table)
                for filename, table in tables.items()
            },
        }
        _write_json(staged / "manifest.json", manifest)
        os.replace(staged, output_dir)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kuppe", "ms"), required=True)
    parser.add_argument("--rc9-run", required=True, type=Path)
    parser.add_argument("--current-run", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(
        dataset=args.dataset,
        rc9_run=args.rc9_run,
        current_run=args.current_run,
        expected_path=args.expected,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
