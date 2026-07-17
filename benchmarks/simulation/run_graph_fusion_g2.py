"""Run and atomically report the preregistered graph-fusion G2 benchmark."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    python_environment,
    sha256_file,
    write_json,
)
from benchmarks.metrics.graph_fusion_g2 import (
    G2GateReport,
    evaluate_graph_fusion_g2_gate,
)
from benchmarks.simulation.graph_fusion_g2 import (
    DEFAULT_CONFIG,
    G2BenchmarkManifest,
    load_g2_manifest,
    manifest_provenance,
    run_g2_simulation,
)

DEFAULT_OUTPUT = Path("benchmark_work/graph_fusion_g2_smoke")
RUN_MANIFEST_SCHEMA = "crychic-g2-graph-fusion-run-manifest-v1"
REPORT_SCHEMA = "crychic-g2-graph-fusion-report-v1"
OUTPUT_FILENAMES = (
    "paired_results.csv",
    "gate.json",
    "report.md",
    "run_manifest.json",
)


def render_g2_report(
    manifest: G2BenchmarkManifest,
    results: pd.DataFrame,
    gate: G2GateReport,
    *,
    profile: str,
    replicates_per_scenario: int,
) -> str:
    """Render a compact report that keeps NE distinct from pass and fail."""

    observed = int(results["status"].astype(str).eq("observed").sum())
    failed = int(len(results) - observed)
    primary = gate.primary_interval
    primary_text = (
        "NE (paired bootstrap is withheld until every scenario cell reaches "
        "the frozen minimum)"
        if primary is None
        else (
            f"estimate={primary.estimate:.4f}, 95% CI "
            f"[{primary.ci_lower:.4f}, {primary.ci_upper:.4f}]"
        )
    )
    control_lines = (
        ["- No topology: NE", "- Wrong topology: NE"]
        if not gate.control_intervals
        else [
            (
                f"- {interval.comparison}: estimate={interval.estimate:.4f}, "
                f"95% CI [{interval.ci_lower:.4f}, {interval.ci_upper:.4f}]"
            )
            for interval in gate.control_intervals
        ]
    )
    return "\n".join(
        [
            "# CRYCHIC Graph-Fusion G2 Decision Report",
            "",
            f"- Report schema: `{REPORT_SCHEMA}`",
            f"- Gate status: **{gate.status}**",
            f"- Reason: `{gate.reason_code}`",
            f"- Profile: `{profile}`",
            f"- Replicates per scenario cell: {replicates_per_scenario}",
            f"- Frozen scenario cells: {len(manifest.scenarios)}",
            f"- Observed method runs: {observed}",
            f"- Failed method runs: {failed}",
            "",
            "## Primary Endpoint",
            "",
            "`driver_family x context` macro-AUPRC; fused and unfused are paired "
            "by scenario seed, then aggregated with equal scenario-cell weight.",
            "",
            f"- Fused minus unfused: {primary_text}",
            (
                "- Required lower confidence bound: "
                f">= {manifest.gate_specification.improvement_margin:.2f}"
            ),
            "",
            "## Topology Sensitivity",
            "",
            *control_lines,
            (
                "- Required noninferiority lower bound: "
                f">= {manifest.gate_specification.noninferiority_margin:.2f}"
            ),
            "",
            "## Interpretation",
            "",
            (
                "This smoke run is not gate-eligible and cannot support a default "
                "switch."
                if gate.status == "NE"
                else (
                    "The preregistered numerical gate passed; graph fusion still "
                    "requires the other release requirements before a default switch."
                    if gate.status == "PASS"
                    else "Graph fusion remains opt-in experimental."
                )
            ),
            "Secondary AUROC, coefficient RMSE, and jump localization are reported "
            "only in `paired_results.csv`; they never replace the primary gate.",
            "",
        ]
    )


def _source_hashes(repo_root: Path, manifest: G2BenchmarkManifest) -> dict[str, str]:
    paths = (
        manifest.source_path,
        repo_root / "benchmarks/metrics/graph_fusion_g2.py",
        repo_root / "benchmarks/simulation/graph_fusion_g2.py",
        repo_root / "benchmarks/simulation/run_graph_fusion_g2.py",
        repo_root / "src/crychic/attribution/graph_fused.py",
    )
    result: dict[str, str] = {}
    for path in paths:
        key = (
            str(path.relative_to(repo_root))
            if path.is_relative_to(repo_root)
            else f"external_config/{path.name}"
        )
        result[key] = sha256_file(path)
    return result


def run_g2_campaign(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
    profile: str | None = None,
    replicates_per_scenario: int | None = None,
) -> Path:
    """Execute one smoke/formal run and atomically publish compact artifacts."""

    manifest = load_g2_manifest(config_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"G2 output directory already exists: {output}")
    selected_profile = manifest.default_profile if profile is None else profile
    if selected_profile not in {"smoke", "formal"}:
        raise ValueError("profile must be 'smoke' or 'formal'")
    default_replicates = (
        manifest.smoke_replicates
        if selected_profile == "smoke"
        else manifest.formal_replicates
    )
    replicates = (
        default_replicates
        if replicates_per_scenario is None
        else replicates_per_scenario
    )
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("replicates_per_scenario must be a positive integer")
    results = run_g2_simulation(
        manifest, replicates_per_scenario=replicates
    )
    gate = evaluate_graph_fusion_g2_gate(results, manifest.gate_specification)
    report = render_g2_report(
        manifest,
        results,
        gate,
        profile=selected_profile,
        replicates_per_scenario=replicates,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    repo_root = Path(__file__).resolve().parents[2]
    try:
        results.to_csv(staging / "paired_results.csv", index=False)
        write_json(staging / "gate.json", gate.to_dict())
        (staging / "report.md").write_text(report, encoding="utf-8")
        records: dict[str, Any] = {
            name: {
                "sha256": sha256_file(staging / name),
                "bytes": (staging / name).stat().st_size,
            }
            for name in OUTPUT_FILENAMES[:-1]
        }
        run_manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "status": "complete",
            "profile": selected_profile,
            "replicates_per_scenario": replicates,
            "gate_status": gate.status,
            "gate_reason_code": gate.reason_code,
            "benchmark": manifest_provenance(manifest),
            "source_sha256": _source_hashes(repo_root, manifest),
            "environment": python_environment(
                environment_name="crychic-g2-graph-fusion",
                packages=("crychic", "numpy", "pandas", "scipy", "scikit-learn"),
                threads=1,
            ),
            "git": git_metadata(repo_root),
            "artifacts": records,
        }
        write_json(staging / "run_manifest.json", run_manifest)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("smoke", "formal"), default=None)
    parser.add_argument("--replicates", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run_g2_campaign(
        config_path=args.config,
        output_dir=args.output,
        profile=args.profile,
        replicates_per_scenario=args.replicates,
    )
    print(output)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "OUTPUT_FILENAMES",
    "REPORT_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "build_parser",
    "main",
    "render_g2_report",
    "run_g2_campaign",
]
