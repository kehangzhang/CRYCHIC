"""CLI for an atomic, auditable G1.5 evidence-table evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml  # type: ignore[import-untyped]

from benchmarks.adapters.common import json_safe, sha256_file
from benchmarks.metrics.mechanism_specificity import (
    OUTPUT_COLUMNS,
    SCHEMA_VERSION,
    component_truth_from_mapping,
    evaluate_mechanism_specificity,
    specification_from_config,
)

OUTPUT_SCHEMA_VERSION = "crychic-mechanism-specificity-evaluation-v1"


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return cast(dict[str, object], value)


def _yaml_object(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be an object: {path}")
    return cast(dict[str, object], value)


def _read_evidence(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("evidence input must be CSV or Parquet")


def _strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def run_evaluation(
    evidence_path: str | Path,
    config_path: str | Path,
    truth_path: str | Path,
    output_dir: str | Path,
    *,
    evaluation_phase: str,
) -> dict[str, object]:
    """Evaluate one supplied evidence table and atomically install outputs."""

    evidence = Path(evidence_path).resolve()
    config_file = Path(config_path).resolve()
    truth_file = Path(truth_path).resolve()
    output = Path(output_dir).resolve()
    for path, label in (
        (evidence, "evidence"),
        (config_file, "config"),
        (truth_file, "truth"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    config = _json_object(config_file)
    truth_mapping = _yaml_object(truth_file)
    specification = specification_from_config(config, evaluation_phase=evaluation_phase)
    truth = component_truth_from_mapping(truth_mapping)
    truth_contract = config.get("component_truth")
    if not isinstance(truth_contract, Mapping):
        raise ValueError("config.component_truth must be an object")
    expected_truth_id = str(truth_contract.get("truth_set_id", ""))
    expected_truth_sha256 = str(truth_contract.get("sha256", ""))
    observed_truth_sha256 = sha256_file(truth_file)
    if expected_truth_id != truth.truth_set_id:
        raise ValueError("truth_set_id does not match the frozen configuration")
    if expected_truth_sha256 != observed_truth_sha256:
        raise ValueError("truth SHA256 does not match the frozen configuration")
    records = _read_evidence(evidence)
    metrics = evaluate_mechanism_specificity(records, specification, truth)
    metrics = metrics.loc[:, list(OUTPUT_COLUMNS)]
    overall_rows = metrics.loc[
        metrics["metric"].eq("g1_5_mechanism_specificity_gate")
        & metrics["aggregation"].eq("overall")
    ]
    if len(overall_rows) != 1:
        raise RuntimeError("G1.5 evaluator did not emit exactly one overall gate row")
    overall = overall_rows.iloc[0]
    evaluation_completed = bool(overall["status"] == "observed")
    gate_passed = bool(overall["gate_passed"]) if evaluation_completed else None
    payload: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "gate_schema_version": SCHEMA_VERSION,
        "evaluation_phase": evaluation_phase,
        "truth_set_id": truth.truth_set_id,
        "known_edge_ids": list(truth.known_edge_ids),
        "expected_seed_count": specification.expected_seed_count,
        "evaluation_completed_for_supplied_input": evaluation_completed,
        "gate_passed_for_supplied_input": gate_passed,
        "overall_status": str(overall["status"]),
        "overall_reason_code": (
            None if pd.isna(overall["reason_code"]) else str(overall["reason_code"])
        ),
        "default_switch_allowed": False,
        "default_policy": "g1_5_alone_never_switches_default",
        "does_not_claim_unsupplied_seed_campaigns": True,
        "inputs": {
            "evidence": {"name": evidence.name, "sha256": sha256_file(evidence)},
            "config": {"name": config_file.name, "sha256": sha256_file(config_file)},
            "truth": {"name": truth_file.name, "sha256": sha256_file(truth_file)},
        },
        "metrics_rows": len(metrics),
        "records": metrics.to_dict(orient="records"),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        metrics.to_csv(
            staging / "metrics.csv",
            index=False,
            lineterminator="\n",
            na_rep="",
        )
        _strict_json(staging / "metrics.json", payload)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate supplied multi-edge G1.5 component evidence"
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("development", "independent_holdout"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line evaluator."""

    args = _parser().parse_args(argv)
    payload = run_evaluation(
        args.evidence,
        args.config,
        args.truth,
        args.output_dir,
        evaluation_phase=args.phase,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "overall_status": payload["overall_status"],
                "gate_passed_for_supplied_input": payload[
                    "gate_passed_for_supplied_input"
                ],
                "default_switch_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
