"""Run and atomically publish deterministic G1.5 simulation campaigns."""

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

from benchmarks.adapters.common import (
    canonical_digest,
    python_environment,
    sha256_file,
    write_json,
)
from benchmarks.evaluate_mechanism_specificity import run_evaluation
from benchmarks.metrics.mechanism_specificity import (
    ComponentTruthMatrix,
    component_truth_from_mapping,
    specification_from_config,
)
from benchmarks.simulation.mechanism_specificity import (
    EVIDENCE_COLUMNS,
    HOLDOUT_PHASE,
    PHASES,
    GeneratedMechanismEvidence,
    frozen_design_manifest,
    generate_mechanism_specificity_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "benchmarks/configs/mechanism_specificity_v2.json"
DEFAULT_TRUTH = REPO_ROOT / "benchmarks/truth/component_truth_matrix.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmark_work/g1_5_sample_keyed_development_v2"
MANIFEST_SCHEMA_VERSION = "crychic-g1.5-generation-manifest-v2"


def _require_publishable_live_phase(phase: str) -> None:
    if phase == HOLDOUT_PHASE:
        raise ValueError(
            "the historical G1.5 v2 holdout has already been inspected; "
            "the sample-keyed generator requires a new preregistered holdout "
            "configuration and seed namespace before publication"
        )


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


def _phase_policy(config: Mapping[str, object], phase: str) -> Mapping[str, object]:
    raw_phases = config.get("evaluation_phases")
    if not isinstance(raw_phases, Mapping):
        raise ValueError("config.evaluation_phases must be an object")
    raw_policy = raw_phases.get(phase)
    if not isinstance(raw_policy, Mapping):
        raise ValueError(f"config has no policy for phase {phase!r}")
    if phase == HOLDOUT_PHASE and bool(raw_policy.get("may_tune_candidate")):
        raise ValueError("independent holdout configuration must forbid tuning")
    return cast(Mapping[str, object], raw_policy)


def _validate_contract_columns(
    config: Mapping[str, object], evidence: pd.DataFrame
) -> None:
    raw_contract = config.get("evidence_record_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("config.evidence_record_contract must be an object")
    raw_columns = raw_contract.get("required_columns")
    if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, str):
        raise ValueError("config evidence required_columns must be an array")
    configured = tuple(str(value) for value in raw_columns)
    if configured != EVIDENCE_COLUMNS or tuple(evidence.columns) != configured:
        raise ValueError("generated evidence columns differ from the frozen config")


def _component_summary(evidence: pd.DataFrame) -> dict[str, object]:
    columns = (
        "availability_effect",
        "receptor_gate",
        "receiver_program_effect",
        "incremental_downstream_effect",
        "sender_effect",
        "integrated_lr_effect",
    )
    summary: dict[str, object] = {}
    for scenario, group in evidence.groupby("scenario", sort=False, observed=True):
        summary[str(scenario)] = {
            column: {
                "minimum": float(group[column].min()),
                "maximum": float(group[column].max()),
                "mean": float(group[column].mean()),
            }
            for column in columns
        }
    return summary


def _generation_manifest(
    generated: GeneratedMechanismEvidence,
    *,
    evidence_path: Path,
    config_path: Path,
    truth_path: Path,
    truth: ComponentTruthMatrix,
    phase_policy: Mapping[str, object],
) -> dict[str, Any]:
    seed_payload = [lineage.to_dict() for lineage in generated.seed_lineages]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "evaluation_phase": generated.phase,
        "truth_set_id": truth.truth_set_id,
        "known_edge_ids": list(truth.known_edge_ids),
        "scenario_count": 7,
        "seed_count": generated.seed_count,
        "evidence_rows": generated.row_count,
        "evidence_grain": "seed_x_scenario_x_known_edge",
        "seed_namespace": {
            "lineages": seed_payload,
            "derived_seed_digest": canonical_digest(
                {"derived_seeds": [item["derived_seed"] for item in seed_payload]},
                prefix="seed-set",
            ),
        },
        "phase_policy": {
            "may_tune_candidate": bool(phase_policy.get("may_tune_candidate")),
            "tuning_performed": False,
            "holdout_tuning_forbidden": generated.phase == HOLDOUT_PHASE,
        },
        "frozen_design": frozen_design_manifest(),
        "audit": generated.audit_summary(),
        "component_summary": _component_summary(generated.evidence),
        "inputs": {
            "config": {"name": config_path.name, "sha256": sha256_file(config_path)},
            "truth": {"name": truth_path.name, "sha256": sha256_file(truth_path)},
        },
        "outputs": {
            "evidence": {
                "name": evidence_path.name,
                "sha256": sha256_file(evidence_path),
                "rows": generated.row_count,
            }
        },
        "software": python_environment(
            environment_name="crychic-g1.5-deterministic-simulation",
            packages=("crychic", "numpy", "pandas", "pyarrow", "scipy"),
            threads=1,
        ),
        "generator_sources": {
            "core": {
                "name": "mechanism_specificity.py",
                "sha256": sha256_file(
                    Path(__file__).with_name("mechanism_specificity.py")
                ),
            },
            "runner": {"name": Path(__file__).name, "sha256": sha256_file(__file__)},
        },
        "claims": {
            "integrated_values_are_scenario_assigned": False,
            "uses_heldout_receiver_expression": True,
            "default_switch_allowed": False,
            "real_data_accuracy_claim": False,
        },
    }


def publish_generated_campaign(
    generated: GeneratedMechanismEvidence,
    *,
    config_path: str | Path,
    truth_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Atomically publish evidence, provenance, and evaluator outputs."""

    config_file = Path(config_path).resolve()
    truth_file = Path(truth_path).resolve()
    output = Path(output_dir).resolve()
    for path, label in ((config_file, "config"), (truth_file, "truth")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    config = _json_object(config_file)
    truth = component_truth_from_mapping(_yaml_object(truth_file))
    specification = specification_from_config(config, evaluation_phase=generated.phase)
    phase_policy = _phase_policy(config, generated.phase)
    _require_publishable_live_phase(generated.phase)
    if generated.seed_count != specification.expected_seed_count:
        raise ValueError(
            "generated seed count does not equal the frozen phase requirement"
        )
    expected_rows = specification.expected_seed_count * len(truth.known_edge_ids) * 7
    if generated.row_count != expected_rows:
        raise ValueError("generated evidence row count is incomplete")
    _validate_contract_columns(config, generated.evidence)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        evidence_path = staging / "evidence.parquet"
        generated.evidence.to_parquet(evidence_path, index=False)
        manifest = _generation_manifest(
            generated,
            evidence_path=evidence_path,
            config_path=config_file,
            truth_path=truth_file,
            truth=truth,
            phase_policy=phase_policy,
        )
        write_json(staging / "generation_manifest.json", manifest)
        evaluation = run_evaluation(
            evidence_path,
            config_file,
            truth_file,
            staging / "metrics",
            evaluation_phase=generated.phase,
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": str(output),
        "evaluation_phase": generated.phase,
        "seed_count": generated.seed_count,
        "evidence_rows": generated.row_count,
        "overall_status": evaluation["overall_status"],
        "gate_passed_for_supplied_input": evaluation["gate_passed_for_supplied_input"],
        "default_switch_allowed": False,
    }


def run_campaign(
    *,
    phase: str,
    config_path: str | Path = DEFAULT_CONFIG,
    truth_path: str | Path = DEFAULT_TRUTH,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Generate the frozen 50- or 200-seed phase and publish it atomically."""

    if phase not in PHASES:
        raise ValueError(f"unknown G1.5 phase: {phase!r}")
    _require_publishable_live_phase(phase)
    config_file = Path(config_path).resolve()
    truth_file = Path(truth_path).resolve()
    output = (
        (DEFAULT_OUTPUT_ROOT / phase).resolve()
        if output_dir is None
        else Path(output_dir).resolve()
    )
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    config = _json_object(config_file)
    _phase_policy(config, phase)
    specification = specification_from_config(config, evaluation_phase=phase)
    truth = component_truth_from_mapping(_yaml_object(truth_file))
    generated = generate_mechanism_specificity_evidence(
        phase=phase,
        seed_count=specification.expected_seed_count,
        truth=truth,
    )
    return publish_generated_campaign(
        generated,
        config_path=config_file,
        truth_path=truth_file,
        output_dir=output,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate the frozen deterministic G1.5 seed campaign"
        )
    )
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line campaign."""

    args = _parser().parse_args(argv)
    result = run_campaign(
        phase=args.phase,
        config_path=args.config,
        truth_path=args.truth,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
