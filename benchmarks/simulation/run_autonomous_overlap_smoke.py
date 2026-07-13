"""Small deterministic smoke benchmark for autonomous-program projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from crychic.resources import GeneNamespace, Species
from crychic.response import build_receiver_autonomous_program_resource
from crychic.scoring import (
    DownstreamRowManifest,
    apply_incremental_downstream_functional,
    fit_incremental_downstream_functional,
)
from crychic.scoring.downstream import (
    IncrementalDownstreamApplication,
    IncrementalDownstreamFunctional,
)

SCHEMA_VERSION = "crychic-autonomous-overlap-smoke-v1"
FEATURE_IDS = ("lr_target", "generic_anchor", "background")
FAMILY_IDS = ("lr_family",)
GENERIC_PROGRAM = np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0)
AUTONOMOUS_SMOKE_SOURCE_PATHS = (
    "benchmarks/simulation/run_autonomous_overlap_smoke.py",
    "src/crychic/attribution/__init__.py",
    "src/crychic/attribution/contracts.py",
    "src/crychic/attribution/solver.py",
    "src/crychic/core/__init__.py",
    "src/crychic/core/errors.py",
    "src/crychic/core/ids.py",
    "src/crychic/resources/__init__.py",
    "src/crychic/resources/contracts.py",
    "src/crychic/response/__init__.py",
    "src/crychic/response/autonomous.py",
    "src/crychic/scoring/__init__.py",
    "src/crychic/scoring/contracts.py",
    "src/crychic/scoring/downstream.py",
)


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def autonomous_smoke_source_sha256() -> dict[str, str]:
    """Hash every repository source file directly used by this smoke."""

    repository_root = Path(__file__).resolve().parents[2]
    return {
        relative_path: _source_sha256(repository_root / relative_path)
        for relative_path in AUTONOMOUS_SMOKE_SOURCE_PATHS
    }


def canonical_payload_sha256(payload: dict[str, object]) -> str:
    """Hash the semantic JSON payload independently of pretty formatting."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _manifest(subjects: tuple[str, ...], *, prefix: str) -> DownstreamRowManifest:
    return DownstreamRowManifest(
        sample_ids=tuple(
            f"{prefix}:{subject}:{context}"
            for context in ("reference", "target")
            for subject in subjects
        ),
        subject_ids=subjects + subjects,
        context_ids=tuple(["reference"] * len(subjects) + ["target"] * len(subjects)),
    )


def _response(
    subjects: tuple[str, ...],
    *,
    generic_effect: float,
    unique_lr_effect: float,
    baseline_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_subjects = len(subjects)
    baselines = np.linspace(-1.0, 1.0, n_subjects) * baseline_scale
    reference: np.ndarray = np.zeros((n_subjects, len(FEATURE_IDS)), dtype=np.float64)
    reference[:, 0] = baselines
    reference[:, 1] = baselines
    reference[:, 2] = np.linspace(-0.12, 0.12, n_subjects)
    target = reference + generic_effect * GENERIC_PROGRAM
    target[:, 0] += unique_lr_effect
    target[:, 2] += np.linspace(0.04, 0.16, n_subjects)
    matrix = np.vstack([reference, target])
    regressor = np.concatenate([-np.ones(n_subjects), np.ones(n_subjects)])
    return matrix, regressor


def _fit(
    *, generic_effect: float, unique_lr_effect: float
) -> IncrementalDownstreamFunctional:
    subjects = tuple(f"train-{index}" for index in range(8))
    response, regressor = _response(
        subjects,
        generic_effect=generic_effect,
        unique_lr_effect=unique_lr_effect,
        baseline_scale=0.5,
    )
    manifest = _manifest(subjects, prefix="training")
    resource = build_receiver_autonomous_program_resource(
        GENERIC_PROGRAM[:, None],
        feature_ids=FEATURE_IDS,
        program_ids=("generic_program",),
        resource_id="synthetic-generic-program",
        version="1",
        manifest_digest="a" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    return fit_incremental_downstream_functional(
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        reference_mask=regressor < 0,
        nuisance_matrix=np.ones((len(response), 1)),
        context_regressor=regressor,
        receiver="Receiver",
        contrast_name="target_vs_reference",
        fold_id="overlap-smoke-training",
        context_regressor_id="paired_minus1_plus1",
        nuisance_design_id="intercept_only",
        feature_ids=FEATURE_IDS,
        family_ids=FAMILY_IDS,
        nuisance_column_ids=("intercept",),
        training_subject_ids=subjects,
        family_basis=np.asarray([[1.0], [0.0], [0.0]]),
        autonomous_program_resource=resource,
        precision_weights=np.ones(len(FEATURE_IDS)),
    )


def _apply(
    functional: IncrementalDownstreamFunctional,
    *,
    generic_effect: float,
    unique_lr_effect: float,
    baseline_scale: float,
) -> IncrementalDownstreamApplication:
    subjects = tuple(f"test-{index}" for index in range(6))
    response, regressor = _response(
        subjects,
        generic_effect=generic_effect,
        unique_lr_effect=unique_lr_effect,
        baseline_scale=baseline_scale,
    )
    manifest = _manifest(subjects, prefix=f"heldout-{baseline_scale:g}")
    return apply_incremental_downstream_functional(
        functional,
        response,
        row_manifest=manifest,
        design_sample_ids=manifest.sample_ids,
        nuisance_matrix=np.ones((len(response), 1)),
        context_regressor=regressor,
        context_regressor_id="paired_minus1_plus1",
        nuisance_design_id="intercept_only",
        feature_ids=FEATURE_IDS,
        nuisance_column_ids=("intercept",),
    )


def run_smoke() -> dict[str, object]:
    """Compare generic-only and active effects under two baseline magnitudes."""

    records: list[dict[str, Any]] = []
    for scenario, unique_effect in (("generic_only", 0.0), ("active_unique", 1.0)):
        functional = _fit(generic_effect=2.0, unique_lr_effect=unique_effect)
        for baseline_scale in (0.0, 25.0):
            application = _apply(
                functional,
                generic_effect=2.0,
                unique_lr_effect=unique_effect,
                baseline_scale=baseline_scale,
            )
            records.append(
                {
                    "scenario": scenario,
                    "baseline_scale": baseline_scale,
                    "status": application.status,
                    "reason_code": application.reason_code,
                    "model_gain": application.model_gain,
                    "raw_model_gain": application.raw_model_gain,
                    "family_gain": (
                        None
                        if not np.isfinite(application.family_gains[0])
                        else float(application.family_gains[0])
                    ),
                    "family_coefficient": float(functional.family_coefficients[0]),
                    "retained_norm_fraction": float(
                        functional.family_retained_norm_fraction[0]
                    ),
                    "incremental_functional_id": (functional.incremental_functional_id),
                }
            )
    by_key: dict[tuple[str, float], dict[str, Any]] = {
        (str(record["scenario"]), float(record["baseline_scale"])): record
        for record in records
    }
    invariance = {
        scenario: abs(
            cast(float, by_key[(scenario, 0.0)]["model_gain"])
            - cast(float, by_key[(scenario, 25.0)]["model_gain"])
        )
        for scenario in ("generic_only", "active_unique")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "small_algorithm_counterexample_smoke_not_method_benchmark",
        "provenance": {
            "algorithm_contract": ("sample_keyed_autonomous_projected_incremental_v4"),
            "source_sha256": autonomous_smoke_source_sha256(),
        },
        "records": records,
        "baseline_invariance_absolute_difference": invariance,
        "checks": {
            "generic_only_zero_gain": all(
                cast(float, record["model_gain"]) <= 1e-12
                for record in records
                if record["scenario"] == "generic_only"
            ),
            "active_unique_positive_gain": all(
                cast(float, record["model_gain"]) > 0.9
                for record in records
                if record["scenario"] == "active_unique"
            ),
            "subject_baseline_invariant": all(
                difference <= 1e-12 for difference in invariance.values()
            ),
        },
        "claims": {
            "official_incremental_certification": False,
            "method_superiority": False,
            "biological_discovery": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_smoke()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
