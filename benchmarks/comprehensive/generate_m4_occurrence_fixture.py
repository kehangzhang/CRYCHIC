"""Generate independent-subject occurrence/prevalence stress fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-m4-occurrence-fixture-v1"
CONDITIONS = ("ctrl", "stim")
FAMILY_COUNTS: Mapping[str, int] = {
    "occurrence_increase": 30,
    "occurrence_decrease": 30,
    "structural_missing_occurrence": 30,
    "magnitude_only": 60,
    "abundance_only": 60,
    "global_null": 90,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _dataset_seed(root_seed: int) -> int:
    payload = f"crychic:m4-occurrence:v1:{root_seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _probabilities(
    family: str, rng: np.random.Generator
) -> tuple[float, float, int, bool]:
    if family in {"occurrence_increase", "structural_missing_occurrence"}:
        reference = float(rng.uniform(0.05, 0.25))
        target = float(min(0.95, reference + rng.uniform(0.40, 0.65)))
        return reference, target, 1, True
    if family == "occurrence_decrease":
        target = float(rng.uniform(0.05, 0.25))
        reference = float(min(0.95, target + rng.uniform(0.40, 0.65)))
        return reference, target, -1, True
    shared = float(rng.uniform(0.10, 0.80))
    return shared, shared, 0, False


def _simulate_dataset(
    *, root_seed: int, n_subjects_per_group: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(_dataset_seed(root_seed))
    event_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    event_index = 0
    for family, count in FAMILY_COUNTS.items():
        for _ in range(count):
            event_index += 1
            event_id = f"E{event_index:04d}"
            reference_p, target_p, direction, positive = _probabilities(family, rng)
            base_log_magnitude = float(rng.normal(loc=-0.15, scale=0.25))
            nuisance_direction = int(rng.choice(np.asarray([-1, 1])))
            magnitude_shift = (
                1.20 * nuisance_direction if family == "magnitude_only" else 0.0
            )
            abundance_shift = (
                0.28 * nuisance_direction if family == "abundance_only" else 0.0
            )
            for condition, probability, prefix in (
                ("ctrl", reference_p, "C"),
                ("stim", target_p, "T"),
            ):
                for subject_index in range(n_subjects_per_group):
                    occurrence = int(rng.random() < probability)
                    missing_probability = (
                        0.25 if family == "structural_missing_occurrence" else 0.03
                    )
                    missing = bool(rng.random() < missing_probability)
                    log_magnitude = base_log_magnitude
                    if condition == "stim":
                        log_magnitude += magnitude_shift
                    magnitude = occurrence * float(
                        np.exp(log_magnitude + rng.normal(scale=0.18))
                    )
                    activity = 0.08 + 0.42 * magnitude
                    if condition == "stim":
                        activity += abundance_shift
                    activity += float(rng.normal(scale=0.015))
                    activity = float(np.clip(activity, 0.0, 1.0))
                    event_rows.append(
                        {
                            "event_id": event_id,
                            "subject_id": f"{prefix}{subject_index + 1:03d}",
                            "condition": condition,
                            "occurrence": np.nan if missing else occurrence,
                            "m0_activity": np.nan if missing else activity,
                        }
                    )
            true_log_odds_ratio = math_logit(target_p) - math_logit(reference_p)
            truth_rows.append(
                {
                    "event_id": event_id,
                    "family": family,
                    "expected_occurrence_differential": positive,
                    "expected_direction": direction,
                    "true_reference_prevalence": reference_p,
                    "true_target_prevalence": target_p,
                    "true_prevalence_difference": target_p - reference_p,
                    "true_log_odds_ratio": true_log_odds_ratio,
                }
            )
    events = pd.DataFrame.from_records(event_rows)
    truth = pd.DataFrame.from_records(truth_rows)
    return events, truth


def math_logit(probability: float) -> float:
    return float(np.log(probability) - np.log1p(-probability))


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


def generate(
    output_dir: Path,
    *,
    seeds: Sequence[int],
    n_subjects_per_group: int = 16,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write immutable subject-event inputs and separate occurrence truth."""

    output_dir = output_dir.resolve()
    resolved_seeds = tuple(map(int, seeds))
    if not resolved_seeds or len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("seeds must be non-empty and unique")
    if n_subjects_per_group < 8:
        raise ValueError("n_subjects_per_group must be at least eight")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        inputs = staged / "inputs"
        inputs.mkdir()
        records: list[dict[str, Any]] = []
        truth_frames: list[pd.DataFrame] = []
        for replicate, root_seed in enumerate(resolved_seeds, start=1):
            dataset_id = f"m4_occurrence_r{replicate:03d}"
            events, truth = _simulate_dataset(
                root_seed=root_seed,
                n_subjects_per_group=n_subjects_per_group,
            )
            dataset_dir = inputs / dataset_id
            dataset_dir.mkdir()
            input_path = dataset_dir / "subject_events.tsv.gz"
            events.to_csv(
                input_path,
                sep="\t",
                index=False,
                compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
                lineterminator="\n",
            )
            truth.insert(0, "root_seed", root_seed)
            truth.insert(0, "dataset_id", dataset_id)
            truth_frames.append(truth)
            records.append(
                {
                    "dataset_id": dataset_id,
                    "root_seed": root_seed,
                    "dataset_seed": _dataset_seed(root_seed),
                    "input": str(input_path.relative_to(staged)),
                    "input_sha256": sha256_file(input_path),
                    "rows": int(len(events)),
                    "events": int(events["event_id"].nunique()),
                }
            )
        truth_table = pd.concat(truth_frames, ignore_index=True)
        truth_path = staged / "event_truth.tsv.gz"
        truth_table.to_csv(
            truth_path,
            sep="\t",
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
            lineterminator="\n",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "seeds": list(resolved_seeds),
            "conditions": list(CONDITIONS),
            "design": "independent_groups",
            "n_subjects_per_group": int(n_subjects_per_group),
            "family_counts": dict(FAMILY_COUNTS),
            "truth": {
                "filename": truth_path.name,
                "sha256": sha256_file(truth_path),
                "rows": int(len(truth_table)),
            },
            "records": records,
            "leakage_controls": {
                "dataset_ids_mask_family_truth": True,
                "truth_is_separate_from_method_inputs": True,
                "subject_is_inferential_unit": True,
                "missing_occurrence_is_not_encoded_as_absence": True,
            },
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n-subjects-per-group", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate(
        args.output_dir,
        seeds=args.seeds,
        n_subjects_per_group=args.n_subjects_per_group,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
