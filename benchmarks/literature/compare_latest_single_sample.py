"""Compare latest-source single-sample reruns with the frozen literature bundle."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA = "crychic-latest-single-sample-consistency-v1"
CITE_DATASETS = {
    "5k_pbmc_protein_v3": "5k_pbmc",
    "5k_pbmc_protein_v3_nextgem": "5k_pbmc_nextgem",
    "cbmc_seuratdata": "cbmc",
    "malt_10k_protein_v3": "malt_10k",
    "pbmc_10k_protein_v3": "pbmc_10k",
    "sln_111": "sln_111",
    "sln_208": "sln_208",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _compare(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    keys: Sequence[str],
    numeric: Sequence[str],
    text: Sequence[str],
) -> dict[str, Any]:
    selected = [*keys, *numeric, *text]
    if set(selected).difference(old) or set(selected).difference(new):
        raise ValueError("single-sample comparison schema is incomplete")
    if old.duplicated(list(keys)).any() or new.duplicated(list(keys)).any():
        raise ValueError("single-sample comparison keys are not unique")
    merged = old.loc[:, selected].merge(
        new.loc[:, selected],
        on=list(keys),
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
        sort=True,
    )
    both = merged["_merge"].eq("both")
    maximum = 0.0
    numeric_mismatches = 0
    for column in numeric:
        left = pd.to_numeric(
            merged.loc[both, f"{column}_old"], errors="coerce"
        ).to_numpy()
        right = pd.to_numeric(
            merged.loc[both, f"{column}_new"], errors="coerce"
        ).to_numpy()
        equal = np.isclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)
        numeric_mismatches += int((~equal).sum())
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(left[finite] - right[finite]))))
    text_mismatches = 0
    for column in text:
        left = merged.loc[both, f"{column}_old"].astype("string")
        right = merged.loc[both, f"{column}_new"].astype("string")
        text_mismatches += int(
            (~left.eq(right).fillna(left.isna() & right.isna())).sum()
        )
    unmatched = int((~both).sum())
    return {
        "old_rows": len(old),
        "new_rows": len(new),
        "matched_rows": int(both.sum()),
        "unmatched_rows": unmatched,
        "numeric_mismatches": numeric_mismatches,
        "text_mismatches": text_mismatches,
        "max_absolute_numeric_difference": maximum,
        "exactly_consistent": bool(
            unmatched == numeric_mismatches == text_mismatches == 0
            and len(old) == len(new)
        ),
    }


def run(
    literature_root: Path,
    rerun_root: Path,
    output: Path,
    contract: Path,
    repo_root: Path,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    common_keys = (
        "sample_id",
        "subject_id",
        "context_json",
        "sender",
        "receiver",
        "interaction_id",
    )
    common_numeric = (
        "ligand_availability",
        "receptor_availability",
        "availability_state",
    )
    common_text = ("ligand", "receptor", "status", "reason_code")
    rows: list[dict[str, Any]] = []
    for dataset_id, old_name in CITE_DATASETS.items():
        old_path = (
            literature_root
            / "results"
            / "citeseq"
            / old_name
            / "H-common"
            / "crychic_availability"
            / "availability_scores.parquet"
        )
        new_path = rerun_root / "citeseq" / dataset_id / "availability_scores.parquet"
        rows.append(
            {
                "track": "CITE-seq",
                "dataset_id": dataset_id,
                "status": "complete",
                "reason_code": "",
                **_compare(
                    pd.read_parquet(old_path),
                    pd.read_parquet(new_path),
                    keys=common_keys,
                    numeric=common_numeric,
                    text=common_text,
                ),
                "old_sha256": sha256_file(old_path),
                "new_sha256": sha256_file(new_path),
            }
        )
    sample_manifest = pd.read_csv(
        literature_root / "results" / "ipf" / "full_cohort" / "sample_manifest.tsv",
        sep="\t",
    )
    for item in sample_manifest.itertuples(index=False):
        old_path = (
            literature_root
            / "results"
            / "ipf"
            / "full_cohort"
            / item.study_id
            / item.sample_id
            / "H-common"
            / "crychic_availability"
            / "availability_scores.parquet"
        )
        new_path = (
            rerun_root
            / "ipf"
            / item.study_id
            / item.sample_id
            / "availability_scores.parquet"
        )
        rows.append(
            {
                "track": "IPF",
                "dataset_id": f"{item.study_id}/{item.sample_id}",
                "status": "complete",
                "reason_code": "",
                **_compare(
                    pd.read_parquet(old_path),
                    pd.read_parquet(new_path),
                    keys=common_keys,
                    numeric=common_numeric,
                    text=common_text,
                ),
                "old_sha256": sha256_file(old_path),
                "new_sha256": sha256_file(new_path),
            }
        )
    old_path = (
        literature_root
        / "cytokine"
        / "her2"
        / "crychic"
        / "availability_lr_scores.parquet"
    )
    new_path = rerun_root / "cytosig" / "HER2" / "availability_lr_scores.parquet"
    rows.append(
        {
            "track": "CytoSig",
            "dataset_id": "HER2_Wu2021",
            "status": "complete",
            "reason_code": "",
            **_compare(
                pd.read_parquet(old_path),
                pd.read_parquet(new_path),
                keys=(
                    "sample_id",
                    "subject_id",
                    "context_id",
                    "cyto_context",
                    "source",
                    "target",
                    "interaction_id",
                ),
                numeric=(
                    "ligand_availability",
                    "receptor_availability",
                    "sender_proportion",
                    "receiver_proportion",
                    "availability_state",
                    "availability_ecosystem",
                ),
                text=(
                    "ligand",
                    "receptor",
                    "state_status",
                    "state_reason_code",
                    "ecosystem_status",
                    "ecosystem_reason_code",
                ),
            ),
            "old_sha256": sha256_file(old_path),
            "new_sha256": sha256_file(new_path),
        }
    )
    rows.append(
        {
            "track": "CytoSig",
            "dataset_id": "TNBC",
            "status": "NE",
            "reason_code": "prepared_tnbc_h5ad_not_retained_locally",
            "old_rows": 3_623_028,
            "new_rows": None,
            "matched_rows": None,
            "unmatched_rows": None,
            "numeric_mismatches": None,
            "text_mismatches": None,
            "max_absolute_numeric_difference": None,
            "exactly_consistent": None,
            "old_sha256": sha256_file(
                literature_root
                / "cytokine"
                / "results"
                / "crychic"
                / "availability_lr_scores.parquet"
            ),
            "new_sha256": None,
        }
    )
    detail = pd.DataFrame.from_records(rows)
    complete = detail.loc[detail["status"].eq("complete")].copy()
    summary = complete.groupby("track", as_index=False).agg(
        datasets=("dataset_id", "size"),
        old_rows=("old_rows", "sum"),
        new_rows=("new_rows", "sum"),
        unmatched_rows=("unmatched_rows", "sum"),
        numeric_mismatches=("numeric_mismatches", "sum"),
        text_mismatches=("text_mismatches", "sum"),
        max_absolute_numeric_difference=("max_absolute_numeric_difference", "max"),
        exactly_consistent=("exactly_consistent", "all"),
    )
    coverage = pd.DataFrame.from_records(
        [
            {
                "score_head": "availability_state",
                "scenario": "single_sample",
                "status": "complete",
                "reason_code": "",
            },
            {
                "score_head": "RC12_sender_response_detection",
                "scenario": "single_sample",
                "status": "NE",
                "reason_code": (
                    "requires_train_derived_sender_assignment_and_between_condition_"
                    "receiver_program"
                ),
            },
            {
                "score_head": "RC14_differential_DES",
                "scenario": "single_sample",
                "status": "NE",
                "reason_code": (
                    "requires_multicondition_differential_statistics_and_spatial_"
                    "pair_prior"
                ),
            },
        ]
    )
    detail.to_csv(output / "dataset_consistency.tsv", sep="\t", index=False)
    summary.to_csv(output / "track_summary.tsv", sep="\t", index=False)
    coverage.to_csv(output / "score_head_coverage.tsv", sep="\t", index=False)
    rerun_manifests = sorted(rerun_root.glob("**/manifest.json"))
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "complete_with_declared_NE",
            "interpretation": (
                "exact row-level equality implies deterministic point metrics and "
                "ranks are unchanged for the same estimand"
            ),
            "source_repository": git_metadata(repo_root),
            "contract": {
                "path": str(contract.resolve()),
                "sha256": sha256_file(contract),
            },
            "execution": {
                "wall_seconds": time.perf_counter() - started,
                "complete_datasets": len(complete),
                "ne_datasets": int(detail["status"].eq("NE").sum()),
            },
            "rerun_manifests": {
                str(path.relative_to(rerun_root)): sha256_file(path)
                for path in rerun_manifests
            },
            "artifacts": {
                name: {"sha256": sha256_file(output / name)}
                for name in (
                    "dataset_consistency.tsv",
                    "track_summary.tsv",
                    "score_head_coverage.tsv",
                )
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--literature-root", type=Path, required=True)
    parser.add_argument("--rerun-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "suggest_v5_gap_completion_v1.json",
    )
    args = parser.parse_args(argv)
    run(
        args.literature_root,
        args.rerun_root,
        args.output,
        args.contract,
        args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
