"""Complete deterministic suggest-v5 metrics from frozen benchmark artifacts.

This module only performs checksum-bindable post-processing. It does not refit
Tensor-cell2cell or CRYCHIC and never converts unavailable structure to zero.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.run_tensor_cell2cell_under100k import (
    LR_PER_PATHWAY,
    generate_tensor,
    match_factors,
    truth_factor_matrices,
)

SCHEMA = "crychic-suggest-v5-posthoc-gap-completion-v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _prepare_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)


def _stable_top(values: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-np.asarray(values, dtype=float), kind="mergesort")[:count]


def _binary_at_top(values: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros(len(values), dtype=int)
    result[_stable_top(values, count)] = 1
    return result


def _dtw(left: np.ndarray, right: np.ndarray) -> float:
    """Return path-length-normalized absolute-cost dynamic time warping."""

    n, m = len(left), len(right)
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    steps = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                (cost[i - 1, j], steps[i - 1, j]),
                (cost[i, j - 1], steps[i, j - 1]),
                (cost[i - 1, j - 1], steps[i - 1, j - 1]),
            )
            previous_cost, previous_steps = min(choices, key=lambda item: item[0])
            cost[i, j] = previous_cost + abs(float(left[i - 1] - right[j - 1]))
            steps[i, j] = previous_steps + 1
    return float(cost[n, m] / steps[n, m])


def _scale_aligned(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denominator = float(np.dot(predicted, predicted))
    scale = (
        0.0
        if denominator == 0.0
        else max(0.0, float(np.dot(predicted, truth) / denominator))
    )
    return predicted * scale


def _ranking_metrics(
    truth: np.ndarray, score: np.ndarray, *, top: int
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    predicted = _binary_at_top(score, min(top, len(score)))
    result = {
        "auroc": float(roc_auc_score(truth, score)),
        "mcc": float(matthews_corrcoef(truth, predicted)),
        f"precision_at_{top}": float(
            precision_score(truth, predicted, zero_division=0)
        ),
        f"recall_at_{top}": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "ndcg": float(ndcg_score(truth[None, :], score[None, :])),
    }
    for count in (50, 100):
        selected = _stable_top(score, min(count, len(score)))
        result[f"precision_at_{count}"] = float(truth[selected].mean())
        result[f"recall_at_{count}"] = float(
            truth[selected].sum() / max(1, truth.sum())
        )
    return result


def tensor_extended(
    input_dir: Path, output: Path, contract: Path, repo_root: Path
) -> None:
    _prepare_output(output)
    started = time.perf_counter()
    original = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    replicate = pd.read_csv(input_dir / "replicate_metrics.tsv", sep="\t")
    factor_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for row in replicate.itertuples(index=False):
        payload = np.load(input_dir / "workers" / f"{row.label}.npz")
        _, programs = generate_tensor(int(row.seed), float(row.noise))
        context_truth, lr_truth, sender_truth, receiver_truth = truth_factor_matrices(
            programs
        )
        matches = match_factors(
            context_truth,
            lr_truth,
            sender_truth,
            receiver_truth,
            payload["context"],
            payload["lr"],
            payload["sender"],
            payload["receiver"],
        )
        current: list[dict[str, Any]] = []
        for truth_index, predicted_index in matches:
            context = _scale_aligned(
                payload["context"][:, predicted_index], context_truth[:, truth_index]
            )
            lr_score = payload["lr"][:, predicted_index]
            lr_binary = lr_truth[:, truth_index].astype(int)
            sender_score = payload["sender"][:, predicted_index]
            receiver_score = payload["receiver"][:, predicted_index]
            event_truth = (
                np.einsum(
                    "l,s,r->lsr",
                    lr_binary,
                    sender_truth[:, truth_index],
                    receiver_truth[:, truth_index],
                )
                .reshape(-1)
                .astype(int)
            )
            event_score = np.einsum(
                "l,s,r->lsr", lr_score, sender_score, receiver_score
            ).reshape(-1)
            lr_metrics = _ranking_metrics(lr_binary, lr_score, top=LR_PER_PATHWAY)
            event_metrics = _ranking_metrics(
                event_truth, event_score, top=LR_PER_PATHWAY
            )
            detail = {
                "label": row.label,
                "seed": int(row.seed),
                "noise": float(row.noise),
                "program_id": programs[truth_index]["program_id"],
                "truth_factor": truth_index,
                "predicted_factor": predicted_index,
                "context_rmse_scale_aligned": float(
                    np.sqrt(np.mean(np.square(context - context_truth[:, truth_index])))
                ),
                "context_dtw_scale_aligned": _dtw(
                    context, context_truth[:, truth_index]
                ),
                "context_peak_absolute_error": int(
                    abs(
                        int(np.argmax(context))
                        - int(np.argmax(context_truth[:, truth_index]))
                    )
                ),
                "sender_loading_jaccard_top1": float(
                    np.argmax(sender_score) == np.argmax(sender_truth[:, truth_index])
                ),
                "receiver_loading_jaccard_top1": float(
                    np.argmax(receiver_score)
                    == np.argmax(receiver_truth[:, truth_index])
                ),
                **{f"lr_{key}": value for key, value in lr_metrics.items()},
                **{f"event_{key}": value for key, value in event_metrics.items()},
            }
            current.append(detail)
            factor_rows.append(detail)
        numeric = (
            pd.DataFrame(current)
            .select_dtypes(include=[np.number])
            .drop(columns=["seed", "noise", "truth_factor", "predicted_factor"])
        )
        replicate_rows.append(
            {
                "label": row.label,
                "seed": int(row.seed),
                "noise": float(row.noise),
                **numeric.mean().to_dict(),
            }
        )
    factors = pd.DataFrame(factor_rows).sort_values(["noise", "seed", "truth_factor"])
    replicates = pd.DataFrame(replicate_rows).sort_values(["noise", "seed"])
    aggregate = replicates.groupby("noise", as_index=False).agg(
        replicate_count=("seed", "size"),
        **{
            f"{column}_mean": (column, "mean")
            for column in replicates.columns
            if column not in {"label", "seed", "noise"}
        },
    )
    coverage = pd.DataFrame.from_records(
        [
            {
                "endpoint": "extended_factor_metrics",
                "status": "complete",
                "reason_code": "",
            },
            {
                "endpoint": "CorrIndex",
                "status": "NE",
                "reason_code": "single_score_generator_no_pairwise_decomposition",
            },
            {
                "endpoint": "strict_or_partial_hyperedge_recovery",
                "status": "NE",
                "reason_code": (
                    "simulation_truth_contains_simple_lr_events_not_"
                    "multisubunit_hyperedges"
                ),
            },
            {
                "endpoint": "signed_event_recovery",
                "status": "NE",
                "reason_code": "nonnegative_tensor_simulation_has_no_signed_truth",
            },
        ]
    )
    paths = {
        "matched_factor_extended_metrics.tsv": factors,
        "replicate_extended_metrics.tsv": replicates,
        "aggregate_extended_metrics.tsv": aggregate,
        "endpoint_coverage.tsv": coverage,
    }
    for name, frame in paths.items():
        frame.to_csv(output / name, sep="\t", index=False)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "track": "tensor_extended_posthoc",
            "status": "complete",
            "source_repository": git_metadata(repo_root),
            "contract": {
                "path": str(contract.resolve()),
                "sha256": sha256_file(contract),
            },
            "source": {
                "path": str(input_dir.resolve()),
                "manifest_sha256": sha256_file(input_dir / "manifest.json"),
                "source_commit": original["source_repository"]["commit"],
                "source_dirty": original["source_repository"]["dirty"],
            },
            "execution": {
                "wall_seconds": time.perf_counter() - started,
                "replicates": len(replicates),
            },
            "artifacts": {
                name: {"sha256": sha256_file(output / name)} for name in paths
            },
        },
    )


def _bh(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    ordered = numeric.loc[valid].sort_values(kind="mergesort")
    if ordered.empty:
        return result
    adjusted = (ordered.to_numpy() * len(ordered) / np.arange(1, len(ordered) + 1))[
        ::-1
    ]
    result.loc[ordered.index] = np.minimum(np.minimum.accumulate(adjusted)[::-1], 1.0)
    return result


def _context(value: str) -> str:
    parsed = json.loads(value)
    if len(parsed) != 1:
        raise ValueError(f"DCST context is not one-dimensional: {value}")
    return str(next(iter(parsed.values())))


def dcst_binary(input_dir: Path, output: Path, contract: Path, repo_root: Path) -> None:
    _prepare_output(output)
    config = json.loads(contract.read_text(encoding="utf-8"))
    threshold = float(config["dcst"]["binary_score_threshold"])
    started = time.perf_counter()
    result_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    paths = sorted(
        input_dir.glob("realizations/*/*/*/crychic/interactions_long.parquet")
    )
    if not paths:
        raise FileNotFoundError("no DCST realization score tables")
    for score_path in paths:
        realization = score_path.parents[1]
        source_manifest = json.loads(
            (realization / "manifest.json").read_text(encoding="utf-8")
        )
        scores = pd.read_parquet(score_path)
        scores = scores.loc[scores["status"].astype(str).eq("ok")].copy()
        scores["condition"] = scores["context_json"].map(_context)
        scores["present"] = pd.to_numeric(scores["score"], errors="coerce").gt(
            threshold
        )
        keys = ["sender", "receiver", "interaction_id"]
        rows: list[dict[str, Any]] = []
        for key, group in scores.groupby(keys, observed=True, sort=True):
            by_condition = {
                condition: part.drop_duplicates("subject_id")["present"].astype(bool)
                for condition, part in group.groupby("condition", observed=True)
            }
            if set(by_condition) != {"C1", "C2"}:
                raise ValueError("DCST realization lacks C1/C2 subject scores")
            c1, c2 = by_condition["C1"], by_condition["C2"]
            table = np.asarray(
                [[int(c2.sum()), int((~c2).sum())], [int(c1.sum()), int((~c1).sum())]],
                dtype=int,
            )
            odds, p_value = fisher_exact(table, alternative="two-sided")
            rows.append(
                {
                    "dataset_id": source_manifest["dataset_id"],
                    "sender": key[0],
                    "receiver": key[1],
                    "interaction_id": key[2],
                    "threshold": threshold,
                    "c1_present": int(c1.sum()),
                    "c1_absent": int((~c1).sum()),
                    "c2_present": int(c2.sum()),
                    "c2_absent": int((~c2).sum()),
                    "presence_effect_c2_minus_c1": float(c2.mean() - c1.mean()),
                    "odds_ratio": float(odds),
                    "p_value": float(p_value),
                }
            )
        events = pd.DataFrame(rows)
        events["q_value"] = _bh(events["p_value"])
        truth = pd.read_csv(realization / "truth" / "event_truth.tsv", sep="\t")
        events = events.merge(truth, on=["dataset_id", *keys], validate="one_to_one")
        ranking = -np.log10(events["p_value"].clip(lower=np.finfo(float).tiny))
        truth_label = events["truth_label"].astype(int)
        called = events["q_value"].le(0.05)
        active = truth_label.eq(1)
        null = ~active
        direction = np.sign(events["presence_effect_c2_minus_c1"])
        metric_rows.append(
            {
                "sweep": source_manifest["published_sweep"],
                "axis_name": score_path.parents[2].name.rsplit("_", maxsplit=1)[0],
                "axis_value": int(
                    re.search(r"(\d+)$", score_path.parents[2].name).group(1)
                ),
                "seed": int(source_manifest["parameters"]["seed"]),
                "cells": int(source_manifest["dimensions"]["cells"]),
                "method": "crychic_fixed_binary_tau_0.25",
                "auroc": float(roc_auc_score(truth_label, ranking)),
                "auprc": float(average_precision_score(truth_label, ranking)),
                "direction_accuracy": float(
                    (
                        direction.loc[active] == events.loc[active, "truth_direction"]
                    ).mean()
                ),
                "sensitivity_q005": float(called.loc[active].mean()),
                "specificity_q005": float((~called.loc[null]).mean()),
                "type1_error_q005": float(called.loc[null].mean()),
                "empirical_fdr_q005": float(
                    (called & null).sum() / max(1, called.sum())
                ),
            }
        )
        events.insert(
            0, "realization", str(realization.relative_to(input_dir / "realizations"))
        )
        result_rows.append(events)
    event_table = pd.concat(result_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["sweep", "axis_value", "seed"])
    aggregate = metrics.groupby(
        ["sweep", "axis_name", "axis_value", "method"], as_index=False
    ).agg(
        replicates=("seed", "size"),
        **{
            f"{column}_mean": (column, "mean")
            for column in metrics.columns
            if column
            not in {"sweep", "axis_name", "axis_value", "seed", "cells", "method"}
        },
    )
    event_table.to_csv(
        output / "event_results.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    metrics.to_csv(output / "replicate_metrics.tsv", sep="\t", index=False)
    aggregate.to_csv(output / "aggregate_metrics.tsv", sep="\t", index=False)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "track": "dcst_fixed_binary_posthoc",
            "status": "complete",
            "source_repository": git_metadata(repo_root),
            "contract": {
                "path": str(contract.resolve()),
                "sha256": sha256_file(contract),
            },
            "parameters": {
                "threshold": threshold,
                "presence_rule": "score > threshold",
                "test": "two-sided Fisher exact",
                "multiplicity": "BH per realization",
            },
            "execution": {
                "wall_seconds": time.perf_counter() - started,
                "realizations": len(paths),
            },
            "source": {
                "path": str(input_dir.resolve()),
                "manifest_sha256": sha256_file(input_dir / "manifest.json"),
            },
            "artifacts": {
                name: {"sha256": sha256_file(output / name)}
                for name in (
                    "event_results.tsv.gz",
                    "replicate_metrics.tsv",
                    "aggregate_metrics.tsv",
                )
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("tensor", "dcst"):
        child = subparsers.add_parser(command)
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--repo-root", type=Path, default=Path.cwd())
        child.add_argument(
            "--contract",
            type=Path,
            default=Path(__file__).resolve().parents[1]
            / "configs"
            / "suggest_v5_gap_completion_v1.json",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "tensor":
        tensor_extended(args.input, args.output, args.contract, args.repo_root)
    else:
        dcst_binary(args.input, args.output, args.contract, args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
