"""Run the under-100k Tensor-cell2cell planted-program benchmark.

The original Code Ocean generation script is not publicly retrievable from this
host. This runner therefore reconstructs the published 12-context protocol and
labels every artifact accordingly. It never presents the result as a byte-level
reproduction of the authors' random realization.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

SCHEMA_VERSION = "crychic-tensor-cell2cell-under100k-v1"
N_CONTEXTS = 12
N_LR = 300
N_CELLS = 3
N_FACTORS = 4
LR_PER_PATHWAY = 100
BACKGROUND_MAXIMUM_SCORE = 0.05
DEFAULT_NOISES = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0)
DEFAULT_REPLICATES = 25


def sha256_file(path: Path) -> str:
    """Return a content digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON with a terminal newline."""

    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def program_definitions() -> tuple[dict[str, Any], ...]:
    """Return the four documented programs and their non-overlapping LR groups.

    The paper has 300 LR pairs but four programs. As described in the
    supplementary notes, an LR pathway can be reused by a different cell-pair
    program. The fourth program thus intentionally reuses pathway 0 with a
    distinct sender-receiver pair.
    """

    return (
        {
            "program_id": "oscillatory",
            "lr_start": 0,
            "lr_stop": 100,
            "sender": 0,
            "receiver": 0,
            "trajectory": np.asarray(
                (
                    0.08,
                    0.50,
                    0.92,
                    0.50,
                    0.08,
                    0.50,
                    0.92,
                    0.50,
                    0.08,
                    0.50,
                    0.92,
                    0.50,
                ),
                dtype=float,
            ),
        },
        {
            "program_id": "pulsatile",
            "lr_start": 100,
            "lr_stop": 200,
            "sender": 1,
            "receiver": 2,
            "trajectory": np.asarray(
                (
                    0.04,
                    0.04,
                    0.08,
                    0.18,
                    0.66,
                    0.96,
                    0.66,
                    0.18,
                    0.08,
                    0.04,
                    0.04,
                    0.04,
                ),
                dtype=float,
            ),
        },
        {
            "program_id": "exponential_decay",
            "lr_start": 200,
            "lr_stop": 300,
            "sender": 2,
            "receiver": 1,
            "trajectory": np.asarray(
                (
                    1.00,
                    0.78,
                    0.61,
                    0.47,
                    0.37,
                    0.29,
                    0.22,
                    0.17,
                    0.13,
                    0.10,
                    0.08,
                    0.06,
                ),
                dtype=float,
            ),
        },
        {
            "program_id": "linear_decrease",
            "lr_start": 0,
            "lr_stop": 100,
            "sender": 0,
            "receiver": 1,
            "trajectory": np.linspace(1.0, 0.04, N_CONTEXTS, dtype=float),
        },
    )


def generate_tensor(
    seed: int, noise: float
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Generate one reconstruction of the published planted-tensor design."""

    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be in [0, 1]")
    expected = np.zeros((N_CONTEXTS, N_LR, N_CELLS, N_CELLS), dtype=float)
    programs = program_definitions()
    for program in programs:
        expected[
            :,
            program["lr_start"] : program["lr_stop"],
            program["sender"],
            program["receiver"],
        ] = program["trajectory"][:, None]
    if noise == 0.0:
        return expected, programs

    rng = np.random.default_rng(seed)
    scale = 1.1 * np.maximum(expected, BACKGROUND_MAXIMUM_SCORE) * noise
    observed = rng.normal(loc=expected, scale=scale)
    return np.clip(observed, 0.0, 1.0), programs


def truth_factor_matrices(
    programs: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build factor-shaped truth arrays in program order."""

    context = np.column_stack(
        [np.asarray(program["trajectory"]) for program in programs]
    )
    lr = np.zeros((N_LR, len(programs)), dtype=float)
    sender = np.zeros((N_CELLS, len(programs)), dtype=float)
    receiver = np.zeros((N_CELLS, len(programs)), dtype=float)
    for index, program in enumerate(programs):
        lr[program["lr_start"] : program["lr_stop"], index] = 1.0
        sender[program["sender"], index] = 1.0
        receiver[program["receiver"], index] = 1.0
    return context, lr, sender, receiver


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return 0.0 if not np.isfinite(value) else value


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    """Return deterministic top indices, retaining the fixed truth cardinality."""

    return np.argsort(-np.asarray(values), kind="mergesort")[:count]


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(np.asarray(left, dtype=int).tolist())
    right_set = set(np.asarray(right, dtype=int).tolist())
    union = left_set.union(right_set)
    return float(len(left_set.intersection(right_set)) / len(union)) if union else 1.0


def match_factors(
    context_truth: np.ndarray,
    lr_truth: np.ndarray,
    sender_truth: np.ndarray,
    receiver_truth: np.ndarray,
    context_pred: np.ndarray,
    lr_pred: np.ndarray,
    sender_pred: np.ndarray,
    receiver_pred: np.ndarray,
) -> list[tuple[int, int]]:
    """Use a factor-order-invariant composite score and Hungarian matching."""

    n_truth = context_truth.shape[1]
    n_pred = context_pred.shape[1]
    if n_truth != n_pred:
        raise ValueError("truth and predicted factor counts must agree")
    similarity = np.zeros((n_truth, n_pred), dtype=float)
    for truth_index in range(n_truth):
        truth_lr = _top_indices(lr_truth[:, truth_index], LR_PER_PATHWAY)
        for pred_index in range(n_pred):
            context_score = abs(
                _safe_pearson(
                    context_truth[:, truth_index], context_pred[:, pred_index]
                )
            )
            lr_score = _jaccard(
                truth_lr, _top_indices(lr_pred[:, pred_index], LR_PER_PATHWAY)
            )
            sender_score = float(
                np.argmax(sender_truth[:, truth_index])
                == np.argmax(sender_pred[:, pred_index])
            )
            receiver_score = float(
                np.argmax(receiver_truth[:, truth_index])
                == np.argmax(receiver_pred[:, pred_index])
            )
            similarity[truth_index, pred_index] = (
                context_score + lr_score + sender_score + receiver_score
            ) / 4.0
    rows, columns = linear_sum_assignment(-similarity)
    return [(int(row), int(column)) for row, column in zip(rows, columns, strict=False)]


def evaluate_factorization(
    tensor: np.ndarray,
    programs: Sequence[Mapping[str, Any]],
    context_pred: np.ndarray,
    lr_pred: np.ndarray,
    sender_pred: np.ndarray,
    receiver_pred: np.ndarray,
    reconstruction: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Score a CP decomposition against known program truth."""

    context_truth, lr_truth, sender_truth, receiver_truth = truth_factor_matrices(
        programs
    )
    matches = match_factors(
        context_truth,
        lr_truth,
        sender_truth,
        receiver_truth,
        context_pred,
        lr_pred,
        sender_pred,
        receiver_pred,
    )
    details: list[dict[str, Any]] = []
    for truth_index, pred_index in matches:
        program = programs[truth_index]
        truth_lr = lr_truth[:, truth_index]
        predicted_lr = lr_pred[:, pred_index]
        true_event = np.einsum(
            "l,s,r->lsr",
            truth_lr,
            sender_truth[:, truth_index],
            receiver_truth[:, truth_index],
        ).reshape(-1)
        predicted_event = np.einsum(
            "l,s,r->lsr",
            predicted_lr,
            sender_pred[:, pred_index],
            receiver_pred[:, pred_index],
        ).reshape(-1)
        details.append(
            {
                "program_id": str(program["program_id"]),
                "truth_factor": truth_index,
                "predicted_factor": pred_index,
                "context_pearson": _safe_pearson(
                    context_truth[:, truth_index], context_pred[:, pred_index]
                ),
                "context_spearman": _safe_spearman(
                    context_truth[:, truth_index], context_pred[:, pred_index]
                ),
                "lr_jaccard_top100": _jaccard(
                    _top_indices(truth_lr, LR_PER_PATHWAY),
                    _top_indices(predicted_lr, LR_PER_PATHWAY),
                ),
                "lr_auprc": float(average_precision_score(truth_lr, predicted_lr)),
                "sender_top1_correct": float(
                    np.argmax(sender_truth[:, truth_index])
                    == np.argmax(sender_pred[:, pred_index])
                ),
                "receiver_top1_correct": float(
                    np.argmax(receiver_truth[:, truth_index])
                    == np.argmax(receiver_pred[:, pred_index])
                ),
                "event_auprc": float(
                    average_precision_score(true_event, predicted_event)
                ),
            }
        )
    squared_error = float(np.square(tensor - reconstruction).sum())
    denominator = float(np.square(tensor).sum())
    summary = {
        "matched_context_pearson": float(
            np.mean([row["context_pearson"] for row in details])
        ),
        "matched_context_spearman": float(
            np.mean([row["context_spearman"] for row in details])
        ),
        "lr_jaccard_top100": float(
            np.mean([row["lr_jaccard_top100"] for row in details])
        ),
        "lr_auprc": float(np.mean([row["lr_auprc"] for row in details])),
        "sender_top1_accuracy": float(
            np.mean([row["sender_top1_correct"] for row in details])
        ),
        "receiver_top1_accuracy": float(
            np.mean([row["receiver_top1_correct"] for row in details])
        ),
        "event_auprc": float(np.mean([row["event_auprc"] for row in details])),
        "normalized_reconstruction_error": squared_error / denominator,
    }
    return summary, details


def _worker_main(args: argparse.Namespace) -> int:
    """Run one native Tensor-cell2cell decomposition in its pinned environment."""

    import cell2cell as c2c
    import tensorly as tl
    from tensorly.cp_tensor import cp_to_tensor

    tensor, _ = generate_tensor(args.seed, args.noise)
    started = time.monotonic()
    model = c2c.tensor.PreBuiltTensor(
        tensor,
        [
            [f"Context-{index + 1}" for index in range(N_CONTEXTS)],
            [f"LR-{index + 1}" for index in range(N_LR)],
            [f"Sender-{index + 1}" for index in range(N_CELLS)],
            [f"Receiver-{index + 1}" for index in range(N_CELLS)],
        ],
        order_labels=["context", "lr", "sender", "receiver"],
    )
    model.compute_tensor_factorization(
        rank=N_FACTORS,
        tf_type="non_negative_cp",
        init="svd",
        random_state=args.seed,
        verbose=False,
        n_iter_max=args.max_iterations,
    )
    factors = model.factors
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        context=factors["context"].to_numpy(dtype=float),
        lr=factors["lr"].to_numpy(dtype=float),
        sender=factors["sender"].to_numpy(dtype=float),
        receiver=factors["receiver"].to_numpy(dtype=float),
        reconstruction=np.asarray(cp_to_tensor(model.tl_object), dtype=float),
    )
    write_json(
        output.with_suffix(".json"),
        {
            "schema_version": SCHEMA_VERSION,
            "method": "Tensor-cell2cell non_negative_cp",
            "cell2cell_version": str(c2c.__version__),
            "tensorly_version": str(tl.__version__),
            "seed": args.seed,
            "noise": args.noise,
            "rank": N_FACTORS,
            "max_iterations": args.max_iterations,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return 0


def _invoke_worker(
    command: Sequence[str],
    label: str,
) -> tuple[str, float, str]:
    """Execute one isolated factorization and retain its captured diagnostic log."""

    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    log = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"Tensor-cell2cell worker failed for {label}: {log}")
    return label, elapsed, log


def _parse_noise(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError("noise levels must be comma-separated values in [0, 1]")
    return values


def run_campaign(args: argparse.Namespace) -> None:
    """Launch the fixed campaign, aggregate factor metrics, and bind provenance."""

    if args.output.exists():
        raise FileExistsError(f"output must not already exist: {args.output}")
    config = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_record = next(
        item
        for item in config["benchmarks"]
        if item["benchmark_id"] == "tensor_cell2cell_planted_12context"
    )
    requested_noises = _parse_noise(args.noise_levels)
    if tuple(contract_record["design"]["noise_levels"]) != requested_noises:
        raise ValueError("noise levels differ from the frozen under-100k contract")
    if args.replicates != contract_record["design"]["replicates_per_noise_level"]:
        raise ValueError("replicate count differs from the frozen under-100k contract")
    if not args.cell2cell_python.is_file():
        raise FileNotFoundError(args.cell2cell_python)
    for asset in (args.supplement, args.source_data):
        if not asset.is_file():
            raise FileNotFoundError(asset)

    args.output.mkdir(parents=True)
    worker_dir = args.output / "workers"
    log_dir = args.output / "logs"
    worker_dir.mkdir()
    log_dir.mkdir()
    jobs: list[tuple[str, int, float, Path]] = []
    for noise_index, noise in enumerate(requested_noises):
        for replicate in range(1, args.replicates + 1):
            seed = args.first_seed + noise_index * 10_000 + replicate
            label = f"noise{noise:g}_rep{replicate:02d}_seed{seed}"
            jobs.append((label, seed, noise, worker_dir / f"{label}.npz"))

    command_rows: list[dict[str, Any]] = []
    commands: list[tuple[list[str], str]] = []
    for label, seed, noise, output_path in jobs:
        command = [
            "env",
            f"OPENBLAS_NUM_THREADS={args.threads_per_worker}",
            f"OMP_NUM_THREADS={args.threads_per_worker}",
            f"MKL_NUM_THREADS={args.threads_per_worker}",
            f"NUMEXPR_NUM_THREADS={args.threads_per_worker}",
            str(args.cell2cell_python),
            str(Path(__file__).resolve()),
            "--worker",
            "--seed",
            str(seed),
            "--noise",
            str(noise),
            "--worker-output",
            str(output_path),
            "--max-iterations",
            str(args.max_iterations),
        ]
        commands.append((command, label))
        command_rows.append({"label": label, "command": " ".join(command)})
    pd.DataFrame(command_rows).to_csv(
        args.output / "commands.tsv", sep="\t", index=False
    )

    campaign_started = time.monotonic()
    invocation: dict[str, tuple[float, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        submitted = {
            executor.submit(_invoke_worker, command, label): label
            for command, label in commands
        }
        for future in concurrent.futures.as_completed(submitted):
            label, elapsed, log = future.result()
            invocation[label] = (elapsed, log)
            (log_dir / f"{label}.log").write_text(log + "\n", encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for label, seed, noise, factorization_path in jobs:
        payload = np.load(factorization_path)
        tensor, programs = generate_tensor(seed, noise)
        summary, details = evaluate_factorization(
            tensor,
            programs,
            payload["context"],
            payload["lr"],
            payload["sender"],
            payload["receiver"],
            payload["reconstruction"],
        )
        worker_metadata = json.loads(
            factorization_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        summary_rows.append(
            {
                "label": label,
                "seed": seed,
                "noise": noise,
                "worker_wall_seconds": invocation[label][0],
                "native_factorization_seconds": worker_metadata["elapsed_seconds"],
                **summary,
            }
        )
        for detail in details:
            detail_rows.append({"label": label, "seed": seed, "noise": noise, **detail})
    summary_table = pd.DataFrame(summary_rows).sort_values(["noise", "seed"])
    detail_table = pd.DataFrame(detail_rows).sort_values(
        ["noise", "seed", "truth_factor"]
    )
    summary_table.to_csv(args.output / "replicate_metrics.tsv", sep="\t", index=False)
    detail_table.to_csv(
        args.output / "matched_factor_metrics.tsv", sep="\t", index=False
    )
    aggregate = (
        summary_table.groupby("noise", as_index=False)
        .agg(
            replicate_count=("seed", "size"),
            matched_context_pearson_mean=("matched_context_pearson", "mean"),
            matched_context_pearson_sd=("matched_context_pearson", "std"),
            matched_context_spearman_mean=("matched_context_spearman", "mean"),
            lr_jaccard_top100_mean=("lr_jaccard_top100", "mean"),
            lr_auprc_mean=("lr_auprc", "mean"),
            sender_top1_accuracy_mean=("sender_top1_accuracy", "mean"),
            receiver_top1_accuracy_mean=("receiver_top1_accuracy", "mean"),
            event_auprc_mean=("event_auprc", "mean"),
            normalized_reconstruction_error_mean=(
                "normalized_reconstruction_error",
                "mean",
            ),
            worker_wall_seconds_mean=("worker_wall_seconds", "mean"),
        )
        .sort_values("noise")
    )
    aggregate.to_csv(args.output / "aggregate_metrics.tsv", sep="\t", index=False)
    write_json(
        args.output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "protocol_status": "paper_protocol_reconstruction_not_byte_identical",
            "contract": {
                "path": str(args.contract),
                "sha256": sha256_file(args.contract),
                "benchmark_id": contract_record["benchmark_id"],
            },
            "source_assets": {
                "supplement": {
                    "path": str(args.supplement),
                    "sha256": sha256_file(args.supplement),
                },
                "source_data": {
                    "path": str(args.source_data),
                    "sha256": sha256_file(args.source_data),
                },
            },
            "source_repository": _git_metadata(args.repo_root),
            "execution": {
                "orchestrator_python": sys.version,
                "platform": platform.platform(),
                "workers": args.workers,
                "threads_per_worker": args.threads_per_worker,
                "max_iterations": args.max_iterations,
                "noise_levels": list(requested_noises),
                "replicates_per_noise": args.replicates,
                "first_seed": args.first_seed,
                "campaign_wall_seconds": time.monotonic() - campaign_started,
                "worker_count": len(jobs),
            },
            "artifacts": {
                name: {"sha256": sha256_file(args.output / name)}
                for name in (
                    "commands.tsv",
                    "replicate_metrics.tsv",
                    "matched_factor_metrics.tsv",
                    "aggregate_metrics.tsv",
                )
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for both the orchestrator and pinned worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "suggest_v5_under100k_v1.json",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--cell2cell-python", type=Path)
    parser.add_argument("--noise-levels", default=",".join(map(str, DEFAULT_NOISES)))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--first-seed", type=int, default=20_260_724)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 24))
    parser.add_argument("--threads-per-worker", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP007
    args = build_parser().parse_args(argv)
    if args.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if args.worker:
        if args.worker_output is None:
            raise ValueError("--worker-output is required in worker mode")
        return _worker_main(args)
    if args.output is None or args.supplement is None or args.source_data is None:
        raise ValueError("--output, --supplement, and --source-data are required")
    if args.cell2cell_python is None:
        raise ValueError("--cell2cell-python is required")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be positive")
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
