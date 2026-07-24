"""Run the frozen 20-seed score-generator by differential-engine crossover."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import psutil  # type: ignore[import-untyped]
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.evaluate_component_crossover import (
    _read_component_long,
)

SCHEMA_VERSION = "crychic-score-engine-crossover-under100k-v1"
SCORE_GENERATORS = (
    "crychic_native",
    "crychic_availability",
    "crychic_availability_prior",
    "crychic_sender_response",
    "crychic_downstream_support",
    "liana_magnitude",
    "cellchat_probability",
    "cellphonedb_mean",
    "simple_lr_mean",
    "simple_lr_product",
)
OBSERVED_ENGINES = ("sample_glm", "clustered_cr2", "staccato")
ALL_ENGINES = (
    *OBSERVED_ENGINES,
    "crychic_common_functional_oof",
    "scseqcommdiff_multi_sample",
)
EDGE_KEYS = ("sender", "receiver", "interaction_id")
CONTRASTS = {
    "B_vs_A": ("B", "A"),
    "C_vs_A": ("C", "A"),
    "C_vs_B": ("C", "B"),
}
SCORE_COLUMNS = (
    "dataset_id",
    "scenario",
    "seed",
    "score_generator",
    "contrast",
    "target",
    "reference",
    "sample_id",
    "subject_id",
    "condition",
    *EDGE_KEYS,
    "ligand",
    "receptor",
    "score",
    "status",
    "reason_code",
)


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _bound_table(directory: Path, manifest: Mapping[str, Any]) -> Path:
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"adapter output provenance is absent: {directory}")
    path = directory / str(output.get("table"))
    if not path.is_file() or sha256_file(path) != output.get("sha256"):
        raise ValueError(f"adapter output checksum mismatch: {path}")
    return path


def _condition(context_json: object) -> str:
    payload = json.loads(str(context_json))
    if not isinstance(payload, dict) or "condition" not in payload:
        raise ValueError(f"context_json lacks condition: {context_json}")
    return str(payload["condition"])


def _dataset_metadata(fixture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = fixture.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("fixture records are absent")
    result: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("fixture record must be an object")
        dataset_id = str(raw.get("dataset_id", ""))
        if not dataset_id or dataset_id in result:
            raise ValueError(f"invalid fixture dataset_id: {dataset_id!r}")
        result[dataset_id] = {
            "scenario": str(raw["scenario"]),
            "seed": int(raw["seed"]),
            "manifest": str(raw["manifest"]),
        }
    return result


def _duplicate_contrasts(table: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for contrast, (target, reference) in CONTRASTS.items():
        selected = table.copy()
        selected["contrast"] = contrast
        selected["target"] = target
        selected["reference"] = reference
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def _generator_runtime_row(
    *,
    dataset_id: str,
    generator: str,
    elapsed_seconds: float,
    accounting: str,
    source: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "score_generator": generator,
        "elapsed_seconds": elapsed_seconds,
        "runtime_accounting": accounting,
        "source": source,
    }


def _load_internal_scores(
    runs_dir: Path,
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    provenance: list[dict[str, Any]] = []
    components = {
        "crychic_native": "comm_strength",
        "crychic_availability": "availability",
        "crychic_sender_response": "sender_component",
        "crychic_downstream_support": "downstream",
    }
    for run_dir in sorted(runs_dir.glob("crychic__*")):
        manifest = _read_json(run_dir / "manifest.json")
        dataset_id = str(manifest.get("dataset_id", ""))
        if dataset_id not in metadata:
            continue
        table, source_provenance = _read_component_long(run_dir)
        source_provenance["dataset_id"] = dataset_id
        provenance.append(source_provenance)
        meta = metadata[dataset_id]
        for contrast, (target, reference) in CONTRASTS.items():
            view = table.loc[
                table["contrast_view"].astype(str).eq(f"global:'{target}'")
            ].copy()
            keys = ["sample_id", *EDGE_KEYS]
            if view.empty or view.duplicated(keys).any():
                raise ValueError(f"invalid CRYCHIC score view: {dataset_id}/{target}")
            view["condition"] = view["context_json"].map(_condition)
            base = view.loc[
                :,
                [
                    "sample_id",
                    "subject_id",
                    "condition",
                    *EDGE_KEYS,
                    "ligand",
                    "receptor",
                ],
            ].copy()
            generated = dict(components)
            for generator, column in generated.items():
                selected = base.copy()
                selected["score"] = pd.to_numeric(view[column], errors="coerce")
                selected["score_generator"] = generator
                selected["status"] = np.where(
                    np.isfinite(selected["score"]), "observed", "not_estimable"
                )
                selected["reason_code"] = np.where(
                    selected["status"].eq("observed"),
                    "component_ledger_observed",
                    "component_score_nonfinite",
                )
                selected["dataset_id"] = dataset_id
                selected["scenario"] = meta["scenario"]
                selected["seed"] = meta["seed"]
                selected["contrast"] = contrast
                selected["target"] = target
                selected["reference"] = reference
                frames.append(selected.loc[:, SCORE_COLUMNS])
            selected = base.copy()
            selected["score"] = (
                pd.to_numeric(view["availability"], errors="coerce")
                * pd.to_numeric(view["prior_quality"], errors="coerce")
            )
            selected["score_generator"] = "crychic_availability_prior"
            selected["status"] = np.where(
                np.isfinite(selected["score"]), "observed", "not_estimable"
            )
            selected["reason_code"] = np.where(
                selected["status"].eq("observed"),
                "component_ledger_observed",
                "component_score_nonfinite",
            )
            selected["dataset_id"] = dataset_id
            selected["scenario"] = meta["scenario"]
            selected["seed"] = meta["seed"]
            selected["contrast"] = contrast
            selected["target"] = target
            selected["reference"] = reference
            frames.append(selected.loc[:, SCORE_COLUMNS])
        elapsed = float(manifest.get("elapsed_seconds", math.nan))
        for generator in SCORE_GENERATORS[:5]:
            runtime_rows.append(
                _generator_runtime_row(
                    dataset_id=dataset_id,
                    generator=generator,
                    elapsed_seconds=elapsed,
                    accounting="shared_parent_run_not_additive",
                    source=str(run_dir),
                )
            )
    scores = pd.concat(frames, ignore_index=True)
    return scores, pd.DataFrame.from_records(runtime_rows), provenance


def _load_external_scores(
    panel_dir: Path,
    *,
    method: str,
    generator: str,
    metadata: Mapping[str, Mapping[str, Any]],
    lower_is_stronger: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel_manifest = _read_json(panel_dir / "panel_manifest.json")
    if panel_manifest.get("status") != "complete":
        raise ValueError(f"external panel is incomplete: {panel_dir}")
    frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for run_dir in sorted(panel_dir.glob(f"{method}__*")):
        manifest = _read_json(run_dir / "manifest.json")
        if manifest.get("status") != "complete":
            raise ValueError(f"external adapter is incomplete: {run_dir}")
        dataset_id = str(manifest.get("dataset_id", ""))
        if dataset_id not in metadata or dataset_id in seen:
            raise ValueError(f"unexpected or duplicate external dataset: {dataset_id}")
        seen.add(dataset_id)
        table = pd.read_parquet(_bound_table(run_dir, manifest))
        table["condition"] = table["context_json"].map(_condition)
        score = pd.to_numeric(table["score"], errors="coerce")
        if lower_is_stronger:
            score = 1.0 - score
        selected = table.loc[
            :,
            ["sample_id", "subject_id", "condition", *EDGE_KEYS, "ligand", "receptor"],
        ].copy()
        selected["score"] = score
        selected["score_generator"] = generator
        observed = table["status"].astype(str).eq("ok") & np.isfinite(score)
        selected["status"] = np.where(observed, "observed", "not_estimable")
        selected["reason_code"] = np.where(
            observed,
            "external_score_observed",
            table["reason_code"].fillna("external_score_not_returned").astype(str),
        )
        selected["dataset_id"] = dataset_id
        selected["scenario"] = metadata[dataset_id]["scenario"]
        selected["seed"] = metadata[dataset_id]["seed"]
        frames.append(_duplicate_contrasts(selected).loc[:, SCORE_COLUMNS])
        runtime_rows.append(
            _generator_runtime_row(
                dataset_id=dataset_id,
                generator=generator,
                elapsed_seconds=float(manifest.get("elapsed_seconds", math.nan)),
                accounting="complete_adapter_run",
                source=str(run_dir),
            )
        )
    if seen != set(metadata):
        raise ValueError(
            f"{method} coverage differs: missing={sorted(set(metadata) - seen)}"
        )
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame.from_records(runtime_rows),
        {
            "directory": str(panel_dir),
            "manifest_sha256": sha256_file(panel_dir / "panel_manifest.json"),
            "adapter_commit": panel_manifest.get("adapter_code", {}).get("commit"),
        },
    )


def _bound_fixture_input(fixture_dir: Path, relative_manifest: str) -> Path:
    manifest_path = fixture_dir / relative_manifest
    manifest = _read_json(manifest_path)
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"fixture input binding is absent: {manifest_path}")
    path = manifest_path.parent / str(output.get("filename"))
    if not path.is_file() or sha256_file(path) != output.get("sha256"):
        raise ValueError(f"fixture input checksum mismatch: {path}")
    return path


def _simple_scores_for_dataset(
    input_path: Path,
    resource: pd.DataFrame,
    *,
    dataset_id: str,
    scenario: str,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    adata = ad.read_h5ad(input_path)
    required_obs = {"sample_id", "subject_id", "condition", "cell_type"}
    if missing := required_obs.difference(adata.obs.columns):
        raise ValueError(f"simple score input lacks obs fields: {sorted(missing)}")
    genes = sorted(set(resource["ligand"]) | set(resource["receptor"]))
    missing_genes = set(genes).difference(adata.var_names.astype(str))
    if missing_genes:
        raise ValueError(f"simple score genes are absent: {sorted(missing_genes)}")
    gene_index = [int(adata.var_names.get_loc(gene)) for gene in genes]
    expression = adata.X[:, gene_index]
    samples = sorted(adata.obs["sample_id"].astype(str).unique())
    cell_types = sorted(adata.obs["cell_type"].astype(str).unique())
    means: dict[tuple[str, str], np.ndarray] = {}
    for sample_id in samples:
        sample_mask = adata.obs["sample_id"].astype(str).eq(sample_id).to_numpy()
        for cell_type in cell_types:
            cell_type_mask = (
                adata.obs["cell_type"].astype(str).eq(cell_type).to_numpy()
            )
            mask = sample_mask & cell_type_mask
            if not mask.any():
                means[(sample_id, cell_type)] = np.full(len(genes), np.nan)
                continue
            values = expression[mask]
            average = values.mean(axis=0)
            means[(sample_id, cell_type)] = np.asarray(average).ravel()
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    sample_metadata = (
        adata.obs.loc[:, ["sample_id", "subject_id", "condition"]]
        .astype(str)
        .drop_duplicates()
        .set_index("sample_id")
    )
    if sample_metadata.index.duplicated().any():
        raise ValueError("sample metadata is not one-to-one")
    rows: list[dict[str, object]] = []
    for sample_id in samples:
        for sender in cell_types:
            for receiver in cell_types:
                for interaction in resource.itertuples(index=False):
                    ligand = str(interaction.ligand)
                    receptor = str(interaction.receptor)
                    ligand_value = means[(sample_id, sender)][gene_lookup[ligand]]
                    receptor_value = means[(sample_id, receiver)][gene_lookup[receptor]]
                    for generator, score in (
                        ("simple_lr_mean", (ligand_value + receptor_value) / 2.0),
                        ("simple_lr_product", ligand_value * receptor_value),
                    ):
                        observed = bool(np.isfinite(score))
                        rows.append(
                            {
                                "dataset_id": dataset_id,
                                "scenario": scenario,
                                "seed": seed,
                                "score_generator": generator,
                                "sample_id": sample_id,
                                "subject_id": sample_metadata.loc[
                                    sample_id, "subject_id"
                                ],
                                "condition": sample_metadata.loc[
                                    sample_id, "condition"
                                ],
                                "sender": sender,
                                "receiver": receiver,
                                "interaction_id": str(
                                    interaction.harmonized_interaction_id
                                ),
                                "ligand": ligand,
                                "receptor": receptor,
                                "score": float(score) if observed else math.nan,
                                "status": "observed" if observed else "not_estimable",
                                "reason_code": (
                                    "log1p_mean_expression"
                                    if observed
                                    else "structural_cell_type_absence"
                                ),
                            }
                        )
    table = _duplicate_contrasts(pd.DataFrame.from_records(rows))
    return table.loc[:, SCORE_COLUMNS], time.perf_counter() - started


def _load_simple_scores(
    fixture_dir: Path,
    fixture: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    resource_record = fixture.get("resource")
    if not isinstance(resource_record, Mapping):
        raise ValueError("fixture resource provenance is absent")
    resource_path = fixture_dir / str(resource_record.get("filename"))
    if not resource_path.is_file() or sha256_file(resource_path) != resource_record.get(
        "sha256"
    ):
        raise ValueError("fixture resource checksum mismatch")
    resource = pd.read_csv(resource_path, sep="\t")
    frames: list[pd.DataFrame] = []
    runtimes: list[dict[str, object]] = []
    for dataset_id, meta in sorted(metadata.items()):
        input_path = _bound_fixture_input(fixture_dir, str(meta["manifest"]))
        table, elapsed = _simple_scores_for_dataset(
            input_path,
            resource,
            dataset_id=dataset_id,
            scenario=str(meta["scenario"]),
            seed=int(meta["seed"]),
        )
        frames.append(table)
        for generator in ("simple_lr_mean", "simple_lr_product"):
            runtimes.append(
                _generator_runtime_row(
                    dataset_id=dataset_id,
                    generator=generator,
                    elapsed_seconds=elapsed,
                    accounting="shared_expression_aggregation_not_additive",
                    source=str(input_path),
                )
            )
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame.from_records(runtimes),
        {"path": str(resource_path), "sha256": sha256_file(resource_path)},
    )


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    observed = np.clip(values[valid], 0.0, 1.0)
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result[valid] = restored
    return result


def _p_from_t(effect: float, standard_error: float, df: int) -> float:
    if not math.isfinite(standard_error) or standard_error < 0.0 or df < 1:
        return math.nan
    if standard_error == 0.0:
        return 0.0 if effect != 0.0 else 1.0
    return float(2.0 * stats.t.sf(abs(effect / standard_error), df=df))


def _fit_pairwise_event(
    values: Sequence[float], target_indicator: Sequence[int], *, robust: bool
) -> tuple[float, float, float]:
    response = np.asarray(values, dtype=float)
    indicator = np.asarray(target_indicator, dtype=float)
    design = np.column_stack((np.ones(len(indicator)), indicator))
    inverse = np.linalg.inv(design.T @ design)
    coefficient = inverse @ design.T @ response
    fitted = design @ coefficient
    residual = response - fitted
    df = len(response) - design.shape[1]
    contrast = np.array([0.0, 1.0])
    if robust:
        leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
        projection = design @ inverse @ contrast
        denominator = np.maximum(1.0 - leverage, np.finfo(float).eps)
        variance = float(np.sum(projection**2 * residual**2 / denominator))
    else:
        sigma_squared = float(np.sum(residual**2) / df)
        variance = float(sigma_squared * (contrast @ inverse @ contrast))
    standard_error = math.sqrt(max(variance, 0.0))
    effect = float(coefficient[1])
    return effect, standard_error, _p_from_t(effect, standard_error, df)


def _fit_engine_arm(
    arm: pd.DataFrame, *, engine: str, min_subjects: int = 4
) -> tuple[pd.DataFrame, float]:
    robust = engine == "clustered_cr2"
    if engine not in {"sample_glm", "clustered_cr2"}:
        raise ValueError(f"unsupported common engine: {engine}")
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    identity = {
        "dataset_id": str(arm["dataset_id"].iloc[0]),
        "seed": int(arm["seed"].iloc[0]),
        "score_generator": str(arm["score_generator"].iloc[0]),
        "differential_engine": engine,
        "contrast": str(arm["contrast"].iloc[0]),
    }
    target = str(arm["target"].iloc[0])
    reference = str(arm["reference"].iloc[0])
    for raw_edge, event in arm.groupby(list(EDGE_KEYS), observed=True, sort=True):
        edge = dict(zip(EDGE_KEYS, raw_edge, strict=True))
        eligible = event.loc[
            event["status"].eq("observed")
            & np.isfinite(pd.to_numeric(event["score"], errors="coerce"))
            & event["condition"].isin([reference, target])
        ].copy()
        if eligible["sample_id"].duplicated().any():
            raise ValueError(f"duplicate event/sample score: {identity | edge}")
        counts = eligible.groupby("condition", observed=True)["subject_id"].nunique()
        if (
            counts.get(reference, 0) < min_subjects
            or counts.get(target, 0) < min_subjects
        ):
            records.append(
                identity
                | edge
                | {
                    "effect": math.nan,
                    "standard_error": math.nan,
                    "p_value": math.nan,
                    "status": "not_estimable",
                    "reason_code": "fewer_than_four_subjects_per_condition",
                }
            )
            continue
        indicator = eligible["condition"].eq(target).astype(int).to_numpy()
        effect, standard_error, p_value = _fit_pairwise_event(
            eligible["score"].to_numpy(dtype=float), indicator, robust=robust
        )
        records.append(
            identity
            | edge
            | {
                "effect": effect,
                "standard_error": standard_error,
                "p_value": p_value,
                "status": "observed",
                "reason_code": (
                    "singleton_subject_cr2_hc2_equivalent"
                    if robust
                    else "ordinary_sample_level_glm"
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    result["q_value"] = _bh_adjust(result["p_value"].to_numpy(dtype=float))
    elapsed = time.perf_counter() - started
    result["engine_elapsed_seconds"] = elapsed
    result["bootstrap_replicates"] = 0
    return result, elapsed


def _fit_common_engines(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    grouped = scores.groupby(
        ["dataset_id", "score_generator", "contrast"], observed=True, sort=True
    )
    for (dataset_id, generator, contrast), arm in grouped:
        for engine in ("sample_glm", "clustered_cr2"):
            fitted, elapsed = _fit_engine_arm(arm, engine=engine)
            frames.append(fitted)
            runtime_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": dataset_id,
                    "score_generator": generator,
                    "differential_engine": engine,
                    "contrast": contrast,
                    "elapsed_seconds": elapsed,
                    "peak_rss_mb": math.nan,
                    "status": "complete",
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame.from_records(runtime_rows)


def _process_tree_rss(process: psutil.Process) -> int:
    total = 0
    try:
        descendants = [process, *process.children(recursive=True)]
    except psutil.Error:
        return 0
    for descendant in descendants:
        try:
            total += int(descendant.memory_info().rss)
        except psutil.Error:
            continue
    return total


def _run_staccato_task(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    *,
    rscript: Path,
    source: Path,
    r_library: Path,
    bootstrap_replicates: int,
    bootstrap_cores: int,
    memory_start_fraction: float,
    memory_hard_fraction: float,
) -> dict[str, object]:
    while psutil.virtual_memory().percent / 100.0 >= memory_start_fraction:
        time.sleep(2.0)
    command = [
        str(rscript),
        str(Path(__file__).with_name("score_engine_staccato_worker.R")),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--staccato-source",
        str(source),
        "--bootstrap-replicates",
        str(bootstrap_replicates),
        "--bootstrap-cores",
        str(bootstrap_cores),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "R_LIBS_USER": str(r_library),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    peak = 0
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"command": command}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        tracked = psutil.Process(process.pid)
        while process.poll() is None:
            peak = max(peak, _process_tree_rss(tracked))
            if psutil.virtual_memory().percent / 100.0 >= memory_hard_fraction:
                process.terminate()
                raise MemoryError("STACCato campaign reached the hard memory limit")
            time.sleep(0.25)
    if process.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"STACCato task failed; inspect {log_path}")
    return {
        "dataset_id": input_path.name.removesuffix(".tsv.gz"),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mb": peak / (1024**2),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "log": str(log_path),
        "status": "complete",
    }


def _fit_staccato(
    scores: pd.DataFrame,
    stage: Path,
    *,
    rscript: Path,
    source: Path,
    r_library: Path,
    bootstrap_replicates: int,
    bootstrap_cores: int,
    max_workers: int,
    memory_start_fraction: float,
    memory_hard_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_root = stage / "staccato_tasks"
    input_root = task_root / "inputs"
    output_root = task_root / "outputs"
    log_root = task_root / "logs"
    for directory in (input_root, output_root, log_root):
        directory.mkdir(parents=True, exist_ok=True)
    task_specs: list[tuple[Path, Path, Path]] = []
    for dataset_id, selected in scores.groupby("dataset_id", observed=True, sort=True):
        input_path = input_root / f"{dataset_id}.tsv.gz"
        selected.to_csv(input_path, sep="\t", index=False, compression="gzip")
        task_specs.append(
            (
                input_path,
                output_root / f"{dataset_id}.tsv",
                log_root / f"{dataset_id}.log",
            )
        )
    task_records: list[dict[str, object]] = []
    futures: dict[Future[dict[str, object]], tuple[Path, Path, Path]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for input_path, output_path, log_path in task_specs:
            futures[
                executor.submit(
                    _run_staccato_task,
                    input_path,
                    output_path,
                    log_path,
                    rscript=rscript,
                    source=source,
                    r_library=r_library,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_cores=bootstrap_cores,
                    memory_start_fraction=memory_start_fraction,
                    memory_hard_fraction=memory_hard_fraction,
                )
            ] = (input_path, output_path, log_path)
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            task_records.append(record)
            print(
                f"[STACCato {completed}/{len(futures)}] "
                f"{record['dataset_id']} {record['elapsed_seconds']:.1f}s",
                flush=True,
            )
    frames = [pd.read_csv(output_path, sep="\t") for _, output_path, _ in task_specs]
    effects = pd.concat(frames, ignore_index=True)
    expected_rows = scores["dataset_id"].nunique() * len(SCORE_GENERATORS) * 3 * 45
    if len(effects) != expected_rows:
        raise ValueError(
            "STACCato row count differs: "
            f"observed={len(effects)} expected={expected_rows}"
        )
    runtime = (
        effects.groupby(
            ["dataset_id", "score_generator", "differential_engine", "contrast"],
            observed=True,
            sort=True,
        )
        .agg(
            elapsed_seconds=("engine_elapsed_seconds", "max"),
            status=(
                "status",
                lambda values: (
                    "complete" if (values == "observed").any() else "not_estimable"
                ),
            ),
        )
        .reset_index()
    )
    runtime["schema_version"] = SCHEMA_VERSION
    runtime["peak_rss_mb"] = math.nan
    return effects, runtime, pd.DataFrame.from_records(task_records)


def _truth_table(
    fixture_dir: Path, fixture: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outputs = fixture.get("outputs")
    truth_record = outputs.get("truth") if isinstance(outputs, Mapping) else None
    if not isinstance(truth_record, Mapping):
        raise ValueError("fixture truth provenance is absent")
    path = fixture_dir / str(truth_record.get("filename"))
    if not path.is_file() or sha256_file(path) != truth_record.get("sha256"):
        raise ValueError("fixture truth checksum mismatch")
    truth = pd.read_csv(path, sep="\t")
    return truth, {"path": str(path), "sha256": sha256_file(path), "rows": len(truth)}


def _attach_truth(effects: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_id", "contrast", *EDGE_KEYS]
    truth_columns = [
        *keys,
        "scenario",
        "seed",
        "truth_effect",
        "truth_label",
        "truth_direction",
    ]
    if truth.duplicated(keys).any():
        raise ValueError("truth event keys are not unique")
    result = effects.merge(
        truth.loc[:, truth_columns],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_truth"),
    )
    if result["truth_label"].isna().any():
        raise ValueError("engine output contains events outside truth")
    if "seed_truth" in result:
        if not result["seed"].eq(result["seed_truth"]).all():
            raise ValueError("engine and truth seeds differ")
        result = result.drop(columns="seed_truth")
    return result


def _safe_detection_metrics(
    labels: np.ndarray, evidence: np.ndarray
) -> tuple[float, float]:
    valid = np.isfinite(evidence)
    labels = labels[valid]
    evidence = evidence[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return math.nan, math.nan
    return (
        float(average_precision_score(labels, evidence)),
        float(roc_auc_score(labels, evidence)),
    )


def _dataset_metrics(effects: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_keys = ["dataset_id", "score_generator", "differential_engine"]
    for identity, arm in effects.groupby(group_keys, observed=True, sort=True):
        dataset_id, generator, engine = identity
        observed = arm["status"].astype(str).eq("observed") & np.isfinite(
            pd.to_numeric(arm["effect"], errors="coerce")
        )
        selected = arm.loc[observed].copy()
        coverage = float(observed.mean())
        evidence = np.full(len(selected), np.nan)
        finite_p = np.isfinite(pd.to_numeric(selected["p_value"], errors="coerce"))
        evidence[finite_p] = -np.log10(
            np.maximum(
                selected.loc[finite_p, "p_value"].to_numpy(dtype=float),
                np.finfo(float).tiny,
            )
        )
        auprc, auroc = _safe_detection_metrics(
            selected["truth_label"].to_numpy(dtype=int), evidence
        )
        prediction = selected["effect"].to_numpy(dtype=float)
        truth_effect = selected["truth_effect"].to_numpy(dtype=float)
        native_rmse = (
            float(np.sqrt(np.mean((prediction - truth_effect) ** 2)))
            if len(prediction)
            else math.nan
        )
        denominator = float(prediction @ prediction)
        scale = (
            max(0.0, float(prediction @ truth_effect) / denominator)
            if denominator > 0.0
            else 0.0
        )
        calibrated_rmse = (
            float(np.sqrt(np.mean((scale * prediction - truth_effect) ** 2)))
            if len(prediction)
            else math.nan
        )
        active = selected["truth_direction"].astype(int).ne(0)
        direction_accuracy = (
            float(
                (
                    np.sign(selected.loc[active, "effect"].to_numpy(dtype=float))
                    == selected.loc[active, "truth_direction"].to_numpy(dtype=int)
                ).mean()
            )
            if active.any()
            else math.nan
        )
        formal = np.isfinite(pd.to_numeric(selected["q_value"], errors="coerce"))
        discoveries = formal & selected["q_value"].astype(float).le(0.05)
        discovery_count = int(discoveries.sum())
        false_discoveries = int(
            (discoveries & selected["truth_label"].astype(int).eq(0)).sum()
        )
        empirical_fdr = (
            false_discoveries / discovery_count
            if discovery_count
            else 0.0
            if formal.any()
            else math.nan
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "scenario": str(arm["scenario"].iloc[0]),
                "seed": int(arm["seed"].iloc[0]),
                "score_generator": generator,
                "differential_engine": engine,
                "event_contrast_cells": len(arm),
                "observed_event_contrast_cells": int(observed.sum()),
                "coverage": coverage,
                "failure_rate": 1.0 - coverage,
                "contrast_event_auprc": auprc,
                "contrast_event_auroc": auroc,
                "native_scale_effect_rmse": native_rmse,
                "nonnegative_scale_calibrated_rmse": calibrated_rmse,
                "effect_scale_factor": scale,
                "direction_accuracy": direction_accuracy,
                "formal_q_coverage": float(formal.mean()) if len(formal) else 0.0,
                "discoveries_q05": discovery_count,
                "false_discoveries_q05": false_discoveries,
                "empirical_fdr_q05": empirical_fdr,
                "null_false_positive_rate_q05": (
                    float(discoveries.mean())
                    if str(arm["scenario"].iloc[0]) == "global_null"
                    and formal.any()
                    else math.nan
                ),
                "engine_elapsed_seconds": float(
                    arm["engine_elapsed_seconds"].max()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _bootstrap_mean_ci(values: pd.Series, *, seed: int) -> tuple[float, float]:
    observed = values.dropna().to_numpy(dtype=float)
    if len(observed) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(observed, size=(10_000, len(observed)), replace=True)
    lower, upper = np.quantile(sampled.mean(axis=1), (0.025, 0.975))
    return float(lower), float(upper)


def _arm_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    active = metrics.loc[metrics["scenario"].eq("active")].copy()
    expected = active["dataset_id"].nunique()
    summary = (
        active.groupby(["score_generator", "differential_engine"], observed=True)
        .agg(
            active_datasets=("dataset_id", "nunique"),
            mean_auprc=("contrast_event_auprc", "mean"),
            mean_auroc=("contrast_event_auroc", "mean"),
            mean_calibrated_rmse=("nonnegative_scale_calibrated_rmse", "mean"),
            mean_direction_accuracy=("direction_accuracy", "mean"),
            mean_empirical_fdr=("empirical_fdr_q05", "mean"),
            mean_coverage=("coverage", "mean"),
            mean_failure_rate=("failure_rate", "mean"),
            mean_engine_seconds=("engine_elapsed_seconds", "mean"),
        )
        .reset_index()
    )
    null = metrics.loc[metrics["scenario"].eq("global_null")]
    null_summary = (
        null.groupby(["score_generator", "differential_engine"], observed=True)
        .agg(
            mean_null_false_positive_rate=("null_false_positive_rate_q05", "mean"),
            null_datasets=("dataset_id", "nunique"),
        )
        .reset_index()
    )
    summary = summary.merge(
        null_summary,
        on=["score_generator", "differential_engine"],
        how="left",
        validate="one_to_one",
    )
    ci_records: list[dict[str, object]] = []
    for index, (identity, selected) in enumerate(
        active.groupby(["score_generator", "differential_engine"], observed=True)
    ):
        lower, upper = _bootstrap_mean_ci(
            selected["contrast_event_auprc"], seed=20260724 + index
        )
        ci_records.append(
            {
                "score_generator": identity[0],
                "differential_engine": identity[1],
                "auprc_ci_lower": lower,
                "auprc_ci_upper": upper,
            }
        )
    summary = summary.merge(
        pd.DataFrame.from_records(ci_records),
        on=["score_generator", "differential_engine"],
        how="left",
        validate="one_to_one",
    )
    summary["rank_eligible"] = (
        summary["active_datasets"].eq(expected) & summary["mean_coverage"].ge(0.80)
    )
    for column, ascending in (
        ("mean_auprc", False),
        ("mean_auroc", False),
        ("mean_calibrated_rmse", True),
        ("mean_direction_accuracy", False),
    ):
        summary[f"{column}_rank"] = (
            summary[column]
            .where(summary["rank_eligible"])
            .rank(method="min", ascending=ascending)
        )
    summary["primary_rank"] = summary["mean_auprc_rank"]
    return summary.sort_values(
        ["primary_rank", "score_generator", "differential_engine"],
        na_position="last",
        ignore_index=True,
    )


def _eligibility_matrix(effects: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for generator in SCORE_GENERATORS:
        for engine in ALL_ENGINES:
            selected = effects.loc[
                effects["score_generator"].eq(generator)
                & effects["differential_engine"].eq(engine)
            ]
            if engine == "crychic_common_functional_oof":
                status = "not_estimable"
                reason = "independent_subject_groups_violate_paired_oof_contract"
                observed_fraction = 0.0
            elif engine == "scseqcommdiff_multi_sample":
                status = "not_estimable"
                reason = "public_engine_has_no_external_score_injection_interface"
                observed_fraction = 0.0
            else:
                observed_fraction = (
                    float(selected["status"].astype(str).eq("observed").mean())
                    if len(selected)
                    else 0.0
                )
                status = (
                    "observed"
                    if observed_fraction == 1.0
                    else "partially_observed"
                    if observed_fraction > 0.0
                    else "not_estimable"
                )
                reason = (
                    "estimable_public_score_tensor_interface"
                    if observed_fraction > 0.0
                    else "no_estimable_event_tensor"
                )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "score_generator": generator,
                    "differential_engine": engine,
                    "status": status,
                    "reason_code": reason,
                    "observed_event_fraction": observed_fraction,
                }
            )
    return pd.DataFrame.from_records(records)


def _report(summary: pd.DataFrame, matrix: pd.DataFrame) -> str:
    ranked = summary.loc[summary["rank_eligible"]].head(15)
    lines = [
        "# Score generator x differential engine crossover",
        "",
        "This is the frozen 20-seed, 45-event, three-condition H-common panel. "
        "All effects use biological samples as replicates. Structural missingness "
        "remains missing and is never converted to zero.",
        "",
        "## Estimability",
        "",
        f"- Matrix cells: {len(matrix)}",
        f"- Observed: {(matrix['status'] == 'observed').sum()}",
        f"- Partially observed: {(matrix['status'] == 'partially_observed').sum()}",
        f"- Not estimable: {(matrix['status'] == 'not_estimable').sum()}",
        "- CRYCHIC common-functional OOF is NE because subjects are independent "
        "across conditions.",
        "- scSeqCommDiff crossover cells are NE because its public engine cannot "
        "accept externally generated score tensors; its native end-to-end arm is "
        "reported separately.",
        "",
        "## Rank-eligible active arms",
        "",
        "| Rank | Score generator | Engine | AUPRC | AUROC | Direction | Coverage |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in ranked.itertuples(index=False):
        lines.append(
            f"| {int(row.primary_rank)} | {row.score_generator} | "
            f"{row.differential_engine} | {row.mean_auprc:.4f} | "
            f"{row.mean_auroc:.4f} | {row.mean_direction_accuracy:.4f} | "
            f"{row.mean_coverage:.4f} |"
        )
    lines.extend(
        [
            "",
            "AUPRC/AUROC use pairwise event cells and -log10 formal p-values. "
            "Native-scale RMSE is retained but not ranked across score generators; "
            "the comparable RMSE is calibrated by one non-negative scale factor.",
            "",
        ]
    )
    return "\n".join(lines)


def run_crossover(
    *,
    fixture_dir: Path,
    crychic_runs: Path,
    external_panel: Path,
    cellphonedb_panel: Path,
    output_dir: Path,
    rscript: Path,
    staccato_source: Path,
    r_library: Path,
    bootstrap_replicates: int,
    bootstrap_cores: int,
    staccato_workers: int,
    memory_start_fraction: float,
    memory_hard_fraction: float,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    code = git_metadata(repo_root)
    if code.get("dirty") is not False or not code.get("commit"):
        raise ValueError(f"formal crossover requires a clean worktree: {code}")
    for path in (rscript, staccato_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not r_library.is_dir():
        raise FileNotFoundError(r_library)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    fixture_dir = fixture_dir.resolve()
    fixture = _read_json(fixture_dir / "manifest.json")
    if fixture.get("schema_version") != "crychic-three-group-fixture-v2":
        raise ValueError("the crossover requires fixture v2")
    metadata = _dataset_metadata(fixture)
    if len(metadata) != 40:
        raise ValueError(
            f"the frozen crossover requires 40 datasets, got {len(metadata)}"
        )
    started = time.perf_counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        internal, runtime_internal, internal_provenance = _load_internal_scores(
            crychic_runs.resolve(), metadata
        )
        liana, runtime_liana, liana_provenance = _load_external_scores(
            external_panel.resolve(),
            method="liana",
            generator="liana_magnitude",
            metadata=metadata,
            lower_is_stronger=True,
        )
        cellchat, runtime_cellchat, cellchat_provenance = _load_external_scores(
            external_panel.resolve(),
            method="cellchat",
            generator="cellchat_probability",
            metadata=metadata,
        )
        cellphonedb, runtime_cellphonedb, cellphonedb_provenance = (
            _load_external_scores(
                cellphonedb_panel.resolve(),
                method="cellphonedb",
                generator="cellphonedb_mean",
                metadata=metadata,
            )
        )
        simple, runtime_simple, resource_provenance = _load_simple_scores(
            fixture_dir, fixture, metadata
        )
        scores = pd.concat(
            [internal, liana, cellchat, cellphonedb, simple], ignore_index=True
        )
        if set(scores["score_generator"].astype(str)) != set(SCORE_GENERATORS):
            raise ValueError(
                "score generator coverage differs from the frozen contract"
            )
        score_keys = [
            "dataset_id",
            "score_generator",
            "contrast",
            "sample_id",
            *EDGE_KEYS,
        ]
        if scores.duplicated(score_keys).any():
            raise ValueError("score generator table contains duplicate identities")
        scores.to_parquet(stage / "generator_scores.parquet", compression="zstd")
        generator_runtime = pd.concat(
            [
                runtime_internal,
                runtime_liana,
                runtime_cellchat,
                runtime_cellphonedb,
                runtime_simple,
            ],
            ignore_index=True,
        )
        generator_runtime.to_csv(
            stage / "generator_runtime.tsv", sep="\t", index=False
        )

        common_effects, common_runtime = _fit_common_engines(scores)
        staccato_effects, staccato_runtime, staccato_tasks = _fit_staccato(
            scores,
            stage,
            rscript=rscript,
            source=staccato_source,
            r_library=r_library,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_cores=bootstrap_cores,
            max_workers=staccato_workers,
            memory_start_fraction=memory_start_fraction,
            memory_hard_fraction=memory_hard_fraction,
        )
        effects = pd.concat([common_effects, staccato_effects], ignore_index=True)
        truth, truth_provenance = _truth_table(fixture_dir, fixture)
        effects = _attach_truth(effects, truth)
        effects.to_parquet(stage / "event_effects.parquet", compression="zstd")
        engine_runtime = pd.concat(
            [common_runtime, staccato_runtime], ignore_index=True
        )
        engine_runtime.to_csv(stage / "engine_runtime.tsv", sep="\t", index=False)
        staccato_tasks.to_csv(
            stage / "staccato_task_runtime.tsv", sep="\t", index=False
        )

        metrics = _dataset_metrics(effects)
        summary = _arm_summary(metrics)
        matrix = _eligibility_matrix(effects)
        metrics.to_csv(stage / "dataset_metrics.tsv", sep="\t", index=False)
        summary.to_csv(stage / "arm_summary.tsv", sep="\t", index=False)
        matrix.to_csv(stage / "eligibility_matrix.tsv", sep="\t", index=False)
        (stage / "REPORT.md").write_text(
            _report(summary, matrix), encoding="utf-8"
        )

        primary_outputs = (
            "generator_scores.parquet",
            "generator_runtime.tsv",
            "event_effects.parquet",
            "engine_runtime.tsv",
            "staccato_task_runtime.tsv",
            "dataset_metrics.tsv",
            "arm_summary.tsv",
            "eligibility_matrix.tsv",
            "REPORT.md",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "code": code,
            "fixture": {
                "manifest": str(fixture_dir / "manifest.json"),
                "manifest_sha256": sha256_file(fixture_dir / "manifest.json"),
                "code": fixture.get("code"),
            },
            "sources": {
                "truth": truth_provenance,
                "resource": resource_provenance,
                "crychic": internal_provenance,
                "liana_panel": liana_provenance,
                "cellchat_panel": cellchat_provenance,
                "cellphonedb_panel": cellphonedb_provenance,
                "staccato_source": {
                    "path": str(staccato_source),
                    "sha256": sha256_file(staccato_source),
                },
            },
            "design": {
                "replicates": 20,
                "datasets": len(metadata),
                "active_datasets": sum(
                    item["scenario"] == "active" for item in metadata.values()
                ),
                "global_null_datasets": sum(
                    item["scenario"] == "global_null" for item in metadata.values()
                ),
                "score_generators": list(SCORE_GENERATORS),
                "differential_engines": list(ALL_ENGINES),
                "contrasts": CONTRASTS,
                "events_per_contrast": 45,
                "structural_missingness": "retained_as_not_estimable_never_zero",
                "staccato_bootstrap_replicates": bootstrap_replicates,
                "staccato_bootstrap_cores": bootstrap_cores,
                "staccato_workers": staccato_workers,
            },
            "counts": {
                "generator_score_rows": len(scores),
                "event_effect_rows": len(effects),
                "dataset_metric_rows": len(metrics),
                "matrix_cells": len(matrix),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "outputs": {
                name: {
                    "sha256": sha256_file(stage / name),
                    "bytes": (stage / name).stat().st_size,
                }
                for name in primary_outputs
            },
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--crychic-runs", required=True, type=Path)
    parser.add_argument("--external-panel", required=True, type=Path)
    parser.add_argument("--cellphonedb-panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--staccato-source", required=True, type=Path)
    parser.add_argument("--r-library", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--bootstrap-cores", type=int, default=4)
    parser.add_argument("--staccato-workers", type=int, default=8)
    parser.add_argument("--memory-start-fraction", type=float, default=0.70)
    parser.add_argument("--memory-hard-fraction", type=float, default=0.80)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.bootstrap_replicates <= 10_000:
        raise ValueError("bootstrap_replicates must lie in [1, 10000]")
    if not 1 <= args.bootstrap_cores <= 16:
        raise ValueError("bootstrap_cores must lie in [1, 16]")
    if not 1 <= args.staccato_workers <= 16:
        raise ValueError("staccato_workers must lie in [1, 16]")
    if not (
        0.2
        <= args.memory_start_fraction
        < args.memory_hard_fraction
        <= 0.80
    ):
        raise ValueError("memory fractions must satisfy 0.2 <= start < hard <= 0.8")
    manifest = run_crossover(
        fixture_dir=args.fixture_dir,
        crychic_runs=args.crychic_runs,
        external_panel=args.external_panel,
        cellphonedb_panel=args.cellphonedb_panel,
        output_dir=args.output_dir,
        rscript=args.rscript,
        staccato_source=args.staccato_source,
        r_library=args.r_library,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_cores=args.bootstrap_cores,
        staccato_workers=args.staccato_workers,
        memory_start_fraction=args.memory_start_fraction,
        memory_hard_fraction=args.memory_hard_fraction,
    )
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
