"""Evaluate fold-training residualized M2 sender coupling against decoys."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.comprehensive.evaluate_m1_mechanism_concordance import (
    _program_definition,
    _program_sample_table,
    signed_geometric_program_concordance,
)
from benchmarks.comprehensive.generate_m2_sender_coupling_fixture import (
    CANDIDATE_ROLES,
)
from benchmarks.comprehensive.generate_m2_sender_coupling_fixture import (
    SCHEMA_VERSION as FIXTURE_SCHEMA,
)
from crychic.availability import AvailabilityParameters
from crychic.core import CrychicConfig
from crychic.scoring import build_absolute_activity_heads
from crychic.sender import (
    RESIDUALIZED_COUPLING_VERSION,
    ResidualizedCouplingSpec,
    fit_residualized_sender_coupling,
)
from crychic.workflow import fit_baseline

SCHEMA_VERSION = "crychic-m2-sender-coupling-evaluation-v1"
CONFIG_SCHEMA_VERSION = "crychic-suggestions-next-m2-config-v1"
M0_METHOD = "crychic_m0_mean_ligand_contrast"
M1_METHOD = "crychic_m1_group_program_concordance"
RAW_METHOD = "crychic_unadjusted_sender_program_coupling"
M2_METHOD = "crychic_m2_residualized_sender_program_coupling"
METHOD_SCORE_COLUMNS: Mapping[str, str] = {
    M0_METHOD: "m0_ranking_score",
    M1_METHOD: "m1_ranking_score",
    RAW_METHOD: "raw_coupling_score",
    M2_METHOD: "m2_coupling_score",
}
NUMERICAL_COVARIATES = (
    "batch_contrast",
    "composition_design_contrast",
    "sender_proportion_effect",
)
FOLD_NAMESPACE = "crychic:m2-sender-coupling:subject-fold:v1"


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


def _validated_config(
    path: Path, *, role: str, fixture_manifest: Path
) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("M2 configuration schema is unsupported")
    expected_candidate = {
        "name": M2_METHOD,
        "score_version": RESIDUALIZED_COUPLING_VERSION,
        "input_unit": "subject_level_paired_condition_contrast",
        "sender_input": "m0_ligand_absolute_evidence",
        "receiver_input": "m1_frozen_direction_program_evidence",
        "numerical_covariates": list(NUMERICAL_COVARIATES),
        "categorical_covariates": [],
        "folds": 4,
        "fold_aggregation": "median_positive_coupling_support",
        "minimum_training_subjects": 10,
        "tuned_parameters": 0,
        "formal_inference_allowed": False,
    }
    candidate = config.get("candidate")
    if not isinstance(candidate, Mapping) or any(
        candidate.get(key) != value for key, value in expected_candidate.items()
    ):
        raise ValueError("M2 candidate identity or frozen formula changed")
    boundary = config.get("claim_boundary")
    if not isinstance(boundary, Mapping) or not all(
        boundary.get(field) is True
        for field in (
            "experimental_head_only",
            "m0_and_m1_heads_are_retained_separately",
            "held_out_subjects_are_excluded_from_each_fit",
            "generation_latents_are_not_method_inputs",
            "recorded_nuisance_covariates_are_required",
            "probability_is_not_emitted",
            "p_and_q_values_are_not_emitted",
            "synthetic_sender_fixture_is_not_general_sota_evidence",
        )
    ):
        raise ValueError("M2 claim boundary changed")
    if role not in {"development", "validation"}:
        raise ValueError("role must be development or validation")
    section = config.get(role)
    if not isinstance(section, Mapping):
        raise ValueError(f"M2 configuration lacks {role} section")
    expected_sha = section.get("fixture_manifest_sha256")
    if not isinstance(expected_sha, str) or expected_sha.startswith("PENDING"):
        raise ValueError(f"{role} fixture checksum is not frozen")
    if sha256_file(fixture_manifest) != expected_sha:
        raise ValueError(f"{role} fixture manifest checksum differs")
    return config


def _subject_folds(
    subject_ids: Sequence[str], *, root_seed: int, n_folds: int
) -> dict[str, int]:
    subjects = tuple(sorted(map(str, subject_ids)))
    if len(set(subjects)) != len(subjects):
        raise ValueError("fold input subjects must be unique")
    if n_folds < 2 or len(subjects) < n_folds * 2:
        raise ValueError("fold split requires at least two subjects per fold")
    keyed = sorted(
        subjects,
        key=lambda subject: hashlib.sha256(
            f"{FOLD_NAMESPACE}:{root_seed}:{subject}".encode()
        ).digest(),
    )
    return {subject: index % n_folds for index, subject in enumerate(keyed)}


def _sample_proportions(adata: ad.AnnData) -> pd.DataFrame:
    obs = adata.obs.loc[:, ["sample_id", "cell_type"]].astype(str)
    counts = (
        obs.groupby(["sample_id", "cell_type"], observed=True, sort=True)
        .size()
        .rename("sender_cells")
        .reset_index()
    )
    totals = (
        obs.groupby("sample_id", observed=True, sort=True).size().rename("sample_cells")
    )
    counts = counts.merge(totals, on="sample_id", how="left", validate="many_to_one")
    counts["sender_cell_proportion"] = counts["sender_cells"] / counts["sample_cells"]
    return counts.rename(columns={"cell_type": "sender"})


def _fit_dataset(
    dataset_id: str,
    dataset_spec: Mapping[str, Any],
    resource_path: str,
    resource_manifest_path: str,
    definition_records: list[dict[str, Any]],
    candidates: tuple[str, ...],
    expression_reference: float,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    input_path = Path(str(dataset_spec["input"])).resolve()
    if sha256_file(input_path) != str(dataset_spec["input_sha256"]):
        raise ValueError(f"M2 input checksum mismatch: {dataset_id}")
    adata = ad.read_h5ad(input_path)
    try:
        bundle = harmonized_resource_bundle(resource_path, resource_manifest_path)
        workflow = cast(Mapping[str, Any], dataset_spec["workflow"])
        with threadpool_limits(limits=1):
            artifacts = fit_baseline(
                adata,
                CrychicConfig.from_dict(
                    cast(Mapping[str, Any], dataset_spec["config"])
                ),
                bundle,
                target_prior=None,
                availability_parameters=AvailabilityParameters(
                    absolute_expression_reference=expression_reference
                ),
                **dict(workflow),
            )
        heads = build_absolute_activity_heads(
            artifacts.availability, artifacts.sender_assignment
        )
        definition = pd.DataFrame.from_records(definition_records)
        identity = definition.iloc[0]
        selected = heads.loc[
            heads["sender"].astype(str).isin(candidates)
            & heads["receiver"].astype(str).eq(str(identity["receiver"]))
            & heads["interaction_id"].astype(str).eq(str(identity["interaction_id"])),
            [
                "sample_id",
                "subject_id",
                "sender",
                "receiver",
                "interaction_id",
                "ligand_absolute_evidence",
                "sender_detection",
            ],
        ].copy()
        if selected.empty or selected.duplicated(["sample_id", "sender"]).any():
            raise ValueError(
                f"M2 candidate evidence is not one row per sample: {dataset_id}"
            )
        sample_context = (
            adata.obs.loc[
                :,
                [
                    "sample_id",
                    "subject_id",
                    "condition",
                    "assay_batch",
                    "composition_design_index",
                ],
            ]
            .astype(str)
            .drop_duplicates(ignore_index=True)
        )
        if sample_context.duplicated("sample_id").any():
            raise ValueError("sample_id must map to one subject and condition")
        sample_context["batch_index"] = sample_context["assay_batch"].map(
            {"A": 0.0, "B": 1.0}
        )
        if sample_context["batch_index"].isna().any():
            raise ValueError("M2 assay batch must be A or B")
        sample_context["composition_design_index"] = pd.to_numeric(
            sample_context["composition_design_index"], errors="raise"
        ).astype(float)
        program = _program_sample_table(
            artifacts.aggregate,
            definition,
            expression_reference=expression_reference,
        )
        proportions = _sample_proportions(adata)
        evidence = selected.merge(
            sample_context,
            on=["sample_id", "subject_id"],
            how="left",
            validate="many_to_one",
        ).merge(
            proportions.loc[
                lambda table: table["sender"].isin(candidates),
                ["sample_id", "sender", "sender_cell_proportion"],
            ],
            on=["sample_id", "sender"],
            how="left",
            validate="one_to_one",
        )
        evidence = evidence.merge(
            program.loc[
                :,
                [
                    "sample_id",
                    "subject_id",
                    "condition",
                    "receiver",
                    "interaction_id",
                    "receiver_program_raw",
                    "expected_program_direction",
                ],
            ],
            on=[
                "sample_id",
                "subject_id",
                "condition",
                "receiver",
                "interaction_id",
            ],
            how="left",
            validate="many_to_one",
        )
        required = [
            "condition",
            "batch_index",
            "composition_design_index",
            "sender_cell_proportion",
            "ligand_absolute_evidence",
            "receiver_program_raw",
        ]
        if evidence[required].isna().any().any():
            raise ValueError(f"M2 sample evidence is incomplete: {dataset_id}")
        expected_rows = len(sample_context) * len(candidates)
        if len(evidence) != expected_rows:
            raise ValueError(
                f"M2 candidate coverage differs: {len(evidence)} != {expected_rows}"
            )
        timing = {
            "dataset_id": dataset_id,
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "subjects": int(adata.obs["subject_id"].astype(str).nunique()),
            "elapsed_seconds": time.perf_counter() - started,
            **{
                f"stage_{name}_seconds": float(value)
                for name, value in artifacts.stage_seconds.items()
            },
        }
        return dataset_id, evidence.reset_index(drop=True), timing
    finally:
        if adata.isbacked:
            adata.file.close()


def _paired_difference(table: pd.DataFrame, value: str) -> pd.Series:
    collapsed = table.groupby(["subject_id", "condition"], observed=True, sort=False)[
        value
    ].mean()
    wide = collapsed.unstack("condition")
    if not {"ctrl", "stim"}.issubset(wide.columns):
        raise ValueError("paired effects require ctrl and stim")
    paired = wide.loc[:, ["ctrl", "stim"]].dropna()
    return (paired["stim"] - paired["ctrl"]).sort_index()


def _subject_effects(sample_evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (dataset_id, root_seed, sender), table in sample_evidence.groupby(
        ["dataset_id", "root_seed", "sender"], observed=True, sort=True
    ):
        effects: dict[str, pd.Series] = {
            "sender_effect": _paired_difference(table, "ligand_absolute_evidence"),
            "receiver_effect": _paired_difference(table, "receiver_program_raw"),
            "batch_contrast": _paired_difference(table, "batch_index"),
            "composition_design_contrast": _paired_difference(
                table, "composition_design_index"
            ),
            "sender_proportion_effect": _paired_difference(
                table, "sender_cell_proportion"
            ),
        }
        subject_ids = effects["sender_effect"].index
        if any(not values.index.equals(subject_ids) for values in effects.values()):
            raise ValueError("M2 subject-effect coverage differs by component")
        frame = pd.DataFrame(effects, index=subject_ids).reset_index()
        frame.insert(0, "sender", str(sender))
        frame.insert(0, "root_seed", int(root_seed))
        frame.insert(0, "dataset_id", str(dataset_id))
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(
        ["root_seed", "sender", "subject_id"], kind="stable", ignore_index=True
    )


def _candidate_couplings(
    effects: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    n_folds: int,
    minimum_training_subjects: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    truth_by_sender = truth.set_index("sender")
    for (dataset_id, root_seed), dataset in effects.groupby(
        ["dataset_id", "root_seed"], observed=True, sort=True
    ):
        subject_ids = tuple(sorted(dataset["subject_id"].astype(str).unique()))
        folds = _subject_folds(subject_ids, root_seed=int(root_seed), n_folds=n_folds)
        for sender, candidate in dataset.groupby("sender", observed=True, sort=True):
            candidate = candidate.loc[
                :,
                [
                    "subject_id",
                    "sender_effect",
                    "receiver_effect",
                    *NUMERICAL_COVARIATES,
                ],
            ].reset_index(drop=True)
            raw_supports: list[float] = []
            adjusted_supports: list[float] = []
            for fold in range(n_folds):
                held_out = tuple(
                    sorted(
                        subject
                        for subject, assigned in folds.items()
                        if assigned == fold
                    )
                )
                training = candidate.loc[
                    ~candidate["subject_id"].astype(str).isin(held_out)
                ].copy()
                fits = {
                    RAW_METHOD: fit_residualized_sender_coupling(
                        training,
                        sender=str(sender),
                        receiver=str(truth_by_sender.loc[str(sender), "receiver"]),
                        interaction_id=str(
                            truth_by_sender.loc[str(sender), "interaction_id"]
                        ),
                        fold_id=f"fold-{fold + 1}",
                        spec=ResidualizedCouplingSpec(
                            minimum_subjects=minimum_training_subjects
                        ),
                    ),
                    M2_METHOD: fit_residualized_sender_coupling(
                        training,
                        sender=str(sender),
                        receiver=str(truth_by_sender.loc[str(sender), "receiver"]),
                        interaction_id=str(
                            truth_by_sender.loc[str(sender), "interaction_id"]
                        ),
                        fold_id=f"fold-{fold + 1}",
                        spec=ResidualizedCouplingSpec(
                            numerical_covariates=NUMERICAL_COVARIATES,
                            minimum_subjects=minimum_training_subjects,
                        ),
                    ),
                }
                for method, fitted in fits.items():
                    support = fitted.positive_coupling_support
                    if support is not None:
                        (
                            raw_supports if method == RAW_METHOD else adjusted_supports
                        ).append(float(support))
                    fold_rows.append(
                        {
                            "dataset_id": str(dataset_id),
                            "root_seed": int(root_seed),
                            "sender": str(sender),
                            "sender_role": str(
                                truth_by_sender.loc[str(sender), "sender_role"]
                            ),
                            "expected_coupled_sender": bool(
                                truth_by_sender.loc[
                                    str(sender), "expected_coupled_sender"
                                ]
                            ),
                            "fold": fold + 1,
                            "method": method,
                            "training_subject_ids": "|".join(
                                fitted.training_subject_ids
                            ),
                            "held_out_subject_ids": "|".join(held_out),
                            "n_training_subjects": fitted.n_complete_subjects,
                            "status": fitted.status.value,
                            "reason_code": fitted.reason_code,
                            "signed_correlation": fitted.signed_correlation,
                            "positive_coupling_support": support,
                            "design_rank": fitted.design_rank,
                            "design_condition_number": (fitted.design_condition_number),
                            "coupling_id": fitted.coupling_id,
                            "formal_inference_allowed": False,
                        }
                    )
            lr_effect = float(candidate["sender_effect"].mean())
            program_effect = float(candidate["receiver_effect"].mean())
            m1 = signed_geometric_program_concordance(
                lr_effect, program_effect, expected_program_direction=1
            )
            score_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "root_seed": int(root_seed),
                    "sender": str(sender),
                    "sender_role": str(truth_by_sender.loc[str(sender), "sender_role"]),
                    "expected_coupled_sender": bool(
                        truth_by_sender.loc[str(sender), "expected_coupled_sender"]
                    ),
                    "subjects": len(candidate),
                    "m0_mean_ligand_effect": lr_effect,
                    "receiver_program_mean_effect": program_effect,
                    "m0_ranking_score": abs(lr_effect),
                    "m1_ranking_score": abs(m1),
                    "raw_coupling_score": (
                        float(np.median(raw_supports)) if raw_supports else np.nan
                    ),
                    "m2_coupling_score": (
                        float(np.median(adjusted_supports))
                        if adjusted_supports
                        else np.nan
                    ),
                    "raw_fold_coverage": len(raw_supports) / n_folds,
                    "m2_fold_coverage": len(adjusted_supports) / n_folds,
                    "formal_inference_allowed": False,
                }
            )
    scores = pd.DataFrame.from_records(score_rows).sort_values(
        ["root_seed", "sender"], kind="stable", ignore_index=True
    )
    fold_table = pd.DataFrame.from_records(fold_rows).sort_values(
        ["root_seed", "sender", "method", "fold"],
        kind="stable",
        ignore_index=True,
    )
    return scores, fold_table


def _seed_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, family in scores.groupby("root_seed", observed=True, sort=True):
        labels = family["expected_coupled_sender"].astype(int).to_numpy()
        if len(labels) != len(CANDIDATE_ROLES) or labels.sum() != 1:
            raise ValueError("each M2 seed requires one true sender and three decoys")
        for method, column in METHOD_SCORE_COLUMNS.items():
            values = pd.to_numeric(family[column], errors="coerce")
            coverage = float(values.notna().mean())
            if coverage != 1.0:
                rows.append(
                    {
                        "root_seed": int(seed),
                        "method": method,
                        "candidate_coverage": coverage,
                        "average_precision": np.nan,
                        "auroc": np.nan,
                        "true_sender_rank": np.nan,
                        "true_sender_score": np.nan,
                        "maximum_decoy_score": np.nan,
                        "maximum_decoy_to_true_ratio": np.nan,
                    }
                )
                continue
            method_scores = values.to_numpy(dtype=float)
            true_index = int(np.flatnonzero(labels == 1)[0])
            true_score = float(method_scores[true_index])
            maximum_decoy = float(method_scores[labels == 0].max())
            rank = float(
                pd.Series(method_scores)
                .rank(method="average", ascending=False)
                .iloc[true_index]
            )
            rows.append(
                {
                    "root_seed": int(seed),
                    "method": method,
                    "candidate_coverage": coverage,
                    "average_precision": float(
                        average_precision_score(labels, method_scores)
                    ),
                    "auroc": float(roc_auc_score(labels, method_scores)),
                    "true_sender_rank": rank,
                    "true_sender_score": true_score,
                    "maximum_decoy_score": maximum_decoy,
                    "maximum_decoy_to_true_ratio": (
                        maximum_decoy / true_score if true_score > 0.0 else np.nan
                    ),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(
        ["method", "root_seed"], kind="stable", ignore_index=True
    )


def _method_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, table in metrics.groupby("method", observed=True, sort=True):
        rows.append(
            {
                "method": str(method),
                "seeds": int(table["root_seed"].nunique()),
                "evaluable_seeds": int(table["average_precision"].notna().sum()),
                "candidate_coverage": float(table["candidate_coverage"].mean()),
                "average_precision": float(table["average_precision"].mean()),
                "auroc": float(table["auroc"].mean()),
                "median_true_sender_rank": float(table["true_sender_rank"].median()),
                "mean_true_sender_score": float(table["true_sender_score"].mean()),
                "mean_maximum_decoy_score": float(table["maximum_decoy_score"].mean()),
                "q95_maximum_decoy_to_true_ratio": float(
                    table["maximum_decoy_to_true_ratio"].quantile(0.95)
                ),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    summary["average_precision_rank"] = summary["average_precision"].rank(
        method="min", ascending=False
    )
    summary["auroc_rank"] = summary["auroc"].rank(method="min", ascending=False)
    return summary.sort_values(
        ["average_precision_rank", "auroc_rank", "method"],
        kind="stable",
        ignore_index=True,
    )


def _paired_bootstrap(
    metrics: pd.DataFrame,
    *,
    comparison: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    candidate = metrics.loc[metrics["method"].eq(M2_METHOD), ["root_seed", metric]]
    baseline = metrics.loc[metrics["method"].eq(comparison), ["root_seed", metric]]
    paired = candidate.merge(
        baseline,
        on="root_seed",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    ).dropna()
    differences = (
        paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
    ).to_numpy(dtype=float)
    if len(differences) < 2:
        raise ValueError("paired bootstrap requires at least two complete seeds")
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(differences), size=(replicates, len(differences)))
    bootstrap = differences[draw].mean(axis=1)
    return {
        "candidate": M2_METHOD,
        "baseline": comparison,
        "metric": metric,
        "paired_seeds": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "win_fraction": float((differences > 0.0).mean()),
        "tie_fraction": float((differences == 0.0).mean()),
    }


def _gate(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    section = cast(Mapping[str, Any], config[role])
    gates = cast(Mapping[str, Any], config["gates"])
    indexed = summary.set_index("method")
    m2 = indexed.loc[M2_METHOD]

    def comparison(baseline: str, metric: str) -> pd.Series:
        selected = paired.loc[
            paired["baseline"].eq(baseline) & paired["metric"].eq(metric)
        ]
        if len(selected) != 1:
            raise ValueError(f"missing paired M2 comparison: {baseline}, {metric}")
        return selected.iloc[0]

    raw_ap = comparison(RAW_METHOD, "average_precision")
    raw_auroc = comparison(RAW_METHOD, "auroc")
    m1_ap = comparison(M1_METHOD, "average_precision")
    checks = {
        "minimum_paired_seeds": int(raw_ap["paired_seeds"])
        >= int(section["minimum_paired_seeds"]),
        "ap_gain_vs_raw": float(raw_ap["ci_low"])
        > float(gates["ap_delta_ci_lower_minimum_exclusive"]),
        "auroc_gain_vs_raw": float(raw_auroc["ci_low"])
        > float(gates["auroc_delta_ci_lower_minimum_exclusive"]),
        "ap_gain_vs_m1": float(m1_ap["ci_low"])
        > float(gates["ap_delta_vs_m1_ci_lower_minimum_exclusive"]),
        "minimum_win_fraction_vs_raw": min(
            float(raw_ap["win_fraction"]), float(raw_auroc["win_fraction"])
        )
        >= float(gates["minimum_seed_win_fraction_vs_raw"]),
        "true_sender_rank": float(m2["median_true_sender_rank"])
        <= float(gates["maximum_median_true_sender_rank"]),
        "candidate_coverage": float(m2["candidate_coverage"])
        >= float(gates["minimum_candidate_coverage"]),
        "true_sender_retention": float(m2["mean_true_sender_score"])
        >= float(gates["minimum_mean_true_sender_support"]),
        "decoy_suppression": float(m2["q95_maximum_decoy_to_true_ratio"])
        <= float(gates["maximum_q95_decoy_to_true_ratio"]),
        "fold_coverage": float(scores["m2_fold_coverage"].mean())
        >= float(gates["minimum_fold_coverage"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "status": ("DEVELOPMENT_PASS" if role == "development" else "ACCEPT")
        if all(checks.values())
        else "REJECT",
        "checks": checks,
        "candidate": M2_METHOD,
        "claim_boundary": "synthetic sender-specificity evidence only",
    }


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


def evaluate(
    fixture_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    role: str,
    n_jobs: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the locked M2 sender-specificity evaluation and release gates."""

    started = time.perf_counter()
    fixture_dir = fixture_dir.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    fixture_manifest_path = fixture_dir / "manifest.json"
    fixture = _read_json(fixture_manifest_path)
    if fixture.get("schema_version") != FIXTURE_SCHEMA:
        raise ValueError("M2 fixture schema is unsupported")
    config = _validated_config(
        config_path, role=role, fixture_manifest=fixture_manifest_path
    )
    section = cast(Mapping[str, Any], config[role])
    if list(map(int, fixture.get("seeds", []))) != list(
        map(int, cast(Sequence[int], section["seeds"]))
    ):
        raise ValueError("M2 fixture seeds differ from frozen role seeds")
    fixture_config_record = cast(Mapping[str, Any], fixture["crychic_config"])
    fixture_config_path = fixture_dir / str(fixture_config_record["filename"])
    if sha256_file(fixture_config_path) != fixture_config_record["sha256"]:
        raise ValueError("M2 CRYCHIC configuration checksum differs")
    crychic_config = _read_json(fixture_config_path)
    datasets = cast(Mapping[str, Mapping[str, Any]], crychic_config["datasets"])
    resource_record = cast(Mapping[str, Any], fixture["resource"])
    resource_path = fixture_dir / str(resource_record["filename"])
    resource_manifest_path = fixture_dir / str(resource_record["manifest"])
    if sha256_file(resource_path) != resource_record["sha256"]:
        raise ValueError("M2 resource checksum differs")
    truth_record = cast(Mapping[str, Any], fixture["truth"])
    truth_path = fixture_dir / str(truth_record["filename"])
    if sha256_file(truth_path) != truth_record["sha256"]:
        raise ValueError("M2 truth checksum differs")
    truth = pd.read_csv(truth_path, sep="\t")
    expected_senders = tuple(CANDIDATE_ROLES)
    if tuple(truth["sender"].astype(str)) != expected_senders:
        raise ValueError("M2 frozen candidate sender order changed")
    if truth["expected_coupled_sender"].astype(bool).sum() != 1:
        raise ValueError("M2 truth must contain one coupled sender")
    definition = _program_definition(fixture_dir, fixture)

    candidate_config = cast(Mapping[str, Any], config["candidate"])
    expression_reference = float(config["absolute_expression_reference"])
    if not math.isfinite(expression_reference) or expression_reference <= 0.0:
        raise ValueError("M2 absolute expression reference is invalid")
    workers = min(int(n_jobs), len(datasets))
    if workers < 1:
        raise ValueError("n_jobs must be positive")
    evidence_by_dataset: dict[str, pd.DataFrame] = {}
    timings: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(
                _fit_dataset,
                dataset_id,
                spec,
                str(resource_path),
                str(resource_manifest_path),
                definition.to_dict(orient="records"),
                expected_senders,
                expression_reference,
            )
            for dataset_id, spec in datasets.items()
        ]
        for future in as_completed(futures):
            dataset_id, evidence, timing = future.result()
            evidence_by_dataset[dataset_id] = evidence
            timings.append(timing)
    seed_by_dataset = {
        str(record["dataset_id"]): int(record["root_seed"])
        for record in cast(Sequence[Mapping[str, Any]], fixture["records"])
    }
    evidence_frames: list[pd.DataFrame] = []
    for dataset_id in sorted(evidence_by_dataset):
        evidence = evidence_by_dataset[dataset_id].copy()
        evidence.insert(0, "root_seed", seed_by_dataset[dataset_id])
        evidence.insert(0, "dataset_id", dataset_id)
        evidence_frames.append(evidence)
    sample_evidence = pd.concat(evidence_frames, ignore_index=True)
    effects = _subject_effects(sample_evidence)
    scores, fold_couplings = _candidate_couplings(
        effects,
        truth,
        n_folds=int(candidate_config["folds"]),
        minimum_training_subjects=int(candidate_config["minimum_training_subjects"]),
    )
    metrics = _seed_metrics(scores)
    summary = _method_summary(metrics)
    comparisons = pd.DataFrame.from_records(
        [
            _paired_bootstrap(
                metrics,
                comparison=baseline,
                metric=metric,
                replicates=int(section["bootstrap_replicates"]),
                seed=int(section["bootstrap_seed"])
                + comparison_index * 10
                + metric_index,
            )
            for comparison_index, baseline in enumerate(
                (M0_METHOD, M1_METHOD, RAW_METHOD)
            )
            for metric_index, metric in enumerate(("average_precision", "auroc"))
        ]
    )
    acceptance = _gate(summary, comparisons, scores, config=config, role=role)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        tables = {
            "sample_evidence.tsv": sample_evidence,
            "subject_effects.tsv": effects,
            "fold_couplings.tsv": fold_couplings,
            "candidate_scores.tsv": scores,
            "seed_metrics.tsv": metrics,
            "method_summary.tsv": summary,
            "paired_comparisons.tsv": comparisons,
            "timing.tsv": pd.DataFrame.from_records(timings).sort_values(
                "dataset_id", kind="stable", ignore_index=True
            ),
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        _write_json(staged / "acceptance.json", acceptance)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "role": role,
            "acceptance": acceptance,
            "code": git_metadata(Path(__file__).resolve().parents[2]),
            "fixture": {
                "path": str(fixture_dir),
                "manifest_sha256": sha256_file(fixture_manifest_path),
            },
            "configuration": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "candidate": dict(candidate_config),
            "datasets": len(datasets),
            "seeds": int(metrics["root_seed"].nunique()),
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "tables": {
                filename: {
                    "sha256": sha256_file(staged / filename),
                    "rows": int(len(table)),
                }
                for filename, table in tables.items()
            },
            "acceptance_sha256": sha256_file(staged / "acceptance.json"),
            "claim_boundary": {
                "formal_inference_allowed": False,
                "general_sota_claim_allowed": False,
                "validated_estimand": (
                    "sender specificity with recorded batch/composition covariates"
                ),
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
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "validation"), required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = evaluate(
        args.fixture_dir,
        args.config,
        args.output_dir,
        role=args.role,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
