"""Evaluate known-truth paired 2x2 effects from per-sample CCC scores."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file

SCHEMA_VERSION = "crychic-brca-semisynthetic-evaluation-v3"
EVENT_KEYS = ("sender", "receiver", "interaction_id", "ligand", "receptor")
SCSEQ_EVENT_KEYS = ("ligand", "receptor", "cluster_L", "cluster_R")
REQUIRED_SCORE_COLUMNS = {
    "run_id",
    "dataset_id",
    "method_id",
    "resource_mode",
    "sample_id",
    "score",
    "score_direction",
    "status",
    *EVENT_KEYS,
}
ENGINES = ("native_raw_mean", "within_sample_rank_mean")
SCSEQ_NATIVE_ENGINE = "native_pairwise_difference_in_differences"
_DIRECTION_MULTIPLIER = {
    "higher": 1.0,
    "higher_is_stronger": 1.0,
    "lower": -1.0,
    "lower_is_stronger": -1.0,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


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


def _bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.notna() & np.isfinite(values) & values.between(0.0, 1.0)
    observed = values.loc[finite].to_numpy(dtype=float)
    if not len(observed):
        return result
    order = np.argsort(observed, kind="stable")
    ranked = observed[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result.loc[finite] = restored
    return result


def _design_table(path: Path) -> pd.DataFrame:
    source = ad.read_h5ad(path, backed="r")
    try:
        required = {"sample_id", "subject_id", "timepoint", "expansion"}
        missing = required.difference(source.obs.columns)
        if missing:
            raise ValueError(f"factorial metadata is missing: {sorted(missing)}")
        raw = source.obs.loc[:, sorted(required)]
        if raw.isna().any().any():
            raise ValueError("factorial metadata contains missing values")
        design = raw.astype(str).drop_duplicates()
    finally:
        source.file.close()
    if design.duplicated("sample_id", keep=False).any():
        raise ValueError("one sample maps to multiple factorial design rows")
    if set(design["timepoint"]) != {"Pre", "On"}:
        raise ValueError("timepoint must contain Pre and On")
    if set(design["expansion"]) != {"E", "NE"}:
        raise ValueError("expansion must contain E and NE")
    subject_expansion = design[["subject_id", "expansion"]].drop_duplicates()
    if subject_expansion.duplicated("subject_id", keep=False).any():
        raise ValueError("one subject maps to multiple expansion groups")
    support = design.groupby(
        ["subject_id", "timepoint"], observed=True, sort=False
    ).size()
    if len(support) != 2 * design["subject_id"].nunique() or not support.eq(1).all():
        raise ValueError("every subject must have one Pre and one On sample")
    return design.sort_values("sample_id", kind="stable", ignore_index=True)


def _view_labels(manifest_path: Path | None) -> dict[str, str]:
    if manifest_path is None or not manifest_path.is_file():
        return {}
    value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"method manifest must be an object: {manifest_path}")
    source_result = value.get("source_result")
    if not isinstance(source_result, dict):
        return {}
    views = source_result.get("score_views")
    if not isinstance(views, list):
        return {}
    labels: dict[str, str] = {}
    for index, item in enumerate(views, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str):
            continue
        explicit_label = item.get("view_label")
        if isinstance(explicit_label, str) and explicit_label:
            labels[str(item["run_id"])] = explicit_label
            continue
        candidates = item.get("contrast_candidates")
        labels[str(item["run_id"])] = (
            ";".join(map(str, candidates))
            if isinstance(candidates, list) and candidates
            else f"score_view_{index}"
        )
    return labels


def _runtime_metadata(
    manifest_path: Path | None,
    time_path: Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "adapter_elapsed_seconds": math.nan,
        "process_wall_seconds": math.nan,
        "peak_rss_kib": math.nan,
    }
    if manifest_path is not None and manifest_path.is_file():
        value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            elapsed = value.get("elapsed_seconds")
            if isinstance(elapsed, int | float) and not isinstance(elapsed, bool):
                result["adapter_elapsed_seconds"] = float(elapsed)
    if time_path is None or not time_path.is_file():
        return result
    text = time_path.read_text(encoding="utf-8")
    wall_match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)",
        text,
    )
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if wall_match:
        parts = [float(item) for item in wall_match.group(1).strip().split(":")]
        if len(parts) == 2:
            result["process_wall_seconds"] = 60.0 * parts[0] + parts[1]
        elif len(parts) == 3:
            result["process_wall_seconds"] = (
                3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
            )
    if rss_match:
        result["peak_rss_kib"] = float(rss_match.group(1))
    return result


def _scseq_manifest_metadata(
    manifest_path: Path,
    score_path: Path,
    *,
    expected_target: str,
    expected_reference: str,
) -> dict[str, Any]:
    value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"scSeqCommDiff manifest must be an object: {manifest_path}")
    protocol = value.get("protocol")
    outputs = value.get("outputs")
    if value.get("status") != "complete" or not isinstance(protocol, dict):
        raise ValueError(f"scSeqCommDiff run is not complete: {manifest_path}")
    if protocol.get("scenario") != "multi-sample":
        raise ValueError("scSeqCommDiff DID requires the native multi-sample scenario")
    if protocol.get("resource_mode") != "H-common":
        raise ValueError("scSeqCommDiff DID requires the frozen H-common resource")
    if protocol.get("contrast_order") != [expected_target, expected_reference]:
        raise ValueError("scSeqCommDiff manifest contrast order is inconsistent")
    if not isinstance(outputs, dict):
        raise ValueError("scSeqCommDiff manifest outputs are missing")
    score_record = outputs.get(score_path.name)
    if not isinstance(score_record, dict) or score_record.get("sha256") != sha256_file(
        score_path
    ):
        raise ValueError("scSeqCommDiff score checksum is not bound by its manifest")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "score_path": str(score_path.resolve()),
        "score_sha256": sha256_file(score_path),
        "dataset_id": value.get("dataset_id"),
        "elapsed_seconds": value.get("elapsed_seconds"),
    }


def _scseq_arm_scores(
    score_path: Path,
    truth: pd.DataFrame,
    *,
    expected_target: str,
    expected_reference: str,
) -> pd.DataFrame:
    table = pd.read_csv(score_path, sep="\t", keep_default_na=False)
    required = {
        *SCSEQ_EVENT_KEYS,
        "effect",
        "status",
        "reason_code",
        "target",
        "reference",
    }
    missing = required.difference(table.columns)
    if table.empty or missing:
        raise ValueError(
            f"invalid scSeqCommDiff event scores: missing={sorted(missing)}"
        )
    if table.duplicated(list(SCSEQ_EVENT_KEYS)).any():
        raise ValueError("scSeqCommDiff event scores contain duplicate event keys")
    if set(table["target"].astype(str)) != {expected_target} or set(
        table["reference"].astype(str)
    ) != {expected_reference}:
        raise ValueError("scSeqCommDiff score contrast labels are inconsistent")
    numeric = pd.to_numeric(table["effect"], errors="coerce")
    observed = table["status"].astype(str).eq("observed")
    if not np.isfinite(numeric.loc[observed]).all():
        raise ValueError("observed scSeqCommDiff effects must be finite")
    lr_map = truth[["interaction_id", "ligand", "receptor"]].drop_duplicates()
    if lr_map.duplicated(["ligand", "receptor"]).any():
        raise ValueError("truth interaction identifiers are not unique by LR pair")
    selected = table.loc[
        :,
        [*SCSEQ_EVENT_KEYS, "effect", "status", "reason_code"],
    ].rename(columns={"cluster_L": "sender", "cluster_R": "receiver"})
    selected["effect"] = pd.to_numeric(selected["effect"], errors="coerce")
    selected = selected.merge(
        lr_map,
        on=["ligand", "receptor"],
        how="inner",
        validate="many_to_one",
    )
    return selected.loc[:, [*EVENT_KEYS, "effect", "status", "reason_code"]]


def _scseq_native_did_effects(
    expansion_score_path: Path,
    nonexpansion_score_path: Path,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    expansion = _scseq_arm_scores(
        expansion_score_path,
        truth,
        expected_target="OnE",
        expected_reference="PreE",
    ).rename(
        columns={
            "effect": "effect_E",
            "status": "status_E",
            "reason_code": "reason_code_E",
        }
    )
    nonexpansion = _scseq_arm_scores(
        nonexpansion_score_path,
        truth,
        expected_target="OnNE",
        expected_reference="PreNE",
    ).rename(
        columns={
            "effect": "effect_NE",
            "status": "status_NE",
            "reason_code": "reason_code_NE",
        }
    )
    result = expansion.merge(
        nonexpansion,
        on=list(EVENT_KEYS),
        how="outer",
        validate="one_to_one",
    )
    observed = (
        result["status_E"].eq("observed")
        & result["status_NE"].eq("observed")
        & np.isfinite(result["effect_E"])
        & np.isfinite(result["effect_NE"])
    )
    result["difference_in_differences"] = np.where(
        observed,
        result["effect_E"] - result["effect_NE"],
        np.nan,
    )
    result["mean_delta_E"] = result["effect_E"]
    result["mean_delta_NE"] = result["effect_NE"]
    result["n_subjects_E"] = np.nan
    result["n_subjects_NE"] = np.nan
    result["p_value"] = np.nan
    result["q_value"] = np.nan
    result["status"] = np.where(observed, "observed", "not_estimable")
    result["reason_code"] = np.where(
        observed,
        "",
        "one_or_both_native_pairwise_effects_not_estimable",
    )
    return result.loc[
        :,
        [
            *EVENT_KEYS,
            "n_subjects_E",
            "n_subjects_NE",
            "mean_delta_E",
            "mean_delta_NE",
            "difference_in_differences",
            "p_value",
            "q_value",
            "status",
            "reason_code",
        ],
    ]


def _combined_native_runtime(
    expansion_manifest: Path,
    nonexpansion_manifest: Path,
    expansion_time: Path | None,
    nonexpansion_time: Path | None,
) -> dict[str, Any]:
    arms = (
        _runtime_metadata(expansion_manifest, expansion_time),
        _runtime_metadata(nonexpansion_manifest, nonexpansion_time),
    )

    def finite_values(key: str) -> list[float]:
        return [
            float(arm[key])
            for arm in arms
            if np.isfinite(float(arm[key]))
        ]

    adapter = finite_values("adapter_elapsed_seconds")
    wall = finite_values("process_wall_seconds")
    rss = finite_values("peak_rss_kib")
    return {
        "adapter_elapsed_seconds": sum(adapter) if len(adapter) == 2 else math.nan,
        "process_wall_seconds": sum(wall) if len(wall) == 2 else math.nan,
        "peak_rss_kib": max(rss) if len(rss) == 2 else math.nan,
        "native_arm_parallel_wall_lower_bound_seconds": (
            max(wall) if len(wall) == 2 else math.nan
        ),
    }


def _score_variants(
    score_path: Path,
    design: pd.DataFrame,
    *,
    manifest_path: Path | None,
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    table = pd.read_parquet(score_path)
    missing = REQUIRED_SCORE_COLUMNS.difference(table.columns)
    if table.empty or missing:
        raise ValueError(f"invalid score table {score_path}: missing={sorted(missing)}")
    dataset_ids = tuple(sorted(table["dataset_id"].astype(str).unique()))
    if len(dataset_ids) != 1:
        raise ValueError("one method table must contain exactly one dataset")
    if set(table["resource_mode"].astype(str)) != {"H-common"}:
        raise ValueError("semi-synthetic comparison requires H-common scores")
    unknown_samples = set(table["sample_id"].astype(str)).difference(
        design["sample_id"]
    )
    if unknown_samples:
        raise ValueError(
            f"score table has unknown samples: {sorted(unknown_samples)[:5]}"
        )
    labels = _view_labels(manifest_path)
    variants: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for run_id, group in table.groupby("run_id", observed=True, sort=True):
        methods = tuple(sorted(group["method_id"].astype(str).unique()))
        directions = tuple(sorted(group["score_direction"].astype(str).unique()))
        if len(methods) != 1 or len(directions) != 1:
            raise ValueError("one score view must have one method and direction")
        if directions[0] not in _DIRECTION_MULTIPLIER:
            raise ValueError(f"unsupported score direction: {directions[0]}")
        keys = ["sample_id", *EVENT_KEYS]
        if group.duplicated(keys).any():
            raise ValueError(f"score view {run_id} has duplicate sample-event rows")
        selected = group.loc[:, [*keys, "score", "status"]].copy()
        numeric = pd.to_numeric(selected["score"], errors="coerce")
        observed = selected["status"].astype(str).eq("ok") & np.isfinite(numeric)
        selected["oriented_score"] = np.where(
            observed,
            numeric * _DIRECTION_MULTIPLIER[directions[0]],
            np.nan,
        )
        selected = selected.merge(
            design,
            on="sample_id",
            how="left",
            validate="many_to_one",
        )
        variants.append(
            (
                {
                    "dataset_id": dataset_ids[0],
                    "base_method_id": methods[0],
                    "run_id": str(run_id),
                    "view_label": labels.get(str(run_id), "canonical_sample_score"),
                    "score_direction": directions[0],
                    "score_path": str(score_path.resolve()),
                    "score_sha256": sha256_file(score_path),
                    "manifest_path": (
                        None if manifest_path is None else str(manifest_path.resolve())
                    ),
                },
                selected,
            )
        )
    return variants


def _engine_scores(table: pd.DataFrame, engine: str) -> pd.DataFrame:
    result = table.copy()
    if engine == "native_raw_mean":
        result["engine_score"] = result["oriented_score"]
    elif engine == "within_sample_rank_mean":
        result["engine_score"] = result.groupby(
            "sample_id", observed=True, sort=False
        )["oriented_score"].rank(method="average", pct=True, na_option="keep")
    else:
        raise ValueError(f"unknown differential engine: {engine}")
    return result


def _subject_deltas(table: pd.DataFrame) -> pd.DataFrame:
    index = ["subject_id", "expansion", *EVENT_KEYS]
    scores = (
        table.groupby([*index, "timepoint"], observed=True, sort=False)[
            "engine_score"
        ]
        .mean()
        .unstack("timepoint")
    )
    if not {"Pre", "On"}.issubset(scores.columns):
        return pd.DataFrame(columns=[*index, "paired_delta"])
    paired = scores.loc[:, ["Pre", "On"]].dropna()
    paired["paired_delta"] = paired["On"] - paired["Pre"]
    return paired[["paired_delta"]].reset_index()


def _welch_inference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    effect: float,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Return two-sided Welch inference with an explicit degenerate-null case."""

    left_variance = float(np.var(left, ddof=1))
    right_variance = float(np.var(right, ddof=1))
    left_component = left_variance / len(left)
    right_component = right_variance / len(right)
    standard_error = math.sqrt(left_component + right_component)
    if standard_error == 0.0:
        if effect == 0.0:
            return {
                "standard_error": 0.0,
                "degrees_of_freedom": math.inf,
                "p_value": 1.0,
                "ci_low_95": 0.0,
                "ci_high_95": 0.0,
            }
        return {
            "standard_error": 0.0,
            "degrees_of_freedom": math.nan,
            "p_value": math.nan,
            "ci_low_95": math.nan,
            "ci_high_95": math.nan,
        }
    denominator = (
        left_component**2 / (len(left) - 1)
        + right_component**2 / (len(right) - 1)
    )
    degrees_of_freedom = (
        (left_component + right_component) ** 2 / denominator
        if denominator > 0.0
        else math.inf
    )
    statistic = effect / standard_error
    p_value = float(
        2.0 * student_t.sf(abs(statistic), df=degrees_of_freedom)
    )
    critical = float(
        student_t.ppf(0.5 + confidence / 2.0, df=degrees_of_freedom)
    )
    radius = critical * standard_error
    return {
        "standard_error": standard_error,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "ci_low_95": effect - radius,
        "ci_high_95": effect + radius,
    }


def _paired_effects(
    deltas: pd.DataFrame,
    *,
    min_subjects_per_group: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for event, group in deltas.groupby(list(EVENT_KEYS), observed=True, sort=False):
        values = {
            str(expansion): frame["paired_delta"].to_numpy(dtype=float)
            for expansion, frame in group.groupby("expansion", observed=True)
        }
        left = values.get("E", np.empty(0, dtype=float))
        right = values.get("NE", np.empty(0, dtype=float))
        estimable = (
            len(left) >= min_subjects_per_group
            and len(right) >= min_subjects_per_group
            and np.isfinite(left).all()
            and np.isfinite(right).all()
        )
        effect = float(left.mean() - right.mean()) if estimable else math.nan
        inference = (
            _welch_inference(left, right, effect=effect)
            if estimable
            else {
                "standard_error": math.nan,
                "degrees_of_freedom": math.nan,
                "p_value": math.nan,
                "ci_low_95": math.nan,
                "ci_high_95": math.nan,
            }
        )
        records.append(
            {
                **dict(zip(EVENT_KEYS, event, strict=True)),
                "n_subjects_E": len(left),
                "n_subjects_NE": len(right),
                "mean_delta_E": float(left.mean()) if len(left) else math.nan,
                "mean_delta_NE": float(right.mean()) if len(right) else math.nan,
                "difference_in_differences": effect,
                **inference,
                "status": "observed" if estimable else "not_estimable",
                "reason_code": (
                    ""
                    if estimable
                    else "insufficient_complete_paired_subjects_per_expansion"
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return pd.DataFrame(
            columns=[
                *EVENT_KEYS,
                "n_subjects_E",
                "n_subjects_NE",
                "mean_delta_E",
                "mean_delta_NE",
                "difference_in_differences",
                "standard_error",
                "degrees_of_freedom",
                "p_value",
                "ci_low_95",
                "ci_high_95",
                "q_value",
                "status",
                "reason_code",
            ]
        )
    result["q_value"] = _bh_adjust(result["p_value"])
    return result


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or not len(labels):
        return math.nan
    return float(average_precision_score(labels, scores))


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2 or not len(labels):
        return math.nan
    return float(roc_auc_score(labels, scores))


def _metrics(
    effects: pd.DataFrame,
    *,
    formal_did_p_value: bool = True,
) -> dict[str, Any]:
    observed = effects["status"].eq("observed") & np.isfinite(
        pd.to_numeric(effects["difference_in_differences"], errors="coerce")
    )
    scored = effects.loc[observed].copy()
    labels = scored["truth_label"].to_numpy(dtype=int)
    directions = scored["truth_direction"].to_numpy(dtype=int)
    values = scored["difference_in_differences"].to_numpy(dtype=float)
    active = directions != 0
    positive = directions > 0
    negative = directions < 0
    main_control = scored["planted_main_effect_control"].astype(bool).to_numpy()
    no_effect = ~(active | main_control)
    null = ~active
    p_values = pd.to_numeric(
        scored.get("p_value", pd.Series(np.nan, index=scored.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    q_values = pd.to_numeric(
        scored.get("q_value", pd.Series(np.nan, index=scored.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    ci_low = pd.to_numeric(
        scored.get("ci_low_95", pd.Series(np.nan, index=scored.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    ci_high = pd.to_numeric(
        scored.get("ci_high_95", pd.Series(np.nan, index=scored.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    finite_p = np.isfinite(p_values)
    finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)
    null_finite_p = null & finite_p
    null_finite_ci = null & finite_ci
    discovered = np.isfinite(q_values) & (q_values < 0.05)
    true_positives = int(np.sum(discovered & active))
    false_positives = int(np.sum(discovered & ~active))
    active_median = (
        float(np.median(np.abs(values[active]))) if active.any() else math.nan
    )
    main_median = (
        float(np.median(np.abs(values[main_control])))
        if main_control.any()
        else math.nan
    )
    return {
        "n_truth_events": len(effects),
        "n_truth_active": int(effects["truth_label"].sum()),
        "n_estimable_events": int(observed.sum()),
        "coverage": float(observed.mean()),
        "active_coverage": float(
            observed.loc[effects["truth_label"].eq(1)].mean()
        ),
        "omnibus_auprc": _safe_ap(labels, np.abs(values)),
        "omnibus_auroc": _safe_auc(labels, np.abs(values)),
        "positive_direction_ap": _safe_ap(positive.astype(int), values),
        "negative_direction_ap": _safe_ap(negative.astype(int), -values),
        "direction_accuracy_active": (
            float(np.mean(np.sign(values[active]) == directions[active]))
            if active.any()
            else math.nan
        ),
        "effect_all_zero_fraction": float(np.mean(values == 0.0)),
        "effect_dynamic_range": float(np.max(values) - np.min(values)),
        "active_median_abs_effect": active_median,
        "main_only_median_abs_leakage": main_median,
        "main_to_active_abs_ratio": (
            main_median / active_median
            if np.isfinite(main_median) and active_median > 0.0
            else math.nan
        ),
        "no_effect_standard_deviation": (
            float(np.std(values[no_effect], ddof=1))
            if np.sum(no_effect) > 1
            else math.nan
        ),
        "n_formal_tests": int(finite_p.sum()) if formal_did_p_value else math.nan,
        "formal_test_fraction": (
            float(finite_p.mean()) if formal_did_p_value and len(finite_p) else math.nan
        ),
        "null_unadjusted_rejections_p_lt_005": (
            int(np.sum(null_finite_p & (p_values < 0.05)))
            if formal_did_p_value
            else math.nan
        ),
        "null_type_i_005": (
            float(np.mean(p_values[null_finite_p] < 0.05))
            if formal_did_p_value and null_finite_p.any()
            else math.nan
        ),
        "null_ci_95_coverage": (
            float(
                np.mean(
                    (ci_low[null_finite_ci] <= 0.0)
                    & (ci_high[null_finite_ci] >= 0.0)
                )
            )
            if formal_did_p_value and null_finite_ci.any()
            else math.nan
        ),
        "null_bh_any_false_discovery": (
            int(np.any(discovered & null)) if formal_did_p_value else math.nan
        ),
        "discoveries_q_lt_005": (
            int(discovered.sum()) if formal_did_p_value else math.nan
        ),
        "true_positive_discoveries": (
            true_positives if formal_did_p_value else math.nan
        ),
        "false_positive_discoveries": (
            false_positives if formal_did_p_value else math.nan
        ),
        "empirical_fdr": (
            false_positives / int(discovered.sum())
            if formal_did_p_value and discovered.any()
            else (0.0 if formal_did_p_value else math.nan)
        ),
        "power": (
            true_positives / int(np.sum(active))
            if formal_did_p_value and active.any()
            else math.nan
        ),
        "main_only_false_discoveries": (
            int(np.sum(discovered & main_control))
            if formal_did_p_value
            else math.nan
        ),
    }


def _truth_for_dataset(truth: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    selected = truth.loc[truth["dataset_id"].astype(str).eq(dataset_id)].copy()
    required = {
        *EVENT_KEYS,
        "event_class",
        "truth_label",
        "truth_direction",
        "planted_main_effect_control",
    }
    missing = required.difference(selected.columns)
    if selected.empty or missing:
        raise ValueError(
            f"truth is missing dataset or columns for {dataset_id}: {sorted(missing)}"
        )
    if selected.duplicated(list(EVENT_KEYS)).any():
        raise ValueError("truth contains duplicate event keys")
    return selected


def evaluate(
    truth_path: Path,
    dataset_inputs: Mapping[str, Path],
    method_specs: Sequence[tuple[Path, Path | None, Path | None]],
    output_dir: Path,
    *,
    native_did_specs: Sequence[
        tuple[
            str,
            str,
            Path,
            Path,
            Path,
            Path,
            Path | None,
            Path | None,
        ]
    ] = (),
    min_subjects_per_group: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    if min_subjects_per_group < 2:
        raise ValueError("minimum subjects per expansion must be at least two")
    truth_path = truth_path.expanduser().resolve()
    truth = pd.read_csv(truth_path, sep="\t", keep_default_na=False)
    designs = {
        dataset_id: _design_table(path.expanduser().resolve())
        for dataset_id, path in dataset_inputs.items()
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    effects_parts: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    published = False
    try:
        for score_path, manifest_path, time_path in method_specs:
            resolved_score = score_path.expanduser().resolve()
            method_table = pd.read_parquet(
                resolved_score, columns=["dataset_id"]
            )
            dataset_ids = tuple(
                sorted(method_table["dataset_id"].astype(str).unique())
            )
            if len(dataset_ids) != 1 or dataset_ids[0] not in designs:
                raise ValueError(
                    f"method dataset is not registered: {dataset_ids}"
                )
            dataset_id = dataset_ids[0]
            runtime = _runtime_metadata(manifest_path, time_path)
            for metadata, table in _score_variants(
                resolved_score,
                designs[dataset_id],
                manifest_path=manifest_path,
            ):
                truth_dataset = _truth_for_dataset(truth, dataset_id)
                for engine in ENGINES:
                    variant_id = (
                        f"{metadata['base_method_id']}::{metadata['run_id']}::{engine}"
                    )
                    deltas = _subject_deltas(_engine_scores(table, engine))
                    estimated = _paired_effects(
                        deltas,
                        min_subjects_per_group=min_subjects_per_group,
                    )
                    merged = truth_dataset.merge(
                        estimated,
                        on=list(EVENT_KEYS),
                        how="left",
                        validate="one_to_one",
                    )
                    merged["status"] = merged["status"].fillna("not_estimable")
                    merged["reason_code"] = merged["reason_code"].fillna(
                        "event_not_emitted_or_incomplete_pairs"
                    )
                    merged.insert(0, "method_variant_id", variant_id)
                    merged.insert(1, "base_method_id", metadata["base_method_id"])
                    merged.insert(2, "run_id", metadata["run_id"])
                    merged.insert(3, "view_label", metadata["view_label"])
                    merged.insert(4, "differential_engine", engine)
                    merged.insert(5, "formal_did_p_value", True)
                    effects_parts.append(merged)
                    metric_records.append(
                        {
                            "method_variant_id": variant_id,
                            "dataset_id": dataset_id,
                            "base_method_id": metadata["base_method_id"],
                            "run_id": metadata["run_id"],
                            "view_label": metadata["view_label"],
                            "differential_engine": engine,
                            "formal_did_p_value": True,
                            **_metrics(merged),
                            **runtime,
                        }
                    )
                variants.append(
                    {
                        **metadata,
                        "time_path": (
                            None if time_path is None else str(time_path.resolve())
                        ),
                    }
                )

        for (
            method_id,
            dataset_id,
            expansion_score,
            nonexpansion_score,
            expansion_manifest,
            nonexpansion_manifest,
            expansion_time,
            nonexpansion_time,
        ) in native_did_specs:
            if dataset_id not in designs:
                raise ValueError(f"native DID dataset is not registered: {dataset_id}")
            truth_dataset = _truth_for_dataset(truth, dataset_id)
            paths = [
                expansion_score,
                nonexpansion_score,
                expansion_manifest,
                nonexpansion_manifest,
            ]
            if expansion_time is not None:
                paths.append(expansion_time)
            if nonexpansion_time is not None:
                paths.append(nonexpansion_time)
            resolved = [path.expanduser().resolve() for path in paths]
            missing = [path for path in resolved if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            expansion_score = expansion_score.expanduser().resolve()
            nonexpansion_score = nonexpansion_score.expanduser().resolve()
            expansion_manifest = expansion_manifest.expanduser().resolve()
            nonexpansion_manifest = nonexpansion_manifest.expanduser().resolve()
            expansion_metadata = _scseq_manifest_metadata(
                expansion_manifest,
                expansion_score,
                expected_target="OnE",
                expected_reference="PreE",
            )
            nonexpansion_metadata = _scseq_manifest_metadata(
                nonexpansion_manifest,
                nonexpansion_score,
                expected_target="OnNE",
                expected_reference="PreNE",
            )
            estimated = _scseq_native_did_effects(
                expansion_score,
                nonexpansion_score,
                truth_dataset,
            )
            merged = truth_dataset.merge(
                estimated,
                on=list(EVENT_KEYS),
                how="left",
                validate="one_to_one",
            )
            merged["status"] = merged["status"].fillna("not_estimable")
            merged["reason_code"] = merged["reason_code"].fillna(
                "event_not_emitted_by_one_or_both_native_pairwise_runs"
            )
            variant_id = f"{method_id}::{SCSEQ_NATIVE_ENGINE}"
            view_label = "native (OnE-PreE) - (OnNE-PreNE)"
            merged.insert(0, "method_variant_id", variant_id)
            merged.insert(1, "base_method_id", method_id)
            merged.insert(2, "run_id", SCSEQ_NATIVE_ENGINE)
            merged.insert(3, "view_label", view_label)
            merged.insert(4, "differential_engine", SCSEQ_NATIVE_ENGINE)
            merged.insert(5, "formal_did_p_value", False)
            runtime = _combined_native_runtime(
                expansion_manifest,
                nonexpansion_manifest,
                None if expansion_time is None else expansion_time.resolve(),
                None if nonexpansion_time is None else nonexpansion_time.resolve(),
            )
            effects_parts.append(merged)
            metric_records.append(
                {
                    "method_variant_id": variant_id,
                    "dataset_id": dataset_id,
                    "base_method_id": method_id,
                    "run_id": SCSEQ_NATIVE_ENGINE,
                    "view_label": view_label,
                    "differential_engine": SCSEQ_NATIVE_ENGINE,
                    "formal_did_p_value": False,
                    **_metrics(merged, formal_did_p_value=False),
                    **runtime,
                }
            )
            variants.append(
                {
                    "dataset_id": dataset_id,
                    "base_method_id": method_id,
                    "run_id": SCSEQ_NATIVE_ENGINE,
                    "view_label": view_label,
                    "formal_did_p_value": False,
                    "expansion_arm": expansion_metadata,
                    "nonexpansion_arm": nonexpansion_metadata,
                    "expansion_time_path": (
                        None
                        if expansion_time is None
                        else str(expansion_time.resolve())
                    ),
                    "nonexpansion_time_path": (
                        None
                        if nonexpansion_time is None
                        else str(nonexpansion_time.resolve())
                    ),
                }
            )

        if not effects_parts:
            raise ValueError(
                "at least one sample-score or native DID method is required"
            )

        all_effects = pd.concat(effects_parts, ignore_index=True, sort=False)
        metrics = pd.DataFrame.from_records(metric_records).sort_values(
            ["dataset_id", "omnibus_auprc", "method_variant_id"],
            ascending=[True, False, True],
            kind="stable",
            ignore_index=True,
        )
        metrics["omnibus_auprc_rank"] = metrics.groupby(
            "dataset_id", observed=True
        )["omnibus_auprc"].rank(method="min", ascending=False)
        effects_path = staged / "event_effects.tsv.gz"
        metrics_path = staged / "metrics.tsv"
        all_effects.to_csv(effects_path, sep="\t", index=False, compression="gzip")
        metrics.to_csv(metrics_path, sep="\t", index=False)
        report_lines = [
            "# BRCA-shaped 2x2 semi-synthetic benchmark",
            "",
            "Known interaction truth is evaluated with missing events left missing.",
            "No best CRYCHIC context-trained view is selected post hoc.",
            "",
            "| Rank | Method | View | Engine | DID p? | AUPRC | AUROC | "
            "Direction | Main/active leakage | Power | FDR |",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in metrics.itertuples(index=False):
            report_lines.append(
                "| {rank:.0f} | {method} | {view} | {engine} | {formal} | "
                "{ap:.4f} | {auc:.4f} | {direction:.4f} | {leakage:.4f} | "
                "{power:.4f} | {fdr:.4f} |".format(
                    rank=row.omnibus_auprc_rank,
                    method=row.base_method_id,
                    view=str(row.view_label).replace("|", "/"),
                    engine=row.differential_engine,
                    formal="yes" if row.formal_did_p_value else "no",
                    ap=row.omnibus_auprc,
                    auc=row.omnibus_auroc,
                    direction=row.direction_accuracy_active,
                    leakage=row.main_to_active_abs_ratio,
                    power=row.power,
                    fdr=row.empirical_fdr,
                )
            )
        report_path = staged / "REPORT.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "estimand": "(On-Pre in E) - (On-Pre in NE)",
            "engines": list(ENGINES),
            "native_did_engine": SCSEQ_NATIVE_ENGINE,
            "minimum_subjects_per_expansion": min_subjects_per_group,
            "missing_policy": "complete_paired_subjects_per_event;never_zero_imputed",
            "formal_inference": (
                "subject_level_within_pair_deltas;between_arm_welch_t;"
                "two_sided_95pct_welch_ci;BH_within_dataset_method_view_engine"
            ),
            "native_did_inference_policy": (
                "subtract_native_pairwise_effects;do_not_reuse_arm_p_values_for_DID"
            ),
            "native_did_runtime_policy": (
                "sum_arm_wall_and_adapter_time;max_arm_rss;parallel_wall_is_lower_bound"
            ),
            "truth": {
                "filename": truth_path.name,
                "sha256": sha256_file(truth_path),
                "rows": len(truth),
            },
            "datasets": {
                dataset_id: {
                    "input": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for dataset_id, path in dataset_inputs.items()
            },
            "variants": variants,
            "outputs": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in (effects_path, metrics_path, report_path)
            },
            "code": git_metadata(Path(__file__).resolve().parents[2]),
        }
        _write_json(staged / "manifest.json", manifest)
        _publish(staged, output_dir, overwrite=overwrite)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _method_spec(value: str) -> tuple[Path, Path | None, Path | None]:
    parts = value.split("::")
    if not 1 <= len(parts) <= 3 or not parts[0]:
        raise argparse.ArgumentTypeError(
            "--method expects SCORE[::MANIFEST[::TIME_FILE]]"
        )
    paths = [Path(part) if part else None for part in parts]
    return (
        paths[0],
        paths[1] if len(paths) > 1 else None,
        paths[2] if len(paths) > 2 else None,
    )  # type: ignore[return-value]


def _native_did_spec(
    value: str,
) -> tuple[
    str,
    str,
    Path,
    Path,
    Path,
    Path,
    Path | None,
    Path | None,
]:
    parts = value.split("::")
    if len(parts) not in {6, 8} or any(not item for item in parts[:6]):
        raise argparse.ArgumentTypeError(
            "--native-did expects METHOD::DATASET::E_SCORE::NE_SCORE::"
            "E_MANIFEST::NE_MANIFEST[::E_TIME::NE_TIME]"
        )
    if len(parts) == 8 and any(not item for item in parts[6:]):
        raise argparse.ArgumentTypeError("both native DID time files are required")
    return (
        parts[0],
        parts[1],
        Path(parts[2]),
        Path(parts[3]),
        Path(parts[4]),
        Path(parts[5]),
        Path(parts[6]) if len(parts) == 8 else None,
        Path(parts[7]) if len(parts) == 8 else None,
    )


def _dataset_spec(value: str) -> tuple[str, Path]:
    dataset, separator, path = value.partition("=")
    if not separator or not dataset or not path:
        raise argparse.ArgumentTypeError("--dataset expects DATASET_ID=H5AD")
    return dataset, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--dataset", required=True, action="append", type=_dataset_spec)
    parser.add_argument("--method", action="append", type=_method_spec, default=[])
    parser.add_argument(
        "--native-did", action="append", type=_native_did_spec, default=[]
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-subjects-per-group", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    datasets = dict(args.dataset)
    if len(datasets) != len(args.dataset):
        raise ValueError("dataset identifiers must be unique")
    manifest = evaluate(
        args.truth,
        datasets,
        args.method,
        args.output_dir,
        native_did_specs=args.native_did,
        min_subjects_per_group=args.min_subjects_per_group,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest["outputs"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
