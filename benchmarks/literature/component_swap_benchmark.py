"""Run the RC0 component-swap diagnosis requested in suggestions_v4.

The runner consumes checksum-bound completed runs.  It never refits or changes
the CRYCHIC backbone.  Arm C is explicitly a continuous-score Wald analogue,
because PyDESeq2's negative-binomial count likelihood cannot legitimately be
applied to a bounded CRYCHIC score.  Arm E remains not estimable until a
calibrated soft posterior exists; an arbitrary ``1 - p`` is not substituted.
"""

from __future__ import annotations

import argparse
import json
import math
import resource as process_resource
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)
from benchmarks.adapters.crychic.des_postprocess import (
    adjusted_subject_directed_lr_effects,
    sender_specific_direct_scores,
)
from benchmarks.adapters.crychic.replay_v3_differential import (
    _bound_output,
    _validated_crossfit_semantic_source,
)
from benchmarks.adapters.crychic.replay_v3_sample import (
    _SOURCE_CONTRACTS,
    _sample_effects,
    _validate_sample_layer,
    _validate_source_contract,
)
from benchmarks.literature.evaluate_spatial_des_benchmark import evaluate_rankings

SCHEMA_VERSION = "crychic-component-swap-rc0-v1"
METHOD_VERSION = "component-swap-rc0-v1"
RESOURCE_ID = "ConnectomeDB2020_Hou_2020_human"
EDGE_COLUMNS = (
    "sender",
    "receiver",
    "interaction_id",
    "ligand",
    "receptor",
    "family_id",
    "driver_id",
)
MIN_WILCOXON_UNITS = 4
ALPHA = 0.05

_CONTRACTS: dict[str, dict[str, Any]] = {
    "kuppe": {
        "truth_dataset": "Kuppe_MI_spatial_CTRL_vs_IZ",
        "expected_sha256": (
            "1a9ad459a7c2cf7eb1e52d47ec7f6d77815b28b604b63315fe8524a95fdf7655"
        ),
        "condition_map": {},
        "expected_filters": {"variant": "spatial_neighbor_max"},
    },
    "ms": {
        "truth_dataset": "lerma_martin_ms_ctrl_vs_chronic_active",
        "expected_sha256": (
            "402f6e8255cf032a40f456c441d6881d07131d05d21608406475fdf0de0a44a2"
        ),
        "condition_map": {"CA": "chronic_active", "Ctrl": "control"},
        "expected_filters": {},
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return cast(dict[str, Any], value)


def _input_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "columns": list(table.columns),
        "sha256": sha256_file(path),
    }


def _load_crychic_inputs(
    source: Path,
    replay: Path,
    *,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_manifest_path = source / "manifest.json"
    replay_manifest_path = replay / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    replay_manifest = _read_json(replay_manifest_path)
    contract = _validate_source_contract(source_manifest, dataset=dataset)
    if replay_manifest.get("status") != "complete":
        raise ValueError("CRYCHIC sample replay must be complete")
    replay_source = replay_manifest.get("source_run")
    if not isinstance(replay_source, Mapping):
        raise ValueError("CRYCHIC replay lacks source-run provenance")
    if replay_source.get("path") != str(source.resolve()):
        raise ValueError("CRYCHIC replay is not bound to the requested source run")
    if replay_source.get("manifest_sha256") != sha256_file(source_manifest_path):
        raise ValueError("CRYCHIC replay/source manifest checksum mismatch")

    score_path = _bound_output(
        source, source_manifest, "sender_lr_score_layers.parquet"
    )
    _, semantic_path, _, semantic_sha = _validated_crossfit_semantic_source(
        source, source_manifest
    )
    if replay_source.get("score_layers_sha256") != sha256_file(score_path):
        raise ValueError("CRYCHIC replay score-layer checksum mismatch")
    if replay_source.get("semantic_availability_sha256") != semantic_sha:
        raise ValueError("CRYCHIC replay semantic checksum mismatch")

    condition_column = str(contract["condition_column"])
    score_columns = [
        "fold_id",
        "sample_id",
        "subject_id",
        condition_column,
        *EDGE_COLUMNS,
        "mechanistic_status",
        "mechanistic_reason_code",
    ]
    score_layers = pd.read_parquet(score_path, columns=score_columns)
    coverage = _validate_sample_layer(
        score_layers,
        condition_column=condition_column,
        samples_by_condition=contract["samples_by_condition"],
        subjects_by_condition=contract["subjects_by_condition"],
    )
    semantic = pd.read_parquet(
        semantic_path,
        columns=[
            "fold_id",
            "sample_id",
            "subject_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "availability_score",
            "status",
            "reason_code",
        ],
        filters=[("mode", "==", "state")],
    )
    direct = sender_specific_direct_scores(
        score_layers,
        semantic,
        condition_column=condition_column,
        edge_columns=EDGE_COLUMNS,
        response_transform="identity",
    )
    del score_layers, semantic
    recomputed = _sample_effects(
        direct,
        reference=str(contract["reference"]),
        target=str(contract["target"]),
        condition_column=condition_column,
    )
    official_effect_path = _bound_output(
        replay, replay_manifest, "sample_specific_directed_lr_effects.parquet"
    )
    official_rank_path = _bound_output(
        replay, replay_manifest, "condition_cell_pair_rankings.tsv"
    )
    official_effects = pd.read_parquet(official_effect_path)
    pd.testing.assert_frame_equal(
        recomputed.reset_index(drop=True),
        official_effects.reset_index(drop=True),
        check_exact=True,
        check_dtype=False,
    )
    official_rankings = pd.read_csv(official_rank_path, sep="\t")
    provenance = {
        "source_manifest": _input_record(source_manifest_path),
        "replay_manifest": _input_record(replay_manifest_path),
        "score_layers": _input_record(score_path),
        "semantic_availability": _input_record(semantic_path),
        "official_effects": _input_record(official_effect_path),
        "official_rankings": _input_record(official_rank_path),
        "sample_coverage": coverage,
    }
    return direct, official_effects, official_rankings, provenance


def _effect_z(effect: pd.Series, standard_error: pd.Series) -> np.ndarray:
    beta = pd.to_numeric(effect, errors="coerce").to_numpy(dtype=float)
    se = pd.to_numeric(standard_error, errors="coerce").to_numpy(dtype=float)
    z = np.full(len(beta), np.nan, dtype=float)
    positive = np.isfinite(beta) & np.isfinite(se) & (se > 0.0)
    z[positive] = beta[positive] / se[positive]
    separated = np.isfinite(beta) & (se == 0.0) & (beta != 0.0)
    z[separated] = np.sign(beta[separated]) * math.inf
    tied_zero = np.isfinite(beta) & (se == 0.0) & (beta == 0.0)
    z[tied_zero] = 0.0
    return z


def _effect_arm(
    effects: pd.DataFrame,
    *,
    dataset_id: str,
    arm: str,
    mode: str,
    threshold: float = 1.0,
) -> pd.DataFrame:
    required = {
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "reference_condition",
        "target_condition",
        "effect_target_minus_reference",
        "effect_standard_error_hc2",
        "n_samples_reference",
        "n_samples_target",
        "status",
        "reason_code",
    }
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"effect table is missing: {sorted(missing)}")
    result = effects.loc[
        :,
        [
            "sender",
            "receiver",
            "interaction_id",
            "ligand",
            "receptor",
            "effect_target_minus_reference",
            "effect_standard_error_hc2",
            "n_samples_reference",
            "n_samples_target",
            "status",
            "reason_code",
        ],
    ].copy()
    result = result.rename(
        columns={
            "effect_target_minus_reference": "effect",
            "effect_standard_error_hc2": "standard_error",
            "n_samples_reference": "n_reference",
            "n_samples_target": "n_target",
        }
    )
    observed = result["status"].eq("observed")
    effect = pd.to_numeric(result["effect"], errors="coerce")
    z = _effect_z(effect, result["standard_error"])
    result["test_statistic"] = z
    result["p_value"] = np.nan
    result["target_weight"] = 0.0
    result["reference_weight"] = 0.0
    if mode == "one_se":
        selected = (
            observed
            & effect.ne(0.0)
            & pd.Series(np.abs(z), index=result.index).ge(threshold)
        )
        result.loc[selected & effect.gt(0.0), "target_weight"] = 1.0
        result.loc[selected & effect.lt(0.0), "reference_weight"] = 1.0
        selection_rule = f"abs_hc2_z>={threshold:g}"
    elif mode == "wald":
        p_value = np.full(len(result), np.nan, dtype=float)
        p_value[observed.to_numpy()] = 2.0 * norm.sf(
            np.abs(z[observed.to_numpy()])
        )
        result["p_value"] = p_value
        selected = observed & pd.Series(p_value, index=result.index).lt(threshold)
        result.loc[selected & effect.gt(0.0), "target_weight"] = 1.0
        result.loc[selected & effect.lt(0.0), "reference_weight"] = 1.0
        selection_rule = f"nominal_gaussian_hc2_wald_p<{threshold:g}"
    elif mode == "continuous":
        signed_evidence = np.zeros(len(result), dtype=float)
        finite = observed.to_numpy() & np.isfinite(z)
        signed_evidence[finite] = 2.0 * (norm.cdf(np.abs(z[finite])) - 0.5)
        infinite = observed.to_numpy() & np.isinf(z)
        signed_evidence[infinite] = 1.0
        positive = observed.to_numpy() & effect.gt(0.0).to_numpy()
        negative = observed.to_numpy() & effect.lt(0.0).to_numpy()
        result.loc[positive, "target_weight"] = signed_evidence[positive]
        result.loc[negative, "reference_weight"] = signed_evidence[negative]
        selected = observed & effect.ne(0.0)
        selection_rule = "continuous_two_sided_hc2_sign_evidence_no_hard_selection"
    else:
        raise ValueError("mode must be one_se, wald, or continuous")
    result["selected_target"] = result["target_weight"].gt(0.0)
    result["selected_reference"] = result["reference_weight"].gt(0.0)
    result.insert(0, "dataset", dataset_id)
    result.insert(1, "arm", arm)
    result["reference_condition"] = str(effects["reference_condition"].iloc[0])
    result["target_condition"] = str(effects["target_condition"].iloc[0])
    result["selection_rule"] = selection_rule
    return result


def _wilcoxon_arm(
    direct: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    dataset_id: str,
    condition_column: str,
    reference: str,
    target: str,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    edge_columns = list(EDGE_COLUMNS)
    design = direct.loc[:, ["sample_id", condition_column]].drop_duplicates()
    if design["sample_id"].duplicated().any():
        raise ValueError("CRYCHIC sample IDs map to multiple conditions")
    samples = design["sample_id"].astype(str).tolist()
    pivot = direct.pivot(
        index=edge_columns,
        columns="sample_id",
        values="direct_response_score",
    ).reindex(columns=samples)
    effect_index = pd.MultiIndex.from_frame(effects.loc[:, edge_columns])
    pivot = pivot.reindex(effect_index)
    values = pivot.to_numpy(dtype=float)
    condition = design.set_index("sample_id").loc[samples, condition_column]
    reference_mask = condition.astype(str).eq(reference).to_numpy()
    target_mask = condition.astype(str).eq(target).to_numpy()
    reference_values = values[:, reference_mask]
    target_values = values[:, target_mask]
    n_reference = np.isfinite(reference_values).sum(axis=1)
    n_target = np.isfinite(target_values).sum(axis=1)
    supported = (n_reference >= MIN_WILCOXON_UNITS) & (
        n_target >= MIN_WILCOXON_UNITS
    )
    p_value = np.full(len(values), np.nan, dtype=float)
    u_statistic = np.full(len(values), np.nan, dtype=float)
    if supported.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic = mannwhitneyu(
                target_values[supported],
                reference_values[supported],
                alternative="two-sided",
                axis=1,
                method="asymptotic",
                use_continuity=True,
                nan_policy="omit",
            )
        p_value[supported] = np.asarray(statistic.pvalue, dtype=float)
        u_statistic[supported] = np.asarray(statistic.statistic, dtype=float)
    constant = np.zeros(len(values), dtype=bool)
    constant[supported] = np.nanmin(values[supported], axis=1) == np.nanmax(
        values[supported], axis=1
    )
    p_value[constant] = 1.0
    u_statistic[constant] = n_reference[constant] * n_target[constant] / 2.0
    target_sum = np.nansum(target_values, axis=1)
    reference_sum = np.nansum(reference_values, axis=1)
    target_mean = np.divide(
        target_sum,
        n_target,
        out=np.full(len(values), np.nan, dtype=float),
        where=n_target > 0,
    )
    reference_mean = np.divide(
        reference_sum,
        n_reference,
        out=np.full(len(values), np.nan, dtype=float),
        where=n_reference > 0,
    )
    effect = target_mean - reference_mean
    observed = supported & np.isfinite(p_value) & np.isfinite(effect)
    selected = observed & (p_value < alpha) & (effect != 0.0)
    result = effects.loc[
        :, ["sender", "receiver", "interaction_id", "ligand", "receptor"]
    ].copy()
    result.insert(0, "dataset", dataset_id)
    result.insert(1, "arm", "B")
    result["reference_condition"] = reference
    result["target_condition"] = target
    result["effect"] = effect
    result["standard_error"] = np.nan
    result["test_statistic"] = u_statistic
    result["p_value"] = p_value
    result["n_reference"] = n_reference
    result["n_target"] = n_target
    result["status"] = np.where(observed, "observed", "not_estimable")
    result["reason_code"] = np.where(
        observed, None, "fewer_than_four_finite_scores_per_condition"
    )
    result["target_weight"] = (selected & (effect > 0.0)).astype(float)
    result["reference_weight"] = (selected & (effect < 0.0)).astype(float)
    result["selected_target"] = result["target_weight"].gt(0.0)
    result["selected_reference"] = result["reference_weight"].gt(0.0)
    result["selection_rule"] = f"scseqcomm_fasterwilcox_equivalent_raw_p<{alpha:g}"
    return result


def _load_scseq_effects(
    score_path: Path,
    scseq_run: Path,
    *,
    reference: str,
    target: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest_path = scseq_run / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("scSeqCommDiff source run must be complete")
    metadata_record = manifest.get("metadata")
    if not isinstance(metadata_record, Mapping):
        raise ValueError("scSeqCommDiff run lacks metadata provenance")
    metadata_path = scseq_run / str(metadata_record.get("filename"))
    if metadata_record.get("sha256") != sha256_file(metadata_path):
        raise ValueError("scSeqCommDiff metadata checksum mismatch")
    design = pd.read_csv(metadata_path, sep="\t", dtype=str).loc[
        :, ["Sample_ID", "Condition_ID"]
    ].drop_duplicates()
    if design["Sample_ID"].duplicated().any():
        raise ValueError("scSeqCommDiff sample IDs map to multiple conditions")
    if set(design["Condition_ID"]) != {reference, target}:
        raise ValueError("scSeqCommDiff score conditions disagree with CRYCHIC")
    wide = pd.read_csv(score_path, sep="\t")
    identity = ["sender", "receiver", "interaction_id", "ligand", "receptor"]
    missing = set(identity).difference(wide.columns)
    samples = design["Sample_ID"].tolist()
    if missing or set(samples).difference(wide.columns):
        raise ValueError("scSeqCommDiff sample score export is incomplete")
    if wide.duplicated(["sender", "receiver", "interaction_id"]).any():
        raise ValueError("scSeqCommDiff sample score export has duplicate edges")
    long = wide.loc[:, [*identity, *samples]].melt(
        id_vars=identity,
        value_vars=samples,
        var_name="sample_id",
        value_name="score",
    )
    long = long.merge(
        design.rename(
            columns={"Sample_ID": "sample_id", "Condition_ID": "condition"}
        ),
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    long["subject_id"] = long["sample_id"]
    long["status"] = np.where(long["score"].notna(), "observed", "not_estimable")
    long["direct_response_transform"] = "identity"
    effects = adjusted_subject_directed_lr_effects(
        long,
        reference=reference,
        target=target,
        condition_column="condition",
        edge_columns=identity,
        score_column="score",
        status_column="status",
        min_subjects_per_condition=2,
    )
    provenance = {
        "manifest": _input_record(manifest_path),
        "metadata": _input_record(metadata_path),
        "sample_scores": _input_record(score_path),
        "rows": len(wide),
        "samples": len(samples),
    }
    return effects, provenance


def _pair_universe(effects: pd.DataFrame) -> pd.DataFrame:
    sender = effects["sender"].astype(str).to_numpy()
    receiver = effects["receiver"].astype(str).to_numpy()
    return pd.DataFrame(
        {
            "sender": np.minimum(sender, receiver),
            "receiver": np.maximum(sender, receiver),
        }
    ).drop_duplicates(ignore_index=True).sort_values(
        ["sender", "receiver"], kind="stable", ignore_index=True
    )


def _build_pair_rankings(
    edges: pd.DataFrame,
    pair_universe: pd.DataFrame,
    *,
    dataset_id: str,
    method: str,
    semantics: str,
) -> pd.DataFrame:
    working = edges.loc[edges["status"].eq("observed")].copy()
    sender = working["sender"].astype(str).to_numpy()
    receiver = working["receiver"].astype(str).to_numpy()
    working["_sender"] = np.minimum(sender, receiver)
    working["_receiver"] = np.maximum(sender, receiver)
    grouped = (
        working.groupby(["_sender", "_receiver"], observed=True, sort=True)
        .agg(
            target_score=("target_weight", "sum"),
            reference_score=("reference_weight", "sum"),
            target_selected=("selected_target", "sum"),
            reference_selected=("selected_reference", "sum"),
            estimable_directed_lr=("interaction_id", "size"),
        )
        .reset_index()
        .rename(columns={"_sender": "sender", "_receiver": "receiver"})
    )
    pairs = pair_universe.merge(
        grouped, on=["sender", "receiver"], how="left", validate="one_to_one"
    )
    for column in (
        "target_score",
        "reference_score",
        "target_selected",
        "reference_selected",
        "estimable_directed_lr",
    ):
        pairs[column] = pairs[column].fillna(0)
    observed_pair = pairs["estimable_directed_lr"].gt(0)
    target = str(edges["target_condition"].iloc[0])
    reference = str(edges["reference_condition"].iloc[0])

    def condition_rows(
        condition: str, score: str, selected: str
    ) -> pd.DataFrame:
        result = pairs.loc[:, ["sender", "receiver", "estimable_directed_lr"]].copy()
        result["condition"] = condition
        result["ranked_strength"] = pairs[score].where(observed_pair)
        result["condition_specific_directed_lr"] = pairs[selected]
        result["status"] = np.where(observed_pair, "observed", "not_estimable")
        result["reason_code"] = np.where(
            observed_pair, None, "no_estimable_directed_lr"
        )
        return result

    result = pd.concat(
        [
            condition_rows(target, "target_score", "target_selected"),
            condition_rows(reference, "reference_score", "reference_selected"),
        ],
        ignore_index=True,
    )
    result.insert(0, "dataset", dataset_id)
    result.insert(1, "method", method)
    result.insert(2, "method_version", METHOD_VERSION)
    result.insert(3, "resource", RESOURCE_ID)
    result.insert(4, "ranking_semantics", semantics)
    return result.sort_values(
        ["condition", "sender", "receiver"], kind="stable", ignore_index=True
    )


def _assert_a_parity(
    arm_a: pd.DataFrame, official: pd.DataFrame
) -> dict[str, object]:
    columns = [
        "condition",
        "sender",
        "receiver",
        "ranked_strength",
        "condition_specific_directed_lr",
        "estimable_directed_lr",
        "status",
    ]
    left = arm_a.loc[:, columns].reset_index(drop=True)
    right = official.loc[:, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)
    return {"rows": len(left), "exact": True}


def _native_ranking(path: Path, *, method: str) -> pd.DataFrame:
    result = pd.read_csv(path, sep="\t")
    required = {
        "dataset",
        "condition",
        "sender",
        "receiver",
        "ranked_strength",
        "status",
    }
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"native ranking is missing: {sorted(missing)}")
    result["method"] = method
    if "method_version" not in result:
        result["method_version"] = "native"
    if "resource" not in result:
        result["resource"] = RESOURCE_ID
    if "ranking_semantics" not in result:
        result["ranking_semantics"] = "native"
    if "estimable_directed_lr" not in result:
        result["estimable_directed_lr"] = np.nan
    return result


def _evaluate(
    rankings: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contract = _CONTRACTS[dataset]
    ranking_dataset = str(_SOURCE_CONTRACTS[dataset]["ranking_dataset_id"])
    scores, coverage = evaluate_rankings(
        rankings,
        expected,
        scenario="multi_sample",
        dataset_map={ranking_dataset: str(contract["truth_dataset"])},
        condition_map=dict(contract["condition_map"]),
        expected_filters=dict(contract["expected_filters"]),
        score_type="pos",
        weight_exponent=1.0,
        tie_policy="fgsea_native",
        exclude_self_pairs=True,
        ranking_statistic="raw_cardinality",
    )
    summary = (
        scores.groupby(
            ["method", "method_version", "resource", "ranking_semantics"],
            observed=True,
            sort=True,
        )["des"]
        .agg(["count", "median", "mean", "min", "max"])
        .reset_index()
    )
    return scores, coverage, summary


def _arm_diagnostics(
    edge_tables: Mapping[str, pd.DataFrame], rankings: pd.DataFrame
) -> pd.DataFrame:
    edge_counts: dict[str, dict[str, int]] = {}
    for arm, edges in edge_tables.items():
        observed = edges.loc[edges["status"].eq("observed")]
        effect = pd.to_numeric(observed["effect"], errors="coerce")
        edge_counts[arm] = {
            "positive_edges": int(effect.gt(0.0).sum()),
            "negative_edges": int(effect.lt(0.0).sum()),
            "zero_edges": int(effect.eq(0.0).sum()),
            "selected_target_edges": int(observed["selected_target"].sum()),
            "selected_reference_edges": int(observed["selected_reference"].sum()),
        }
    records: list[dict[str, object]] = []
    for (method, condition), group in rankings.groupby(
        ["method", "condition"], observed=True, sort=True
    ):
        observed = group.loc[
            group["status"].eq("observed")
            & group["sender"].astype(str).ne(group["receiver"].astype(str))
        ]
        score = pd.to_numeric(observed["ranked_strength"], errors="coerce")
        opportunity = pd.to_numeric(
            observed["estimable_directed_lr"], errors="coerce"
        )
        tied = score.duplicated(keep=False)
        correlation = (
            score.corr(opportunity, method="spearman")
            if score.nunique() > 1 and opportunity.nunique() > 1
            else np.nan
        )
        counts = edge_counts.get(str(method), {})
        records.append(
            {
                "method": method,
                "condition": condition,
                "observed_pairs": len(observed),
                "positive_edges": counts.get("positive_edges", 0),
                "negative_edges": counts.get("negative_edges", 0),
                "zero_edges": counts.get("zero_edges", 0),
                "selected_target_edges": counts.get("selected_target_edges", 0),
                "selected_reference_edges": counts.get(
                    "selected_reference_edges", 0
                ),
                "selected_edges_per_pair_median": float(score.median())
                if len(score)
                else np.nan,
                "selected_edges_per_pair_max": float(score.max())
                if len(score)
                else np.nan,
                "n_estimable_lr_per_pair_median": float(opportunity.median())
                if len(opportunity)
                else np.nan,
                "n_estimable_lr_per_pair_min": float(opportunity.min())
                if len(opportunity)
                else np.nan,
                "n_estimable_lr_per_pair_max": float(opportunity.max())
                if len(opportunity)
                else np.nan,
                "pair_score_vs_n_estimable_lr_spearman": correlation,
                "tie_fraction": float(tied.mean()) if len(tied) else 1.0,
                "zero_pair_fraction": float(score.eq(0.0).mean())
                if len(score)
                else 1.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _pairwise_diagnostics(
    edge_tables: Mapping[str, pd.DataFrame], rankings: pd.DataFrame
) -> pd.DataFrame:
    edge_metrics: dict[tuple[str, str], dict[str, object]] = {}
    arms = sorted(edge_tables)
    # CRYCHIC persists model-native interaction_* identifiers, whereas the
    # scSeqComm export carries harmonized ConnectomeDB identifiers.  The frozen
    # resource is one-to-one on simple ligand/receptor pairs, so biological
    # keys are the valid cross-method alignment axis.
    edge_key = ["sender", "receiver", "ligand", "receptor"]
    for index, left_arm in enumerate(arms):
        left = edge_tables[left_arm]
        left = left.loc[left["status"].eq("observed"), [*edge_key, "effect"]]
        for right_arm in arms[index + 1 :]:
            right = edge_tables[right_arm]
            right = right.loc[
                right["status"].eq("observed"), [*edge_key, "effect"]
            ]
            merged = left.merge(
                right,
                on=edge_key,
                how="inner",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            left_effect = pd.to_numeric(merged["effect_left"], errors="coerce")
            right_effect = pd.to_numeric(merged["effect_right"], errors="coerce")
            nonzero = left_effect.ne(0.0) & right_effect.ne(0.0)
            edge_metrics[(left_arm, right_arm)] = {
                "common_estimable_edges": len(merged),
                "common_nonzero_edges": int(nonzero.sum()),
                "edge_sign_concordance": float(
                    np.sign(left_effect.loc[nonzero]).eq(
                        np.sign(right_effect.loc[nonzero])
                    ).mean()
                )
                if nonzero.any()
                else np.nan,
                "edge_rank_spearman": left_effect.corr(
                    right_effect, method="spearman"
                )
                if left_effect.nunique() > 1 and right_effect.nunique() > 1
                else np.nan,
            }
    records: list[dict[str, object]] = []
    primary = rankings.loc[rankings["method"].isin(arms)].copy()
    for condition, condition_table in primary.groupby(
        "condition", observed=True, sort=True
    ):
        for index, left_arm in enumerate(arms):
            left = condition_table.loc[
                condition_table["method"].eq(left_arm)
                & condition_table["status"].eq("observed")
                & condition_table["sender"].astype(str).ne(
                    condition_table["receiver"].astype(str)
                ),
                ["sender", "receiver", "ranked_strength"],
            ]
            for right_arm in arms[index + 1 :]:
                right = condition_table.loc[
                    condition_table["method"].eq(right_arm)
                    & condition_table["status"].eq("observed")
                    & condition_table["sender"].astype(str).ne(
                        condition_table["receiver"].astype(str)
                    ),
                    ["sender", "receiver", "ranked_strength"],
                ]
                merged = left.merge(
                    right,
                    on=["sender", "receiver"],
                    how="inner",
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                left_score = pd.to_numeric(
                    merged["ranked_strength_left"], errors="coerce"
                )
                right_score = pd.to_numeric(
                    merged["ranked_strength_right"], errors="coerce"
                )

                def top_ten(table: pd.DataFrame) -> set[tuple[str, str]]:
                    ordered = table.sort_values(
                        ["ranked_strength", "sender", "receiver"],
                        ascending=[False, True, True],
                        kind="stable",
                    ).head(10)
                    return set(
                        ordered.loc[:, ["sender", "receiver"]].itertuples(
                            index=False, name=None
                        )
                    )

                left_top = top_ten(left)
                right_top = top_ten(right)
                union = left_top | right_top
                metrics = edge_metrics[(left_arm, right_arm)]
                records.append(
                    {
                        "condition": condition,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        **metrics,
                        "common_estimable_pairs": len(merged),
                        "pair_rank_spearman": left_score.corr(
                            right_score, method="spearman"
                        )
                        if left_score.nunique() > 1 and right_score.nunique() > 1
                        else np.nan,
                        "top10_pair_overlap_count": len(left_top & right_top),
                        "top10_pair_overlap_jaccard": len(left_top & right_top)
                        / len(union)
                        if union
                        else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def _pair_rank_comparison(
    rankings: pd.DataFrame,
    expected: pd.DataFrame,
    arm_a_edges: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    contract = _CONTRACTS[dataset]
    base = rankings.loc[
        rankings["method"].eq("A")
        & rankings["status"].eq("observed")
        & rankings["sender"].astype(str).ne(rankings["receiver"].astype(str)),
        ["dataset", "condition", "sender", "receiver"],
    ].copy()
    base["unordered_pair"] = base["sender"].astype(str) + " | " + base[
        "receiver"
    ].astype(str)

    truth = expected.loc[
        expected["scenario"].eq("multi_sample")
        & pd.to_numeric(expected["top_fraction"], errors="coerce").eq(0.1)
    ].copy()
    for column, value in dict(contract["expected_filters"]).items():
        truth = truth.loc[truth[column].astype(str).eq(value)].copy()
    reverse_condition = {
        str(target): str(source)
        for source, target in dict(contract["condition_map"]).items()
    }
    truth["condition"] = truth["condition"].astype(str).replace(reverse_condition)
    truth = truth.loc[
        truth["sender"].astype(str).ne(truth["receiver"].astype(str)),
        ["condition", "sender", "receiver", "is_expected", "spatial_rank"],
    ].rename(
        columns={
            "is_expected": "expected_spatial_top10pct",
            "spatial_rank": "expected_spatial_rank",
        }
    )
    base = base.merge(
        truth,
        on=["condition", "sender", "receiver"],
        how="left",
        validate="one_to_one",
    )

    methods = ("A", "B", "C", "D", "F", "R_scseq_native", "R_liana_native")
    for method in methods:
        table = rankings.loc[
            rankings["method"].eq(method)
            & rankings["sender"].astype(str).ne(rankings["receiver"].astype(str)),
            [
                "condition",
                "sender",
                "receiver",
                "ranked_strength",
                "estimable_directed_lr",
                "status",
            ],
        ].copy()
        observed = table["status"].eq("observed")
        table["_rank"] = np.nan
        table.loc[observed, "_rank"] = (
            table.loc[observed]
            .groupby("condition", observed=True, sort=False)["ranked_strength"]
            .rank(method="min", ascending=False)
        )
        table = table.rename(
            columns={
                "ranked_strength": f"{method}_score",
                "_rank": f"{method}_rank",
                "estimable_directed_lr": f"{method}_n_estimable_lr",
                "status": f"{method}_status",
            }
        )
        base = base.merge(
            table,
            on=["condition", "sender", "receiver"],
            how="left",
            validate="one_to_one",
        )

    effects = arm_a_edges.loc[arm_a_edges["status"].eq("observed")].copy()
    sender = effects["sender"].astype(str).to_numpy()
    receiver = effects["receiver"].astype(str).to_numpy()
    effects["_sender"] = np.minimum(sender, receiver)
    effects["_receiver"] = np.maximum(sender, receiver)
    mean_effect = (
        effects.groupby(["_sender", "_receiver"], observed=True, sort=True)[
            "effect"
        ]
        .mean()
        .rename("mean_effect_target_minus_reference")
        .reset_index()
        .rename(columns={"_sender": "sender", "_receiver": "receiver"})
    )
    base = base.merge(
        mean_effect,
        on=["sender", "receiver"],
        how="left",
        validate="many_to_one",
    )
    target = str(arm_a_edges["target_condition"].iloc[0])
    base["condition_oriented_mean_effect"] = np.where(
        base["condition"].astype(str).eq(target),
        base["mean_effect_target_minus_reference"],
        -base["mean_effect_target_minus_reference"],
    )
    base["crychic_rank"] = base["A_rank"]
    base["scseqcommdiff_rank"] = base["R_scseq_native_rank"]
    base["liana_rank"] = base["R_liana_native_rank"]
    base["n_estimable_lr"] = base["A_n_estimable_lr"]
    base["hard_selected_count"] = base["A_score"]
    base["expected_selected_count"] = base["F_score"]
    for unavailable in (
        "occurrence_effect",
        "program_effect",
        "downstream_support",
        "sender_abundance",
        "receiver_abundance",
    ):
        base[unavailable] = np.nan
    base["unavailable_component_reason"] = (
        "not_persisted_in_frozen_component_swap_inputs"
    )
    return base.sort_values(
        ["condition", "crychic_rank", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )


def _threshold_rankings(
    effects_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    effects_d: pd.DataFrame,
    pair_universe: pd.DataFrame,
    *,
    dataset_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for arm, effects in (("A", effects_a), ("D", effects_d)):
        for threshold in (0.0, 0.5, 1.0, 1.645, 1.96, 2.576):
            edge = _effect_arm(
                effects,
                dataset_id=dataset_id,
                arm=arm,
                mode="one_se",
                threshold=threshold,
            )
            method = f"{arm}_abs_hc2_z_ge_{threshold:g}"
            tables.append(
                _build_pair_rankings(
                    edge,
                    pair_universe,
                    dataset_id=dataset_id,
                    method=method,
                    semantics=edge["selection_rule"].iloc[0],
                )
            )
            metadata.append(
                {
                    "method": method,
                    "arm": arm,
                    "threshold_type": "absolute_hc2_z",
                    "threshold": threshold,
                }
            )
    for arm, base in (("B", arm_b),):
        effect = pd.to_numeric(base["effect"], errors="coerce")
        observed = base["status"].eq("observed")
        p_value = pd.to_numeric(base["p_value"], errors="coerce")
        for threshold in (0.01, 0.025, 0.05, 0.1, 0.2):
            edge = base.copy()
            selected = observed & p_value.lt(threshold) & effect.ne(0.0)
            edge["target_weight"] = (selected & effect.gt(0.0)).astype(float)
            edge["reference_weight"] = (selected & effect.lt(0.0)).astype(float)
            edge["selected_target"] = edge["target_weight"].gt(0.0)
            edge["selected_reference"] = edge["reference_weight"].gt(0.0)
            method = f"{arm}_raw_p_lt_{threshold:g}"
            tables.append(
                _build_pair_rankings(
                    edge,
                    pair_universe,
                    dataset_id=dataset_id,
                    method=method,
                    semantics=f"raw_p<{threshold:g}",
                )
            )
            metadata.append(
                {
                    "method": method,
                    "arm": arm,
                    "threshold_type": "raw_p",
                    "threshold": threshold,
                }
            )
    for threshold in (0.01, 0.025, 0.05, 0.1, 0.2):
        edge = _effect_arm(
            effects_a,
            dataset_id=dataset_id,
            arm="C",
            mode="wald",
            threshold=threshold,
        )
        method = f"C_nominal_wald_p_lt_{threshold:g}"
        tables.append(
            _build_pair_rankings(
                edge,
                pair_universe,
                dataset_id=dataset_id,
                method=method,
                semantics=edge["selection_rule"].iloc[0],
            )
        )
        metadata.append(
            {
                "method": method,
                "arm": "C",
                "threshold_type": "nominal_wald_p",
                "threshold": threshold,
            }
        )
    return pd.concat(tables, ignore_index=True), pd.DataFrame(metadata)


def run(
    *,
    dataset: str,
    crychic_source_run: Path,
    crychic_replay_run: Path,
    scseq_run: Path,
    scseq_sample_scores: Path,
    scseq_native_ranking: Path,
    liana_native_ranking: Path,
    expected_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if dataset not in _CONTRACTS:
        raise ValueError("dataset must be 'kuppe' or 'ms'")
    started = time.perf_counter()
    output = prepare_output(output_dir, overwrite=overwrite)
    source_contract = _SOURCE_CONTRACTS[dataset]
    dataset_id = str(source_contract["ranking_dataset_id"])
    reference = str(source_contract["reference"])
    target = str(source_contract["target"])
    condition_column = str(source_contract["condition_column"])
    expected_sha = sha256_file(expected_path)
    if expected_sha != _CONTRACTS[dataset]["expected_sha256"]:
        raise ValueError(
            "spatial expected-set checksum is not the frozen Figure 3 input"
        )

    direct, effects_a, official_a, crychic_provenance = _load_crychic_inputs(
        crychic_source_run.resolve(),
        crychic_replay_run.resolve(),
        dataset=dataset,
    )
    pair_universe = _pair_universe(effects_a)
    arm_a = _effect_arm(
        effects_a, dataset_id=dataset_id, arm="A", mode="one_se", threshold=1.0
    )
    arm_b = _wilcoxon_arm(
        direct,
        effects_a,
        dataset_id=dataset_id,
        condition_column=condition_column,
        reference=reference,
        target=target,
    )
    del direct
    arm_c = _effect_arm(
        effects_a, dataset_id=dataset_id, arm="C", mode="wald", threshold=ALPHA
    )
    effects_d, scseq_provenance = _load_scseq_effects(
        scseq_sample_scores,
        scseq_run,
        reference=reference,
        target=target,
    )
    arm_d = _effect_arm(
        effects_d, dataset_id=dataset_id, arm="D", mode="one_se", threshold=1.0
    )
    arm_f = _effect_arm(
        effects_a, dataset_id=dataset_id, arm="F", mode="continuous"
    )
    edge_tables = {"A": arm_a, "B": arm_b, "C": arm_c, "D": arm_d, "F": arm_f}
    edge_table = pd.concat(edge_tables.values(), ignore_index=True)

    pair_tables = {
        arm: _build_pair_rankings(
            edges,
            pair_universe,
            dataset_id=dataset_id,
            method=arm,
            semantics=str(edges["selection_rule"].iloc[0]),
        )
        for arm, edges in edge_tables.items()
    }
    parity = _assert_a_parity(pair_tables["A"], official_a)
    component_rankings = pd.concat(pair_tables.values(), ignore_index=True)
    scseq_native = _native_ranking(scseq_native_ranking, method="R_scseq_native")
    liana_native = _native_ranking(liana_native_ranking, method="R_liana_native")
    all_rankings = pd.concat(
        [component_rankings, scseq_native, liana_native], ignore_index=True
    )
    expected = pd.read_csv(expected_path, sep="\t")
    des_scores, des_coverage, des_summary = _evaluate(
        all_rankings, expected, dataset=dataset
    )
    diagnostics = _arm_diagnostics(edge_tables, component_rankings)
    pairwise = _pairwise_diagnostics(edge_tables, component_rankings)
    pair_comparison = _pair_rank_comparison(
        all_rankings, expected, arm_a, dataset=dataset
    )

    threshold_rankings, threshold_metadata = _threshold_rankings(
        effects_a,
        arm_b,
        effects_d,
        pair_universe,
        dataset_id=dataset_id,
    )
    threshold_scores, threshold_coverage, threshold_summary = _evaluate(
        threshold_rankings, expected, dataset=dataset
    )
    threshold_summary = threshold_metadata.merge(
        threshold_summary, on="method", how="left", validate="one_to_one"
    )

    availability = pd.DataFrame.from_records(
        [
            {
                "arm": "A",
                "status": "complete",
                "fidelity": "exact",
                "edge_score": "CRYCHIC sender-specific held-out score",
                "test": "sample-level HC2 contrast",
                "selection": "one-SE stability",
                "pair_score": "hard count",
                "reason_code": None,
            },
            {
                "arm": "B",
                "status": "complete",
                "fidelity": "exact_head_swap",
                "edge_score": "CRYCHIC sender-specific held-out score",
                "test": "scSeqCommDiff-equivalent Mann-Whitney",
                "selection": "raw p<0.05",
                "pair_score": "hard count",
                "reason_code": None,
            },
            {
                "arm": "C",
                "status": "complete",
                "fidelity": "declared_proxy",
                "edge_score": "CRYCHIC sender-specific held-out score",
                "test": "Gaussian HC2 Wald analogue",
                "selection": "nominal raw p<0.05",
                "pair_score": "hard count",
                "reason_code": (
                    "bounded_crychic_score_is_not_a_raw_count_and_cannot_enter_"
                    "pydeseq2_negative_binomial_likelihood"
                ),
            },
            {
                "arm": "D",
                "status": "complete",
                "fidelity": "exact_head_swap",
                "edge_score": "native scSeqCommDiff per-sample S_inter",
                "test": "sample-level HC2 contrast",
                "selection": "one-SE stability",
                "pair_score": "hard count",
                "reason_code": None,
            },
            {
                "arm": "E",
                "status": "not_estimable",
                "fidelity": "unavailable",
                "edge_score": "LIANA interaction_stat",
                "test": "CRYCHIC hypergraph reweighting",
                "selection": "calibrated soft posterior",
                "pair_score": "posterior expected count",
                "reason_code": (
                    "no_checksum_bound_full_pipeline_null_and_g3p_calibrated_"
                    "soft_posterior_in_current_run"
                ),
            },
            {
                "arm": "F",
                "status": "complete",
                "fidelity": "diagnostic_uncalibrated",
                "edge_score": "CRYCHIC sender-specific held-out score",
                "test": "current HC2 effect/SE",
                "selection": "none",
                "pair_score": "continuous signed HC2 evidence count",
                "reason_code": "not_a_released_posterior_probability",
            },
        ]
    )

    paths: dict[str, tuple[Path, pd.DataFrame]] = {
        "directed_edge_statistics.parquet": (
            output / "directed_edge_statistics.parquet",
            edge_table,
        ),
        "condition_cell_pair_rankings.tsv": (
            output / "condition_cell_pair_rankings.tsv",
            all_rankings,
        ),
        "spatial_des_scores.tsv": (output / "spatial_des_scores.tsv", des_scores),
        "spatial_des_coverage.tsv": (
            output / "spatial_des_coverage.tsv",
            des_coverage,
        ),
        "spatial_des_summary.tsv": (
            output / "spatial_des_summary.tsv",
            des_summary,
        ),
        "arm_diagnostics.tsv": (output / "arm_diagnostics.tsv", diagnostics),
        "pairwise_diagnostics.tsv": (output / "pairwise_diagnostics.tsv", pairwise),
        "pair_rank_comparison.tsv": (
            output / "pair_rank_comparison.tsv",
            pair_comparison,
        ),
        "arm_availability.tsv": (output / "arm_availability.tsv", availability),
        "threshold_path_rankings.tsv": (
            output / "threshold_path_rankings.tsv",
            threshold_rankings,
        ),
        "threshold_path_scores.tsv": (
            output / "threshold_path_scores.tsv",
            threshold_scores,
        ),
        "threshold_path_coverage.tsv": (
            output / "threshold_path_coverage.tsv",
            threshold_coverage,
        ),
        "threshold_path_summary.tsv": (
            output / "threshold_path_summary.tsv",
            threshold_summary,
        ),
    }
    for name, (path, table) in paths.items():
        if name.endswith(".parquet"):
            table.to_parquet(path, index=False, compression="zstd")
        else:
            table.to_csv(path, sep="\t", index=False, lineterminator="\n")

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "dataset_id": dataset_id,
        "reference": reference,
        "target": target,
        "algorithm_modified": False,
        "component_swap_only": True,
        "spatial_des": {
            "scenario": "multi_sample",
            "score_type": "pos",
            "weight_exponent": 1.0,
            "tie_policy": "fgsea_native",
            "ranking_statistic": "raw_cardinality",
            "self_pairs": "excluded",
        },
        "arm_a_parity": parity,
        "arm_availability": availability.to_dict(orient="records"),
        "implementation": {
            "script_sha256": sha256_file(Path(__file__)),
            "scseq_sample_exporter_sha256": sha256_file(
                Path(__file__).with_name("export_scseqcomm_sample_scores.R")
            ),
        },
        "inputs": {
            "crychic": crychic_provenance,
            "scseqcommdiff": scseq_provenance,
            "scseq_native_ranking": _input_record(scseq_native_ranking),
            "liana_native_ranking": _input_record(liana_native_ranking),
            "expected_sets": _input_record(expected_path),
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(
                process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "code": git_metadata(Path(__file__).resolve().parents[2]),
        "outputs": {
            name: _output_record(path, table) for name, (path, table) in paths.items()
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(_CONTRACTS), required=True)
    parser.add_argument("--crychic-source-run", type=Path, required=True)
    parser.add_argument("--crychic-replay-run", type=Path, required=True)
    parser.add_argument("--scseq-run", type=Path, required=True)
    parser.add_argument("--scseq-sample-scores", type=Path, required=True)
    parser.add_argument("--scseq-native-ranking", type=Path, required=True)
    parser.add_argument("--liana-native-ranking", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        dataset=args.dataset,
        crychic_source_run=args.crychic_source_run,
        crychic_replay_run=args.crychic_replay_run,
        scseq_run=args.scseq_run,
        scseq_sample_scores=args.scseq_sample_scores,
        scseq_native_ranking=args.scseq_native_ranking,
        liana_native_ranking=args.liana_native_ranking,
        expected_path=args.expected,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "output": str(args.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
