"""Evaluate bounded RC11 rankings against resampled spatial DES endpoints.

This evaluator consumes persisted RC11 cell-pair rankings and persisted spatial
truth tables.  It never refits CRYCHIC.  The available RC11 table is an
unordered, pair-aggregated ranking, so event-level DES variants are emitted as
typed ``not_estimable`` records rather than approximated from pair scores.

Bootstrap, leave-one-slice-out, and condition-label permutation operate on the
spatial endpoint only while the candidate ranking remains fixed.  They are
diagnostic endpoint-resampling analyses, not full-pipeline inferential tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.literature.evaluate_spatial_des_benchmark import evaluate_rankings

SCHEMA_VERSION = "crychic-bounded-des-extensions-v1"
BOUNDED_SCHEMA_VERSION = "crychic-bounded-core-evidence-real-v1"
METHOD_ID = "CRYCHIC_RC11_bounded_core_evidence"
TOP_FRACTIONS = (0.1, 0.2, 0.3, 0.4)
EVENT_BUDGETS = (100, 250, 500, 1000)
MECHANISMS = ("contact", "ecm_receptor", "secreted", "diffusible_long_range")

DatasetName = Literal["kuppe", "ms"]
ResampleKind = Literal[
    "slice_bootstrap", "leave_one_slice_out", "endpoint_condition_permutation"
]


@dataclass(frozen=True, slots=True)
class DatasetContract:
    dataset: DatasetName
    role: str
    independent_validation: bool
    truth_schema: str
    truth_dataset_id: str
    source_value_column: str
    source_output_key: str
    reference_condition: str
    target_condition: str
    candidate_condition_map: Mapping[str, str]
    expected_filters: Mapping[str, str]


CONTRACTS: Mapping[DatasetName, DatasetContract] = {
    "kuppe": DatasetContract(
        dataset="kuppe",
        role="development",
        independent_validation=False,
        truth_schema="crychic-kuppe-misty-des-truth-v1",
        truth_dataset_id="Kuppe_MI_spatial_CTRL_vs_IZ",
        source_value_column="spatial_importance",
        source_output_key="sample_pair_strengths",
        reference_condition="CTRL",
        target_condition="IZ",
        candidate_condition_map={"CTRL": "CTRL", "IZ": "IZ"},
        expected_filters={"variant": "spatial_neighbor_max"},
    ),
    "ms": DatasetContract(
        dataset="ms",
        role="independent_validation",
        independent_validation=True,
        truth_schema="crychic-ms-spatial-des-truth-v1",
        truth_dataset_id="lerma_martin_ms_ctrl_vs_chronic_active",
        source_value_column="pearson_r",
        source_output_key="sample_correlations",
        reference_condition="control",
        target_condition="chronic_active",
        candidate_condition_map={"Ctrl": "control", "CA": "chronic_active"},
        expected_filters={},
    ),
}


@dataclass(frozen=True, slots=True)
class BoundInputs:
    contract: DatasetContract
    bounded_manifest_path: Path
    truth_manifest_path: Path
    bounded_manifest: Mapping[str, Any]
    truth_manifest: Mapping[str, Any]
    candidate: pd.DataFrame
    stored_scores: pd.DataFrame
    stored_coverage: pd.DataFrame
    expected: pd.DataFrame
    sample_values: pd.DataFrame
    input_records: Mapping[str, Any]


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


def _bound_output(
    root: Path,
    manifest: Mapping[str, Any],
    key: str,
    *,
    expected_filename: str | None = None,
) -> Path:
    outputs = manifest.get("outputs")
    record = outputs.get(key) if isinstance(outputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"manifest does not bind output {key!r}: {root}")
    filename = record.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"manifest output {key!r} has an invalid filename")
    if expected_filename is not None and filename != expected_filename:
        raise ValueError(f"manifest output {key!r} filename changed")
    path = root / filename
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"bound output checksum mismatch: {path}")
    rows = record.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError(f"manifest output {key!r} has an invalid row count")
    return path


def _read_bound_table(
    root: Path,
    manifest: Mapping[str, Any],
    key: str,
    *,
    expected_filename: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    path = _bound_output(
        root, manifest, key, expected_filename=expected_filename
    )
    table = pd.read_csv(path, sep="\t", low_memory=False)
    record = cast(Mapping[str, Any], cast(Mapping[str, Any], manifest["outputs"])[key])
    if len(table) != int(record["rows"]):
        raise ValueError(f"bound output row count mismatch: {path}")
    return table, path


def _validate_typed_values(
    table: pd.DataFrame,
    *,
    value_column: str,
    status_column: str = "status",
    label: str,
) -> None:
    missing = {value_column, status_column}.difference(table.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    status = table[status_column]
    if status.isna().any() or not status.map(
        lambda value: isinstance(value, str) and value == value.strip() and bool(value)
    ).all():
        raise ValueError(f"{label} statuses must be canonical strings")
    numeric = pd.to_numeric(table[value_column], errors="coerce")
    supplied = table[value_column].notna()
    if (supplied & numeric.isna()).any() or np.isinf(numeric.dropna()).any():
        raise ValueError(f"{label}.{value_column} must be finite or missing")
    observed = status.astype(str).eq("observed")
    if numeric.loc[observed].isna().any():
        raise ValueError(f"{label} observed rows require finite values")
    if numeric.loc[~observed].notna().any():
        raise ValueError(
            f"{label} non-observed rows must stay missing; zero imputation is forbidden"
        )


def _load_inputs(bounded_run: Path, truth_manifest_path: Path) -> BoundInputs:
    bounded_run = bounded_run.resolve()
    bounded_manifest_path = bounded_run / "manifest.json"
    bounded_manifest = _read_json(bounded_manifest_path)
    dataset = bounded_manifest.get("dataset")
    if dataset not in CONTRACTS:
        raise ValueError("bounded run dataset must be kuppe or ms")
    contract = CONTRACTS[cast(DatasetName, dataset)]
    acceptance = bounded_manifest.get("acceptance")
    if (
        bounded_manifest.get("schema_version") != BOUNDED_SCHEMA_VERSION
        or bounded_manifest.get("status") != "complete"
        or bounded_manifest.get("method") != METHOD_ID
        or bounded_manifest.get("formal_inference_allowed") is not False
        or not isinstance(acceptance, Mapping)
        or acceptance.get("dataset_role")
        != ("validation" if contract.independent_validation else "development")
        or acceptance.get("independent_validation")
        is not contract.independent_validation
    ):
        raise ValueError("bounded run provenance or dataset role is invalid")

    candidate, candidate_path = _read_bound_table(
        bounded_run,
        bounded_manifest,
        "candidate_condition_cell_pair_rankings.tsv",
    )
    stored_scores, stored_scores_path = _read_bound_table(
        bounded_run, bounded_manifest, "spatial_des_scores.tsv"
    )
    stored_coverage, stored_coverage_path = _read_bound_table(
        bounded_run, bounded_manifest, "spatial_des_coverage.tsv"
    )
    _validate_typed_values(
        candidate,
        value_column="ranked_strength",
        label="bounded candidate ranking",
    )
    if set(candidate["method"].astype(str)) != {METHOD_ID}:
        raise ValueError("candidate ranking contains an unexpected method")
    if candidate.duplicated(["condition", "sender", "receiver"]).any():
        raise ValueError("candidate ranking keys must be unique")
    expected_conditions = set(contract.candidate_condition_map)
    if set(candidate["condition"].astype(str)) != expected_conditions:
        raise ValueError("candidate conditions disagree with the dataset contract")

    truth_manifest_path = truth_manifest_path.resolve()
    truth_root = truth_manifest_path.parent
    truth_manifest = _read_json(truth_manifest_path)
    if (
        truth_manifest.get("schema_version") != contract.truth_schema
        or truth_manifest.get("status") != "complete"
        or truth_manifest.get("dataset_id") != contract.truth_dataset_id
    ):
        raise ValueError("spatial truth manifest does not match the dataset contract")
    expected, expected_path = _read_bound_table(
        truth_root, truth_manifest, "expected_sets"
    )
    sample_values, sample_values_path = _read_bound_table(
        truth_root, truth_manifest, contract.source_output_key
    )
    bounded_expected = bounded_manifest.get("inputs", {}).get("expected_sets", {})
    if not isinstance(bounded_expected, Mapping) or (
        bounded_expected.get("sha256") != sha256_file(expected_path)
    ):
        raise ValueError("bounded run and truth manifest bind different expected sets")
    _validate_typed_values(
        sample_values,
        value_column=contract.source_value_column,
        label="spatial sample endpoint",
    )

    required_sample = {
        "sample_id",
        "subject_id",
        "condition",
        "sender",
        "receiver",
        contract.source_value_column,
        "status",
        "reason_code",
    }
    missing_sample = required_sample.difference(sample_values.columns)
    if missing_sample:
        raise ValueError(
            f"spatial sample endpoint is missing columns: {sorted(missing_sample)}"
        )
    if contract.dataset == "kuppe":
        if "variant" not in sample_values:
            raise ValueError("Kuppe sample endpoint lacks spatial variant")
        sample_values = sample_values.loc[
            sample_values["variant"].astype(str).eq("spatial_neighbor_max")
        ].copy()
    if sample_values.empty:
        raise ValueError("primary spatial sample endpoint is empty")
    sample_conditions = set(sample_values["condition"].astype(str))
    if sample_conditions != {
        contract.reference_condition,
        contract.target_condition,
    }:
        raise ValueError("spatial sample conditions disagree with the contract")

    return BoundInputs(
        contract=contract,
        bounded_manifest_path=bounded_manifest_path,
        truth_manifest_path=truth_manifest_path,
        bounded_manifest=bounded_manifest,
        truth_manifest=truth_manifest,
        candidate=candidate,
        stored_scores=stored_scores,
        stored_coverage=stored_coverage,
        expected=expected,
        sample_values=sample_values,
        input_records={
            "bounded_manifest": _input_record(bounded_manifest_path),
            "candidate_rankings": _input_record(candidate_path),
            "stored_spatial_des_scores": _input_record(stored_scores_path),
            "stored_spatial_des_coverage": _input_record(stored_coverage_path),
            "truth_manifest": _input_record(truth_manifest_path),
            "expected_sets": _input_record(expected_path),
            "sample_endpoint": _input_record(sample_values_path),
        },
    )


def _normalize_sample_values(inputs: BoundInputs) -> pd.DataFrame:
    contract = inputs.contract
    columns = [
        "sample_id",
        "subject_id",
        "condition",
        "sender",
        "receiver",
        contract.source_value_column,
        "status",
        "reason_code",
    ]
    result = inputs.sample_values.loc[:, columns].rename(
        columns={contract.source_value_column: "endpoint_value"}
    )
    result["endpoint_value"] = pd.to_numeric(
        result["endpoint_value"], errors="coerce"
    )
    return result.sort_values(
        ["condition", "sample_id", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )


def _build_spatial_expected(
    sample_values: pd.DataFrame,
    *,
    contract: DatasetContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the frozen multi-sample spatial ranking from compact slice rows."""

    required = {
        "sample_id",
        "condition",
        "sender",
        "receiver",
        "endpoint_value",
        "status",
    }
    missing = required.difference(sample_values.columns)
    if missing:
        raise ValueError(f"resampled endpoint is missing columns: {sorted(missing)}")
    _validate_typed_values(
        sample_values,
        value_column="endpoint_value",
        label="resampled spatial endpoint",
    )
    all_pairs = (
        sample_values.loc[:, ["sender", "receiver"]]
        .drop_duplicates()
        .sort_values(["sender", "receiver"], kind="stable")
    )
    observed = sample_values.loc[sample_values["status"].eq("observed")]
    units = (
        observed.groupby(
            ["sample_id", "condition", "sender", "receiver"],
            sort=True,
            observed=True,
            as_index=False,
        )["endpoint_value"]
        .mean()
        .sort_values(
            ["condition", "sample_id", "sender", "receiver"], kind="stable"
        )
    )
    records: list[dict[str, object]] = []
    reference_name = contract.reference_condition
    target_name = contract.target_condition
    for sender, receiver in all_pairs.itertuples(index=False, name=None):
        pair = units.loc[
            units["sender"].astype(str).eq(str(sender))
            & units["receiver"].astype(str).eq(str(receiver))
        ]
        reference = pair.loc[
            pair["condition"].eq(reference_name), "endpoint_value"
        ].to_numpy(dtype=float)
        target = pair.loc[
            pair["condition"].eq(target_name), "endpoint_value"
        ].to_numpy(dtype=float)
        reference_mean = float(np.mean(reference)) if reference.size else math.nan
        target_mean = float(np.mean(target)) if target.size else math.nan
        effect = target_mean - reference_mean
        if reference.size >= 2 and target.size >= 2:
            test = mannwhitneyu(
                target,
                reference,
                alternative="two-sided",
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
            status = "observed"
            reason_code: str | None = None
        else:
            statistic = p_value = math.nan
            status = "not_estimable"
            reason_code = "fewer_than_two_observed_slices_in_a_condition"
        direction = (
            target_name
            if effect > 0.0
            else reference_name
            if effect < 0.0
            else "tied"
        )
        records.append(
            {
                "dataset": contract.truth_dataset_id,
                "scenario": "multi_sample",
                "sender": str(sender),
                "receiver": str(receiver),
                "mean_reference": reference_mean,
                "mean_target": target_mean,
                "effect_target_minus_reference": effect,
                "abs_effect": abs(effect),
                "u_statistic": statistic,
                "p_value": p_value,
                "n_reference_units": int(reference.size),
                "n_target_units": int(target.size),
                "direction_condition": direction,
                "status": status,
                "reason_code": reason_code,
            }
        )
    rankings = pd.DataFrame.from_records(records)
    eligible = (
        rankings["status"].eq("observed")
        & rankings["effect_target_minus_reference"].ne(0.0)
        & rankings["p_value"].notna()
    )
    ordered = rankings.loc[eligible].sort_values(
        ["p_value", "abs_effect", "sender", "receiver"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    rank_map = pd.Series(
        np.arange(1, len(ordered) + 1, dtype=np.int64), index=ordered.index
    )
    rankings["spatial_rank"] = rank_map.reindex(rankings.index).astype("Int64")
    rankings["ranking_eligible"] = eligible
    rankings = rankings.sort_values(
        ["spatial_rank", "sender", "receiver"],
        kind="stable",
        na_position="last",
        ignore_index=True,
    )

    rankable_pairs = int(eligible.sum())
    expected_records: list[dict[str, object]] = []
    for condition in (reference_name, target_name):
        for fraction in TOP_FRACTIONS:
            top_count = math.floor(fraction * rankable_pairs) if rankable_pairs else 0
            for row in rankings.itertuples(index=False):
                rank = row.spatial_rank
                expected_records.append(
                    {
                        "dataset": contract.truth_dataset_id,
                        "scenario": "multi_sample",
                        "condition": condition,
                        "top_fraction": fraction,
                        "sender": row.sender,
                        "receiver": row.receiver,
                        "is_expected": bool(
                            not pd.isna(rank)
                            and int(rank) <= top_count
                            and row.direction_condition == condition
                        ),
                        "spatial_rank": rank,
                        "rankable_pairs": rankable_pairs,
                        "top_count": top_count,
                        "top_count_rule": "floor",
                        "cell_pair_direction": "unordered_canonical",
                    }
                )
    expected = pd.DataFrame.from_records(expected_records)
    expected["spatial_rank"] = expected["spatial_rank"].astype("Int64")
    expected = expected.sort_values(
        ["condition", "top_fraction", "sender", "receiver"],
        kind="stable",
        ignore_index=True,
    )
    return rankings, expected


def _frozen_primary_expected(inputs: BoundInputs) -> pd.DataFrame:
    expected = inputs.expected.loc[
        inputs.expected["scenario"].astype(str).eq("multi_sample")
    ].copy()
    for column, value in inputs.contract.expected_filters.items():
        if column not in expected:
            raise ValueError(f"frozen expected table lacks filter column {column!r}")
        expected = expected.loc[expected[column].astype(str).eq(value)].copy()
    if expected.empty:
        raise ValueError("frozen primary expected table is empty")
    return expected


def _assert_expected_parity(rebuilt: pd.DataFrame, frozen: pd.DataFrame) -> None:
    keys = ["condition", "top_fraction", "sender", "receiver"]
    values = ["is_expected", "spatial_rank", "rankable_pairs", "top_count"]
    missing = set((*keys, *values)).difference(frozen.columns)
    if missing:
        raise ValueError(
            f"frozen expected table lacks parity columns: {sorted(missing)}"
        )
    left = rebuilt.loc[:, [*keys, *values]].sort_values(keys, kind="stable")
    right = frozen.loc[:, [*keys, *values]].sort_values(keys, kind="stable")
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if len(left) != len(right) or not left.loc[:, keys].equals(right.loc[:, keys]):
        raise ValueError("rebuilt and frozen expected-set axes disagree")
    if not left["is_expected"].astype(bool).equals(right["is_expected"].astype(bool)):
        raise ValueError("rebuilt expected membership differs from frozen truth")
    for column in ("spatial_rank", "rankable_pairs", "top_count"):
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if not np.allclose(left_values, right_values, equal_nan=True, rtol=0, atol=0):
            raise ValueError(f"rebuilt expected {column} differs from frozen truth")


def _evaluate_candidate(
    candidate: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    contract: DatasetContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_dataset = str(candidate["dataset"].iloc[0])
    dataset_map = (
        {candidate_dataset: contract.truth_dataset_id}
        if candidate_dataset != contract.truth_dataset_id
        else None
    )
    return evaluate_rankings(
        candidate,
        expected,
        scenario="multi_sample",
        dataset_map=dataset_map,
        condition_map=dict(contract.candidate_condition_map),
        score_type="pos",
        weight_exponent=1.0,
        tie_policy="fgsea_native",
        exclude_self_pairs=True,
        ranking_statistic="raw_cardinality",
    )


def _candidate_rows(table: pd.DataFrame) -> pd.DataFrame:
    result = table.loc[table["method"].astype(str).eq(METHOD_ID)].copy()
    if result.empty:
        raise ValueError("stored DES table lacks the bounded candidate")
    return result.sort_values(
        ["condition", "top_fraction"], kind="stable", ignore_index=True
    )


def _assert_des_parity(recomputed: pd.DataFrame, stored: pd.DataFrame) -> None:
    stored = _candidate_rows(stored)
    recomputed = _candidate_rows(recomputed)
    keys = ["condition", "top_fraction"]
    if not recomputed.loc[:, keys].equals(stored.loc[:, keys]):
        raise ValueError("recomputed and stored DES axes disagree")
    for column in ("status",):
        if not recomputed[column].astype(str).equals(stored[column].astype(str)):
            raise ValueError(f"recomputed and stored DES {column} disagree")
    left_reason = recomputed["reason_code"].fillna("").astype(str)
    right_reason = stored["reason_code"].fillna("").astype(str)
    if not left_reason.equals(right_reason):
        raise ValueError("recomputed and stored DES reason_code disagree")
    for column in ("des", "signed_peak_es", "peak_rank"):
        left = pd.to_numeric(recomputed[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(stored[column], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, equal_nan=True, rtol=0, atol=1e-12):
            raise ValueError(f"recomputed and stored DES {column} disagree")


def _assert_coverage_parity(recomputed: pd.DataFrame, stored: pd.DataFrame) -> None:
    stored = _candidate_rows(stored)
    recomputed = _candidate_rows(recomputed)
    keys = ["condition", "top_fraction"]
    if not recomputed.loc[:, keys].equals(stored.loc[:, keys]):
        raise ValueError("recomputed and stored DES coverage axes disagree")
    for column in ("status", "expected_input_mode", "rank_status_counts_json"):
        if not recomputed[column].astype(str).equals(stored[column].astype(str)):
            raise ValueError(f"recomputed and stored DES coverage {column} disagree")
    left_reason = recomputed["reason_code"].fillna("").astype(str)
    right_reason = stored["reason_code"].fillna("").astype(str)
    if not left_reason.equals(right_reason):
        raise ValueError("recomputed and stored DES coverage reason_code disagree")
    numeric_columns = (
        "expected_set_available",
        "expected_input_rows",
        "expected_pairs",
        "expected_pairs_covered",
        "expected_pairs_missing",
        "expected_pair_coverage_fraction",
        "rank_rows_total",
        "rank_pairs_eligible",
        "rank_pairs_noneligible",
        "rank_eligible_fraction",
        "ranked_background_pairs",
        "tied_strength_blocks",
        "ranked_pairs_in_ties",
    )
    for column in numeric_columns:
        left = pd.to_numeric(recomputed[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(stored[column], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, equal_nan=True, rtol=0, atol=1e-12):
            raise ValueError(
                f"recomputed and stored DES coverage {column} disagree"
            )


def _role_columns(contract: DatasetContract) -> dict[str, object]:
    return {
        "dataset_key": contract.dataset,
        "dataset_role": contract.role,
        "independent_validation": contract.independent_validation,
        "cross_dataset_pooling_allowed": False,
        "formal_inference_allowed": False,
    }


def _endpoint_availability(contract: DatasetContract) -> pd.DataFrame:
    common = {
        **_role_columns(contract),
        "available_input_grain": "condition_x_unordered_cell_pair",
    }
    records: list[dict[str, object]] = [
        {
            **common,
            "des_variant": "legacy_bounded_pair_rank_des",
            "event_budget": pd.NA,
            "mechanism": None,
            "status": "observed",
            "reason_code": None,
            "required_input_grain": "condition_x_unordered_cell_pair",
            "claim_scope": (
                "existing_protocol_spatial_sensitivity;not_literal_original_count_DES"
            ),
        },
        {
            **common,
            "des_variant": "original_count_des",
            "event_budget": pd.NA,
            "mechanism": None,
            "status": "not_estimable",
            "reason_code": "condition_specific_selected_event_count_not_exported",
            "required_input_grain": "condition_x_directed_lr_event_selection",
            "claim_scope": "suggestion_v6_strict_endpoint",
        },
    ]
    records.extend(
        {
            **common,
            "des_variant": "top_k_count_matched_des",
            "event_budget": budget,
            "mechanism": None,
            "status": "not_estimable",
            "reason_code": "candidate_event_ranking_ledger_not_exported",
            "required_input_grain": "condition_x_directed_lr_event_rank",
            "claim_scope": "suggestion_v6_strict_endpoint",
        }
        for budget in EVENT_BUDGETS
    )
    records.extend(
        [
            {
                **common,
                "des_variant": "continuous_weighted_des",
                "event_budget": pd.NA,
                "mechanism": None,
                "status": "not_estimable",
                "reason_code": (
                    "pair_score_is_not_sum_of_event_effects_or_signed_p_evidence"
                ),
                "required_input_grain": "condition_x_directed_lr_event_effect",
                "claim_scope": "suggestion_v6_strict_endpoint",
            },
            {
                **common,
                "des_variant": "direction_preserving_des",
                "event_budget": pd.NA,
                "mechanism": None,
                "status": "not_estimable",
                "reason_code": "candidate_axis_collapses_sender_receiver_directions",
                "required_input_grain": "condition_x_ordered_sender_receiver_x_event",
                "claim_scope": "suggestion_v6_strict_endpoint",
            },
        ]
    )
    records.extend(
        {
            **common,
            "des_variant": "mechanism_stratified_des",
            "event_budget": pd.NA,
            "mechanism": mechanism,
            "status": "not_estimable",
            "reason_code": "candidate_event_mechanism_annotation_not_exported",
            "required_input_grain": "condition_x_directed_lr_event_x_mechanism",
            "claim_scope": "suggestion_v6_strict_endpoint",
        }
        for mechanism in MECHANISMS
    )
    result = pd.DataFrame.from_records(records)
    result["event_budget"] = result["event_budget"].astype("Int64")
    return result


def _supporting_analysis_availability(contract: DatasetContract) -> pd.DataFrame:
    common = {
        **_role_columns(contract),
        "evaluation_scope": "persisted_tables_only_no_core_refit",
    }
    return pd.DataFrame.from_records(
        [
            {
                **common,
                "analysis": "slice_bootstrap_95pct_ci",
                "status": "observed",
                "reason_code": None,
                "analysis_scope": "spatial_endpoint_only_candidate_ranking_fixed",
            },
            {
                **common,
                "analysis": "leave_one_slice_out",
                "status": "observed",
                "reason_code": None,
                "analysis_scope": "spatial_endpoint_only_candidate_ranking_fixed",
            },
            {
                **common,
                "analysis": "condition_label_permutation_null",
                "status": "observed",
                "reason_code": None,
                "analysis_scope": (
                    "spatial_endpoint_labels_only_candidate_ranking_fixed"
                ),
            },
            {
                **common,
                "analysis": "spatial_rank_stability",
                "status": "observed",
                "reason_code": None,
                "analysis_scope": "spatial_endpoint_only_candidate_ranking_fixed",
            },
            {
                **common,
                "analysis": "predicted_event_count_curve",
                "status": "not_estimable",
                "reason_code": "candidate_event_selection_ledger_not_exported",
                "analysis_scope": "suggestion_v6_strict_endpoint",
            },
            {
                **common,
                "analysis": "cell_pair_coverage",
                "status": "observed",
                "reason_code": None,
                "analysis_scope": "condition_x_unordered_cell_pair",
            },
        ]
    )


def _resample_slices(
    sample_values: pd.DataFrame,
    *,
    contract: DatasetContract,
    rng: np.random.Generator,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for condition in (contract.reference_condition, contract.target_condition):
        condition_rows = sample_values.loc[sample_values["condition"].eq(condition)]
        sample_ids = np.array(
            sorted(condition_rows["sample_id"].astype(str).unique()), dtype=object
        )
        selected = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        for draw_index, sample_id in enumerate(selected):
            drawn = condition_rows.loc[
                condition_rows["sample_id"].astype(str).eq(str(sample_id))
            ].copy()
            drawn["sample_id"] = f"{sample_id}__bootstrap_{draw_index:04d}"
            records.append(drawn)
    return pd.concat(records, ignore_index=True, sort=False)


def _permute_slice_labels(
    sample_values: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = sample_values.copy(deep=True)
    sample_labels = (
        result.loc[:, ["sample_id", "condition"]]
        .drop_duplicates()
        .sort_values("sample_id", kind="stable")
    )
    if sample_labels["sample_id"].duplicated().any():
        raise ValueError("each spatial sample must have one condition")
    shuffled = rng.permutation(sample_labels["condition"].to_numpy(dtype=object))
    mapping = dict(zip(sample_labels["sample_id"].astype(str), shuffled, strict=True))
    result["condition"] = result["sample_id"].astype(str).map(mapping)
    return result


def _rank_stability(
    reference: pd.DataFrame,
    resampled: pd.DataFrame,
) -> tuple[float, int, str, str | None]:
    left = reference.loc[
        reference["ranking_eligible"], ["sender", "receiver", "spatial_rank"]
    ]
    right = resampled.loc[
        resampled["ranking_eligible"], ["sender", "receiver", "spatial_rank"]
    ]
    common = left.merge(
        right,
        on=["sender", "receiver"],
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_resampled"),
    )
    if len(common) < 2:
        return math.nan, len(common), "not_estimable", "fewer_than_two_common_ranks"
    rho = common["spatial_rank_reference"].corr(
        common["spatial_rank_resampled"], method="spearman"
    )
    if pd.isna(rho):
        return math.nan, len(common), "not_estimable", "constant_spatial_ranks"
    return float(rho), len(common), "observed", None


def _annotate_resample_scores(
    scores: pd.DataFrame,
    *,
    contract: DatasetContract,
    kind: ResampleKind,
    replicate_id: int,
    omitted_sample_id: str | None,
) -> pd.DataFrame:
    result = scores.copy(deep=True)
    leading = {
        **_role_columns(contract),
        "resampling_scope": "spatial_endpoint_only_candidate_ranking_fixed",
        "resample_kind": kind,
        "replicate_id": replicate_id,
        "omitted_sample_id": omitted_sample_id,
    }
    for position, (column, value) in enumerate(leading.items()):
        result.insert(position, column, value)
    return result


def _resample_endpoint(
    inputs: BoundInputs,
    *,
    full_rankings: pd.DataFrame,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_values = _normalize_sample_values(inputs)
    contract = inputs.contract
    score_tables: list[pd.DataFrame] = []
    stability_records: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    def evaluate_one(
        values: pd.DataFrame,
        *,
        kind: ResampleKind,
        replicate_id: int,
        omitted_sample_id: str | None = None,
    ) -> None:
        rankings, expected = _build_spatial_expected(values, contract=contract)
        scores, _ = _evaluate_candidate(
            inputs.candidate, expected, contract=contract
        )
        score_tables.append(
            _annotate_resample_scores(
                scores,
                contract=contract,
                kind=kind,
                replicate_id=replicate_id,
                omitted_sample_id=omitted_sample_id,
            )
        )
        rho, common, status, reason = _rank_stability(full_rankings, rankings)
        stability_records.append(
            {
                **_role_columns(contract),
                "resampling_scope": "spatial_endpoint_only_candidate_ranking_fixed",
                "resample_kind": kind,
                "replicate_id": replicate_id,
                "omitted_sample_id": omitted_sample_id,
                "common_rankable_pairs": common,
                "spatial_rank_spearman": rho,
                "status": status,
                "reason_code": reason,
            }
        )

    for replicate in range(bootstrap_replicates):
        evaluate_one(
            _resample_slices(sample_values, contract=contract, rng=rng),
            kind="slice_bootstrap",
            replicate_id=replicate,
        )
    for replicate, sample_id in enumerate(
        sorted(sample_values["sample_id"].astype(str).unique())
    ):
        evaluate_one(
            sample_values.loc[
                sample_values["sample_id"].astype(str).ne(sample_id)
            ].copy(),
            kind="leave_one_slice_out",
            replicate_id=replicate,
            omitted_sample_id=sample_id,
        )
    for replicate in range(permutation_replicates):
        evaluate_one(
            _permute_slice_labels(sample_values, rng=rng),
            kind="endpoint_condition_permutation",
            replicate_id=replicate,
        )
    scores = pd.concat(score_tables, ignore_index=True, sort=False)
    stability = pd.DataFrame.from_records(stability_records)
    return scores, stability


def _summary_value(values: pd.Series, operation: str) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not finite.size:
        return math.nan
    if operation == "mean":
        return float(np.mean(finite))
    if operation == "median":
        return float(np.median(finite))
    if operation == "q025":
        return float(np.quantile(finite, 0.025))
    if operation == "q975":
        return float(np.quantile(finite, 0.975))
    raise ValueError(f"unsupported summary operation: {operation}")


def _summarize_resamples(
    scores: pd.DataFrame,
    full_scores: pd.DataFrame,
    *,
    contract: DatasetContract,
) -> pd.DataFrame:
    full = _candidate_rows(full_scores).loc[
        :, ["condition", "top_fraction", "des", "status"]
    ].rename(columns={"des": "full_des", "status": "full_status"})
    records: list[dict[str, object]] = []
    for (kind, condition, fraction), group in scores.groupby(
        ["resample_kind", "condition", "top_fraction"],
        observed=True,
        sort=True,
    ):
        observed = group["status"].eq("observed") & group["des"].notna()
        record: dict[str, object] = {
            **_role_columns(contract),
            "resampling_scope": "spatial_endpoint_only_candidate_ranking_fixed",
            "resample_kind": kind,
            "condition": condition,
            "top_fraction": float(fraction),
            "resamples_requested": len(group),
            "resamples_observed": int(observed.sum()),
            "resamples_not_estimable": int((~observed).sum()),
            "resample_observed_fraction": float(observed.mean()),
            "resampled_des_mean": _summary_value(group.loc[observed, "des"], "mean"),
            "resampled_des_median": _summary_value(
                group.loc[observed, "des"], "median"
            ),
            "resampled_des_q025": _summary_value(
                group.loc[observed, "des"], "q025"
            ),
            "resampled_des_q975": _summary_value(
                group.loc[observed, "des"], "q975"
            ),
            "status": "observed" if observed.any() else "not_estimable",
            "reason_code": (
                None if observed.any() else "no_estimable_endpoint_resamples"
            ),
        }
        records.append(record)
    summary = pd.DataFrame.from_records(records).merge(
        full,
        on=["condition", "top_fraction"],
        how="left",
        validate="many_to_one",
    )
    bootstrap = summary["resample_kind"].eq("slice_bootstrap")
    summary["bootstrap_ci_low_95"] = summary["resampled_des_q025"].where(bootstrap)
    summary["bootstrap_ci_high_95"] = summary["resampled_des_q975"].where(
        bootstrap
    )
    summary["interval_semantics"] = summary["resample_kind"].map(
        {
            "slice_bootstrap": "percentile_bootstrap_95pct_interval",
            "leave_one_slice_out": "empirical_omission_distribution_quantiles",
            "endpoint_condition_permutation": "empirical_null_distribution_quantiles",
        }
    )
    permutation = summary["resample_kind"].eq("endpoint_condition_permutation")
    summary["diagnostic_empirical_upper_tail_p"] = np.nan
    for index in summary.index[permutation]:
        row = summary.loc[index]
        null = scores.loc[
            scores["resample_kind"].eq("endpoint_condition_permutation")
            & scores["condition"].eq(row["condition"])
            & scores["top_fraction"].eq(row["top_fraction"])
            & scores["status"].eq("observed"),
            "des",
        ].dropna()
        full_des = pd.to_numeric(pd.Series([row["full_des"]]), errors="coerce").iloc[0]
        if len(null) and pd.notna(full_des):
            summary.loc[index, "diagnostic_empirical_upper_tail_p"] = (
                1.0 + float(pd.to_numeric(null, errors="raise").ge(full_des).sum())
            ) / (1.0 + len(null))
    return summary.sort_values(
        ["resample_kind", "condition", "top_fraction"],
        kind="stable",
        ignore_index=True,
    )


def _pair_coverage(candidate: pd.DataFrame, contract: DatasetContract) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for condition, group in candidate.groupby("condition", observed=True, sort=True):
        score = pd.to_numeric(group["ranked_strength"], errors="coerce")
        observed = group["status"].eq("observed")
        opportunity = pd.to_numeric(group["estimable_directed_lr"], errors="coerce")
        records.append(
            {
                **_role_columns(contract),
                "condition": condition,
                "pair_rows": len(group),
                "observed_pairs": int(observed.sum()),
                "not_estimable_pairs": int((~observed).sum()),
                "observed_pair_fraction": float(observed.mean()),
                "observed_zero_strength_pairs": int(
                    (observed & score.eq(0.0)).sum()
                ),
                "observed_unique_strengths": int(score.loc[observed].nunique()),
                "finite_opportunity_pairs": int(opportunity.notna().sum()),
                "estimable_directed_lr_sum": (
                    float(opportunity.sum(min_count=1))
                    if opportunity.notna().any()
                    else math.nan
                ),
                "missing_policy": "typed_missing_never_zero_imputed",
            }
        )
    return pd.DataFrame.from_records(records)


def run(
    *,
    bounded_run: Path,
    truth_manifest: Path,
    output_dir: Path,
    bootstrap_replicates: int = 1000,
    permutation_replicates: int = 1000,
    seed: int = 20260724,
) -> dict[str, Any]:
    """Evaluate one checksum-bound Kuppe or MS RC11 result."""

    for name, value in (
        ("bootstrap_replicates", bootstrap_replicates),
        ("permutation_replicates", permutation_replicates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    started = time.perf_counter()
    inputs = _load_inputs(bounded_run, truth_manifest)
    sample_values = _normalize_sample_values(inputs)
    full_spatial_rankings, rebuilt_expected = _build_spatial_expected(
        sample_values, contract=inputs.contract
    )
    _assert_expected_parity(rebuilt_expected, _frozen_primary_expected(inputs))
    full_scores, full_coverage = _evaluate_candidate(
        inputs.candidate, rebuilt_expected, contract=inputs.contract
    )
    _assert_des_parity(full_scores, inputs.stored_scores)
    _assert_coverage_parity(full_coverage, inputs.stored_coverage)

    role = _role_columns(inputs.contract)
    full_scores = _candidate_rows(full_scores)
    full_coverage = _candidate_rows(full_coverage)
    for table in (full_scores, full_coverage):
        for position, (column, value) in enumerate(role.items()):
            table.insert(position, column, value)
    resample_scores, rank_stability = _resample_endpoint(
        inputs,
        full_rankings=full_spatial_rankings,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        seed=seed,
    )
    _validate_typed_values(
        full_scores, value_column="des", label="legacy bounded pair-rank DES"
    )
    _validate_typed_values(
        resample_scores, value_column="des", label="endpoint-resampled DES"
    )
    _validate_typed_values(
        rank_stability,
        value_column="spatial_rank_spearman",
        label="spatial rank stability",
    )
    resample_summary = _summarize_resamples(
        resample_scores, full_scores, contract=inputs.contract
    )
    tables = {
        "des_endpoint_availability.tsv": _endpoint_availability(inputs.contract),
        "supporting_analysis_availability.tsv": _supporting_analysis_availability(
            inputs.contract
        ),
        "legacy_bounded_pair_rank_des.tsv": full_scores,
        "legacy_bounded_pair_rank_coverage.tsv": full_coverage,
        "endpoint_resample_scores.tsv": resample_scores,
        "endpoint_resample_summary.tsv": resample_summary,
        "spatial_rank_stability.tsv": rank_stability,
        "pair_coverage.tsv": _pair_coverage(inputs.candidate, inputs.contract),
    }

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    published = False
    try:
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "dataset": inputs.contract.dataset,
            **role,
            "method": METHOD_ID,
            "evaluation_scope": "persisted_tables_only_no_core_refit",
            "resampling_scope": "spatial_endpoint_only_candidate_ranking_fixed",
            "formal_inference_allowed": False,
            "v6_strict_des_variants_estimable": [],
            "legacy_sensitivity_estimable": ["legacy_bounded_pair_rank_des"],
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "sampling_unit": "sample_id_spatial_slice",
                "stratified_by_condition": True,
                "interval": "empirical_quantile_2.5_97.5",
            },
            "leave_one_out": {"unit": "sample_id_spatial_slice"},
            "permutation": {
                "replicates": permutation_replicates,
                "unit": "sample_id_spatial_slice",
                "preserves_condition_sizes": True,
                "claim_limit": "endpoint_label_permutation_not_algorithm_refit",
            },
            "seed": seed,
            "missing_policy": "typed_missing_never_zero_imputed",
            "development_validation_boundary": {
                "kuppe": "development_non_independent",
                "ms": "independent_validation",
                "cross_dataset_pooling_allowed": False,
            },
            "inputs": inputs.input_records,
            "source_code": git_metadata(Path(__file__).resolve().parents[2]),
            "performance": {"elapsed_seconds": time.perf_counter() - started},
            "outputs": {
                filename: _output_record(staged / filename, table)
                for filename, table in tables.items()
            },
            "limitations": [
                "rc11_exports_only_aggregated_unordered_cell_pair_rankings",
                "strict_v6_event_level_des_variants_are_not_estimable",
                "endpoint_resampling_does_not_refit_or_permute_crychic",
                "spatial_colocalization_is_an_indirect_proxy",
                "kuppe_was_used_for_candidate_development",
                "ms_is_the_only_independent_validation_cohort",
            ],
        }
        _write_json(staged / "manifest.json", manifest)
        os.replace(staged, output_dir)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded-run", required=True, type=Path)
    parser.add_argument("--truth-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--permutation-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(
        bounded_run=args.bounded_run,
        truth_manifest=args.truth_manifest,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
        seed=args.seed,
    )
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
