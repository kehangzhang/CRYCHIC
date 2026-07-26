"""Align frozen v7 real-data effects with sparse Visium geometry diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    json_safe,
    prepare_output,
    python_environment,
    sha256_file,
    write_json,
)
from benchmarks.literature.event_level_des import (
    mechanism_annotations,
    pair_rankings_from_events,
    select_top_k_events_by_scope,
)
from benchmarks.literature.run_v7_spatial_geometry import (
    DEFAULT_CONFIG,
    load_spatial_geometry_config,
)
from benchmarks.metrics.spatial_des import SpatialDESSpec, evaluate_spatial_des

SCHEMA_VERSION = "crychic-suggest-next2-v7-spatial-geometry-evaluation-v1"
REAL_RUN_SCHEMA = "crychic-suggest-next2-v7-real-e1-run-v1"
GEOMETRY_RUN_SCHEMA = "crychic-suggest-next2-v7-spatial-geometry-run-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("kuppe", "ms")
TRUTH_SUPPORT_COLUMNS = {
    "rank_top": (),
    "coordinate_max_t_supported_rank_top": ("coordinate_max_t_p_value",),
    "cell_label_max_t_supported_rank_top": ("cell_label_max_t_p_value",),
    "coordinate_and_cell_label_max_t_supported_rank_top": (
        "coordinate_max_t_p_value",
        "cell_label_max_t_p_value",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


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
        "sha256": sha256_file(path),
    }


def _append_log(path: Path, event: str, **fields: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                json_safe({"event": event, **fields}),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def _manifest_output(
    run_dir: Path,
    manifest: Mapping[str, object],
    filename: str,
) -> Path:
    outputs = manifest.get("outputs")
    record = outputs.get(filename) if isinstance(outputs, Mapping) else None
    if not isinstance(record, Mapping) or record.get("filename") != filename:
        raise ValueError(f"run manifest lacks checksum-bound output {filename!r}")
    path = run_dir / filename
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"run output checksum mismatch: {filename}")
    return path


def _validate_real_run(
    run_dir: Path,
    *,
    dataset: str,
    contract: Mapping[str, object],
) -> tuple[dict[str, Any], Path]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != REAL_RUN_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != dataset
        or manifest.get("dataset_id") != contract["algorithm_dataset_id"]
        or manifest.get("formal_inference_allowed") is not False
    ):
        raise ValueError("v7 real run does not satisfy the geometry contract")
    ledger_path = _manifest_output(
        run_dir, manifest, "sender_resolved_event_ledger.parquet"
    )
    return manifest, ledger_path


def _validate_geometry_run(
    run_dir: Path,
    *,
    dataset: str,
    contract: Mapping[str, object],
    config_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    inputs = manifest.get("inputs")
    protocol = inputs.get("protocol") if isinstance(inputs, Mapping) else None
    if (
        manifest.get("schema_version") != GEOMETRY_RUN_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != dataset
        or manifest.get("dataset_id") != contract["geometry_dataset_id"]
        or manifest.get("profile") != "formal"
        or manifest.get("formal_inference_allowed") is not False
        or not isinstance(protocol, Mapping)
        or protocol.get("sha256") != sha256_file(config_path)
    ):
        raise ValueError("formal spatial geometry run does not satisfy its contract")
    effects_path = _manifest_output(
        run_dir, manifest, "condition_geometry_effects.tsv"
    )
    expected_path = _manifest_output(run_dir, manifest, "geometry_expected_sets.tsv")
    return manifest, effects_path, expected_path


def _validate_resource(
    resource_path: Path,
    manifest_path: Path,
    contract: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if (
        sha256_file(resource_path) != contract["payload_sha256"]
        or sha256_file(manifest_path) != contract["manifest_sha256"]
    ):
        raise ValueError("spatial mechanism resource checksum mismatch")
    manifest = _read_json(manifest_path)
    payload = manifest.get("payload")
    if (
        manifest.get("schema_version") != "crychic-connectomedb2020-resource-v1"
        or manifest.get("resource_id") != contract["resource_id"]
        or not isinstance(payload, Mapping)
        or payload.get("filename") != resource_path.name
        or payload.get("sha256") != contract["payload_sha256"]
    ):
        raise ValueError("spatial mechanism resource manifest changed")
    resource = pd.read_csv(resource_path, sep="\t")
    if len(resource) != manifest.get("rows"):
        raise ValueError("spatial mechanism resource row count changed")
    return resource, manifest


def annotate_event_mechanisms(
    ledger: pd.DataFrame,
    resource: pd.DataFrame,
    *,
    generators: Sequence[str],
) -> pd.DataFrame:
    """Attach frozen ligand-location mechanisms to a sender-resolved ledger."""

    required = {
        "generator_id",
        "condition",
        "sender",
        "receiver",
        "pair_sender",
        "pair_receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "effect_target_minus_reference",
        "abs_effect",
        "event_evidence",
        "native_selected",
        "status",
    }
    missing = required.difference(ledger.columns)
    if missing or ledger.empty:
        raise ValueError(f"v7 event ledger is empty or missing: {sorted(missing)}")
    expected_generators = set(map(str, generators))
    observed_generators = set(ledger["generator_id"].astype(str))
    if observed_generators != expected_generators:
        raise ValueError(
            "v7 event-ledger generator axis changed: "
            f"{sorted(observed_generators)} != {sorted(expected_generators)}"
        )
    lookup = mechanism_annotations(resource)
    result = ledger.merge(
        lookup,
        on=["ligand", "receptor"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if result[["ligand_location", "mechanism"]].isna().any(axis=None):
        raise ValueError("one or more v7 events lack a frozen mechanism annotation")
    sender = result["sender"].astype(str).to_numpy()
    receiver = result["receiver"].astype(str).to_numpy()
    if not (
        result["pair_sender"].astype(str).eq(np.minimum(sender, receiver)).all()
        and result["pair_receiver"].astype(str).eq(np.maximum(sender, receiver)).all()
    ):
        raise ValueError("v7 event ledger contains non-canonical unordered pairs")
    observed = result["status"].astype(str).eq("observed")
    for column in ("abs_effect", "event_evidence"):
        values = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(values.loc[observed]).all() or (
            values.loc[observed] < 0.0
        ).any():
            raise ValueError(f"observed v7 events require finite {column}")
        result[column] = values
    result["native_selected"] = result["native_selected"].fillna(False).astype(bool)
    return result.sort_values(
        ["generator_id", "condition", "sender", "receiver", "interaction_id"],
        kind="stable",
        ignore_index=True,
    )


def geometry_pair_axes(
    effects: pd.DataFrame,
    *,
    dataset_id: str,
    analysis_unit: str,
    bands: Sequence[str],
    algorithm_conditions: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the geometry universe and return algorithm-labelled pair axes."""

    required = {
        "dataset",
        "analysis_unit",
        "band",
        "sender",
        "receiver",
        "status",
        "absolute_effect",
        "effect_target_minus_reference",
        "reference",
        "target",
        "coordinate_max_t_p_value",
        "cell_label_max_t_p_value",
    }
    missing = required.difference(effects.columns)
    if missing or effects.empty:
        raise ValueError(f"geometry effects are empty or missing: {sorted(missing)}")
    selected = effects.loc[
        effects["dataset"].astype(str).eq(dataset_id)
        & effects["analysis_unit"].astype(str).eq(analysis_unit)
    ].copy()
    if selected.empty or set(selected["band"].astype(str)) != set(map(str, bands)):
        raise ValueError("geometry effect dataset, unit, or distance-band axis changed")
    sender = selected["sender"].astype(str)
    receiver = selected["receiver"].astype(str)
    if not sender.lt(receiver).all():
        raise ValueError("geometry effects must use canonical non-self cell pairs")
    pair_sets = {
        band: set(
            selected.loc[selected["band"].astype(str).eq(band), ["sender", "receiver"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        for band in map(str, bands)
    }
    first_pairs = next(iter(pair_sets.values()))
    if not first_pairs or any(pairs != first_pairs for pairs in pair_sets.values()):
        raise ValueError("geometry distance bands do not share one fixed pair universe")
    cell_types = sorted({value for pair in first_pairs for value in pair})
    if len(first_pairs) != len(cell_types) * (len(cell_types) - 1) // 2:
        raise ValueError(
            "geometry effects do not contain the complete non-self universe"
        )
    conditions = tuple(map(str, algorithm_conditions))
    if len(conditions) != 2 or len(set(conditions)) != 2:
        raise ValueError("algorithm condition axis must contain exactly two labels")
    axes = pd.DataFrame.from_records(
        {
            "condition": condition,
            "sender": left,
            "receiver": right,
            "ranked_strength": 0.0,
            "status": "observed",
        }
        for condition in conditions
        for left, right in sorted(first_pairs)
    )
    return axes, selected.sort_values(
        ["band", "sender", "receiver"], kind="stable", ignore_index=True
    )


def build_mechanism_pair_rankings(
    ledger: pd.DataFrame,
    pair_axes: pd.DataFrame,
    *,
    dataset_id: str,
    resource_id: str,
    alignment: Mapping[str, object],
) -> pd.DataFrame:
    """Build the preregistered v7 endpoint-by-mechanism pair rankings."""

    generators = tuple(map(str, cast(Sequence[object], alignment["generators"])))
    mechanisms = tuple(map(str, cast(Sequence[object], alignment["mechanisms"])))
    budgets = tuple(map(int, cast(Sequence[object], alignment["event_budgets"])))
    rankings: list[pd.DataFrame] = []
    for generator in generators:
        local = ledger.loc[ledger["generator_id"].astype(str).eq(generator)].copy()
        observed = local["status"].astype(str).eq("observed")
        nonself = local["pair_sender"].astype(str).ne(
            local["pair_receiver"].astype(str)
        )
        for mechanism in mechanisms:
            mechanism_mask = (
                pd.Series(True, index=local.index)
                if mechanism == "all"
                else local["mechanism"].astype(str).eq(mechanism)
            )
            nonzero = local["effect_target_minus_reference"].ne(0.0)
            eligible = observed & nonself & mechanism_mask & nonzero
            endpoint_specs: list[tuple[str, int | None, pd.Series, str | None]] = [
                (
                    "continuous_weighted_des",
                    None,
                    eligible,
                    str(alignment["continuous_weight"]),
                ),
                (
                    "diagnostic_one_se_native_count_des",
                    None,
                    eligible & local["native_selected"],
                    None,
                ),
            ]
            for budget in budgets:
                selected = select_top_k_events_by_scope(
                    local,
                    budget=budget,
                    evidence_column=str(alignment["top_k_evidence"]),
                    eligible=eligible,
                    scope=str(alignment["top_k_scope"]),
                )
                endpoint_specs.append(("top_k_count_des", budget, selected, None))
            for endpoint, budget, selected, weight_column in endpoint_specs:
                semantics = (
                    "sum_absolute_subject_level_effect"
                    if weight_column is not None
                    else "count_selected_directed_lr_events"
                )
                ranking = pair_rankings_from_events(
                    local,
                    pair_axes,
                    selected=selected,
                    weight_column=weight_column,
                    metadata={
                        "dataset": dataset_id,
                        "method": f"CRYCHIC_v7_{generator}_I1_{mechanism}",
                        "method_version": "suggest-next2-v7-primary",
                        "resource": resource_id,
                        "ranking_semantics": semantics,
                    },
                )
                ranking.insert(0, "generator_id", generator)
                ranking.insert(1, "mechanism", mechanism)
                ranking.insert(2, "des_variant", endpoint)
                ranking.insert(3, "event_budget", budget)
                rankings.append(ranking)
    return pd.concat(rankings, ignore_index=True, sort=False)


def geometry_truth_scenarios(
    expected_sets: pd.DataFrame,
    *,
    dataset_id: str,
    analysis_unit: str,
    bands: Sequence[str],
    truth_variants: Sequence[str],
    geometry_conditions: Sequence[str],
    diagnostic_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand signed geometry effects to a full condition-specific truth universe."""

    required = {
        "dataset",
        "analysis_unit",
        "band",
        "sender",
        "receiver",
        "top_fraction",
        "is_expected",
        "expected_condition",
        "status",
        "coordinate_max_t_p_value",
        "cell_label_max_t_p_value",
    }
    missing = required.difference(expected_sets.columns)
    if missing or expected_sets.empty:
        raise ValueError(
            f"geometry expected sets are empty or missing: {sorted(missing)}"
        )
    variants = tuple(map(str, truth_variants))
    if variants != tuple(TRUTH_SUPPORT_COLUMNS):
        raise ValueError("geometry truth-variant axis changed")
    if not math.isfinite(diagnostic_alpha) or diagnostic_alpha != 0.05:
        raise ValueError("geometry diagnostic alpha changed")
    selected = expected_sets.loc[
        expected_sets["dataset"].astype(str).eq(dataset_id)
        & expected_sets["analysis_unit"].astype(str).eq(analysis_unit)
        & expected_sets["band"].astype(str).isin(tuple(map(str, bands)))
    ].copy()
    if selected.empty:
        raise ValueError("geometry expected sets lack the preregistered primary unit")
    conditions = tuple(map(str, geometry_conditions))
    if len(conditions) != 2 or len(set(conditions)) != 2:
        raise ValueError("geometry condition axis must contain exactly two labels")
    if not set(selected["expected_condition"].astype(str)).issubset(
        {*conditions, "tied"}
    ):
        raise ValueError("geometry expected-condition labels changed")
    records: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for band in map(str, bands):
        band_rows = selected.loc[selected["band"].astype(str).eq(band)].copy()
        if band_rows.empty:
            raise ValueError(f"geometry expected sets lack band {band!r}")
        for variant in variants:
            support = band_rows["status"].astype(str).eq("observed")
            for column in TRUTH_SUPPORT_COLUMNS[variant]:
                values = pd.to_numeric(band_rows[column], errors="coerce")
                support &= values.notna() & values.le(diagnostic_alpha)
            scenario = f"geometry_{band}__{variant}"
            metadata.append(
                {"scenario": scenario, "band": band, "truth_variant": variant}
            )
            for condition in conditions:
                encoded = band_rows.copy()
                encoded["scenario"] = scenario
                encoded["condition"] = condition
                encoded["is_expected"] = (
                    encoded["is_expected"].fillna(False).astype(bool)
                    & encoded["expected_condition"].astype(str).eq(condition)
                    & support
                )
                records.append(
                    encoded.loc[
                        :,
                        [
                            "dataset",
                            "scenario",
                            "condition",
                            "top_fraction",
                            "sender",
                            "receiver",
                            "is_expected",
                        ],
                    ]
                )
    truth = pd.concat(records, ignore_index=True, sort=False)
    keys = ["dataset", "scenario", "condition", "top_fraction", "sender", "receiver"]
    if truth.duplicated(keys).any():
        raise ValueError("expanded geometry truth contains duplicate pair axes")
    return truth, pd.DataFrame.from_records(metadata)


def evaluate_geometry_des(
    rankings: pd.DataFrame,
    truth: pd.DataFrame,
    scenario_metadata: pd.DataFrame,
    *,
    algorithm_dataset_id: str,
    geometry_dataset_id: str,
    condition_map: Mapping[str, str],
    alignment: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every frozen endpoint against every geometry scenario in one pass."""

    ranked = rankings.copy()
    if set(ranked["dataset"].astype(str)) != {algorithm_dataset_id}:
        raise ValueError("algorithm ranking dataset axis changed")
    observed_conditions = set(ranked["condition"].astype(str))
    if observed_conditions != set(map(str, condition_map)):
        raise ValueError("algorithm ranking condition axis changed")
    ranked["dataset"] = geometry_dataset_id
    ranked["condition"] = ranked["condition"].replace(dict(condition_map))
    if set(ranked["condition"].astype(str)) != set(truth["condition"].astype(str)):
        raise ValueError("algorithm and geometry conditions do not align")
    scenario_axis = scenario_metadata.loc[:, ["scenario"]].drop_duplicates()
    ranked = ranked.merge(scenario_axis, how="cross")
    ranked["event_budget_label"] = ranked["event_budget"].map(
        lambda value: "not_applicable" if pd.isna(value) else f"K{int(value)}"
    )
    tables = evaluate_spatial_des(
        ranked,
        truth,
        SpatialDESSpec(
            method_columns=(
                "generator_id",
                "mechanism",
                "des_variant",
                "event_budget_label",
            ),
            stratum_columns=("dataset", "scenario"),
            expected_member_column="is_expected",
            score_type=cast(Any, alignment["score_type"]),
            weight_exponent=float(alignment["weight_exponent"]),
            tie_policy=cast(Any, alignment["tie_policy"]),
            cell_pair_mode="unordered",
        ),
    )
    hypotheses = cast(
        Mapping[str, Sequence[str]], alignment["mechanism_band_hypotheses"]
    )
    outputs: list[pd.DataFrame] = []
    for table in (tables.scores, tables.coverage):
        result = table.merge(
            scenario_metadata,
            on="scenario",
            how="left",
            validate="many_to_one",
        )
        result["event_budget"] = result["event_budget_label"].map(
            lambda value: (
                pd.NA
                if value == "not_applicable"
                else int(str(value).removeprefix("K"))
            )
        ).astype("Int64")
        result["is_preregistered_range_hypothesis"] = [
            band in hypotheses[str(mechanism)]
            for mechanism, band in result.loc[:, ["mechanism", "band"]].itertuples(
                index=False, name=None
            )
        ]
        result["formal_inference_allowed"] = False
        result["claim_scope"] = "indirect_spot_geometry_diagnostic_only"
        outputs.append(result.drop(columns="event_budget_label"))
    return outputs[0], outputs[1]


def geometry_rank_concordance(
    rankings: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    condition_map: Mapping[str, str],
    alignment: Mapping[str, object],
) -> pd.DataFrame:
    """Correlate continuous pair strength with signed geometry-effect magnitude."""

    continuous = rankings.loc[
        rankings["des_variant"].astype(str).eq("continuous_weighted_des")
    ].copy()
    continuous["condition"] = continuous["condition"].replace(dict(condition_map))
    hypotheses = cast(
        Mapping[str, Sequence[str]], alignment["mechanism_band_hypotheses"]
    )
    records: list[dict[str, object]] = []
    for (generator, mechanism, condition), ranked in continuous.groupby(
        ["generator_id", "mechanism", "condition"], observed=True, sort=True
    ):
        pair_scores = ranked.loc[:, ["sender", "receiver", "ranked_strength", "status"]]
        if pair_scores.duplicated(["sender", "receiver"]).any():
            raise ValueError("continuous geometry rankings contain duplicate pairs")
        for band, geometry in effects.groupby("band", observed=True, sort=True):
            local = geometry.loc[
                :,
                [
                    "sender",
                    "receiver",
                    "absolute_effect",
                    "effect_target_minus_reference",
                    "reference",
                    "target",
                    "status",
                ],
            ].copy()
            effect = pd.to_numeric(
                local["effect_target_minus_reference"], errors="coerce"
            )
            local["expected_condition"] = np.select(
                (effect.gt(0.0), effect.lt(0.0)),
                (local["target"].astype(str), local["reference"].astype(str)),
                default="tied",
            )
            local["geometry_directional_strength"] = np.where(
                local["status"].astype(str).eq("observed")
                & local["expected_condition"].astype(str).eq(str(condition)),
                pd.to_numeric(local["absolute_effect"], errors="coerce"),
                0.0,
            )
            merged = pair_scores.merge(
                local.loc[:, ["sender", "receiver", "geometry_directional_strength"]],
                on=["sender", "receiver"],
                how="inner",
                validate="one_to_one",
            )
            predicted = pd.to_numeric(merged["ranked_strength"], errors="coerce")
            geometry_values = pd.to_numeric(
                merged["geometry_directional_strength"], errors="coerce"
            )
            finite = np.isfinite(predicted) & np.isfinite(geometry_values)
            reason: str | None = None
            if int(finite.sum()) != len(local):
                reason = "incomplete_pair_alignment"
            elif predicted.loc[finite].nunique() < 2:
                reason = "constant_algorithm_pair_strength"
            elif geometry_values.loc[finite].nunique() < 2:
                reason = "constant_directional_geometry_strength"
            rho = (
                math.nan
                if reason is not None
                else float(
                    predicted.loc[finite].corr(
                        geometry_values.loc[finite], method="spearman"
                    )
                )
            )
            records.append(
                {
                    "generator_id": generator,
                    "mechanism": mechanism,
                    "condition": condition,
                    "band": band,
                    "cell_pairs": len(merged),
                    "algorithm_nonzero_pairs": int(predicted.gt(0.0).sum()),
                    "geometry_directional_nonzero_pairs": int(
                        geometry_values.gt(0.0).sum()
                    ),
                    "spearman": rho,
                    "status": "observed" if reason is None else "not_estimable",
                    "reason_code": reason,
                    "is_preregistered_range_hypothesis": (
                        str(band) in hypotheses[str(mechanism)]
                    ),
                    "formal_inference_allowed": False,
                    "claim_scope": "indirect_spot_geometry_diagnostic_only",
                }
            )
    return pd.DataFrame.from_records(records)


def mechanism_distance_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize descriptive DES and rank generators within identical endpoints."""

    grouping = [
        "mechanism",
        "band",
        "truth_variant",
        "des_variant",
        "event_budget",
        "is_preregistered_range_hypothesis",
        "generator_id",
    ]
    observed = scores.loc[scores["status"].astype(str).eq("observed")].copy()
    if observed.empty:
        return pd.DataFrame(
            columns=[
                *grouping,
                "observations",
                "median_des",
                "mean_des",
                "generator_rank",
            ]
        )
    summary = (
        observed.groupby(grouping, observed=True, sort=True, dropna=False)["des"]
        .agg(observations="size", median_des="median", mean_des="mean")
        .reset_index()
    )
    rank_groups = [column for column in grouping if column != "generator_id"]
    summary["generator_rank"] = (
        summary.groupby(rank_groups, observed=True, dropna=False)["median_des"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    summary["formal_inference_allowed"] = False
    summary["claim_scope"] = "indirect_spot_geometry_diagnostic_only"
    return summary.sort_values(
        [*rank_groups, "generator_rank", "generator_id"],
        kind="stable",
        ignore_index=True,
    )


def run(
    *,
    dataset: str,
    real_run_dir: Path,
    geometry_run_dir: Path,
    resource_path: Path,
    resource_manifest_path: Path,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    overwrite: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run checksum-bound v7-to-geometry evaluation for one real cohort."""

    if dataset not in DATASETS:
        raise ValueError("dataset must be 'kuppe' or 'ms'")
    protocol, _, _ = load_spatial_geometry_config(config_path)
    alignment = cast(Mapping[str, object], protocol["algorithm_alignment"])
    release_contract = cast(Mapping[str, object], protocol["release"])
    dataset_contracts = cast(Mapping[str, Mapping[str, object]], alignment["datasets"])
    contract = dataset_contracts[dataset]
    code = git_metadata(REPOSITORY_ROOT)
    if code["dirty"] and not allow_dirty:
        raise RuntimeError("spatial geometry evaluation refuses a dirty worktree")
    real_manifest, ledger_path = _validate_real_run(
        real_run_dir, dataset=dataset, contract=contract
    )
    geometry_manifest, effects_path, expected_path = _validate_geometry_run(
        geometry_run_dir,
        dataset=dataset,
        contract=contract,
        config_path=config_path,
    )
    resource_contract = cast(Mapping[str, object], alignment["resource"])
    resource, resource_manifest = _validate_resource(
        resource_path, resource_manifest_path, resource_contract
    )
    output = prepare_output(output_dir, overwrite=overwrite)
    log_path = output / "run.jsonl"
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "dataset": dataset,
        "algorithm_dataset_id": contract["algorithm_dataset_id"],
        "geometry_dataset_id": contract["geometry_dataset_id"],
        "formal_inference_allowed": False,
        "inputs": {
            "protocol": _input_record(config_path),
            "real_run_manifest": _input_record(real_run_dir / "manifest.json"),
            "sender_resolved_event_ledger": _input_record(ledger_path),
            "geometry_run_manifest": _input_record(geometry_run_dir / "manifest.json"),
            "condition_geometry_effects": _input_record(effects_path),
            "geometry_expected_sets": _input_record(expected_path),
            "resource": _input_record(resource_path),
            "resource_manifest": _input_record(resource_manifest_path),
        },
        "parameters": dict(alignment),
        "code": code,
        "outputs": None,
        "failure": None,
    }
    write_json(output / "manifest.json", manifest)
    _append_log(log_path, "run_started", elapsed_seconds=0.0, dataset=dataset)
    try:
        ledger = annotate_event_mechanisms(
            pd.read_parquet(ledger_path),
            resource,
            generators=cast(Sequence[str], alignment["generators"]),
        )
        effects = pd.read_csv(effects_path, sep="\t")
        expected = pd.read_csv(expected_path, sep="\t")
        bands = tuple(map(str, cast(Sequence[object], alignment["evaluated_bands"])))
        condition_map = cast(
            Mapping[str, str], contract["condition_map_algorithm_to_geometry"]
        )
        pair_axes, primary_effects = geometry_pair_axes(
            effects,
            dataset_id=str(contract["geometry_dataset_id"]),
            analysis_unit=str(alignment["analysis_unit"]),
            bands=bands,
            algorithm_conditions=tuple(condition_map),
        )
        event_cell_types = set(ledger["sender"].astype(str)) | set(
            ledger["receiver"].astype(str)
        )
        geometry_cell_types = set(pair_axes["sender"].astype(str)) | set(
            pair_axes["receiver"].astype(str)
        )
        if event_cell_types != geometry_cell_types:
            raise ValueError(
                "v7 event and geometry cell-type axes disagree: "
                f"{sorted(event_cell_types)} != {sorted(geometry_cell_types)}"
            )
        _append_log(
            log_path,
            "inputs_aligned",
            elapsed_seconds=time.perf_counter() - started,
            event_rows=len(ledger),
            cell_types=len(geometry_cell_types),
            pair_rows=len(pair_axes),
        )
        rankings = build_mechanism_pair_rankings(
            ledger,
            pair_axes,
            dataset_id=str(contract["algorithm_dataset_id"]),
            resource_id=str(resource_contract["resource_id"]),
            alignment=alignment,
        )
        truth, scenario_metadata = geometry_truth_scenarios(
            expected,
            dataset_id=str(contract["geometry_dataset_id"]),
            analysis_unit=str(alignment["analysis_unit"]),
            bands=bands,
            truth_variants=cast(Sequence[str], alignment["truth_variants"]),
            geometry_conditions=tuple(condition_map.values()),
            diagnostic_alpha=float(alignment["diagnostic_alpha"]),
        )
        _append_log(
            log_path,
            "rankings_and_truth_built",
            elapsed_seconds=time.perf_counter() - started,
            ranking_rows=len(rankings),
            truth_rows=len(truth),
            scenarios=len(scenario_metadata),
        )
        scores, coverage = evaluate_geometry_des(
            rankings,
            truth,
            scenario_metadata,
            algorithm_dataset_id=str(contract["algorithm_dataset_id"]),
            geometry_dataset_id=str(contract["geometry_dataset_id"]),
            condition_map=condition_map,
            alignment=alignment,
        )
        concordance = geometry_rank_concordance(
            rankings,
            primary_effects,
            condition_map=condition_map,
            alignment=alignment,
        )
        summary = mechanism_distance_summary(scores)
        audit = (
            ledger.assign(observed=ledger["status"].astype(str).eq("observed"))
            .groupby(
                ["generator_id", "mechanism"], observed=True, sort=True, dropna=False
            )
            .agg(
                event_rows=("interaction_id", "size"),
                observed_rows=("observed", "sum"),
                nonzero_rows=("abs_effect", lambda values: int(values.gt(0.0).sum())),
                native_selected_rows=("native_selected", "sum"),
            )
            .reset_index()
        )
        tables = {
            "mechanism_event_audit.tsv": audit,
            "geometry_aligned_pair_rankings.parquet": rankings,
            "geometry_des_scores.tsv": scores,
            "geometry_des_coverage.tsv": coverage,
            "geometry_rank_concordance.tsv": concordance,
            "mechanism_distance_summary.tsv": summary,
        }
        output_records: dict[str, object] = {}
        for filename, table in tables.items():
            path = output / filename
            if path.suffix == ".parquet":
                table.to_parquet(path, index=False, compression="zstd")
            else:
                table.to_csv(path, sep="\t", index=False)
            output_records[filename] = _output_record(path, table)
        _append_log(
            log_path,
            "run_completed",
            elapsed_seconds=time.perf_counter() - started,
            score_rows=len(scores),
            observed_score_rows=int(scores["status"].astype(str).eq("observed").sum()),
        )
        output_records[log_path.name] = {
            "filename": log_path.name,
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        }
        manifest.update(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "real_run": {
                    "schema_version": real_manifest["schema_version"],
                    "code": real_manifest.get("code"),
                },
                "geometry_run": {
                    "schema_version": geometry_manifest["schema_version"],
                    "profile": geometry_manifest["profile"],
                    "code": geometry_manifest.get("code"),
                },
                "resource": {
                    "resource_id": resource_manifest["resource_id"],
                    "version": resource_manifest["version"],
                },
                "release": {
                    "geometry_is_indirect_silver_evidence": True,
                    "coordinate_and_label_p_values_are_endpoint_diagnostics_only": True,
                    "causal_or_direct_contact_claim_allowed": False,
                    "cross_platform_replication_status": release_contract[
                        "cross_platform_replication_status"
                    ],
                },
                "environment": python_environment(
                    environment_name="crychic_project_python",
                    packages=("numpy", "pandas", "pyarrow", "scipy"),
                    threads=1,
                ),
                "outputs": output_records,
            }
        )
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        _append_log(
            log_path,
            "run_failed",
            elapsed_seconds=time.perf_counter() - started,
            error_type=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error),
        )
        manifest.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "failure": {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": str(error),
                },
            }
        )
        write_json(output / "manifest.json", manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--real-run-dir", type=Path, required=True)
    parser.add_argument("--geometry-run-dir", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    manifest = run(
        dataset=arguments.dataset,
        real_run_dir=arguments.real_run_dir.resolve(),
        geometry_run_dir=arguments.geometry_run_dir.resolve(),
        resource_path=arguments.resource.resolve(),
        resource_manifest_path=arguments.resource_manifest.resolve(),
        output_dir=arguments.output_dir.resolve(),
        config_path=arguments.config.resolve(),
        overwrite=arguments.overwrite,
        allow_dirty=arguments.allow_dirty,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "annotate_event_mechanisms",
    "build_mechanism_pair_rankings",
    "evaluate_geometry_des",
    "geometry_pair_axes",
    "geometry_rank_concordance",
    "geometry_truth_scenarios",
    "mechanism_distance_summary",
    "run",
]
