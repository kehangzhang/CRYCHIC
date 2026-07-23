"""Run checksum-bound event-level DES variants on Kuppe or MS.

This runner never refits either method. It consumes persisted CRYCHIC directed
effects, the frozen RC11 pair ranking, and the persisted scSeqCommDiff result.
Kuppe/MS spatial co-localization is an unordered silver endpoint, so directed
and diffusible-long-range variants remain explicitly not estimable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.literature.evaluate_spatial_des_benchmark import evaluate_rankings
from benchmarks.literature.event_level_des import (
    EVENT_BUDGETS,
    MECHANISMS,
    assert_original_count_parity,
    mechanism_annotations,
    original_count_version_audit,
    pair_rankings_from_events,
    prepare_crychic_event_ledger,
    prepare_scseqcommdiff_event_ledger,
    select_top_k_events_by_scope,
    selected_event_diagnostics,
)

SCHEMA_VERSION = "crychic-event-level-des-real-v1"
CONFIG_SCHEMA_VERSION = "crychic-bounded-event-evidence-config-v1"
RC11_SCHEMA_VERSION = "crychic-bounded-core-evidence-real-v1"
COMPONENT_SCHEMA_VERSION = "crychic-component-swap-rc0-v1"
SCSEQ_SCHEMA_VERSIONS = frozenset(
    {
        "crychic-scseqcommdiff-paper-benchmark-v1",
        "crychic-scseqcommdiff-paper-benchmark-v2",
    }
)
RESOURCE_SCHEMA_VERSION = "crychic-connectomedb2020-resource-v1"
CORE_EVENT_FILENAME = "sender_specific_directed_lr_effects.parquet"

CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "kuppe": {
        "core_schemas": {"crychic-kuppe-ctrl-iz-crossfit-run-v3"},
        "truth_schema": "crychic-kuppe-misty-des-truth-v1",
        "truth_dataset": "Kuppe_MI_spatial_CTRL_vs_IZ",
        "condition_map": {},
        "expected_filters": {"variant": "spatial_neighbor_max"},
        "role": "development",
    },
    "ms": {
        "core_schemas": {
            "crychic-ms-ctrl-ca-crossfit-run-v2",
            "crychic-ms-ctrl-ca-crossfit-run-v3",
        },
        "truth_schema": "crychic-ms-spatial-des-truth-v1",
        "truth_dataset": "lerma_martin_ms_ctrl_vs_chronic_active",
        "condition_map": {"CA": "chronic_active", "Ctrl": "control"},
        "expected_filters": {},
        "role": "reused_external_cohort_not_fresh_independent_validation",
    },
}


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
) -> Path:
    outputs = manifest.get("outputs")
    record = outputs.get(key) if isinstance(outputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"manifest does not bind output {key!r}: {root}")
    filename = record.get("filename", key)
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"manifest output {key!r} has an invalid filename")
    path = root / filename
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"bound output checksum mismatch: {path}")
    return path


def _bound_input_record(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"component manifest lacks input {label!r}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise ValueError(f"component input {label!r} lacks an absolute path")
    path = Path(raw_path).resolve()
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"component input checksum mismatch for {label}: {path}")
    return path


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("event-level DES config schema is unsupported")
    budgets = config.get("event_budgets")
    if budgets != list(EVENT_BUDGETS):
        raise ValueError("event-level DES budgets differ from the frozen contract")
    policy = config.get("policy")
    release = config.get("release")
    if not isinstance(policy, Mapping) or not isinstance(release, Mapping):
        raise ValueError("event-level DES config lacks policy/release contracts")
    if any(
        (
            release.get("algorithm_backbone_modified") is not False,
            release.get("canonical_score_replaced") is not False,
            release.get("formal_inference_allowed") is not False,
            release.get("ranking_head_only") is not True,
        )
    ):
        raise ValueError("event-level DES release boundary was altered")
    floor = policy.get("pair_gate_floor")
    if isinstance(floor, bool) or not isinstance(floor, int | float):
        raise ValueError("event-level DES config pair_gate_floor must be numeric")
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError("event-level DES pair_gate_floor must lie in [0, 1]")
    scope = policy.get("top_k_budget_scope")
    if scope not in {"global_across_both_effect_directions", "per_condition"}:
        raise ValueError("event-level DES config has an unsupported Top-K scope")
    comparator = config.get("scseqcommdiff_comparator")
    if (
        not isinstance(comparator, Mapping)
        or comparator.get("top_k_budget_scope") != scope
    ):
        raise ValueError("CRYCHIC and scSeqCommDiff Top-K scopes must match")
    return config


def _load_component_sources(
    component_run: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, dict[str, Any], dict[str, object]]:
    manifest_path = component_run / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != COMPONENT_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("component-swap source is not a completed compatible run")
    input_payload = manifest.get("inputs")
    if not isinstance(input_payload, Mapping):
        raise ValueError("component-swap manifest lacks input provenance")
    scseq_ranking_path = _bound_input_record(
        input_payload.get("scseq_native_ranking"), label="scseq_native_ranking"
    )
    scseq_payload = input_payload.get("scseqcommdiff")
    if not isinstance(scseq_payload, Mapping):
        raise ValueError("component-swap manifest lacks scSeqCommDiff provenance")
    scseq_manifest_path = _bound_input_record(
        scseq_payload.get("manifest"), label="scseqcommdiff.manifest"
    )
    scseq_manifest = _read_json(scseq_manifest_path)
    if (
        scseq_manifest.get("schema_version") not in SCSEQ_SCHEMA_VERSIONS
        or scseq_manifest.get("status") != "complete"
    ):
        raise ValueError("scSeqCommDiff source is not a completed paper run")
    scseq_rds_path = _bound_output(
        scseq_manifest_path.parent, scseq_manifest, "differential_comm.rds"
    )
    ranking = pd.read_csv(scseq_ranking_path, sep="\t")
    component_rank_path = _bound_output(
        component_run, manifest, "condition_cell_pair_rankings.tsv"
    )
    component_rankings = pd.read_csv(component_rank_path, sep="\t")
    arm_a = component_rankings.loc[
        component_rankings["method"].astype(str).eq("A")
    ].copy()
    if arm_a.empty:
        raise ValueError("component-swap source lacks CRYCHIC arm A")
    provenance = {
        "component_manifest": _input_record(manifest_path),
        "component_rankings": _input_record(component_rank_path),
        "scseq_manifest": _input_record(scseq_manifest_path),
        "scseq_rds": _input_record(scseq_rds_path),
        "scseq_native_ranking": _input_record(scseq_ranking_path),
    }
    return arm_a, ranking, scseq_rds_path, scseq_manifest, provenance


def _load_core_events(
    core_run: Path, *, contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    manifest_path = core_run / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") not in contract["core_schemas"]
        or manifest.get("status") != "complete"
    ):
        raise ValueError("current-core source is not a completed compatible run")
    event_path = _bound_output(core_run, manifest, CORE_EVENT_FILENAME)
    events = pd.read_parquet(event_path)
    outputs = cast(Mapping[str, Any], manifest["outputs"])
    record = cast(Mapping[str, Any], outputs[CORE_EVENT_FILENAME])
    if len(events) != int(record["rows"]):
        raise ValueError("current-core event ledger row count mismatch")
    return events, manifest, {
        "manifest": _input_record(manifest_path),
        "events": _input_record(event_path),
    }


def _load_rc11(
    rc11_run: Path, *, dataset: str
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    manifest_path = rc11_run / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != RC11_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("dataset") != dataset
        or manifest.get("formal_inference_allowed") is not False
    ):
        raise ValueError("RC11 source is not a completed benchmark-only run")
    path = _bound_output(
        rc11_run, manifest, "candidate_condition_cell_pair_rankings.tsv"
    )
    ranking = pd.read_csv(path, sep="\t")
    return ranking, manifest, {
        "manifest": _input_record(manifest_path),
        "rankings": _input_record(path),
    }


def _load_truth(
    truth_manifest_path: Path,
    *,
    contract: Mapping[str, Any],
    component_expected: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    manifest = _read_json(truth_manifest_path)
    if (
        manifest.get("schema_version") != contract["truth_schema"]
        or manifest.get("status") != "complete"
        or manifest.get("dataset_id") != contract["truth_dataset"]
    ):
        raise ValueError("spatial truth manifest does not match the dataset contract")
    expected_path = _bound_output(
        truth_manifest_path.parent, manifest, "expected_sets"
    )
    if expected_path.resolve() != Path(str(component_expected.get("path"))).resolve():
        raise ValueError("truth and component-swap expected-set paths disagree")
    if sha256_file(expected_path) != component_expected.get("sha256"):
        raise ValueError("truth and component-swap expected-set checksums disagree")
    return pd.read_csv(expected_path, sep="\t"), manifest, {
        "manifest": _input_record(truth_manifest_path),
        "expected_sets": _input_record(expected_path),
    }


def _load_resource(
    resource_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    manifest = _read_json(resource_manifest_path)
    payload = manifest.get("payload")
    if (
        manifest.get("schema_version") != RESOURCE_SCHEMA_VERSION
        or not isinstance(payload, Mapping)
    ):
        raise ValueError("ConnectomeDB2020 manifest is unsupported")
    filename = payload.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("resource manifest has an invalid payload filename")
    path = resource_manifest_path.parent / filename
    if not path.is_file() or sha256_file(path) != payload.get("sha256"):
        raise ValueError("ConnectomeDB2020 payload checksum mismatch")
    table = pd.read_csv(path, sep="\t")
    if len(table) != int(manifest["rows"]):
        raise ValueError("ConnectomeDB2020 payload row count mismatch")
    return table, manifest, {
        "manifest": _input_record(resource_manifest_path),
        "payload": _input_record(path),
    }


def _export_scseq_ledger(
    *,
    rscript: Path,
    exporter: Path,
    source_rds: Path,
    source_manifest: Mapping[str, Any],
    output_path: Path,
) -> tuple[pd.DataFrame, list[str], float]:
    protocol = source_manifest.get("protocol")
    contrast = protocol.get("contrast_order") if isinstance(protocol, Mapping) else None
    if (
        not isinstance(contrast, list)
        or len(contrast) != 2
        or not all(isinstance(value, str) and value for value in contrast)
    ):
        raise ValueError("scSeqCommDiff manifest lacks a two-level contrast order")
    target, reference = contrast
    command = [
        str(rscript.resolve()),
        str(exporter.resolve()),
        str(source_rds.resolve()),
        str(output_path.resolve()),
        target,
        reference,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            "scSeqCommDiff event export failed: "
            + completed.stderr[-2000:].replace("\n", " ")
        )
    if not output_path.is_file():
        raise RuntimeError("scSeqCommDiff event exporter produced no ledger")
    return pd.read_csv(output_path, sep="\t"), command, elapsed


def _metadata(
    *,
    dataset_id: str,
    method: str,
    method_version: str,
    resource: str,
    semantics: str,
) -> dict[str, str]:
    return {
        "dataset": dataset_id,
        "method": method,
        "method_version": method_version,
        "resource": resource,
        "ranking_semantics": semantics,
    }


def _annotate(
    table: pd.DataFrame,
    *,
    method_family: str,
    variant: str,
    budget: int | None = None,
    mechanism: str | None = None,
) -> pd.DataFrame:
    result = table.copy()
    result.insert(0, "method_family", method_family)
    result.insert(1, "des_variant", variant)
    result.insert(2, "event_budget", budget)
    result.insert(3, "mechanism", mechanism)
    return result


def _evaluate_one(
    rankings: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    method_family: str,
    variant: str,
    budget: int | None = None,
    mechanism: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_datasets = rankings["dataset"].astype(str).unique()
    if len(source_datasets) != 1:
        raise ValueError("one DES variant must contain exactly one source dataset")
    scores, coverage = evaluate_rankings(
        rankings,
        expected,
        scenario="multi_sample",
        dataset_map={str(source_datasets[0]): str(contract["truth_dataset"])},
        condition_map=dict(contract["condition_map"]),
        expected_filters=dict(contract["expected_filters"]),
        score_type="pos",
        weight_exponent=1.0,
        tie_policy="fgsea_native",
        exclude_self_pairs=True,
        ranking_statistic="raw_cardinality",
    )
    return (
        _annotate(
            scores,
            method_family=method_family,
            variant=variant,
            budget=budget,
            mechanism=mechanism,
        ),
        _annotate(
            coverage,
            method_family=method_family,
            variant=variant,
            budget=budget,
            mechanism=mechanism,
        ),
    )


def _summarize(scores: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "method_family",
        "method",
        "method_version",
        "des_variant",
        "event_budget",
        "mechanism",
    ]
    observed = scores.loc[scores["status"].astype(str).eq("observed")].copy()
    return (
        observed.groupby(keys, observed=True, sort=True, dropna=False)["des"]
        .agg(["count", "median", "mean", "min", "max"])
        .reset_index()
    )


def _compare(summary: pd.DataFrame) -> pd.DataFrame:
    endpoint_keys = ["des_variant", "event_budget", "mechanism"]
    records: list[dict[str, object]] = []
    for endpoint, group in summary.groupby(
        endpoint_keys, observed=True, sort=True, dropna=False
    ):
        by_family = group.set_index("method_family")
        if not {"CRYCHIC", "scSeqCommDiff"}.issubset(by_family.index):
            continue
        cry = by_family.loc["CRYCHIC"]
        scseq = by_family.loc["scSeqCommDiff"]
        if isinstance(cry, pd.DataFrame) or isinstance(scseq, pd.DataFrame):
            raise ValueError("an endpoint has duplicate method-family summaries")
        records.append(
            {
                **dict(zip(endpoint_keys, endpoint, strict=True)),
                "crychic_method": cry["method"],
                "crychic_count": int(cry["count"]),
                "crychic_median": float(cry["median"]),
                "crychic_mean": float(cry["mean"]),
                "scseqcommdiff_count": int(scseq["count"]),
                "scseqcommdiff_median": float(scseq["median"]),
                "scseqcommdiff_mean": float(scseq["mean"]),
                "median_delta_crychic_minus_scseqcommdiff": float(
                    cry["median"] - scseq["median"]
                ),
                "mean_delta_crychic_minus_scseqcommdiff": float(
                    cry["mean"] - scseq["mean"]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _availability(dataset: str) -> pd.DataFrame:
    role = str(CONTRACTS[dataset]["role"])
    common = {
        "dataset": dataset,
        "dataset_role": role,
        "formal_inference_allowed": False,
        "spatial_truth_direction": "unordered_canonical",
    }
    records: list[dict[str, object]] = [
        {
            **common,
            "des_variant": "original_count_des",
            "event_budget": pd.NA,
            "mechanism": None,
            "status": "observed",
            "reason_code": None,
        },
        {
            **common,
            "des_variant": "continuous_weighted_des",
            "event_budget": pd.NA,
            "mechanism": None,
            "status": "observed",
            "reason_code": None,
        },
        {
            **common,
            "des_variant": "direction_preserving_des",
            "event_budget": pd.NA,
            "mechanism": None,
            "status": "not_estimable",
            "reason_code": "spatial_truth_collapses_sender_receiver_direction",
        },
        {
            **common,
            "des_variant": "mechanism_stratified_des",
            "event_budget": pd.NA,
            "mechanism": "diffusible_long_range",
            "status": "not_estimable",
            "reason_code": "connectomedb2020_has_no_diffusion_range_annotation",
        },
    ]
    records.extend(
        {
            **common,
            "des_variant": "top_k_count_matched_des",
            "event_budget": budget,
            "mechanism": None,
            "status": "observed",
            "reason_code": None,
        }
        for budget in EVENT_BUDGETS
    )
    records.extend(
        {
            **common,
            "des_variant": "mechanism_stratified_des",
            "event_budget": pd.NA,
            "mechanism": mechanism,
            "status": "observed",
            "reason_code": None,
        }
        for mechanism in MECHANISMS
    )
    result = pd.DataFrame.from_records(records)
    result["event_budget"] = result["event_budget"].astype("Int64")
    return result


def run(
    *,
    dataset: str,
    core_run: Path,
    rc11_run: Path,
    component_run: Path,
    truth_manifest_path: Path,
    resource_manifest_path: Path,
    config_path: Path,
    rscript: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    if dataset not in CONTRACTS:
        raise ValueError("dataset must be kuppe or ms")
    contract = CONTRACTS[dataset]
    config = _load_config(config_path.resolve())
    policy = cast(Mapping[str, Any], config["policy"])
    arm_a, scseq_axes, scseq_rds, scseq_manifest, component_provenance = (
        _load_component_sources(component_run.resolve())
    )
    component_manifest = _read_json(component_run.resolve() / "manifest.json")
    if component_manifest.get("dataset") != dataset:
        raise ValueError("component-swap dataset does not match the request")
    component_inputs = cast(Mapping[str, Any], component_manifest["inputs"])
    expected_record = component_inputs.get("expected_sets")
    if not isinstance(expected_record, Mapping):
        raise ValueError("component-swap source does not bind expected sets")
    core_events, core_manifest, core_provenance = _load_core_events(
        core_run.resolve(), contract=contract
    )
    rc11_axes, rc11_manifest, rc11_provenance = _load_rc11(
        rc11_run.resolve(), dataset=dataset
    )
    expected, truth_manifest, truth_provenance = _load_truth(
        truth_manifest_path.resolve(),
        contract=contract,
        component_expected=expected_record,
    )
    resource, resource_manifest, resource_provenance = _load_resource(
        resource_manifest_path.resolve()
    )
    resource_id = str(resource_manifest["resource_id"])
    lookup = mechanism_annotations(resource)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    published = False
    try:
        exporter = Path(__file__).with_name("export_scseqcommdiff_event_ledger.R")
        raw_scseq_path = staged / "scseqcommdiff_event_ledger.tsv.gz"
        raw_scseq, export_command, export_elapsed = _export_scseq_ledger(
            rscript=rscript,
            exporter=exporter,
            source_rds=scseq_rds,
            source_manifest=scseq_manifest,
            output_path=raw_scseq_path,
        )
        cry_ledger = prepare_crychic_event_ledger(
            core_events,
            rc11_axes,
            lookup,
            pair_gate_floor=float(policy["pair_gate_floor"]),
        )
        scseq_ledger = prepare_scseqcommdiff_event_ledger(raw_scseq, lookup)

        cry_dataset = str(arm_a["dataset"].iloc[0])
        scseq_dataset = str(scseq_axes["dataset"].iloc[0])
        cry_original = pair_rankings_from_events(
            cry_ledger,
            arm_a,
            selected=cry_ledger["native_selected"],
            weight_column=None,
            metadata=_metadata(
                dataset_id=cry_dataset,
                method="CRYCHIC_current_native_one_se",
                method_version=str(core_manifest.get("package_version", "current")),
                resource=resource_id,
                semantics=(
                    "cardinality_of_one_standard_error_stable_directed_lr_after_"
                    "unordered_pair_collapse"
                ),
            ),
        )
        count_version_audit, count_version_summary = original_count_version_audit(
            cry_original, arm_a
        )
        scseq_original = pair_rankings_from_events(
            scseq_ledger,
            scseq_axes,
            selected=scseq_ledger["native_selected"],
            weight_column=None,
            metadata=_metadata(
                dataset_id=scseq_dataset,
                method="scseqcommdiff",
                method_version="2.0.0",
                resource=resource_id,
                semantics=(
                    "native_raw_p_lt_0.05_and_intracellular_gate_after_"
                    "unordered_pair_collapse"
                ),
            ),
        )
        assert_original_count_parity(scseq_original, scseq_axes)

        rankings: list[pd.DataFrame] = []
        score_tables: list[pd.DataFrame] = []
        coverage_tables: list[pd.DataFrame] = []
        cry_selections: dict[str, pd.Series] = {
            "original_count_des": cry_ledger["native_selected"]
        }
        scseq_selections: dict[str, pd.Series] = {
            "original_count_des": scseq_ledger["native_selected"]
        }

        def evaluate_pair(
            cry_ranking: pd.DataFrame,
            scseq_ranking: pd.DataFrame,
            *,
            variant: str,
            budget: int | None = None,
            mechanism: str | None = None,
        ) -> None:
            rankings.extend([cry_ranking, scseq_ranking])
            for ranking, family in (
                (cry_ranking, "CRYCHIC"),
                (scseq_ranking, "scSeqCommDiff"),
            ):
                scores, coverage = _evaluate_one(
                    ranking,
                    expected,
                    contract=contract,
                    method_family=family,
                    variant=variant,
                    budget=budget,
                    mechanism=mechanism,
                )
                score_tables.append(scores)
                coverage_tables.append(coverage)

        evaluate_pair(
            cry_original,
            scseq_original,
            variant="original_count_des",
        )

        cry_top_eligible = (
            cry_ledger["status"].astype(str).eq("observed")
            & cry_ledger["abs_effect"].gt(0.0)
        )
        scseq_top_eligible = scseq_ledger["top_k_eligible"].astype(bool)
        top_k_scope = str(policy["top_k_budget_scope"])
        top_k_semantics = (
            "per_condition" if top_k_scope == "per_condition" else "global"
        )
        for budget in EVENT_BUDGETS:
            cry_selected = select_top_k_events_by_scope(
                cry_ledger,
                budget=budget,
                evidence_column="bounded_event_evidence",
                eligible=cry_top_eligible,
                scope=top_k_scope,
            )
            scseq_selected = select_top_k_events_by_scope(
                scseq_ledger,
                budget=budget,
                evidence_column="event_evidence",
                eligible=scseq_top_eligible,
                scope=top_k_scope,
            )
            key = f"top_k_count_matched_des_k{budget}"
            cry_selections[key] = cry_selected
            scseq_selections[key] = scseq_selected
            evaluate_pair(
                pair_rankings_from_events(
                    cry_ledger,
                    rc11_axes,
                    selected=cry_selected,
                    weight_column=None,
                    metadata=_metadata(
                        dataset_id=cry_dataset,
                        method=str(config["method"]),
                        method_version=str(config["method_version"]),
                        resource=resource_id,
                        semantics=(
                            f"{top_k_semantics}_top_{budget}_pair_gate_times_abs_hc2_z"
                        ),
                    ),
                ),
                pair_rankings_from_events(
                    scseq_ledger,
                    scseq_axes,
                    selected=scseq_selected,
                    weight_column=None,
                    metadata=_metadata(
                        dataset_id=scseq_dataset,
                        method="scseqcommdiff",
                        method_version="2.0.0",
                        resource=resource_id,
                        semantics=(
                            f"{top_k_semantics}_top_{budget}_negative_log10_native_p"
                        ),
                    ),
                ),
                variant="top_k_count_matched_des",
                budget=budget,
            )

        cry_continuous = (
            cry_ledger["status"].astype(str).eq("observed")
            & cry_ledger["abs_effect"].gt(0.0)
        )
        scseq_continuous = (
            scseq_ledger["top_k_eligible"].astype(bool)
            & scseq_ledger["abs_effect"].gt(0.0)
        )
        cry_selections["continuous_weighted_des"] = cry_continuous
        scseq_selections["continuous_weighted_des"] = scseq_continuous
        evaluate_pair(
            pair_rankings_from_events(
                cry_ledger,
                rc11_axes,
                selected=cry_continuous,
                weight_column="bounded_abs_effect",
                metadata=_metadata(
                    dataset_id=cry_dataset,
                    method=str(config["method"]),
                    method_version=str(config["method_version"]),
                    resource=resource_id,
                    semantics="sum_abs_effect_times_bounded_rc11_pair_gate",
                ),
            ),
            pair_rankings_from_events(
                scseq_ledger,
                scseq_axes,
                selected=scseq_continuous,
                weight_column="continuous_weight",
                metadata=_metadata(
                    dataset_id=scseq_dataset,
                    method="scseqcommdiff",
                    method_version="2.0.0",
                    resource=resource_id,
                    semantics="sum_abs_native_intercellular_score_difference",
                ),
            ),
            variant="continuous_weighted_des",
        )

        for mechanism in MECHANISMS:
            cry_selected = cry_continuous & cry_ledger["mechanism"].eq(mechanism)
            scseq_selected = scseq_continuous & scseq_ledger["mechanism"].eq(
                mechanism
            )
            key = f"mechanism_stratified_des_{mechanism}"
            cry_selections[key] = cry_selected
            scseq_selections[key] = scseq_selected
            evaluate_pair(
                pair_rankings_from_events(
                    cry_ledger,
                    rc11_axes,
                    selected=cry_selected,
                    weight_column="bounded_abs_effect",
                    metadata=_metadata(
                        dataset_id=cry_dataset,
                        method=str(config["method"]),
                        method_version=str(config["method_version"]),
                        resource=resource_id,
                        semantics=f"{mechanism}_sum_abs_effect_times_bounded_pair_gate",
                    ),
                ),
                pair_rankings_from_events(
                    scseq_ledger,
                    scseq_axes,
                    selected=scseq_selected,
                    weight_column="continuous_weight",
                    metadata=_metadata(
                        dataset_id=scseq_dataset,
                        method="scseqcommdiff",
                        method_version="2.0.0",
                        resource=resource_id,
                        semantics=f"{mechanism}_sum_abs_native_score_difference",
                    ),
                ),
                variant="mechanism_stratified_des",
                mechanism=mechanism,
            )

        combined_rankings = pd.concat(rankings, ignore_index=True, sort=False)
        scores = pd.concat(score_tables, ignore_index=True, sort=False)
        coverage = pd.concat(coverage_tables, ignore_index=True, sort=False)
        summary = _summarize(scores)
        comparison = _compare(summary)
        diagnostics = pd.concat(
            [
                selected_event_diagnostics(cry_ledger, cry_selections).assign(
                    method_family="CRYCHIC"
                ),
                selected_event_diagnostics(scseq_ledger, scseq_selections).assign(
                    method_family="scSeqCommDiff"
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        availability = _availability(dataset)

        cry_ledger_path = staged / "crychic_event_ledger.parquet"
        scseq_ledger_path = staged / "scseqcommdiff_event_ledger.parquet"
        cry_ledger.to_parquet(cry_ledger_path, index=False)
        scseq_ledger.to_parquet(scseq_ledger_path, index=False)
        tables = {
            "condition_cell_pair_rankings.tsv": combined_rankings,
            "spatial_des_scores.tsv": scores,
            "spatial_des_coverage.tsv": coverage,
            "spatial_des_summary.tsv": summary,
            "method_comparison.tsv": comparison,
            "selected_event_diagnostics.tsv": diagnostics,
            "endpoint_availability.tsv": availability,
            "crychic_original_count_version_audit.tsv": count_version_audit,
        }
        for filename, table in tables.items():
            table.to_csv(staged / filename, sep="\t", index=False, lineterminator="\n")
        output_records = {
            filename: _output_record(staged / filename, table)
            for filename, table in tables.items()
        }
        output_records.update(
            {
                "crychic_event_ledger.parquet": _output_record(
                    cry_ledger_path, cry_ledger
                ),
                "scseqcommdiff_event_ledger.parquet": _output_record(
                    scseq_ledger_path, scseq_ledger
                ),
                "scseqcommdiff_event_ledger.tsv.gz": _output_record(
                    raw_scseq_path, raw_scseq
                ),
            }
        )
        original = comparison.loc[
            comparison["des_variant"].eq("original_count_des")
        ]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "dataset": dataset,
            "dataset_role": contract["role"],
            "method": config["method"],
            "method_version": config["method_version"],
            "formal_inference_allowed": False,
            "canonical_score_replaced": False,
            "algorithm_backbone_modified": False,
            "ranking_head_modified": True,
            "protocol": {
                "event_budgets": list(EVENT_BUDGETS),
                "top_k_budget_scope": str(policy["top_k_budget_scope"]),
                "cell_pair_direction": "unordered_canonical",
                "missing_policy": "typed_not_estimable_never_zero_imputed",
                "pair_gate_floor": float(policy["pair_gate_floor"]),
                "spatial_endpoint": "silver_standard_not_event_truth",
            },
            "original_count_parity": {
                "crychic_current_vs_historical_component_arm_a": (
                    count_version_summary
                ),
                "scseqcommdiff_native_exact": True,
            },
            "original_count_comparison": (
                original.iloc[0].to_dict() if len(original) == 1 else None
            ),
            "inputs": {
                "config": _input_record(config_path.resolve()),
                "current_core": core_provenance,
                "rc11": rc11_provenance,
                "component_swap": component_provenance,
                "truth": truth_provenance,
                "resource": resource_provenance,
                "scseq_exporter": _input_record(exporter),
            },
            "source_code": {
                "current_core": core_manifest.get("code"),
                "rc11": rc11_manifest.get("source_code"),
                "truth": truth_manifest.get("code"),
                "runner": git_metadata(Path(__file__).resolve().parents[2]),
            },
            "execution": {
                "r_export_command": export_command,
                "r_export_elapsed_seconds": export_elapsed,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "limitations": [
                "kuppe_is_a_development_dataset",
                "ms_was_previously_inspected_for_rc11_and_is_not_fresh_for_rc13",
                "spatial_colocalization_is_an_indirect_unordered_pair_proxy",
                "rc13_is_an_unreleased_benchmark_only_ranking_head",
                "direction_preserving_des_is_not_estimable_from_the_spatial_truth",
                "diffusible_long_range_des_is_not_estimable_from_connectomedb2020",
                "no_formal_p_or_q_values_are_claimed_for_rc13",
            ],
            "outputs": output_records,
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
    parser.add_argument("--dataset", choices=("kuppe", "ms"), required=True)
    parser.add_argument("--core-run", type=Path, required=True)
    parser.add_argument("--rc11-run", type=Path, required=True)
    parser.add_argument("--component-run", type=Path, required=True)
    parser.add_argument("--truth-manifest", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(
        dataset=args.dataset,
        core_run=args.core_run,
        rc11_run=args.rc11_run,
        component_run=args.component_run,
        truth_manifest_path=args.truth_manifest,
        resource_manifest_path=args.resource_manifest,
        config_path=args.config,
        rscript=args.rscript,
        output_dir=args.output_dir,
    )
    print(json.dumps(json_safe(manifest), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
