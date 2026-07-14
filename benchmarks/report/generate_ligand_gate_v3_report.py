"""Generate the ligand-gate v3 synthetic diagnostic report.

The report is a presentation layer over frozen G1.5 campaign summaries. It does
not rerun scoring, reinterpret ranks as probabilities, or promote synthetic
diagnostics to biological validation.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

REPORT_SCHEMA_VERSION = "crychic-ligand-gate-v3-report.v1"
CAMPAIGN_SCHEMA_VERSION = "crychic-public-family-common-g1.5-campaign-v1"

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
BLACK = "#222222"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"
VERY_LIGHT_GRAY = "#F2F2F2"

CLAIMS: dict[str, bool] = {
    "biological_validation": False,
    "complete_pipeline_oof_certification": False,
    "family_common_full_oof_certification": False,
    "default_switch_allowed": False,
    "method_superiority": False,
}

SNAPSHOT_LABELS = {
    "pre_gate": "Historical pre-gate\n(two-seed full seven)",
    "full7": "Ligand-gate v3\n(two-seed full seven)",
    "seed003": "Ligand-gate v3\n(seed003 debug)",
}
SNAPSHOT_ORDER = {"pre_gate": 0, "full7": 1, "seed003": 2}
SCENARIO_ORDER = (
    "global_null",
    "abundance_only",
    "ligand_only",
    "target_only",
    "receiver_autonomous",
    "receptor_knockout",
)
ALL_SCENARIOS = ("active", *SCENARIO_ORDER)
CAMPAIGN_SEEDS = ("public-g15-001", "public-g15-002")
DEBUG_SEEDS = ("public-g15-003",)
DEBUG_SCENARIOS = (
    "active",
    "ligand_only",
    "target_only",
    "receptor_knockout",
)
CAMPAIGN_TRUE_CHECKS = (
    "active_and_paired_ligand_only_executed",
    "all_generated_inputs_complete_paired",
    "all_observed_receiver_programs_use_subject_equal_reference_transform",
    "all_runs_used_public_crossfit_workflow",
    "eligible_for_campaign_metric_interpretation",
    "executed_full_seven_scenario_contract",
    "executed_multiple_seeds",
    "frozen_contract_contains_all_seven_g1_5_scenarios",
    "frozen_contract_contains_multiple_known_edges",
    "frozen_contract_contains_multiple_seeds",
    "not_estimable_components_never_count_as_pass",
)
SVG_HASHSALT = "crychic-ligand-gate-v3-report-v1"
SCENARIO_LABELS = {
    "global_null": "Global\nnull",
    "abundance_only": "Abundance\nonly",
    "ligand_only": "Ligand\nonly",
    "target_only": "Target\nonly",
    "receiver_autonomous": "Receiver\nautonomous",
    "receptor_knockout": "Receptor\nknockout",
}
VIEW_ORDER = (
    ("state", "member_unresolved"),
    ("state", "sender_resolved"),
    ("ecosystem", "member_unresolved"),
    ("ecosystem", "sender_resolved"),
)
VIEW_LABELS = {
    ("state", "member_unresolved"): "State\nmember",
    ("state", "sender_resolved"): "State\nsender",
    ("ecosystem", "member_unresolved"): "Ecosystem\nmember",
    ("ecosystem", "sender_resolved"): "Ecosystem\nsender",
}


@dataclass(frozen=True)
class Snapshot:
    """One validated frozen campaign summary."""

    key: str
    role: str
    path: Path
    payload: Mapping[str, Any]

    @property
    def seeds(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(record["seed_id"])
                    for record in self.payload["records"]
                    if isinstance(record, Mapping) and "seed_id" in record
                }
            )
        )

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(record["scenario"])
                for record in self.payload["records"]
                if isinstance(record, Mapping) and "scenario" in record
            )
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _git_state(repo_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = result.stdout if result.returncode == 0 else ""
    return {
        "dirty": bool(status),
        "status_entry_count": len(status.splitlines()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _generated_at(value: str | None) -> str:
    if value is not None:
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("generated_at must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError("generated_at must include an explicit timezone")
    else:
        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if source_date_epoch is not None:
            timestamp = datetime.fromtimestamp(int(source_date_epoch), tz=UTC)
        else:
            timestamp = datetime.now(tz=UTC)
    return timestamp.isoformat()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"campaign summary must be a JSON object: {path}")
    return payload


def _control_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    aggregate = payload.get("aggregate_metrics")
    if not isinstance(aggregate, Mapping):
        raise ValueError("aggregate_metrics must be an object")
    rows = aggregate.get("control_family_false_positive_and_selection")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("control false-positive metrics must be a list of objects")
    return rows


def _validate_snapshot(
    key: str,
    role: str,
    path: Path,
    payload: Mapping[str, Any],
) -> Snapshot:
    schema = payload.get("schema_version")
    if schema != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(
            f"{role} schema must be {CAMPAIGN_SCHEMA_VERSION!r}; got {schema!r}"
        )
    claims = payload.get("claims")
    if not isinstance(claims, Mapping) or claims.get("development_only") is not True:
        raise ValueError(f"{role} must declare development_only=true")
    invalid_claims = [
        name for name, value in CLAIMS.items() if claims.get(name) is not value
    ]
    if invalid_claims:
        raise ValueError(f"{role} has unsafe or missing claim flags: {invalid_claims}")
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or not records
        or not all(isinstance(record, Mapping) for record in records)
    ):
        raise ValueError(f"{role} records must be a non-empty list")
    snapshot = Snapshot(key=key, role=role, path=path, payload=payload)
    registry = payload.get("registry")
    if not isinstance(registry, Mapping):
        raise ValueError(f"{role} registry must be an object")
    registry_sha256 = registry.get("sha256")
    profile = registry.get("profile")
    if not isinstance(registry_sha256, str) or len(registry_sha256) != 64:
        raise ValueError(f"{role} registry sha256 must be a 64-character string")
    if not isinstance(profile, str) or not profile:
        raise ValueError(f"{role} registry profile must be a non-empty string")
    record_profiles = {record.get("profile") for record in records}
    if record_profiles != {profile}:
        raise ValueError(
            f"{role} record profiles must equal registry profile {profile!r}"
        )
    known_edges = payload.get("known_edges")
    if (
        not isinstance(known_edges, list)
        or len(known_edges) < 2
        or not all(isinstance(edge, Mapping) for edge in known_edges)
    ):
        raise ValueError(f"{role} must contain multiple known-edge objects")
    edge_identities = [_known_edge_identity(edge, role) for edge in known_edges]
    if len(set(edge_identities)) != len(edge_identities):
        raise ValueError(f"{role} known-edge identities must be unique")

    if key in {"pre_gate", "full7"}:
        if payload.get("scope") != "development_full_seven_scenario_diagnostic":
            raise ValueError(f"{role} must have the full-seven campaign scope")
        if snapshot.seeds != CAMPAIGN_SEEDS:
            raise ValueError(f"{role} must contain exactly seeds 001 and 002")
        if set(snapshot.scenarios) != set(ALL_SCENARIOS):
            raise ValueError(
                f"{role} must contain exactly the seven campaign scenarios"
            )
        _validate_record_grid(snapshot, CAMPAIGN_SEEDS, ALL_SCENARIOS)
        checks = payload.get("checks")
        if not isinstance(checks, Mapping):
            raise ValueError(f"{role} checks must be an object")
        failed_checks = [
            name for name in CAMPAIGN_TRUE_CHECKS if checks.get(name) is not True
        ]
        if failed_checks:
            raise ValueError(f"{role} failed campaign contract checks: {failed_checks}")
        _validate_registry_execution(snapshot, CAMPAIGN_SEEDS, ALL_SCENARIOS)
    if key == "seed003":
        if snapshot.seeds != DEBUG_SEEDS:
            raise ValueError("seed003 summary must contain only public-g15-003")
        if (
            payload.get("scope")
            != "single_seed_public_workflow_debug_excluded_from_campaign"
        ):
            raise ValueError(
                "seed003 summary must remain excluded from campaign claims"
            )
        if set(snapshot.scenarios) != set(DEBUG_SCENARIOS):
            raise ValueError(
                "seed003 summary must contain exactly the four debug scenarios"
            )
        _validate_record_grid(snapshot, DEBUG_SEEDS, DEBUG_SCENARIOS)
        _validate_registry_execution(snapshot, DEBUG_SEEDS, DEBUG_SCENARIOS)
    _control_rows(payload)
    return snapshot


def _known_edge_identity(edge: Mapping[str, Any], role: str) -> tuple[str, ...]:
    fields = ("known_edge_id", "harmonized_interaction_id", "ligand", "receptor")
    values = tuple(edge.get(field) for field in fields)
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{role} known-edge identity fields must be non-empty strings")
    return tuple(str(value) for value in values)


def _snapshot_known_edges(snapshot: Snapshot) -> frozenset[tuple[str, ...]]:
    known_edges = snapshot.payload["known_edges"]
    return frozenset(_known_edge_identity(edge, snapshot.role) for edge in known_edges)


def _validate_record_grid(
    snapshot: Snapshot,
    seeds: Sequence[str],
    scenarios: Sequence[str],
) -> None:
    observed = [
        (str(record["seed_id"]), str(record["scenario"]))
        for record in snapshot.payload["records"]
    ]
    expected = {(seed, scenario) for seed in seeds for scenario in scenarios}
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(
            f"{snapshot.role} records must form the exact seed-scenario grid"
        )


def _validate_registry_execution(
    snapshot: Snapshot,
    seeds: Sequence[str],
    scenarios: Sequence[str],
) -> None:
    registry = snapshot.payload["registry"]
    if set(registry.get("executed_seed_ids", ())) != set(seeds):
        raise ValueError(
            f"{snapshot.role} registry executed_seed_ids disagree with records"
        )
    if set(registry.get("executed_scenarios", ())) != set(scenarios):
        raise ValueError(
            f"{snapshot.role} registry executed_scenarios disagree with records"
        )
    if set(registry.get("all_preregistered_scenarios", ())) != set(ALL_SCENARIOS):
        raise ValueError(
            f"{snapshot.role} registry does not preregister all seven scenarios"
        )
    if set(registry.get("all_preregistered_seed_ids", ())) != {
        *CAMPAIGN_SEEDS,
        *DEBUG_SEEDS,
    }:
        raise ValueError(f"{snapshot.role} registry seed preregistration is incomplete")


def _validate_snapshot_set(snapshots: Sequence[Snapshot]) -> None:
    by_key = {snapshot.key: snapshot for snapshot in snapshots}
    if set(by_key) != {"pre_gate", "full7", "seed003"}:
        raise ValueError("report requires pre_gate, full7, and seed003 snapshots")
    pre = by_key["pre_gate"]
    full7 = by_key["full7"]
    debug = by_key["seed003"]
    registry_shas = {
        str(snapshot.payload["registry"]["sha256"]) for snapshot in snapshots
    }
    if len(registry_shas) != 1:
        raise ValueError("all snapshots must bind the same registry SHA256")
    profiles = {str(snapshot.payload["registry"]["profile"]) for snapshot in snapshots}
    if len(profiles) != 1:
        raise ValueError("all snapshots must use the same profile")
    known_edge_sets = {_snapshot_known_edges(snapshot) for snapshot in snapshots}
    if len(known_edge_sets) != 1:
        raise ValueError("all snapshots must contain the same known-edge set")
    campaign_ids = {snapshot.payload.get("campaign_id") for snapshot in snapshots}
    if len(campaign_ids) != 1:
        raise ValueError("all snapshots must use the same campaign_id")
    if pre.seeds != full7.seeds or pre.scenarios != full7.scenarios:
        raise ValueError("pre_gate and full7 execution contracts must be identical")
    debug_checks = debug.payload.get("checks")
    if not isinstance(debug_checks, Mapping):
        raise ValueError("seed003 debug checks must be an object")
    if debug_checks.get("eligible_for_campaign_metric_interpretation") is not False:
        raise ValueError(
            "seed003 debug summary must remain ineligible for campaign metrics"
        )


def _load_snapshot(key: str, role: str, path: Path) -> Snapshot:
    resolved = path.expanduser().resolve()
    return _validate_snapshot(key, role, resolved, _read_json(resolved))


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    for prefix, root in (("repo", repo_root), ("workspace", repo_root.parent)):
        try:
            return f"{prefix}:{resolved.relative_to(root.resolve()).as_posix()}"
        except ValueError:
            continue
    return path.name


def _input_manifest(snapshots: Sequence[Snapshot], repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        rows.append(
            {
                "snapshot": snapshot.key,
                "role": snapshot.role,
                "path": _display_path(snapshot.path, repo_root),
                "bytes": snapshot.path.stat().st_size,
                "sha256": _sha256(snapshot.path),
                "schema_version": snapshot.payload["schema_version"],
                "campaign_id": snapshot.payload.get("campaign_id"),
                "scope": snapshot.payload.get("scope"),
                "registry_sha256": snapshot.payload["registry"]["sha256"],
                "profile": snapshot.payload["registry"]["profile"],
                "seed_ids": ";".join(snapshot.seeds),
                "scenarios": ";".join(snapshot.scenarios),
                "known_edge_ids": ";".join(
                    sorted(identity[0] for identity in _snapshot_known_edges(snapshot))
                ),
            }
        )
    return pd.DataFrame(rows)


def _source_binding_audit(
    snapshots: Sequence[Snapshot], repo_root: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        source_sha256 = snapshot.payload.get("source_sha256")
        if not isinstance(source_sha256, Mapping):
            raise ValueError(f"{snapshot.role} source_sha256 must be an object")
        for relative_path, recorded_sha256 in sorted(source_sha256.items()):
            candidate = repo_root / str(relative_path)
            exists = candidate.is_file()
            current_sha256 = _sha256(candidate) if exists else None
            rows.append(
                {
                    "snapshot": snapshot.key,
                    "relative_path": str(relative_path),
                    "recorded_sha256": str(recorded_sha256),
                    "current_sha256": current_sha256,
                    "current_file_exists": exists,
                    "matches_current_tree": bool(
                        exists and current_sha256 == str(recorded_sha256)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _positive_evaluation_count(rate: float, denominator: int) -> int:
    if not 0 <= rate <= 1 or denominator <= 0:
        raise ValueError("evaluation rate and denominator are outside their contract")
    expected = rate * denominator
    rounded = round(expected)
    if not np.isclose(expected, rounded, rtol=0, atol=1e-8):
        raise ValueError("evaluation rate is incompatible with its integer denominator")
    return rounded


def _target_only_data(snapshots: Sequence[Snapshot]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        target_rows = [
            row
            for row in _control_rows(snapshot.payload)
            if row["scenario"] == "target_only"
        ]
        if len(target_rows) != 1:
            raise ValueError(f"{snapshot.role} must contain one target_only aggregate")
        row = target_rows[0]
        denominator = int(row["n_seed_mode_evaluations"])
        common = {
            "panel": "A",
            "snapshot": snapshot.key,
            "snapshot_label": SNAPSHOT_LABELS[snapshot.key].replace("\n", " "),
            "snapshot_order": SNAPSHOT_ORDER[snapshot.key],
            "scenario": "target_only",
            "n_seed_runs": int(row["n_seed_runs"]),
            "n_seed_mode_evaluations": denominator,
        }
        paired_rate = float(row["any_positive_integrated_family_rate"])
        raw_rate = float(row["any_positive_raw_integrated_family_rate"])
        rows.extend(
            (
                {
                    **common,
                    "estimand": "paired_integrated",
                    "any_positive_family_rate": paired_rate,
                    "positive_seed_mode_evaluations": _positive_evaluation_count(
                        paired_rate, denominator
                    ),
                    "mean_positive_families": float(
                        row["mean_positive_integrated_families"]
                    ),
                },
                {
                    **common,
                    "estimand": "raw_integrated",
                    "any_positive_family_rate": raw_rate,
                    "positive_seed_mode_evaluations": _positive_evaluation_count(
                        raw_rate, denominator
                    ),
                    "mean_positive_families": float(
                        row["mean_positive_raw_integrated_families"]
                    ),
                },
            )
        )
    return pd.DataFrame(rows)


def _active_data(snapshots: Sequence[Snapshot]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if snapshot.key == "pre_gate":
            continue
        aggregate = snapshot.payload["aggregate_metrics"]
        active = aggregate.get("active_vs_ligand_only")
        if not isinstance(active, Mapping) or not isinstance(
            active.get("paired_records"), list
        ):
            raise ValueError(f"{snapshot.role} active paired records are missing")
        for record in active["paired_records"]:
            if not isinstance(record, Mapping):
                raise ValueError("active paired records must be objects")
            mode = str(record["mode"])
            score_kind = str(record["score_kind"])
            view = (mode, score_kind)
            if view not in VIEW_ORDER:
                raise ValueError(f"unexpected active score view: {view}")
            detail_rows.append(
                {
                    "panel": "A",
                    "snapshot": snapshot.key,
                    "snapshot_label": SNAPSHOT_LABELS[snapshot.key].replace("\n", " "),
                    "seed_id": record["seed_id"],
                    "known_edge_id": record["known_edge_id"],
                    "mode": mode,
                    "score_kind": score_kind,
                    "view_order": VIEW_ORDER.index(view),
                    "view_label": VIEW_LABELS[view].replace("\n", " "),
                    "active_minus_ligand_only_margin": float(
                        record["active_minus_ligand_only_margin"]
                    ),
                    "active_recovered": bool(record["active_recovered"]),
                    "coverage_loss": float(record["coverage_loss"]),
                    "active_mean": float(record["active_mean"]),
                    "ligand_only_mean": float(record["reference_mean"]),
                }
            )
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise ValueError("no active paired records were found")
    summary = (
        detail.groupby(
            [
                "snapshot",
                "snapshot_label",
                "mode",
                "score_kind",
                "view_order",
                "view_label",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            n_pairs=("active_recovered", "size"),
            recovered_fraction=("active_recovered", "mean"),
            positive_margin_fraction=(
                "active_minus_ligand_only_margin",
                lambda values: float((values > 0).mean()),
            ),
            mean_margin=("active_minus_ligand_only_margin", "mean"),
            minimum_margin=("active_minus_ligand_only_margin", "min"),
            maximum_coverage_loss=("coverage_loss", "max"),
        )
        .sort_values(["snapshot", "view_order"])
        .reset_index(drop=True)
    )
    summary.insert(0, "panel", "B")
    return detail, summary


def _control_specificity_data(full7: Snapshot) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in _control_rows(full7.payload):
        scenario = str(record["scenario"])
        if scenario not in SCENARIO_ORDER:
            continue
        denominator = int(record["n_seed_mode_evaluations"])
        common = {
            "panel": "A",
            "scenario": scenario,
            "scenario_order": SCENARIO_ORDER.index(scenario),
            "scenario_label": SCENARIO_LABELS[scenario].replace("\n", " "),
            "n_seed_runs": int(record["n_seed_runs"]),
            "n_seed_mode_evaluations": denominator,
        }
        paired_rate = float(record["any_positive_integrated_family_rate"])
        raw_rate = float(record["any_positive_raw_integrated_family_rate"])
        rows.extend(
            (
                {
                    **common,
                    "estimand": "paired_integrated",
                    "any_positive_family_rate": paired_rate,
                    "positive_seed_mode_evaluations": _positive_evaluation_count(
                        paired_rate, denominator
                    ),
                },
                {
                    **common,
                    "estimand": "raw_integrated",
                    "any_positive_family_rate": raw_rate,
                    "positive_seed_mode_evaluations": _positive_evaluation_count(
                        raw_rate, denominator
                    ),
                },
            )
        )
    table = pd.DataFrame(rows)
    expected = set(SCENARIO_ORDER)
    if set(table["scenario"]) != expected:
        raise ValueError("full7 control specificity table is incomplete")
    return table


def _gate_reason_data(full7: Snapshot) -> pd.DataFrame:
    counts: dict[tuple[str, str], int] = {}
    totals: dict[str, int] = {}
    for record in full7.payload["records"]:
        if not isinstance(record, Mapping):
            raise ValueError("campaign records must be objects")
        scenario = str(record["scenario"])
        if scenario not in SCENARIO_ORDER:
            continue
        known_scores = record.get("known_edge_scores")
        if not isinstance(known_scores, list):
            raise ValueError("known_edge_scores must be a list")
        for score in known_scores:
            if not isinstance(score, Mapping):
                raise ValueError("known-edge score must be an object")
            reason_counts = score.get("reason_counts")
            if not isinstance(reason_counts, Mapping):
                raise ValueError("known-edge reason_counts must be an object")
            for reason, count in reason_counts.items():
                numeric_count = int(count)
                counts[(scenario, str(reason))] = (
                    counts.get((scenario, str(reason)), 0) + numeric_count
                )
                totals[scenario] = totals.get(scenario, 0) + numeric_count
    reasons = sorted({reason for _, reason in counts})
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        if totals.get(scenario, 0) <= 0:
            raise ValueError(f"no known-edge gate reasons for {scenario}")
        for reason in reasons:
            count = counts.get((scenario, reason), 0)
            rows.append(
                {
                    "panel": "B",
                    "scenario": scenario,
                    "scenario_order": SCENARIO_ORDER.index(scenario),
                    "scenario_label": SCENARIO_LABELS[scenario].replace("\n", " "),
                    "reason_code": reason,
                    "count": count,
                    "total_known_edge_score_rows": totals[scenario],
                    "fraction": count / totals[scenario],
                }
            )
    return pd.DataFrame(rows)


def _panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save_figure(
    figure: Figure,
    figures_dir: Path,
    stem: str,
    generated_at: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fixed_datetime = datetime.fromisoformat(generated_at).astimezone(UTC)
    figure.savefig(
        figures_dir / f"{stem}.svg",
        facecolor="white",
        metadata={
            "Creator": "CRYCHIC ligand-gate v3 report generator",
            "Date": generated_at,
            "Title": stem,
            "Description": "Synthetic development diagnostic; noncertifying.",
        },
    )
    figure.savefig(
        figures_dir / f"{stem}.pdf",
        facecolor="white",
        metadata={
            "Title": stem,
            "Author": "CRYCHIC developers",
            "Subject": "Synthetic development diagnostic; noncertifying.",
            "Creator": "CRYCHIC ligand-gate v3 report generator",
            "Producer": "CRYCHIC ligand-gate v3 report generator",
            "CreationDate": fixed_datetime,
            "ModDate": fixed_datetime,
        },
    )
    figure.savefig(
        figures_dir / f"{stem}.png",
        dpi=300,
        facecolor="white",
        metadata={
            "Software": "CRYCHIC ligand-gate v3 report generator",
            "Creation Time": generated_at,
        },
    )
    plt.close(figure)


def _plot_target_only(
    table: pd.DataFrame,
    figures_dir: Path,
    generated_at: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    snapshots = ["pre_gate", "full7", "seed003"]
    positions: npt.NDArray[np.float64] = np.arange(len(snapshots), dtype=float)
    width = 0.34
    styles = (
        ("paired_integrated", -width / 2, BLUE, None, "Paired integrated"),
        ("raw_integrated", width / 2, "white", "///", "Raw integrated"),
    )
    for estimand, offset, color, hatch, label in styles:
        values = [
            float(
                table.loc[
                    (table["snapshot"] == snapshot) & (table["estimand"] == estimand),
                    "any_positive_family_rate",
                ].iloc[0]
            )
            for snapshot in snapshots
        ]
        bars = axis.bar(
            positions + offset,
            values,
            width=width,
            color=color,
            edgecolor=BLUE if estimand == "raw_integrated" else "white",
            hatch=hatch,
            label=label,
            zorder=2,
        )
        subset = table.loc[table["estimand"] == estimand].set_index("snapshot")
        for bar, value, snapshot in zip(bars, values, snapshots, strict=True):
            numerator = int(
                cast(Any, subset.loc[snapshot, "positive_seed_mode_evaluations"])
            )
            denominator = int(
                cast(Any, subset.loc[snapshot, "n_seed_mode_evaluations"])
            )
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(value + 0.035, 0.035),
                f"{numerator}/{denominator}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            if value == 0:
                axis.plot(
                    bar.get_x() + bar.get_width() / 2,
                    0,
                    marker="_",
                    color=BLACK,
                    markersize=8,
                    zorder=3,
                )
    for position, snapshot in zip(positions, snapshots, strict=True):
        n_seeds = int(table.loc[table["snapshot"] == snapshot, "n_seed_runs"].iloc[0])
        axis.text(
            position, 1.045, f"{n_seeds} seed{'s' if n_seeds != 1 else ''}", ha="center"
        )
    axis.set_xticks(positions, [SNAPSHOT_LABELS[item] for item in snapshots])
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Positive seed-mode evaluation fraction")
    axis.set_title("Target-only synthetic negative control")
    axis.legend(loc="center right")
    axis.grid(axis="y", zorder=0)
    _panel_label(axis, "A")
    _save_figure(
        figure,
        figures_dir,
        "figure01_target_only_pre_post",
        generated_at,
    )


def _jitter(record: pd.Series) -> float:
    key = "|".join(
        str(record[column])
        for column in ("snapshot", "seed_id", "known_edge_id", "mode", "score_kind")
    )
    raw = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return ((raw / 0xFFFFFFFF) - 0.5) * 0.18


def _plot_active(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    figures_dir: Path,
    generated_at: str,
) -> None:
    if (detail["active_minus_ligand_only_margin"] <= 0).any():
        raise ValueError("active-margin log plot requires strictly positive values")
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    figure.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.28, wspace=0.22)
    markers = {"full7": ("o", BLUE), "seed003": ("D", ORANGE)}
    for snapshot, (marker, color) in markers.items():
        subset = detail.loc[detail["snapshot"] == snapshot]
        axes[0].scatter(
            subset["view_order"] + subset.apply(_jitter, axis=1),
            subset["active_minus_ligand_only_margin"],
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            label="Full-seven seeds 001-002"
            if snapshot == "full7"
            else "Debug seed 003",
            zorder=3,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(
        range(len(VIEW_ORDER)), [VIEW_LABELS[item] for item in VIEW_ORDER]
    )
    axes[0].set_ylabel("Active - ligand-only paired mean (log scale)")
    axes[0].set_title("Known-edge paired margins")
    axes[0].grid(axis="y", zorder=0)
    _panel_label(axes[0], "A")

    width = 0.32
    for snapshot, offset, color, hatch, label in (
        ("full7", -width / 2, BLUE, None, "Full-seven seeds 001-002"),
        ("seed003", width / 2, "white", "///", "Debug seed 003"),
    ):
        subset = summary.loc[summary["snapshot"] == snapshot].sort_values("view_order")
        bars = axes[1].bar(
            subset["view_order"] + offset,
            subset["recovered_fraction"],
            width=width,
            color=color,
            edgecolor=ORANGE if snapshot == "seed003" else "white",
            hatch=hatch,
            label=label,
            zorder=2,
        )
        for bar, (_, row) in zip(bars, subset.iterrows(), strict=True):
            n_pairs = int(row["n_pairs"])
            n_recovered = round(float(row["recovered_fraction"]) * n_pairs)
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                float(row["recovered_fraction"]) + 0.025,
                f"{n_recovered}/{n_pairs}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    axes[1].set_xticks(
        range(len(VIEW_ORDER)), [VIEW_LABELS[item] for item in VIEW_ORDER]
    )
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("Recovered fraction")
    axes[1].set_title("Active synthetic recovery")
    axes[1].grid(axis="y", zorder=0)
    _panel_label(axes[1], "B")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
    )
    _save_figure(
        figure,
        figures_dir,
        "figure02_active_recovery_margins",
        generated_at,
    )


def _reason_label(reason: str) -> str:
    replacements = {
        "family_not_selected": "Family not\nselected",
        "ligand_contrast_not_supported": "Ligand contrast\nnot supported",
        "receptor_interaction_ineligible": "Receptor interaction\nineligible",
    }
    return replacements.get(reason, reason.replace("_", "\n"))


def _plot_controls(
    specificity: pd.DataFrame,
    reasons: pd.DataFrame,
    figures_dir: Path,
    generated_at: str,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.5, 4.0),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
        constrained_layout=True,
    )
    positions: npt.NDArray[np.float64] = np.arange(len(SCENARIO_ORDER), dtype=float)
    width = 0.32
    for estimand, offset, marker, color, label in (
        ("paired_integrated", -width / 2, "o", BLUE, "Paired integrated"),
        ("raw_integrated", width / 2, "s", VERMILLION, "Raw integrated"),
    ):
        subset = specificity.loc[specificity["estimand"] == estimand].sort_values(
            "scenario_order"
        )
        axes[0].bar(
            positions + offset,
            subset["any_positive_family_rate"],
            width=width,
            color="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=2,
        )
        axes[0].scatter(
            positions + offset,
            subset["any_positive_family_rate"],
            marker=marker,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            label=label,
            zorder=3,
        )
        for x_value, (_, row) in zip(
            positions + offset, subset.iterrows(), strict=True
        ):
            rate = float(row["any_positive_family_rate"])
            numerator = int(row["positive_seed_mode_evaluations"])
            denominator = int(row["n_seed_mode_evaluations"])
            high = rate >= 0.9
            axes[0].text(
                x_value,
                rate - 0.015 if high else rate + 0.015,
                f"{numerator}/{denominator}",
                ha="center",
                va="top" if high else "bottom",
                fontsize=6,
                rotation=90,
            )
    axes[0].set_xticks(
        positions,
        [SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER],
        rotation=35,
        ha="right",
    )
    axes[0].set_ylim(-0.03, 1.02)
    axes[0].set_ylabel("Positive seed-mode evaluation fraction")
    axes[0].set_title("Current full-seven control specificity")
    axes[0].grid(axis="y", zorder=0)
    axes[0].legend(loc="upper right")
    _panel_label(axes[0], "A")

    reason_order = sorted(reasons["reason_code"].unique())
    matrix: npt.NDArray[np.float64] = np.zeros(
        (len(SCENARIO_ORDER), len(reason_order)), dtype=float
    )
    for row_index, scenario in enumerate(SCENARIO_ORDER):
        for column_index, reason in enumerate(reason_order):
            values = reasons.loc[
                (reasons["scenario"] == scenario) & (reasons["reason_code"] == reason),
                "fraction",
            ]
            matrix[row_index, column_index] = (
                float(values.iloc[0]) if len(values) else 0.0
            )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gate_reason_fraction", [VERY_LIGHT_GRAY, GREEN]
    )
    image = axes[1].imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    axes[1].set_xticks(
        range(len(reason_order)), [_reason_label(item) for item in reason_order]
    )
    axes[1].set_yticks(
        range(len(SCENARIO_ORDER)),
        [SCENARIO_LABELS[item].replace("\n", " ") for item in SCENARIO_ORDER],
    )
    axes[1].tick_params(axis="x", rotation=35)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axes[1].text(
                column_index,
                row_index,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else BLACK,
                fontsize=7,
            )
    axes[1].set_title("Known-edge row reason composition")
    colorbar = figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of scored known-edge rows")
    _panel_label(axes[1], "B")
    _save_figure(
        figure,
        figures_dir,
        "figure03_control_specificity_gate_reasons",
        generated_at,
    )


def _save_source(table: pd.DataFrame, source_dir: Path, stem: str) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(source_dir / f"{stem}.csv", index=False, na_rep="")


def _format_float(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(value) for value in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def _report_markdown(
    snapshots: Sequence[Snapshot],
    input_manifest: pd.DataFrame,
    source_audit: pd.DataFrame,
    target: pd.DataFrame,
    active_detail: pd.DataFrame,
    active_summary: pd.DataFrame,
    specificity: pd.DataFrame,
    reasons: pd.DataFrame,
    generated_at: str,
) -> str:
    target_pivot = target.pivot(
        index="snapshot", columns="estimand", values="any_positive_family_rate"
    )
    full_active = active_detail.loc[active_detail["snapshot"] == "full7"]
    seed_active = active_detail.loc[active_detail["snapshot"] == "seed003"]
    specificity_zero = int((specificity["any_positive_family_rate"] == 0).sum())
    audit_summary = (
        source_audit.groupby("snapshot", as_index=False)
        .agg(
            recorded_sources=("relative_path", "size"),
            current_tree_matches=("matches_current_tree", "sum"),
            current_files_missing=(
                "current_file_exists",
                lambda values: int((~values.astype(bool)).sum()),
            ),
        )
        .sort_values("snapshot")
    )
    input_rows = [
        (
            row.snapshot,
            row.role,
            row.profile,
            f"`{cast(str, row.registry_sha256)[:16]}`",
            row.seed_ids,
            row.scenarios,
            f"`{cast(str, row.sha256)[:16]}`",
        )
        for row in input_manifest.itertuples(index=False)
    ]
    active_rows = [
        (
            row.snapshot,
            row.view_label,
            int(cast(Any, row.n_pairs)),
            _format_float(float(cast(Any, row.mean_margin))),
            _format_float(float(cast(Any, row.minimum_margin))),
            f"{float(cast(Any, row.recovered_fraction)):.1%}",
            _format_float(float(cast(Any, row.maximum_coverage_loss))),
        )
        for row in active_summary.sort_values(["snapshot", "view_order"]).itertuples(
            index=False
        )
    ]
    control_rows = []
    for scenario in SCENARIO_ORDER:
        paired_row = specificity.loc[
            (specificity["scenario"] == scenario)
            & (specificity["estimand"] == "paired_integrated")
        ].iloc[0]
        raw_row = specificity.loc[
            (specificity["scenario"] == scenario)
            & (specificity["estimand"] == "raw_integrated")
        ].iloc[0]
        paired = float(cast(Any, paired_row["any_positive_family_rate"]))
        raw = float(cast(Any, raw_row["any_positive_family_rate"]))
        paired_count = (
            f"{int(cast(Any, paired_row['positive_seed_mode_evaluations']))}/"
            f"{int(cast(Any, paired_row['n_seed_mode_evaluations']))} ({paired:.1%})"
        )
        raw_count = (
            f"{int(cast(Any, raw_row['positive_seed_mode_evaluations']))}/"
            f"{int(cast(Any, raw_row['n_seed_mode_evaluations']))} ({raw:.1%})"
        )
        dominant = reasons.loc[
            (reasons["scenario"] == scenario) & (reasons["fraction"] > 0),
            ["reason_code", "fraction"],
        ]
        reason_text = "; ".join(
            f"`{row.reason_code}` ({float(cast(Any, row.fraction)):.0%})"
            for row in dominant.itertuples(index=False)
        )
        control_rows.append((scenario, paired_count, raw_count, reason_text))
    audit_rows = [
        (
            row.snapshot,
            int(cast(Any, row.recorded_sources)),
            int(cast(Any, row.current_tree_matches)),
            int(cast(Any, row.current_files_missing)),
        )
        for row in audit_summary.itertuples(index=False)
    ]
    target_index = target.set_index(["snapshot", "estimand"])

    def target_count(snapshot: str, estimand: str) -> str:
        row = cast(Any, target_index.loc)[(snapshot, estimand)]
        return (
            f"{int(cast(Any, row['positive_seed_mode_evaluations']))}/"
            f"{int(cast(Any, row['n_seed_mode_evaluations']))}"
        )

    by_key = {snapshot.key: snapshot for snapshot in snapshots}
    reproduction_command = "\n".join(
        (
            "OUTPUT_DIR=reports/algorithm_ligand_gate_v3",
            "uv run --extra plotting python -m \\",
            "  benchmarks.report.generate_ligand_gate_v3_report \\",
            f"  --pre-summary {shlex.quote(str(by_key['pre_gate'].path))} \\",
            f"  --seed003-summary {shlex.quote(str(by_key['seed003'].path))} \\",
            f"  --full7-summary {shlex.quote(str(by_key['full7'].path))} \\",
            '  --output "$OUTPUT_DIR" \\',
            f"  --generated-at {shlex.quote(generated_at)}",
        )
    )
    return f"""# CRYCHIC ligand-gate v3 synthetic diagnostic report

> **Evidence boundary.** This report is a development-only presentation of frozen synthetic artifacts. It is noncertifying, contains no real biological validation, does not compare CRYCHIC with external methods, does not establish method superiority, and cannot authorize a default-method switch.

Generated at `{generated_at}`. All plotted values were reconstructed from the three checksummed JSON summaries listed below; no score was recomputed by this report generator.

## Executive summary

The frozen historical pre-gate two-seed artifact produced a positive integrated family in every `target_only` seed-mode evaluation: paired {target_count("pre_gate", "paired_integrated")} ({target_pivot.loc["pre_gate", "paired_integrated"]:.1%}), raw {target_count("pre_gate", "raw_integrated")} ({target_pivot.loc["pre_gate", "raw_integrated"]:.1%}). Under ligand-gate v3, the corresponding frozen two-seed full-seven artifact results were paired {target_count("full7", "paired_integrated")} ({target_pivot.loc["full7", "paired_integrated"]:.1%}) and raw {target_count("full7", "raw_integrated")} ({target_pivot.loc["full7", "raw_integrated"]:.1%}); the separate frozen seed003 debug artifact was paired {target_count("seed003", "paired_integrated")} and raw {target_count("seed003", "raw_integrated")}. This is a targeted regression result, not an uncertainty-adjusted performance estimate.

In the active synthetic scenario, the two-seed full-seven artifact recovered {int(full_active["active_recovered"].sum())}/{len(full_active)} preregistered edge-view pairs with {int((full_active["active_minus_ligand_only_margin"] > 0).sum())}/{len(full_active)} positive paired margins and maximum coverage loss {_format_float(float(full_active["coverage_loss"].max()))}. Seed003 recovered {int(seed_active["active_recovered"].sum())}/{len(seed_active)} pairs with {int((seed_active["active_minus_ligand_only_margin"] > 0).sum())}/{len(seed_active)} positive margins. These are implanted synthetic edges; “recovery” is not discovery of known biology.

Across the current full-seven summary, {specificity_zero}/{len(specificity)} paired/raw control rates were exactly zero. `target_only` known-edge rows were materialized as `ligand_contrast_not_supported`; receptor-knockout rows were `receptor_interaction_ineligible`. Structural zeros remain observed rows in the registered coverage denominator.

## Frozen inputs and provenance

{_markdown_table(("Snapshot", "Role", "Profile", "Registry SHA prefix", "Seeds", "Scenarios", "Summary SHA prefix"), input_rows)}

Full hashes, byte counts, scopes, and paths are in [input_manifest.tsv](input_manifest.tsv). Embedded source bindings were also compared with the current checkout:

{_markdown_table(("Snapshot", "Recorded paths", "Current hash matches", "Missing now"), audit_rows)}

These are frozen-artifact-to-current-checkout comparisons. A mismatch can arise whenever the current checkout differs from the source bound to any frozen artifact; the report does not assume that every mismatch is historical. Frozen summary hashes remain unchanged, but a mismatched artifact is not presented as a rerun of the current tree. Complete path-level results are in [source_binding_audit.csv](source_data/source_binding_audit.csv).

## Metrics and plots

### Target-only regression

![Target-only pre/post false-positive diagnostic](figures/figure01_target_only_pre_post.png)

**Figure 1.** Fractions and exact numerators/denominators of seed-mode evaluations with any positive integrated family in the target-only synthetic negative control. Paired integrated is the registered subject-level stim-minus-control estimand; raw integrated is retained as a separate diagnostic. The pre-gate and current full-seven snapshots each contain four seed-mode evaluations (two seeds by two modes); seed003 is explicitly debug-only and contains two. Exact source data: [figure01_target_only_pre_post.csv](source_data/figure01_target_only_pre_post.csv).

### Active recovery and paired margins

![Active known-edge recovery and margins](figures/figure02_active_recovery_margins.png)

**Figure 2.** Panel A shows every strictly positive active-minus-ligand-only paired margin on a logarithmic scale, separated by state/ecosystem and member/sender views. Panel B reports the preregistered known-edge recovery fraction. Full-seven includes seeds 001-002; seed003 remains a supplemental debug observation. Exact source data: [figure02_active_recovery_margins.csv](source_data/figure02_active_recovery_margins.csv).

{_markdown_table(("Snapshot", "View", "Pairs", "Mean margin", "Minimum margin", "Recovered", "Max coverage loss"), active_rows)}

### Control specificity and gate reasons

![Full-seven control specificity and gate reasons](figures/figure03_control_specificity_gate_reasons.png)

**Figure 3.** Panel A shows paired and raw any-positive integrated-family fractions and exact counts across four seed-mode evaluations for all six negative controls in the current two-seed full-seven artifact. Panel B decomposes the persisted reason codes across known-edge score rows; it does not infer a biological mechanism from those codes. Exact source data: [figure03_control_specificity_gate_reasons.csv](source_data/figure03_control_specificity_gate_reasons.csv).

{_markdown_table(("Scenario", "Paired positive evaluations", "Raw positive evaluations", "Known-edge reason composition"), control_rows)}

## Interpretation

The bounded conclusion is that, in these registered quick-profile simulations, ligand-gate v3 closes the previously observed target-only pathway to a positive family score while preserving positive active-versus-ligand-only margins for the implanted edge views. The persisted reason codes agree with the intended precedence: ligand support gates target-only, while receptor eligibility gates receptor knockout.

This evidence is useful for algorithm regression control. It is not sufficient for biological validation, calibration, real-data sensitivity, cross-method benchmarking, or publication claims about superiority. In particular, the campaign has two primary seeds plus one excluded debug seed, uses a small synthetic gene universe, and the frozen pre-gate and ligand-gate artifacts are not a randomized intervention. No confidence interval or hypothesis-test p-value is reported because these seed-mode evaluations are not independent biological replicates.

## Reproduction contract

Run from the repository root. The three input paths and `generated-at` value below are fully fixed; change only `OUTPUT_DIR`. With identical software, fonts, platform, and checkout, two output directories have byte-identical generated files. PDF/SVG dates and SVG object IDs are fixed by the generator.

```bash
{reproduction_command}
```

- `REPORT.md`: this report.
- `figures/*.png`: 300 dpi raster figures; matching PDF and SVG files retain vector text.
- `source_data/figure*.csv`: exact plotting tables, with panel labels.
- `input_manifest.tsv`: consumed input identity and SHA256.
- `source_data/source_binding_audit.csv`: recorded source hashes versus the current checkout.
- `metrics_summary.json`: compact machine-readable values used in the narrative.
- `report_manifest.json`: claim boundary, environment, inputs, and output contract.
- `artifact_manifest.tsv`: generated-file sizes and SHA256 values (excluding itself).
"""


def _metrics_summary(
    target: pd.DataFrame,
    active_detail: pd.DataFrame,
    active_summary: pd.DataFrame,
    specificity: pd.DataFrame,
    reasons: pd.DataFrame,
) -> dict[str, Any]:
    target_records = json.loads(target.to_json(orient="records"))
    active_macro = json.loads(active_summary.to_json(orient="records"))
    specificity_records = json.loads(specificity.to_json(orient="records"))
    reason_records = json.loads(reasons.to_json(orient="records"))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evidence_class": "synthetic_development_noncertifying",
        "claims": dict(CLAIMS),
        "target_only_pre_post": target_records,
        "active_macro": active_macro,
        "active_pair_count": len(active_detail),
        "control_specificity": specificity_records,
        "known_edge_gate_reason_composition": reason_records,
    }


def _artifact_manifest(output_dir: Path, roles: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for relative_path, role in sorted(roles.items()):
        path = output_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": relative_path,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def generate_report(
    *,
    pre_summary: Path,
    seed003_summary: Path,
    full7_summary: Path,
    output_dir: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Generate figures, source data, manifests, and Markdown report."""

    repo_root = Path(__file__).resolve().parents[2]
    snapshots = (
        _load_snapshot(
            "pre_gate", "immutable historical pre-gate failure", pre_summary
        ),
        _load_snapshot(
            "full7", "immutable ligand-gate v3 full-seven campaign", full7_summary
        ),
        _load_snapshot("seed003", "ligand-gate v3 seed003 debug", seed003_summary),
    )
    _validate_snapshot_set(snapshots)
    timestamp = _generated_at(generated_at)
    output = output_dir.expanduser().resolve()
    figures_dir = output / "figures"
    source_dir = output / "source_data"
    output.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    style_path = repo_root / "benchmarks/report/publication.mplstyle"
    if not style_path.is_file():
        raise FileNotFoundError(style_path)
    plt.style.use(style_path)
    mpl.rcParams["svg.hashsalt"] = SVG_HASHSALT

    input_manifest = _input_manifest(snapshots, repo_root)
    source_audit = _source_binding_audit(snapshots, repo_root)
    target = _target_only_data(snapshots)
    active_detail, active_summary = _active_data(snapshots)
    specificity = _control_specificity_data(snapshots[1])
    reasons = _gate_reason_data(snapshots[1])

    input_manifest.to_csv(output / "input_manifest.tsv", sep="\t", index=False)
    source_audit.to_csv(source_dir / "source_binding_audit.csv", index=False)
    _save_source(target, source_dir, "figure01_target_only_pre_post")
    figure02_source = pd.concat((active_detail, active_summary), ignore_index=True)
    _save_source(figure02_source, source_dir, "figure02_active_recovery_margins")
    figure03_source = pd.concat((specificity, reasons), ignore_index=True)
    _save_source(
        figure03_source, source_dir, "figure03_control_specificity_gate_reasons"
    )

    _plot_target_only(target, figures_dir, timestamp)
    _plot_active(active_detail, active_summary, figures_dir, timestamp)
    _plot_controls(specificity, reasons, figures_dir, timestamp)

    metrics = _metrics_summary(
        target, active_detail, active_summary, specificity, reasons
    )
    (output / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _report_markdown(
        snapshots,
        input_manifest,
        source_audit,
        target,
        active_detail,
        active_summary,
        specificity,
        reasons,
        timestamp,
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    source_match_summary = {
        snapshot: {
            "recorded": len(group),
            "matches_current_tree": int(group["matches_current_tree"].sum()),
            "missing_current_file": int((~group["current_file_exists"]).sum()),
        }
        for snapshot, group in source_audit.groupby("snapshot")
    }
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": timestamp,
        "evidence_class": "synthetic_development_noncertifying",
        "claims": dict(CLAIMS),
        "generator": {
            "path": "benchmarks/report/generate_ligand_gate_v3_report.py",
            "sha256": _sha256(Path(__file__)),
            "svg_hashsalt": SVG_HASHSALT,
            "figure_metadata_timestamp": timestamp,
        },
        "inputs": json.loads(input_manifest.to_json(orient="records")),
        "source_binding_audit": source_match_summary,
        "environment": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "matplotlib": _package_version("matplotlib"),
            "platform": platform.platform(),
            "git_revision": _git_revision(repo_root),
            "git_state": _git_state(repo_root),
        },
        "figures": [
            {
                "id": stem,
                "source_data": f"source_data/{stem}.csv",
                "formats": [
                    f"figures/{stem}.{suffix}" for suffix in ("png", "pdf", "svg")
                ],
            }
            for stem in (
                "figure01_target_only_pre_post",
                "figure02_active_recovery_margins",
                "figure03_control_specificity_gate_reasons",
            )
        ],
    }
    manifest_path = output / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    roles: dict[str, str] = {
        "REPORT.md": "Markdown report",
        "input_manifest.tsv": "input checksum manifest",
        "metrics_summary.json": "machine-readable metric summary",
        "report_manifest.json": "report provenance and claim contract",
        "source_data/source_binding_audit.csv": "source binding audit",
    }
    for stem in (
        "figure01_target_only_pre_post",
        "figure02_active_recovery_margins",
        "figure03_control_specificity_gate_reasons",
    ):
        roles[f"source_data/{stem}.csv"] = f"figure source data: {stem}"
        for suffix in ("png", "pdf", "svg"):
            roles[f"figures/{stem}.{suffix}"] = f"figure {stem} ({suffix})"
    artifacts = _artifact_manifest(output, roles)
    artifacts.to_csv(output / "artifact_manifest.tsv", sep="\t", index=False)

    return {
        "output_dir": str(output),
        "report": str(output / "REPORT.md"),
        "artifact_count": int(len(artifacts) + 1),
        "input_sha256": {
            row.snapshot: row.sha256 for row in input_manifest.itertuples(index=False)
        },
        "claims": manifest["claims"],
    }


def _parser(repo_root: Path) -> argparse.ArgumentParser:
    workspace = repo_root.parent
    artifact_root = workspace / "benchmark_work/algorithm_smoke"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-summary",
        type=Path,
        default=artifact_root
        / "public_family_common_g15_campaign_v1_f48e4d0_pre_ligand_gate_failure_summary.json",
        help="immutable historical pre-ligand-gate failure summary",
    )
    parser.add_argument(
        "--seed003-summary",
        type=Path,
        default=artifact_root
        / "public_family_common_g15_seed003_holm_v3_final_summary.json",
        help="final seed003 ligand-gate v3 debug summary",
    )
    parser.add_argument(
        "--full7-summary",
        type=Path,
        default=artifact_root
        / "public_family_common_g15_holm_v3_seen001_002_full7_summary.json",
        help="immutable two-seed ligand-gate v3 full-seven summary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "reports/algorithm_ligand_gate_v3",
        help="report output directory",
    )
    parser.add_argument(
        "--generated-at",
        help="fixed ISO-8601 timestamp for deterministic report metadata",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    args = _parser(repo_root).parse_args(argv)
    result = generate_report(
        pre_summary=args.pre_summary,
        seed003_summary=args.seed003_summary,
        full7_summary=args.full7_summary,
        output_dir=args.output,
        generated_at=args.generated_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
