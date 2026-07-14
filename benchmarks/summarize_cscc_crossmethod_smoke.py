"""Summarize the bounded Ji cSCC cross-method smoke benchmark.

Track A compares paired tumor-normal rank effects on one frozen LR universe.
Track B retains NicheNet's source-agnostic ligand-target-program proxy and is
never mixed into sender-resolved LR comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmarks.adapters.common import git_metadata, sha256_file, write_json
from benchmarks.metrics.multicondition import (
    external_long_to_score_table,
    paired_edge_effects,
)
from benchmarks.metrics.track_b import (
    PROXY_SEMANTICS,
    prepare_track_b_long,
    track_b_edge_effects,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SCHEMA_VERSION = "crychic-cscc-crossmethod-summary-v1"
CRYCHIC_SCHEMA_VERSION = "crychic-cscc-paired-gate-smoke-v2"
DATASET_ID = "GSE144236_Ji_cSCC"
REFERENCE_CONTEXT = "Normal"
TARGET_CONTEXT = "Tumor"
CONTRAST = "tumor_vs_normal"
MIN_PAIRED_SUBJECTS = 3
DEFAULT_WORKSPACE_ROOT = Path("/media/subunit/bioinfo/crychic_dev")
DEFAULT_CRYCHIC_ARTIFACT = Path("benchmark_work/cscc_paired_gate_smoke.json")
DEFAULT_METHODS_DIR = Path("benchmark_work/cscc_crossmethod_smoke/methods")
DEFAULT_OUTPUT_DIR = Path("benchmark_work/cscc_crossmethod_smoke/summary")
TRACK_A_RUNS = (
    ("cellchat", "CellChat"),
    ("cellphonedb", "CellPhoneDB"),
    ("liana", "LIANA"),
)
TRACK_A_MAIN_ORDER = (
    "cellchat",
    "cellphonedb",
    "liana_rank_aggregate",
    "crychic_state",
)
DISPLAY_NAMES = {
    "cellchat": "CellChat",
    "cellphonedb": "CellPhoneDB",
    "liana_rank_aggregate": "LIANA",
    "crychic_state": "CRYCHIC state*",
    "crychic_ecosystem": "CRYCHIC ecosystem*",
    "nichenet_prior_activity": "NicheNet prior proxy",
}
FOCUS_EDGES = (
    ("CD1C", "CD1C", "CCL3", "CCR5"),
    ("Epithelial", "CD1C", "CCL3", "CCR5"),
    ("CD1C", "Epithelial", "HBEGF", "EGFR"),
    ("Epithelial", "Epithelial", "HBEGF", "EGFR"),
    ("Epithelial", "Epithelial", "TGFA", "EGFR"),
)
EDGE_KEYS = ("sender", "receiver", "interaction_id", "ligand", "receptor")


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return cast(Sequence[object], value)


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, field=str(path))


def _constant(table: pd.DataFrame, column: str) -> str:
    values = table[column].astype("string").dropna().astype(str).drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{column} must be constant, observed {values.tolist()}")
    return str(values.iloc[0])


def _finite_float(value: object) -> float | None:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _status_counts(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in values.astype(str).value_counts().sort_index().items()
    }


def _load_adapter_run(run_dir: Path) -> tuple[pd.DataFrame, Mapping[str, object]]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("failure") is not None:
        raise ValueError(f"adapter run is not complete: {run_dir}")
    sample_failures = manifest.get("sample_failures")
    if isinstance(sample_failures, Mapping) and sample_failures:
        raise ValueError(f"adapter run contains sample failures: {run_dir}")
    output = _mapping(manifest.get("output"), field="manifest.output")
    table_path = run_dir / str(output.get("table", ""))
    if not table_path.is_file():
        raise FileNotFoundError(f"adapter output is missing: {table_path}")
    expected_sha = str(output.get("sha256", ""))
    observed_sha = sha256_file(table_path)
    if expected_sha != observed_sha:
        raise ValueError(f"adapter output checksum mismatch: {table_path}")
    table = pd.read_parquet(table_path)
    if int(cast(Any, output.get("rows", -1))) != len(table):
        raise ValueError(f"adapter output row count mismatch: {table_path}")
    if _constant(table, "dataset_id") != DATASET_ID:
        raise ValueError(f"unexpected adapter dataset: {run_dir}")
    return table, manifest


def _standardize_external_effects(
    table: pd.DataFrame,
) -> pd.DataFrame:
    mapped = external_long_to_score_table(
        table,
        context_key="condition",
        contrast=CONTRAST,
        dataset=DATASET_ID,
    )
    effects = paired_edge_effects(
        mapped,
        reference=REFERENCE_CONTEXT,
        target=TARGET_CONTEXT,
        min_pairs=MIN_PAIRED_SUBJECTS,
        contrast=CONTRAST,
        validated=True,
    )
    method = _constant(effects, "method")
    result = effects.loc[
        :,
        [
            *EDGE_KEYS,
            "effect",
            "median_effect",
            "direction_consistency",
            "direction_comparable_pairs",
            "n_pairs",
            "status",
            "reason_code",
        ],
    ].copy()
    result.insert(0, "method", method)
    result.insert(1, "method_label", DISPLAY_NAMES.get(method, method))
    result.insert(2, "mode", "native")
    result["analysis_track"] = "lr_stlr"
    result["effect_semantics"] = "tumor_minus_normal_paired_comparison_strength"
    result["evidence_scope"] = "exploratory_unadjusted_paired_rank_effects"
    result["is_official"] = False
    result["is_oof_certified"] = False
    return cast(pd.DataFrame, result)


def _external_coverage(
    table: pd.DataFrame,
    manifest: Mapping[str, object],
    effects: pd.DataFrame,
) -> dict[str, object]:
    method = _constant(table, "method_id")
    native_observed = int(table["status"].astype(str).eq("ok").sum())
    empty_results = manifest.get("sample_empty_results")
    empty_count = len(empty_results) if isinstance(empty_results, Sequence) else 0
    return {
        "method": method,
        "method_label": DISPLAY_NAMES.get(method, method),
        "analysis_track": "lr_stlr",
        "n_input_rows": len(table),
        "n_native_observed_rows": native_observed,
        "native_observed_fraction": native_observed / len(table),
        "input_status_counts": _status_counts(table["status"]),
        "n_effect_rows": len(effects),
        "n_estimable_effect_rows": int(effects["status"].eq("exploratory").sum()),
        "n_samples": int(table["sample_id"].nunique()),
        "n_subjects": int(table["subject_id"].nunique()),
        "empty_sample_results": empty_count,
        "elapsed_seconds": _finite_float(manifest.get("elapsed_seconds")),
    }


def _crychic_effects_and_coverage(
    artifact: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if artifact.get("schema_version") != CRYCHIC_SCHEMA_VERSION:
        raise ValueError("unexpected CRYCHIC cSCC artifact schema")
    if (
        artifact.get("scope")
        != "bounded_real_data_algorithm_smoke_not_biological_validation"
    ):
        raise ValueError("unexpected CRYCHIC cSCC artifact scope")
    summary = _mapping(
        artifact.get("diagnostic_score_summary"),
        field="diagnostic_score_summary",
    )
    if summary.get("scope") != (
        "exploratory_unadjusted_noncertified_paired_rank_effects"
    ):
        raise ValueError("CRYCHIC diagnostic score scope is not noncertified")
    if summary.get("raw_sample_and_subject_rows_exported") is not False:
        raise ValueError("CRYCHIC summary must remain deidentified")
    if list(
        _sequence(
            summary.get("inferential_fields_available"),
            field="inferential_fields_available",
        )
    ):
        raise ValueError(
            "CRYCHIC diagnostic summary must not expose inferential fields"
        )

    mode_payloads = {
        str(_mapping(item, field="diagnostic mode").get("mode")): _mapping(
            item, field="diagnostic mode"
        )
        for item in _sequence(summary.get("modes"), field="diagnostic modes")
    }
    if set(mode_payloads) != {"state", "ecosystem"}:
        raise ValueError("CRYCHIC diagnostic summary must contain both modes")
    records = [
        dict(_mapping(item, field="paired effect"))
        for item in _sequence(summary.get("paired_effects"), field="paired_effects")
    ]
    effects = pd.DataFrame.from_records(records)
    required = {*EDGE_KEYS, "mode", "effect", "status", "n_pairs"}
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"CRYCHIC paired effects lack columns: {sorted(missing)}")
    if effects.duplicated(["mode", *EDGE_KEYS]).any():
        raise ValueError("CRYCHIC paired effects contain duplicate mode-edge rows")
    if set(effects["mode"].astype(str)) != {"state", "ecosystem"}:
        raise ValueError("CRYCHIC paired effect modes disagree with summary")
    if not set(effects["status"].astype(str)).issubset(
        {"exploratory", "not_estimable"}
    ):
        raise ValueError("CRYCHIC diagnostic effects contain an official status")

    effects["method"] = "crychic_" + effects["mode"].astype(str)
    effects["method_label"] = effects["method"].map(DISPLAY_NAMES)
    effects["analysis_track"] = "lr_stlr"
    effects["effect_semantics"] = "tumor_minus_normal_paired_comparison_strength"
    effects["evidence_scope"] = str(summary["scope"])
    effects["is_official"] = False
    effects["is_oof_certified"] = False
    for column in ("median_effect", "direction_consistency", "reason_code"):
        if column not in effects:
            effects[column] = np.nan
    if "direction_comparable_pairs" not in effects:
        effects["direction_comparable_pairs"] = 0
    effects = effects.loc[
        :,
        [
            "method",
            "method_label",
            "mode",
            *EDGE_KEYS,
            "effect",
            "median_effect",
            "direction_consistency",
            "direction_comparable_pairs",
            "n_pairs",
            "status",
            "reason_code",
            "analysis_track",
            "effect_semantics",
            "evidence_scope",
            "is_official",
            "is_oof_certified",
        ],
    ].copy()

    runtime = _mapping(artifact.get("runtime"), field="runtime")
    coverage_rows: list[dict[str, object]] = []
    for mode, payload in mode_payloads.items():
        counts = _mapping(
            payload.get("input_status_counts"), field="input_status_counts"
        )
        n_samples = int(cast(Any, payload.get("n_samples")))
        universe_size = int(cast(Any, payload.get("universe_size")))
        total = sum(int(cast(Any, value)) for value in counts.values())
        if total != n_samples * universe_size:
            raise ValueError(f"CRYCHIC {mode} input status counts are incomplete")
        selected = effects.loc[effects["mode"].eq(mode)]
        if len(selected) != universe_size:
            raise ValueError(f"CRYCHIC {mode} effect universe is incomplete")
        observed = int(cast(Any, counts.get("ok", 0)))
        coverage_rows.append(
            {
                "method": f"crychic_{mode}",
                "method_label": DISPLAY_NAMES[f"crychic_{mode}"],
                "analysis_track": "lr_stlr",
                "n_input_rows": total,
                "n_native_observed_rows": observed,
                "native_observed_fraction": observed / total,
                "input_status_counts": json.dumps(
                    {str(key): int(cast(Any, value)) for key, value in counts.items()},
                    sort_keys=True,
                ),
                "n_effect_rows": len(selected),
                "n_estimable_effect_rows": int(
                    selected["status"].eq("exploratory").sum()
                ),
                "n_samples": n_samples,
                "n_subjects": int(cast(Any, payload.get("n_subjects"))),
                "empty_sample_results": 0,
                "elapsed_seconds": _finite_float(runtime.get("total_elapsed_seconds")),
            }
        )
    return effects, pd.DataFrame.from_records(coverage_rows)


def pairwise_effect_concordance(
    effects: pd.DataFrame,
    *,
    methods: Sequence[str] = TRACK_A_MAIN_ORDER,
) -> pd.DataFrame:
    """Return full-universe Spearman concordance without imputing missing effects."""

    keys = list(EDGE_KEYS)
    subset = effects.loc[effects["method"].isin(methods), ["method", *keys, "effect"]]
    if subset.duplicated(["method", *keys]).any():
        raise ValueError("Track-A effects contain duplicate method-edge rows")
    wide = subset.pivot(index=keys, columns="method", values="effect")
    rows: list[dict[str, object]] = []
    for left in methods:
        for right in methods:
            if left not in wide or right not in wide:
                rows.append(
                    {
                        "method_left": left,
                        "method_right": right,
                        "n_shared_edges": 0,
                        "spearman": np.nan,
                        "status": "not_estimable",
                        "reason_code": "method_missing",
                    }
                )
                continue
            shared = pd.concat(
                [wide[left].rename("left"), wide[right].rename("right")], axis=1
            ).dropna()
            constant = shared["left"].nunique() < 2 or shared["right"].nunique() < 2
            if len(shared) < 3 or constant:
                rho = np.nan
                status = "not_estimable"
                reason = "fewer_than_three_shared_or_nonconstant_edges"
            else:
                rho = float(spearmanr(shared["left"], shared["right"]).statistic)
                status = "exploratory"
                reason = None
            rows.append(
                {
                    "method_left": left,
                    "method_right": right,
                    "n_shared_edges": len(shared),
                    "spearman": rho,
                    "status": status,
                    "reason_code": reason,
                }
            )
    return pd.DataFrame.from_records(rows)


def _track_b_effects(
    table: pd.DataFrame,
) -> pd.DataFrame:
    prepared = prepare_track_b_long(table, context_key="condition")
    effects = track_b_edge_effects(
        prepared,
        reference=REFERENCE_CONTEXT,
        target=TARGET_CONTEXT,
        design="paired",
        min_support=MIN_PAIRED_SUBJECTS,
    )
    if set(effects["proxy_semantics"].astype(str)) != {PROXY_SEMANTICS}:
        raise ValueError("Track-B proxy semantics changed unexpectedly")
    return cast(pd.DataFrame, effects)


def _figure_edge_selection(effects: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    main = effects.loc[effects["method"].isin(TRACK_A_MAIN_ORDER)].copy()
    main["edge_tuple"] = list(
        zip(
            main["sender"],
            main["receiver"],
            main["ligand"],
            main["receptor"],
            strict=True,
        )
    )
    focus = [edge for edge in FOCUS_EDGES if edge in set(main["edge_tuple"])]
    ranked: list[tuple[str, str, str, str]] = []
    for method in TRACK_A_MAIN_ORDER:
        selected = main.loc[
            main["method"].eq(method) & main["effect"].notna()
        ].nlargest(2, "effect")
        ranked.extend(
            cast(list[tuple[str, str, str, str]], selected["edge_tuple"].tolist())
        )
    candidates = list(dict.fromkeys([*focus, *ranked]))
    maximum = (
        main.assign(abs_effect=main["effect"].abs())
        .groupby("edge_tuple", observed=True)["abs_effect"]
        .max()
        .to_dict()
    )
    ordered = [*focus]
    ordered.extend(
        sorted(
            (edge for edge in candidates if edge not in focus),
            key=lambda edge: (-float(maximum.get(edge, -math.inf)), edge),
        )
    )
    ordered = ordered[:limit]
    records = []
    for order, (sender, receiver, ligand, receptor) in enumerate(ordered):
        records.append(
            {
                "figure_order": order,
                "sender": sender,
                "receiver": receiver,
                "ligand": ligand,
                "receptor": receptor,
                "focus_edge": (sender, receiver, ligand, receptor) in FOCUS_EDGES,
                "label": f"{sender} -> {receiver} | {ligand}-{receptor}",
            }
        )
    return pd.DataFrame.from_records(records)


def _plot_summary(
    *,
    track_a: pd.DataFrame,
    coverage: pd.DataFrame,
    concordance: pd.DataFrame,
    track_b: pd.DataFrame,
    selection: pd.DataFrame,
    style_path: Path,
    output_stem: Path,
) -> None:
    if not style_path.is_file():
        raise FileNotFoundError(f"publication style is missing: {style_path}")
    with plt.style.context(style_path):
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), layout="constrained")
        ax_a, ax_b, ax_c, ax_d = axes.ravel()
        colors = {
            "cellchat": "#3B6FB6",
            "cellphonedb": "#4E9F3D",
            "liana_rank_aggregate": "#D9822B",
            "crychic_state": "#7A5195",
        }

        main_coverage = coverage.set_index("method").reindex(TRACK_A_MAIN_ORDER)
        values = main_coverage["native_observed_fraction"].to_numpy(float) * 100.0
        positions = np.arange(len(TRACK_A_MAIN_ORDER))
        ax_a.barh(
            positions,
            values,
            color=[colors[method] for method in TRACK_A_MAIN_ORDER],
            height=0.62,
        )
        ax_a.set_yticks(
            positions,
            [DISPLAY_NAMES[method] for method in TRACK_A_MAIN_ORDER],
        )
        ax_a.invert_yaxis()
        ax_a.set_xlim(0, 100)
        ax_a.set_xlabel("Native result density (%; not accuracy)")
        ax_a.set_title("A  Sample-edge result density", loc="left")
        for position, value in zip(positions, values, strict=True):
            ax_a.text(min(value + 1.5, 96), position, f"{value:.1f}", va="center")

        selected_keys = selection.loc[:, ["sender", "receiver", "ligand", "receptor"]]
        selected = track_a.merge(
            selected_keys.assign(figure_order=np.arange(len(selected_keys))),
            on=["sender", "receiver", "ligand", "receptor"],
            how="inner",
        )
        heatmap = selected.pivot_table(
            index="figure_order", columns="method", values="effect", aggfunc="first"
        ).reindex(index=selection["figure_order"], columns=TRACK_A_MAIN_ORDER)
        matrix = heatmap.to_numpy(float)
        finite = np.abs(matrix[np.isfinite(matrix)])
        scale = max(float(np.quantile(finite, 0.95)) if len(finite) else 0.0, 0.05)
        cmap = matplotlib.colormaps["RdBu_r"].copy()
        cmap.set_bad("#E5E7EB")
        image = ax_b.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=-scale,
            vmax=scale,
        )
        ax_b.set_xticks(
            np.arange(len(TRACK_A_MAIN_ORDER)),
            [DISPLAY_NAMES[method] for method in TRACK_A_MAIN_ORDER],
            rotation=25,
            ha="right",
        )
        ax_b.set_yticks(
            np.arange(len(selection)), selection["label"].astype(str).tolist()
        )
        ax_b.set_title("B  Paired tumor-normal LR effects", loc="left")
        figure.colorbar(image, ax=ax_b, label="Rank-strength effect", shrink=0.82)

        correlation = concordance.pivot(
            index="method_left", columns="method_right", values="spearman"
        ).reindex(index=TRACK_A_MAIN_ORDER, columns=TRACK_A_MAIN_ORDER)
        corr_matrix = correlation.to_numpy(float)
        corr_cmap = matplotlib.colormaps["coolwarm"].copy()
        corr_cmap.set_bad("#E5E7EB")
        ax_c.imshow(
            corr_matrix,
            vmin=-1,
            vmax=1,
            cmap=corr_cmap,
            aspect="equal",
        )
        labels = [DISPLAY_NAMES[method] for method in TRACK_A_MAIN_ORDER]
        ax_c.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax_c.set_yticks(np.arange(len(labels)), labels)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = corr_matrix[row, column]
                ax_c.text(
                    column,
                    row,
                    "NE" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white"
                    if np.isfinite(value) and abs(value) > 0.55
                    else "black",
                )
        ax_c.set_title("C  Full-universe effect concordance", loc="left")

        estimable_b = track_b.loc[
            track_b["status"].eq("exploratory") & track_b["effect"].gt(0)
        ].copy()
        top_b = (
            estimable_b.sort_values(
                ["receiver", "effect", "ligand"],
                ascending=[True, False, True],
                kind="stable",
            )
            .groupby("receiver", sort=True, observed=True)
            .head(3)
            .sort_values("effect", kind="stable")
        )
        if top_b.empty:
            ax_d.text(0.5, 0.5, "No positive Track-B effects", ha="center", va="center")
            ax_d.set_axis_off()
        else:
            labels_b = [
                f"{row.ligand} -> {row.receiver}"
                for row in top_b.itertuples(index=False)
            ]
            receiver_names = sorted(top_b["receiver"].astype(str).unique())
            receiver_colors = dict(
                zip(receiver_names, ("#B23A48", "#2A9D8F", "#6C757D"), strict=False)
            )
            ax_d.barh(
                np.arange(len(top_b)),
                top_b["effect"].to_numpy(float),
                color=[receiver_colors[str(value)] for value in top_b["receiver"]],
                height=0.62,
            )
            ax_d.set_yticks(np.arange(len(top_b)), labels_b)
            ax_d.set_xlabel("Paired receiver-program rank effect")
        ax_d.set_title("D  NicheNet frozen-prior proxy (Track B)", loc="left")

        figure.suptitle(
            "Ji cSCC paired tumor-normal benchmark | 8 subjects",
            fontsize=11,
            fontweight="bold",
        )
        figure.text(
            0.01,
            0.003,
            "* CRYCHIC scores are exploratory and non-certified in this smoke run. "
            "Panel A is output sparsity, not accuracy. Track B is source-agnostic "
            "and is not an LR-edge or sender comparison.",
            fontsize=7,
        )
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300)
        plt.close(figure)


def summarize_crossmethod_smoke(
    *,
    crychic_artifact_path: Path,
    methods_dir: Path,
    output_dir: Path,
    repo_root: Path,
    overwrite: bool = False,
) -> Mapping[str, object]:
    """Validate, summarize, and plot one frozen cSCC smoke benchmark bundle."""

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        crychic_artifact = _read_json(crychic_artifact_path)
        crychic_effects, crychic_coverage = _crychic_effects_and_coverage(
            crychic_artifact
        )
        effect_frames = [crychic_effects]
        coverage_records = cast(
            list[dict[str, object]], crychic_coverage.to_dict(orient="records")
        )
        input_records: list[dict[str, object]] = [
            {
                "method": "crychic",
                "artifact": str(crychic_artifact_path),
                "sha256": sha256_file(crychic_artifact_path),
                "schema_version": crychic_artifact["schema_version"],
            }
        ]

        for directory, _ in TRACK_A_RUNS:
            table, adapter_manifest = _load_adapter_run(methods_dir / directory)
            if _constant(table, "analysis_track") != "lr_stlr":
                raise ValueError(f"Track-A method emitted a non-LR track: {directory}")
            effects = _standardize_external_effects(table)
            effect_frames.append(effects)
            coverage_records.append(
                _external_coverage(table, adapter_manifest, effects)
            )
            table_path = methods_dir / directory / "interactions_long.parquet"
            input_records.append(
                {
                    "method": _constant(table, "method_id"),
                    "artifact": str(table_path),
                    "sha256": sha256_file(table_path),
                    "schema_version": _constant(table, "schema_version"),
                }
            )

        nichenet_table, nichenet_manifest = _load_adapter_run(
            methods_dir / "nichenet_proxy"
        )
        if _constant(nichenet_table, "analysis_track") != "ligand_target_program":
            raise ValueError("NicheNet proxy must remain Track B")
        track_b = _track_b_effects(nichenet_table)
        input_records.append(
            {
                "method": _constant(nichenet_table, "method_id"),
                "artifact": str(
                    methods_dir / "nichenet_proxy" / "interactions_long.parquet"
                ),
                "sha256": sha256_file(
                    methods_dir / "nichenet_proxy" / "interactions_long.parquet"
                ),
                "schema_version": _constant(nichenet_table, "schema_version"),
            }
        )

        track_a = pd.concat(effect_frames, ignore_index=True)
        coverage = pd.DataFrame.from_records(coverage_records)
        if set(TRACK_A_MAIN_ORDER).difference(track_a["method"].astype(str)):
            raise ValueError("Track-A summary lacks a required method")
        external_universes = {
            frozenset(
                table.loc[:, list(EDGE_KEYS)]
                .astype(str)
                .itertuples(index=False, name=None)
            )
            for table in (
                track_a.loc[track_a["method"].eq(method)]
                for method in TRACK_A_MAIN_ORDER
            )
        }
        if len(external_universes) != 1:
            raise ValueError("Track-A methods do not share the exact edge universe")

        concordance = pairwise_effect_concordance(track_a)
        selection = _figure_edge_selection(track_a)
        if selection.empty:
            raise ValueError("figure edge selection is empty")
        track_a.to_csv(staging / "track_a_effects.tsv", sep="\t", index=False)
        coverage.to_csv(staging / "track_a_coverage.tsv", sep="\t", index=False)
        concordance.to_csv(staging / "track_a_concordance.tsv", sep="\t", index=False)
        selection.to_csv(staging / "track_a_figure_edges.tsv", sep="\t", index=False)
        track_b.to_csv(staging / "track_b_nichenet_effects.tsv", sep="\t", index=False)

        style_path = repo_root / "benchmarks/report/publication.mplstyle"
        _plot_summary(
            track_a=track_a,
            coverage=coverage,
            concordance=concordance,
            track_b=track_b,
            selection=selection,
            style_path=style_path,
            output_stem=staging / "cscc_crossmethod_smoke",
        )
        outputs = []
        for path in sorted(staging.iterdir()):
            outputs.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "rows": (
                        int(pd.read_csv(path, sep="\t").shape[0])
                        if path.suffix == ".tsv"
                        else None
                    ),
                }
            )
        summary_manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "design": {
                "type": "paired",
                "reference": REFERENCE_CONTEXT,
                "target": TARGET_CONTEXT,
                "minimum_paired_subjects": MIN_PAIRED_SUBJECTS,
                "effect_semantics": "tumor_minus_normal_paired_rank_strength",
            },
            "inputs": input_records,
            "outputs": outputs,
            "runtime": {
                "nichenet_elapsed_seconds": _finite_float(
                    nichenet_manifest.get("elapsed_seconds")
                )
            },
            "source": {
                "script_sha256": sha256_file(Path(__file__)),
                "style_sha256": sha256_file(style_path),
                "git": git_metadata(repo_root),
            },
            "guardrails": {
                "track_a_exact_edge_universe": True,
                "crychic_official_claimed": False,
                "crychic_oof_certified_claimed": False,
                "inferential_p_values_reported": False,
                "nichenet_mixed_with_track_a": False,
                "nichenet_sender_or_lr_edge_claimed": False,
                "nichenet_proxy_semantics": PROXY_SEMANTICS,
                "method_superiority_claimed": False,
                "native_result_density_used_as_accuracy": False,
            },
            "interpretation_scope": (
                "bounded_algorithm_smoke; descriptive paired effects only; "
                "not comparative biological validation"
            ),
        }
        write_json(staging / "summary.json", summary_manifest)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        return summary_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve(path: Path, workspace_root: Path) -> Path:
    return path if path.is_absolute() else workspace_root / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--crychic-artifact", type=Path, default=DEFAULT_CRYCHIC_ARTIFACT
    )
    parser.add_argument("--methods-dir", type=Path, default=DEFAULT_METHODS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    output_dir = _resolve(args.output_dir, workspace_root)
    manifest = summarize_crossmethod_smoke(
        crychic_artifact_path=_resolve(args.crychic_artifact, workspace_root),
        methods_dir=_resolve(args.methods_dir, workspace_root),
        output_dir=output_dir,
        repo_root=args.repo_root.resolve(),
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "schema_version": manifest["schema_version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
