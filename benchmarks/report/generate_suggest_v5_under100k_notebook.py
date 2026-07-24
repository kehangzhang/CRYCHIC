"""Build and execute the publication notebook for suggest_v5 under-100k runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


def _markdown(source: str) -> Any:
    return nbformat.v4.new_markdown_cell(source.strip())


def _code(source: str) -> Any:
    body = source.strip()
    if not body.startswith("%%time"):
        body = f"%%time\n{body}"
    return nbformat.v4.new_code_cell(body)


def build_notebook() -> Any:
    """Return an unexecuted notebook with timing on every code cell."""

    cells = [
        _markdown(
            """
# suggest_v5: under-100k multi-context communication benchmarks

Frozen evidence report for all eligible section II and III benchmarks. The
notebook distinguishes exact reruns, paper-protocol reconstructions, public
cohort reconstructions, and non-estimable cells.
"""
        ),
        _code(
            """
from pathlib import Path
import hashlib
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

workspace = Path(os.environ.get("CRYCHIC_BENCHMARK_WORKSPACE", Path.cwd()))
results_root = workspace / "benchmark_work" / "suggest_v5_under100k_20260724"
figure_dir = results_root / "publication_figures"
figure_dir.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.6,
})
sns.set_style("ticks")

COLORS = {
    "CRYCHIC": "#C43C39",
    "scSeqCommDiff": "#3568A8",
    "CellChat": "#2D8C6F",
    "LIANA": "#D18B2C",
    "STACCato": "#6B4C9A",
    "DCST": "#4A7C59",
    "simple": "#626262",
}

def save_figure(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figure_dir / f"{stem}.{suffix}", bbox_inches="tight")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
"""
        ),
        _markdown("## Evidence inventory"),
        _code(
            """
result_dirs = {
    "Tensor-cell2cell": results_root / "tensor_cell2cell",
    "STACCato": results_root / "staccato",
    "DCST": results_root / "dcst",
    "PDAC": results_root / "scaccordion_pdac",
    "AKI": results_root / "scaccordion_aki",
    "RCC fine": results_root / "scaccordion_rcc",
    "RCC coarse": results_root / "scaccordion_rcc_coarse",
    "Signed estimand": results_root / "signed_estimand_contract",
    "Score-engine crossover": (
        results_root / "score_generator_engine_crossover" /
        "formal_20seed_100boot"
    ),
    "Native end-to-end": (
        results_root / "score_generator_engine_crossover" /
        "native_end_to_end_evaluation"
    ),
}

inventory = []
for benchmark, directory in result_dirs.items():
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inventory.append({
        "benchmark": benchmark,
        "status": manifest.get("status"),
        "schema": manifest.get("schema_version"),
        "manifest_sha256": sha256(manifest_path)[:16],
    })
inventory = pd.DataFrame(inventory)
assert inventory["status"].eq("complete").all()
display(inventory)
"""
        ),
        _markdown("## Known-truth context programs"),
        _code(
            """
tensor = pd.read_csv(
    result_dirs["Tensor-cell2cell"] / "aggregate_metrics.tsv", sep="\t"
)
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
axes[0].plot(
    tensor["noise"], tensor["event_auprc_mean"], marker="o", color="#3568A8"
)
axes[0].plot(
    tensor["noise"], tensor["lr_auprc_mean"], marker="s", color="#D18B2C"
)
axes[0].set(xlabel="Noise", ylabel="AUPRC", title="A  Event and LR recovery")
axes[0].legend(["Event", "LR"], frameon=False)
axes[0].set_ylim(0, 1.04)
axes[1].plot(
    tensor["noise"],
    tensor["normalized_reconstruction_error_mean"],
    marker="o",
    color="#C43C39",
)
axes[1].set(
    xlabel="Noise",
    ylabel="Normalized reconstruction error",
    title="B  Tensor reconstruction",
)
save_figure(fig, "figure_01_tensor_cell2cell")
plt.show()
display(tensor.round(4))
"""
        ),
        _markdown("## Condition and batch-confounding simulation"),
        _code(
            """
staccato = pd.read_csv(
    result_dirs["STACCato"] / "aggregate_metrics.tsv", sep="\t"
)
keep = staccato[staccato["method"].isin([
    "eventwise_ols_adjusted", "staccato_adjusted"
])].copy()
labels = {
    "eventwise_ols_adjusted": "Eventwise OLS",
    "staccato_adjusted": "STACCato",
}
keep["Method"] = keep["method"].map(labels)
order = ["balanced", "moderate", "extreme"]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
sns.barplot(
    data=keep, x="design", y="mse_mean", hue="Method", order=order,
    palette=["#626262", COLORS["STACCato"]], ax=axes[0]
)
axes[0].set(xlabel="Design", ylabel="Disease-effect MSE", title="A  Effect error")
sns.barplot(
    data=keep, x="design", y="sign_accuracy_mean", hue="Method", order=order,
    palette=["#626262", COLORS["STACCato"]], ax=axes[1]
)
axes[1].set(
    xlabel="Design", ylabel="Sign accuracy", title="B  Direction recovery",
    ylim=(0, 1.04)
)
axes[0].legend(frameon=False, title=None)
axes[1].legend_.remove()
save_figure(fig, "figure_02_staccato")
plt.show()
display(keep[["design", "Method", "mse_mean", "sign_accuracy_mean"]].round(4))
"""
        ),
        _markdown("## DCST paper-native independent sweeps"),
        _code(
            """
dcst = pd.read_csv(result_dirs["DCST"] / "aggregate_metrics.tsv", sep="\t")
dcst["Method"] = dcst["method"].map({
    "dcst_protocol_reimplementation": "DCST",
    "crychic_generic_synthetic_prior": "CRYCHIC generic prior",
})
palette = {"DCST": COLORS["DCST"], "CRYCHIC generic prior": COLORS["CRYCHIC"]}
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
for ax, sweep, title, xlabel in (
    (axes[0], "subject_count", "A  Subject-count sweep", "Subjects / condition"),
    (axes[1], "receiver_cell_count", "B  Receiver-cell sweep", "Receiver cells"),
):
    selected = dcst[dcst["sweep"].eq(sweep)]
    for method, group in selected.groupby("Method", sort=False):
        ax.plot(
            group["axis_value"], group["auroc_mean"], marker="o",
            label=method, color=palette[method]
        )
    ax.axhline(0.5, color="#999999", lw=0.8, ls="--")
    ax.set(xlabel=xlabel, ylabel="AUROC", title=title, ylim=(0, 1.04))
axes[0].legend(frameon=False)
save_figure(fig, "figure_03_dcst")
plt.show()
display(
    dcst.groupby(["sweep", "Method"], observed=True)[
        ["auroc_mean", "auprc_mean", "direction_accuracy_mean"]
    ].mean().round(4)
)
"""
        ),
        _markdown("## Patient graph phenotype recovery"),
        _code(
            """
cohorts = []
for cohort, key in (("PDAC", "PDAC"), ("AKI", "AKI"), ("RCC", "RCC fine")):
    table = pd.read_csv(result_dirs[key] / "summary_metrics.tsv", sep="\t")
    table["Cohort"] = cohort
    cohorts.append(table)
patient = pd.concat(cohorts, ignore_index=True)
method_labels = {
    "corr_ot": "CORR-OT",
    "crychic_canonical_event_correlation": "CRYCHIC projection",
    "got": "GOT",
    "scaccordion_dw_ot": "scACCorDiON DW-OT",
    "tabular_pca": "Tabular PCA",
}
patient["Method"] = patient["method"].map(method_labels)
patient_palette = ["#3568A8", "#C43C39", "#D18B2C", "#2D8C6F", "#777777"]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
sns.barplot(
    data=patient, x="Cohort", y="ari_at_true_k", hue="Method",
    palette=patient_palette, ax=axes[0]
)
axes[0].axhline(0, color="#555555", lw=0.7)
axes[0].set(xlabel="", ylabel="ARI", title="A  ARI at known phenotype k")
sns.barplot(
    data=patient, x="Cohort", y="maximum_ari_over_k", hue="Method",
    palette=patient_palette, ax=axes[1]
)
axes[1].axhline(0, color="#555555", lw=0.7)
axes[1].set(xlabel="", ylabel="Maximum ARI", title="B  Best k in 2-7")
axes[0].legend(frameon=False, title=None, ncol=1, loc="best")
axes[1].legend_.remove()
save_figure(fig, "figure_04_patient_graphs")
plt.show()
display(patient[["Cohort", "Method", "ari_at_true_k", "maximum_ari_over_k"]].round(4))
"""
        ),
        _markdown("## RCC annotation-resolution robustness"),
        _code(
            """
rcc_rows = []
for resolution, key in (("Fine", "RCC fine"), ("Coarse", "RCC coarse")):
    table = pd.read_csv(result_dirs[key] / "summary_metrics.tsv", sep="\t")
    table["Resolution"] = resolution
    rcc_rows.append(table)
rcc = pd.concat(rcc_rows, ignore_index=True)
rcc["Method"] = rcc["method"].map(method_labels)

distance_correlations = []
for method in sorted(rcc["method"].unique()):
    fine_path = result_dirs["RCC fine"] / "distances" / f"{method}.npz"
    coarse_path = result_dirs["RCC coarse"] / "distances" / f"{method}.npz"
    fine = np.load(fine_path)["distance"]
    coarse = np.load(coarse_path)["distance"]
    upper = np.triu_indices_from(fine, k=1)
    correlation = pd.Series(fine[upper]).corr(
        pd.Series(coarse[upper]), method="spearman"
    )
    distance_correlations.append({
        "Method": method_labels[method], "Spearman": correlation
    })
distance_correlations = pd.DataFrame(distance_correlations)

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
sns.barplot(
    data=rcc, x="Method", y="ari_at_true_k", hue="Resolution",
    palette=["#3568A8", "#D18B2C"], ax=axes[0]
)
axes[0].set(xlabel="", ylabel="ARI", title="A  Phenotype recovery")
axes[0].tick_params(axis="x", rotation=35)
axes[0].legend(frameon=False, title=None)
sns.barplot(
    data=distance_correlations, x="Method", y="Spearman", color="#2D8C6F", ax=axes[1]
)
axes[1].set(
    xlabel="", ylabel="Fine-coarse Spearman", title="B  Distance robustness",
    ylim=(-1, 1)
)
axes[1].tick_params(axis="x", rotation=35)
save_figure(fig, "figure_05_rcc_annotation")
plt.show()
display(distance_correlations.round(4))
"""
        ),
        _markdown("## Native end-to-end three-group comparison"),
        _code(
            """
native = pd.read_csv(
    result_dirs["Native end-to-end"] / "multigroup_method_summary.tsv", sep="\t"
)
native["Method"] = native["method_label"]
native_palette = {
    "CRYCHIC generic baseline": COLORS["CRYCHIC"],
    "scSeqCommDiff": COLORS["scSeqCommDiff"],
    "CellChat": COLORS["CellChat"],
    "LIANA": COLORS["LIANA"],
}
native = native.sort_values("primary_rank")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
sns.barplot(
    data=native, x="Method", y="mean_omnibus_auprc",
    palette=native_palette, hue="Method", legend=False, ax=axes[0]
)
axes[0].set(xlabel="", ylabel="Mean omnibus AUPRC", title="A  Event detection")
axes[0].tick_params(axis="x", rotation=30)
sns.barplot(
    data=native, x="Method", y="mean_direction_accuracy_all_active",
    palette=native_palette, hue="Method", legend=False, ax=axes[1]
)
axes[1].set(
    xlabel="", ylabel="Direction accuracy", title="B  Signed direction",
    ylim=(0, 1)
)
axes[1].tick_params(axis="x", rotation=30)
save_figure(fig, "figure_06_native_three_group")
plt.show()
display(native[[
    "primary_rank", "Method", "mean_omnibus_auprc", "mean_omnibus_auroc",
    "mean_direction_accuracy_all_active", "mean_event_coverage"
]].round(4))
"""
        ),
        _markdown("## Score generator x differential engine crossover"),
        _code(
            """
crossover = pd.read_csv(
    result_dirs["Score-engine crossover"] / "arm_summary.tsv", sep="\t"
)
matrix = pd.read_csv(
    result_dirs["Score-engine crossover"] / "eligibility_matrix.tsv", sep="\t"
)
generator_order = [
    "crychic_native", "crychic_availability", "crychic_availability_prior",
    "crychic_sender_response", "crychic_downstream_support", "liana_magnitude",
    "cellchat_probability", "cellphonedb_mean", "simple_lr_mean",
    "simple_lr_product",
]
engine_order = [
    "sample_glm", "clustered_cr2", "staccato",
    "crychic_common_functional_oof", "scseqcommdiff_multi_sample",
]
heat = crossover.pivot(
    index="score_generator", columns="differential_engine", values="mean_auprc"
).reindex(index=generator_order, columns=engine_order)
fig, ax = plt.subplots(figsize=(6.7, 4.5), constrained_layout=True)
sns.heatmap(
    heat, cmap="viridis", vmin=0, vmax=max(0.5, np.nanmax(heat.to_numpy())),
    annot=True, fmt=".3f", linewidths=0.4, linecolor="white",
    cbar_kws={"label": "Mean contrast-event AUPRC"}, ax=ax
)
ax.set(
    xlabel="Differential engine", ylabel="Score generator",
    title="20-seed component crossover"
)
ax.tick_params(axis="x", rotation=25)
save_figure(fig, "figure_07_score_engine_crossover")
plt.show()
display(crossover.sort_values("primary_rank").head(15).round(4))
display(matrix.groupby(["status", "reason_code"], dropna=False).size().rename("cells"))
"""
        ),
        _markdown("## Contract gate and compact conclusions"),
        _code(
            """
cases = pd.read_csv(
    result_dirs["Signed estimand"] / "case_results.tsv", sep="\t"
)
invariants = pd.read_csv(
    result_dirs["Signed estimand"] / "invariance_results.tsv", sep="\t"
)
conclusions = pd.DataFrame([
    {
        "Endpoint": "Signed estimand contract",
        "Result": f"{cases.passed.sum()}/{len(cases)} cases; "
                  f"{invariants.passed.sum()}/{len(invariants)} invariants",
        "Interpretation": (
            "Gate A passed; formal p values remain NE without full-pipeline "
            "resampling."
        ),
    },
    {
        "Endpoint": "Native three-group omnibus AUPRC",
        "Result": "CRYCHIC rank 3/4",
        "Interpretation": (
            "No native canonical accuracy advantage on this planted fixture."
        ),
    },
    {
        "Endpoint": "Component crossover",
        "Result": f"{matrix.status.eq('observed').sum()} fully observed; "
                  f"{matrix.status.eq('partially_observed').sum()} partial; "
                  f"{matrix.status.eq('not_estimable').sum()} NE",
        "Interpretation": (
            "NE reflects design/API constraints, never numerical zero filling."
        ),
    },
    {
        "Endpoint": "Patient graph cohorts",
        "Result": "PDAC, AKI, and RCC completed",
        "Interpretation": (
            "These test phenotype recovery, not event-level causal truth."
        ),
    },
])
display(conclusions)
"""
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )


def generate(
    output: Path,
    *,
    execute: bool,
    workspace_root: Path | None,
    timeout_seconds: int,
) -> None:
    notebook = build_notebook()
    output.parent.mkdir(parents=True, exist_ok=True)
    if execute:
        if workspace_root is None:
            raise ValueError("workspace_root is required when executing")
        environment_value = os.environ.get("CRYCHIC_BENCHMARK_WORKSPACE")
        os.environ["CRYCHIC_BENCHMARK_WORKSPACE"] = str(workspace_root.resolve())
        try:
            client = NotebookClient(
                notebook,
                timeout=timeout_seconds,
                kernel_name="python3",
                resources={"metadata": {"path": str(workspace_root.resolve())}},
            )
            client.execute()
        finally:
            if environment_value is None:
                os.environ.pop("CRYCHIC_BENCHMARK_WORKSPACE", None)
            else:
                os.environ["CRYCHIC_BENCHMARK_WORKSPACE"] = environment_value
    nbformat.write(notebook, output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main() -> int:
    args = _parser().parse_args()
    generate(
        args.output,
        execute=args.execute,
        workspace_root=args.workspace_root,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
