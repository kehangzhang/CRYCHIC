"""Generate the Dimitrov cytokine, receptor-protein and robustness report."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.openproblems.common import sha256_file, write_json

METHOD_ORDER = (
    "CRYCHIC availability-state",
    "CellChat composite",
    "CellPhoneDB composite",
    "Connectome specificity",
    "logFC specificity",
    "NATMI specificity",
    "SingleCellSignalR LRscore",
    "LIANA specificity consensus",
)
DISPLAY = {
    "CRYCHIC availability-state": "CRYCHIC",
    "CellChat composite": "CellChat",
    "CellPhoneDB composite": "CellPhoneDB",
    "Connectome specificity": "Connectome",
    "logFC specificity": "logFC Mean",
    "NATMI specificity": "NATMI",
    "SingleCellSignalR LRscore": "SingleCellSignalR",
    "LIANA specificity consensus": "LIANA consensus",
}
COLORS = {
    "CRYCHIC availability-state": "#B53A32",
    "CellChat composite": "#2878B5",
    "CellPhoneDB composite": "#2A9D8F",
    "Connectome specificity": "#D6A62E",
    "logFC specificity": "#4D7C58",
    "NATMI specificity": "#E07A3F",
    "SingleCellSignalR LRscore": "#8A6FA8",
    "LIANA specificity consensus": "#666666",
}
CYTOKINE_IDS = {
    "crychic_availability_state": "CRYCHIC availability-state",
    "cellchat_composite": "CellChat composite",
    "cellphonedb_composite": "CellPhoneDB composite",
    "connectome_specificity": "Connectome specificity",
    "logfc_specificity": "logFC specificity",
    "natmi_specificity": "NATMI specificity",
    "singlecellsignalr_lrscore": "SingleCellSignalR LRscore",
    "liana_specificity_consensus": "LIANA specificity consensus",
}
PERTURBATION_TITLES = {
    "cell_subsampling": "Cell subsampling",
    "label_reshuffling": "Cell-label reshuffling",
    "resource_selective": "Selective LR replacement",
    "resource_nonselective": "Non-selective LR replacement",
}


def _method_name(value: object) -> str:
    text = str(value)
    aliases = {
        "CRYCHIC availability": "CRYCHIC availability-state",
        "LIANA specificity": "LIANA specificity consensus",
        "LIANA magnitude": "LIANA specificity consensus",
        "SingleCellSignalR": "SingleCellSignalR LRscore",
    }
    return aliases.get(text, text)


def _save(figure: plt.Figure, directory: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def _cytokine_tables(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = pd.read_csv(directory / "fisher_rank_curves.tsv", sep="\t")
    curves = curves.loc[
        curves["protocol"].eq("paper_union_max_imputed")
        & curves["fisher_definition"].eq("author_released_code_top_vs_total")
    ].copy()
    controlled = pd.read_csv(directory / "ranking_metrics.tsv", sep="\t")
    controlled = controlled.loc[
        controlled["protocol"].eq("controlled_unique_ligand_target")
        & controlled["aggregation"].eq("max")
    ].copy()
    curves["method"] = curves["method_id"].map(CYTOKINE_IDS)
    controlled["method"] = controlled["method_id"].map(CYTOKINE_IDS)
    return curves.loc[curves["method"].notna()].copy(), controlled.loc[
        controlled["method"].notna()
    ].copy()


def _plot_cytokine(
    curves: pd.DataFrame,
    controlled: pd.DataFrame,
    figures: Path,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    observed = curves.loc[curves["status"].eq("observed")]
    for method_name in METHOD_ORDER:
        method = observed.loc[observed["method"].eq(method_name)].sort_values(
            "rank_cutoff"
        )
        if method.empty:
            continue
        axes[0].plot(
            method["rank_cutoff"],
            method["odds_ratio"],
            marker="o",
            label=DISPLAY[method_name],
            color=COLORS[method_name],
        )
    axes[0].axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Top-ranked LR rows")
    axes[0].set_ylabel("CytoSig enrichment odds ratio")
    axes[0].set_title("A  Released-code union/max-rank enrichment", loc="left")
    axes[0].legend(ncol=2, fontsize=6, loc="best")
    selected = controlled.loc[controlled["aggregation"].eq("max")].copy()
    selected = selected.loc[selected["method"].isin(METHOD_ORDER)].sort_values("auroc")
    colors = [COLORS[value] for value in selected["method"]]
    labels = [DISPLAY[value] for value in selected["method"]]
    axes[1].barh(labels, selected["auroc"], color=colors, height=0.68)
    axes[1].axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set_xlim(0.38, max(0.55, float(selected["auroc"].max()) + 0.02))
    axes[1].set_xlabel("AUROC")
    axes[1].set_title("B  Unique ligand-target control", loc="left")
    figure.tight_layout()
    return _save(figure, figures, "figure01_cytokine_activity")


def _cite_selected(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["method"] = result["method"].map(_method_name)
    result = result.loc[result["method"].isin(METHOD_ORDER)].copy()
    return result.drop_duplicates("method")


def _citeseq_tables(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(directory / "metrics_by_dataset.tsv", sep="\t")
    metrics = metrics.loc[
        metrics["comparison_arm"].eq("H-common/resource_fixed")
        & metrics["direct_method_comparison_valid"].astype(bool)
    ].copy()
    metrics["method"] = metrics["method"].map(_method_name)
    summary = (
        metrics.groupby("method", sort=True, observed=True)
        .agg(
            datasets_evaluable=("dataset", "nunique"),
            auroc_mean=("auroc", "mean"),
            auroc_min=("auroc", "min"),
            auroc_max=("auroc", "max"),
            balanced_auprc_mean=("paper_balanced_auprc_mean", "mean"),
            balanced_auprc_min=("paper_balanced_auprc_mean", "min"),
            balanced_auprc_max=("paper_balanced_auprc_mean", "max"),
            all_scores_tied_datasets=("all_evaluated_scores_tied", "sum"),
            truth_key_coverage_mean=("truth_key_coverage_fraction", "mean"),
        )
        .reset_index()
    )
    return metrics, summary


def _plot_citeseq(
    common: pd.DataFrame,
    figures: Path,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for axis, mean_column, min_column, max_column, title, xlabel in (
        (
            axes[0],
            "auroc_mean",
            "auroc_min",
            "auroc_max",
            "A  AUROC",
            "Mean AUROC (range across 4 datasets)",
        ),
        (
            axes[1],
            "balanced_auprc_mean",
            "balanced_auprc_min",
            "balanced_auprc_max",
            "B  Paper-balanced AUPRC",
            "Mean AUPRC (range across 4 datasets)",
        ),
    ):
        table = _cite_selected(common)
        table = table.sort_values(mean_column).reset_index(drop=True)
        y = np.arange(len(table))
        labels = [DISPLAY[value] for value in table["method"]]
        for index, row in table.iterrows():
            color = COLORS[str(row["method"])]
            axis.plot(
                [row[min_column], row[max_column]],
                [index, index],
                color=color,
                linewidth=1.8,
            )
            axis.scatter(row[mean_column], index, color=color, s=24, zorder=3)
        axis.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_yticks(y, labels)
        axis.set_title(title, loc="left")
        axis.set_xlabel(xlabel)
        axis.set_xlim(0.20, 0.80)
    figure.tight_layout()
    return _save(figure, figures, "figure02_receptor_protein")


def _plot_robustness(summary: pd.DataFrame, figures: Path) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
    for axis, kind in zip(axes.ravel(), PERTURBATION_TITLES, strict=True):
        selected = summary.loc[summary["perturbation"].eq(kind)]
        for method_name in METHOD_ORDER:
            method = selected.loc[selected["method"].eq(method_name)].sort_values(
                "proportion"
            )
            if method.empty:
                continue
            x = 100 * method["proportion"].to_numpy(float)
            median = method["baseline_recovery_median"].to_numpy(float)
            q1 = method["baseline_recovery_q1"].to_numpy(float)
            q3 = method["baseline_recovery_q3"].to_numpy(float)
            color = COLORS[method_name]
            axis.plot(
                x,
                median,
                marker="o",
                color=color,
                label=DISPLAY[method_name],
            )
            axis.fill_between(x, q1, q3, color=color, alpha=0.10, linewidth=0)
        axis.set_title(PERTURBATION_TITLES[kind], loc="left")
        axis.set_xlim(0, 40)
        axis.set_ylim(0, 1.03)
        axis.set_xlabel("Modification (%)")
        axis.set_ylabel("Baseline top-250 recovery")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        fontsize=6,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(figure, figures, "figure03_robustness")


def _overview_table(
    curves: pd.DataFrame,
    controlled: pd.DataFrame,
    common: pd.DataFrame,
    robustness: pd.DataFrame,
) -> pd.DataFrame:
    rows = pd.DataFrame({"method": METHOD_ORDER})
    odds = curves.loc[
        curves["rank_cutoff"].eq(250) & curves["status"].eq("observed"),
        ["method", "odds_ratio"],
    ].drop_duplicates("method")
    auc = controlled.loc[
        controlled["aggregation"].eq("max"), ["method", "auroc"]
    ].drop_duplicates("method")
    common_selected = _cite_selected(common).loc[
        :, ["method", "auroc_mean", "balanced_auprc_mean"]
    ]
    robust = robustness.drop_duplicates("method").loc[
        :, ["method", "normalized_recovery_auc_macro"]
    ]
    result = rows.merge(odds, on="method", how="left")
    result = result.merge(auc, on="method", how="left")
    result = result.merge(
        common_selected.rename(
            columns={
                "auroc_mean": "citeseq_hcommon_auroc",
                "balanced_auprc_mean": "citeseq_hcommon_auprc",
            }
        ),
        on="method",
        how="left",
    )
    return result.merge(robust, on="method", how="left")


def _plot_overview(table: pd.DataFrame, figures: Path) -> list[Path]:
    columns = (
        "odds_ratio",
        "auroc",
        "citeseq_hcommon_auroc",
        "citeseq_hcommon_auprc",
        "normalized_recovery_auc_macro",
    )
    labels = (
        "CytoSig\nOR @250",
        "CytoSig\nAUROC",
        "CITE-seq\nH-common AUROC",
        "CITE-seq\nH-common AUPRC",
        "Robustness\ncurve AUC",
    )
    values = table.loc[:, columns].to_numpy(float)
    ranks = np.full_like(values, np.nan)
    for index in range(values.shape[1]):
        valid = np.isfinite(values[:, index])
        if valid.any():
            ranks[valid, index] = (
                pd.Series(values[valid, index]).rank(pct=True).to_numpy()
            )
    figure, axis = plt.subplots(figsize=(6.8, 3.4))
    masked = np.ma.masked_invalid(ranks)
    image = axis.imshow(masked, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            label = (
                "NA"
                if not np.isfinite(values[row, column])
                else f"{values[row, column]:.3f}"
            )
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white"
                if np.isfinite(ranks[row, column]) and ranks[row, column] > 0.7
                else "black",
            )
    axis.set_xticks(np.arange(len(columns)), labels)
    axis.set_yticks(
        np.arange(len(table)), [DISPLAY[value] for value in table["method"]]
    )
    axis.set_title(
        "Descriptive validation overview (colour = within-column percentile)"
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    colorbar.set_label("Within-endpoint percentile")
    figure.tight_layout()
    return _save(figure, figures, "figure04_validation_overview")


def _markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    selected = table.loc[:, columns].copy()

    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if numeric != 0.0 and abs(numeric) < 0.001:
                return f"{numeric:.2e}"
            return f"{numeric:.3f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join((header, separator, *rows))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run(
    cytokine_dir: Path,
    citeseq_tables: Path,
    robustness_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "REPORT.md"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"Dimitrov validation report exists: {report_path}")
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    style = Path(__file__).resolve().parents[1] / "report" / "publication.mplstyle"
    plt.style.use(style)
    curves, controlled = _cytokine_tables(cytokine_dir)
    cite_metrics, common = _citeseq_tables(citeseq_tables)
    robust_summary = pd.read_csv(
        robustness_dir / "top250_overlap_summary.tsv", sep="\t"
    )
    robust_methods = pd.read_csv(
        robustness_dir / "method_robustness_summary.tsv", sep="\t"
    )
    robustness_manifest = json.loads(
        (robustness_dir / "manifest.json").read_text(encoding="utf-8")
    )
    robustness_protocol = robustness_manifest["protocol"]
    robustness_replicates = int(robustness_protocol["replicates"])
    liana_manifest_path = robustness_dir.parent / "liana_full" / "manifest.json"
    if not liana_manifest_path.is_file():
        liana_manifest_path = robustness_dir.parent / "liana_smoke" / "manifest.json"
    liana_protocol = (
        json.loads(liana_manifest_path.read_text(encoding="utf-8"))["protocol"]
        if liana_manifest_path.is_file()
        else {}
    )
    robustness_permutations = int(liana_protocol.get("n_perms", 0))
    figure_paths = []
    figure_paths.extend(_plot_cytokine(curves, controlled, figures))
    figure_paths.extend(_plot_citeseq(common, figures))
    figure_paths.extend(_plot_robustness(robust_summary, figures))
    overview = _overview_table(curves, controlled, common, robust_methods)
    figure_paths.extend(_plot_overview(overview, figures))
    source_tables = {
        "cytokine_fisher_curves.tsv": curves,
        "cytokine_controlled_metrics.tsv": controlled,
        "citeseq_metrics_by_dataset.tsv": cite_metrics,
        "citeseq_hcommon_summary.tsv": common,
        "robustness_curve_summary.tsv": robust_summary,
        "robustness_method_summary.tsv": robust_methods,
        "validation_overview.tsv": overview,
    }
    table_paths = []
    for filename, table in source_tables.items():
        path = tables / filename
        table.to_csv(path, sep="\t", index=False, na_rep="")
        table_paths.append(path)
    cytokine_250 = curves.loc[
        curves["rank_cutoff"].eq(250) & curves["status"].eq("observed")
    ].sort_values("odds_ratio", ascending=False)
    cytokine_auc = controlled.sort_values("auroc", ascending=False)
    common_selected = _cite_selected(common).sort_values("auroc_mean", ascending=False)
    robust_macro = robust_methods.drop_duplicates("method").sort_values(
        "normalized_recovery_auc_macro", ascending=False
    )
    crychic_250 = cytokine_250.loc[
        cytokine_250["method"].eq("CRYCHIC availability-state")
    ].iloc[0]
    crychic_auc = cytokine_auc.loc[
        cytokine_auc["method"].eq("CRYCHIC availability-state")
    ].iloc[0]
    crychic_common = common_selected.loc[
        common_selected["method"].eq("CRYCHIC availability-state")
    ].iloc[0]
    crychic_robust = robust_macro.loc[
        robust_macro["method"].eq("CRYCHIC availability-state")
    ].iloc[0]
    crychic_cite_datasets = cite_metrics.loc[
        cite_metrics["method"].eq("CRYCHIC availability-state")
    ].sort_values("dataset")
    cite_external = common.loc[common["method"].ne("CRYCHIC availability-state")]
    best_cite_auroc = cite_external.sort_values("auroc_mean", ascending=False).iloc[0]
    best_cite_auprc = cite_external.sort_values(
        "balanced_auprc_mean", ascending=False
    ).iloc[0]
    crychic_robust_by_kind = robust_methods.loc[
        robust_methods["method"].eq("CRYCHIC availability-state")
    ].sort_values("perturbation")
    cytokine_rank = int(
        cytokine_250.reset_index(drop=True).index[
            cytokine_250.reset_index(drop=True)["method"].eq(
                "CRYCHIC availability-state"
            )
        ][0]
        + 1
    )
    robustness_rank = int(
        robust_macro.reset_index(drop=True).index[
            robust_macro.reset_index(drop=True)["method"].eq(
                "CRYCHIC availability-state"
            )
        ][0]
        + 1
    )
    report = f"""# Dimitrov 2022 三类验证复现与 CRYCHIC 对比

生成日期：2026-07-17

## 结论摘要

- 本报告使用真实 TNBC、4 个真实 CITE-seq 数据集和 PBMC3k，复现细胞因子活性、受体蛋白和稳健性三个终点。CRYCHIC 是确定性非深度学习方法，不存在训练集/验证集训练。
- CytoSig 的 union/max-imputed 作者代码口径下，CRYCHIC top-250 OR 为 **{crychic_250["odds_ratio"]:.3f}**，主 8-arm 排名 **{cytokine_rank}/{len(cytokine_250)}**，BH q={crychic_250["q_value_bh_primary_8arm"]:.2g}；但受控 unique ligand-target AUROC 仅 **{crychic_auc["auroc"]:.3f}**。这支持前列局部富集，不支持全排序领先。
- CITE-seq 同一 H-common 固定全集中，CRYCHIC 平均 AUROC/论文式平衡 AUPRC 为 **{crychic_common["auroc_mean"]:.3f}/{crychic_common["balanced_auprc_mean"]:.3f}**；最佳外部 score arm 分别为 {best_cite_auroc["method"]} {best_cite_auroc["auroc_mean"]:.3f} 和 {best_cite_auprc["method"]} {best_cite_auprc["balanced_auprc_mean"]:.3f}。这是当前最明确的相对优势，但覆盖 LR 很少且若干基线分数完全并列。
- PBMC3k 五重复稳健性中，CRYCHIC 四扰动宏平均恢复曲线 AUC 为 **{crychic_robust["normalized_recovery_auc_macro"]:.3f}**，排名 **{robustness_rank}/{len(robust_macro)}**；属于较稳健方法之一，不是第一名。
- 综合结论是“CRYCHIC 在部分同资源指标上领先，并在 top-rank 与稳健性上处于前列”，不能表述为全面超过 CellChat、CellPhoneDB、LIANA 或多数方法。

## 设计与数据

| 验证 | 数据 | 终点 | 本报告主口径 |
| --- | --- | --- | --- |
| 细胞因子活性 | Wu TNBC，42,512 cells，29 target cell types | CytoSig response agreement | union/max-imputed Fisher OR；812 个 unique ligand-target 受控 AUROC/AP |
| 受体蛋白 | 5k PBMC、5k PBMC NextGem、10k PBMC、10k MALT，共 27,051 cells | ADT receptor `z >= 1.645` | 638-pair H-common 固定全集；AUROC；100 次 1:1 有放回负样本梯形 PR-AUC |
| 稳健性 | PBMC3k，2,638 QC-passing cells，9 clusters | 各方法自身 baseline top-250 恢复率 | 5--40%，5% 步长，{robustness_replicates} repeats，四类共享扰动 |

这些数据均为单 condition。CRYCHIC 在本报告中只评估静态 `availability_state`，不是完整多条件 `comm_strength`、sender attribution 或 downstream response。

## 细胞因子活性

![Cytokine validation](figures/figure01_cytokine_activity.png)

Union/max-imputed、作者 released-code top-vs-total 的 top-250 结果：

{_markdown_table(cytokine_250.assign(display=cytokine_250["method"].map(DISPLAY)), ["display", "odds_ratio", "q_value_bh_primary_8arm", "true_positive", "false_positive"])}

旧报告的 `OR=2.559, rank 1/8` 使用各方法独立 returned universe，只能作为敏感性分析。正确扩展 union 中 CRYCHIC 排名第 3；同时该 185,020-row union 由 CRYCHIC 的广覆盖输出主导，外部方法只返回 2,207--9,161 rows，所以不能把 union OR 当作纯算法公平比较。

受控 unique ligand-target 口径消除了 sender/receptor 复制。CRYCHIC AUROC/AP/100 次平衡 AP 为 **{crychic_auc["auroc"]:.3f}/{crychic_auc["average_precision"]:.3f}/{crychic_auc["balanced_average_precision_mean"]:.3f}**；其 AUROC 低于 0.5，说明静态 availability 并不等同于下游细胞因子转录活性。

## 受体蛋白验证

![CITE-seq receptor validation](figures/figure02_receptor_protein.png)

相同 H-common 资源、相同 sender x receiver x LR 固定全集的 8 个代表性 score arms：

{_markdown_table(common_selected.assign(display=common_selected["method"].map(DISPLAY)), ["display", "auroc_mean", "auroc_min", "auroc_max", "balanced_auprc_mean", "all_scores_tied_datasets", "truth_key_coverage_mean"])}

CRYCHIC 分数据集结果：

{_markdown_table(crychic_cite_datasets, ["dataset", "auroc", "paper_balanced_auprc_mean", "n_evaluated_edges", "n_evaluated_lr_pairs", "all_evaluated_scores_tied"])}

论文式 AUPRC 使用全部 positives 加等量、**有放回** negatives，100 次，seed 1234，并按 yardstick 0.0.8 计算梯形 PR-AUC。NumPy MT19937 与 R `sample()` 不会逐 draw 完全一致。PBMC 10k/MALT 中多个外部 score arms 全部并列；梯形 PR-AUC 对全并列分数可给出 0.75，因此必须与 AUROC=0.5 和 tied flag 一起解释。

ADT 终点只验证 receiver-receptor protein specificity，不验证 ligand、sender、物理结合或因果通信。CRYCHIC availability 直接包含 receptor availability，任务结构与其静态分数更匹配。

## 稳健性

![Robustness validation](figures/figure03_robustness.png)

{_markdown_table(robust_macro.assign(display=robust_macro["method"].map(DISPLAY)), ["display", "normalized_recovery_auc_macro", "recovery_mean_at_40_macro", "jaccard_mean_at_40_macro"])}

CRYCHIC 分扰动：

{_markdown_table(crychic_robust_by_kind, ["perturbation", "normalized_recovery_auc_0_40", "recovery_mean_at_40", "jaccard_mean_at_40"])}

恢复率是与每种方法自身未扰动 top-250 的一致性，不是生物学准确率。作者代码所谓 TPR 分母实际是扰动后 top 集；本报告保留该值、以 baseline 为分母的 recovery 和 Jaccard，并以 recovery 为主。

## 跨终点概览

![Validation overview](figures/figure04_validation_overview.png)

颜色仅表示每列内部百分位，不同终点不可加权求和。原始数值位于 `tables/validation_overview.tsv`。

## 与原论文的差异和限制

1. 当前外部方法来自 LIANA 1.7.3 components/current consensus，而不是作者 LIANA 0.0.5、旧 OmniPath 和独立历史软件；CytoSig 用 100 permutations，稳健性用 {robustness_permutations} permutations。Crosstalk 不可用。
2. TNBC truth 是 OpenProblems 冻结的 812-row binary proxy，只有 28 ligands，不含原始 MLM score/p/q。HER2 原始 19,311 cells、5 patients 和发表前 43-signature centroid 已 checksum 验证，但本轮没有生成可比方法结果。
3. CITE-seq 完成 4/7 数据集；CBMC 对象版本不一致，SLN111/208 protein matrix/name 维度不一致，未猜测修补。standalone CellChat、CellPhoneDB 和 NicheNet 没有真实 CITE-seq 预测，主表中的同名项是 LIANA components。
4. PBMC3k 的细胞下采样、标签重排、选择性和非选择性 LR **替换**均完成 8 个比例 x 5 repeats；不是向资源额外注入假边。当前聚类是 Scanpy 对论文 Seurat 协议的近似。
5. NicheNet 输出 ligand-to-target regulatory potential，需要目标基因程序，不是直接 source-target-LR 排名，因此没有强行映射到这三个静态 LR 终点。

## 可复现产物

- `tables/`：每张图的源数据、逐数据集指标和统一 overview。
- `figures/`：300 dpi PNG，以及矢量 PDF、SVG。
- 上游 CytoSig、CITE-seq、LIANA robustness、CRYCHIC robustness 和统一评价目录均含 checksum manifest。

## 参考

Dimitrov D, et al. Comparison of methods and resources for cell-cell communication inference from single-cell RNA-Seq data. *Nature Communications*. 2022;13:3224. doi:10.1038/s41467-022-30755-0.
"""
    report_path.write_text(report, encoding="utf-8")
    inputs = [
        cytokine_dir / "fisher_rank_curves.tsv",
        cytokine_dir / "ranking_metrics.tsv",
        cytokine_dir / "manifest.json",
        citeseq_tables / "metrics_by_dataset.tsv",
        citeseq_tables / "method_summary.tsv",
        citeseq_tables / "manifest.json",
        robustness_dir / "top250_overlap_summary.tsv",
        robustness_dir / "method_robustness_summary.tsv",
        robustness_dir / "manifest.json",
    ]
    outputs = [report_path, *figure_paths, *table_paths]
    manifest: dict[str, object] = {
        "schema_version": "crychic-dimitrov-validation-report-v2",
        "status": "complete",
        "inputs": {str(path): sha256_file(path) for path in inputs},
        "outputs": {
            str(path.relative_to(output_dir)): sha256_file(path) for path in outputs
        },
        "headline": _json_safe(
            {
                "cytokine_top250_or": crychic_250["odds_ratio"],
                "cytokine_unique_auroc": crychic_auc["auroc"],
                "citeseq_hcommon_auroc_mean": crychic_common["auroc_mean"],
                "citeseq_hcommon_paper_balanced_auprc_mean": crychic_common[
                    "balanced_auprc_mean"
                ],
                "robustness_normalized_auc_macro": crychic_robust[
                    "normalized_recovery_auc_macro"
                ],
            }
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cytokine_dir", type=Path)
    parser.add_argument("citeseq_tables", type=Path)
    parser.add_argument("robustness_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(
        args.cytokine_dir,
        args.citeseq_tables,
        args.robustness_dir,
        args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
