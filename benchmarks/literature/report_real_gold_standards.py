"""Summarize the real CITE-seq and IPF literature gold-standard runs."""

# ruff: noqa: E501, RUF001 -- embedded Chinese Markdown preserves native prose.

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve  # type: ignore[import-untyped]

from benchmarks.adapters.common import sha256_file

CITE_DATASETS = (
    ("5k_pbmc", "5k PBMC"),
    ("5k_pbmc_nextgem", "5k PBMC NextGem"),
    ("pbmc_10k", "10k PBMC"),
    ("malt_10k", "10k MALT"),
)

CITE_NATIVE_METHODS = (
    "CRYCHIC availability-state",
    "CellChat composite",
    "CellPhoneDB composite",
    "LIANA magnitude consensus",
    "NATMI specificity",
    "SingleCellSignalR LRscore",
)

CITE_COMMON_METHODS = (
    "CRYCHIC availability-state",
    "CellChat p-value",
    "CellPhoneDB p-value",
    "LIANA magnitude consensus",
    "NATMI specificity",
    "logFC specificity",
)

IPF_METHODS = (
    "CRYCHIC availability-state",
    "CellChat composite",
    "CellChat p-value",
    "CellPhoneDB composite",
    "CellPhoneDB p-value",
    "LIANA specificity consensus",
    "NATMI specificity",
    "SingleCellSignalR LRscore",
)

DISPLAY_NAMES = {
    "CRYCHIC availability-state": "CRYCHIC availability",
    "CellChat composite": "CellChat composite",
    "CellChat p-value": "CellChat p-value",
    "CellPhoneDB composite": "CellPhoneDB composite",
    "CellPhoneDB p-value": "CellPhoneDB p-value",
    "LIANA magnitude consensus": "LIANA magnitude",
    "LIANA specificity consensus": "LIANA specificity",
    "NATMI specificity": "NATMI specificity",
    "SingleCellSignalR LRscore": "SingleCellSignalR",
    "logFC specificity": "logFC specificity",
}

METHOD_COLORS = {
    "CRYCHIC availability-state": "#0072B2",
    "CellChat p-value": "#D55E00",
    "CellPhoneDB p-value": "#009E73",
    "LIANA specificity consensus": "#CC79A7",
    "NATMI specificity": "#E69F00",
    "SingleCellSignalR LRscore": "#6F6F6F",
}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _load_evaluation(path: Path, **identifiers: str) -> pd.DataFrame:
    point_path = path / "point_estimates.tsv"
    negative_path = path / "negative_sampling_summary.tsv"
    bootstrap_path = path / "bootstrap_summary.tsv"
    point = pd.read_csv(point_path, sep="\t")
    negative = pd.read_csv(negative_path, sep="\t")
    bootstrap = pd.read_csv(bootstrap_path, sep="\t")
    balanced = negative.loc[
        negative["metric"].eq("average_precision"),
        ["method", "replicate_mean", "ci_lower", "ci_upper"],
    ].rename(
        columns={
            "replicate_mean": "balanced_auprc",
            "ci_lower": "balanced_auprc_ci_lower",
            "ci_upper": "balanced_auprc_ci_upper",
        }
    )
    auroc_interval = bootstrap.loc[
        bootstrap["metric"].eq("auroc"), ["method", "ci_lower", "ci_upper"]
    ].rename(
        columns={
            "ci_lower": "auroc_ci_lower",
            "ci_upper": "auroc_ci_upper",
        }
    )
    result = point.merge(
        balanced, on="method", how="left", validate="one_to_one"
    ).merge(auroc_interval, on="method", how="left", validate="one_to_one")
    for key, value in reversed(tuple(identifiers.items())):
        result.insert(0, key, value)
    result["evaluation_dir"] = str(path.resolve())
    return result


def load_citeseq(results_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory, display in CITE_DATASETS:
        evaluations = (
            ("native", "native/evaluation_independent"),
            ("H-common", "H-common/evaluation_resource_fixed"),
            ("H-common-independent", "H-common/evaluation_independent"),
        )
        for arm, relative_path in evaluations:
            frames.append(
                _load_evaluation(
                    results_root / "citeseq" / directory / relative_path,
                    dataset_short=directory,
                    dataset_display=display,
                    arm=arm,
                )
            )
    return pd.concat(frames, ignore_index=True)


def load_ipf(results_root: Path) -> pd.DataFrame:
    root = results_root / "ipf" / "022I"
    arms = (
        ("native", "Native resources", root / "native/evaluation_paper_base_full"),
        (
            "H-common_full",
            "H-common on full base",
            root / "H-common/evaluation_paper_base_full",
        ),
        (
            "H-common_representable",
            "H-common representable",
            root / "H-common/evaluation_representable",
        ),
    )
    return pd.concat(
        [
            _load_evaluation(path, arm=arm, arm_display=display)
            for arm, display, path in arms
        ],
        ignore_index=True,
    )


def _summarize_cite(table: pd.DataFrame, arm: str) -> pd.DataFrame:
    selected = table.loc[table["arm"].eq(arm) & table["status"].eq("observed")]
    return (
        selected.groupby("method", sort=True, observed=True)
        .agg(
            datasets_evaluable=("dataset_short", "nunique"),
            auroc_mean=("auroc", "mean"),
            auroc_min=("auroc", "min"),
            auroc_max=("auroc", "max"),
            balanced_auprc_mean=("balanced_auprc", "mean"),
            balanced_auprc_min=("balanced_auprc", "min"),
            balanced_auprc_max=("balanced_auprc", "max"),
        )
        .reset_index()
        .sort_values(["auroc_mean", "method"], ascending=[False, True])
    )


def _configure_matplotlib(style_path: Path) -> None:
    plt.style.use(style_path)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def _save_figure(figure: Figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "svg"):
        figure.savefig(output / f"{stem}.{suffix}", facecolor="white")
    figure.savefig(output / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(figure)


def _heatmap(
    cite: pd.DataFrame,
    *,
    arm: str,
    methods: tuple[str, ...],
    output: Path,
    stem: str,
    title: str,
) -> None:
    selected = cite.loc[
        cite["arm"].eq(arm) & cite["method"].isin(methods),
        ["dataset_display", "method", "auroc"],
    ]
    matrix = selected.pivot(index="method", columns="dataset_display", values="auroc")
    matrix = matrix.reindex(
        index=list(methods), columns=[item[1] for item in CITE_DATASETS]
    )
    matrix["Mean"] = matrix.mean(axis=1, skipna=True)
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.MaskedArray(  # type: ignore[no-untyped-call]
        values, mask=~np.isfinite(values)
    )
    cmap = mpl.colormaps["RdYlBu"].copy()
    cmap.set_bad("#E6E6E6")
    figure, axis = plt.subplots(figsize=(8.1, 4.3), constrained_layout=True)
    image = axis.imshow(masked, cmap=cmap, vmin=0.4, vmax=0.75, aspect="auto")
    axis.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=25, ha="right")
    axis.set_yticks(
        np.arange(matrix.shape[0]),
        [DISPLAY_NAMES.get(method, method) for method in matrix.index],
    )
    axis.set_title(title, loc="left")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "NA" if not np.isfinite(value) else f"{value:.3f}"
            color = (
                "#222222" if not np.isfinite(value) or 0.49 < value < 0.68 else "white"
            )
            axis.text(
                column, row, text, ha="center", va="center", color=color, fontsize=8
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("AUROC")
    _save_figure(figure, output, stem)


def _ipf_metric_plot(ipf: pd.DataFrame, output: Path) -> None:
    arms = ("native", "H-common_representable")
    titles = ("Native resources", "H-common representable universe")
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.6), sharex=True, sharey=True)
    y = np.arange(len(IPF_METHODS))
    for axis, arm, title in zip(axes, arms, titles, strict=True):
        selected = (
            ipf.loc[ipf["arm"].eq(arm) & ipf["method"].isin(IPF_METHODS)]
            .set_index("method")
            .reindex(IPF_METHODS)
        )
        axis.axvline(0.5, color="#888888", linewidth=1, linestyle="--", zorder=0)
        auroc = selected["auroc"].to_numpy(dtype=float)
        auroc_error = np.vstack(
            (
                auroc - selected["auroc_ci_lower"].to_numpy(dtype=float),
                selected["auroc_ci_upper"].to_numpy(dtype=float) - auroc,
            )
        )
        balanced = selected["balanced_auprc"].to_numpy(dtype=float)
        balanced_error = np.vstack(
            (
                balanced - selected["balanced_auprc_ci_lower"].to_numpy(dtype=float),
                selected["balanced_auprc_ci_upper"].to_numpy(dtype=float) - balanced,
            )
        )
        axis.errorbar(
            auroc,
            y - 0.11,
            xerr=auroc_error,
            fmt="o",
            color="#0072B2",
            capsize=2,
            linewidth=1,
            label="AUROC",
            zorder=3,
        )
        axis.errorbar(
            balanced,
            y + 0.11,
            xerr=balanced_error,
            fmt="s",
            color="#D55E00",
            capsize=2,
            linewidth=1,
            label="Balanced AUPRC",
            zorder=3,
        )
        axis.set_title(title, loc="left")
        axis.set_xlim(0.45, 0.80)
        axis.set_xlabel("Performance")
        axis.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    axes[0].set_yticks(y, [DISPLAY_NAMES[method] for method in IPF_METHODS])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=2, frameon=False)
    figure.suptitle("IPF 022I literature gold-standard benchmark", x=0.08, ha="left")
    figure.subplots_adjust(bottom=0.18, left=0.22, right=0.98, top=0.84, wspace=0.12)
    _save_figure(figure, output, "ipf_022i_performance")


def _ipf_pr_plot(results_root: Path, ipf: pd.DataFrame, output: Path) -> None:
    score_path = (
        results_root
        / "ipf/022I/native/evaluation_paper_base_full/materialized_scores.parquet"
    )
    scores = pd.read_parquet(score_path)
    selected_methods = (
        "CRYCHIC availability-state",
        "CellPhoneDB p-value",
        "CellChat p-value",
        "LIANA specificity consensus",
    )
    figure, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    baseline = float(scores["is_positive"].mean())
    axis.axhline(
        baseline,
        color="#777777",
        linestyle="--",
        linewidth=1,
        label=f"Class prevalence ({baseline:.3f})",
    )
    point = ipf.loc[ipf["arm"].eq("native")].set_index("method")
    for method in selected_methods:
        method_scores = scores.loc[scores["method"].eq(method)]
        precision, recall, _ = precision_recall_curve(
            method_scores["is_positive"].astype(int), method_scores["score"]
        )
        axis.step(
            recall,
            precision,
            where="post",
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=(
                f"{DISPLAY_NAMES[method]} "
                f"(AP={float(str(point.loc[method, 'average_precision'])):.3f})"
            ),
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("IPF 022I: native-resource STLR ranking", loc="left")
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    axis.grid(color="#E5E5E5", linewidth=0.7)
    _save_figure(figure, output, "ipf_022i_precision_recall")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.select_dtypes(include="number"):
        column_name = str(column)
        if column_name == "datasets_evaluable" or column_name.endswith("_edges"):
            selected[column] = selected[column].map(
                lambda value: "NA" if pd.isna(value) else str(int(value))
            )
        else:
            selected[column] = selected[column].map(
                lambda value: "NA" if pd.isna(value) else f"{float(value):.3f}"
            )
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join((header, rule, *rows))


def _write_report(
    output: Path,
    cite: pd.DataFrame,
    cite_native: pd.DataFrame,
    cite_common: pd.DataFrame,
    ipf: pd.DataFrame,
) -> None:
    native_selected = cite_native.loc[
        cite_native["method"].isin(CITE_NATIVE_METHODS)
    ].copy()
    native_selected["method"] = native_selected["method"].map(DISPLAY_NAMES)
    common_selected = cite_common.loc[
        cite_common["method"].isin(CITE_COMMON_METHODS)
    ].copy()
    common_selected["method"] = common_selected["method"].map(DISPLAY_NAMES)
    ipf_selected = ipf.loc[
        ipf["arm"].isin(["native", "H-common_representable"])
        & ipf["method"].isin(IPF_METHODS),
        [
            "arm_display",
            "method",
            "auroc",
            "average_precision",
            "balanced_auprc",
            "raw_returned_positive_edges",
            "truth_positive_edges",
        ],
    ].copy()
    ipf_selected["method"] = ipf_selected["method"].map(DISPLAY_NAMES)
    report = f"""# IPF 人工金标准与 CITE-seq 蛋白金标准真实数据复现

生成日期：{date.today().isoformat()}

## 结论摘要

- 本轮使用 4 个真实人 CITE-seq 数据集和 IPF GSE136831 样本 022I，不含训练集/验证集；CRYCHIC 为确定性非深度学习方法。
- CITE-seq native 独立资源臂中，CRYCHIC availability 的 4 数据集平均 AUROC 为 {float(cite_native.loc[cite_native["method"].eq("CRYCHIC availability-state"), "auroc_mean"].iloc[0]):.3f}，低于 SingleCellSignalR 的 {float(cite_native.loc[cite_native["method"].eq("SingleCellSignalR LRscore"), "auroc_mean"].iloc[0]):.3f}；CRYCHIC 没有在该主平均指标上获胜。
- CRYCHIC native CITE-seq 的最差数据集 AUROC 为 {float(cite_native.loc[cite_native["method"].eq("CRYCHIC availability-state"), "auroc_min"].iloc[0]):.3f}，是本轮所有方法中最高的最差值，显示出较好的跨数据集稳定性。
- 固定 H-common CITE-seq 臂中，CRYCHIC 平均 AUROC/AUPRC(1:1 平衡负样本) 为 {float(cite_common.loc[cite_common["method"].eq("CRYCHIC availability-state"), "auroc_mean"].iloc[0]):.3f}/{float(cite_common.loc[cite_common["method"].eq("CRYCHIC availability-state"), "balanced_auprc_mean"].iloc[0]):.3f}；所有方法在每个数据集使用相同 U/P/N，但其余多种方法在小型共同资源上出现大量并列分数，不能把差距解释为普遍算法优势。
- IPF H-common 可覆盖集合包含 960 个 STLR、53 个去重阳性。CRYCHIC AUROC/平衡 AUPRC 为 {float(ipf.loc[(ipf["arm"].eq("H-common_representable")) & ipf["method"].eq("CRYCHIC availability-state"), "auroc"].iloc[0]):.3f}/{float(ipf.loc[(ipf["arm"].eq("H-common_representable")) & ipf["method"].eq("CRYCHIC availability-state"), "balanced_auprc"].iloc[0]):.3f}，优于该臂最佳外部结果的 0.625/0.620。
- IPF native 独立资源臂中，CRYCHIC AUROC/平衡 AUPRC 为 {float(ipf.loc[(ipf["arm"].eq("native")) & ipf["method"].eq("CRYCHIC availability-state"), "auroc"].iloc[0]):.3f}/{float(ipf.loc[(ipf["arm"].eq("native")) & ipf["method"].eq("CRYCHIC availability-state"), "balanced_auprc"].iloc[0]):.3f}；下一名约为 0.610/0.605。该比较同时混合了资源差异，因此以 H-common 结果作为更直接的算法对照。
- IPF H-common 的 edge-bootstrap/负样本采样区间较宽且方法间重叠；上述结论是点估计优势，不是独立生物学重复支持的显著性结论。

## CITE-seq 蛋白金标准

复现 Dimitrov et al. 2022：每个 ADT 在细胞簇间做样本标准差 z-score，`z >= 1.645` 为阳性；AUPRC 按论文进行 100 次 1:1 负样本下采样。四个数据集分别为 5k PBMC、5k PBMC NextGem、10k PBMC 和 10k MALT，共 27,051 个细胞。CRYCHIC 在此处仅评估单样本静态 `availability_state`，不是完整差异 `comm_strength`。

### Native 资源汇总

{_markdown_table(native_selected, ["method", "datasets_evaluable", "auroc_mean", "auroc_min", "auroc_max", "balanced_auprc_mean"])}

### H-common 资源汇总

{_markdown_table(common_selected, ["method", "datasets_evaluable", "auroc_mean", "auroc_min", "auroc_max", "balanced_auprc_mean"])}

Native 臂按论文使用各方法自己的返回集合，因此覆盖集合和 estimand 不同。H-common 横向比较冻结 638 个简单 LR，并为所有方法构造完全相同的 sender-receiver-LR truth universe；未返回边排在已返回边之后，不作为数值零。原论文 independent-universe 结果仍保存在 `citeseq_all_metrics.tsv` 的 `H-common-independent` 行。CellChat/CellPhoneDB composite 在 PBMC/MALT 的显著性过滤后返回 0 条可评估边，固定 universe 后为并列末位。多个 LIANA component 也完全并列，AUROC=0.5，这会放大 CRYCHIC 连续 availability 分数的表面优势。

## IPF 人工金标准

复现 Xie et al. 2023。上游 253 行包含 3 个重复 STLR，规范化后为 250 个唯一阳性、37 个完整 LR、8 类细胞。论文的基础伪阴性空间是 `8 x 8 x 37 = 2,368`，不是 ligand 与 receptor 分开组合。缺失预测按论文的操作定义排在已返回预测之后；未标注项不是实验阴性。

{_markdown_table(ipf_selected, ["arm_display", "method", "auroc", "average_precision", "balanced_auprc", "raw_returned_positive_edges", "truth_positive_edges"])}

论文报告 022I CellPhoneDB ST 模型 PR AUC 0.83，但归档 notebook 没有生成该曲线的代码或最终排序文件。本轮报告的是去重 STLR 固定基础空间，不能把当前 AP 与 0.83 视为同一 estimand。原论文方法并集的 100,352 条 universe 也无法从仓库完整重建。

## 生物学与方法学解释

IPF gold 包含 CCL2-CCR2、PDGF-PDGFR、TGFB-TGFBR、VEGFA-FLT1 等已知纤维化相关通讯。CRYCHIC native 在返回 211/250 个 gold 阳性的情况下获得较高排序指标，而广覆盖 LIANA components 返回 228/250，说明点估计差异不是简单来自返回更多阳性；更可能来自配体和受体可用性联合排序。这个结果仍是 022I 单样本对疾病级人工 gold 的一致性，不是患者级因果验证。

CITE-seq gold 只验证 receiver-receptor 蛋白特异性，却在评分时复制到不同 sender/ligand edge；它不验证发送细胞、配体作用或下游转录反应。availability 分数直接包含 receptor RNA availability，与该终点存在结构性对齐。这不是标签泄漏，但会使 benchmark 更匹配 CRYCHIC 静态模块；不能据此推断完整 CRYCHIC 在差异通讯、发送者归因或下游响应上优于所有方法。

NicheNet 没有强行纳入这两张主表：其输出是 ligand-to-target regulatory potential/ligand activity，需要目标基因程序，并非直接的 receptor/STLR 连续分数。把它映射到 CITE receptor 标签或 022I 静态 STLR 会改变任务定义。现有多组别 benchmark 中的 NicheNet 结果应单独保留。

## 图表

- `figures/citeseq_native_auroc.*`：4 个 CITE-seq 数据集 native AUROC。
- `figures/citeseq_hcommon_auroc.*`：H-common AUROC；灰色 NA 为不可估计。
- `figures/ipf_022i_performance.*`：IPF native 与 H-common 可覆盖集合的 AUROC/平衡 AUPRC。
- `figures/ipf_022i_precision_recall.*`：IPF native 固定 STLR 空间 PR 曲线。

## 限制

1. IPF 未标注项是开放世界伪阴性；特异度和 MCC 不等于真实生物阴性识别。
2. 022I 只有一个样本，不能用于 CRYCHIC 多组别差异推断。
3. 描述性 edge bootstrap 不是生物学重复推断；CI 不应解释为患者间置信区间。
4. CITE-seq 各 native 方法覆盖集合不同，独立 universe 的 AUROC 不完全可交换。
5. H-common 只含简单 LR，覆盖 IPF 250 个去重阳性中的 53 个；full-base 结果同时受资源覆盖影响。
6. 本轮结论支持“部分指标和特定任务有优势”，不支持“全面优于 CellChat/CellPhoneDB/LIANA”。
7. 阈值指标采用 top-k tie-expanded 规则；当大量缺失边并列时会选入超过 k 条边，主结论因此只使用 AUROC/AP。
8. IPF 按论文把复合物展开为单亚基并给同分；CRYCHIC native 的 211 个返回阳性中有 134 个依赖该宽松 credit，LIANA 为 150/228。
9. IPF 仅使用可重建的 2,368 条 gold-derived base；当前方法在其外的预测未加入 universe，不能称为原论文 100,352 条方法并集的精确复现。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def _write_manifest(output: Path, inputs: list[Path], generated: list[Path]) -> None:
    payload: dict[str, Any] = {
        "schema_version": "crychic-real-literature-gold-report-v1",
        "generated_on": date.today().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "pandas": _package_version("pandas"),
            "numpy": _package_version("numpy"),
            "matplotlib": _package_version("matplotlib"),
            "scikit_learn": _package_version("scikit-learn"),
        },
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(inputs))
        ],
        "outputs": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(generated))
            if path.is_file()
        ],
        "interpretation_boundary": (
            "Single-sample static availability benchmark; not full differential "
            "CRYCHIC and not biological-replicate inference."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workspace = repo_root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=workspace / "benchmark_work/literature_reproduction/results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            workspace
            / "benchmark_work/literature_reproduction/reports/real_gold_standards"
        ),
    )
    args = parser.parse_args()
    results_root = args.results_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib(repo_root / "benchmarks/report/publication.mplstyle")

    cite = load_citeseq(results_root)
    ipf = load_ipf(results_root)
    cite_native = _summarize_cite(cite, "native")
    cite_common = _summarize_cite(cite, "H-common")
    cite.to_csv(tables / "citeseq_all_metrics.tsv", sep="\t", index=False)
    cite_native.to_csv(tables / "citeseq_native_summary.tsv", sep="\t", index=False)
    cite_common.to_csv(tables / "citeseq_hcommon_summary.tsv", sep="\t", index=False)
    ipf.to_csv(tables / "ipf_022i_all_metrics.tsv", sep="\t", index=False)

    _heatmap(
        cite,
        arm="native",
        methods=CITE_NATIVE_METHODS,
        output=figures,
        stem="citeseq_native_auroc",
        title="CITE-seq receptor-protein benchmark: native resources",
    )
    _heatmap(
        cite,
        arm="H-common",
        methods=CITE_COMMON_METHODS,
        output=figures,
        stem="citeseq_hcommon_auroc",
        title="CITE-seq receptor-protein benchmark: fixed H-common universe",
    )
    _ipf_metric_plot(ipf, figures)
    _ipf_pr_plot(results_root, ipf, figures)
    _write_report(output, cite, cite_native, cite_common, ipf)

    inputs = [
        Path(value)
        for value in pd.concat((cite["evaluation_dir"], ipf["evaluation_dir"]))
        .drop_duplicates()
        .tolist()
        for value in (
            str(Path(value) / "point_estimates.tsv"),
            str(Path(value) / "negative_sampling_summary.tsv"),
            str(Path(value) / "bootstrap_summary.tsv"),
        )
    ]
    inputs.append(
        results_root
        / "ipf/022I/native/evaluation_paper_base_full/materialized_scores.parquet"
    )
    generated = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    _write_manifest(output, inputs, generated)


if __name__ == "__main__":
    main()
