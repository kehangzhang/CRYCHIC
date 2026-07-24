"""Add gap-completion evidence to the literature benchmark notebook and run it."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

SECTION_TAG = "suggest-v5-gap-completion-20260724"


def _markdown(source: str) -> Any:
    return nbformat.v4.new_markdown_cell(
        source.strip(), metadata={"tags": [SECTION_TAG]}
    )


def _code(source: str) -> Any:
    body = source.strip()
    if not body.startswith("%%time\n"):
        body = f"%%time\n{body}"
    return nbformat.v4.new_code_cell(body, metadata={"tags": [SECTION_TAG]})


def _ensure_timing(notebook: Any) -> None:
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if not source.startswith("%%time\n"):
            cell.source = f"%%time\n{source}"


def _gap_cells() -> list[Any]:
    return [
        _markdown(
            """
## 7. 最新源码单样本逐行一致性

这里读取的是用当前源码重新计算后的独立结果目录，而不是只重新绘制旧表。
可共同定义的 `availability_state` 做逐行键和值比较；需要跨条件训练信息的
RC12/RC14 保留为 NE。
"""
        ),
        _code(
            """
from IPython.display import display

gap_candidates = []
if os.environ.get("CRYCHIC_GAP_RESULT_ROOT"):
    gap_candidates.append(
        Path(os.environ["CRYCHIC_GAP_RESULT_ROOT"]).expanduser()
    )
for ancestor in (REPO_ROOT, *REPO_ROOT.parents):
    gap_candidates.append(
        ancestor
        / "benchmark_work"
        / "suggest_v5_gap_completion_20260724"
    )
GAP_ROOT = next(
    path.resolve()
    for path in gap_candidates
    if (path / "single_sample_consistency/manifest.json").is_file()
)

single_root = GAP_ROOT / "single_sample_consistency"
single_manifest = read_json(single_root / "manifest.json")
assert single_manifest["status"] == "complete_with_declared_NE"
assert single_manifest["source_repository"]["dirty"] is False
for relative, record in single_manifest["artifacts"].items():
    assert sha256_file(single_root / relative) == record["sha256"]

single_consistency = pd.read_csv(
    single_root / "track_summary.tsv", sep="\t"
)
score_head_coverage = pd.read_csv(
    single_root / "score_head_coverage.tsv", sep="\t"
)
assert int(single_consistency["datasets"].sum()) == 64
assert int(single_consistency["old_rows"].sum()) == 4_757_604
assert single_consistency["exactly_consistent"].astype(bool).all()
assert single_consistency[
    ["unmatched_rows", "numeric_mismatches", "text_mismatches"]
].fillna(0).to_numpy().sum() == 0
assert set(score_head_coverage.loc[
    score_head_coverage["status"].eq("NE"), "score_head"
]) == {"RC12_sender_response_detection", "RC14_differential_DES"}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
axes[0].bar(
    single_consistency["track"],
    single_consistency["new_rows"],
    color=[CRYCHIC_COLOR, ACCENT_COLOR, "#548C2F"],
)
axes[0].set_yscale("log")
axes[0].set_ylabel("Recomputed rows (log scale)")
axes[0].set_title("Current-source single-sample rerun")
for index, row in single_consistency.reset_index(drop=True).iterrows():
    axes[0].text(
        index,
        row["new_rows"] * 1.12,
        f"{int(row['datasets'])} datasets",
        ha="center",
        fontsize=8,
    )

mismatch_columns = [
    "unmatched_rows", "numeric_mismatches", "text_mismatches"
]
mismatch_values = single_consistency[mismatch_columns].fillna(0).sum()
axes[1].barh(mismatch_values.index, mismatch_values.values, color=EXTERNAL_COLOR)
axes[1].set_xlim(0, 1)
axes[1].set_xlabel("Count")
axes[1].set_title("Exact comparison discrepancies")
for index, value in enumerate(mismatch_values):
    axes[1].text(0.02, index, str(int(value)), va="center")
single_figure = OUTPUT_ROOT / "latest_single_sample_consistency.png"
fig.savefig(single_figure, bbox_inches="tight")
plt.show()

display(single_consistency)
display(score_head_coverage)
"""
        ),
        _markdown(
            """
逐行一致性覆盖 CITE-seq 7 个数据集、IPF 56 位患者和 HER2 CytoSig，
共 64 个数据集、4,757,604 行；未匹配行、数值差异、文本差异和最大绝对
数值差均为 0。因此本 notebook 前面展示的同一单样本 estimand 的 AUROC、
AP 和排名保持不变。TNBC prepared H5AD 未保留，故当前重跑为 NE，未放入
一致性分母。

RC12 需要训练折生成的 sender assignment 和跨条件 receiver program；
RC14 需要多条件差异统计及空间 pair prior。它们不是单样本 score head，
不能用 receptor expression 临时替代。
"""
        ),
        _markdown(
            """
## 8. suggest_v5 剩余端点补齐

下表汇总实际运行后的状态。`complete_diagnostic` 与
`diagnostic_nonconverged` 不等同于正式推断完成，NE 也不计为零分。
"""
        ),
        _code(
            """
gap_runs = {
    "Tensor extended": GAP_ROOT / "tensor_extended",
    "STACCato calibration": GAP_ROOT / "staccato",
    "DCST fixed binary": GAP_ROOT / "dcst_binary",
    "scACCorDiON PDAC": GAP_ROOT / "scaccordion_pdac_v2",
    "scACCorDiON AKI": GAP_ROOT / "scaccordion_aki_v2",
    "scACCorDiON RCC fine": GAP_ROOT / "scaccordion_rcc_v2",
    "scACCorDiON RCC coarse": GAP_ROOT / "scaccordion_rcc_coarse_v2",
    "Cross-cohort ranks": GAP_ROOT / "scaccordion_crosscohort_v2",
    "TCGA-PAAD survival": GAP_ROOT / "tcga_paad_survival",
}
gap_inventory = []
for label, directory in gap_runs.items():
    manifest = read_json(directory / "manifest.json")
    source = manifest.get("source_repository", {})
    assert source.get("dirty") is False
    gap_inventory.append(
        {
            "endpoint": label,
            "status": manifest["status"],
            "source_commit": source.get("commit", "")[:12],
        }
    )
gap_inventory = pd.DataFrame(gap_inventory)

coverage_frames = []
for label, relative in (
    ("Tensor", "tensor_extended/endpoint_coverage.tsv"),
    ("STACCato", "staccato/endpoint_coverage.tsv"),
    ("PDAC", "scaccordion_pdac_v2/endpoint_coverage.tsv"),
    ("AKI", "scaccordion_aki_v2/endpoint_coverage.tsv"),
    ("RCC fine", "scaccordion_rcc_v2/endpoint_coverage.tsv"),
    ("RCC coarse", "scaccordion_rcc_coarse_v2/endpoint_coverage.tsv"),
):
    frame = pd.read_csv(GAP_ROOT / relative, sep="\t").fillna("")
    frame.insert(0, "benchmark", label)
    coverage_frames.append(frame)
gap_coverage = pd.concat(coverage_frames, ignore_index=True)

dcst_binary = pd.read_csv(
    GAP_ROOT / "dcst_binary/aggregate_metrics.tsv", sep="\t"
)
dcst_binary_summary = (
    dcst_binary.groupby(["sweep", "method"], observed=True)[
        ["auroc_mean", "auprc_mean", "sensitivity_q005_mean"]
    ]
    .mean()
    .reset_index()
)
cross_ranks = pd.read_csv(
    GAP_ROOT / "scaccordion_crosscohort_v2/friedman_average_ranks.tsv",
    sep="\t",
)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
axes[0].bar(
    dcst_binary_summary["sweep"],
    dcst_binary_summary["auroc_mean"],
    color=CRYCHIC_COLOR,
)
axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=0.9)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Mean AUROC")
axes[0].set_title("CRYCHIC fixed binary projection (tau=0.25)")

rank_plot = cross_ranks.sort_values("average_rank", ascending=False)
rank_colors = np.where(
    rank_plot["method"].eq("crychic_canonical_event_spearman"),
    CRYCHIC_COLOR,
    EXTERNAL_COLOR,
)
axes[1].barh(rank_plot["method"], rank_plot["average_rank"], color=rank_colors)
axes[1].set_xlabel("Average rank (lower is better)")
axes[1].set_title("Patient graphs: 3-cohort descriptive rank")
gap_figure = OUTPUT_ROOT / "suggest_v5_gap_completion.png"
fig.savefig(gap_figure, bbox_inches="tight")
plt.show()

display(gap_inventory)
display(gap_coverage)
display(dcst_binary_summary.round(4))
display(cross_ranks.round(4))
"""
        ),
        _code(
            """
patient_rows = []
for cohort, relative in (
    ("PDAC", "scaccordion_pdac_v2/summary_metrics.tsv"),
    ("AKI", "scaccordion_aki_v2/summary_metrics.tsv"),
    ("RCC fine", "scaccordion_rcc_v2/summary_metrics.tsv"),
    ("RCC coarse", "scaccordion_rcc_coarse_v2/summary_metrics.tsv"),
):
    frame = pd.read_csv(GAP_ROOT / relative, sep="\t")
    frame = frame.loc[
        frame["clustering_backend"].eq("kmedoids")
        & frame["method"].isin(
            [
                "crychic_canonical_event_spearman",
                "crychic_native_availability_hyperedge_spearman",
            ]
        )
    ].copy()
    frame.insert(0, "cohort", cohort)
    patient_rows.append(frame)
patient_crychic = pd.concat(patient_rows, ignore_index=True)

cox = pd.read_csv(
    GAP_ROOT / "tcga_paad_survival/cox_coefficients.tsv", sep="\t"
)
survival_hits = cox.loc[
    cox["mode"].eq("ligand_receptor_geometric_mean")
    & cox["is_interaction_term"].astype(bool)
    & cox["q.value.within.mode"].lt(0.05),
    ["term", "estimate", "p.value", "q.value.within.mode"],
].rename(columns={"estimate": "hazard_ratio"})

staccato_coverage = pd.read_csv(
    GAP_ROOT / "staccato/endpoint_coverage.tsv", sep="\t"
)
final_consistency = pd.DataFrame(
    [
        {
            "claim": "Latest estimable single-sample score",
            "result": "64/64 exact; 4,757,604/4,757,604 rows identical",
        },
        {
            "claim": "Existing single-sample AUROC/AP/ranks",
            "result": "unchanged because all score rows are identical",
        },
        {
            "claim": "RC12/RC14 on one sample",
            "result": "NE: requires between-condition training/statistics",
        },
        {
            "claim": "Patient-graph rank claim",
            "result": "CRYCHIC-compatible Spearman 1/6 descriptively; Friedman p=0.133",
        },
    ]
)
display(patient_crychic[
    [
        "cohort", "method", "ari_at_protocol_selected_true_k",
        "maximum_ari_over_grid", "numerical_status",
    ]
])
display(survival_hits.round(4))
display(staccato_coverage)
display(final_consistency)
"""
        ),
        _markdown(
            """
PDAC cell downsampling remains NE because the released archive contains patient
graphs and metadata but not the author-processed cell-level H5AD. STACCato
full-pipeline p/q remains NE because the diagnostic residual bootstrap does not
resample upstream score generation. RCC-coarse correlation-OT k-barycenter is
retained as `diagnostic_nonconverged`. These boundaries prevent unavailable or
numerically invalid evidence from being converted into a zero or a favorable
score.
"""
        ),
    ]


def update_notebook(notebook: Any) -> Any:
    notebook.cells = [
        cell
        for cell in notebook.cells
        if SECTION_TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    _ensure_timing(notebook)
    insertion = next(
        (
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "markdown"
            and cell.source.startswith("## 解释边界与重跑范围")
        ),
        len(notebook.cells),
    )
    notebook.cells[insertion:insertion] = _gap_cells()
    return notebook


def run(
    notebook_path: Path,
    *,
    execute: bool,
    repo_root: Path,
    timeout_seconds: int,
) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook = update_notebook(notebook)
    if execute:
        previous_pythonpath = os.environ.get("PYTHONPATH")
        path_entries = [str((repo_root / "src").resolve()), str(repo_root.resolve())]
        if previous_pythonpath:
            path_entries.append(previous_pythonpath)
        os.environ["PYTHONPATH"] = os.pathsep.join(path_entries)
        try:
            client = NotebookClient(
                notebook,
                timeout=timeout_seconds,
                kernel_name="python3",
                resources={"metadata": {"path": str(repo_root.resolve())}},
            )
            client.execute()
        finally:
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run(
        args.notebook,
        execute=args.execute,
        repo_root=args.repo_root,
        timeout_seconds=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
