# IPF 全队列 benchmark 报告

## 完成状态

Xie 等人公开处理后的 IPF 队列已全量运行完成。输入来自 Zenodo 记录
`10.5281/zenodo.6497091`，覆盖 4 个 GEO study、56 个患者样本和 138,248 个细胞：

| Study | 患者数 |
| --- | ---: |
| GSE122960 | 4 |
| GSE128033 | 8 |
| GSE135893 | 12 |
| GSE136831 | 32 |

正式队列为 `full_task_ledger.tsv`，168/168 个任务完成、0 个失败：每个样本各运行
LIANA H-common、CRYCHIC availability-state H-common 和统一评估。不可用的外部独立算法未临时安装，
本报告仅比较本环境中可复现的 CRYCHIC 与 LIANA/CellChat 导出的 10 个评分组件。

## 主要结果

主汇总采用患者等权，而不是 study 等权；因此 GSE136831 的 32 个患者占 56 个患者中的
32 份权重。统一评估空间含 960 条可表示边，其中 53 条为文献金标准阳性、907 条为未标注补集。

| 方法/组件 | AUROC | AP | balanced AUPRC |
| --- | ---: | ---: | ---: |
| CRYCHIC availability-state | 0.5855 | **0.1777** | **0.6517** |
| CellPhoneDB composite | **0.5881** | 0.1210 | 0.5830 |
| SingleCellSignalR LRscore | 0.5815 | 0.1291 | 0.5917 |
| Connectome specificity | 0.5462 | 0.1423 | 0.5932 |

CRYCHIC 的 AUROC 点估计为第二，AP 和 balanced AUPRC 点估计最高。按 study 分层后，
CRYCHIC AUROC 分别为 0.7690、0.7821、0.5310、0.5338，说明跨研究异质性明显。

患者 bootstrap 在每个 study 内分层重采样 2,000 次。CRYCHIC 的描述性 95% 区间为：
AUROC 0.5637-0.6087、AP 0.1604-0.1963、balanced AUPRC 0.6346-0.6696。
CellPhoneDB composite 的 AUROC 区间为 0.5768-0.5999，与 CRYCHIC 重叠。
这些区间描述患者异质性，不是成对方法优越性检验，不能据此声称统计显著优于其他方法。

## 运行资源

H5AD 准备使用 12 个 worker，耗时 18.86 秒，`GNU time -v` 最大 RSS 930,364 kB。
正式评分使用最多 24 个并发 worker、每任务 1 线程，软件内存阈值为 60%、硬阈值为 70%；
168 个任务总墙钟时间 114.94 秒。任务账本记录的最大单任务进程树 RSS 为 2,324,107,264 字节，
任务启动时最大整机内存占比为 0.2160，未触发内存失败。运行期间约 60/251 GiB 的整机占用仅为
非连续抽查值，不应解释为连续监控得到的峰值。

## 解释边界

这是 IPF 病例样本针对疾病级、静态配体-受体金标准的排序 benchmark。金标准不是患者特异，
未标注补集也不是实验确认的阴性，因此 specificity 和 MCC 使用的是操作性伪阴性。
CRYCHIC 此处运行的是单样本 availability-state 分支，不是多组别差异通讯分析；本结果也不构成
下游通路、因果方向或功能实验验证。

## Published artifacts

- [患者等权汇总](cohort_patient_equal.tsv)
- [Study 分层汇总](metrics_by_study.tsv)
- [患者 bootstrap 汇总](patient_bootstrap.tsv)
- [清理后的样本级结果](sample_metrics.tsv)
- [发布证明](manifest.json)
