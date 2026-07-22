# 多组差异通讯优化与 benchmark 总结（2026-07-22）

## 结论

当前通过独立模拟、完整真实队列和安全门的最佳版本是 **CRYCHIC RC9
bounded receiver-program blend**。它在不改变 RC2/RC3 backbone、LIANA call set、
结构缺失语义和 receiver-program 来源的前提下，将冻结的 soft program expert 与
mirror-tail gate expert 以 20% 上限做均值保持的凸组合。

| Dataset | scSeqCommDiff | CRYCHIC RC9 | 之前最佳 CRYCHIC | RC9 增益 | 与 scSeq 差距 |
|---|---:|---:|---:|---:|---:|
| Kuppe | 0.701 | 0.613 | 0.576 (RC3) | +0.0367 | -0.0878 |
| MS | 0.825 | 0.796 | 0.768 (RC4) | +0.0276 | -0.0289 |

数值为完整 8/8 strata 的 median DES。RC9 在 MS 的 mean DES 为 `0.754`，高于
scSeqCommDiff 的 `0.673`，但预注册主排名是 median，因此当前仍不能声称总体超过
scSeqCommDiff。RC9 在两个队列均为完整方法第 2 名，是本轮唯一可支持的默认候选。

## 正式完整排名

### Kuppe multi-sample

| Rank | Method | Median DES | Mean DES | Strata |
|---:|---|---:|---:|---:|
| 1 | scSeqCommDiff native | 0.701 | 0.691 | 8/8 |
| 2 | CRYCHIC RC9 bounded program blend | 0.613 | 0.583 | 8/8 |
| 3 | CRYCHIC RC3 receiver-program soft | 0.576 | 0.563 | 8/8 |
| 4 | CRYCHIC RC4 adaptive program | 0.560 | 0.548 | 8/8 |
| 5 | CRYCHIC RC2 LIANA + signed residual | 0.557 | 0.556 | 8/8 |
| 6 | CRYCHIC RC6 subject hurdle | 0.514 | 0.543 | 8/8 |
| 7 | LIANA+ native | 0.490 | 0.470 | 8/8 |

### MS multi-sample

| Rank | Method | Median DES | Mean DES | Strata |
|---:|---|---:|---:|---:|
| 1 | scSeqCommDiff native | 0.825 | 0.673 | 8/8 |
| 2 | CRYCHIC RC9 bounded program blend | 0.796 | 0.754 | 8/8 |
| 3 | CRYCHIC RC4 adaptive program | 0.768 | 0.736 | 8/8 |
| 4 | CRYCHIC RC3 receiver-program soft | 0.743 | 0.734 | 8/8 |
| 4 | CRYCHIC RC6 subject hurdle | 0.743 | 0.705 | 8/8 |
| 6 | CRYCHIC RC2 LIANA + signed residual | 0.700 | 0.628 | 8/8 |
| incomplete | LIANA+ native | 0.500 | 0.401 | 6/8 |

LIANA+ 的 MS 结果不完整，不进入正式完整排名。所有方法使用相同
ConnectomeDB2020、Figure 3 expected sets、unordered non-self pair universe、
`scoreType=pos`、`gseaParam=1` 和 tie policy。

## 版本迭代

| Iteration | 方向 | 模拟决定 | 真实 benchmark | 最终决定 |
|---|---|---|---|---|
| RC0 | component swap | diagnostic | 0.428 / 0.050 | 定位 one-SE/head 与 representation 双重问题 |
| RC1 | signed soft cardinality | accepted | 0.360 / 0.520 | MS 修复但 Kuppe 回退，拒绝默认替换 |
| RC2 | LIANA baseline + signed hypergraph residual | accepted | 0.557 / 0.700 | 接受为新基线 |
| RC3 | receiver-program soft evidence | accepted | 0.576 / 0.743 | 接受为通用最佳 |
| RC4 | adaptive multiview residual | accepted | 0.560 / 0.768 | MS specialist，不替换 Kuppe 默认 |
| RC5 | two-sided program attenuation | rejected | not run | holdout 主指标失败 |
| RC6 | occurrence + magnitude hurdle | accepted | 0.514 / 0.743 | 真实队列不复现，拒绝替换 |
| RC7 | cross-fitted occurrence SVD program | rejected | not run | 重建 MSE 与差异排序不对齐 |
| RC8 | adaptive hard program gate | rejected | not run | diffuse/isolated safety 失败 |
| RC9 | bounded soft/gate blend | accepted | **0.613 / 0.796** | 当前推荐候选 |
| RC10 | paired rank contrast | rejected | not run | 破坏合法双向重塑 |

斜杠前后分别为 Kuppe/MS median DES。各 iteration 的完整 development、holdout、
manifest、checksum 和真实结果保存在同名结果目录。

## RC9 合同

- 冻结模拟提交：`fc5089a`；真实 runner 提交：`40a30f2`。
- simulation 选择：mirror-tail reduction `0.50`，gate blend `0.20`。
- Kuppe gate：`sign_only`；MS gate：跨两个稳定 edge folds 的 enriched tail，
  threshold `4.86071`。
- Kuppe/MS 分别保持 `3,203/782` 个 RC3 calls，rank order parity 完全一致。
- soft 与 gate expert 在 call set 上分别归一化，最终 blended call mean 均为 1.0。
- missing program evidence 为 neutral；structural absence 始终为 missing，不填零。
- 新 head 仅用于 benchmark ranking；`formal_inference_allowed=false`。

## 未采用的高分路径

以下数值只能用于诊断，不能进入正式 leaderboard：

- RC8 hard gate 的 post-hoc Kuppe/MS 约为 `0.633/0.822`，但正式模拟 diffuse safety
  回退，RC8 被拒绝。
- RC10 的 MS rank contrast sensitivity 可达约 `0.900`，但正式 holdout 对独立和
  asymmetric bidirectional truth 分别回退 `-0.201/-0.457`，不能作为通讯算法。
- subject-pooled ligand/receptor specificity 在最弱 `gamma=0.10` 时已同时降低 Kuppe
  (`0.576 -> 0.535`) 和 MS (`0.743 -> 0.717`)。
- encounter-opportunity 只在强权重下改善 Kuppe并伤害 MS；RC3/RC4 简单混合也几乎
  没有超过各自最佳 expert 的空间。

## 后续优化方向

1. **停止在 Kuppe/MS 上继续选择 pair 公式。** 两个 test cohorts 已被反复用于诊断；
   新候选需要第三个独立 multi-sample development cohort 或 leave-study-out spatial
   calibration。
2. **实现真正的 sample x directed-edge program model。** RC7 只对 occurrence 做
   label-free SVD，不能代表 `suggestions_v4.md` 的 low-rank + sparse signed effect
   模型；下一版应在 outer-training subjects 学 loadings，再对 held-out subject 投影。
3. **升级 signed coefficient solver。** 在成熟 pseudobulk baseline 上实现一次拟合的
   forward/reverse posterior 与 signed fused/TV prior，而不是继续增加 edge multiplier。
4. **补充独立 downstream truth。** RC10 证明 pair scores 本身无法区分共享偏置与合法
   双向重塑；需要 receiver TF/program 或 matched spatial evidence 才能安全缩小剩余差距。
5. **科学输出与 benchmark head 分离。** RC9 可作为 ranking candidate，但正式 effect、
   SE/lfsr/q-value 仍必须来自校准的 subject-level inference，不能由 DES 权重替代。

## 版本位置

- 当前审计分支：`improve/rc10-shared-pair-background-20260722`。
- 推荐 RC9 实现提交：`6433a2e`（包含源码、冻结 runner 与真实结果）。
- RC9 核心：`benchmarks/literature/bounded_program_blend.py`。
- RC9 runner：`benchmarks/literature/run_bounded_program_blend_real.py`。
- RC9 正式结果：`benchmarks/results/bounded_program_blend_rc9_real_20260722/`。

RC9 是目前“性能提升可复现且安全门通过”的终点；RC10 作为阴性审计保留，不应覆盖
RC9 的推荐状态。
