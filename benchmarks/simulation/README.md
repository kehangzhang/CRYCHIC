# V0.1 synthetic controls

Run the complete smoke benchmark from the repository root:

```bash
uv run --extra benchmark python -m benchmarks.simulation.run_negative_controls \
  --output-dir benchmark_work/synthetic_v01 \
  --seed 20260712
```

The runner executes one active positive control and six required negative-control
scenarios through input validation, sample-level pseudobulk, paired receiver
response, LR availability, and a toy target-prior attribution. It writes:

- `scenario_metrics.csv`: compact component diagnostics for every scenario;
- `edge_metrics.csv`: synthetic edge truth and exploratory recovery scores;
- `checks.csv`: pre-specified mechanism checks and thresholds;
- `summary.json`: reproducibility metadata and truth-scoped aggregate metrics.

AUROC and average precision use synthetic edge labels only. The runner does not
calculate real-data truth metrics, p-values, q-values, FDR, or a communication
probability. The recovery score is a smoke-test geometric mean of positive state
change, downstream response, and attribution fit; it is not calibrated.

## External-method synthetic suite

Export the normalized/count-layer inputs and construct the five-edge frozen
H-common resource before running the external adapters. Then execute all seven
scenarios with isolated method environments:

```bash
uv run python -m benchmarks.simulation.run_multimethod_controls \
  "$SYNTHETIC_ROOT/manifest.json" "$BENCHMARK_ROOT/synthetic_multimethod" \
  --database-root "$DATABASE_ROOT" \
  --harmonized-resource "$SYNTHETIC_RESOURCE/harmonized_lr.tsv" \
  --harmonized-manifest "$SYNTHETIC_RESOURCE/manifest.json" \
  --cellchat-environment "$CELLCHAT_ENV" \
  --cellphonedb-python "$CPDB_ENV/bin/python" \
  --liana-python "$LIANA_ENV/bin/python"
```

CellChat, CellPhoneDB, LIANA, and CRYCHIC use the exact five-edge H-common
fixture. New CRYCHIC runs use the versioned
`synthetic_multimethod_nocap_v02.json` paired design, the frozen NicheNet target
prior, state scoring, `min_cells=10`,
`min_subjects_per_context=4`, and no data-driven top-k interaction cap. The
H-common adapter rejects non-null `max_interactions` so the comparison universe
cannot be selected from evaluation data. NicheNet runs its separate,
source-agnostic ligand-to-target-program Track B and auto-normalizes integer
count input to library-size 10,000 plus `log1p`; its adapter manifest records
the resolved transformation.

The suite writes one independent method/scenario directory, sampled wall time
and process-tree RSS, stdout/stderr logs, `runs.tsv`, and a checksum-pinned
`suite_manifest.json`. A nonzero adapter exit or any sample-level method failure
has suite status `failed` and missing measurements; failures are never converted
to zero scores. Completed runs resume only after output checksums verify.
`--overwrite --method METHOD` reruns only the selected method while preserving
other completed outputs. `track_a_edge_truth.tsv` labels the complete 45-edge
cell-pair/LR universe; only the active Sender-to-Receiver CXCL10-CXCR3 edge is a
positive. All-negative mechanisms carry an explicit single-class reason.
`track_b_scenario_truth.tsv` stores receiver-response expectations separately
and is never passed off as LR edge truth.

Evaluate paired differential truth after all external runs complete:

```bash
uv run python -m benchmarks.metrics.track_a_differential_truth \
  benchmarks/configs/track_a_differential_truth_v01.json \
  benchmark_work/multicondition_v01/track_a_differential_truth
```

The evaluator uses direction-oriented native paired score changes as its
primary estimand and paired rank-strength changes as a sensitivity analysis.
Native missing results are not zero-filled. It writes edge-level effects, a
compact 7-scenario by 4-method by 2-estimand summary, finalizer-compatible
Track-A records, and `simulation_records.tsv` with the existing Track-B records
appended unchanged. All-zero truth scenarios remain explicitly single-class
and do not receive AUROC, average precision, FDR, or type-I-error estimates.

## Post-benchmark downstream-support candidate

The gate-aware downstream attribution-support v2 is opt-in and does not alter
the frozen benchmark or finalizer inputs. Run its pre-specified unseen-seed
CRYCHIC-only holdout audit with:

```bash
uv run python -m benchmarks.simulation.evaluate_downstream_support_candidate \
  benchmarks/configs/downstream_support_candidate_holdout_nocap_v02.json \
  benchmark_work/multicondition_v02/downstream_support_candidate_nocap_v02_holdout
```

The runner generates seed `20260819` holdouts, executes complete v1 and v2
pipelines for active, ligand-only, target-only and receptor-knockout scenarios,
and verifies that every adapter and v2 result manifest retains the explicit
support method. Its report is labeled post-benchmark development and is never
merged into `simulation_records.tsv`. Passing the decision rule qualifies v2
for a future default-change evaluation; it does not switch the current v1
default or authorize a real-data rerun.

## G1.5 mechanism-specificity gate

The published v2 development and independent-holdout campaigns are historical
source locks. Reproduce them from commit `5e8b4b8`, not with the live
sample-keyed generator:

```bash
uv run --extra benchmark python -m benchmarks.simulation.run_mechanism_specificity \
  --phase development \
  --output-dir benchmark_work/g1_5_v2/development
uv run --extra benchmark python -m benchmarks.simulation.run_mechanism_specificity \
  --phase independent_holdout \
  --output-dir benchmark_work/g1_5_v2/independent_holdout
```

The runner writes atomic evidence, generation provenance, evaluator metrics,
and structural-zero versus positive-denominator counts globally and by
scenario. The live runner uses `mechanism_specificity_v3.json`, a distinct v3
seed namespace, and the default development output
`benchmark_work/g1_5_sample_keyed_development_v3/`. It refuses to republish the
already inspected historical holdout; the v3 holdout namespace is
nonpublication test-only and cannot be promoted to an independent holdout.
Published historical lightweight results and exact artifact hashes are
in [`benchmarks/results/g1_5_v2_summary.json`](../results/g1_5_v2_summary.json).
The synthetic gate never switches the public default by itself.

## Suggest-next2 v7 integrated protocol

The integrated v7 campaign is frozen in
[`suggest_next2_v7_benchmark_v1.json`](../configs/suggest_next2_v7_benchmark_v1.json).
It separates DGP families, not only seeds: six families are development-only,
22 are locked family holdouts, and four are null-calibration families. The
locked set includes receptor-only, program-only, inhibitory, generic-state,
batch/composition null, multi-sender, topology-corruption, structural-absence,
and missingness mechanisms that are absent from the old component-specific
fixtures.

Validate the protocol and write a checksum-bound plan before generating data:

```bash
uv run --extra benchmark python -m benchmarks.simulation.v7_protocol \
  --phase smoke \
  --profile score_primary \
  --output-dir benchmark_work/suggest_next2_v7/plans/smoke_score_primary
```

Use `--profile score_primary` for `G0--G5` under the same design-aware `I1`
engine. Use `--profile inference_crossover` for the same frozen `G3` table
under `I0--I2`. This prevents simultaneous score-generator and inference
changes. `--maximum-replicates` may reduce a phase only for local diagnostics;
it cannot exceed or redefine the frozen tier.

The full registered sizes are 800 development, 5,750 locked, and 14,000 null
calibration datasets before method arms. The null tier contains 1,000 complete
pipeline replicates for every registered family/design pair. These counts are
intentional release evidence targets, not claims that those campaigns have
already run. A plan manifest is `planned_not_executed` until a separate runner
has checksum-bound every generated fixture, method result, failure, timing,
and peak process-tree RSS record.

Execute both frozen profiles while fitting each generated dataset only once:

```bash
uv run --extra benchmark python -m benchmarks.simulation.run_v7_campaign \
  --phase smoke \
  --profiles score_primary inference_crossover \
  --output-dir benchmark_work/suggest_next2_v7/smoke
```

The runner first completes one single-core pilot, then derives process-level
parallelism from the pilot peak RSS, host CPU count, and the frozen 80% system
memory ceiling. Every worker is limited to one numerical-library thread, so
process and BLAS parallelism cannot oversubscribe one another. Each dataset
directory contains stage-by-stage JSONL logs, compact Zstandard Parquet tables,
and a manifest that binds the raw fixture, cross-fit, method arms, timings,
process-tree peak RSS, versions, Git state, and output checksums. Completed
datasets resume only after every checksum verifies. `--maximum-replicates` and
`--maximum-datasets` are diagnostic subsets and do not redefine a frozen tier.

`G0` and `G2` are intentionally both executed. Under the frozen formulas they
are constant rescalings whenever the `1e6` CPM clip does not bind; every result
therefore includes an explicit equivalence diagnostic rather than presenting
their matching rank metrics as independent evidence. `G4` combines standardized
M0-parent and signed-program effects only after separate I1 fits, and withholds
an analytic SE because their covariance is unknown. `G5` leaves raw M0 sender
detection unchanged and evaluates M1, M2 coupling, and conditional attribution
as separate heads.
