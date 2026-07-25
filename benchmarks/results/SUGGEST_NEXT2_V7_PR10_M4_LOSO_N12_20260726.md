# Suggest-next2 v7 PR10 M4 LOSO n=12 scale check

Status: **paired/repeated M4 LOSO estimability gate passed; formal release not
evaluated**.

## Frozen evidence

- Source commit: `21226276f6d094a190e7c28d8dce0bfb77c1ff98`
  (`dirty=false` in both dataset manifests).
- Runner: `benchmarks/simulation/run_v7_full_refit_campaign.py`.
- Frozen config:
  `benchmarks/configs/suggest_next2_v7_pr10_m4_loso_n12_v1.json`.
- Config SHA-256:
  `34f1f7f0e3775b7bb98acdb53ef7115a1d687257a91f7269d6a3c485d285c2ef`.
- M4 protocol SHA-256:
  `89ca902e09e4828ba910f8641e613b764fed72bfdcc2efcedc7419c670e1aae5`.
- Output:
  `/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/pr10_m4_loso_n12_v1_20260726`.
- Campaign manifest SHA-256:
  `fb81bdcdcdf786d7950ec6d7510c4b78f4dcdce011f78a51074eff3d8126af1c`.
- Every dataset table and compressed plan/record artifact was independently
  rehashed after completion; no checksum mismatch was found.

This preregistered development diagnostic changed only the paired and repeated
subject count from 8 to 12. It retained the global-null DGP, one replicate,
two bootstraps, two legal condition permutations, complete LOSO, the frozen M4
protocol, and full ten-stage refitting per resample. Its primary pass rule was
that every point-observed M4 hypothesis have a complete LOSO distribution.

## Result

| Design | Full refits | Successful | Point-observed M4 | Complete bootstrap | Complete permutation | Complete LOSO | Wall seconds | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paired | 16 | 16 | 35 | 35 | 35 | 35 | 322.62 | 286.5 |
| repeated | 16 | 16 | 105 | 105 | 105 | 105 | 747.54 | 329.9 |
| **Total** | **32** | **32** | **140** | **140** | **140** | **140** | **1,070.16** | **329.9 max** |

Campaign wall time was 1,072.5 seconds (17.88 minutes). The maximum observed
system-memory fraction was 18.36%, below the frozen 80% boundary. All 32 child
records succeeded and both datasets observed all ten required refit stages:
fold split, availability, M0, M1, sender attribution, M2 coupling, held-out
application, design-aware effects, M4, and M5.

The primary endpoint passed for both designs: paired had 35/35 and repeated had
105/105 point-observed M4 hypotheses with 12/12 observed LOSO effects. No
point-observed M4 row had an incomplete LOSO distribution.

## n=8 comparison

| Design | n=8 complete LOSO | n=12 complete LOSO | Total-time ratio | Per-resample ratio | Peak-RSS ratio |
|---|---:|---:|---:|---:|---:|
| paired | 0/35 | 35/35 | 1.658 | 1.243 | 1.029 |
| repeated | 0/105 | 105/105 | 1.768 | 1.326 | 1.074 |

At n=8, LOSO leaves seven complete trajectories and one side of a two-fold
cross-fit receives only three training subjects. That is below the frozen M4
fold-calibration minimum of four, so every paired/repeated M4 LOSO effect was
correctly marked `occurrence_design_not_estimable`. At n=12, LOSO leaves 11
trajectories and the smaller training side contains five. The complete recovery
at that scale supports a minimum-scale preflight rather than an algorithmic
fallback that would change the estimator.

The total runtime increase includes four additional LOSO plans. After dividing
by the number of full refits, n=12 increased time by 24.3% for paired and 32.6%
for repeated, while peak RSS increased by only 2.9% and 7.4%, respectively.

## Integrity checks

- Paired dataset manifest SHA-256:
  `e9fff2a8fed7b801ed4cf0110ea83ed07bf10c4c5a109a18b5c71d0a305e9cbe`.
- Repeated dataset manifest SHA-256:
  `09d2d7afc2528ffbe48cb0ffec0b379c1ef08bb128a7e5f871d61deb7994eded`.
- `runs.tsv` SHA-256:
  `b0a0cb35216975b3121013f677685e29a2ba02a595797a9d8a730a92085fbbf0`.
- `run_plan.tsv` SHA-256:
  `284264e3450c9aaae6d4090c31fa825ffaf858557452d797d8cd4f591328fa9c`.
- Aggregate continuous/M4/M5 table SHA-256 values:
  `2961243c45ad1a13f076fd544098a9c1ab614249c1f69416e53e5ff6d8980e3a`,
  `86a2536336578bcd722021148f3bcb75233d7586b22f25d97cc34c2c1d92c5ba`,
  and `aee163331f3d2a5ad469bd902c91d2ea5b09c82cb103417a259e315282484ab2`.

## Boundary and next gate

- **M4 paired/repeated LOSO estimability gate at n=12: pass.**
- **PR10 formal inference gate: not evaluated and not passed.** The run has
  only two bootstraps and two permutations, no calibration gate, zero formal
  rows, and all formal statuses remain `calibration_gate_missing`.
- These results do not estimate Type-I error, FDR, CI coverage, or power and do
  not justify releasing p/q values.
- Formal paired/repeated campaigns must preflight enough complete trajectories;
  n=12 is the empirically validated scale for the current two-fold M4 protocol.
- Before the required 1,000-bootstrap plus 1,000-permutation distributions, the
  next engineering gate is a deterministic process-based resample backend. The
  existing shared-snapshot thread backend was slower than serial in the frozen
  scaling diagnostic.
