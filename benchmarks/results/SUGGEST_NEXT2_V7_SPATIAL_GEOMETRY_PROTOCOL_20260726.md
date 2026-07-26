# suggest_next2 v7 spatial geometry protocol

Date: 2026-07-26

Status: implementation and real-section smoke validation complete; formal Kuppe/MS
geometry runs and v7-to-geometry scoring are pending the active PR10 calibration job.
No benchmark result is claimed by this document.

## Scope

This protocol adds the spatial endpoints requested by `suggest_next2.md` without
changing the CRYCHIC estimator:

- sparse Visium distance bands: contact `(0, 1.01]`, short `(1.01, 2.01]`, local
  `(2.01, 4.01]`, and diffuse `(4.01, 8.01]` nearest-neighbor spot pitches;
- within-section rank-normalized, symmetric row-normalized bivariate cross-Moran
  association;
- within-section coordinate permutation of each complete abundance row-vector;
- one dataset-global cell-type-label permutation shared by all sections and
  conditions in each replicate;
- contact versus secreted ConnectomeDB2020 mechanism stratification;
- Kuppe subject-level primary analysis with section-level sensitivity;
- MS subject-level primary analysis with section-level sensitivity.

Both cohorts are Visium and use non-aligned cell ontologies. Cross-platform spatial
replication is therefore not estimable from these two datasets and must not be
reported as complete.

## Frozen evaluation matrix

The algorithm comparison evaluates `G0`, `G2`, `G3`, and `G5`. `G4` is excluded
because its two score heads do not define a sender-resolved event ledger. Directed
LR effects are assigned to the condition favored by their signed subject-level
effect, then summed over both directions only when aligned to the unordered spatial
cell-pair axis.

Axes:

- mechanisms: all, contact, secreted;
- bands: contact, short, local, diffuse;
- endpoints: continuous weighted DES, diagnostic one-SE native count DES, and
  fixed-K count DES for K = 100, 250, 500, 1000;
- expected-set fractions: 0.1, 0.2, 0.3, 0.4;
- truth variants: rank-top, coordinate max-T supported, cell-label max-T
  supported, and joint coordinate/cell-label max-T supported;
- diagnostic threshold: 0.05, with no formal-inference interpretation.

The preregistered range hypotheses are contact LR events at the contact band and
secreted LR events at the local/diffuse bands. All mechanism-band combinations are
still emitted so that range specificity can be checked rather than assumed.

## Binding and claims

The protocol binds both input manifests, every persisted table, the exact
ConnectomeDB2020 payload, and the code commit by SHA-256. MS condition labels are
explicitly mapped from algorithm `Ctrl/CA` to geometry
`control/chronic_active`; no implicit string matching is allowed.

Geometry is indirect silver evidence for spot-level encounter opportunity. It is
not direct evidence of ligand-receptor binding, direction, causality, or physical
cell contact. Coordinate and label permutation p-values are endpoint diagnostics,
not released formal CCC p-values.

## Validation completed

- Kuppe `control_P1`: 4 bands x 55 cell pairs; sparse edge counts 12,453,
  24,365, 82,458, and 330,296; one contact component.
- MS `CO37`: 4 bands x 36 cell pairs; sparse edge counts 11,692, 22,912,
  77,683, and 310,574; one contact component.
- geometry core and real-input contracts: 7 tests passed;
- v7 alignment, mechanism filtering, signed condition mapping, truth support,
  DES, and rank concordance: 5 tests passed.

## Formal execution order

Run these only after the current 192-core calibration campaign releases capacity.
The geometry jobs are section-parallel, force one BLAS thread per worker, record
every completed section, memory fraction, and wall-clock ETA, and refuse to start
above 80% system memory.

```bash
python -m benchmarks.literature.run_v7_spatial_geometry \
  --dataset kuppe \
  --profile formal \
  --input-root /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/kuppe_spatial_misty \
  --input-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/kuppe_spatial_misty/kuppe_spatial_misty_input.manifest.json \
  --truth-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/spatial_truth/kuppe_figure3_sample_floor_exclude_self_v2/manifest.json \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/spatial_geometry_kuppe_formal_v1_20260726 \
  --sample-jobs 13
```

```bash
python -m benchmarks.literature.run_v7_spatial_geometry \
  --dataset ms \
  --profile formal \
  --input-root /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/downloads/ms_spatial_meta \
  --truth-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/spatial_truth/ms_figure3_sample_floor_exclude_self_v2/manifest.json \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/spatial_geometry_ms_formal_v1_20260726 \
  --sample-jobs 11
```

After the checksum-bound v7 real runs finish, evaluate each cohort with:

```bash
python -m benchmarks.literature.evaluate_v7_spatial_geometry \
  --dataset DATASET \
  --real-run-dir REAL_RUN_DIR \
  --geometry-run-dir GEOMETRY_RUN_DIR \
  --resource /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --output-dir OUTPUT_DIR
```

Primary outputs are `geometry_des_scores.tsv`, `geometry_des_coverage.tsv`,
`geometry_rank_concordance.tsv`, and `mechanism_distance_summary.tsv`. Every row
retains the dataset, mechanism, distance band, truth variant, endpoint, event
budget, claim scope, and formal-inference prohibition.
