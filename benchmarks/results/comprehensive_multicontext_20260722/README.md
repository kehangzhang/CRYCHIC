# Comprehensive multi-context benchmark freeze

Status: design, source pinning, asset audit, MIS-C preparation, LIANA component
export, and core metric implementations complete. No new cross-method accuracy
leaderboard was run in this freeze.

## Frozen inventory

| Item | Count | Current state |
|---|---:|---|
| Datasets | 30 | 22 complete asset sets; partial, blocked, and undeclared assets remain explicit |
| Methods | 27 | 15 runnable, 1 frozen-output-only, 11 planned |
| Metrics | 36 | Separate event, downstream, factor, Olink, spatial, phenotype, robustness, and performance endpoints |
| Panels | 13 | Minimal, full, legacy, static, scaling, and patient-graph tracks |
| Contrasts | 8 | MIS-C omnibus/pairwise/one-vs-rest, anti-PD-1 interaction, and Figure 3 legacy contrasts |
| Hidden scenarios | 23 | Positive multi-group designs plus all required null and perturbation controls |
| Literature records | 18 | 17 local full texts; STACCato is metadata/source-code only because the PMC record is non-OA |
| Pinned source trees | 6 | All exact commits and declared license files verified |

The expanded execution matrix contains 133 eligible new-run resource
combinations, 2 reusable frozen outputs, and 125 explicit skips. A skip or NE
is never encoded as zero.

## New real-data panel

The MIS-C input was converted and read back successfully:

- 20,332 genes x 13,343 cells;
- 16 subjects and samples in three conditions;
- NK, T, and Monocyte broad cell types;
- 11 valid family blocks after splitting the shared `UNR` code into four
  unrelated adult blocks;
- sparse logcounts in X and raw counts in the `counts` layer;
- H5AD SHA256
  `c0caeb5b0a25e97218315626641c4366e2de9b4be0f09333569fe7df33bd8df9`.

The external Olink truth bundle contains 1,463 analytes. At q<=0.05, 788 have
a positive MIS-C-minus-healthy effect and 206 a negative effect. No analyte ID
changes under `make.names`; eight underscore composite analytes remain unsplit.
The truth TSV SHA256 is
`820034f368fe540d2b5ab6e9b7f3bf5ca1fc576167d114660c766ee5bd636b46`.

Olink is evaluation-only. RNA-only method outputs must be checksummed first;
the official MultiNicheNet Olink prioritization extension is disabled. The
tutorial/Zenodo call the specimen serum while the Diorio Methods call it
plasma, and the manifest preserves that discrepancy.

## Readiness gates

Only P07, P10, and P11 currently pass the preregistered 80% method-dataset
coverage gate. P04 MIS-C/Olink is 7/11: CRYCHIC, by-sample CellChat,
CellPhoneDB, LIANA RRA, LIANA+, the frozen NicheNet proxy, and scSeqCommDiff are
runnable; native NicheNet, MultiNicheNet, STACCato, and DCST remain planned.
P06 is 6/8 because MultiNicheNet is still missing for both legacy datasets.

The hidden G1 and Tensor-cell2cell panels intentionally remain at zero until
the method commit, generator, seeds, and truth are frozen. The scACCorDiON P13
panel also remains planned: its PDAC graph is ready, COVID/MI are partial, and
four paper-exact cohorts are not downloaded.

See `panel_readiness.tsv` for every panel.

## Implemented metric contracts

- The G1 evaluator computes prevalence-adjusted AP, AUROC, MCC, effect RMSE,
  effect Spearman, and sign accuracy. Native AP and coverage are diagnostics;
  single-class or invalid effects are NE.
- The Olink evaluator computes direction-matched q<=0.05 ligand AP, signed
  logFC Spearman, and directional effect-magnitude NDCG within a frozen,
  Olink-blind representable ligand universe. Less than 80% scored coverage of
  its measured intersection is diagnostic-only; assay-wide representability is
  reported separately. Only the shared H-common forward contrast is rankable.
- The LIANA exporter produces six current component arms from one raw run:
  CellPhoneDB mean and p-value, Connectome, logFC mean, NATMI, and
  SingleCellSignalR. It preserves the fixed universe and structural statuses
  and leaves differential effect/p/q empty.

A real cSCC LIANA export smoke covered a 906,304-row universe with 258,795
observed rows for all six components in 41.37 seconds; peak RSS was 836,916
KiB. This is an adapter/performance smoke, not an accuracy result.

## Legacy context

Kuppe/MS Figure 3 and CRYCHIC RC9 results are reused only for regression and
paper comparability. Both cohorts were used repeatedly during RC0-RC10
development and cannot support a new independent superiority claim. The
current summaries also use a Python positive running-sum analogue rather than
the original R `fgsea`, so exact R reproduction remains open.

CytoSig TNBC/HER2 and CITE-seq/IPF are static appendix panels. CytoSig is a
downstream activity reference, not a multi-group differential CCC algorithm.

## Verification

- Registry, G1, Olink, and LIANA component tests: 27 passed.
- Extended external-adapter and multi-condition regression selection: 106
  passed with two pre-existing pandas future warnings.
- Ruff lint and format: passed.
- MIS-C H5AD readback: passed.
- Actual Olink 1,463-row R export and Python contract readback: passed.
- All six upstream Git commits and licenses: verified.

## Remaining work

1. Commit this freeze, generate hidden G1 truth, and run the runnable methods.
2. Validate at least two of native NicheNet, MultiNicheNet, STACCato, or DCST
   to bring P04 to the 80% gate; implementing all four remains the target
   matrix.
3. Normalize the upstream Tensor-cell2cell simulation and implement the tensor
   adapters before running P03.
4. Freeze the scACCorDiON environment/adapter and acquire its exact remaining
   cohort objects.
5. Rerun Figure 3 with original R `fgsea` before any exact-reproduction claim.

Raw audits, prepared matrices, truth tables, third-party checkouts, and future
method outputs remain under
`benchmark_work/comprehensive_multicontext_20260722` and are not committed.
