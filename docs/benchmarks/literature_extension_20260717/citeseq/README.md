# Extended seven-dataset CITE-seq benchmark

## Scope

The frozen inventory contains 7 public CITE-seq datasets (5 human, 2 mouse). CBMC, SLN111 and SLN208 are the newly completed datasets. ADT was held out from all communication methods and used only for receiver-receptor truth.

The same-resource `H-common/resource_fixed` arm is the only direct method comparison within a dataset/species resource. Native independent results are descriptive because LIANA and CRYCHIC use different native resources.

## Newly completed results

Top H-common/resource-fixed AUROC rows (CBMC is not evaluable):

| dataset | species | method | auroc | average_precision |
| --- | --- | --- | --- | --- |
| sln_111 | mouse | CRYCHIC availability-state | 0.6262 | 0.1993 |
| sln_111 | mouse | SingleCellSignalR LRscore | 0.6209 | 0.1738 |
| sln_111 | mouse | Connectome specificity | 0.6150 | 0.2024 |
| sln_208 | mouse | SingleCellSignalR LRscore | 0.5623 | 0.1074 |
| sln_208 | mouse | CRYCHIC availability-state | 0.5411 | 0.1216 |
| sln_208 | mouse | CellChat composite | 0.5387 | 0.0945 |

Top native/independent AUROC rows:

| dataset | species | method | auroc | average_precision |
| --- | --- | --- | --- | --- |
| cbmc_seuratdata | human | Connectome specificity | 0.6841 | 0.2558 |
| cbmc_seuratdata | human | logFC specificity | 0.6828 | 0.2360 |
| cbmc_seuratdata | human | NATMI specificity | 0.6797 | 0.2659 |
| sln_111 | mouse | CRYCHIC availability-state | 0.7355 | 0.4074 |
| sln_111 | mouse | SingleCellSignalR LRscore | 0.6805 | 0.2902 |
| sln_111 | mouse | CellChat composite | 0.6573 | 0.3529 |
| sln_208 | mouse | CRYCHIC availability-state | 0.7227 | 0.3589 |
| sln_208 | mouse | SingleCellSignalR LRscore | 0.6975 | 0.2714 |
| sln_208 | mouse | CellChat composite | 0.6639 | 0.3360 |

## Skips and non-evaluable arms

| dataset | species | comparison_arm | method | status | reason_code |
| --- | --- | --- | --- | --- | --- |
| cbmc_seuratdata | human | H-common/resource_fixed | all | not_evaluable | no_receptor_truth_coverage |
| cbmc_seuratdata | human | H-common/independent | all | not_evaluable | no_receptor_truth_coverage |
| sln_111 | mouse | H-common/resource_fixed | CellPhoneDB composite | skipped | cellphonedb_mouse_unavailable |
| sln_111 | mouse | H-common/independent | CellPhoneDB composite | skipped | cellphonedb_mouse_unavailable |
| sln_111 | mouse | native/independent | CellPhoneDB composite | skipped | cellphonedb_mouse_unavailable |
| sln_111 | mouse | H-common/resource_fixed | CellPhoneDB p-value | skipped | cellphonedb_mouse_unavailable |
| sln_111 | mouse | H-common/independent | CellPhoneDB p-value | skipped | cellphonedb_mouse_unavailable |
| sln_111 | mouse | native/independent | CellPhoneDB p-value | skipped | cellphonedb_mouse_unavailable |
| sln_208 | mouse | H-common/resource_fixed | CellPhoneDB composite | skipped | cellphonedb_mouse_unavailable |
| sln_208 | mouse | H-common/independent | CellPhoneDB composite | skipped | cellphonedb_mouse_unavailable |
| sln_208 | mouse | native/independent | CellPhoneDB composite | skipped | cellphonedb_mouse_unavailable |
| sln_208 | mouse | H-common/resource_fixed | CellPhoneDB p-value | skipped | cellphonedb_mouse_unavailable |
| sln_208 | mouse | H-common/independent | CellPhoneDB p-value | skipped | cellphonedb_mouse_unavailable |
| sln_208 | mouse | native/independent | CellPhoneDB p-value | skipped | cellphonedb_mouse_unavailable |

## Audit limits

- Mouse CellPhoneDB is unavailable and was skipped. LIANA's placeholder CellPhoneDB component columns were explicitly excluded from mouse metrics.
- CBMC has 14 observed receptor truth genes, but none overlap the frozen 638-pair human H-common resource; reporting a metric would be invalid.
- All seven inputs have one sample, one subject and one context. They cannot estimate condition-specific or differential communication.
- ADT validates receiver-receptor specificity, not ligand binding, sender causality or downstream signaling activation.
- Aggregate means in `method_summary.tsv` are descriptive and species stratified; they are not pooled inferential estimates.

## Resource and memory audit

The highest measured peak RSS was 4.75 GiB (1.89% of host RAM), below the requested 70% ceiling. Runtime records and checksums are in the TSV and JSON manifests.

The human H-common resource contains 638 frozen simple LR pairs. The mouse H-common resource contains 705 LIANA-consensus x CellChatDB-mouse simple pairs. Native mouse LIANA uses the checksum-pinned 4,015-pair MGI resource.

## Published artifacts

- [Dataset inventory](dataset_inventory.tsv)
- [All dataset-level metrics](metrics_by_dataset.tsv)
- [Method summary](method_summary.tsv)
- [Skipped and non-evaluable arms](skipped_arms.tsv)
- [Runtime and memory observations](runtime_memory.tsv)
- [Publication manifest](manifest.json)
