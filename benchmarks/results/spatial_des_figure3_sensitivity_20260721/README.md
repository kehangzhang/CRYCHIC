# Figure 3 DES sensitivity index

This directory indexes 20 checksum-isolated comparison panels covering:

- fgsea-default `std`, `gseaParam=1`, raw cardinality;
- Figure-compatible `pos`, `gseaParam=1`, average cardinality rank;
- former `pos`, `gseaParam=0`, raw cardinality;
- `pos`, `gseaParam=1`, include-self plus ceil top sets for scSeqCommDiff;
- `pos`, `gseaParam=1`, strict native pair estimability for scSeqCommDiff.

The table must not be collapsed across `comparison_panel_id`. Different score
semantics, expected-set checksums, self-pair policies, tie policies, datasets,
scenarios, and analysis units are deliberately separate. The main
Figure-compatible report is in `../spatial_des_figure3_reanalysis_20260721`.

See
`docs/benchmarks/multigroup_spatial_20260717/FIGURE3_REANALYSIS_20260721.md`
for interpretation and the limits of exact reproduction.
