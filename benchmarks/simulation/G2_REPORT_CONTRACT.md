# Graph-Fusion G2 Report Contract

The default command runs the complete frozen 16-cell factorial with one paired
seed per cell. It is a runtime smoke test, not a decision-gate execution.

```bash
python -m benchmarks.simulation.run_graph_fusion_g2
```

Every output directory contains:

- `paired_results.csv`: one row per scenario, replicate, and method variant;
- `gate.json`: primary paired-bootstrap decision and topology controls;
- `report.md`: human-readable decision summary;
- `run_manifest.json`: config, source, environment, seed, and artifact provenance.

The report must display `NE` when any scenario cell has fewer than the frozen
minimum paired replicates. Secondary AUROC, RMSE, and jump-localization metrics
cannot substitute for the primary `driver_family x context` macro-AUPRC gate.
Formal replicate counts are selected with `--profile formal` or an explicit
`--replicates`; neither option can lower the frozen gate minimum.
