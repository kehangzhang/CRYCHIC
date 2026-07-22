from __future__ import annotations

from pathlib import Path

import pandas as pd
from benchmarks.adapters.common import sha256_file
from benchmarks.comprehensive.derive_misc_complete_case import derive
from benchmarks.comprehensive.evaluate_olink import ligand_universe_id


def test_complete_case_derivation_keeps_only_all_method_observed_ligands(
    tmp_path: Path,
) -> None:
    source = tmp_path / "predictions.tsv"
    rows = []
    for method in ("a", "b"):
        for ligand in ("L1", "L2"):
            observed = ligand == "L1" or method == "a"
            rows.append(
                {
                    "dataset_id": "misc_olink",
                    "method_id": method,
                    "resource_mode": "H-common",
                    "universe_id": ligand_universe_id(["L1", "L2"]),
                    "contrast": "misc_m_vs_s",
                    "ligand": ligand,
                    "score": 1.0 if observed else None,
                    "status": "observed" if observed else "not_estimable",
                }
            )
    pd.DataFrame(rows).to_csv(source, sep="\t", index=False)

    manifest = derive(
        source,
        tmp_path / "output",
        predictions_sha256=sha256_file(source),
    )

    result = pd.read_csv(tmp_path / "output/complete_case_predictions.tsv", sep="\t")
    assert manifest["ligands"] == 1
    assert set(result["ligand"]) == {"L1"}
    assert set(result["method_id"]) == {"a", "b"}
