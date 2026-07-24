from __future__ import annotations

import pandas as pd

from benchmarks.comprehensive.run_liana_cellphonedb_cohort import (
    _stable_seed,
    standardize_liana_result,
)


def test_stable_sample_seed() -> None:
    assert _stable_seed(7, "sample-a") == _stable_seed(7, "sample-a")
    assert _stable_seed(7, "sample-a") != _stable_seed(7, "sample-b")


def test_liana_result_uses_registered_pvalue_filter() -> None:
    frame = pd.DataFrame(
        {
            "source": ["A", "A", "B"],
            "target": ["B", "B", "A"],
            "ligand": ["L2", "L1", "L3"],
            "receptor": ["R2", "R1", "R3"],
            "lr_means": [2.0, 1.0, 3.0],
            "cellphone_pvals": [0.01, 0.0, 0.02],
        }
    )
    result = standardize_liana_result(frame, p_value_threshold=0.01)

    assert len(result) == 2
    assert result["ligand"].tolist() == ["L1", "L2"]
