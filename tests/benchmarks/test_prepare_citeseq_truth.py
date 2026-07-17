from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.literature.prepare_citeseq_truth import (
    normalize_author_receptor_truth,
)


def test_normalize_author_receptor_truth_maps_without_changing_labels() -> None:
    source = pd.DataFrame(
        {
            "dataset_id": ["cite", "cite"],
            "target_cell_type": ["cluster.1", "cluster.0"],
            "receptor_gene": ["il7r", "CD14"],
            "label": [1, 0],
            "is_positive": [True, False],
            "protein_z_max": [2.0, -1.0],
        }
    )

    result = normalize_author_receptor_truth(source)

    assert tuple(result.columns) == (
        "dataset",
        "receiver",
        "receptor",
        "is_positive",
        "truth_status",
    )
    assert result["receptor"].tolist() == ["CD14", "IL7R"]
    assert result["is_positive"].tolist() == [0, 1]
    assert set(result["truth_status"]) == {"observed"}


def test_normalize_author_receptor_truth_rejects_duplicate_keys() -> None:
    source = pd.DataFrame(
        {
            "dataset_id": ["cite", "cite"],
            "target_cell_type": ["cluster.0", "cluster.0"],
            "receptor_gene": ["CD14", "cd14"],
            "label": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="duplicate canonical keys"):
        normalize_author_receptor_truth(source)
