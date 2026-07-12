from __future__ import annotations

from pathlib import Path

import anndata as ad
import benchmarks.datasets.subset_ms_primary as module
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


def _fixture(path: Path) -> str:
    obs = pd.DataFrame(
        {
            "sample_id": ["CA1", "CA2", "CO1", "CO2", "CI1"],
            "subject_id": ["P1", "P2", "P3", "P4", "P5"],
            "lesion_type": ["CA", "CA", "Ctrl", "Ctrl", "CI"],
            "cell_type": ["MG", "AS", "OL", "MG", "AS"],
            "batch": ["1", "2", "1", "2", "1"],
        },
        index=[f"C{index}" for index in range(5)],
    )
    counts = sparse.csr_matrix(np.arange(15, dtype=np.int32).reshape(5, 3))
    adata = ad.AnnData(
        X=counts.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2", "G3"]),
    )
    adata.layers["counts"] = counts
    adata.uns["crychic_conversion"] = {
        "dataset_id": "UCSC_Lerma_Martin_MS_snRNA"
    }
    adata.write_h5ad(path)
    return module._sha256(path)


def test_subset_ms_primary_preserves_order_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.h5ad"
    digest = _fixture(source)
    monkeypatch.setattr(module, "EXPECTED_PRIMARY_SHAPE", (4, 3))
    monkeypatch.setattr(module, "EXPECTED_SUBJECTS", {"CA": 2, "Ctrl": 2})

    output = tmp_path / "primary.h5ad"
    result = module.subset_ms_primary(
        source,
        output,
        expected_sha256=digest,
    )

    assert result.obs_names.tolist() == ["C0", "C1", "C2", "C3"]
    assert set(result.obs["lesion_type"]) == {"CA", "Ctrl"}
    assert np.array_equal(
        result.layers["counts"].toarray(),
        np.arange(12, dtype=np.int32).reshape(4, 3),
    )
    assert output.is_file()
    assert output.with_suffix(".subset.json").is_file()


def test_subset_ms_primary_rejects_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.h5ad"
    _fixture(source)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.subset_ms_primary(
            source,
            tmp_path / "primary.h5ad",
            expected_sha256="0" * 64,
        )
