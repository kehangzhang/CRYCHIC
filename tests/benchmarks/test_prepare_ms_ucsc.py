from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import benchmarks.datasets.prepare_ms_ucsc as module
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _write_fixture(path: Path) -> None:
    barcodes = ["C1", "C2", "C3"]
    genes = ["G1", "G2"]
    with gzip.open(path / "barcodes.tsv.gz", "wt") as handle:
        handle.write("\n".join(barcodes))
    with gzip.open(path / "features.tsv.gz", "wt") as handle:
        handle.write("\n".join(genes))
    with gzip.open(path / "matrix.mtx.gz", "wb") as handle:
        mmwrite(
            handle,
            sparse.coo_matrix(np.asarray([[1, 0, 3], [0, 2, 1]], dtype=float)),
        )
    pd.DataFrame(
        {
            "cellId": barcodes,
            "patient_id": ["P1", "P1", "P2"],
            "sample_id": ["S1", "S1", "S2"],
            "condition": ["Control", "Control", "MS"],
            "lesion_type": ["Ctrl", "Ctrl", "CA"],
            "batch_sn": [1, 1, 2],
            "celltype": ["MG", "OL", "MG"],
            "subtype": ["MG1", "OL1", "MG2"],
        }
    ).to_csv(path / "meta.tsv", sep="\t", index=False)


def test_prepare_ms_ucsc_preserves_counts_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture(tmp_path)
    expected = {
        name: {
            "size": source.stat().st_size,
            "md5_prefix": _md5(source)[:10],
        }
        for name in module.EXPECTED
        if (source := tmp_path / name).is_file()
    }
    monkeypatch.setattr(module, "EXPECTED", expected)

    output = tmp_path / "prepared.h5ad"
    result = module.prepare_ms_ucsc(tmp_path, output)

    assert result.shape == (3, 2)
    assert result.obs_names.tolist() == ["C1", "C2", "C3"]
    assert result.var_names.tolist() == ["G1", "G2"]
    assert result.obs["batch"].tolist() == ["1", "1", "2"]
    assert np.array_equal(
        result.layers["counts"].toarray(),
        np.asarray([[1, 0], [0, 2], [3, 1]], dtype=np.int32),
    )
    assert result.layers["counts"].dtype == np.int32
    assert output.is_file()
    assert output.with_suffix(".conversion.json").is_file()


def test_integer_counts_rejects_fractional_values() -> None:
    matrix = sparse.coo_matrix(np.asarray([[1.0, 0.5]]))
    with pytest.raises(ValueError, match="non-integer"):
        module._integer_counts(matrix)
