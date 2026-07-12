from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from benchmarks.simulation.export_multimethod_controls import (
    SCENARIOS,
    export_controls,
)


def test_export_controls_writes_normalized_x_and_integer_counts(
    tmp_path: Path,
) -> None:
    manifest = export_controls(
        tmp_path,
        n_subjects=2,
        mean_cells_per_sample=30,
        seed=7,
        overwrite=False,
    )

    assert len(manifest["records"]) == len(SCENARIOS)  # type: ignore[arg-type]
    active = ad.read_h5ad(tmp_path / "synthetic_active.h5ad")
    assert active.layers["counts"].dtype == np.int32
    assert active.X.dtype == np.float32
    assert np.isfinite(active.X.data).all()
    assert set(active.obs["condition"]) == {"ctrl", "stim"}
    assert set(active.obs["scenario"]) == {"active"}
    assert (tmp_path / "scenario_truth.tsv").is_file()


def test_export_controls_refuses_nonempty_directory(tmp_path: Path) -> None:
    (tmp_path / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_controls(
            tmp_path,
            n_subjects=2,
            mean_cells_per_sample=30,
            seed=7,
            overwrite=False,
        )
