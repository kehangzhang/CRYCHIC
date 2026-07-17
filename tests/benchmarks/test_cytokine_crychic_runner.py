from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from benchmarks.literature import run_cytokine_crychic as runner


def test_runner_defaults_preserve_tnbc_cli_compatibility() -> None:
    parameters = inspect.signature(runner.run).parameters
    assert parameters["dataset_id"].default == "TNBC"
    assert parameters["label_key"].default == "label"


def test_runner_records_explicit_dataset_and_label_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = ad.AnnData(
        X=np.ones((2, 2), dtype=np.int32),
        obs=pd.DataFrame(
            {"minor_type": ["A", "B"]},
            index=["a", "b"],
        ),
        var=pd.DataFrame(index=["L", "R"]),
    )
    input_path = tmp_path / "input.h5ad"
    data.write_h5ad(input_path)
    resource_path = tmp_path / "resource.parquet"
    pd.DataFrame({"ligand": ["L"], "receptor": ["R"]}).to_parquet(
        resource_path, index=False
    )
    captured = {}

    monkeypatch.setattr(
        runner,
        "_bundle",
        lambda _path: (
            SimpleNamespace(resource_id="toy", interactions=("lr",)),
            {"input_available_interactions": 1},
            resource_path,
        ),
    )

    def fake_validate(value, _schema):
        captured["obs"] = value.obs.copy()
        return SimpleNamespace(mode=SimpleNamespace(value="counts"))

    monkeypatch.setattr(runner, "validate_anndata", fake_validate)
    monkeypatch.setattr(
        runner,
        "aggregate_pseudobulk",
        lambda _value, min_cells: SimpleNamespace(matrix_unit_ids=(min_cells,)),
    )
    monkeypatch.setattr(
        runner,
        "estimate_bundle_availability",
        lambda *_args, **_kwargs: SimpleNamespace(
            sample_interactions=pd.DataFrame(
                {
                    "sender": ["A"],
                    "receiver": ["B"],
                    "ligand": ["L"],
                    "receptor": ["R"],
                }
            )
        ),
    )

    manifest = runner.run(
        input_path,
        tmp_path,
        tmp_path / "output",
        min_cells=1,
        dataset_id="HER2_Wu2021",
        label_key="minor_type",
        overwrite=False,
    )

    assert manifest["parameters"]["dataset_id"] == "HER2_Wu2021"
    assert manifest["parameters"]["label_key"] == "minor_type"
    assert set(captured["obs"]["cyto_context"]) == {"HER2_Wu2021"}
    assert captured["obs"]["cyto_cell_type"].tolist() == ["A", "B"]
