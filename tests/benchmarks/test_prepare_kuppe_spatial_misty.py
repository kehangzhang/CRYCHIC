from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import benchmarks.literature.prepare_kuppe_spatial_misty as module
import numpy as np
import pandas as pd
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_slide(path: Path, offset: int = 0) -> None:
    n_spots = 12
    abundance = np.full((n_spots, len(module.ABUNDANCE_COLUMNS)), 1.0 / 11.0)
    abundance[2, 3] = np.nan
    obs = pd.DataFrame(
        {
            "array_row": np.arange(n_spots) + offset,
            "array_col": np.arange(n_spots) * 2 + offset,
            "in_tissue": [1, 0, 1, *([1] * 9)],
            **{
                name: abundance[:, index]
                for index, name in enumerate(module.ABUNDANCE_COLUMNS)
            },
        },
        index=pd.Index([f"spot_{offset}_{index}" for index in range(n_spots)]),
    )
    source = ad.AnnData(
        X=np.full((n_spots, 2), 999.0),
        obs=obs,
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    # Pixel coordinates are deliberately incompatible with array coordinates.
    source.obsm["spatial"] = np.column_stack(
        [np.arange(n_spots) * 1000 + 123, np.arange(n_spots) * 2000 + 456]
    )
    source.write_h5ad(path)


def test_prepare_kuppe_spatial_misty_uses_array_geometry_and_filters_spots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "prepared"
    source_dir.mkdir()
    frozen = (
        ("control_A.cellxgene.h5ad", "CTRL"),
        ("IZ_B.cellxgene.h5ad", "IZ"),
    )
    monkeypatch.setattr(module, "FROZEN_SLIDES", frozen)
    _write_slide(source_dir / frozen[0][0], offset=0)
    _write_slide(source_dir / frozen[1][0], offset=100)

    read_modes: list[str | None] = []
    real_read_h5ad = module.ad.read_h5ad

    def audited_read_h5ad(*args: object, **kwargs: object) -> ad.AnnData:
        read_modes.append(kwargs.get("backed"))  # type: ignore[arg-type]
        return real_read_h5ad(*args, **kwargs)

    monkeypatch.setattr(module.ad, "read_h5ad", audited_read_h5ad)

    manifest = module.prepare_kuppe_spatial_misty(source_dir, output_dir)

    assert read_modes == ["r", "r"]
    assert manifest["status"] == "complete"
    assert manifest["cohort"] == {
        "conditions": ["CTRL", "IZ"],
        "slides_by_condition": {"CTRL": 1, "IZ": 1},
        "n_slides": 2,
        "n_retained_spots": 20,
    }
    assert manifest["protocol"]["geometry_source"] == [
        "obs.array_row",
        "obs.array_col",
    ]
    assert manifest["protocol"]["pixel_coordinates_used"] is False
    assert manifest["provenance"]["author_importance_table"] is False
    assert manifest["manifest_payload_sha256"] == module._payload_sha256(manifest)

    first_record = manifest["slides"][0]
    output = output_dir / first_record["input"]["filename"]
    table = pd.read_csv(output, sep="\t")
    assert table.columns.tolist() == [
        "spot_id",
        "array_row",
        "array_col",
        *module.ABUNDANCE_COLUMNS,
    ]
    assert table["spot_id"].tolist() == [
        "spot_0_0",
        *[f"spot_0_{index}" for index in range(3, 12)],
    ]
    assert table["array_row"].tolist() == [0, *range(3, 12)]
    assert table["array_col"].max() == 22
    assert table["array_col"].max() < 1000
    assert first_record["spots"] == {
        "total": 12,
        "in_tissue": 11,
        "complete_abundance": 11,
        "retained": 10,
        "excluded_out_of_tissue": 1,
        "excluded_in_tissue_incomplete_abundance": 1,
    }
    assert first_record["input"]["sha256"] == _sha256(output)
    checksum_path = output_dir / first_record["input"]["checksum_filename"]
    assert checksum_path.read_text(encoding="ascii") == (
        f"{_sha256(output)}  {output.name}\n"
    )

    on_disk = json.loads(
        (output_dir / module.COHORT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert on_disk == manifest


def test_prepare_kuppe_spatial_misty_rejects_roster_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setattr(
        module,
        "FROZEN_SLIDES",
        (("control_A.cellxgene.h5ad", "CTRL"), ("IZ_B.cellxgene.h5ad", "IZ")),
    )
    _write_slide(source_dir / "control_A.cellxgene.h5ad")
    _write_slide(source_dir / "unexpected.cellxgene.h5ad", offset=100)

    with pytest.raises(ValueError, match="roster mismatch") as error:
        module.prepare_kuppe_spatial_misty(source_dir, tmp_path / "output")
    assert "IZ_B.cellxgene.h5ad" in str(error.value)
    assert "unexpected.cellxgene.h5ad" in str(error.value)


def test_prepare_kuppe_spatial_misty_rejects_duplicate_array_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    filename = "control_A.cellxgene.h5ad"
    monkeypatch.setattr(module, "FROZEN_SLIDES", ((filename, "CTRL"),))
    path = source_dir / filename
    _write_slide(path)
    source = ad.read_h5ad(path)
    source.obs.loc["spot_0_3", ["array_row", "array_col"]] = source.obs.loc[
        "spot_0_0", ["array_row", "array_col"]
    ]
    source.write_h5ad(path)

    with pytest.raises(ValueError, match="duplicate array geometry"):
        module.prepare_kuppe_spatial_misty(source_dir, tmp_path / "output")


def test_r_runner_is_parseable_and_pins_author_protocol() -> None:
    runner = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "literature"
        / "run_kuppe_spatial_misty.R"
    )
    text = runner.read_text(encoding="utf-8")
    assert 'packageVersion("mistyR")) != "1.3.5"' in text
    assert 'MISTYR_TAG_COMMIT <- "19248ea7e02803063d1e1112a8af6c3f06c59e03"' in text
    assert "EXPECTED_SAMPLE_CONDITIONS <- c(" in text
    assert "JUXTA_NEIGHBOR_THRESHOLD <- 5" in text
    assert "PARA_LENGTH_SCALE <- 15" in text
    assert "mistyR::add_juxtaview" in text
    assert "mistyR::add_paraview" in text
    assert "mistyR::run_misty" in text
    assert "mistyR::collect_results" in text
    assert 'gzfile(temporary, open = "wb"' in text
    assert "not the authors' published MISTy importance table" in text
    assert "Seurat" not in text

    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is unavailable for syntax validation")
    completed = subprocess.run(
        [rscript, "-e", f"parse(file={str(runner)!r})"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
