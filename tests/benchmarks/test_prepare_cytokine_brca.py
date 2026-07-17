from __future__ import annotations

import io
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from benchmarks.literature.prepare_cytokine_brca import (
    EXPECTED_FIGURE6A_CUTOFFS,
    EXPECTED_FIGURE6A_DATASETS,
    EXPECTED_FIGURE6A_METHODS,
    _normalise_author_label,
    build_activity_truth,
    build_brca_input,
    build_cytosig_network,
    infer_cytosig_activity,
    validate_figure6a,
)
from scipy import sparse
from scipy.io import mmwrite


def _tar_bytes(files: dict[str, bytes], *, gzipped: bool) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz" if gzipped else "w") as handle:
        for name, content in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            handle.addfile(item, io.BytesIO(content))
    return payload.getvalue()


def _sample_archive(
    sample_id: str,
    matrix: sparse.spmatrix,
    barcodes: list[str],
    genes: list[str],
) -> bytes:
    matrix_buffer = io.BytesIO()
    mmwrite(matrix_buffer, matrix)
    prefix = f"{sample_id}/"
    return _tar_bytes(
        {
            f"{prefix}count_matrix_sparse.mtx": matrix_buffer.getvalue(),
            f"{prefix}count_matrix_genes.tsv": ("\n".join(genes) + "\n").encode(),
            f"{prefix}count_matrix_barcodes.tsv": ("\n".join(barcodes) + "\n").encode(),
        },
        gzipped=True,
    )


def test_build_brca_input_uses_per_sample_counts_and_author_filter(
    tmp_path: Path,
) -> None:
    metadata = pd.DataFrame(
        {
            "orig.ident": ["CID1", "CID1", "CID2", "CID2"],
            "subtype": ["HER2+"] * 4,
            "celltype_minor": ["Keep type", "Drop_type", "Keep type", "Keep type"],
            "celltype_subset": ["subset"] * 4,
            "celltype_major": ["major"] * 4,
        },
        index=["CID1_A", "CID1_B", "CID2_C", "CID2_D"],
    )
    combined = tmp_path / "combined.tar.gz"
    metadata_bytes = metadata.to_csv().encode()
    combined.write_bytes(
        _tar_bytes({"root/metadata.csv": metadata_bytes}, gzipped=True)
    )
    genes = ["G1", "G2", "G3"]
    first = _sample_archive(
        "CID1",
        sparse.coo_matrix(np.array([[1, 0], [0, 2], [3, 4]], dtype=np.int32)),
        ["A", "B"],
        genes,
    )
    second = _sample_archive(
        "CID2",
        sparse.coo_matrix(np.array([[5, 6], [0, 1], [2, 0]], dtype=np.int32)),
        ["C", "D"],
        genes,
    )
    raw = tmp_path / "raw.tar"
    raw.write_bytes(
        _tar_bytes(
            {
                "GSM1_CID1.tar.gz": first,
                "GSM2_CID2.tar.gz": second,
            },
            gzipped=False,
        )
    )

    data, audit = build_brca_input(
        raw,
        combined,
        subtype="HER2",
        min_celltype_cells=2,
    )

    assert data.shape == (3, 3)
    assert list(data.obs_names) == ["CID1_A", "CID2_C", "CID2_D"]
    assert set(data.obs["label"].astype(str)) == {"Keep.type"}
    assert audit["raw_cells"] == 4
    assert audit["prepared_cells"] == 3
    assert audit["removed_cell_types"] == ["Drop.type"]
    assert np.asarray(data.X.sum(axis=1)).ravel().tolist() == [4, 7, 7]


def test_network_activity_and_truth_are_unique_and_leakage_free(
    tmp_path: Path,
) -> None:
    signature = tmp_path / "signature.centroid"
    pd.DataFrame(
        {
            "S1": [4.0, 3.0, 2.0, 1.0],
            "S2": [-4.0, -3.0, -2.0, -1.0],
        },
        index=["G1", "G2", "G3", "G4"],
    ).to_csv(signature, sep="\t")
    network = build_cytosig_network(signature, top_targets=3)
    assert network.groupby("source").size().to_dict() == {"S1": 3, "S2": 3}

    data = ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [
                    [4, 3, 2, 1],
                    [2, 2, 1, 1],
                    [3, 1, 2, 1],
                    [4, 4, 2, 1],
                ],
                dtype=np.int32,
            )
        ),
        obs=pd.DataFrame(
            {"label": pd.Categorical(["A", "A", "B", "B"])},
            index=["a1", "a2", "b1", "b2"],
        ),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4"]),
    )

    def fake_mlm(
        expression: pd.DataFrame,
        local_network: pd.DataFrame,
        _tmin: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        columns = sorted(local_network["source"].unique())
        label = expression.index[0]
        scores = [
            2.0 if source == "S1" and label == "A" else -1.0 for source in columns
        ]
        pvalues = [0.001 if source == "S1" else 0.9 for source in columns]
        return (
            pd.DataFrame([scores], index=[label], columns=columns),
            pd.DataFrame([pvalues], index=[label], columns=columns),
        )

    activity = infer_cytosig_activity(
        data,
        network,
        detection_fraction=0.1,
        minimum_sum=1,
        tmin=2,
        mlm=fake_mlm,
    )
    assert len(activity) == 4
    assert activity["response"].sum() == 1
    assert activity.loc[activity["response"].eq(1), ["signature", "target"]].iloc[
        0
    ].tolist() == ["S1", "A"]

    resource = tmp_path / "resource.parquet"
    pd.DataFrame({"ligand": ["S1", "S2"]}).to_parquet(resource, index=False)
    truth = build_activity_truth(activity, resource)
    assert len(truth) == 4
    assert not truth.duplicated(["ligand", "target"]).any()
    assert truth["response"].sum() == 1


def test_validate_official_figure6a_requires_complete_published_grid() -> None:
    rows = []
    for dataset in EXPECTED_FIGURE6A_DATASETS:
        for method in EXPECTED_FIGURE6A_METHODS:
            for cutoff in EXPECTED_FIGURE6A_CUTOFFS:
                rows.append(
                    {
                        "method_name": method,
                        "pval": 0.1,
                        "odds_ratio": 1.0,
                        "padj": 0.2,
                        "n_rank": cutoff,
                        "dataset": dataset,
                        "max_rank": 10000,
                    }
                )
    result = validate_figure6a(pd.DataFrame.from_records(rows))
    assert len(result) == 112
    assert not result.duplicated(["dataset", "method_name", "n_rank"]).any()


def test_author_label_normalisation_matches_released_r_code() -> None:
    assert _normalise_author_label("T cells CD8+") == "T.cells.CD8"
    assert _normalise_author_label("CAFs MSC_iCAF-like") == "CAFs.MSC.iCAF.like"
