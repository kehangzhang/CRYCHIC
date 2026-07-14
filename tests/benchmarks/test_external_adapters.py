from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.adapters.cellchat.run_by_sample import (
    _is_valid_empty_result,
    _r_seed,
    _seed_provenance,
)
from benchmarks.adapters.cellchat.run_by_sample import (
    _normalize_sample as normalize_cellchat,
)
from benchmarks.adapters.cellphonedb.run_by_sample import (
    _filtered_database,
)
from benchmarks.adapters.cellphonedb.run_by_sample import (
    _normalize_sample as normalize_cellphonedb,
)
from benchmarks.adapters.common import (
    LONG_TABLE_COLUMNS,
    materialize_fixed_universe,
    method_frozen_resource,
    validate_long_table,
)
from benchmarks.adapters.liana.run_by_sample import _normalize as normalize_liana
from benchmarks.adapters.nichenet.run_by_sample import (
    _expression,
    _prior_operator,
    _score_samples,
)
from scipy import sparse


def _sample_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "subject_id": ["p1", "p2"],
            "context_json": ['{"condition":"A"}', '{"condition":"B"}'],
        }
    )


def _support() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "cell_type": ["A", "B", "A", "B"],
            "n_cells": [20, 20, 20, 2],
        }
    )


def _resource() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interaction_id": ["i1", "i2"],
            "native_interaction_id": ["n1", pd.NA],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "method_covered": [True, False],
        }
    )


def test_cellchat_seed_is_mapped_into_r_integer_range() -> None:
    assert _r_seed(20260712, 0) == 20260712
    assert _r_seed(4_285_072_227, 0) == 2_137_588_580
    assert _r_seed(2_147_483_646, 0) == 2_147_483_646
    assert _r_seed(2_147_483_647, 0) == 2_147_483_647
    assert _r_seed(2_147_483_647, 1) == 1
    assert _r_seed(0, 0) == 2_147_483_647
    with pytest.raises(ValueError, match="uint32"):
        _r_seed(-1, 0)
    with pytest.raises(ValueError, match="uint32"):
        _r_seed(2**32, 0)


def test_cellchat_seed_provenance_records_requested_and_effective_values() -> None:
    provenance = _seed_provenance(["s1", "s2"], 4_285_072_227)

    assert provenance["requested_seed"] == 4_285_072_227
    assert provenance["requested_seed_type"] == "uint32"
    assert provenance["effective_seed_range"] == [1, 2_147_483_647]
    assert provenance["effective_sample_seeds"] == {
        "s1": 2_137_588_580,
        "s2": 2_137_588_581,
    }


def test_cellchat_distinguishes_valid_empty_result_from_method_failure() -> None:
    valid_empty = subprocess.CompletedProcess(
        args=["Rscript"],
        returncode=1,
        stdout="CellChat inference is done. Parameter values are stored.\n",
        stderr=(
            "Error in subsetCommunication_internal(...): "
            "No significant signaling interactions are inferred based on the input!\n"
        ),
    )
    unrelated_failure = subprocess.CompletedProcess(
        args=["Rscript"],
        returncode=1,
        stdout=valid_empty.stdout,
        stderr="Error in computeCommunProb: numerical failure\n",
    )
    premature_subset_failure = subprocess.CompletedProcess(
        args=["Rscript"],
        returncode=1,
        stdout="",
        stderr=valid_empty.stderr,
    )

    assert _is_valid_empty_result(valid_empty)
    assert not _is_valid_empty_result(unrelated_failure)
    assert not _is_valid_empty_result(premature_subset_failure)


def test_fixed_universe_materializes_absence_and_support_states() -> None:
    observed = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "sender": ["A"],
            "receiver": ["B"],
            "interaction_id": ["i1"],
            "target": [pd.NA],
            "score": [0.8],
        }
    )
    table = materialize_fixed_universe(
        observed,
        sample_metadata=_sample_metadata(),
        support=_support(),
        resource=_resource(),
        dataset_id="toy",
        run_id="run",
        method_id="method",
        method_version="1",
        analysis_track="lr_stlr",
        resource_mode="H-covered",
        resource_id="resource",
        resource_version="1",
        score_name="native",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
    )

    assert tuple(table.columns) == LONG_TABLE_COLUMNS
    assert table.groupby("sample_id").size().to_dict() == {"s1": 8, "s2": 8}
    assert table["universe_size"].unique().tolist() == [8]
    assert table["universe_id"].nunique() == 1
    status = table.groupby(["sample_id", "status"]).size().to_dict()
    assert status[("s1", "ok")] == 1
    assert status[("s1", "not_returned")] == 3
    assert status[("s1", "resource_unavailable")] == 4
    assert status[("s2", "not_returned")] == 1
    assert status[("s2", "insufficient_cells")] == 3
    assert status[("s2", "resource_unavailable")] == 4
    assert table.loc[~table["status"].eq("ok"), "score"].isna().all()


def test_target_program_universe_uses_one_source_agnostic_sender() -> None:
    observed = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "sender": ["__source_agnostic__"],
            "receiver": ["B"],
            "interaction_id": ["i1"],
            "target": [pd.NA],
            "score": [0.8],
        }
    )
    table = materialize_fixed_universe(
        observed,
        sample_metadata=_sample_metadata(),
        support=_support(),
        resource=_resource(),
        dataset_id="toy",
        run_id="run",
        method_id="target_method",
        method_version="1",
        analysis_track="ligand_target_program",
        resource_mode="native",
        resource_id="resource",
        resource_version="1",
        score_name="target_activity",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
        sender_types=("__source_agnostic__",),
        sender_requires_cells=False,
        interaction_direction="ligand_to_target_program",
    )

    assert table.groupby("sample_id").size().to_dict() == {"s1": 4, "s2": 4}
    assert set(table["sender"]) == {"__source_agnostic__"}
    assert set(table["interaction_direction"]) == {"ligand_to_target_program"}
    receiver_status = table.loc[
        (table["sample_id"] == "s2") & (table["receiver"] == "B"),
        ["interaction_id", "status"],
    ].set_index("interaction_id")["status"]
    assert receiver_status.to_dict() == {
        "i1": "insufficient_cells",
        "i2": "resource_unavailable",
    }


def test_validate_long_table_rejects_score_on_absent_result() -> None:
    observed = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "sender": ["A"],
            "receiver": ["B"],
            "interaction_id": ["i1"],
            "target": [pd.NA],
            "score": [0.8],
        }
    )
    table = materialize_fixed_universe(
        observed,
        sample_metadata=_sample_metadata(),
        support=_support(),
        resource=_resource(),
        dataset_id="toy",
        run_id="run",
        method_id="method",
        method_version="1",
        analysis_track="lr_stlr",
        resource_mode="H-covered",
        resource_id="resource",
        resource_version="1",
        score_name="native",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
    )
    index = table.index[table["status"].eq("not_returned")][0]
    table.loc[index, "score"] = 0.0
    with pytest.raises(ValueError, match="non-ok statuses"):
        validate_long_table(table)


def test_validate_long_table_rejects_sample_specific_universe() -> None:
    table = materialize_fixed_universe(
        pd.DataFrame(
            columns=["sample_id", "sender", "receiver", "interaction_id", "target"]
        ),
        sample_metadata=_sample_metadata(),
        support=_support(),
        resource=_resource(),
        dataset_id="toy",
        run_id="run",
        method_id="method",
        method_version="1",
        analysis_track="lr_stlr",
        resource_mode="H-covered",
        resource_id="resource",
        resource_version="1",
        score_name="native",
        score_direction="higher",
        specificity_score_name=None,
        min_cells=10,
    )
    index = table.index[table["sample_id"].eq("s2")][0]
    table.loc[index, "interaction_id"] = "sample_specific_edge"
    with pytest.raises(ValueError, match="same frozen edge universe"):
        validate_long_table(table)


def test_h_covered_resource_retains_method_specific_coverage() -> None:
    harmonized = pd.DataFrame(
        {
            "harmonized_interaction_id": ["i1", "i2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "cellchat_source_interaction_id": ["cc1", ""],
            "cellchat_covered": ["true", "false"],
        }
    )
    covered = method_frozen_resource(
        harmonized, method="cellchat", resource_mode="H-covered"
    )
    assert covered["method_covered"].tolist() == [True, False]
    with pytest.raises(ValueError, match="H-common"):
        method_frozen_resource(harmonized, method="cellchat", resource_mode="H-common")


def test_cellchat_normalizer_maps_and_collapses_native_duplicates() -> None:
    native = pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["B", "B"],
            "interaction_name": ["native1", "native2"],
            "prob": [0.2, 0.7],
            "pval": [0.4, 0.1],
        }
    )
    source_map = pd.DataFrame(
        {
            "source_interaction_id": ["native1", "native2"],
            "interaction_id": ["edge", "edge"],
        }
    )
    result = normalize_cellchat(native, sample_id="s1", source_map=source_map)
    assert len(result) == 1
    assert result.loc[0, "score"] == pytest.approx(0.7)
    assert result.loc[0, "within_dataset_p_value"] == pytest.approx(0.1)


def test_cellphonedb_normalizer_preserves_score_and_cell_label_pvalue() -> None:
    identifiers = {
        "id_cp_interaction": ["native1"],
        "interacting_pair": ["L_R"],
        "A|B": [7.5],
    }
    native = {
        "interaction_scores": pd.DataFrame(identifiers),
        "pvalues": pd.DataFrame({**identifiers, "A|B": [0.03]}),
    }
    source_map = pd.DataFrame(
        {"source_interaction_id": ["native1"], "interaction_id": ["edge"]}
    )
    result = normalize_cellphonedb(
        native,
        sample_id="s1",
        source_map=source_map,
        statistical=True,
        separator="|",
    )
    assert result.loc[0, "sender"] == "A"
    assert result.loc[0, "receiver"] == "B"
    assert result.loc[0, "score"] == pytest.approx(7.5)
    assert result.loc[0, "within_dataset_p_value"] == pytest.approx(0.03)


def test_liana_normalizer_maps_custom_pair_and_keeps_lower_score() -> None:
    native = pd.DataFrame(
        {
            "sample": ["s1", "s1"],
            "source": ["A", "A"],
            "target": ["B", "B"],
            "ligand_complex": ["L", "L"],
            "receptor_complex": ["R", "R"],
            "magnitude_rank": [0.4, 0.2],
            "specificity_rank": [0.3, 0.1],
        }
    )
    pair_map = pd.DataFrame(
        {"ligand": ["L"], "receptor": ["R"], "interaction_id": ["edge"]}
    )
    result = normalize_liana(native, pair_map=pair_map)
    assert len(result) == 1
    assert result.loc[0, "score"] == pytest.approx(0.2)
    assert result.loc[0, "specificity_score"] == pytest.approx(0.1)


def test_cellphonedb_database_filter_keeps_archive_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "filtered.zip"
    repeated = tmp_path / "filtered_repeated.zip"
    interactions = pd.DataFrame(
        {
            "id_cp_interaction": ["keep", "drop"],
            "multidata_1_id": [1, 2],
            "multidata_2_id": [3, 4],
        }
    )
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("interaction_table.csv", interactions.to_csv(index=False))
        archive.writestr("gene_table.csv", "gene\nG\n")
    _filtered_database(source, output, source_interaction_ids={"keep"})
    _filtered_database(source, repeated, source_interaction_ids={"keep"})
    with zipfile.ZipFile(output) as archive:
        filtered = pd.read_csv(io.BytesIO(archive.read("interaction_table.csv")))
        assert archive.read("gene_table.csv") == b"gene\nG\n"
    assert filtered["id_cp_interaction"].tolist() == ["keep"]
    assert output.read_bytes() == repeated.read_bytes()


def test_nichenet_prior_operator_preserves_ligand_order_and_coverage() -> None:
    prior = pd.DataFrame(
        {
            "ligand": ["L1", "L1", "L2"],
            "target": ["T1", "T2", "T1"],
            "weight": [1.0, 3.0, 2.0],
        }
    )
    operator, weight_sum, target_count, ligand_index = _prior_operator(
        prior,
        var_names=pd.Index(["L1", "T1", "T2"]),
        ligand_order=["L1", "L2"],
    )
    assert operator.shape == (3, 2)
    np.testing.assert_allclose(weight_sum, [4.0, 2.0])
    np.testing.assert_array_equal(target_count, [2, 1])
    np.testing.assert_array_equal(ligand_index, [0, -1])


def test_nichenet_target_program_score_does_not_multiply_sender_expression() -> None:
    resource = pd.DataFrame({"interaction_id": ["i1"]})
    operator = pd.DataFrame([[0.0], [1.0]])
    observed = _score_samples(
        {
            ("s1", "Sender"): np.array([100.0, 0.0]),
            ("s1", "Receiver"): np.array([0.0, 4.0]),
        },
        sample_ids=["s1"],
        cell_types=["Sender", "Receiver"],
        resource=resource,
        operator=sparse.csr_matrix(operator),
        weight_sum=np.array([1.0]),
        support=pd.DataFrame(
            {
                "sample_id": ["s1", "s1"],
                "cell_type": ["Sender", "Receiver"],
                "n_cells": [20, 20],
            }
        ),
        min_cells=10,
    )

    assert set(observed["sender"]) == {"__source_agnostic__"}
    assert observed.loc[observed["receiver"] == "Receiver", "score"].iloc[0] == 4.0


def test_nichenet_count_input_is_library_normalized_and_log_transformed() -> None:
    counts = sparse.csr_matrix([[1, 1], [1, 3]], dtype=np.int32)
    adata = ad.AnnData(X=counts)

    expression, transform = _expression(adata, None)

    assert transform == "counts_library_size_1e4_log1p"
    np.testing.assert_allclose(
        expression.toarray(),
        np.log1p([[5000.0, 5000.0], [2500.0, 7500.0]]),
    )


def test_nichenet_continuous_expression_is_preserved() -> None:
    continuous = sparse.csr_matrix([[0.0, 0.25], [1.5, 2.75]])
    adata = ad.AnnData(X=continuous)

    expression, transform = _expression(adata, None)

    assert transform == "input_continuous_expression_preserved"
    np.testing.assert_allclose(expression.toarray(), continuous.toarray())
