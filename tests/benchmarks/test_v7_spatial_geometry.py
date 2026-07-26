from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.literature.run_v7_spatial_geometry import (
    DEFAULT_CONFIG,
    load_spatial_geometry_config,
)
from benchmarks.literature.spatial_geometry import (
    DistanceBand,
    SectionGeometryResult,
    build_band_geometry,
    cell_type_permutations,
    compute_section_geometry,
    condition_geometry_effects,
    distance_bands,
    expected_geometry_sets,
    rank_normal_abundance,
    section_association_table,
    visium_hex_coordinates,
)


def _hex_table(rows: int = 5, columns: int = 6) -> pd.DataFrame:
    records = [
        {"array_row": row, "array_col": 2 * column + row % 2}
        for row in range(rows)
        for column in range(columns)
    ]
    return pd.DataFrame.from_records(records)


def _bands() -> tuple[DistanceBand, ...]:
    return (
        DistanceBand("contact", 0.0, 1.01),
        DistanceBand("short", 1.01, 2.01),
        DistanceBand("local", 2.01, 4.01),
        DistanceBand("diffuse", 4.01, 8.01),
    )


def _result(
    sample: str,
    subject: str,
    condition: str,
    values: tuple[float, float, float],
    *,
    null_replicates: int = 3,
    cell_types: tuple[str, str, str] = ("A", "B", "C"),
) -> SectionGeometryResult:
    observed = np.asarray([values], dtype=float)
    coordinate_null = np.zeros((null_replicates, 1, 3), dtype=np.float32)
    return SectionGeometryResult(
        dataset="fixture",
        sample_id=sample,
        subject_id=subject,
        condition=condition,
        cell_types=cell_types,
        bands=("contact",),
        observed=observed,
        coordinate_null=coordinate_null,
        pair_left=np.asarray([0, 0, 1], dtype=np.int16),
        pair_right=np.asarray([1, 2, 2], dtype=np.int16),
        band_audit=pd.DataFrame(
            {
                "sample_id": [sample],
                "band": ["contact"],
                "mean_distance": [1.0],
            }
        ),
    )


def test_spatial_geometry_protocol_loads_and_rejects_release_drift(
    tmp_path: Path,
) -> None:
    protocol, contracts, bands = load_spatial_geometry_config(DEFAULT_CONFIG)

    assert tuple(contracts) == ("kuppe", "ms")
    assert contracts["kuppe"].primary_analysis_unit == "subject_id"
    assert contracts["ms"].samples == 11
    assert [band.name for band in bands] == ["contact", "short", "local", "diffuse"]
    assert protocol["release"]["ordinary_spot_pearson_is_not_spatial_geometry"]
    assert protocol["algorithm_alignment"]["generators"] == ["G0", "G2", "G3", "G5"]
    assert protocol["algorithm_alignment"]["datasets"]["ms"][
        "condition_map_algorithm_to_geometry"
    ] == {"Ctrl": "control", "CA": "chronic_active"}

    changed = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed["release"]["causal_sender_claim_allowed"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="release boundary changed"):
        load_spatial_geometry_config(path)

    changed = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed["algorithm_alignment"]["diagnostic_alpha"] = 0.1
    path = tmp_path / "changed-alignment.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="algorithm-alignment contract changed"):
        load_spatial_geometry_config(path)


def test_visium_hex_transform_makes_both_neighbor_offsets_unit_distance() -> None:
    table = pd.DataFrame({"array_row": [0, 0, 1], "array_col": [0, 2, 1]})
    coordinates = visium_hex_coordinates(table)

    assert np.linalg.norm(coordinates[0] - coordinates[1]) == pytest.approx(1.0)
    assert np.linalg.norm(coordinates[0] - coordinates[2]) == pytest.approx(1.0)

    broken = table.copy()
    broken.loc[2, "array_col"] = 2
    with pytest.raises(ValueError, match="parity"):
        visium_hex_coordinates(broken)


def test_sparse_bands_are_row_normalized_and_contact_connected() -> None:
    coordinates = visium_hex_coordinates(_hex_table())
    geometry = build_band_geometry(coordinates, _bands())

    assert len(geometry) == 4
    assert geometry[0].unordered_spot_pairs > 0
    assert geometry[0].connected_components == 1
    for item in geometry:
        row_mass = np.asarray(item.weights.sum(axis=1)).reshape(-1)
        assert np.allclose(row_mass[row_mass > 0.0], 1.0)
        assert item.weights.nnz == 2 * item.unordered_spot_pairs


def test_rank_normal_transform_keeps_constant_cell_type_not_estimable() -> None:
    transformed, variable = rank_normal_abundance(
        pd.DataFrame({"varying": [0.0, 1.0, 2.0], "constant": [1.0, 1.0, 1.0]})
    )

    assert variable.tolist() == [True, False]
    assert np.mean(transformed[:, 0]) == pytest.approx(0.0, abs=1e-7)
    assert np.isnan(transformed[:, 1]).all()


def test_section_geometry_uses_whole_vector_coordinate_permutations() -> None:
    coordinates = _hex_table(rows=6, columns=7)
    xy = visium_hex_coordinates(coordinates)
    first = xy[:, 0] - xy[:, 0].min() + 1.0
    second = first.copy()
    third = first.max() - first + 1.0
    abundance = pd.DataFrame({"A": first, "B": second, "C": third})

    result = compute_section_geometry(
        dataset="fixture",
        sample_id="sample-1",
        subject_id="subject-1",
        condition="target",
        coordinates=coordinates,
        abundance=abundance,
        bands=_bands(),
        coordinate_permutations=5,
        seed=19,
    )
    repeated = compute_section_geometry(
        dataset="fixture",
        sample_id="sample-1",
        subject_id="subject-1",
        condition="target",
        coordinates=coordinates,
        abundance=abundance,
        bands=_bands(),
        coordinate_permutations=5,
        seed=19,
    )

    assert result.observed.shape == (4, 3)
    assert result.coordinate_null.shape == (5, 4, 3)
    assert np.array_equal(result.coordinate_null, repeated.coordinate_null)
    assert result.observed[0, 0] > 0.0
    table = section_association_table(result)
    assert len(table) == 12
    assert set(table["coordinate_permutations"]) == {5}
    assert set(table["claim_scope"]) == {"indirect_spot_geometry_diagnostic_only"}


def test_condition_effects_merge_repeated_sections_before_contrast() -> None:
    results = (
        _result("r1", "r1", "reference", (0.0, 0.0, 0.0)),
        _result("r2", "r2", "reference", (0.0, 0.0, 0.0)),
        _result("t1-a", "t1", "target", (2.0, 1.0, -1.0)),
        _result("t1-b", "t1", "target", (4.0, 1.0, -1.0)),
        _result("t2", "t2", "target", (1.0, 1.0, -1.0)),
    )
    label_axis = cell_type_permutations(
        ("A", "B", "C"), replicates=5, seed=23, dataset="fixture"
    )

    subject = condition_geometry_effects(
        results,
        reference="reference",
        target="target",
        analysis_unit="subject_id",
        label_permutations=label_axis,
        minimum_units_per_condition=2,
    )
    sample = condition_geometry_effects(
        results,
        reference="reference",
        target="target",
        analysis_unit="sample_id",
        label_permutations=label_axis,
        minimum_units_per_condition=2,
    )

    subject_ab = subject.loc[
        subject["sender"].eq("A") & subject["receiver"].eq("B")
    ].iloc[0]
    sample_ab = sample.loc[sample["sender"].eq("A") & sample["receiver"].eq("B")].iloc[
        0
    ]
    assert subject_ab["n_target_units"] == 2
    assert sample_ab["n_target_units"] == 3
    assert subject_ab["effect_target_minus_reference"] == pytest.approx(2.0)
    assert sample_ab["effect_target_minus_reference"] == pytest.approx(7 / 3)
    assert not subject["formal_inference_allowed"].any()

    expected = expected_geometry_sets(subject, top_fractions=(0.5,))
    assert len(expected) == len(subject)
    assert int(expected["is_expected"].sum()) == 1


def test_distance_band_contract_rejects_gaps() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        distance_bands(
            (
                {"name": "contact", "lower_exclusive": 0.0, "upper_inclusive": 1.0},
                {"name": "short", "lower_exclusive": 2.0, "upper_inclusive": 3.0},
            )
        )


def test_materialized_unordered_pairs_are_canonical_for_nonlexical_axis() -> None:
    names = ("A", "C", "B")
    results = (
        _result("r1", "r1", "reference", (0.0, 0.0, 0.0), cell_types=names),
        _result("r2", "r2", "reference", (0.0, 0.0, 0.0), cell_types=names),
        _result("t1", "t1", "target", (1.0, 2.0, 3.0), cell_types=names),
        _result("t2", "t2", "target", (1.0, 2.0, 3.0), cell_types=names),
    )
    associations = section_association_table(results[0])
    label_axis = cell_type_permutations(
        names, replicates=3, seed=31, dataset="fixture"
    )
    effects = condition_geometry_effects(
        results,
        reference="reference",
        target="target",
        analysis_unit="subject_id",
        label_permutations=label_axis,
        minimum_units_per_condition=2,
    )
    expected_pairs = {("A", "B"), ("A", "C"), ("B", "C")}

    for table in (associations, effects):
        observed_pairs = set(
            table[["sender", "receiver"]].itertuples(index=False, name=None)
        )
        assert observed_pairs == expected_pairs
