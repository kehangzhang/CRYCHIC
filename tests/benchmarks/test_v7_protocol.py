from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from benchmarks.adapters.common import sha256_file
from benchmarks.simulation.v7_protocol import (
    DEFAULT_CONFIG,
    GENERATORS,
    INFERENCES,
    PLAN_COLUMNS,
    REQUIRED_DGP_FAMILIES,
    expand_v7_benchmark_plan,
    load_v7_benchmark_protocol,
    resolve_v7_dgp_scale,
    write_v7_benchmark_plan,
)


def _config() -> dict[str, object]:
    value: object = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(
        json.dumps(config, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def test_frozen_v7_protocol_has_disjoint_complete_dgp_axes() -> None:
    protocol = load_v7_benchmark_protocol()
    roles = protocol.family_roles
    family_sets = {role: set(roles[role]) for role in roles}

    assert family_sets["development"].isdisjoint(family_sets["locked"])
    assert family_sets["development"].isdisjoint(family_sets["null_calibration"])
    assert family_sets["locked"].isdisjoint(family_sets["null_calibration"])
    assert set().union(*family_sets.values()) == REQUIRED_DGP_FAMILIES
    assert protocol.replicate_count("smoke") == 20
    assert protocol.replicate_count("development") == 100
    assert protocol.replicate_count("locked") == 250
    assert protocol.replicate_count("null_calibration") == 1000
    assert protocol.config["generator_matrix"].keys() == set(GENERATORS)
    assert protocol.config["inference_matrix"].keys() == set(INFERENCES)


def test_plan_expansion_pairs_generators_on_identical_datasets_and_seeds() -> None:
    protocol = load_v7_benchmark_protocol()
    primary = expand_v7_benchmark_plan(
        protocol,
        phase="development",
        profile="score_primary",
        maximum_replicates=2,
    )

    assert tuple(primary.columns) == PLAN_COLUMNS
    assert len(primary) == 8 * 2 * len(GENERATORS)
    assert primary["dataset_id"].nunique() == 16
    assert primary["run_id"].is_unique
    assert set(primary["generator_id"]) == set(GENERATORS)
    assert set(primary["inference_id"]) == {"I1"}
    grouped = primary.groupby("dataset_id", observed=True)
    assert grouped["seed"].nunique().eq(1).all()
    assert grouped["candidate_sender_count"].nunique().eq(1).all()
    assert grouped["cells_per_type"].nunique().eq(1).all()
    assert grouped["subjects_per_level"].nunique().eq(1).all()
    assert grouped.size().eq(len(GENERATORS)).all()
    datasets = primary.loc[:, ["dataset_id", "seed"]].drop_duplicates()
    assert datasets["seed"].is_unique

    crossover = expand_v7_benchmark_plan(
        protocol,
        phase="development",
        profile="inference_crossover",
        maximum_replicates=2,
    )
    assert len(crossover) == 8 * 2 * len(INFERENCES)
    assert set(crossover["generator_id"]) == {"G3"}
    assert set(crossover["inference_id"]) == set(INFERENCES)
    paired = primary.loc[
        primary["generator_id"].eq("G3") & primary["inference_id"].eq("I1"),
        ["dataset_id", "seed"],
    ].merge(
        crossover.loc[crossover["inference_id"].eq("I1"), ["dataset_id", "seed"]],
        on="dataset_id",
        suffixes=("_primary", "_crossover"),
        validate="one_to_one",
    )
    assert (paired["seed_primary"] == paired["seed_crossover"]).all()


def test_smoke_covers_every_family_without_changing_family_role() -> None:
    protocol = load_v7_benchmark_protocol()
    plan = expand_v7_benchmark_plan(
        protocol,
        phase="smoke",
        profile="inference_crossover",
        maximum_replicates=1,
    )
    assert set(plan["dgp_family"]) == REQUIRED_DGP_FAMILIES
    assert set(plan["family_role"]) == {
        "development",
        "locked",
        "null_calibration",
    }
    assert plan["dataset_id"].nunique() == 45
    assert len(plan) == 45 * len(INFERENCES)


def test_dgp_scale_cycles_only_on_preregistered_stress_families() -> None:
    protocol = load_v7_benchmark_protocol()
    cardinalities = [
        resolve_v7_dgp_scale(
            protocol,
            phase="smoke",
            dgp_family="candidate_cardinality",
            replicate_index=index,
        ).candidate_sender_count
        for index in range(1, 9)
    ]
    assert cardinalities == [2, 5, 10, 20, 2, 5, 10, 20]
    cell_counts = [
        resolve_v7_dgp_scale(
            protocol,
            phase="null_calibration",
            dgp_family="fixed_subject_increasing_cell_null",
            replicate_index=index,
        ).cells_per_type
        for index in range(1, 9)
    ]
    assert cell_counts == [1, 2, 4, 8, 1, 2, 4, 8]
    default = resolve_v7_dgp_scale(
        protocol,
        phase="locked",
        dgp_family="ligand_only",
        replicate_index=7,
    )
    assert (default.candidate_sender_count, default.cells_per_type) == (5, 4)
    assert default.subjects_per_level == 8


def test_protocol_digest_and_plan_ignore_json_object_order(tmp_path: Path) -> None:
    original = _config()
    reordered = dict(reversed(tuple(original.items())))
    families = reordered["dgp_families"]
    assert isinstance(families, dict)
    reordered["dgp_families"] = {
        role: dict(reversed(tuple(values.items())))
        for role, values in reversed(tuple(families.items()))
    }
    reordered_path = tmp_path / "reordered.json"
    _write_config(reordered_path, reordered)

    first = load_v7_benchmark_protocol()
    second = load_v7_benchmark_protocol(reordered_path)
    assert first.protocol_digest == second.protocol_digest
    first_plan = expand_v7_benchmark_plan(
        first,
        phase="locked",
        profile="score_primary",
        maximum_replicates=1,
    )
    second_plan = expand_v7_benchmark_plan(
        second,
        phase="locked",
        profile="score_primary",
        maximum_replicates=1,
    )
    pd.testing.assert_frame_equal(first_plan, second_plan)


def test_protocol_refuses_family_leakage_and_underpowered_locked_tier(
    tmp_path: Path,
) -> None:
    leaked = _config()
    families = leaked["dgp_families"]
    assert isinstance(families, dict)
    development = families["development"]
    assert isinstance(development, dict)
    development["ligand_only"] = ["independent_two_group"]
    leaked_path = tmp_path / "leaked.json"
    _write_config(leaked_path, leaked)
    with pytest.raises(ValueError, match="leaks across"):
        load_v7_benchmark_protocol(leaked_path)

    underpowered = _config()
    tiers = underpowered["execution_tiers"]
    assert isinstance(tiers, dict)
    locked = tiers["locked"]
    assert isinstance(locked, dict)
    locked["replicates_per_family_design"] = 199
    underpowered_path = tmp_path / "underpowered.json"
    _write_config(underpowered_path, underpowered)
    with pytest.raises(ValueError, match=r"\[200, 500\]"):
        load_v7_benchmark_protocol(underpowered_path)


def test_plan_writer_binds_checksum_and_refuses_implicit_overwrite(
    tmp_path: Path,
) -> None:
    protocol = load_v7_benchmark_protocol()
    output = tmp_path / "plan"
    manifest = write_v7_benchmark_plan(
        protocol,
        output,
        phase="development",
        profile="score_primary",
        maximum_replicates=1,
    )
    table_path = output / "run_plan.tsv"
    assert manifest["rows"] == 48
    assert manifest["datasets"] == 8
    assert manifest["status"] == "planned_not_executed"
    assert manifest["output"]["sha256"] == sha256_file(table_path)
    saved = pd.read_csv(table_path, sep="\t")
    assert tuple(saved.columns) == PLAN_COLUMNS

    with pytest.raises(FileExistsError, match="pass overwrite"):
        write_v7_benchmark_plan(
            protocol,
            output,
            phase="development",
            profile="score_primary",
            maximum_replicates=1,
        )
