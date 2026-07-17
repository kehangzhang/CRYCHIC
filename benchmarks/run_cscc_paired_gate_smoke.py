"""Run the checksum-pinned Ji cSCC paired ligand-gate algorithm smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource as process_resource
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import validate_prepared_input
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.metrics.multicondition import paired_edge_effects, validate_score_table
from crychic import __version__ as crychic_version
from crychic import load_nichenet_target_prior
from crychic.attribution import GainCalibrationSpec, PenaltyTuningSpec
from crychic.core import CrychicConfig, canonical_digest
from crychic.design import balanced_contrast, context_id
from crychic.resources import (
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.sender import (
    ContrastCommonSenderParameters,
    interaction_ligand_contrast_gate,
)
from crychic.workflow import (
    AutonomousProgramUseScope,
    CrossFitArtifacts,
    CrossFitSpec,
    FoldTrainingSpec,
    run_subject_crossfit,
    write_crossfit_result,
)

SCHEMA_VERSION = "crychic-cscc-paired-gate-smoke-v2"
CONFIG_SCHEMA_VERSION = "crychic-cscc-paired-gate-smoke-config-v1"
GAIN_CALIBRATION_CONFIG_SCHEMA_VERSION = "crychic-cscc-gain-calibration-smoke-config-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DEFAULT_CONFIG = REPOSITORY_ROOT / "benchmarks/configs/cscc_paired_gate_smoke.json"
DEFAULT_GAIN_CALIBRATION_CONFIG = (
    REPOSITORY_ROOT / "benchmarks/configs/cscc_gain_calibration_smoke_v1.json"
)
DEFAULT_OUTPUT = Path("benchmark_work/cscc_paired_gate_smoke.json")

EXPECTED_BASE_SMOKE_CONFIG_SHA256 = (
    "f7cb4ce4b6cfc6762a9c636497fb8b4af34a49ad01abc7c96973186f5a50c3b4"
)
_GAIN_CALIBRATION_SMOKE_SCOPE: dict[str, object] = {
    "analysis_class": "small_scale_descriptive_gain_calibration_smoke",
    "formal_inference_allowed": False,
    "biological_validation_claim_allowed": False,
    "comparative_method_advantage_claim_allowed": False,
}
_GAIN_CALIBRATION_SMOKE_VALUES = {
    "min_inner_folds": 2,
    "min_subjects": 4,
    "min_supported_families": 1,
    "min_subjects_per_family": 4,
    "min_positive_observations": 4,
    "min_distinct_positive_gains": 3,
}

EXPECTED_H5AD_SHA256 = (
    "b15759def47df2c2aa5e1936398c9e57fab92dac6814b77d31839ae50b675b81"
)
EXPECTED_HARMONIZED_TABLE_SHA256 = (
    "e24148ad3d6ee0be0d1a71503c98934085072e22fcb46bf3f9a8f315d3424988"
)
EXPECTED_HARMONIZED_MANIFEST_SHA256 = (
    "15f5fb2711616282d95059319d8cdcbe05078c12df22b65bac953aebfae59e98"
)
EXPECTED_NICHENET_PAYLOAD_SHA256 = (
    "42a6fa3746ad3ab4ba3f03c35fe693d7d2ef0a9b28677295ec838cfc2e0f0293"
)
EXPECTED_NICHENET_MANIFEST_SHA256 = (
    "4e2a107768e5dee7457fdb5b73ff3659d1f5dbb0a1f82f09dd5ddfc12749f146"
)
EXPECTED_CELL_ID_DIGEST = (
    "e9d4a487cff6227e1af94f5e6967e181048b7bb86759e158d527153b12d13e6d"
)
EXPECTED_INTERACTION_ID_DIGEST = (
    "af50febfb9fe78358084ade42e97ef25a7c591bcfadb5855eecde1516bd4d01f"
)
EXPECTED_INTERACTION_FIELDS_DIGEST = (
    "981b73475435dd6ddfb20c9635003e799dddd9da8959bd3bb1ee0a51bfe4ed7b"
)
EXPECTED_TARGET_LINK_DIGEST = (
    "f716190c833b367d97a96162767c87eaee2a27a5e1a66277495d4154fbde2f2f"
)


class TargetLinkRecord(TypedDict):
    """One ranked NicheNet link."""

    target: str
    rank: int
    weight: float


def sha256_file(path: Path) -> str:
    """Hash a file without retaining its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return cast(Sequence[object], value)


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    result = tuple(str(item).strip() for item in _sequence(value, field=field))
    if (
        not result
        or any(not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return result


def _int_value(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _float_value(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _require_equal(value: object, expected: object, *, field: str) -> None:
    if value != expected:
        raise ValueError(f"{field} changed from the frozen smoke contract")


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise ValueError(
            f"{field} fields changed: missing={missing}, unexpected={unexpected}"
        )


def load_gain_calibration_smoke_config(
    path: str | Path = DEFAULT_GAIN_CALIBRATION_CONFIG,
) -> GainCalibrationSpec:
    """Load the separately versioned, base-config-bound descriptive policy."""

    config = dict(
        _mapping(
            json.loads(Path(path).read_text(encoding="utf-8")),
            field="gain_calibration_config",
        )
    )
    _require_exact_keys(
        config,
        {
            "schema_version",
            "base_smoke_config",
            "scope",
            "gain_calibration_spec",
        },
        field="gain_calibration_config",
    )
    _require_equal(
        config["schema_version"],
        GAIN_CALIBRATION_CONFIG_SCHEMA_VERSION,
        field="gain_calibration_config.schema_version",
    )

    base = _mapping(
        config["base_smoke_config"],
        field="gain_calibration_config.base_smoke_config",
    )
    _require_exact_keys(
        base,
        {"relative_path", "sha256"},
        field="gain_calibration_config.base_smoke_config",
    )
    _require_equal(
        base["relative_path"],
        "benchmarks/configs/cscc_paired_gate_smoke.json",
        field="gain_calibration_config.base_smoke_config.relative_path",
    )
    _require_equal(
        base["sha256"],
        EXPECTED_BASE_SMOKE_CONFIG_SHA256,
        field="gain_calibration_config.base_smoke_config.sha256",
    )
    if sha256_file(DEFAULT_CONFIG) != EXPECTED_BASE_SMOKE_CONFIG_SHA256:
        raise ValueError("base cSCC smoke configuration sha256 changed")
    load_smoke_config(DEFAULT_CONFIG)

    scope = _mapping(config["scope"], field="gain_calibration_config.scope")
    _require_exact_keys(
        scope,
        set(_GAIN_CALIBRATION_SMOKE_SCOPE),
        field="gain_calibration_config.scope",
    )
    for name, expected in _GAIN_CALIBRATION_SMOKE_SCOPE.items():
        observed = scope[name]
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"gain_calibration_config.scope.{name} changed from the "
                "descriptive smoke contract"
            )

    raw_spec = _mapping(
        config["gain_calibration_spec"],
        field="gain_calibration_config.gain_calibration_spec",
    )
    _require_exact_keys(
        raw_spec,
        set(_GAIN_CALIBRATION_SMOKE_VALUES),
        field="gain_calibration_config.gain_calibration_spec",
    )
    values: dict[str, int] = {}
    for name, expected in _GAIN_CALIBRATION_SMOKE_VALUES.items():
        observed = _int_value(
            raw_spec[name],
            field=f"gain_calibration_config.gain_calibration_spec.{name}",
        )
        _require_equal(
            observed,
            expected,
            field=f"gain_calibration_config.gain_calibration_spec.{name}",
        )
        values[name] = observed
    return GainCalibrationSpec(**values)


def load_smoke_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Load the small policy while locking only input and resource identities."""

    config = dict(
        _mapping(
            json.loads(Path(path).read_text(encoding="utf-8")),
            field="config",
        )
    )
    _require_equal(
        config.get("schema_version"),
        CONFIG_SCHEMA_VERSION,
        field="schema_version",
    )
    dataset = _mapping(config.get("dataset"), field="dataset")
    _require_equal(dataset.get("dataset_id"), "GSE144236_Ji_cSCC", field="dataset_id")
    _require_equal(
        _strings(dataset.get("cell_types"), field="dataset.cell_types"),
        ("CD1C", "Epithelial"),
        field="dataset.cell_types",
    )
    _require_equal(
        _mapping(dataset.get("conditions"), field="dataset.conditions"),
        {"positive": "Tumor", "negative": "Normal"},
        field="dataset.conditions",
    )
    h5ad = _mapping(dataset.get("expected_h5ad"), field="dataset.expected_h5ad")
    _require_equal(h5ad.get("sha256"), EXPECTED_H5AD_SHA256, field="H5AD sha256")
    _require_equal(h5ad.get("bytes"), 579531847, field="H5AD bytes")
    _require_equal(
        tuple(_sequence(h5ad.get("shape"), field="H5AD shape")),
        (47068, 32738),
        field="H5AD shape",
    )
    dataset_locks: tuple[tuple[str, object], ...] = (
        ("expected_subject_count", 8),
        ("expected_condition_count", 2),
        ("expected_cell_type_count", 2),
        ("expected_stratum_count", 32),
        ("expected_selected_cell_count", 5297),
        ("minimum_cells_per_stratum", 10),
        ("cell_cap_per_stratum", 200),
    )
    for key, expected_value in dataset_locks:
        _require_equal(dataset.get(key), expected_value, field=f"dataset.{key}")
    _require_equal(
        dataset.get("cell_cap_policy"),
        "sha256_utf8_cell_id_then_cell_id",
        field="dataset.cell_cap_policy",
    )
    _require_equal(
        dataset.get("selected_cell_id_digest_policy"),
        "canonical_digest_sorted_cell_ids",
        field="dataset.selected_cell_id_digest_policy",
    )
    _require_equal(
        dataset.get("expected_selected_cell_id_digest"),
        EXPECTED_CELL_ID_DIGEST,
        field="dataset.expected_selected_cell_id_digest",
    )

    harmonized = _mapping(
        config.get("harmonized_resource"), field="harmonized_resource"
    )
    _require_equal(
        harmonized.get("table_sha256"),
        EXPECTED_HARMONIZED_TABLE_SHA256,
        field="harmonized table sha256",
    )
    _require_equal(
        harmonized.get("manifest_sha256"),
        EXPECTED_HARMONIZED_MANIFEST_SHA256,
        field="harmonized manifest sha256",
    )
    _require_equal(
        harmonized.get("expected_selected_id_digest"),
        EXPECTED_INTERACTION_ID_DIGEST,
        field="selected interaction ID digest",
    )
    _require_equal(
        harmonized.get("expected_selected_full_fields_digest"),
        EXPECTED_INTERACTION_FIELDS_DIGEST,
        field="selected interaction fields digest",
    )

    nichenet = _mapping(config.get("nichenet_resource"), field="nichenet_resource")
    _require_equal(
        nichenet.get("payload_sha256"),
        EXPECTED_NICHENET_PAYLOAD_SHA256,
        field="NicheNet payload sha256",
    )
    _require_equal(
        nichenet.get("manifest_sha256"),
        EXPECTED_NICHENET_MANIFEST_SHA256,
        field="NicheNet manifest sha256",
    )
    nichenet_locks: tuple[tuple[str, object], ...] = (
        ("top_h5ad_observable_targets_per_ligand", 20),
        ("expected_driver_count", 14),
        ("expected_link_count", 280),
        ("expected_target_count", 171),
        ("expected_selected_link_digest", EXPECTED_TARGET_LINK_DIGEST),
    )
    for key, expected_value in nichenet_locks:
        _require_equal(
            nichenet.get(key), expected_value, field=f"nichenet_resource.{key}"
        )

    definitions = [
        _mapping(item, field="diagnostic_interactions[]")
        for item in _sequence(
            config.get("diagnostic_interactions"),
            field="diagnostic_interactions",
        )
    ]
    interaction_ids = [str(item.get("interaction_id", "")) for item in definitions]
    if len(definitions) != 14 or len(set(interaction_ids)) != 14:
        raise ValueError("diagnostic_interactions must contain 14 unique IDs")
    _require_equal(
        canonical_digest(sorted(interaction_ids)),
        EXPECTED_INTERACTION_ID_DIGEST,
        field="diagnostic_interactions ID digest",
    )

    policy = _mapping(config.get("crossfit"), field="crossfit")
    _require_equal(
        policy.get("autonomous_program_use_scope"),
        "biological_analysis",
        field="crossfit.autonomous_program_use_scope",
    )
    _require_equal(
        tuple(_sequence(policy.get("outer_allowed_n_splits"), field="outer splits")),
        (2,),
        field="crossfit.outer_allowed_n_splits",
    )
    for key in (
        "minimum_train_subjects_per_context",
        "minimum_test_subjects_per_context",
        "sender_minimum_subjects",
    ):
        _require_equal(policy.get(key), 4, field=f"crossfit.{key}")
    outer_seed = _int_value(
        policy.get("outer_fold_partition_seed"),
        field="crossfit.outer_fold_partition_seed",
    )
    random_seed = _int_value(policy.get("random_seed"), field="crossfit.random_seed")
    if outer_seed == random_seed:
        raise ValueError("outer partition seed must be explicit and independent")
    return config


def _verify_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(f"{label} byte size changed")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{label} sha256 changed")
    return {"bytes": observed_bytes, "sha256": observed_sha256}


def deterministic_cell_cap(
    obs: pd.DataFrame,
    *,
    cell_types: Sequence[str],
    conditions: Sequence[str],
    minimum_cells: int,
    cap: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose complete subjects and cap each stratum by SHA256(cell ID)."""

    required = {"subject_id", "condition", "cell_type"}
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"obs lacks cell-cap columns: {sorted(missing)}")
    if not obs.index.is_unique:
        raise ValueError("cell IDs must be unique")
    if minimum_cells < 1 or cap < minimum_cells:
        raise ValueError("cell cap must be >= the positive minimum cell count")
    if obs.loc[:, list(required)].isna().any(axis=None):
        raise ValueError("cell-cap metadata contains null values")

    selected_types = tuple(map(str, cell_types))
    selected_conditions = tuple(map(str, conditions))
    if len(selected_types) != 2 or len(set(selected_types)) != 2:
        raise ValueError("cell cap requires exactly two unique cell types")
    if len(selected_conditions) != 2 or len(set(selected_conditions)) != 2:
        raise ValueError("cell cap requires exactly two unique conditions")
    frame = pd.DataFrame(
        {
            "subject_id": obs["subject_id"].astype(str).to_numpy(),
            "condition": obs["condition"].astype(str).to_numpy(),
            "cell_type": obs["cell_type"].astype(str).to_numpy(),
            "cell_id": obs.index.astype(str).to_numpy(),
            "position": np.arange(len(obs), dtype=np.int64),
        }
    )
    counts = frame.groupby(
        ["subject_id", "condition", "cell_type"], observed=True, sort=True
    ).size()
    subjects = tuple(
        subject
        for subject in sorted(frame["subject_id"].unique())
        if all(
            int(counts.get((subject, condition, cell_type), 0)) >= minimum_cells
            for condition in selected_conditions
            for cell_type in selected_types
        )
    )
    if not subjects:
        raise ValueError("no subject has complete cell support")

    keep = (
        frame["subject_id"].isin(subjects)
        & frame["condition"].isin(selected_conditions)
        & frame["cell_type"].isin(selected_types)
    )
    eligible = frame.loc[keep].copy()
    selected_positions: list[int] = []
    selected_ids: list[str] = []
    source_counts: list[int] = []
    retained_counts: list[int] = []
    grouped = eligible.groupby(
        ["subject_id", "condition", "cell_type"], observed=True, sort=True
    )
    for _, group in grouped:
        ranked = sorted(
            zip(group["cell_id"], group["position"], strict=True),
            key=lambda item: (
                hashlib.sha256(str(item[0]).encode("utf-8")).digest(),
                str(item[0]),
            ),
        )
        retained = ranked[:cap]
        selected_ids.extend(str(cell_id) for cell_id, _ in retained)
        selected_positions.extend(int(position) for _, position in retained)
        source_counts.append(len(ranked))
        retained_counts.append(len(retained))

    expected_strata = len(subjects) * len(selected_conditions) * len(selected_types)
    if len(source_counts) != expected_strata:
        raise ValueError("eligible subset lacks a complete subject-condition-type grid")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("cell cap selected duplicate cell IDs")
    audit = {
        "n_subjects": len(subjects),
        "subject_set_digest": canonical_digest(list(subjects)),
        "n_conditions": len(selected_conditions),
        "n_cell_types": len(selected_types),
        "n_strata": len(source_counts),
        "n_selected_cells": len(selected_ids),
        "selected_cell_id_digest": canonical_digest(sorted(selected_ids)),
        "cell_cap_policy": "sha256_utf8_cell_id_then_cell_id",
        "selected_cell_id_digest_policy": "canonical_digest_sorted_cell_ids",
        "minimum_source_cells_per_stratum": min(source_counts),
        "maximum_source_cells_per_stratum": max(source_counts),
        "minimum_selected_cells_per_stratum": min(retained_counts),
        "maximum_selected_cells_per_stratum": max(retained_counts),
        "cell_cap_per_stratum": cap,
    }
    return np.asarray(sorted(selected_positions), dtype=np.int64), audit


def prepare_input(
    input_path: Path,
    *,
    dataset_config: Mapping[str, object],
) -> tuple[ad.AnnData, tuple[str, ...], dict[str, object]]:
    """Verify the H5AD, select cells from backed metadata, and retain all genes."""

    expected = _mapping(dataset_config["expected_h5ad"], field="expected_h5ad")
    file_audit = _verify_file(
        input_path,
        expected_bytes=_int_value(expected["bytes"], field="expected_h5ad.bytes"),
        expected_sha256=str(expected["sha256"]),
        label="paired cSCC H5AD",
    )
    expected_shape = tuple(
        _int_value(value, field="expected_h5ad.shape[]")
        for value in _sequence(expected["shape"], field="expected_h5ad.shape")
    )
    backed = ad.read_h5ad(input_path, backed="r")
    try:
        validate_prepared_input(
            backed,
            sample_key="sample_id",
            subject_key="subject_id",
            cell_type_key="cell_type",
            context_keys=("condition",),
        )
        if backed.shape != expected_shape:
            raise ValueError("paired cSCC H5AD shape changed")
        genes = tuple(map(str, backed.var_names))
        if len(set(genes)) != len(genes) or genes != tuple(sorted(genes)):
            raise ValueError("paired cSCC genes must remain unique and sorted")
        if "counts" not in backed.layers:
            raise ValueError("paired cSCC H5AD lacks the raw counts layer")
        counts = backed.layers["counts"]
        if getattr(counts, "format", None) != "csr" or str(counts.dtype) != "int32":
            raise ValueError("paired cSCC counts must remain CSR int32")

        cell_types = _strings(dataset_config["cell_types"], field="cell_types")
        conditions_config = _mapping(dataset_config["conditions"], field="conditions")
        conditions = (
            str(conditions_config["negative"]),
            str(conditions_config["positive"]),
        )
        positions, cap_audit = deterministic_cell_cap(
            backed.obs,
            cell_types=cell_types,
            conditions=conditions,
            minimum_cells=_int_value(
                dataset_config["minimum_cells_per_stratum"],
                field="minimum_cells_per_stratum",
            ),
            cap=_int_value(
                dataset_config["cell_cap_per_stratum"],
                field="cell_cap_per_stratum",
            ),
        )
        for key, expected_value in (
            ("n_subjects", dataset_config["expected_subject_count"]),
            ("n_conditions", dataset_config["expected_condition_count"]),
            ("n_cell_types", dataset_config["expected_cell_type_count"]),
            ("n_strata", dataset_config["expected_stratum_count"]),
            ("n_selected_cells", dataset_config["expected_selected_cell_count"]),
            (
                "selected_cell_id_digest",
                dataset_config["expected_selected_cell_id_digest"],
            ),
        ):
            if cap_audit[key] != expected_value:
                raise ValueError(f"deterministic cSCC cell cap changed: {key}")
        subset = backed[positions, :].to_memory()
    finally:
        backed.file.close()

    if subset.n_vars != expected_shape[1] or tuple(map(str, subset.var_names)) != genes:
        raise RuntimeError("cell cap changed the full-transcriptome gene axis")
    return (
        subset,
        genes,
        {
            **file_audit,
            "source_shape": list(expected_shape),
            "analysis_shape": [int(subset.n_obs), int(subset.n_vars)],
            **cap_audit,
            "counts_layer": "counts",
            "counts_dtype": "int32",
            "normalization_denominator_gene_count": int(subset.n_vars),
            "full_gene_axis_retained_for_fold_cpm": True,
        },
    )


def _interaction_fields(interaction: Interaction) -> dict[str, object]:
    return {
        "interaction_id": interaction.interaction_id,
        "source_interaction_id": interaction.source_interaction_id,
        "ligand_name": interaction.ligand_name,
        "ligand_subunits": list(interaction.ligand_subunits),
        "receptor_name": interaction.receptor_name,
        "receptor_subunits": list(interaction.receptor_subunits),
        "ligand_is_complex": interaction.ligand_is_complex,
        "receptor_is_complex": interaction.receptor_is_complex,
        "direction": interaction.direction,
        "source": interaction.source,
        "version": interaction.version,
        "evidence": list(interaction.evidence),
    }


def select_anchor_interactions(
    bundle: ResourceBundle,
    definitions: Sequence[Mapping[str, object]],
    *,
    expected_id_digest: str,
    expected_fields_digest: str,
) -> tuple[ResourceBundle, list[dict[str, object]]]:
    """Select exact harmonized IDs without reconstructing resource semantics."""

    by_id = {item.interaction_id: item for item in bundle.interactions}
    selected: list[Interaction] = []
    manifest: list[dict[str, object]] = []
    for definition in definitions:
        interaction_id = str(definition["interaction_id"])
        interaction = by_id.get(interaction_id)
        if interaction is None:
            raise ValueError(f"harmonized resource lacks anchor {interaction_id}")
        ligand = str(definition["ligand"])
        receptor = str(definition["receptor"])
        evidence = tuple(
            sorted(
                (
                    f"cellchat:{definition['cellchat_source_interaction_id']}",
                    f"cellphonedb:{definition['cellphonedb_source_interaction_id']}",
                )
            )
        )
        expected_identity = (
            interaction_id,
            ligand,
            (ligand,),
            receptor,
            (receptor,),
            False,
            False,
            "Ligand-Receptor",
            evidence,
        )
        observed_identity = (
            interaction.source_interaction_id,
            interaction.ligand_name,
            interaction.ligand_subunits,
            interaction.receptor_name,
            interaction.receptor_subunits,
            interaction.ligand_is_complex,
            interaction.receptor_is_complex,
            interaction.direction,
            interaction.evidence,
        )
        if observed_identity != expected_identity:
            raise ValueError(
                f"harmonized molecular identity changed for {interaction_id}"
            )
        selected.append(interaction)
        manifest.append(
            {
                "diagnostic_id": str(definition["diagnostic_id"]),
                "interaction_id": interaction_id,
                "ligand": ligand,
                "receptor": receptor,
            }
        )

    ids = sorted(item.interaction_id for item in selected)
    fields = sorted(
        (_interaction_fields(item) for item in selected),
        key=lambda row: str(row["interaction_id"]),
    )
    if len(selected) != len(definitions) or len(set(ids)) != len(definitions):
        raise ValueError("anchor selection is not one-to-one")
    if canonical_digest(ids) != expected_id_digest:
        raise ValueError("anchor interaction ID digest changed")
    if canonical_digest(fields) != expected_fields_digest:
        raise ValueError("anchor interaction field digest changed")
    genes = {
        gene
        for item in selected
        for gene in (*item.ligand_subunits, *item.receptor_subunits)
    }
    return (
        replace(
            bundle,
            interactions=tuple(selected),
            mapping_report=MappingReport(
                source_rows=bundle.mapping_report.source_rows,
                loaded_rows=len(selected),
                mapped_entities=len(genes),
                notes=(
                    "pre_specified_cscc_fourteen_interaction_smoke_subset",
                    "source_rows_checksum_verified_before_selection",
                ),
            ),
        ),
        sorted(manifest, key=lambda row: str(row["interaction_id"])),
    )


def _top_observable_targets(
    prior: TargetPrior,
    *,
    ligand: str,
    top_n: int,
    available_genes: set[str],
) -> list[TargetLinkRecord]:
    if ligand not in prior.driver_ids:
        raise ValueError(f"NicheNet target prior has no driver {ligand!r}")
    ranks = prior.ranks
    if ranks is None:
        raise ValueError("cSCC smoke requires source-ranked NicheNet links")
    driver_index = prior.driver_ids.index(ligand)
    start, stop = prior.indptr[driver_index : driver_index + 2]
    records = [
        TargetLinkRecord(
            target=prior.target_ids[prior.target_indices[offset]],
            rank=ranks[offset],
            weight=float(prior.weights[offset]),
        )
        for offset in range(start, stop)
    ]
    records.sort(key=lambda row: (row["rank"], row["target"]))
    selected = [row for row in records if row["target"] in available_genes][:top_n]
    if len(selected) != top_n or len({row["target"] for row in selected}) != top_n:
        raise ValueError(
            f"NicheNet {ligand} does not provide {top_n} unique observable targets"
        )
    return selected


def build_smoke_target_prior(
    source: TargetPrior,
    *,
    definitions: Sequence[Mapping[str, object]],
    available_genes: Sequence[str],
    top_n: int,
    expected_driver_count: int,
    expected_link_count: int,
    expected_target_count: int,
    expected_link_digest: str,
) -> tuple[TargetPrior, dict[str, object]]:
    """Freeze the first ranked H5AD-observable links for every anchor ligand."""

    if source.driver_kind != "ligand":
        raise ValueError("cSCC smoke requires a ligand-indexed target prior")
    ligands = tuple(sorted({str(item["ligand"]) for item in definitions}))
    if len(ligands) != len(definitions):
        raise ValueError("each cSCC anchor must have one unique ligand driver")
    available = set(map(str, available_genes))
    selected_by_ligand = {
        ligand: _top_observable_targets(
            source,
            ligand=ligand,
            top_n=top_n,
            available_genes=available,
        )
        for ligand in ligands
    }
    target_ids = tuple(
        sorted(
            {
                row["target"]
                for records in selected_by_ligand.values()
                for row in records
            }
        )
    )
    target_index = {target: index for index, target in enumerate(target_ids)}
    target_indices: list[int] = []
    weights: list[float] = []
    ranks: list[int] = []
    indptr = [0]
    link_manifest: list[dict[str, object]] = []
    for ligand in ligands:
        for record in selected_by_ligand[ligand]:
            target_indices.append(target_index[record["target"]])
            weights.append(record["weight"])
            ranks.append(record["rank"])
            link_manifest.append({"ligand": ligand, **record})
        indptr.append(len(target_indices))

    observed = (len(ligands), len(weights), len(target_ids))
    expected = (expected_driver_count, expected_link_count, expected_target_count)
    if observed != expected:
        raise ValueError(
            f"diagnostic TargetPrior coverage changed: {observed} != {expected}"
        )
    selected_link_digest = canonical_digest(link_manifest)
    if selected_link_digest != expected_link_digest:
        raise ValueError("diagnostic TargetPrior link identity changed")
    prior = TargetPrior(
        resource_id=f"{source.resource_id}_cscc_paired_gate_smoke",
        version=f"{source.version}+fourteen_ligand_top{top_n}_h5ad_observable",
        species=source.species,
        gene_namespace=source.gene_namespace,
        driver_kind=source.driver_kind,
        target_ids=target_ids,
        driver_ids=ligands,
        indptr=tuple(indptr),
        target_indices=tuple(target_indices),
        weights=tuple(weights),
        ranks=tuple(ranks),
        direction=source.direction,
        evidence=(
            f"{source.evidence}; checksum-pinned first {top_n} H5AD-observable "
            "links per cSCC anchor ligand by source rank"
        ),
        mapping_report=MappingReport(
            source_rows=source.mapping_report.source_rows,
            loaded_rows=len(weights),
            mapped_entities=len(ligands) + len(target_ids),
            notes=(
                "pre_specified_cscc_fourteen_ligand_target_prior",
                f"selected_link_digest={selected_link_digest}",
            ),
        ),
        manifest_digest=source.manifest_digest,
    )
    return prior, {
        "selection_policy": "first_20_h5ad_observable_links_per_ligand_by_source_rank",
        "n_drivers": len(ligands),
        "n_links": len(weights),
        "n_targets": len(target_ids),
        "selected_target_link_digest": selected_link_digest,
    }


def build_crossfit_spec(
    config: Mapping[str, object],
    *,
    gain_calibration_spec: GainCalibrationSpec | None = None,
) -> tuple[CrychicConfig, CrossFitSpec]:
    """Build the explicit two-fold Tumor-minus-Normal smoke specification."""

    dataset = _mapping(config["dataset"], field="dataset")
    conditions = _mapping(dataset["conditions"], field="dataset.conditions")
    policy = _mapping(config["crossfit"], field="crossfit")
    sender_parameters = ContrastCommonSenderParameters(
        min_subjects=_int_value(
            policy["sender_minimum_subjects"], field="sender_minimum_subjects"
        ),
        prevalence_threshold=_float_value(
            policy["sender_prevalence_threshold"],
            field="sender_prevalence_threshold",
        ),
        softmax_temperature=_float_value(
            policy["sender_softmax_temperature"],
            field="sender_softmax_temperature",
        ),
        ligand_contrast_confidence_level=_float_value(
            policy["ligand_contrast_confidence_level"],
            field="ligand_contrast_confidence_level",
        ),
        ligand_contrast_minimum_effect=_float_value(
            policy["ligand_contrast_minimum_effect"],
            field="ligand_contrast_minimum_effect",
        ),
    )
    tuning = PenaltyTuningSpec(
        lambda1_fractions=tuple(
            _float_value(value, field="penalty_lambda1_fractions[]")
            for value in _sequence(
                policy["penalty_lambda1_fractions"],
                field="penalty_lambda1_fractions",
            )
        ),
        lambda2_fractions=tuple(
            _float_value(value, field="penalty_lambda2_fractions[]")
            for value in _sequence(
                policy["penalty_lambda2_fractions"],
                field="penalty_lambda2_fractions",
            )
        ),
        inner_allowed_n_splits=tuple(
            _int_value(value, field="inner_allowed_n_splits[]")
            for value in _sequence(
                policy["inner_allowed_n_splits"],
                field="inner_allowed_n_splits",
            )
        ),
        min_inner_train_subjects_per_context=_int_value(
            policy["minimum_inner_train_subjects_per_context"],
            field="minimum_inner_train_subjects_per_context",
        ),
        min_inner_validation_subjects_per_context=_int_value(
            policy["minimum_inner_validation_subjects_per_context"],
            field="minimum_inner_validation_subjects_per_context",
        ),
        root_seed=_int_value(policy["random_seed"], field="random_seed"),
    )
    contrast = balanced_contrast(
        (str(conditions["positive"]),),
        (str(conditions["negative"]),),
        name="tumor_vs_normal",
    )
    spec = CrossFitSpec(
        contrasts=(contrast,),
        outer_fold_partition_seed=_int_value(
            policy["outer_fold_partition_seed"],
            field="outer_fold_partition_seed",
        ),
        training_spec=FoldTrainingSpec(
            min_cells=_int_value(
                policy["minimum_cells_per_sample_cell_type"],
                field="minimum_cells_per_sample_cell_type",
            ),
            min_pooled_availability=_float_value(
                policy["minimum_pooled_availability"],
                field="minimum_pooled_availability",
            ),
            max_interactions=None,
            sender_parameters=sender_parameters,
        ),
        allowed_n_splits=tuple(
            _int_value(value, field="outer_allowed_n_splits[]")
            for value in _sequence(
                policy["outer_allowed_n_splits"],
                field="outer_allowed_n_splits",
            )
        ),
        min_train_subjects_per_context=_int_value(
            policy["minimum_train_subjects_per_context"],
            field="minimum_train_subjects_per_context",
        ),
        min_test_subjects_per_context=_int_value(
            policy["minimum_test_subjects_per_context"],
            field="minimum_test_subjects_per_context",
        ),
        receptor_gate_threshold=_float_value(
            policy["receptor_gate_threshold"], field="receptor_gate_threshold"
        ),
        family_cosine_threshold=_float_value(
            policy["family_cosine_threshold"], field="family_cosine_threshold"
        ),
        downstream_minimum_scale=_float_value(
            policy["downstream_minimum_scale"], field="downstream_minimum_scale"
        ),
        autonomous_program_resource=None,
        autonomous_program_use_scope=cast(
            AutonomousProgramUseScope,
            policy["autonomous_program_use_scope"],
        ),
        penalty_tuning_spec=tuning,
        gain_calibration_spec=gain_calibration_spec,
    )
    crychic_config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=_int_value(policy["random_seed"], field="random_seed"),
    )
    return crychic_config, spec


def _missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _finite_float(value: object) -> float | None:
    if _missing(value):
        return None
    result = float(cast(float, value))
    return result if math.isfinite(result) else None


def _count_values(values: Sequence[object]) -> dict[str, int]:
    counts: Counter[str] = Counter(
        "none" if _missing(value) else str(value) for value in values
    )
    return dict(sorted(counts.items()))


def _expected_fold_keys(
    artifacts: CrossFitArtifacts,
    interaction_ids: Sequence[str],
) -> set[tuple[str, str, str]]:
    return {
        (fold.fold_id, receiver, interaction_id)
        for fold in artifacts.folds
        for receiver in fold.training.cell_type_ids
        for interaction_id in interaction_ids
    }


def compact_fold_supports(
    artifacts: CrossFitArtifacts,
    *,
    interaction_manifest: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return exactly one training-fold support row per receiver and anchor."""

    diagnostics = {str(item["interaction_id"]): item for item in interaction_manifest}
    records: list[dict[str, object]] = []
    keys: list[tuple[str, str, str]] = []
    for fold in artifacts.folds:
        for functional in fold.training.sender_functionals:
            for support in functional.contrast_supports:
                definition = diagnostics.get(support.interaction_id)
                if definition is None:
                    continue
                key = (fold.fold_id, support.receiver, support.interaction_id)
                keys.append(key)
                gate = interaction_ligand_contrast_gate(
                    functional,
                    support.receiver,
                    support.interaction_id,
                )
                records.append(
                    {
                        "fold_id": fold.fold_id,
                        "receiver": support.receiver,
                        "diagnostic_id": str(definition["diagnostic_id"]),
                        "interaction_id": support.interaction_id,
                        "n_complete": support.n_complete,
                        "minimum_complete_subjects": support.minimum_complete_subjects,
                        "mean_effect": _finite_float(support.mean_effect),
                        "holm_adjusted_p_value": _finite_float(
                            support.holm_adjusted_p_value
                        ),
                        "status": support.status.value,
                        "reason_code": support.reason_code,
                        "ligand_contrast_gate": gate.gate,
                        "ligand_contrast_gate_status": gate.status.value,
                        "ligand_contrast_gate_reason_code": gate.reason_code,
                    }
                )
    expected = _expected_fold_keys(artifacts, tuple(diagnostics))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("support rows do not have exact fold-receiver-anchor coverage")
    return sorted(
        records,
        key=lambda row: (
            str(row["fold_id"]),
            str(row["receiver"]),
            str(row["diagnostic_id"]),
        ),
    )


def summarize_fold_supports(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Count support across unique folds; identical duplicate rows cannot inflate it."""

    unique: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in records:
        key = (
            str(row["fold_id"]),
            str(row["receiver"]),
            str(row["interaction_id"]),
        )
        previous = unique.get(key)
        if previous is not None and canonical_digest(previous) != canonical_digest(row):
            raise ValueError("conflicting duplicate fold support row")
        unique[key] = row
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in unique.values():
        key = (
            str(row["receiver"]),
            str(row["diagnostic_id"]),
            str(row["interaction_id"]),
        )
        groups.setdefault(key, []).append(row)
    return [
        {
            "receiver": receiver,
            "diagnostic_id": diagnostic_id,
            "interaction_id": interaction_id,
            "n_unique_folds": len(rows),
            "support_status_counts": _count_values([row.get("status") for row in rows]),
            "support_reason_counts": _count_values(
                [row.get("reason_code") for row in rows]
            ),
            "ligand_contrast_gate_counts": _count_values(
                [row.get("ligand_contrast_gate") for row in rows]
            ),
        }
        for (receiver, diagnostic_id, interaction_id), rows in sorted(groups.items())
    ]


def compact_receptor_gates(
    artifacts: CrossFitArtifacts,
    *,
    interaction_manifest: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return exact fold-local receiver receptor eligibility for every anchor."""

    diagnostics = {str(item["interaction_id"]): item for item in interaction_manifest}
    records: list[dict[str, object]] = []
    keys: list[tuple[str, str, str]] = []
    for fold in artifacts.folds:
        for model in fold.receiver_family_models:
            artifact = model.receiver_family_artifact
            driver_by_interaction = dict(artifact.driver_by_interaction)
            gate_by_driver = dict(artifact.receptor_gates)
            source_basis = artifact.source_basis
            eligible_by_driver = dict(
                zip(
                    source_basis.driver_ids,
                    source_basis.receptor_eligible.tolist(),
                    strict=True,
                )
            )
            for interaction_id, definition in diagnostics.items():
                driver = driver_by_interaction.get(interaction_id)
                if (
                    driver is None
                    or driver not in gate_by_driver
                    or driver not in eligible_by_driver
                ):
                    raise ValueError("receiver artifact lacks an anchor receptor gate")
                key = (fold.fold_id, artifact.receiver, interaction_id)
                keys.append(key)
                records.append(
                    {
                        "fold_id": fold.fold_id,
                        "receiver": artifact.receiver,
                        "diagnostic_id": str(definition["diagnostic_id"]),
                        "interaction_id": interaction_id,
                        "driver_id": driver,
                        "receptor": str(definition["receptor"]),
                        "receptor_gate": float(gate_by_driver[driver]),
                        "receptor_gate_threshold": source_basis.receptor_gate_threshold,
                        "receptor_eligible": bool(eligible_by_driver[driver]),
                    }
                )
    expected = _expected_fold_keys(artifacts, tuple(diagnostics))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("receptor gates lack exact fold-receiver-anchor coverage")
    return sorted(
        records,
        key=lambda row: (
            str(row["fold_id"]),
            str(row["receiver"]),
            str(row["diagnostic_id"]),
        ),
    )


def _fold_partitions(artifacts: CrossFitArtifacts) -> list[dict[str, object]]:
    records = []
    for fold in artifacts.folds:
        identity = {
            "training": sorted(fold.training.training_subject_ids),
            "heldout": sorted(fold.application.heldout_subject_ids),
        }
        records.append(
            {
                "fold_id": fold.fold_id,
                "n_training_subjects": len(identity["training"]),
                "n_heldout_subjects": len(identity["heldout"]),
                "subject_partition_digest": canonical_digest(identity),
            }
        )
    return sorted(records, key=lambda row: str(row["fold_id"]))


def compact_not_estimable_metrics(
    artifacts: CrossFitArtifacts,
) -> dict[str, object]:
    """Summarize current incomplete-stage status without exporting score tables."""

    manifest = artifacts.to_manifest()
    training_models = [
        model for fold in artifacts.folds for model in fold.receiver_incremental_models
    ]
    applications = [
        application
        for fold in artifacts.folds
        for application in fold.receiver_incremental_applications
    ]
    receiver_coverage = artifacts.oof_receiver_coverage
    return {
        "crossfit_id": artifacts.crossfit_id,
        "crossfit_spec_id": artifacts.spec.spec_id,
        "certification_status": artifacts.certification_status,
        "completed_stage_oof_verified": artifacts.completed_stage_oof_verified,
        "complete_pipeline_oof_certified": artifacts.is_oof_certified,
        "remaining_stages": manifest["remaining_stages"],
        "n_receiver_models": len(training_models),
        "training_diagnostic_status_counts": _count_values(
            [model.diagnostic_status for model in training_models]
        ),
        "training_official_status_counts": _count_values(
            [model.official_incremental_status for model in training_models]
        ),
        "training_reason_counts": _count_values(
            [model.reason_code for model in training_models]
        ),
        "heldout_diagnostic_status_counts": _count_values(
            [application.diagnostic_status for application in applications]
        ),
        "heldout_official_status_counts": _count_values(
            [application.official_incremental_status for application in applications]
        ),
        "heldout_reason_counts": _count_values(
            [application.reason_code for application in applications]
        ),
        "family_common_functional_status_counts": manifest[
            "family_common_functional_status_counts"
        ],
        "family_common_application_status_counts": manifest[
            "family_common_application_status_counts"
        ],
        "n_oof_sender_coverage_rows": len(artifacts.oof_coverage),
        "n_oof_receiver_coverage_rows": len(receiver_coverage),
        "oof_receiver_diagnostic_status_counts": _count_values(
            receiver_coverage["diagnostic_status"].tolist()
        ),
        "oof_receiver_diagnostic_reason_counts": _count_values(
            receiver_coverage["diagnostic_reason_code"].tolist()
        ),
        "oof_receiver_official_status_counts": _count_values(
            receiver_coverage["official_incremental_status"].tolist()
        ),
        "oof_receiver_official_reason_counts": _count_values(
            receiver_coverage["reason_code"].tolist()
        ),
    }


def _diagnostic_sender_score_table(
    sender_scores: pd.DataFrame,
    *,
    interaction_manifest: Sequence[Mapping[str, object]],
    resource_id: str,
    resource_version: str,
    mode: str,
    context_label_by_id: Mapping[str, str],
) -> pd.DataFrame:
    """Map one CRYCHIC diagnostic mode to the shared paired-metric contract."""

    required = {
        "sample_id",
        "subject_id",
        "context_id",
        "sender",
        "receiver",
        "interaction_id",
        "mode",
        "sender_resolved_strength",
        "status",
    }
    missing = required.difference(sender_scores.columns)
    if missing:
        raise ValueError(
            f"CRYCHIC sender diagnostic scores lack columns: {sorted(missing)}"
        )
    definitions = {
        str(item["interaction_id"]): (str(item["ligand"]), str(item["receptor"]))
        for item in interaction_manifest
    }
    selected = sender_scores.loc[
        sender_scores["mode"].astype(str).eq(mode), list(required)
    ].copy()
    if selected.empty:
        raise ValueError(f"CRYCHIC sender diagnostic mode {mode!r} is empty")
    selected["interaction_id"] = selected["interaction_id"].astype(str)
    unknown = set(selected["interaction_id"]).difference(definitions)
    if unknown:
        raise ValueError(
            "CRYCHIC sender scores contain interactions outside anchors: "
            f"{sorted(unknown)}"
        )
    duplicate_key = ["sample_id", "sender", "receiver", "interaction_id"]
    if selected.duplicated(duplicate_key).any():
        raise ValueError("CRYCHIC sender diagnostic scores contain duplicate edge rows")
    status_map = {
        "ok": "observed",
        "structural_zero": "observed",
        "not_estimable": "not_estimable",
    }
    statuses = selected["status"].astype(str)
    invalid_statuses = set(statuses).difference(status_map)
    if invalid_statuses:
        raise ValueError(
            "CRYCHIC sender scores contain unsupported statuses: "
            f"{sorted(invalid_statuses)}"
        )
    metric_status = statuses.map(status_map)
    score = pd.to_numeric(selected["sender_resolved_strength"], errors="coerce")
    score = score.where(metric_status.eq("observed"))
    edge_records = sorted(
        {
            (
                str(row.sender),
                str(row.receiver),
                str(row.interaction_id),
                *definitions[str(row.interaction_id)],
            )
            for row in selected.itertuples(index=False)
        }
    )
    universe_id = canonical_digest(
        {
            "analysis_track": "lr_stlr",
            "dataset": "GSE144236_Ji_cSCC",
            "edges": edge_records,
            "mode": mode,
            "resource_id": resource_id,
            "resource_version": resource_version,
        }
    )
    raw_context = selected["context_id"].astype(str)
    context = raw_context.map(
        lambda value: context_label_by_id.get(str(value), str(value))
    )
    result = pd.DataFrame(
        {
            "dataset": "GSE144236_Ji_cSCC",
            "method": "crychic",
            "method_version": crychic_version,
            "analysis_track": "lr_stlr",
            "resource": resource_id,
            "resource_version": resource_version,
            "resource_mode": "H-common",
            "score_semantics": f"{mode}_sender_resolved_strength_diagnostic",
            "universe_id": universe_id,
            "contrast": "tumor_vs_normal",
            "sample_id": selected["sample_id"].astype(str),
            "subject_id": selected["subject_id"].astype(str),
            "context": context,
            "sender": selected["sender"].astype(str),
            "receiver": selected["receiver"].astype(str),
            "interaction_id": selected["interaction_id"],
            "ligand": selected["interaction_id"].map(
                lambda value: definitions[str(value)][0]
            ),
            "receptor": selected["interaction_id"].map(
                lambda value: definitions[str(value)][1]
            ),
            "score": score,
            "score_direction": "higher",
            "status": metric_status,
            "universe_member": True,
            "universe_size": len(edge_records),
        }
    )
    if set(result["context"]) != {"Normal", "Tumor"}:
        raise ValueError("CRYCHIC sender score contexts must be Normal and Tumor")
    return cast(pd.DataFrame, validate_score_table(result))


def compact_diagnostic_score_summary(
    artifacts: CrossFitArtifacts,
    *,
    interaction_manifest: Sequence[Mapping[str, object]],
    resource_id: str,
    resource_version: str,
) -> dict[str, object]:
    """Return deidentified paired effects from noncertified held-out scores."""

    frames = [
        application.sender_scores
        for fold in artifacts.folds
        for application in fold.family_common_applications
    ]
    if not frames:
        raise ValueError("CRYCHIC cSCC smoke has no family-common sender score tables")
    sender_scores = pd.concat(frames, ignore_index=True)
    context_label_by_id = {
        str(context_id({"condition": label}, ("condition",))): label
        for label in ("Normal", "Tumor")
    }
    mode_summaries: list[dict[str, object]] = []
    effect_records: list[dict[str, object]] = []
    for mode in ("state", "ecosystem"):
        scores = _diagnostic_sender_score_table(
            sender_scores,
            interaction_manifest=interaction_manifest,
            resource_id=resource_id,
            resource_version=resource_version,
            mode=mode,
            context_label_by_id=context_label_by_id,
        )
        effects = paired_edge_effects(
            scores,
            reference="Normal",
            target="Tumor",
            min_pairs=3,
            contrast="tumor_vs_normal",
            validated=True,
        )
        for row in effects.itertuples(index=False):
            effect_records.append(
                {
                    "mode": mode,
                    "sender": str(row.sender),
                    "receiver": str(row.receiver),
                    "interaction_id": str(row.interaction_id),
                    "ligand": str(row.ligand),
                    "receptor": str(row.receptor),
                    "effect": _finite_float(row.effect),
                    "median_effect": _finite_float(row.median_effect),
                    "direction_consistency": _finite_float(row.direction_consistency),
                    "direction_comparable_pairs": int(row.direction_comparable_pairs),
                    "n_pairs": int(row.n_pairs),
                    "status": str(row.status),
                    "reason_code": (
                        None if _missing(row.reason_code) else str(row.reason_code)
                    ),
                }
            )
        mode_summaries.append(
            {
                "mode": mode,
                "n_samples": int(scores["sample_id"].nunique()),
                "n_subjects": int(scores["subject_id"].nunique()),
                "universe_id": str(scores["universe_id"].iloc[0]),
                "universe_size": int(scores["universe_size"].iloc[0]),
                "input_status_counts": _count_values(
                    sender_scores.loc[
                        sender_scores["mode"].astype(str).eq(mode), "status"
                    ].tolist()
                ),
                "effect_status_counts": _count_values(effects["status"].tolist()),
            }
        )
    penalties = []
    for fold in artifacts.folds:
        for model in fold.receiver_incremental_models:
            tuning = model.penalty_tuning_artifact
            selected = None if tuning is None else tuning.selected_candidate
            penalties.append(
                {
                    "fold_id": fold.fold_id,
                    "receiver": model.receiver,
                    "diagnostic_status": model.diagnostic_status,
                    "official_status": model.official_incremental_status,
                    "selected_lambda1_fraction": (
                        None if selected is None else selected.lambda1_fraction
                    ),
                    "selected_lambda2_fraction": (
                        None if selected is None else selected.lambda2_fraction
                    ),
                }
            )
    return {
        "scope": "exploratory_unadjusted_noncertified_paired_rank_effects",
        "effect_semantics": "tumor_minus_normal_comparison_strength",
        "inferential_fields_available": [],
        "raw_sample_and_subject_rows_exported": False,
        "modes": mode_summaries,
        "selected_penalties": penalties,
        "paired_effects": sorted(
            effect_records,
            key=lambda row: (
                str(row["mode"]),
                str(row["sender"]),
                str(row["receiver"]),
                str(row["interaction_id"]),
            ),
        ),
    }


def _definitions(config: Mapping[str, object]) -> list[Mapping[str, object]]:
    return [
        _mapping(item, field="diagnostic_interactions[]")
        for item in _sequence(
            config["diagnostic_interactions"], field="diagnostic_interactions"
        )
    ]


def _portable_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _gain_calibration_configuration_summary(
    *,
    config_path: Path,
    workspace_root: Path,
    spec: GainCalibrationSpec,
) -> dict[str, object]:
    loaded = load_gain_calibration_smoke_config(config_path)
    if loaded.spec_id != spec.spec_id:
        raise ValueError(
            "explicit gain calibration specification differs from its smoke config"
        )
    raw = _mapping(
        json.loads(config_path.read_text(encoding="utf-8")),
        field="gain_calibration_config",
    )
    return {
        "path": _portable_path(config_path, workspace_root),
        "sha256": sha256_file(config_path),
        "schema_version": GAIN_CALIBRATION_CONFIG_SCHEMA_VERSION,
        "base_smoke_config_sha256": EXPECTED_BASE_SMOKE_CONFIG_SHA256,
        "gain_calibration_spec": spec.to_dict(),
        "scope": dict(_mapping(raw["scope"], field="gain_calibration_config.scope")),
    }


def _crossfit_result_summary(
    *,
    path: Path,
    workspace_root: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    if manifest.get("schema_version") not in {"5.0.0", "6.0.0"}:
        raise ValueError(
            "cSCC calibration smoke requires cross-fit result schema v5 or v6"
        )
    eligibility: list[dict[str, object]] = []
    collections = _sequence(
        manifest.get("contrast_common_collections"),
        field="crossfit_result.contrast_common_collections",
    )
    for raw_collection in collections:
        collection = _mapping(
            raw_collection,
            field="crossfit_result.contrast_common_collections[]",
        )
        applications = _sequence(
            collection.get("fold_applications"),
            field="crossfit_result.fold_applications",
        )
        for raw_application in applications:
            application = _mapping(
                raw_application,
                field="crossfit_result.fold_applications[]",
            )
            all_calibrated = application.get("all_receivers_gain_calibrated")
            rank_eligible = application.get("cross_receiver_percentile_rank_eligible")
            if type(all_calibrated) is not bool or type(rank_eligible) is not bool:
                raise ValueError("persisted gain calibration eligibility is invalid")
            eligibility.append(
                {
                    "contrast_common_collection_id": collection.get(
                        "contrast_common_collection_id"
                    ),
                    "fold_id": application.get("fold_id"),
                    "all_receivers_gain_calibrated": all_calibrated,
                    "cross_receiver_percentile_rank_eligible": rank_eligible,
                }
            )
    n_eligible = sum(
        bool(item["cross_receiver_percentile_rank_eligible"]) for item in eligibility
    )
    if not eligibility:
        eligibility_status = "not_available"
    elif n_eligible == len(eligibility):
        eligibility_status = "eligible"
    else:
        eligibility_status = "not_eligible"
    return {
        "path": _portable_path(path, workspace_root),
        "crossfit_result_id": manifest.get("crossfit_result_id"),
        "schema_version": manifest.get("schema_version"),
        "certification_status": manifest.get("certification_status"),
        "complete_pipeline_oof_certified": manifest.get(
            "complete_pipeline_oof_certified"
        ),
        "cross_receiver_percentile_rank_eligibility": {
            "status": eligibility_status,
            "n_fold_applications": len(eligibility),
            "n_eligible_fold_applications": n_eligible,
            "fold_applications": eligibility,
        },
    }


def run_smoke(
    *,
    workspace_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    gain_calibration_spec: GainCalibrationSpec | None = None,
    gain_calibration_config_path: Path = DEFAULT_GAIN_CALIBRATION_CONFIG,
    crossfit_result_output: Path | None = None,
) -> dict[str, object]:
    """Run the bounded real-data smoke, with optional descriptive v5 output."""

    if crossfit_result_output is not None and gain_calibration_spec is None:
        raise ValueError(
            "crossfit_result_output requires an explicit gain_calibration_spec"
        )

    total_started = time.perf_counter()
    workspace_root = workspace_root.resolve()
    config_path = config_path.resolve()
    gain_calibration_summary = None
    if gain_calibration_spec is not None:
        if sha256_file(config_path) != EXPECTED_BASE_SMOKE_CONFIG_SHA256:
            raise ValueError(
                "gain calibration smoke requires the checksum-bound base config"
            )
        gain_calibration_config_path = gain_calibration_config_path.resolve()
        gain_calibration_summary = _gain_calibration_configuration_summary(
            config_path=gain_calibration_config_path,
            workspace_root=workspace_root,
            spec=gain_calibration_spec,
        )
    config = load_smoke_config(config_path)
    dataset = _mapping(config["dataset"], field="dataset")
    harmonized_config = _mapping(
        config["harmonized_resource"], field="harmonized_resource"
    )
    nichenet_config = _mapping(config["nichenet_resource"], field="nichenet_resource")
    definitions = _definitions(config)

    input_path = workspace_root / str(dataset["relative_path"])
    adata, available_genes, input_audit = prepare_input(
        input_path,
        dataset_config=dataset,
    )

    table_path = workspace_root / str(harmonized_config["table_relative_path"])
    manifest_path = workspace_root / str(harmonized_config["manifest_relative_path"])
    table_audit = _verify_file(
        table_path,
        expected_bytes=_int_value(
            harmonized_config["table_bytes"], field="table_bytes"
        ),
        expected_sha256=str(harmonized_config["table_sha256"]),
        label="harmonized LR table",
    )
    manifest_audit = _verify_file(
        manifest_path,
        expected_bytes=_int_value(
            harmonized_config["manifest_bytes"], field="manifest_bytes"
        ),
        expected_sha256=str(harmonized_config["manifest_sha256"]),
        label="harmonized LR manifest",
    )
    full_bundle = harmonized_resource_bundle(table_path, manifest_path)
    expected_bundle_identity = (
        str(harmonized_config["resource_id"]),
        str(harmonized_config["version"]),
        str(harmonized_config["manifest_digest"]),
        _int_value(
            harmonized_config["expected_source_interaction_count"],
            field="expected_source_interaction_count",
        ),
    )
    observed_bundle_identity = (
        full_bundle.resource_id,
        full_bundle.version,
        full_bundle.manifest_digest,
        len(full_bundle.interactions),
    )
    if observed_bundle_identity != expected_bundle_identity:
        raise ValueError("harmonized ResourceBundle identity changed")
    bundle, interaction_manifest = select_anchor_interactions(
        full_bundle,
        definitions,
        expected_id_digest=str(harmonized_config["expected_selected_id_digest"]),
        expected_fields_digest=str(
            harmonized_config["expected_selected_full_fields_digest"]
        ),
    )
    if bundle.mapping_report.mapped_entities != _int_value(
        harmonized_config["expected_selected_gene_count"],
        field="expected_selected_gene_count",
    ):
        raise ValueError("anchor LR gene count changed")
    missing_lr = sorted(
        {
            gene
            for interaction in bundle.interactions
            for gene in (*interaction.ligand_subunits, *interaction.receptor_subunits)
        }.difference(available_genes)
    )
    if missing_lr:
        raise ValueError(f"paired cSCC H5AD lacks anchor LR genes: {missing_lr}")

    nichenet_manifest_path = workspace_root / str(
        nichenet_config["manifest_relative_path"]
    )
    nichenet_payload_path = workspace_root / str(
        nichenet_config["payload_relative_path"]
    )
    nichenet_manifest_audit = _verify_file(
        nichenet_manifest_path,
        expected_bytes=_int_value(
            nichenet_config["manifest_bytes"], field="manifest_bytes"
        ),
        expected_sha256=str(nichenet_config["manifest_sha256"]),
        label="NicheNet manifest",
    )
    nichenet_payload_audit = _verify_file(
        nichenet_payload_path,
        expected_bytes=_int_value(
            nichenet_config["payload_bytes"], field="payload_bytes"
        ),
        expected_sha256=str(nichenet_config["payload_sha256"]),
        label="NicheNet payload",
    )
    database_root = workspace_root / str(nichenet_config["database_root_relative_path"])
    full_target_prior = load_nichenet_target_prior(database_root)
    expected_prior_identity = (
        str(nichenet_config["resource_id"]),
        str(nichenet_config["version"]),
        str(nichenet_config["manifest_digest"]),
    )
    observed_prior_identity = (
        full_target_prior.resource_id,
        full_target_prior.version,
        full_target_prior.manifest_digest,
    )
    if observed_prior_identity != expected_prior_identity:
        raise ValueError("NicheNet TargetPrior identity changed")
    if (
        full_target_prior.species is not Species.HUMAN
        or full_target_prior.species is not bundle.species
        or full_target_prior.gene_namespace is not bundle.gene_namespace
    ):
        raise ValueError("LR bundle and target prior species/namespace differ")
    target_prior, target_manifest = build_smoke_target_prior(
        full_target_prior,
        definitions=definitions,
        available_genes=available_genes,
        top_n=_int_value(
            nichenet_config["top_h5ad_observable_targets_per_ligand"],
            field="top_h5ad_observable_targets_per_ligand",
        ),
        expected_driver_count=_int_value(
            nichenet_config["expected_driver_count"],
            field="expected_driver_count",
        ),
        expected_link_count=_int_value(
            nichenet_config["expected_link_count"], field="expected_link_count"
        ),
        expected_target_count=_int_value(
            nichenet_config["expected_target_count"], field="expected_target_count"
        ),
        expected_link_digest=str(nichenet_config["expected_selected_link_digest"]),
    )

    crychic_config, spec = build_crossfit_spec(
        config,
        gain_calibration_spec=gain_calibration_spec,
    )
    crossfit_started = time.perf_counter()
    artifacts = run_subject_crossfit(
        adata,
        crychic_config,
        bundle,
        target_prior,
        spec=spec,
    )
    crossfit_elapsed = time.perf_counter() - crossfit_started
    if len(artifacts.folds) != 2:
        raise ValueError("cSCC smoke did not produce exactly two outer folds")
    partitions = _fold_partitions(artifacts)
    if any(
        row["n_training_subjects"] != 4 or row["n_heldout_subjects"] != 4
        for row in partitions
    ):
        raise ValueError("cSCC smoke outer folds must remain four-versus-four")
    fold_supports = compact_fold_supports(
        artifacts,
        interaction_manifest=interaction_manifest,
    )
    receptor_gates = compact_receptor_gates(
        artifacts,
        interaction_manifest=interaction_manifest,
    )
    not_estimable_metrics = compact_not_estimable_metrics(artifacts)
    diagnostic_score_summary = compact_diagnostic_score_summary(
        artifacts,
        interaction_manifest=interaction_manifest,
        resource_id=bundle.resource_id,
        resource_version=bundle.version,
    )
    if (
        not_estimable_metrics["n_receiver_models"] != 4
        or not_estimable_metrics["n_oof_receiver_coverage_rows"] != 32
    ):
        raise ValueError("cSCC smoke NE grains changed from 4 models and 32 rows")
    configuration_summary: dict[str, object] = {
        "path": _portable_path(config_path, workspace_root),
        "sha256": sha256_file(config_path),
        "crychic_config": crychic_config.to_dict(),
        "crossfit_spec": spec.to_dict(),
    }
    if gain_calibration_summary is not None:
        configuration_summary["gain_calibration_smoke"] = gain_calibration_summary

    crossfit_result_summary = None
    if crossfit_result_output is not None:
        destination = crossfit_result_output
        if not destination.is_absolute():
            destination = workspace_root / destination
        persisted = write_crossfit_result(artifacts, destination)
        crossfit_result_summary = _crossfit_result_summary(
            path=persisted.path,
            workspace_root=workspace_root,
            manifest=persisted.manifest,
        )

    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "scope": "bounded_real_data_algorithm_smoke_not_biological_validation",
        "configuration": configuration_summary,
        "input": {
            "path": _portable_path(input_path, workspace_root),
            **input_audit,
        },
        "resources": {
            "harmonized_lr": {
                "resource_id": full_bundle.resource_id,
                "version": full_bundle.version,
                "manifest_digest": full_bundle.manifest_digest,
                "table": table_audit,
                "manifest": manifest_audit,
                "n_source_interactions": len(full_bundle.interactions),
                "n_selected_interactions": len(bundle.interactions),
                "selected_interaction_id_digest": EXPECTED_INTERACTION_ID_DIGEST,
                "selected_interaction_fields_digest": (
                    EXPECTED_INTERACTION_FIELDS_DIGEST
                ),
            },
            "nichenet": {
                "resource_id": full_target_prior.resource_id,
                "version": full_target_prior.version,
                "manifest_digest": full_target_prior.manifest_digest,
                "manifest": nichenet_manifest_audit,
                "payload": nichenet_payload_audit,
                **target_manifest,
            },
        },
        "fold_partitions": partitions,
        "fold_receiver_interaction_supports": fold_supports,
        "fold_deduplicated_support_summary": summarize_fold_supports(fold_supports),
        "fold_receiver_receptor_gates": receptor_gates,
        "not_estimable_metrics": not_estimable_metrics,
        "diagnostic_score_summary": diagnostic_score_summary,
        "runtime": {
            "crossfit_elapsed_seconds": crossfit_elapsed,
            "total_elapsed_seconds": time.perf_counter() - total_started,
            "process_peak_rss_kib": int(
                process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "output_exclusions": {
            "plaintext_cell_ids": True,
            "plaintext_subject_ids": True,
            "downstream_score_tables": True,
        },
    }
    if crossfit_result_summary is not None:
        result["crossfit_result"] = crossfit_result_summary
        exclusions = cast(dict[str, object], result["output_exclusions"])
        exclusions["plaintext_subject_ids"] = False
        exclusions["downstream_score_tables"] = False
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=DEFAULT_WORKSPACE_ROOT,
        help="Workspace containing databases/ and benchmark_work/.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gain-calibration-config",
        type=Path,
        default=None,
        help="Enable the separately versioned descriptive gain-calibration smoke.",
    )
    parser.add_argument(
        "--crossfit-result-output",
        type=Path,
        default=None,
        help="Write the validated v5 cross-fit result directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.crossfit_result_output is not None and args.gain_calibration_config is None:
        parser.error("--crossfit-result-output requires --gain-calibration-config")
    workspace_root = args.workspace_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = workspace_root / output
    gain_calibration_spec = (
        None
        if args.gain_calibration_config is None
        else load_gain_calibration_smoke_config(args.gain_calibration_config)
    )
    result = run_smoke(
        workspace_root=workspace_root,
        config_path=args.config,
        gain_calibration_spec=gain_calibration_spec,
        gain_calibration_config_path=(
            DEFAULT_GAIN_CALIBRATION_CONFIG
            if args.gain_calibration_config is None
            else args.gain_calibration_config
        ),
        crossfit_result_output=args.crossfit_result_output,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
