"""Run a small multi-edge, multi-seed public family-common G1.5 campaign.

The campaign is a development diagnostic.  Every score is produced by the
public ``run_subject_crossfit`` workflow with the trusted synthetic autonomous
resource and the tuned subject-blocked path.  It does not certify the complete
pipeline or make a biological validation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource as process_resource
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.adapters.common import canonical_digest, sha256_file
from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.simulation.run_family_common_crossfit_smoke import (
    AUTONOMOUS_REGISTRATION_ID,
    FAMILY_COMMON_SMOKE_SOURCE_PATHS,
    PRIMARY_RECEIVER,
    PRIMARY_SENDER,
)
from crychic import load_nichenet_target_prior
from crychic.attribution import PenaltyTuningSpec
from crychic.core import CrychicConfig
from crychic.design import balanced_contrast
from crychic.resources import ResourceBundle, TargetPrior
from crychic.response import (
    ReceiverAutonomousProgramResource,
    load_receiver_autonomous_program_resource,
)
from crychic.sender import ContrastCommonSenderParameters
from crychic.workflow import (
    CrossFitArtifacts,
    CrossFitSpec,
    FoldTrainingSpec,
    run_subject_crossfit,
)

SCHEMA_VERSION = "crychic-public-family-common-g1.5-campaign-v1"
INPUT_SCHEMA_VERSION = "crychic-public-family-common-g1.5-campaign-inputs-v1"
GENERATOR_SCHEMA_VERSION = "crychic-public-family-common-g1.5-generator-v1"
DEFAULT_INPUT_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "fixtures/family_common_g15_campaign_v1.json"
)
ALL_SCENARIOS = (
    "active",
    "global_null",
    "abundance_only",
    "ligand_only",
    "target_only",
    "receiver_autonomous",
    "receptor_knockout",
)
DEFAULT_DEVELOPMENT_SCENARIOS = (
    "active",
    "ligand_only",
    "receiver_autonomous",
    "global_null",
)
RELEASED_MODES = ("state", "ecosystem")
SCORE_KINDS = ("member_unresolved", "sender_resolved")
FROZEN_LAMBDA1_FRACTIONS = (1.0, 0.3, 0.1)
FROZEN_LAMBDA2_FRACTIONS = (0.0,)
HOUSEKEEPING_GENES = ("GAPDH", "RPLP0", "MALAT1", "ACTB")
OTHER_RESPONSE_GENES = ("ISG15", "IFIT1", "MX1", "OAS1", "STAT1", "B2M")
CELL_TYPES = (PRIMARY_SENDER, PRIMARY_RECEIVER, "Bystander")

CAMPAIGN_SOURCE_PATHS = tuple(
    sorted(
        {
            *FAMILY_COMMON_SMOKE_SOURCE_PATHS,
            "benchmarks/fixtures/family_common_g15_campaign_v1.json",
            "benchmarks/simulation/run_family_common_g15_campaign.py",
            "src/crychic/scoring/receiver_program.py",
        }
    )
)

AUDIT_CLAIMS: dict[str, bool] = {
    "development_only": True,
    "biological_validation": False,
    "method_superiority": False,
    "complete_pipeline_oof_certification": False,
    "family_common_full_oof_certification": False,
    "default_switch_allowed": False,
}


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return cast(Sequence[object], value)


def _required_string(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _float_value(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _int_value(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def load_campaign_registry(
    path: str | Path = DEFAULT_INPUT_REGISTRY,
) -> dict[str, object]:
    """Load and validate the preregistered seed, edge, and profile identities."""

    registry_path = Path(path).resolve()
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = dict(_mapping(value, field="registry"))
    if registry.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported family-common G1.5 input registry")
    if registry.get("generator_schema_version") != GENERATOR_SCHEMA_VERSION:
        raise ValueError("registry generator schema does not match the live generator")
    if registry.get("lock_scope") != "before_first_multiseed_execution":
        raise ValueError(
            "campaign lock scope is not the preregistered multi-seed scope"
        )
    debug_policy = _mapping(
        registry.get("prelock_debug_policy"), field="prelock_debug_policy"
    )
    if debug_policy != {
        "single_seed_public_workflow_debug_was_run": True,
        "debug_result_is_excluded_from_all_campaign_metrics": True,
        "debug_result_may_not_change_edges_seeds_or_signal_policy": True,
        "first_eligible_campaign_requires_at_least_two_frozen_seeds": True,
    }:
        raise ValueError("prelock debug exclusion policy changed")

    scenarios = tuple(
        _required_string(item, field="scenarios[]")
        for item in _sequence(registry.get("scenarios"), field="scenarios")
    )
    if scenarios != ALL_SCENARIOS:
        raise ValueError("registry scenarios must match the frozen campaign order")
    development_scenarios = tuple(
        str(item)
        for item in _sequence(
            registry.get("default_development_scenarios"),
            field="default_development_scenarios",
        )
    )
    if development_scenarios != DEFAULT_DEVELOPMENT_SCENARIOS:
        raise ValueError("default development scenarios changed from the frozen subset")

    raw_edges = _sequence(registry.get("known_edges"), field="known_edges")
    if len(raw_edges) < 2:
        raise ValueError("public G1.5 campaign requires multiple known edges")
    edge_ids: list[str] = []
    interaction_ids: list[str] = []
    target_genes: list[str] = []
    for index, raw in enumerate(raw_edges):
        edge = _mapping(raw, field=f"known_edges[{index}]")
        edge_ids.append(
            _required_string(edge.get("known_edge_id"), field="known_edge_id")
        )
        interaction_ids.append(
            _required_string(
                edge.get("source_interaction_id"), field="source_interaction_id"
            )
        )
        _required_string(edge.get("ligand"), field="ligand")
        _required_string(edge.get("receptor"), field="receptor")
        targets = [
            _required_string(item, field="target_genes[]")
            for item in _sequence(edge.get("target_genes"), field="target_genes")
        ]
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("each known edge requires unique non-empty target genes")
        target_genes.extend(targets)
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("known_edge_id values must be unique")
    if len(set(interaction_ids)) != len(interaction_ids):
        raise ValueError("source interaction identities must be unique")
    if len(set(target_genes)) != len(target_genes):
        raise ValueError("known-edge target programs must be disjoint")

    raw_seeds = _sequence(registry.get("seed_sets"), field="seed_sets")
    if len(raw_seeds) < 2:
        raise ValueError("public G1.5 campaign requires multiple preregistered seeds")
    seed_ids: list[str] = []
    input_seeds: list[int] = []
    crossfit_seeds: list[int] = []
    for index, raw in enumerate(raw_seeds):
        seed = _mapping(raw, field=f"seed_sets[{index}]")
        seed_ids.append(_required_string(seed.get("seed_id"), field="seed_id"))
        input_seeds.append(int(cast(int, seed.get("input_seed"))))
        crossfit_seeds.append(int(cast(int, seed.get("crossfit_seed"))))
    for values, label in (
        (seed_ids, "seed_id"),
        (input_seeds, "input_seed"),
        (crossfit_seeds, "crossfit_seed"),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"registry {label} values must be unique")

    profiles = _mapping(registry.get("profiles"), field="profiles")
    for profile_name in ("tiny", "quick"):
        profile = _mapping(profiles.get(profile_name), field=f"profiles.{profile_name}")
        if int(cast(int, profile.get("n_subjects"))) < 8:
            raise ValueError(
                "campaign profiles require eight subjects for nested tuning"
            )
        if int(cast(int, profile.get("mean_cells_per_sample"))) < 60:
            raise ValueError("campaign profiles require at least 60 cells per sample")

    tuning = _mapping(
        registry.get("penalty_tuning_policy"), field="penalty_tuning_policy"
    )
    lambda1 = tuple(
        _float_value(item, field="lambda1_fractions[]")
        for item in _sequence(
            tuning.get("lambda1_fractions"),
            field="penalty_tuning_policy.lambda1_fractions",
        )
    )
    lambda2 = tuple(
        _float_value(item, field="lambda2_fractions[]")
        for item in _sequence(
            tuning.get("lambda2_fractions"),
            field="penalty_tuning_policy.lambda2_fractions",
        )
    )
    if lambda1 != FROZEN_LAMBDA1_FRACTIONS or lambda2 != FROZEN_LAMBDA2_FRACTIONS:
        raise ValueError("campaign penalty grid changed from the frozen policy")

    truth = _mapping(registry.get("component_truth"), field="component_truth")
    if set(truth) != set(ALL_SCENARIOS):
        raise ValueError("component truth must cover every frozen G1.5 scenario")
    expected_components = {
        "receiver_program",
        "incremental_downstream",
        "integrated",
    }
    allowed_truth_codes = {"positive", "zero", "allowed_positive"}
    for scenario in ALL_SCENARIOS:
        components = _mapping(
            truth.get(scenario), field=f"component_truth.{scenario}"
        )
        if set(components) != expected_components:
            raise ValueError(
                f"component truth is incomplete for scenario {scenario!r}"
            )
        if not set(str(value) for value in components.values()).issubset(
            allowed_truth_codes
        ):
            raise ValueError(f"component truth code is invalid for {scenario!r}")
    component_estimands = _mapping(
        registry.get("component_estimands"), field="component_estimands"
    )
    expected_estimands = {
        "receiver_program": "subject_level_stim_minus_ctrl",
        "incremental_downstream": "raw_heldout_family_gain",
        "integrated": "subject_level_stim_minus_ctrl",
        "raw_integrated_falsification_diagnostic": (
            "raw_heldout_score_over_context_rows"
        ),
    }
    if dict(component_estimands) != expected_estimands:
        raise ValueError("component estimands changed from the frozen policy")

    return registry


def campaign_source_sha256() -> dict[str, str]:
    """Hash the public workflow and benchmark source closure."""

    repository_root = Path(__file__).resolve().parents[2]
    return {
        relative_path: sha256_file(repository_root / relative_path)
        for relative_path in CAMPAIGN_SOURCE_PATHS
    }


def _stable_seed(input_seed: int, *parts: str) -> int:
    payload = ":".join(
        (GENERATOR_SCHEMA_VERSION, str(input_seed), *parts)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _edge_records(registry: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(item, field="known_edges[]")
        for item in _sequence(registry.get("known_edges"), field="known_edges")
    )


def _decoy_records(registry: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(item, field="expressed_decoy_edges[]")
        for item in _sequence(
            registry.get("expressed_decoy_edges"), field="expressed_decoy_edges"
        )
    )


def _gene_catalog(registry: Mapping[str, object]) -> tuple[str, ...]:
    signal = _mapping(registry.get("signal_policy"), field="signal_policy")
    autonomous = tuple(
        str(item)
        for item in _sequence(
            signal.get("autonomous_target_genes"),
            field="signal_policy.autonomous_target_genes",
        )
    )
    ordered: list[str] = []
    for edge in (*_edge_records(registry), *_decoy_records(registry)):
        ordered.extend((str(edge["ligand"]), str(edge["receptor"])))
    for edge in _edge_records(registry):
        ordered.extend(
            str(item) for item in cast(Sequence[object], edge["target_genes"])
        )
    ordered.extend(autonomous)
    ordered.extend(OTHER_RESPONSE_GENES)
    ordered.extend(HOUSEKEEPING_GENES)
    return tuple(dict.fromkeys(ordered))


def _mean_profile(
    registry: Mapping[str, object],
    genes: tuple[str, ...],
    *,
    cell_type: str,
    condition: str,
    scenario: str,
) -> np.ndarray:
    index = {gene: position for position, gene in enumerate(genes)}
    signal = _mapping(registry.get("signal_policy"), field="signal_policy")
    rates: np.ndarray = np.full(len(genes), 0.025, dtype=float)
    for gene in HOUSEKEEPING_GENES:
        rates[index[gene]] = {
            "GAPDH": 2.5,
            "RPLP0": 2.2,
            "MALAT1": 3.0,
            "ACTB": 2.0,
        }[gene]
    for gene in OTHER_RESPONSE_GENES:
        rates[index[gene]] = 0.08
    for edge in _edge_records(registry):
        for target in cast(Sequence[object], edge["target_genes"]):
            rates[index[str(target)]] = _float_value(
                signal["target_baseline"], field="signal_policy.target_baseline"
            )
    for target in cast(Sequence[object], signal["autonomous_target_genes"]):
        rates[index[str(target)]] = 0.08

    all_edges = (*_edge_records(registry), *_decoy_records(registry))
    if cell_type == PRIMARY_SENDER:
        for edge in all_edges:
            rates[index[str(edge["ligand"])]] = _float_value(
                signal["sender_ligand_baseline"],
                field="signal_policy.sender_ligand_baseline",
            )
    elif cell_type == PRIMARY_RECEIVER:
        for edge in all_edges:
            rates[index[str(edge["receptor"])]] = _float_value(
                signal["receiver_receptor_baseline"],
                field="signal_policy.receiver_receptor_baseline",
            )

    if cell_type == PRIMARY_RECEIVER and scenario == "receptor_knockout":
        for edge in _edge_records(registry):
            rates[index[str(edge["receptor"])]] = 0.0

    if condition == "stim" and cell_type == PRIMARY_SENDER and scenario in {
        "active",
        "ligand_only",
        "receptor_knockout",
    }:
        for edge in _edge_records(registry):
            ligand = str(edge["ligand"])
            rates[index[ligand]] *= _float_value(
                signal["active_ligand_fold_change"],
                field="signal_policy.active_ligand_fold_change",
            )
    if condition == "stim" and cell_type == PRIMARY_RECEIVER:
        if scenario in {"active", "target_only", "receptor_knockout"}:
            for edge in _edge_records(registry):
                for target in cast(Sequence[object], edge["target_genes"]):
                    rates[index[str(target)]] *= _float_value(
                        signal["active_target_fold_change"],
                        field="signal_policy.active_target_fold_change",
                    )
        elif scenario == "receiver_autonomous":
            for target in cast(Sequence[object], signal["autonomous_target_genes"]):
                rates[index[str(target)]] *= _float_value(
                    signal["autonomous_target_fold_change"],
                    field="signal_policy.autonomous_target_fold_change",
                )
    return rates


def _input_content_sha256(adata: ad.AnnData) -> str:
    counts = sparse.csr_matrix(adata.layers["counts"], dtype=np.int32)
    digest = hashlib.sha256()
    digest.update(np.asarray(counts.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(counts.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(counts.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(counts.data, dtype=np.int32).tobytes())
    for column in ("sample_id", "subject_id", "condition", "cell_type"):
        digest.update(column.encode("ascii"))
        for value in adata.obs[column].astype(str):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    for gene in adata.var_names.astype(str):
        digest.update(gene.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def generate_campaign_input(
    registry: Mapping[str, object],
    *,
    profile_name: str,
    seed_record: Mapping[str, object],
    scenario: str,
    registry_sha256: str,
) -> tuple[ad.AnnData, dict[str, object]]:
    """Generate one paired multi-positive input from a frozen seed identity."""

    scenarios = tuple(
        str(item) for item in cast(Sequence[object], registry["scenarios"])
    )
    if scenario not in scenarios:
        raise ValueError(f"scenario is not preregistered: {scenario!r}")
    profiles = _mapping(registry.get("profiles"), field="profiles")
    profile = _mapping(profiles.get(profile_name), field=f"profiles.{profile_name}")
    n_subjects = int(cast(int, profile["n_subjects"]))
    mean_cells = int(cast(int, profile["mean_cells_per_sample"]))
    input_seed = int(cast(int, seed_record["input_seed"]))
    genes = _gene_catalog(registry)
    baseline_proportions = np.asarray([0.30, 0.38, 0.32], dtype=float)
    counts_parts: list[np.ndarray] = []
    obs_parts: list[pd.DataFrame] = []

    for subject_index in range(n_subjects):
        subject_id = f"S{subject_index + 1:02d}"
        subject_rng = np.random.default_rng(
            _stable_seed(input_seed, f"subject={subject_id}", "subject-scale")
        )
        subject_scale = float(subject_rng.lognormal(mean=0.0, sigma=0.12))
        for condition in ("ctrl", "stim"):
            sample_id = f"{subject_id}:{condition}"
            composition_rng = np.random.default_rng(
                _stable_seed(input_seed, f"sample={sample_id}", "composition")
            )
            proportions = baseline_proportions
            if scenario == "abundance_only" and condition == "stim":
                proportions = np.asarray([0.67, 0.13, 0.20], dtype=float)
            realized = composition_rng.dirichlet(proportions * 160.0)
            total_cells = max(60, int(composition_rng.poisson(mean_cells)))
            cell_counts = composition_rng.multinomial(total_cells, realized)
            for cell_type, n_cells in zip(CELL_TYPES, cell_counts, strict=True):
                if n_cells == 0:
                    continue
                block_id = f"sample={sample_id}:cell_type={cell_type}"
                depth_rng = np.random.default_rng(
                    _stable_seed(input_seed, block_id, "cell-depth")
                )
                cell_depth = depth_rng.lognormal(mean=0.0, sigma=0.30, size=n_cells)
                rates = _mean_profile(
                    registry,
                    genes,
                    cell_type=cell_type,
                    condition=condition,
                    scenario=scenario,
                ) * subject_scale
                part = np.empty((n_cells, len(genes)), dtype=np.int32)
                for gene_index, gene in enumerate(genes):
                    gene_rng = np.random.default_rng(
                        _stable_seed(input_seed, block_id, f"gene={gene}")
                    )
                    latent_multiplier = gene_rng.gamma(
                        shape=10.0, scale=0.1, size=n_cells
                    )
                    expected = cell_depth * rates[gene_index] * latent_multiplier
                    part[:, gene_index] = np.asarray(
                        gene_rng.poisson(expected), dtype=np.int32
                    )
                counts_parts.append(part)
                obs_parts.append(
                    pd.DataFrame(
                        {
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "condition": condition,
                            "cell_type": cell_type,
                            "scenario": scenario,
                        },
                        index=[
                            f"{sample_id}:{cell_type}:{index:04d}"
                            for index in range(n_cells)
                        ],
                    )
                )

    counts = sparse.csr_matrix(np.vstack(counts_parts), dtype=np.int32)
    obs = pd.concat(obs_parts, axis=0)
    library_size = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(
        10_000.0,
        library_size,
        out=np.zeros_like(library_size, dtype=np.float32),
        where=library_size > 0,
    )
    normalized = (sparse.diags(scale) @ counts.astype(np.float32)).tocsr()
    normalized.data = np.log1p(normalized.data)
    var = pd.DataFrame(index=pd.Index(genes, name="gene_symbol"))
    adata = ad.AnnData(X=normalized, obs=obs, var=var)
    adata.layers["counts"] = counts
    known_edge_ids = [str(edge["known_edge_id"]) for edge in _edge_records(registry)]
    adata.uns["simulation_truth"] = {
        "campaign_id": registry["campaign_id"],
        "seed_id": seed_record["seed_id"],
        "input_seed": input_seed,
        "scenario": scenario,
        "active_known_edge_ids": known_edge_ids if scenario == "active" else [],
        "truth_scope": "simulation",
        "cell_level_values_are_not_independent_replicates": True,
    }
    content_sha256 = _input_content_sha256(adata)
    identity_payload = {
        "registry_sha256": registry_sha256,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "profile": profile_name,
        "seed_id": seed_record["seed_id"],
        "input_seed": input_seed,
        "scenario": scenario,
        "content_sha256": content_sha256,
    }
    return adata, {
        **identity_payload,
        "input_identity": canonical_digest(identity_payload, prefix="g15-input"),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_subjects": n_subjects,
        "n_samples": int(adata.obs["sample_id"].nunique()),
        "paired_contexts_complete": bool(
            adata.obs.loc[:, ["subject_id", "condition"]]
            .drop_duplicates()
            .groupby("subject_id", observed=True)["condition"]
            .nunique()
            .eq(2)
            .all()
        ),
    }


def build_campaign_spec(
    autonomous_resource: ReceiverAutonomousProgramResource,
    *,
    crossfit_seed: int,
) -> CrossFitSpec:
    """Build the public tuned subject-blocked specification for one seed set."""

    tuning = PenaltyTuningSpec(
        lambda1_fractions=FROZEN_LAMBDA1_FRACTIONS,
        lambda2_fractions=FROZEN_LAMBDA2_FRACTIONS,
        inner_allowed_n_splits=(2,),
        min_inner_train_subjects_per_context=2,
        min_inner_validation_subjects_per_context=1,
        root_seed=crossfit_seed,
    )
    return CrossFitSpec(
        contrasts=(balanced_contrast(("stim",), ("ctrl",), name="stim_vs_ctrl"),),
        training_spec=FoldTrainingSpec(
            min_cells=10,
            min_pooled_availability=0.0,
            max_interactions=None,
            sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
        ),
        allowed_n_splits=(2,),
        min_train_subjects_per_context=2,
        min_test_subjects_per_context=1,
        outer_fold_partition_seed=crossfit_seed,
        autonomous_program_resource=autonomous_resource,
        penalty_tuning_spec=tuning,
    )


def _count_strings(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _count_reasons(values: Iterable[object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        missing = value is None or value is pd.NA or value is pd.NaT
        if isinstance(value, (float, np.floating)):
            missing = missing or bool(np.isnan(value))
        counts["none" if missing else str(value)] += 1
    return dict(sorted(counts.items()))


def _paired_effect_summary(
    table: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, object]:
    subjects = tuple(sorted(set(table["subject_id"].astype(str))))
    numeric = pd.to_numeric(table[score_column], errors="coerce")
    finite = table.loc[np.isfinite(numeric.to_numpy())].copy()
    finite["_score"] = pd.to_numeric(finite[score_column], errors="coerce")
    finite["_condition"] = (
        finite["sample_id"].astype(str).str.rsplit(":", n=1).str[-1]
    )
    invalid_conditions = sorted(
        set(finite["_condition"].astype(str)).difference({"ctrl", "stim"})
    )
    if invalid_conditions:
        raise ValueError(
            "campaign sample_id no longer encodes ctrl/stim condition: "
            f"{invalid_conditions}"
        )
    context_means = (
        finite.groupby(["subject_id", "_condition"], observed=True, sort=True)[
            "_score"
        ]
        .mean()
        .unstack("_condition")
    )
    if {"ctrl", "stim"}.issubset(context_means.columns):
        paired = (context_means["stim"] - context_means["ctrl"]).dropna()
    else:
        paired = pd.Series(dtype=float)
    return {
        "n_rows": len(table),
        "n_expected_subjects": len(subjects),
        "n_finite": len(paired),
        "coverage": float(len(paired) / len(subjects)) if subjects else None,
        "mean": float(paired.mean()) if len(paired) else None,
        "minimum": float(paired.min()) if len(paired) else None,
        "maximum": float(paired.max()) if len(paired) else None,
        "estimand": "subject_level_stim_minus_ctrl",
    }


def _raw_strength_summary(values: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return {
        "raw_strength_n_rows": len(values),
        "raw_strength_n_finite": len(finite),
        "raw_strength_coverage": (
            float(len(finite) / len(values)) if len(values) else None
        ),
        "raw_strength_mean": float(np.mean(finite)) if len(finite) else None,
        "raw_strength_minimum": float(np.min(finite)) if len(finite) else None,
        "raw_strength_maximum": float(np.max(finite)) if len(finite) else None,
        "raw_strength_estimand": "raw_heldout_score_over_context_rows",
    }


def _concat_application_table(
    artifacts: CrossFitArtifacts, attribute: str
) -> pd.DataFrame:
    tables = [
        cast(pd.DataFrame, getattr(application, attribute))
        for fold in artifacts.folds
        for application in fold.family_common_applications
    ]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _resolve_edge_catalog(
    registry: Mapping[str, object],
    *,
    lr_table_path: Path,
    bundle: ResourceBundle,
    target_prior: TargetPrior,
) -> tuple[dict[str, object], ...]:
    table = pd.read_csv(lr_table_path, sep="\t", dtype=str)
    resolved: list[dict[str, object]] = []
    bundle_by_id = {item.interaction_id: item for item in bundle.interactions}
    for edge in _edge_records(registry):
        ligand = str(edge["ligand"])
        receptor = str(edge["receptor"])
        selected = table.loc[
            table["ligand"].eq(ligand) & table["receptor"].eq(receptor)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"known edge must map to one harmonized row: {ligand}-{receptor}"
            )
        row = selected.iloc[0]
        source_id = str(edge["source_interaction_id"])
        if str(row["cellchat_source_interaction_id"]) != source_id:
            raise ValueError(f"source interaction mapping changed: {source_id}")
        harmonized_id = str(row["harmonized_interaction_id"])
        if harmonized_id not in bundle_by_id:
            raise ValueError(
                f"harmonized interaction is absent from bundle: {source_id}"
            )
        prior_links = {
            link.target: link
            for link in target_prior.links_for_driver(ligand)
        }
        targets = tuple(
            str(item) for item in cast(Sequence[object], edge["target_genes"])
        )
        missing_targets = sorted(set(targets).difference(prior_links))
        if missing_targets:
            raise ValueError(
                "target prior lacks preregistered targets for "
                f"{ligand}: {missing_targets}"
            )
        if any(prior_links[target].weight <= 0.0 for target in targets):
            raise ValueError(
                f"target prior has non-positive target weights for {ligand}"
            )
        resolved.append(
            {
                "known_edge_id": str(edge["known_edge_id"]),
                "source_interaction_id": source_id,
                "harmonized_interaction_id": harmonized_id,
                "ligand": ligand,
                "receptor": receptor,
                "target_genes": list(targets),
                "target_prior_weights": {
                    target: float(prior_links[target].weight) for target in targets
                },
                "target_prior_ranks": {
                    target: int(prior_links[target].rank) for target in targets
                },
            }
        )
    return tuple(resolved)


def _interaction_score_summaries(
    table: pd.DataFrame,
    *,
    score_column: str,
    score_kind: str,
    edge_catalog: Sequence[Mapping[str, object]],
    positive_tolerance: float,
    recovery_top_k: int,
    sender: str | None = None,
) -> list[dict[str, object]]:
    known_by_harmonized = {
        str(edge["harmonized_interaction_id"]): edge for edge in edge_catalog
    }
    scope = table.loc[table["receiver"].astype(str).eq(PRIMARY_RECEIVER)].copy()
    if sender is not None:
        scope = scope.loc[scope["sender"].astype(str).eq(sender)].copy()
    output: list[dict[str, object]] = []
    for mode in RELEASED_MODES:
        mode_table = scope.loc[scope["mode"].astype(str).eq(mode)]
        candidates: list[dict[str, object]] = []
        for interaction_id, group in mode_table.groupby(
            "interaction_id", sort=True, observed=True
        ):
            summary = _paired_effect_summary(group, score_column=score_column)
            raw_summary = _raw_strength_summary(group[score_column])
            candidates.append(
                {
                    "interaction_id": str(interaction_id),
                    **summary,
                    **raw_summary,
                    "status_counts": _count_strings(group["status"]),
                    "reason_counts": _count_reasons(group["reason_code"]),
                }
            )
        finite_values = sorted(
            {
                cast(float, row["mean"])
                for row in candidates
                if row["mean"] is not None
            },
            reverse=True,
        )
        ranks = {value: index + 1 for index, value in enumerate(finite_values)}
        candidate_by_id = {str(row["interaction_id"]): row for row in candidates}
        for harmonized_id, edge in known_by_harmonized.items():
            row = candidate_by_id.get(harmonized_id)
            if row is None:
                summary = {
                    "n_rows": 0,
                    "n_expected_subjects": 0,
                    "n_finite": 0,
                    "coverage": None,
                    "mean": None,
                    "minimum": None,
                    "maximum": None,
                    "estimand": "subject_level_stim_minus_ctrl",
                    "raw_strength_n_rows": 0,
                    "raw_strength_n_finite": 0,
                    "raw_strength_coverage": None,
                    "raw_strength_mean": None,
                    "raw_strength_minimum": None,
                    "raw_strength_maximum": None,
                    "raw_strength_estimand": (
                        "raw_heldout_score_over_context_rows"
                    ),
                    "status_counts": {},
                    "reason_counts": {},
                }
                rank = None
                tie_count = None
            else:
                summary = {
                    key: value
                    for key, value in row.items()
                    if key != "interaction_id"
                }
                mean = cast(float | None, row["mean"])
                rank = None if mean is None else ranks[mean]
                tie_count = (
                    None
                    if mean is None
                    else sum(other["mean"] == mean for other in candidates)
                )
            mean_value = cast(float | None, summary["mean"])
            output.append(
                {
                    "known_edge_id": str(edge["known_edge_id"]),
                    "harmonized_interaction_id": harmonized_id,
                    "mode": mode,
                    "score_kind": score_kind,
                    "estimand": "subject_level_stim_minus_ctrl",
                    **summary,
                    "dense_rank": rank,
                    "n_interactions_tied_at_rank": tie_count,
                    "n_candidate_interactions": len(candidates),
                    "positive": (
                        None
                        if mean_value is None
                        else bool(mean_value > positive_tolerance)
                    ),
                    "recovered": (
                        None
                        if mean_value is None or rank is None
                        else bool(
                            mean_value > positive_tolerance and rank <= recovery_top_k
                        )
                    ),
                }
            )
    return output


def _family_membership(
    member_scores: pd.DataFrame,
    edge_catalog: Sequence[Mapping[str, object]],
) -> dict[str, tuple[str, ...]]:
    memberships: dict[str, tuple[str, ...]] = {}
    for edge in edge_catalog:
        harmonized_id = str(edge["harmonized_interaction_id"])
        rows = member_scores.loc[
            member_scores["receiver"].astype(str).eq(PRIMARY_RECEIVER)
            & member_scores["interaction_id"].astype(str).eq(harmonized_id)
        ]
        memberships[str(edge["known_edge_id"])] = tuple(
            sorted(set(rows["family_id"].astype(str)))
        )
    return memberships


def _component_summaries(
    family_scores: pd.DataFrame,
    *,
    memberships: Mapping[str, tuple[str, ...]],
    positive_tolerance: float,
) -> list[dict[str, object]]:
    components = {
        "receiver_program": "receiver_program_score",
        "incremental_downstream": "incremental_downstream_gain",
        "integrated": "integrated_lr_score",
    }
    output: list[dict[str, object]] = []
    for known_edge_id, family_ids in memberships.items():
        edge_scope = family_scores.loc[
            family_scores["receiver"].astype(str).eq(PRIMARY_RECEIVER)
            & family_scores["family_id"].astype(str).isin(family_ids)
        ]
        for mode in RELEASED_MODES:
            mode_scope = edge_scope.loc[edge_scope["mode"].astype(str).eq(mode)]
            for component, column in components.items():
                paired_summary = _paired_effect_summary(
                    mode_scope, score_column=column
                )
                raw_summary = _raw_strength_summary(mode_scope[column])
                if component == "incremental_downstream":
                    primary_summary = {
                        "n_rows": raw_summary["raw_strength_n_rows"],
                        "n_finite": raw_summary["raw_strength_n_finite"],
                        "coverage": raw_summary["raw_strength_coverage"],
                        "mean": raw_summary["raw_strength_mean"],
                        "minimum": raw_summary["raw_strength_minimum"],
                        "maximum": raw_summary["raw_strength_maximum"],
                        "estimand": "raw_heldout_family_gain",
                    }
                else:
                    primary_summary = {
                        key: paired_summary[key]
                        for key in (
                            "n_rows",
                            "n_finite",
                            "coverage",
                            "mean",
                            "minimum",
                            "maximum",
                            "estimand",
                        )
                    }
                mean = cast(float | None, primary_summary["mean"])
                if component == "receiver_program" and not mode_scope.empty:
                    status_counts = _count_strings(
                        mode_scope["receiver_program_status"]
                    )
                    reason_counts = _count_reasons(
                        mode_scope["receiver_program_reason_code"]
                    )
                else:
                    status_counts = (
                        {} if mode_scope.empty else _count_strings(mode_scope["status"])
                    )
                    reason_counts = (
                        {}
                        if mode_scope.empty
                        else _count_reasons(mode_scope["reason_code"])
                    )
                output.append(
                    {
                        "known_edge_id": known_edge_id,
                        "family_ids": list(family_ids),
                        "mode": mode,
                        "component": component,
                        **primary_summary,
                        "paired_effect_n_expected_subjects": paired_summary[
                            "n_expected_subjects"
                        ],
                        "paired_effect_n_finite": paired_summary["n_finite"],
                        "paired_effect_coverage": paired_summary["coverage"],
                        "paired_effect_mean": paired_summary["mean"],
                        "paired_effect_minimum": paired_summary["minimum"],
                        "paired_effect_maximum": paired_summary["maximum"],
                        "paired_effect_estimand": paired_summary["estimand"],
                        **raw_summary,
                        "positive": (
                            None if mean is None else bool(mean > positive_tolerance)
                        ),
                        "status_counts": status_counts,
                        "reason_counts": reason_counts,
                    }
                )
    return output


def _family_diagnostics(
    artifacts: CrossFitArtifacts,
    *,
    family_scores: pd.DataFrame,
    family_attribution: pd.DataFrame,
    memberships: Mapping[str, tuple[str, ...]],
    positive_tolerance: float,
) -> dict[str, object]:
    known_families = {
        family for values in memberships.values() for family in values
    }
    training_selected: list[dict[str, object]] = []
    for fold in artifacts.folds:
        for model in fold.receiver_incremental_models:
            if (
                model.receiver != PRIMARY_RECEIVER
                or model.diagnostic_functional is None
            ):
                continue
            functional = model.diagnostic_functional
            for family_id, coefficient in zip(
                functional.family_ids,
                functional.family_coefficients,
                strict=True,
            ):
                if float(coefficient) > positive_tolerance:
                    training_selected.append(
                        {
                            "fold_id": fold.fold_id,
                            "family_id": family_id,
                            "coefficient": float(coefficient),
                            "known_family": family_id in known_families,
                        }
                    )
    heldout_selected = family_attribution.loc[
        family_attribution["family_selected"].fillna(False).astype(bool)
    ]
    mode_records: list[dict[str, object]] = []
    for mode in RELEASED_MODES:
        scope = family_scores.loc[
            family_scores["receiver"].astype(str).eq(PRIMARY_RECEIVER)
            & family_scores["mode"].astype(str).eq(mode)
        ]
        family_means = {
            str(family_id): cast(float | None, summary["mean"])
            for family_id, group in scope.groupby(
                "family_id", observed=True, sort=True
            )
            for summary in (
                _paired_effect_summary(
                    group, score_column="integrated_lr_score"
                ),
            )
        }
        finite_family_means = {
            family_id: value
            for family_id, value in family_means.items()
            if value is not None and math.isfinite(value)
        }
        raw_family_means = {
            str(family_id): cast(float | None, summary["raw_strength_mean"])
            for family_id, group in scope.groupby(
                "family_id", observed=True, sort=True
            )
            for summary in (_raw_strength_summary(group["integrated_lr_score"]),)
        }
        finite_raw_family_means = {
            family_id: value
            for family_id, value in raw_family_means.items()
            if value is not None and math.isfinite(value)
        }
        positive_families = {
            family_id: float(value)
            for family_id, value in finite_family_means.items()
            if float(value) > positive_tolerance
        }
        positive_raw_families = {
            family_id: float(value)
            for family_id, value in finite_raw_family_means.items()
            if float(value) > positive_tolerance
        }
        mode_records.append(
            {
                "mode": mode,
                "n_families": len(family_means),
                "n_positive_integrated_families": len(positive_families),
                "positive_integrated_families": positive_families,
                "n_positive_integrated_nontruth_families": sum(
                    family_id not in known_families for family_id in positive_families
                ),
                "maximum_integrated_family_mean": (
                    float(max(finite_family_means.values()))
                    if len(finite_family_means)
                    else None
                ),
                "integrated_family_estimand": "subject_level_stim_minus_ctrl",
                "n_positive_raw_integrated_families": len(
                    positive_raw_families
                ),
                "positive_raw_integrated_families": positive_raw_families,
                "n_positive_raw_integrated_nontruth_families": sum(
                    family_id not in known_families
                    for family_id in positive_raw_families
                ),
                "maximum_raw_integrated_family_mean": (
                    float(max(finite_raw_family_means.values()))
                    if finite_raw_family_means
                    else None
                ),
                "raw_integrated_family_estimand": (
                    "raw_heldout_score_over_context_rows"
                ),
            }
        )
    return {
        "known_family_ids": sorted(known_families),
        "known_edge_family_ids": {
            key: list(value) for key, value in sorted(memberships.items())
        },
        "training_selected_families": training_selected,
        "n_training_selected_families": len(training_selected),
        "n_training_selected_nontruth_families": sum(
            not cast(bool, row["known_family"]) for row in training_selected
        ),
        "heldout_selected_family_ids": sorted(
            set(heldout_selected["family_id"].astype(str))
        ),
        "n_heldout_selected_rows": len(heldout_selected),
        "modes": mode_records,
    }


def _penalty_selection_summary(artifacts: CrossFitArtifacts) -> dict[str, object]:
    records: list[dict[str, object]] = []
    counts: Counter[tuple[float, float]] = Counter()
    for fold in artifacts.folds:
        for model in fold.receiver_incremental_models:
            if model.receiver != PRIMARY_RECEIVER:
                continue
            tuning = model.penalty_tuning_artifact
            candidate = None if tuning is None else tuning.selected_candidate
            if candidate is not None:
                counts[(candidate.lambda1_fraction, candidate.lambda2_fraction)] += 1
            records.append(
                {
                    "fold_id": fold.fold_id,
                    "receiver": model.receiver,
                    "selection_rule": (
                        None if tuning is None else tuning.spec.selection_rule
                    ),
                    "selected_candidate_id": model.selected_penalty_candidate_id,
                    "lambda1_fraction": (
                        None if candidate is None else candidate.lambda1_fraction
                    ),
                    "lambda2_fraction": (
                        None if candidate is None else candidate.lambda2_fraction
                    ),
                    "official_incremental_status": model.official_incremental_status,
                    "official_incremental_reason_code": model.reason_code,
                }
            )
    return {
        "selected_fraction_counts": [
            {
                "lambda1_fraction": lambda1,
                "lambda2_fraction": lambda2,
                "count": count,
            }
            for (lambda1, lambda2), count in sorted(counts.items())
        ],
        "records": records,
    }


def _receiver_program_reference_audit(
    artifacts: CrossFitArtifacts,
) -> dict[str, object]:
    expected = {
        "reference_summary_method": (
            "technical_row_mean_then_context_equal_subject_mean_v1"
        ),
        "center_method": "reference_subject_equal_feature_median_v2",
        "scale_method": "reference_subject_equal_scaled_mad_floor_v2",
    }
    records: list[dict[str, object]] = []
    for fold in artifacts.folds:
        for model in fold.receiver_program_models:
            if model.receiver != PRIMARY_RECEIVER:
                continue
            functional = model.downstream_functional
            records.append(
                {
                    "fold_id": fold.fold_id,
                    "status": model.status,
                    "reason_code": model.reason_code,
                    "downstream_functional_id": (
                        None
                        if functional is None
                        else functional.downstream_functional_id
                    ),
                    "reference_summary_method": (
                        None
                        if functional is None
                        else functional.reference_summary_method
                    ),
                    "center_method": (
                        None if functional is None else functional.center_method
                    ),
                    "scale_method": (
                        None if functional is None else functional.scale_method
                    ),
                    "reference_transform_id": (
                        None
                        if functional is None
                        else functional.reference_transform_id
                    ),
                    "reference_row_manifest_id": (
                        None
                        if functional is None
                        else functional.reference_row_manifest_id
                    ),
                    "reference_subject_summary_digest": (
                        None
                        if functional is None
                        else functional.reference_subject_summary_digest
                    ),
                    "n_reference_subjects": (
                        0
                        if functional is None
                        else len(functional.reference_summary_subject_ids)
                    ),
                }
            )
    observed = [row for row in records if row["status"] == "observed"]
    return {
        "expected_methods": expected,
        "records": records,
        "n_primary_receiver_models": len(records),
        "n_observed_primary_receiver_models": len(observed),
        "all_observed_models_use_subject_equal_reference_transform": bool(
            observed
            and all(
                all(row[key] == value for key, value in expected.items())
                for row in observed
            )
        ),
    }


def _run_algorithm_metrics(
    artifacts: CrossFitArtifacts,
    *,
    edge_catalog: Sequence[Mapping[str, object]],
    positive_tolerance: float,
    recovery_top_k: int,
) -> dict[str, object]:
    family_attribution = _concat_application_table(artifacts, "family_attribution")
    family_scores = _concat_application_table(artifacts, "family_scores")
    member_scores = _concat_application_table(artifacts, "member_scores")
    sender_scores = _concat_application_table(artifacts, "sender_scores")
    memberships = _family_membership(member_scores, edge_catalog)
    known_scores = [
        *_interaction_score_summaries(
            member_scores,
            score_column="sender_unresolved_strength",
            score_kind="member_unresolved",
            edge_catalog=edge_catalog,
            positive_tolerance=positive_tolerance,
            recovery_top_k=recovery_top_k,
        ),
        *_interaction_score_summaries(
            sender_scores,
            score_column="sender_resolved_strength",
            score_kind="sender_resolved",
            edge_catalog=edge_catalog,
            positive_tolerance=positive_tolerance,
            recovery_top_k=recovery_top_k,
            sender=PRIMARY_SENDER,
        ),
    ]
    models = [
        model for fold in artifacts.folds for model in fold.receiver_incremental_models
    ]
    applications = [
        application
        for fold in artifacts.folds
        for application in fold.family_common_applications
    ]
    manifest = artifacts.to_manifest()
    return {
        "crossfit_id": artifacts.crossfit_id,
        "crossfit_spec_id": artifacts.spec.spec_id,
        "n_folds": len(artifacts.folds),
        "incremental_training_official_status_counts": _count_strings(
            model.official_incremental_status for model in models
        ),
        "family_common_application_reason_counts": _count_reasons(
            application.heldout_reason_code for application in applications
        ),
        "family_common_application_oof_certified_counts": _count_strings(
            application.is_oof_certified for application in applications
        ),
        "selected_penalties": _penalty_selection_summary(artifacts),
        "receiver_program_reference_audit": _receiver_program_reference_audit(
            artifacts
        ),
        "known_edge_scores": known_scores,
        "component_scores": _component_summaries(
            family_scores,
            memberships=memberships,
            positive_tolerance=positive_tolerance,
        ),
        "family_diagnostics": _family_diagnostics(
            artifacts,
            family_scores=family_scores,
            family_attribution=family_attribution,
            memberships=memberships,
            positive_tolerance=positive_tolerance,
        ),
        "completed_stage_oof_verified": artifacts.completed_stage_oof_verified,
        "complete_pipeline_oof_certified": artifacts.is_oof_certified,
        "remaining_stages": manifest["remaining_stages"],
    }


def _index_records(
    records: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    return {
        (str(record["seed_id"]), str(record["scenario"])): record
        for record in records
    }


def _score_index(
    record: Mapping[str, object],
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    return {
        (
            str(row["known_edge_id"]),
            str(row["mode"]),
            str(row["score_kind"]),
        ): row
        for row in cast(Sequence[Mapping[str, object]], record["known_edge_scores"])
    }


def _component_index(
    record: Mapping[str, object],
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    return {
        (
            str(row["known_edge_id"]),
            str(row["mode"]),
            str(row["component"]),
        ): row
        for row in cast(Sequence[Mapping[str, object]], record["component_scores"])
    }


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    observed = np.asarray([value for value in values if value is not None], dtype=float)
    return float(observed.mean()) if len(observed) else None


def _aggregate_score_campaign(
    records: Sequence[Mapping[str, object]],
    *,
    edge_catalog: Sequence[Mapping[str, object]],
    seed_ids: Sequence[str],
    positive_scenario: str,
    reference_scenario: str,
    maximum_allowed_coverage_loss: float,
) -> dict[str, object]:
    by_run = _index_records(records)
    paired: list[dict[str, object]] = []
    for seed_id in seed_ids:
        active = by_run.get((seed_id, positive_scenario))
        reference = by_run.get((seed_id, reference_scenario))
        if active is None or reference is None:
            continue
        active_scores = _score_index(active)
        reference_scores = _score_index(reference)
        for edge in edge_catalog:
            edge_id = str(edge["known_edge_id"])
            for mode in RELEASED_MODES:
                for score_kind in SCORE_KINDS:
                    key = (edge_id, mode, score_kind)
                    active_row = active_scores.get(key)
                    reference_row = reference_scores.get(key)
                    active_mean = (
                        None
                        if active_row is None
                        else cast(float | None, active_row["mean"])
                    )
                    reference_mean = (
                        None
                        if reference_row is None
                        else cast(float | None, reference_row["mean"])
                    )
                    active_coverage = (
                        None
                        if active_row is None
                        else cast(float | None, active_row["coverage"])
                    )
                    reference_coverage = (
                        None
                        if reference_row is None
                        else cast(float | None, reference_row["coverage"])
                    )
                    paired.append(
                        {
                            "seed_id": seed_id,
                            "known_edge_id": edge_id,
                            "mode": mode,
                            "score_kind": score_kind,
                            "active_mean": active_mean,
                            "reference_mean": reference_mean,
                            "primary_estimand": "subject_level_stim_minus_ctrl",
                            "active_raw_strength_mean": (
                                None
                                if active_row is None
                                else active_row["raw_strength_mean"]
                            ),
                            "active_raw_strength_maximum": (
                                None
                                if active_row is None
                                else active_row["raw_strength_maximum"]
                            ),
                            "reference_raw_strength_mean": (
                                None
                                if reference_row is None
                                else reference_row["raw_strength_mean"]
                            ),
                            "reference_raw_strength_maximum": (
                                None
                                if reference_row is None
                                else reference_row["raw_strength_maximum"]
                            ),
                            "raw_strength_estimand": (
                                "raw_heldout_score_over_context_rows"
                            ),
                            "active_minus_ligand_only_margin": (
                                None
                                if active_mean is None or reference_mean is None
                                else float(active_mean - reference_mean)
                            ),
                            "active_dense_rank": (
                                None
                                if active_row is None
                                else active_row["dense_rank"]
                            ),
                            "active_recovered": (
                                None
                                if active_row is None
                                else active_row["recovered"]
                            ),
                            "active_coverage": active_coverage,
                            "reference_coverage": reference_coverage,
                            "coverage_loss": (
                                None
                                if active_coverage is None
                                or reference_coverage is None
                                else float(reference_coverage - active_coverage)
                            ),
                        }
                    )
    summaries: list[dict[str, object]] = []
    for mode in RELEASED_MODES:
        for score_kind in SCORE_KINDS:
            scope = [
                row
                for row in paired
                if row["mode"] == mode and row["score_kind"] == score_kind
            ]
            margins = np.asarray(
                [
                    cast(float, row["active_minus_ligand_only_margin"])
                    for row in scope
                    if row["active_minus_ligand_only_margin"] is not None
                ],
                dtype=float,
            )
            recoveries = [
                cast(bool, row["active_recovered"])
                for row in scope
                if row["active_recovered"] is not None
            ]
            ranks = np.asarray(
                [
                    cast(float, row["active_dense_rank"])
                    for row in scope
                    if row["active_dense_rank"] is not None
                ],
                dtype=float,
            )
            coverage_losses = np.asarray(
                [
                    cast(float, row["coverage_loss"])
                    for row in scope
                    if row["coverage_loss"] is not None
                ],
                dtype=float,
            )
            summaries.append(
                {
                    "mode": mode,
                    "score_kind": score_kind,
                    "primary_estimand": "subject_level_stim_minus_ctrl",
                    "n_expected_pairs": len(seed_ids) * len(edge_catalog),
                    "n_observed_margins": len(margins),
                    "mean_active_minus_ligand_only_margin": (
                        float(margins.mean()) if len(margins) else None
                    ),
                    "minimum_active_minus_ligand_only_margin": (
                        float(margins.min()) if len(margins) else None
                    ),
                    "positive_margin_fraction": (
                        float(np.mean(margins > 0.0)) if len(margins) else None
                    ),
                    "active_recovery_fraction": (
                        float(np.mean(recoveries)) if recoveries else None
                    ),
                    "median_active_dense_rank": (
                        float(np.median(ranks)) if len(ranks) else None
                    ),
                    "maximum_coverage_loss": (
                        float(coverage_losses.max())
                        if len(coverage_losses)
                        else None
                    ),
                    "coverage_noninferior": (
                        None
                        if not len(coverage_losses)
                        else bool(
                            coverage_losses.max()
                            <= maximum_allowed_coverage_loss + 1e-12
                        )
                    ),
                    "raw_strength_retained_as_separate_diagnostic": True,
                }
            )
    return {"paired_records": paired, "macro_summaries": summaries}


def _aggregate_component_semantics(
    records: Sequence[Mapping[str, object]],
    *,
    registry: Mapping[str, object],
    edge_catalog: Sequence[Mapping[str, object]],
    seed_ids: Sequence[str],
    positive_tolerance: float,
) -> dict[str, object]:
    truth = _mapping(registry.get("component_truth"), field="component_truth")
    component_estimands = _mapping(
        registry.get("component_estimands"), field="component_estimands"
    )
    by_run = _index_records(records)
    diagnostics: list[dict[str, object]] = []
    for scenario in ALL_SCENARIOS:
        scenario_truth = _mapping(
            truth.get(scenario), field=f"component_truth.{scenario}"
        )
        for seed_id in seed_ids:
            run = by_run.get((seed_id, scenario))
            if run is None:
                continue
            components = _component_index(run)
            for edge in edge_catalog:
                edge_id = str(edge["known_edge_id"])
                for mode in RELEASED_MODES:
                    for component in (
                        "receiver_program",
                        "incremental_downstream",
                        "integrated",
                    ):
                        row = components.get((edge_id, mode, component))
                        mean = None if row is None else cast(float | None, row["mean"])
                        truth_code = str(scenario_truth[component])
                        if mean is None:
                            status = "not_estimable"
                            conforms: bool | None = None
                        elif truth_code == "positive":
                            status = "observed"
                            conforms = bool(mean > positive_tolerance)
                        elif truth_code == "zero":
                            status = "observed"
                            conforms = bool(abs(mean) <= positive_tolerance)
                        else:
                            status = "diagnostic_only"
                            conforms = None
                        diagnostics.append(
                            {
                                "seed_id": seed_id,
                                "scenario": scenario,
                                "known_edge_id": edge_id,
                                "mode": mode,
                                "component": component,
                                "truth_code": truth_code,
                                "mean": mean,
                                "coverage": (
                                    None if row is None else row["coverage"]
                                ),
                                "estimand": (
                                    None if row is None else row["estimand"]
                                ),
                                "paired_effect_mean": (
                                    None
                                    if row is None
                                    else row["paired_effect_mean"]
                                ),
                                "paired_effect_coverage": (
                                    None
                                    if row is None
                                    else row["paired_effect_coverage"]
                                ),
                                "paired_effect_estimand": (
                                    "subject_level_stim_minus_ctrl"
                                ),
                                "raw_strength_mean": (
                                    None
                                    if row is None
                                    else row["raw_strength_mean"]
                                ),
                                "raw_strength_maximum": (
                                    None
                                    if row is None
                                    else row["raw_strength_maximum"]
                                ),
                                "raw_strength_coverage": (
                                    None
                                    if row is None
                                    else row["raw_strength_coverage"]
                                ),
                                "raw_strength_estimand": (
                                    None
                                    if row is None
                                    else row["raw_strength_estimand"]
                                ),
                                "status": status,
                                "conforms": conforms,
                                "not_estimable_never_counts_as_pass": True,
                            }
                        )
    summaries: list[dict[str, object]] = []
    for scenario in ALL_SCENARIOS:
        for component in (
            "receiver_program",
            "incremental_downstream",
            "integrated",
        ):
            scope = [
                row
                for row in diagnostics
                if row["scenario"] == scenario and row["component"] == component
            ]
            observed = [row for row in scope if row["status"] == "observed"]
            conforming = [cast(bool, row["conforms"]) for row in observed]
            summaries.append(
                {
                    "scenario": scenario,
                    "component": component,
                    "n_expected": (
                        len(seed_ids) * len(edge_catalog) * len(RELEASED_MODES)
                    ),
                    "n_executed": len(scope),
                    "n_observed": len(observed),
                    "n_not_estimable": sum(
                        row["status"] == "not_estimable" for row in scope
                    ),
                    "n_diagnostic_only": sum(
                        row["status"] == "diagnostic_only" for row in scope
                    ),
                    "conformance_fraction_among_observed": (
                        float(np.mean(conforming)) if conforming else None
                    ),
                    "mean_score": _mean_or_none(
                        cast(float | None, row["mean"]) for row in scope
                    ),
                    "mean_raw_strength": _mean_or_none(
                        cast(float | None, row["raw_strength_mean"])
                        for row in scope
                    ),
                    "mean_paired_effect": _mean_or_none(
                        cast(float | None, row["paired_effect_mean"])
                        for row in scope
                    ),
                    "maximum_raw_strength": (
                        max(
                            cast(float, row["raw_strength_maximum"])
                            for row in scope
                            if row["raw_strength_maximum"] is not None
                        )
                        if any(
                            row["raw_strength_maximum"] is not None
                            for row in scope
                        )
                        else None
                    ),
                    "primary_estimand": str(component_estimands[component]),
                    "paired_effect_estimand": "subject_level_stim_minus_ctrl",
                    "raw_strength_estimand": (
                        "raw_heldout_score_over_context_rows"
                    ),
                }
            )
    return {"records": diagnostics, "summaries": summaries}


def _aggregate_control_families(
    records: Sequence[Mapping[str, object]],
    *,
    scenarios: Sequence[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for scenario in scenarios:
        if scenario == "active":
            continue
        runs = [record for record in records if record["scenario"] == scenario]
        positive_counts: list[int] = []
        nontruth_counts: list[int] = []
        raw_positive_counts: list[int] = []
        raw_nontruth_counts: list[int] = []
        raw_family_maxima: list[float] = []
        training_selected: list[int] = []
        heldout_selected: list[int] = []
        for run in runs:
            diagnostic = cast(Mapping[str, object], run["family_diagnostics"])
            training_selected.append(
                _int_value(
                    diagnostic["n_training_selected_families"],
                    field="n_training_selected_families",
                )
            )
            heldout_selected.append(
                _int_value(
                    diagnostic["n_heldout_selected_rows"],
                    field="n_heldout_selected_rows",
                )
            )
            for mode in cast(Sequence[Mapping[str, object]], diagnostic["modes"]):
                positive_counts.append(
                    _int_value(
                        mode["n_positive_integrated_families"],
                        field="n_positive_integrated_families",
                    )
                )
                nontruth_counts.append(
                    _int_value(
                        mode["n_positive_integrated_nontruth_families"],
                        field="n_positive_integrated_nontruth_families",
                    )
                )
                raw_positive_counts.append(
                    _int_value(
                        mode["n_positive_raw_integrated_families"],
                        field="n_positive_raw_integrated_families",
                    )
                )
                raw_nontruth_counts.append(
                    _int_value(
                        mode["n_positive_raw_integrated_nontruth_families"],
                        field="n_positive_raw_integrated_nontruth_families",
                    )
                )
                raw_maximum = cast(
                    float | None, mode["maximum_raw_integrated_family_mean"]
                )
                if raw_maximum is not None:
                    raw_family_maxima.append(raw_maximum)
        output.append(
            {
                "scenario": scenario,
                "n_seed_runs": len(runs),
                "n_seed_mode_evaluations": len(positive_counts),
                "mean_positive_integrated_families": (
                    float(np.mean(positive_counts)) if positive_counts else None
                ),
                "any_positive_integrated_family_rate": (
                    float(np.mean(np.asarray(positive_counts) > 0))
                    if positive_counts
                    else None
                ),
                "mean_positive_integrated_nontruth_families": (
                    float(np.mean(nontruth_counts)) if nontruth_counts else None
                ),
                "paired_integrated_estimand": "subject_level_stim_minus_ctrl",
                "mean_positive_raw_integrated_families": (
                    float(np.mean(raw_positive_counts))
                    if raw_positive_counts
                    else None
                ),
                "any_positive_raw_integrated_family_rate": (
                    float(np.mean(np.asarray(raw_positive_counts) > 0))
                    if raw_positive_counts
                    else None
                ),
                "mean_positive_raw_integrated_nontruth_families": (
                    float(np.mean(raw_nontruth_counts))
                    if raw_nontruth_counts
                    else None
                ),
                "maximum_raw_integrated_family_mean": (
                    float(max(raw_family_maxima)) if raw_family_maxima else None
                ),
                "raw_integrated_estimand": (
                    "raw_heldout_score_over_context_rows"
                ),
                "any_training_family_selection_rate": (
                    float(np.mean(np.asarray(training_selected) > 0))
                    if training_selected
                    else None
                ),
                "mean_training_selected_families": (
                    float(np.mean(training_selected)) if training_selected else None
                ),
                "any_heldout_family_selection_rate": (
                    float(np.mean(np.asarray(heldout_selected) > 0))
                    if heldout_selected
                    else None
                ),
            }
        )
    return output


def aggregate_campaign_metrics(
    records: Sequence[Mapping[str, object]],
    *,
    registry: Mapping[str, object],
    edge_catalog: Sequence[Mapping[str, object]],
    seed_ids: Sequence[str],
    scenarios: Sequence[str],
) -> dict[str, object]:
    """Aggregate paired edge recovery, component semantics, and family controls."""

    evaluation = _mapping(registry.get("evaluation_policy"), field="evaluation_policy")
    positive_tolerance = _float_value(
        evaluation["numerical_positive_tolerance"],
        field="evaluation_policy.numerical_positive_tolerance",
    )
    maximum_coverage_loss = _float_value(
        evaluation["maximum_allowed_coverage_loss"],
        field="evaluation_policy.maximum_allowed_coverage_loss",
    )
    score_campaign = _aggregate_score_campaign(
        records,
        edge_catalog=edge_catalog,
        seed_ids=seed_ids,
        positive_scenario=str(registry["positive_scenario"]),
        reference_scenario=str(registry["paired_margin_reference_scenario"]),
        maximum_allowed_coverage_loss=maximum_coverage_loss,
    )
    components = _aggregate_component_semantics(
        records,
        registry=registry,
        edge_catalog=edge_catalog,
        seed_ids=seed_ids,
        positive_tolerance=positive_tolerance,
    )
    return {
        "active_vs_ligand_only": score_campaign,
        "component_semantics": components,
        "control_family_false_positive_and_selection": _aggregate_control_families(
            records, scenarios=scenarios
        ),
        "coverage_policy": {
            "maximum_allowed_coverage_loss": maximum_coverage_loss,
            "coverage_is_finite_score_rows_over_candidate_rows": True,
            "structural_zero_rows_remain_observed": True,
        },
    }


def _resource_provenance(
    *,
    autonomous: ReceiverAutonomousProgramResource,
    bundle: ResourceBundle,
    target_prior: TargetPrior,
    lr_table_path: Path,
    lr_manifest_path: Path,
) -> dict[str, object]:
    return {
        "receiver_autonomous_program": {
            "artifact_id": autonomous.artifact_id,
            "registration_id": autonomous.registration_id,
            "review_scope": autonomous.review_scope,
            "verification_status": autonomous.verification_status,
            "manifest_digest": autonomous.manifest_digest,
            "matrix_digest": autonomous.matrix_digest,
            "manifest_verified_trusted": autonomous.is_manifest_verified_trusted,
            "biological_reference_trusted": (
                autonomous.is_biological_reference_trusted
            ),
        },
        "harmonized_lr": {
            "resource_id": bundle.resource_id,
            "version": bundle.version,
            "manifest_digest": bundle.manifest_digest,
            "table_filename": lr_table_path.name,
            "table_sha256": sha256_file(lr_table_path),
            "manifest_filename": lr_manifest_path.name,
            "manifest_sha256": sha256_file(lr_manifest_path),
            "n_interactions": len(bundle.interactions),
        },
        "nichenet_target_prior": {
            "resource_id": target_prior.resource_id,
            "version": target_prior.version,
            "manifest_digest": target_prior.manifest_digest,
            "driver_kind": target_prior.driver_kind,
            "shape": list(target_prior.shape),
            "nnz": target_prior.nnz,
        },
    }


def _select_seed_records(
    registry: Mapping[str, object],
    *,
    seed_count: int | None,
    seed_ids: tuple[str, ...] | None,
) -> tuple[Mapping[str, object], ...]:
    raw_seed_records = tuple(
        _mapping(item, field="seed_sets[]")
        for item in cast(Sequence[object], registry["seed_sets"])
    )
    if seed_count is not None and seed_ids is not None:
        raise ValueError("seed_count and seed_ids are mutually exclusive")
    if seed_ids is None:
        selected_count = 2 if seed_count is None else seed_count
        if not 1 <= selected_count <= len(raw_seed_records):
            raise ValueError(
                f"seed_count must lie in [1, {len(raw_seed_records)}]"
            )
        return raw_seed_records[:selected_count]

    requested_ids = tuple(seed_id.strip() for seed_id in seed_ids)
    if not requested_ids or any(not seed_id for seed_id in requested_ids):
        raise ValueError("seed_ids must be a non-empty sequence")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("seed_ids must be unique")
    records_by_id = {
        str(record["seed_id"]): record for record in raw_seed_records
    }
    unknown_ids = sorted(set(requested_ids).difference(records_by_id))
    if unknown_ids:
        raise ValueError(f"unknown campaign seed_ids: {unknown_ids}")
    return tuple(records_by_id[seed_id] for seed_id in requested_ids)


def run_campaign(
    *,
    workspace_root: Path,
    registry_path: Path = DEFAULT_INPUT_REGISTRY,
    profile_name: str = "quick",
    seed_count: int | None = None,
    seed_ids: tuple[str, ...] | None = None,
    scenarios: tuple[str, ...] = DEFAULT_DEVELOPMENT_SCENARIOS,
) -> dict[str, object]:
    """Execute selected preregistered seeds through the public workflow."""

    registry_file = registry_path.expanduser().resolve()
    registry = load_campaign_registry(registry_file)
    registry_sha256 = sha256_file(registry_file)
    profiles = _mapping(registry.get("profiles"), field="profiles")
    if profile_name not in profiles:
        raise ValueError(f"unknown campaign profile: {profile_name!r}")
    registered_scenarios = tuple(
        str(item) for item in cast(Sequence[object], registry["scenarios"])
    )
    unknown_scenarios = sorted(set(scenarios).difference(registered_scenarios))
    if unknown_scenarios:
        raise ValueError(f"unregistered campaign scenarios: {unknown_scenarios}")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be a non-empty unique sequence")
    raw_seed_records = tuple(
        _mapping(item, field="seed_sets[]")
        for item in cast(Sequence[object], registry["seed_sets"])
    )
    seed_records = _select_seed_records(
        registry,
        seed_count=seed_count,
        seed_ids=seed_ids,
    )
    executed_seed_ids = [str(record["seed_id"]) for record in seed_records]
    selected_seed_count = len(seed_records)

    repository_root = Path(__file__).resolve().parents[2]
    fixture_root = (
        repository_root
        / "benchmarks/fixtures/synthetic_receiver_autonomous_program"
    )
    autonomous = load_receiver_autonomous_program_resource(
        fixture_root,
        manifest_path=fixture_root / "manifest.json",
        registration_id=AUTONOMOUS_REGISTRATION_ID,
    )
    benchmark_root = workspace_root / "benchmark_work/multicondition_v01"
    lr_root = benchmark_root / "resources/synthetic_harmonized_lr"
    lr_table_path = lr_root / "harmonized_lr.tsv"
    lr_manifest_path = lr_root / "manifest.json"
    bundle = harmonized_resource_bundle(lr_table_path, lr_manifest_path)
    target_prior = load_nichenet_target_prior(workspace_root / "databases")
    edge_catalog = _resolve_edge_catalog(
        registry,
        lr_table_path=lr_table_path,
        bundle=bundle,
        target_prior=target_prior,
    )
    evaluation = _mapping(registry.get("evaluation_policy"), field="evaluation_policy")
    positive_tolerance = _float_value(
        evaluation["numerical_positive_tolerance"],
        field="evaluation_policy.numerical_positive_tolerance",
    )
    recovery_top_k = int(cast(int, evaluation["active_recovery_top_k"]))

    records: list[dict[str, object]] = []
    campaign_started = time.perf_counter()
    for seed_record in seed_records:
        crossfit_seed = int(cast(int, seed_record["crossfit_seed"]))
        spec = build_campaign_spec(autonomous, crossfit_seed=crossfit_seed)
        config = CrychicConfig(
            context_keys=("condition",),
            counts_layer="counts",
            design="~ condition",
            random_seed=crossfit_seed,
        )
        for scenario in scenarios:
            adata, input_audit = generate_campaign_input(
                registry,
                profile_name=profile_name,
                seed_record=seed_record,
                scenario=scenario,
                registry_sha256=registry_sha256,
            )
            if not cast(bool, input_audit["paired_contexts_complete"]):
                raise RuntimeError(
                    "generated input is not completely paired: "
                    f"{seed_record['seed_id']}"
                )
            started = time.perf_counter()
            artifacts = run_subject_crossfit(
                adata,
                config,
                bundle,
                target_prior,
                spec=spec,
            )
            elapsed = time.perf_counter() - started
            records.append(
                {
                    "seed_id": str(seed_record["seed_id"]),
                    "input_seed": int(cast(int, seed_record["input_seed"])),
                    "crossfit_seed": crossfit_seed,
                    "scenario": scenario,
                    "profile": profile_name,
                    "input": input_audit,
                    "elapsed_seconds": elapsed,
                    "process_peak_rss_kib_after_run": int(
                        process_resource.getrusage(
                            process_resource.RUSAGE_SELF
                        ).ru_maxrss
                    ),
                    **_run_algorithm_metrics(
                        artifacts,
                        edge_catalog=edge_catalog,
                        positive_tolerance=positive_tolerance,
                        recovery_top_k=recovery_top_k,
                    ),
                }
            )

    aggregate = aggregate_campaign_metrics(
        records,
        registry=registry,
        edge_catalog=edge_catalog,
        seed_ids=executed_seed_ids,
        scenarios=scenarios,
    )
    full_scenario_execution = set(scenarios) == set(registered_scenarios)
    tuning_policy = _mapping(
        registry.get("penalty_tuning_policy"), field="penalty_tuning_policy"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": registry["campaign_id"],
        "scope": (
            "single_seed_public_workflow_debug_excluded_from_campaign"
            if selected_seed_count < 2
            else (
                "development_full_seven_scenario_diagnostic"
                if full_scenario_execution
                else "development_preregistered_scenario_subset_diagnostic"
            )
        ),
        "claims": dict(AUDIT_CLAIMS),
        "certification_boundary": (
            "Public outer-heldout family-common diagnostics remain noncertifying; "
            "not-estimable receiver-program rows never count as semantic passes."
        ),
        "public_workflow_entrypoint": "crychic.workflow.run_subject_crossfit",
        "statistical_unit": "subject_id",
        "registry": {
            "relative_path": str(registry_file.relative_to(repository_root)),
            "sha256": registry_sha256,
            "locked_at": registry["locked_at"],
            "generator_schema_version": registry["generator_schema_version"],
            "all_preregistered_scenarios": list(registered_scenarios),
            "executed_scenarios": list(scenarios),
            "all_preregistered_seed_ids": [
                str(record["seed_id"]) for record in raw_seed_records
            ],
            "executed_seed_ids": executed_seed_ids,
            "profile": profile_name,
            "profile_parameters": dict(
                _mapping(profiles[profile_name], field=f"profiles.{profile_name}")
            ),
        },
        "penalty_tuning_policy": dict(tuning_policy),
        "known_edges": list(edge_catalog),
        "source_sha256": campaign_source_sha256(),
        "resource_provenance": _resource_provenance(
            autonomous=autonomous,
            bundle=bundle,
            target_prior=target_prior,
            lr_table_path=lr_table_path,
            lr_manifest_path=lr_manifest_path,
        ),
        "checks": {
            "frozen_contract_contains_all_seven_g1_5_scenarios": (
                tuple(registered_scenarios) == ALL_SCENARIOS
            ),
            "frozen_contract_contains_multiple_known_edges": len(edge_catalog) >= 2,
            "frozen_contract_contains_multiple_seeds": len(raw_seed_records) >= 2,
            "executed_multiple_seeds": selected_seed_count >= 2,
            "eligible_for_campaign_metric_interpretation": (
                selected_seed_count >= 2
            ),
            "executed_full_seven_scenario_contract": full_scenario_execution,
            "active_and_paired_ligand_only_executed": (
                "active" in scenarios and "ligand_only" in scenarios
            ),
            "all_generated_inputs_complete_paired": all(
                cast(
                    bool,
                    cast(Mapping[str, object], record["input"])[
                        "paired_contexts_complete"
                    ],
                )
                for record in records
            ),
            "all_runs_used_public_crossfit_workflow": True,
            "all_observed_receiver_programs_use_subject_equal_reference_transform": all(
                cast(
                    bool,
                    cast(
                        Mapping[str, object],
                        record["receiver_program_reference_audit"],
                    )[
                        "all_observed_models_use_subject_equal_reference_transform"
                    ],
                )
                for record in records
            ),
            "not_estimable_components_never_count_as_pass": True,
        },
        "aggregate_metrics": aggregate,
        "elapsed_seconds": time.perf_counter() - campaign_started,
        "process_peak_rss_kib": int(
            process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
        ),
        "records": records,
    }


def compact_campaign_summary(payload: Mapping[str, object]) -> dict[str, object]:
    """Remove runtime-only detail while retaining all scientific diagnostics."""

    compact_records: list[dict[str, object]] = []
    for raw in cast(Sequence[Mapping[str, object]], payload["records"]):
        input_audit = cast(Mapping[str, object], raw["input"])
        compact_records.append(
            {
                "seed_id": raw["seed_id"],
                "input_seed": raw["input_seed"],
                "crossfit_seed": raw["crossfit_seed"],
                "scenario": raw["scenario"],
                "profile": raw["profile"],
                "input_identity": input_audit["input_identity"],
                "content_sha256": input_audit["content_sha256"],
                "n_cells": input_audit["n_cells"],
                "n_genes": input_audit["n_genes"],
                "n_subjects": input_audit["n_subjects"],
                "n_samples": input_audit["n_samples"],
                "incremental_training_official_status_counts": raw[
                    "incremental_training_official_status_counts"
                ],
                "family_common_application_reason_counts": raw[
                    "family_common_application_reason_counts"
                ],
                "family_common_application_oof_certified_counts": raw[
                    "family_common_application_oof_certified_counts"
                ],
                "selected_penalties": raw["selected_penalties"],
                "receiver_program_reference_audit": raw[
                    "receiver_program_reference_audit"
                ],
                "known_edge_scores": raw["known_edge_scores"],
                "component_scores": raw["component_scores"],
                "family_diagnostics": raw["family_diagnostics"],
                "completed_stage_oof_verified": raw["completed_stage_oof_verified"],
                "complete_pipeline_oof_certified": raw[
                    "complete_pipeline_oof_certified"
                ],
                "remaining_stages": raw["remaining_stages"],
            }
        )
    return {
        key: value
        for key, value in payload.items()
        if key not in {"elapsed_seconds", "process_peak_rss_kib", "records"}
    } | {"records": compact_records}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the configurable development campaign CLI."""

    default_workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument("--registry", type=Path, default=DEFAULT_INPUT_REGISTRY)
    parser.add_argument("--profile", choices=("tiny", "quick"), default="quick")
    seed_selection = parser.add_mutually_exclusive_group()
    seed_selection.add_argument("--seed-count", type=int, default=2)
    seed_selection.add_argument("--seed-ids", nargs="+")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=list(DEFAULT_DEVELOPMENT_SCENARIOS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_work/algorithm_smoke/public_family_common_g15_campaign_v1.json"
        ),
    )
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected frozen prefix and write strict JSON outputs."""

    args = build_parser().parse_args(argv)
    workspace_root = args.workspace_root.expanduser().resolve()
    selected_seed_ids = (
        None if args.seed_ids is None else tuple(args.seed_ids)
    )
    payload = run_campaign(
        workspace_root=workspace_root,
        registry_path=args.registry,
        profile_name=args.profile,
        seed_count=(args.seed_count if selected_seed_ids is None else None),
        seed_ids=selected_seed_ids,
        scenarios=tuple(args.scenarios),
    )
    output = args.output.expanduser()
    if not output.is_absolute():
        output = workspace_root / output
    _write_json(output.resolve(), payload)
    if args.summary_output is not None:
        summary_output = args.summary_output.expanduser()
        if not summary_output.is_absolute():
            summary_output = workspace_root / summary_output
        _write_json(summary_output.resolve(), compact_campaign_summary(payload))
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "summary_output": (
                    None
                    if args.summary_output is None
                    else str(summary_output.resolve())
                ),
                "executed_seed_count": len(
                    cast(
                        Sequence[object],
                        cast(Mapping[str, object], payload["registry"])[
                            "executed_seed_ids"
                        ],
                    )
                ),
                "executed_scenarios": list(args.scenarios),
                "scope": payload["scope"],
                "claims": payload["claims"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
