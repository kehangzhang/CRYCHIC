"""Validate and expand the frozen suggest-next2 v7 benchmark protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmarks.adapters.common import json_safe, sha256_file
from crychic.core import canonical_digest, stable_id

SCHEMA_VERSION = "crychic-suggest-next2-v7-benchmark-protocol-v1"
AMENDMENT_SCHEMA_VERSION = "crychic-suggest-next2-v7-benchmark-protocol-amendment-v2"
PLAN_SCHEMA_VERSION = "crychic-suggest-next2-v7-run-plan-v2"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "suggest_next2_v7_benchmark_v1.json"
)
AMENDMENT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "suggest_next2_v7_benchmark_v2.json"
)

DESIGN_KINDS = (
    "independent_two_group",
    "independent_multi_group",
    "paired",
    "repeated",
    "multi_cohort",
    "continuous",
)
PHASES = ("smoke", "development", "locked", "null_calibration")
FAMILY_ROLES = ("development", "locked", "null_calibration")
GENERATORS = ("G0", "G1", "G2", "G3", "G4", "G5")
INFERENCES = ("I0", "I1", "I2")
PROFILES = ("score_primary", "inference_crossover")
EXPERIMENTS = (
    "E1_m0_integration",
    "E2_hard_gate_attrition",
    "E3_sender_detection_attribution",
    "E4_signed_program",
    "E5_hypergraph",
)
M4_EXPERIMENT = "E6_m4_occurrence"
REQUIRED_DIAGNOSTICS = {
    "gate_attrition",
    "score_geometry",
    "candidate_sender_bias",
    "parent_vs_sender_resolution",
    "full_pipeline_bootstrap",
    "full_pipeline_condition_permutation",
    "full_pipeline_leave_one_subject_out",
}
REQUIRED_DGP_FAMILIES = {
    "expression_joint",
    "occurrence_heterogeneity",
    "sender_decoy",
    "candidate_cardinality",
    "complex_and",
    "graph_smooth",
    "ligand_only",
    "receptor_only",
    "program_only",
    "inhibitory_program",
    "multi_sender",
    "alternative_or",
    "spatial_range",
    "hypergraph_weak_effects",
    "prior_corruption",
    "generic_state_null",
    "batch_context_confounding_null",
    "composition_only_null",
    "receiver_autonomous_null",
    "abundance_only_null",
    "topology_jump",
    "wrong_topology",
    "disconnected_graph",
    "collinear_lr",
    "structural_absence",
    "annotation_perturbation",
    "prior_replacement",
    "context_correlated_sampling_missingness",
    "global_null",
    "scoring_functional_null",
    "fixed_subject_increasing_cell_null",
    "legal_condition_permutation_null",
}

PLAN_COLUMNS = (
    "plan_schema_version",
    "protocol_digest",
    "phase",
    "profile",
    "family_role",
    "dgp_family",
    "design_kind",
    "replicate_index",
    "seed",
    "dataset_id",
    "candidate_sender_count",
    "cells_per_type",
    "subjects_per_level",
    "generator_id",
    "inference_id",
    "run_id",
)


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    observed = set(map(str, value))
    if observed != expected:
        raise ValueError(
            f"{field} keys differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _canonical_config_digest(config: Mapping[str, Any]) -> str:
    return str(canonical_digest(json.loads(json.dumps(config, sort_keys=True))))


def _validate_execution_tiers(config: Mapping[str, Any]) -> None:
    tiers = _mapping(config.get("execution_tiers"), field="execution_tiers")
    _exact_keys(tiers, set(PHASES), field="execution_tiers")
    counts: dict[str, int] = {}
    for phase in PHASES:
        tier = _mapping(tiers[phase], field=f"execution_tiers.{phase}")
        count = _positive_integer(
            tier.get("replicates_per_family_design"),
            field=f"execution_tiers.{phase}.replicates_per_family_design",
        )
        if not isinstance(tier.get("full_pipeline"), bool):
            raise ValueError(f"execution_tiers.{phase}.full_pipeline must be boolean")
        if not isinstance(tier.get("publication_role"), str):
            raise ValueError(
                f"execution_tiers.{phase}.publication_role must be a string"
            )
        counts[phase] = count
    if counts["smoke"] != 20 or counts["development"] != 100:
        raise ValueError("smoke/development replicate counts must be 20/100")
    if not 200 <= counts["locked"] <= 500:
        raise ValueError("locked replicate count must lie in [200, 500]")
    if counts["null_calibration"] < 1000:
        raise ValueError("null calibration requires at least 1000 replicates")
    if bool(_mapping(tiers["smoke"], field="smoke")["full_pipeline"]):
        raise ValueError("smoke must not claim a full-pipeline campaign")
    for phase in ("development", "locked", "null_calibration"):
        if not bool(_mapping(tiers[phase], field=phase)["full_pipeline"]):
            raise ValueError(f"{phase} must rerun the full pipeline")


def _validate_seed_policy(config: Mapping[str, Any]) -> None:
    policy = _mapping(config.get("seed_policy"), field="seed_policy")
    expected_algorithm = (
        "sha256_namespace_family_design_replicate_first_31_bits_nonzero"
    )
    if policy.get("algorithm") != expected_algorithm:
        raise ValueError("seed_policy.algorithm changed")
    namespaces = _mapping(policy.get("namespaces"), field="seed_policy.namespaces")
    _exact_keys(namespaces, set(PHASES), field="seed_policy.namespaces")
    values = tuple(str(namespaces[phase]) for phase in PHASES)
    if any(not value or value != value.strip() for value in values):
        raise ValueError("seed namespaces must be canonical non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("seed namespaces must be disjoint across phases")


def _validate_dgp_generation(config: Mapping[str, Any]) -> None:
    generation = _mapping(config.get("dgp_generation"), field="dgp_generation")
    _exact_keys(
        generation,
        {
            "candidate_sender_default",
            "candidate_cardinality_cycle",
            "fixed_subject_cell_count_cycle",
            "cells_per_type_by_phase",
            "subjects_per_level_by_phase",
            "legacy_g1_tuning",
        },
        field="dgp_generation",
    )
    if generation["candidate_sender_default"] != 5:
        raise ValueError("candidate_sender_default must remain 5")
    if tuple(generation["candidate_cardinality_cycle"]) != (2, 5, 10, 20):
        raise ValueError("candidate_cardinality_cycle must remain 2/5/10/20")
    if tuple(generation["fixed_subject_cell_count_cycle"]) != (1, 2, 4, 8):
        raise ValueError("fixed_subject_cell_count_cycle must remain 1/2/4/8")
    for field_name, expected in (
        (
            "cells_per_type_by_phase",
            {"smoke": 2, "development": 4, "locked": 4, "null_calibration": 2},
        ),
        (
            "subjects_per_level_by_phase",
            {phase: 8 for phase in PHASES},
        ),
    ):
        values = _mapping(generation[field_name], field=f"dgp_generation.{field_name}")
        _exact_keys(values, set(PHASES), field=f"dgp_generation.{field_name}")
        observed = {
            phase: _positive_integer(
                values[phase], field=f"dgp_generation.{field_name}.{phase}"
            )
            for phase in PHASES
        }
        if observed != expected:
            raise ValueError(f"dgp_generation.{field_name} changed")
    tuning = _mapping(
        generation["legacy_g1_tuning"],
        field="dgp_generation.legacy_g1_tuning",
    )
    _exact_keys(
        tuning,
        {
            "lambda1_fractions",
            "lambda2_fractions",
            "inner_allowed_n_splits",
            "minimum_inner_train_subjects_per_context",
            "minimum_inner_validation_subjects_per_context",
            "root_seed",
        },
        field="dgp_generation.legacy_g1_tuning",
    )
    expected_tuning = {
        "lambda1_fractions": [1.0, 0.1],
        "lambda2_fractions": [0.0],
        "inner_allowed_n_splits": [2],
        "minimum_inner_train_subjects_per_context": 1,
        "minimum_inner_validation_subjects_per_context": 1,
        "root_seed": "dataset_seed",
    }
    if dict(tuning) != expected_tuning:
        raise ValueError("legacy G1 tuning policy changed")


@dataclass(frozen=True, slots=True)
class V7DGPScale:
    """Frozen generated-data scale for one planned replicate."""

    candidate_sender_count: int
    cells_per_type: int
    subjects_per_level: int


def resolve_v7_dgp_scale(
    protocol: V7BenchmarkProtocol,
    *,
    phase: str,
    dgp_family: str,
    replicate_index: int,
) -> V7DGPScale:
    """Resolve cardinality and cell/subject scales from the frozen protocol."""

    if phase not in PHASES:
        raise ValueError(f"phase must be one of {list(PHASES)}")
    replicate = _positive_integer(replicate_index, field="replicate_index")
    generation = _mapping(protocol.config["dgp_generation"], field="dgp_generation")
    candidate_cycle = tuple(map(int, generation["candidate_cardinality_cycle"]))
    cell_cycle = tuple(map(int, generation["fixed_subject_cell_count_cycle"]))
    candidate_count = (
        candidate_cycle[(replicate - 1) % len(candidate_cycle)]
        if dgp_family == "candidate_cardinality"
        else int(generation["candidate_sender_default"])
    )
    cells = (
        cell_cycle[(replicate - 1) % len(cell_cycle)]
        if dgp_family == "fixed_subject_increasing_cell_null"
        else int(
            _mapping(
                generation["cells_per_type_by_phase"],
                field="dgp_generation.cells_per_type_by_phase",
            )[phase]
        )
    )
    subjects = int(
        _mapping(
            generation["subjects_per_level_by_phase"],
            field="dgp_generation.subjects_per_level_by_phase",
        )[phase]
    )
    return V7DGPScale(
        candidate_sender_count=candidate_count,
        cells_per_type=cells,
        subjects_per_level=subjects,
    )


def _validate_dgp_families(config: Mapping[str, Any]) -> None:
    declared_designs = tuple(config.get("design_kinds", ()))
    if declared_designs != DESIGN_KINDS:
        raise ValueError("design_kinds must equal the frozen ordered design axis")
    families = _mapping(config.get("dgp_families"), field="dgp_families")
    _exact_keys(families, set(FAMILY_ROLES), field="dgp_families")
    observed: dict[str, str] = {}
    for role in FAMILY_ROLES:
        role_families = _mapping(families[role], field=f"dgp_families.{role}")
        if not role_families:
            raise ValueError(f"dgp_families.{role} cannot be empty")
        for family, design_values in role_families.items():
            family_name = str(family)
            if family_name in observed:
                raise ValueError(
                    f"DGP family {family_name!r} leaks across "
                    f"{observed[family_name]} and {role}"
                )
            if not isinstance(design_values, list) or not design_values:
                raise ValueError(f"DGP family {family_name} has no design kinds")
            designs = tuple(map(str, design_values))
            if len(designs) != len(set(designs)) or not set(designs).issubset(
                DESIGN_KINDS
            ):
                raise ValueError(f"DGP family {family_name} has invalid designs")
            observed[family_name] = role
    if set(observed) != REQUIRED_DGP_FAMILIES:
        raise ValueError(
            "DGP family axis differs; "
            f"missing={sorted(REQUIRED_DGP_FAMILIES - set(observed))}, "
            f"extra={sorted(set(observed) - REQUIRED_DGP_FAMILIES)}"
        )
    calibration = _mapping(
        families["null_calibration"], field="dgp_families.null_calibration"
    )
    if tuple(calibration["global_null"]) != DESIGN_KINDS:
        raise ValueError("global_null must cover every supported design kind")


def _validate_method_matrix(config: Mapping[str, Any]) -> None:
    generators = _mapping(config.get("generator_matrix"), field="generator_matrix")
    _exact_keys(generators, set(GENERATORS), field="generator_matrix")
    for generator_id in GENERATORS:
        generator = _mapping(
            generators[generator_id], field=f"generator_matrix.{generator_id}"
        )
        if not isinstance(generator.get("name"), str) or not isinstance(
            generator.get("formula"), str
        ):
            raise ValueError(f"generator {generator_id} lacks name/formula")
        if not isinstance(generator.get("outcome_agnostic"), bool) or not isinstance(
            generator.get("condition_gate"), bool
        ):
            raise ValueError(f"generator {generator_id} lacks gate semantics")
    if (
        generators["G1"].get("outcome_agnostic") is not False
        or generators["G1"].get("condition_gate") is not True
    ):
        raise ValueError("G1 must remain the outcome-dependent legacy comparator")
    for generator_id in ("G0", "G2", "G3", "G4", "G5"):
        if (
            generators[generator_id].get("outcome_agnostic") is not True
            or generators[generator_id].get("condition_gate") is not False
        ):
            raise ValueError(f"{generator_id} must be outcome agnostic")

    inferences = _mapping(config.get("inference_matrix"), field="inference_matrix")
    _exact_keys(inferences, set(INFERENCES), field="inference_matrix")
    if any(
        _mapping(inferences[item], field=f"inference_matrix.{item}").get(
            "formal_inference_eligible"
        )
        is not False
        for item in INFERENCES
    ):
        raise ValueError("analytic inference arms cannot claim formal eligibility")

    profiles = _mapping(config.get("plan_profiles"), field="plan_profiles")
    _exact_keys(profiles, set(PROFILES), field="plan_profiles")
    primary = _mapping(profiles["score_primary"], field="score_primary")
    crossover = _mapping(profiles["inference_crossover"], field="inference_crossover")
    if tuple(primary.get("generators", ())) != GENERATORS or tuple(
        primary.get("inferences", ())
    ) != ("I1",):
        raise ValueError("score_primary must freeze G0-G5 at I1")
    if (
        tuple(crossover.get("generators", ())) != ("G3",)
        or tuple(crossover.get("inferences", ())) != INFERENCES
    ):
        raise ValueError("inference_crossover must freeze G3 at I0-I2")


def _validate_experiments(config: Mapping[str, Any]) -> None:
    experiments = _mapping(config.get("experiments"), field="experiments")
    amended = "_amendment_provenance" in config
    expected = {*EXPERIMENTS, *((M4_EXPERIMENT,) if amended else ())}
    _exact_keys(experiments, expected, field="experiments")
    diagnostics = config.get("required_diagnostics")
    if not isinstance(diagnostics, list) or set(map(str, diagnostics)) != (
        REQUIRED_DIAGNOSTICS
    ):
        raise ValueError("required diagnostic axis differs")
    e1 = _mapping(experiments["E1_m0_integration"], field="E1")
    if tuple(e1.get("arms", ())) != ("G0-I1", "G1-I1", "G2-I1", "G3-I1"):
        raise ValueError("E1 M0 integration arms changed")
    e3 = _mapping(experiments["E3_sender_detection_attribution"], field="E3")
    if tuple(e3.get("candidate_counts", ())) != (2, 5, 10, 20):
        raise ValueError("E3 candidate counts must be 2/5/10/20")
    if amended:
        m4 = _mapping(experiments[M4_EXPERIMENT], field=M4_EXPERIMENT)
        if (
            m4.get("parameter_role") != "fixed_baseline_before_m4_development"
            or m4.get("activity_head") != "parent_mean_raw"
            or float(m4.get("activity_threshold_raw", math.nan)) != 1.0
            or float(m4.get("probability_transition_scale", math.nan)) != 0.25
        ):
            raise ValueError("E6 M4 fixed baseline policy changed")
        expected_methods = (
            "raw_prevalence_difference",
            "fisher_exact",
            "dcst_protocol_compatible_exact",
            "logistic_model",
            "beta_binomial",
            "crychic_m4_design_aware",
        )
        if tuple(m4.get("methods", ())) != expected_methods:
            raise ValueError("E6 M4 method axis changed")
        required_metrics = {
            "estimable_fraction",
            "occurrence_auprc",
            "occurrence_auroc",
            "prevalence_effect_rmse",
            "prevalence_effect_bias",
            "direction_accuracy",
            "odds_ratio_bias",
            "log_odds_rmse",
            "subject_brier_score",
            "calibration_intercept",
            "calibration_slope",
            "type1_error_alpha_0_05",
            "fdr_at_q_0_10",
            "power_at_q_0_10",
        }
        if set(map(str, m4.get("metrics", ()))) != required_metrics:
            raise ValueError("E6 M4 metric axis changed")


def _validate_release_gates(config: Mapping[str, Any]) -> None:
    gates = _mapping(config.get("release_gates"), field="release_gates")
    if tuple(gates.get("type1_interval", ())) != (0.035, 0.065):
        raise ValueError("type-I release interval changed")
    if float(gates.get("fdr_maximum_at_q_0_10", math.nan)) != 0.12:
        raise ValueError("FDR release limit changed")
    if tuple(gates.get("coverage95_interval", ())) != (0.92, 0.97):
        raise ValueError("coverage release interval changed")


@dataclass(frozen=True, slots=True)
class V7BenchmarkProtocol:
    """Validated frozen configuration and its content digest."""

    path: Path
    config: Mapping[str, Any]
    protocol_digest: str
    source_schema_version: str = SCHEMA_VERSION

    @property
    def family_roles(self) -> Mapping[str, Mapping[str, Sequence[str]]]:
        return cast(
            Mapping[str, Mapping[str, Sequence[str]]],
            self.config["dgp_families"],
        )

    def replicate_count(self, phase: str) -> int:
        tier = _mapping(
            _mapping(self.config["execution_tiers"], field="execution_tiers")[phase],
            field=f"execution_tiers.{phase}",
        )
        return int(tier["replicates_per_family_design"])

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.source_schema_version,
            "protocol_path": str(self.path),
            "protocol_sha256": sha256_file(self.path),
            "protocol_digest": self.protocol_digest,
            "minimum_estimator_commit": self.config["minimum_estimator_commit"],
            "dgp_family_counts": {
                role: len(self.family_roles[role]) for role in FAMILY_ROLES
            },
            "generators": list(GENERATORS),
            "inferences": list(INFERENCES),
            "amendment": self.config.get("_amendment_provenance"),
        }


def _resolve_protocol_config(
    resolved: Path,
) -> tuple[dict[str, Any], str]:
    raw = _read_json(resolved)
    schema = str(raw.get("schema_version", ""))
    if schema == SCHEMA_VERSION:
        if raw.get("status") != "preregistered_before_integrated_v7_campaign":
            raise ValueError("v7 benchmark protocol status is not preregistered")
        return raw, schema
    if schema != AMENDMENT_SCHEMA_VERSION:
        raise ValueError("v7 benchmark protocol schema is unsupported")
    if raw.get("status") != "amended_after_smoke_before_development":
        raise ValueError("v7 benchmark amendment status is unsupported")
    base_record = _mapping(raw.get("base_protocol"), field="base_protocol")
    if set(base_record) != {"filename", "sha256"}:
        raise ValueError("base_protocol must contain filename and sha256")
    filename = str(base_record["filename"])
    if Path(filename).name != filename:
        raise ValueError("base_protocol filename must be a local path component")
    base_path = resolved.parent / filename
    if sha256_file(base_path) != base_record["sha256"]:
        raise ValueError("base_protocol checksum differs from the amendment")
    base = _read_json(base_path)
    if (
        base.get("schema_version") != SCHEMA_VERSION
        or base.get("status") != "preregistered_before_integrated_v7_campaign"
    ):
        raise ValueError("base_protocol is not the supported frozen v1 protocol")
    experiment = _mapping(raw.get("experiment"), field="experiment")
    if experiment.get("name") != M4_EXPERIMENT:
        raise ValueError("v7 amendment must register E6_m4_occurrence")
    merged = json.loads(json.dumps(base))
    merged["minimum_estimator_commit"] = raw.get("minimum_estimator_commit")
    merged["experiments"][M4_EXPERIMENT] = {
        key: value for key, value in experiment.items() if key != "name"
    }
    merged["_amendment_provenance"] = {
        "schema_version": schema,
        "status": raw["status"],
        "amendment_reason": raw.get("amendment_reason"),
        "base_protocol_path": str(base_path.resolve()),
        "base_protocol_sha256": str(base_record["sha256"]),
    }
    return merged, schema


def load_v7_benchmark_protocol(path: Path = DEFAULT_CONFIG) -> V7BenchmarkProtocol:
    """Load the protocol and reject any change to its required experiment axes."""

    resolved = path.resolve()
    config, source_schema = _resolve_protocol_config(resolved)
    commit = str(config.get("minimum_estimator_commit", ""))
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("minimum_estimator_commit must be a full lowercase Git hash")
    _validate_execution_tiers(config)
    _validate_dgp_generation(config)
    _validate_seed_policy(config)
    _validate_dgp_families(config)
    _validate_method_matrix(config)
    _validate_experiments(config)
    _validate_release_gates(config)
    execution = _mapping(config.get("execution"), field="execution")
    fraction = float(execution.get("maximum_memory_fraction", 0.0))
    if fraction != 0.8:
        raise ValueError("maximum_memory_fraction must remain 0.8")
    return V7BenchmarkProtocol(
        path=resolved,
        config=config,
        protocol_digest=_canonical_config_digest(config),
        source_schema_version=source_schema,
    )


def _seed(
    namespace: str,
    *,
    dgp_family: str,
    design_kind: str,
    replicate_index: int,
) -> int:
    payload = f"{namespace}|{dgp_family}|{design_kind}|{replicate_index}".encode(
        "ascii"
    )
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
    return value if value else 1


def _phase_families(
    protocol: V7BenchmarkProtocol,
    phase: str,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    roles = FAMILY_ROLES if phase == "smoke" else (phase,)
    records: list[tuple[str, str, tuple[str, ...]]] = []
    for role in roles:
        for family, designs in protocol.family_roles[role].items():
            records.append((role, str(family), tuple(map(str, designs))))
    return tuple(sorted(records))


def expand_v7_benchmark_plan(
    protocol: V7BenchmarkProtocol,
    *,
    phase: str,
    profile: str,
    maximum_replicates: int | None = None,
) -> pd.DataFrame:
    """Expand one paired score/inference plan with deterministic dataset seeds."""

    if not isinstance(protocol, V7BenchmarkProtocol):
        raise TypeError("protocol must be V7BenchmarkProtocol")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {list(PHASES)}")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {list(PROFILES)}")
    configured = protocol.replicate_count(phase)
    if maximum_replicates is None:
        replicates = configured
    else:
        limit = _positive_integer(maximum_replicates, field="maximum_replicates")
        if limit > configured:
            raise ValueError("maximum_replicates cannot exceed the frozen tier")
        replicates = limit
    namespaces = _mapping(
        _mapping(protocol.config["seed_policy"], field="seed_policy")["namespaces"],
        field="seed_policy.namespaces",
    )
    namespace = str(namespaces[phase])
    profile_config = _mapping(
        _mapping(protocol.config["plan_profiles"], field="plan_profiles")[profile],
        field=f"plan_profiles.{profile}",
    )
    generators = tuple(map(str, profile_config["generators"]))
    inferences = tuple(map(str, profile_config["inferences"]))
    records: list[dict[str, object]] = []
    dataset_keys: dict[str, int] = {}
    for role, family, designs in _phase_families(protocol, phase):
        for design_kind in sorted(designs):
            for replicate_index in range(1, replicates + 1):
                seed = _seed(
                    namespace,
                    dgp_family=family,
                    design_kind=design_kind,
                    replicate_index=replicate_index,
                )
                dataset_id = f"v7_{phase}_{family}_{design_kind}_r{replicate_index:04d}"
                if dataset_id in dataset_keys or seed in dataset_keys.values():
                    raise RuntimeError("v7 dataset ID or seed collision")
                dataset_keys[dataset_id] = seed
                scale = resolve_v7_dgp_scale(
                    protocol,
                    phase=phase,
                    dgp_family=family,
                    replicate_index=replicate_index,
                )
                for generator_id in generators:
                    for inference_id in inferences:
                        run_id = stable_id(
                            "v7_benchmark_run",
                            {
                                "dataset_id": dataset_id,
                                "generator_id": generator_id,
                                "inference_id": inference_id,
                                "profile": profile,
                                "protocol_digest": protocol.protocol_digest,
                            },
                        )
                        records.append(
                            {
                                "plan_schema_version": PLAN_SCHEMA_VERSION,
                                "protocol_digest": protocol.protocol_digest,
                                "phase": phase,
                                "profile": profile,
                                "family_role": role,
                                "dgp_family": family,
                                "design_kind": design_kind,
                                "replicate_index": replicate_index,
                                "seed": seed,
                                "dataset_id": dataset_id,
                                "candidate_sender_count": (
                                    scale.candidate_sender_count
                                ),
                                "cells_per_type": scale.cells_per_type,
                                "subjects_per_level": scale.subjects_per_level,
                                "generator_id": generator_id,
                                "inference_id": inference_id,
                                "run_id": run_id,
                            }
                        )
    plan = pd.DataFrame.from_records(records, columns=PLAN_COLUMNS).sort_values(
        [
            "family_role",
            "dgp_family",
            "design_kind",
            "replicate_index",
            "generator_id",
            "inference_id",
        ],
        kind="stable",
        ignore_index=True,
    )
    if plan.empty or plan["run_id"].duplicated().any():
        raise RuntimeError("expanded v7 plan is empty or has duplicate run IDs")
    dataset_groups = plan.groupby("dataset_id", observed=True)
    if dataset_groups["seed"].nunique().ne(1).any():
        raise RuntimeError("method arms do not share one paired dataset seed")
    expected_arms = len(generators) * len(inferences)
    if dataset_groups.size().ne(expected_arms).any():
        raise RuntimeError("expanded datasets do not cover the exact method matrix")
    return plan


def write_v7_benchmark_plan(
    protocol: V7BenchmarkProtocol,
    output_dir: Path,
    *,
    phase: str,
    profile: str,
    maximum_replicates: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Atomically write a compact TSV plan and checksum-bound manifest."""

    output = output_dir.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}; pass overwrite=True")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        plan = expand_v7_benchmark_plan(
            protocol,
            phase=phase,
            profile=profile,
            maximum_replicates=maximum_replicates,
        )
        plan_path = staged / "run_plan.tsv"
        plan.to_csv(plan_path, sep="\t", index=False, lineterminator="\n")
        manifest = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "status": "planned_not_executed",
            "phase": phase,
            "profile": profile,
            "protocol": protocol.to_manifest(),
            "rows": len(plan),
            "datasets": int(plan["dataset_id"].nunique()),
            "output": {
                "filename": plan_path.name,
                "sha256": sha256_file(plan_path),
            },
        }
        manifest_path = staged / "manifest.json"
        manifest_path.write_text(
            json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            backup = output.with_name(f".{output.name}.previous")
            if backup.exists():
                raise FileExistsError(f"stale plan backup exists: {backup}")
            os.replace(output, backup)
            try:
                os.replace(staged, output)
            except BaseException:
                os.replace(backup, output)
                raise
            for child in backup.iterdir():
                child.unlink()
            backup.rmdir()
        else:
            os.replace(staged, output)
        return manifest
    finally:
        if staged.exists():
            for child in staged.iterdir():
                child.unlink()
            staged.rmdir()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--maximum-replicates", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_v7_benchmark_protocol(args.config)
    manifest = write_v7_benchmark_plan(
        protocol,
        args.output_dir,
        phase=args.phase,
        profile=args.profile,
        maximum_replicates=args.maximum_replicates,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
