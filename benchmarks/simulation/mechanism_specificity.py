"""Deterministic held-out evidence generation for the G1.5 benchmark."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from benchmarks.adapters.common import canonical_digest
from benchmarks.metrics.mechanism_specificity import (
    REQUIRED_SCENARIOS,
    ComponentTruthMatrix,
)
from crychic.core import SeedLineage
from crychic.scoring import (
    apply_incremental_downstream_functional,
    fit_incremental_downstream_functional,
    mechanistic_strength,
)

GENERATOR_SCHEMA_VERSION = "crychic-g1.5-deterministic-generator-v1"
DEVELOPMENT_PHASE = "development"
HOLDOUT_PHASE = "independent_holdout"
PHASES = (DEVELOPMENT_PHASE, HOLDOUT_PHASE)
PHASE_ROOT_SEEDS: Final[dict[str, int]] = {
    DEVELOPMENT_PHASE: 1_517_130_527,
    HOLDOUT_PHASE: 2_026_071_301,
}
PHASE_SEED_NAMESPACES: Final[dict[str, str]] = {
    DEVELOPMENT_PHASE: "crychic:g1.5:v2:development:frozen-20260713",
    HOLDOUT_PHASE: "crychic:g1.5:v2:independent-holdout:frozen-20260713",
}
EVIDENCE_COLUMNS = (
    "seed",
    "scenario",
    "known_edge_id",
    "availability_effect",
    "receptor_gate",
    "receiver_program_effect",
    "incremental_downstream_effect",
    "sender_effect",
    "integrated_lr_effect",
    "comparison_coverage",
    "reference_comparison_coverage",
    "known_edge_rank",
    "known_edge_positive_direction",
    "status",
    "reason_code",
)
FEATURE_IDS = ("lr_target", "autonomous_target", "background_target")
NUISANCE_COLUMN_IDS = ("intercept",)
N_TRAIN_SUBJECTS = 10
N_TEST_SUBJECTS = 8
MINIMUM_SCALE = 0.25
NULL_LOSS_FLOOR = 1e-8
LAMBDA1 = 0.0
LAMBDA2 = 1e-8
NUMERICAL_ZERO_TOLERANCE = 1e-12
SOFTMIN_POWER = 4.0
SOFTMIN_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class EdgeProfile:
    """Frozen edge-specific signal and quality parameters."""

    known_edge_id: str
    receiver: str
    family_id: str
    expression_scale: float
    availability_scale: float
    receptor_level: float
    sender_level: float
    prior_quality: float

    def __post_init__(self) -> None:
        if not self.known_edge_id or not self.receiver or not self.family_id:
            raise ValueError("edge profile identifiers must be non-empty")
        if not math.isfinite(self.expression_scale) or self.expression_scale <= 0:
            raise ValueError("expression_scale must be finite and positive")
        for field_name in (
            "availability_scale",
            "receptor_level",
            "sender_level",
            "prior_quality",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{field_name} must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class ScenarioMechanism:
    """Interventions used to generate components, never integrated scores."""

    lr_program: bool = False
    autonomous_program: bool = False
    state_availability: bool = False
    ecosystem_abundance: bool = False
    receptor_present: bool = False
    sender_present: bool = False


EDGE_PROFILES: Final[dict[str, EdgeProfile]] = {
    "synthetic_cxcl10_cxcr3": EdgeProfile(
        known_edge_id="synthetic_cxcl10_cxcr3",
        receiver="Receiver_CXCR3",
        family_id="interferon_chemokine",
        expression_scale=1.00,
        availability_scale=0.68,
        receptor_level=0.88,
        sender_level=0.84,
        prior_quality=0.96,
    ),
    "synthetic_ccl5_ccr5": EdgeProfile(
        known_edge_id="synthetic_ccl5_ccr5",
        receiver="Receiver_CCR5",
        family_id="inflammatory_chemokine",
        expression_scale=0.92,
        availability_scale=0.62,
        receptor_level=0.83,
        sender_level=0.79,
        prior_quality=0.93,
    ),
    "synthetic_egf_egfr": EdgeProfile(
        known_edge_id="synthetic_egf_egfr",
        receiver="Receiver_EGFR",
        family_id="growth_factor",
        expression_scale=0.84,
        availability_scale=0.57,
        receptor_level=0.78,
        sender_level=0.74,
        prior_quality=0.90,
    ),
}

SCENARIO_MECHANISMS: Final[dict[str, ScenarioMechanism]] = {
    "active": ScenarioMechanism(
        lr_program=True,
        state_availability=True,
        receptor_present=True,
        sender_present=True,
    ),
    "global_null": ScenarioMechanism(),
    "abundance_only": ScenarioMechanism(
        ecosystem_abundance=True,
        receptor_present=True,
        sender_present=True,
    ),
    "ligand_only": ScenarioMechanism(
        state_availability=True,
        receptor_present=True,
        sender_present=True,
    ),
    "target_only": ScenarioMechanism(
        lr_program=True,
        receptor_present=True,
    ),
    "receiver_autonomous": ScenarioMechanism(
        autonomous_program=True,
        receptor_present=True,
    ),
    "receptor_knockout": ScenarioMechanism(
        lr_program=True,
        state_availability=True,
        sender_present=True,
    ),
}


@dataclass(frozen=True, slots=True)
class GeneratedMechanismEvidence:
    """Evidence plus compact provenance needed by the atomic runner."""

    phase: str
    evidence: pd.DataFrame
    seed_lineages: tuple[SeedLineage, ...]
    functional_ids: tuple[str, ...]
    application_statuses: tuple[str, ...]
    null_losses: tuple[float | None, ...]
    ecosystem_effects: tuple[float, ...]

    @property
    def seed_count(self) -> int:
        return len(self.seed_lineages)

    @property
    def row_count(self) -> int:
        return len(self.evidence)

    def audit_summary(self) -> dict[str, object]:
        """Return deterministic aggregate provenance for the manifest."""

        observed_losses = np.asarray(
            [value for value in self.null_losses if value is not None], dtype=float
        )
        ecosystem = np.asarray(self.ecosystem_effects, dtype=float)
        return {
            "model_call_counts": {
                "fit_incremental_downstream_functional": self.row_count,
                "apply_incremental_downstream_functional": self.row_count,
                "mechanistic_strength": self.row_count,
            },
            "application_status_counts": dict(
                sorted(Counter(self.application_statuses).items())
            ),
            "incremental_functional_id_digest": canonical_digest(
                {"ids": list(self.functional_ids)}, prefix="functional-set"
            ),
            "unique_incremental_functional_ids": len(set(self.functional_ids)),
            "heldout_null_loss": {
                "n_observed": len(observed_losses),
                "minimum": (
                    float(observed_losses.min()) if len(observed_losses) else None
                ),
                "maximum": (
                    float(observed_losses.max()) if len(observed_losses) else None
                ),
            },
            "abundance_only_ecosystem_effect": {
                "n": len(ecosystem),
                "minimum": float(ecosystem.min()) if len(ecosystem) else None,
                "maximum": float(ecosystem.max()) if len(ecosystem) else None,
                "mean": float(ecosystem.mean()) if len(ecosystem) else None,
            },
        }


@dataclass(frozen=True, slots=True)
class _SharedSeedEdgeInputs:
    """Scenario-independent paired noise for one seed and registered edge."""

    signal_multiplier: float
    train_reference_expression: np.ndarray
    test_reference_expression: np.ndarray
    state_reference: np.ndarray
    ecosystem_reference: np.ndarray
    receptor_noise: np.ndarray
    sender_noise: np.ndarray


def phase_seed_lineages(phase: str, seed_count: int) -> tuple[SeedLineage, ...]:
    """Return frozen, call-order-independent campaign seeds."""

    if phase not in PHASES:
        raise ValueError(f"unknown G1.5 phase: {phase!r}")
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    root = SeedLineage(PHASE_ROOT_SEEDS[phase]).derive(PHASE_SEED_NAMESPACES[phase])
    lineages = tuple(
        root.derive("campaign-seed", f"index={index:03d}")
        for index in range(seed_count)
    )
    if len({lineage.seed for lineage in lineages}) != len(lineages):
        raise RuntimeError("frozen seed namespace produced a collision")
    return lineages


def _zero(value: float) -> float:
    return 0.0 if abs(value) <= NUMERICAL_ZERO_TOLERANCE else float(value)


def _shared_inputs(lineage: SeedLineage, profile: EdgeProfile) -> _SharedSeedEdgeInputs:
    edge_lineage = lineage.derive(f"edge={profile.known_edge_id}")
    expression_rng = np.random.default_rng(
        edge_lineage.derive("paired-receiver-expression").seed
    )
    component_rng = np.random.default_rng(edge_lineage.derive("paired-components").seed)
    signal_rng = np.random.default_rng(edge_lineage.derive("signal-scale").seed)
    expression_center = np.asarray([4.0, 3.2, 2.4], dtype=float)
    train_reference = expression_center + expression_rng.normal(
        0.0, 0.35, size=(N_TRAIN_SUBJECTS, len(FEATURE_IDS))
    )
    test_reference = expression_center + expression_rng.normal(
        0.0, 0.35, size=(N_TEST_SUBJECTS, len(FEATURE_IDS))
    )
    return _SharedSeedEdgeInputs(
        signal_multiplier=float(signal_rng.uniform(0.90, 1.10)),
        train_reference_expression=train_reference,
        test_reference_expression=test_reference,
        state_reference=np.clip(
            0.12 + component_rng.normal(0.0, 0.015, N_TEST_SUBJECTS),
            0.05,
            0.20,
        ),
        ecosystem_reference=np.clip(
            0.18 + component_rng.normal(0.0, 0.02, N_TEST_SUBJECTS),
            0.08,
            0.28,
        ),
        receptor_noise=component_rng.normal(0.0, 0.015, N_TEST_SUBJECTS),
        sender_noise=component_rng.normal(0.0, 0.015, N_TEST_SUBJECTS),
    )


def _expression_effect(
    mechanism: ScenarioMechanism,
    profile: EdgeProfile,
    shared: _SharedSeedEdgeInputs,
) -> np.ndarray:
    multiplier = profile.expression_scale * shared.signal_multiplier
    result: np.ndarray = np.asarray(
        [
            1.25 * multiplier if mechanism.lr_program else 0.0,
            1.10 * multiplier if mechanism.autonomous_program else 0.0,
            0.0,
        ],
        dtype=float,
    )
    return result


def _paired_context_matrix(
    reference: np.ndarray, effect: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = reference + effect
    matrix = np.vstack([reference, target])
    n_subjects = len(reference)
    regressor = np.concatenate(
        [-np.ones(n_subjects, dtype=float), np.ones(n_subjects, dtype=float)]
    )
    nuisance: np.ndarray = np.ones((2 * n_subjects, 1), dtype=float)
    return matrix, regressor, nuisance


def _component_values(
    mechanism: ScenarioMechanism,
    profile: EdgeProfile,
    shared: _SharedSeedEdgeInputs,
) -> tuple[float, float, float, float]:
    multiplier = shared.signal_multiplier
    availability_increment = (
        profile.availability_scale * multiplier if mechanism.state_availability else 0.0
    )
    state_target = shared.state_reference + availability_increment
    availability_effect = _zero(float(np.mean(state_target - shared.state_reference)))

    ecosystem_increment = (
        0.55 * profile.availability_scale * multiplier
        if mechanism.ecosystem_abundance
        else 0.0
    )
    ecosystem_target = shared.ecosystem_reference + ecosystem_increment
    ecosystem_effect = _zero(
        float(np.mean(ecosystem_target - shared.ecosystem_reference))
    )

    receptor_values = (
        np.clip(
            profile.receptor_level * multiplier + shared.receptor_noise,
            0.0,
            1.0,
        )
        if mechanism.receptor_present
        else np.zeros(N_TEST_SUBJECTS, dtype=float)
    )
    sender_values = (
        np.clip(
            profile.sender_level * multiplier + shared.sender_noise,
            0.0,
            1.0,
        )
        if mechanism.sender_present
        else np.zeros(N_TEST_SUBJECTS, dtype=float)
    )
    return (
        availability_effect,
        _zero(float(np.mean(receptor_values))),
        _zero(float(np.mean(sender_values))),
        ecosystem_effect,
    )


def _receiver_program_effect(
    test_response: np.ndarray, test_regressor: np.ndarray
) -> float:
    reference = test_response[test_regressor < 0]
    target = test_response[test_regressor > 0]
    paired_effect = np.mean(target - reference, axis=0)
    positive_norm = float(np.linalg.norm(np.maximum(paired_effect, 0.0)))
    return _zero(1.0 - math.exp(-positive_norm))


def _simulate_record(
    *,
    phase: str,
    lineage: SeedLineage,
    scenario: str,
    profile: EdgeProfile,
    shared: _SharedSeedEdgeInputs,
) -> tuple[dict[str, object], str, float | None, float]:
    mechanism = SCENARIO_MECHANISMS[scenario]
    expression_effect = _expression_effect(mechanism, profile, shared)
    train_response, train_regressor, train_nuisance = _paired_context_matrix(
        shared.train_reference_expression, expression_effect
    )
    test_response, test_regressor, test_nuisance = _paired_context_matrix(
        shared.test_reference_expression, expression_effect
    )
    functional = fit_incremental_downstream_functional(
        train_response,
        reference_mask=train_regressor < 0,
        nuisance_matrix=train_nuisance,
        context_regressor=train_regressor,
        receiver=profile.receiver,
        contrast_name=f"{scenario}_target_vs_reference",
        fold_id=f"{phase}_simulation_train",
        context_regressor_id="paired_binary_minus1_plus1_v1",
        nuisance_design_id="intercept_only_v1",
        feature_ids=FEATURE_IDS,
        family_ids=(profile.family_id,),
        nuisance_column_ids=NUISANCE_COLUMN_IDS,
        training_subject_ids=tuple(
            f"{lineage.seed}:{profile.known_edge_id}:train:{index:02d}"
            for index in range(N_TRAIN_SUBJECTS)
        ),
        family_basis=np.asarray([[1.0], [0.0], [0.0]], dtype=float),
        precision_weights=np.ones(len(FEATURE_IDS), dtype=float),
        minimum_scale=MINIMUM_SCALE,
        null_loss_floor=NULL_LOSS_FLOOR,
        lambda1=LAMBDA1,
        lambda2=LAMBDA2,
    )
    application = apply_incremental_downstream_functional(
        functional,
        test_response,
        nuisance_matrix=test_nuisance,
        context_regressor=test_regressor,
        feature_ids=FEATURE_IDS,
        nuisance_column_ids=NUISANCE_COLUMN_IDS,
    )
    availability, receptor, sender, ecosystem_effect = _component_values(
        mechanism, profile, shared
    )
    receiver_program = _receiver_program_effect(test_response, test_regressor)
    incremental = (
        _zero(float(application.family_gains[0]))
        if application.status == "observed"
        else None
    )
    effective_availability = (
        float(np.clip(availability * receptor, 0.0, 1.0))
        if incremental is not None
        else None
    )
    _, _, integrated = mechanistic_strength(
        availability=effective_availability,
        incremental_downstream=incremental,
        prior_quality=profile.prior_quality,
        sender_weight=sender,
        softmin_power=SOFTMIN_POWER,
        epsilon=SOFTMIN_EPSILON,
    )
    observed = application.status == "observed" and integrated is not None
    if observed:
        record: dict[str, object] = {
            "seed": lineage.seed,
            "scenario": scenario,
            "known_edge_id": profile.known_edge_id,
            "availability_effect": availability,
            "receptor_gate": receptor,
            "receiver_program_effect": receiver_program,
            "incremental_downstream_effect": incremental,
            "sender_effect": sender,
            "integrated_lr_effect": _zero(float(integrated)),
            "comparison_coverage": 1.0,
            "reference_comparison_coverage": 1.0,
            "known_edge_rank": math.nan,
            "known_edge_positive_direction": bool(float(integrated) > 0.0),
            "status": "observed",
            "reason_code": None,
        }
    else:
        record = {
            "seed": lineage.seed,
            "scenario": scenario,
            "known_edge_id": profile.known_edge_id,
            "availability_effect": math.nan,
            "receptor_gate": math.nan,
            "receiver_program_effect": math.nan,
            "incremental_downstream_effect": math.nan,
            "sender_effect": math.nan,
            "integrated_lr_effect": math.nan,
            "comparison_coverage": math.nan,
            "reference_comparison_coverage": math.nan,
            "known_edge_rank": math.nan,
            "known_edge_positive_direction": False,
            "status": application.status,
            "reason_code": application.reason_code or "integrated_strength_missing",
        }
    return (
        record,
        functional.incremental_functional_id,
        application.null_loss,
        ecosystem_effect,
    )


def _assign_ranks(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy(deep=True)
    for _, group in result.groupby(["seed", "scenario"], sort=False, observed=True):
        observed = group.loc[group["status"].eq("observed")]
        ranks = observed["integrated_lr_effect"].rank(method="first", ascending=False)
        result.loc[observed.index, "known_edge_rank"] = ranks.to_numpy(dtype=float)
    return result


def _validate_catalog(truth: ComponentTruthMatrix) -> None:
    if tuple(truth.known_edge_ids) != tuple(EDGE_PROFILES):
        raise ValueError(
            "truth known-edge order must exactly match the frozen generator catalog"
        )
    if tuple(SCENARIO_MECHANISMS) != REQUIRED_SCENARIOS:
        raise RuntimeError("generator scenarios no longer match the G1.5 contract")


def _validate_generated_table(
    table: pd.DataFrame,
    *,
    truth: ComponentTruthMatrix,
    seed_lineages: tuple[SeedLineage, ...],
) -> None:
    if tuple(table.columns) != EVIDENCE_COLUMNS:
        raise RuntimeError(
            "generated evidence columns do not match the frozen contract"
        )
    expected_rows = (
        len(seed_lineages) * len(truth.known_edge_ids) * len(REQUIRED_SCENARIOS)
    )
    if len(table) != expected_rows:
        raise RuntimeError("generated evidence is missing seed-edge-scenario records")
    if table.duplicated(["seed", "known_edge_id", "scenario"]).any():
        raise RuntimeError("generated evidence contains duplicate records")
    if set(table["seed"]) != {lineage.seed for lineage in seed_lineages}:
        raise RuntimeError("generated evidence seed universe is incomplete")
    if set(table["known_edge_id"]) != set(truth.known_edge_ids):
        raise RuntimeError("generated evidence edge universe is incomplete")
    if set(table["scenario"]) != set(REQUIRED_SCENARIOS):
        raise RuntimeError("generated evidence scenario universe is incomplete")


def generate_mechanism_specificity_evidence(
    *,
    phase: str,
    seed_count: int,
    truth: ComponentTruthMatrix,
) -> GeneratedMechanismEvidence:
    """Generate deterministic model-derived evidence for all registered edges."""

    _validate_catalog(truth)
    seed_lineages = phase_seed_lineages(phase, seed_count)
    rows: list[dict[str, object]] = []
    functional_ids: list[str] = []
    statuses: list[str] = []
    null_losses: list[float | None] = []
    ecosystem_effects: list[float] = []
    for lineage in seed_lineages:
        for edge_id in truth.known_edge_ids:
            profile = EDGE_PROFILES[edge_id]
            shared = _shared_inputs(lineage, profile)
            for scenario in REQUIRED_SCENARIOS:
                record, functional_id, null_loss, ecosystem_effect = _simulate_record(
                    phase=phase,
                    lineage=lineage,
                    scenario=scenario,
                    profile=profile,
                    shared=shared,
                )
                rows.append(record)
                functional_ids.append(functional_id)
                statuses.append(str(record["status"]))
                null_losses.append(null_loss)
                if scenario == "abundance_only":
                    ecosystem_effects.append(ecosystem_effect)
    evidence = _assign_ranks(pd.DataFrame(rows, columns=EVIDENCE_COLUMNS))
    scenario_order = {
        scenario: index for index, scenario in enumerate(REQUIRED_SCENARIOS)
    }
    edge_order = {edge: index for index, edge in enumerate(truth.known_edge_ids)}
    evidence = (
        evidence.assign(
            _scenario_order=evidence["scenario"].map(scenario_order),
            _edge_order=evidence["known_edge_id"].map(edge_order),
        )
        .sort_values(["seed", "_scenario_order", "_edge_order"], kind="stable")
        .drop(columns=["_scenario_order", "_edge_order"])
        .reset_index(drop=True)
    )
    _validate_generated_table(evidence, truth=truth, seed_lineages=seed_lineages)
    return GeneratedMechanismEvidence(
        phase=phase,
        evidence=evidence,
        seed_lineages=seed_lineages,
        functional_ids=tuple(functional_ids),
        application_statuses=tuple(statuses),
        null_losses=tuple(null_losses),
        ecosystem_effects=tuple(ecosystem_effects),
    )


def frozen_design_manifest() -> dict[str, object]:
    """Describe fixed simulation parameters without runtime-derived values."""

    return {
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "phases": {
            phase: {
                "root_seed": PHASE_ROOT_SEEDS[phase],
                "namespace": PHASE_SEED_NAMESPACES[phase],
            }
            for phase in PHASES
        },
        "training_subjects_per_record": N_TRAIN_SUBJECTS,
        "heldout_subjects_per_record": N_TEST_SUBJECTS,
        "feature_ids": list(FEATURE_IDS),
        "family_basis": [[1.0], [0.0], [0.0]],
        "nuisance_column_ids": list(NUISANCE_COLUMN_IDS),
        "context_regressor": "paired_reference_minus1_target_plus1",
        "minimum_scale": MINIMUM_SCALE,
        "null_loss_floor": NULL_LOSS_FLOOR,
        "lambda1": LAMBDA1,
        "lambda2": LAMBDA2,
        "numerical_zero_tolerance": NUMERICAL_ZERO_TOLERANCE,
        "softmin_power": SOFTMIN_POWER,
        "softmin_epsilon": SOFTMIN_EPSILON,
        "edge_profiles": {
            edge_id: {
                "receiver": profile.receiver,
                "family_id": profile.family_id,
                "expression_scale": profile.expression_scale,
                "availability_scale": profile.availability_scale,
                "receptor_level": profile.receptor_level,
                "sender_level": profile.sender_level,
                "prior_quality": profile.prior_quality,
            }
            for edge_id, profile in EDGE_PROFILES.items()
        },
        "scenario_mechanisms": {
            scenario: {
                "lr_program": mechanism.lr_program,
                "autonomous_program": mechanism.autonomous_program,
                "state_availability": mechanism.state_availability,
                "ecosystem_abundance": mechanism.ecosystem_abundance,
                "receptor_present": mechanism.receptor_present,
                "sender_present": mechanism.sender_present,
            }
            for scenario, mechanism in SCENARIO_MECHANISMS.items()
        },
        "component_derivations": {
            "availability_effect": "paired heldout state-availability mean difference",
            "receptor_gate": "heldout mean receptor gate",
            "receiver_program_effect": (
                "one_minus_exp_negative_l2_positive_paired_expression_effect"
            ),
            "incremental_downstream_effect": (
                "apply_incremental_downstream_functional.family_gains[0]"
            ),
            "sender_effect": "heldout mean sender allocation weight",
            "integrated_lr_effect": (
                "mechanistic_strength.sender_resolved with "
                "availability_effect_times_receptor_gate"
            ),
        },
        "integrated_values_are_scenario_assigned": False,
    }


__all__ = [
    "DEVELOPMENT_PHASE",
    "EDGE_PROFILES",
    "EVIDENCE_COLUMNS",
    "GENERATOR_SCHEMA_VERSION",
    "HOLDOUT_PHASE",
    "PHASE_ROOT_SEEDS",
    "PHASE_SEED_NAMESPACES",
    "GeneratedMechanismEvidence",
    "frozen_design_manifest",
    "generate_mechanism_specificity_evidence",
    "phase_seed_lineages",
]
