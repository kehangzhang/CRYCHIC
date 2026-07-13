"""Run a development-only smoke of tuned public family-common cross-fit stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource as process_resource
import time
from collections import Counter
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from benchmarks.simulation.run_crossfit_smoke import CROSSFIT_SMOKE_SOURCE_PATHS
from crychic import load_nichenet_target_prior
from crychic.attribution import PenaltyTuningArtifact, PenaltyTuningSpec
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

SCHEMA_VERSION = "crychic-family-common-crossfit-smoke-v1"
DEFAULT_SCENARIOS = (
    "active",
    "ligand_only",
    "receiver_autonomous",
    "global_null",
)
ALLOWED_SCENARIOS = (
    "active",
    "global_null",
    "abundance_only",
    "receiver_autonomous",
    "ligand_only",
    "target_only",
    "receptor_knockout",
)
AUTONOMOUS_REGISTRATION_ID = "crychic.synthetic_receiver_autonomous_program.v1"
ACTIVE_SOURCE_INTERACTION_ID = "CXCL10_CXCR3"
ACTIVE_LIGAND = "CXCL10"
ACTIVE_RECEPTOR = "CXCR3"
PRIMARY_RECEIVER = "Receiver"
PRIMARY_SENDER = "Sender"
RELEASED_MODES = ("state", "ecosystem")
INPUT_REGISTRY_RELATIVE_PATH = (
    "benchmarks/fixtures/family_common_crossfit_smoke_inputs_v1.json"
)
_INPUT_RECORD_FIELDS = (
    "scenario",
    "dataset_id",
    "path",
    "sha256",
    "scenario_seed",
    "n_subjects",
    "n_samples",
    "active_interaction",
    "expected_state_change",
    "expected_ecosystem_change",
    "expected_receiver_response",
    "expected_integrated_edge",
)

FAMILY_COMMON_SMOKE_SOURCE_PATHS = tuple(
    sorted(
        {
            *CROSSFIT_SMOKE_SOURCE_PATHS,
            INPUT_REGISTRY_RELATIVE_PATH,
            "benchmarks/fixtures/synthetic_receiver_autonomous_program/manifest.json",
            "benchmarks/fixtures/synthetic_receiver_autonomous_program/programs.tsv",
            "benchmarks/simulation/run_family_common_crossfit_smoke.py",
            "src/crychic/resources/autonomous_registry.py",
            "src/crychic/scoring/family_common.py",
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
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_common_smoke_source_sha256() -> dict[str, str]:
    """Hash the repository-owned code and trusted fixture closure."""

    repository_root = Path(__file__).resolve().parents[2]
    return {
        relative_path: _sha256_file(repository_root / relative_path)
        for relative_path in FAMILY_COMMON_SMOKE_SOURCE_PATHS
    }


def build_family_common_smoke_spec(
    autonomous_resource: ReceiverAutonomousProgramResource,
) -> CrossFitSpec:
    """Build the fixed small-grid paired subject-blocked smoke policy."""

    tuning_spec = PenaltyTuningSpec(
        lambda1_fractions=(1.0, 0.1),
        lambda2_fractions=(0.0,),
        inner_allowed_n_splits=(2,),
        min_inner_train_subjects_per_context=2,
        min_inner_validation_subjects_per_context=1,
        root_seed=779404205,
    )
    return CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("stim",),
                ("ctrl",),
                name="stim_vs_ctrl",
            ),
        ),
        training_spec=FoldTrainingSpec(
            min_cells=10,
            min_pooled_availability=0.0,
            max_interactions=None,
            sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
        ),
        allowed_n_splits=(2,),
        min_train_subjects_per_context=2,
        min_test_subjects_per_context=1,
        autonomous_program_resource=autonomous_resource,
        penalty_tuning_spec=tuning_spec,
    )


def _count_strings(values: Iterable[object]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def _count_reasons(values: Iterable[object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        missing = value is None or value is pd.NA or value is pd.NaT
        if isinstance(value, (float, np.floating)):
            missing = missing or bool(np.isnan(value))
        counts["none" if missing else str(value)] += 1
    return dict(sorted(counts.items()))


def _table_status_counts(table: pd.DataFrame, column: str) -> dict[str, int]:
    if table.empty:
        return {}
    return _count_strings(table[column])


def _table_reason_counts(table: pd.DataFrame) -> dict[str, int]:
    if table.empty:
        return {}
    return _count_reasons(table["reason_code"])


def _finite_score_summary(values: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        return {
            "n_rows": len(values),
            "n_finite": 0,
            "mean": None,
            "sum": None,
            "minimum": None,
            "maximum": None,
            "all_finite_scores_exactly_zero": None,
        }
    return {
        "n_rows": len(values),
        "n_finite": len(finite),
        "mean": float(np.mean(finite)),
        "sum": float(np.sum(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "all_finite_scores_exactly_zero": bool(np.all(finite == 0.0)),
    }


def _rank_interaction_table(
    table: pd.DataFrame,
    *,
    score_column: str,
    active_interaction_id: str,
    receiver: str,
    sender: str | None = None,
) -> dict[str, object]:
    """Rank mean finite scores with dense ties while retaining exact zeros."""

    scope = table.loc[table["receiver"].astype(str) == receiver].copy()
    if sender is not None:
        scope = scope.loc[scope["sender"].astype(str) == sender].copy()
    mode_records: list[dict[str, object]] = []
    for mode in RELEASED_MODES:
        mode_table = scope.loc[scope["mode"].astype(str) == mode]
        rows: list[dict[str, object]] = []
        for interaction_id, group in mode_table.groupby(
            "interaction_id", observed=True, sort=True
        ):
            summary = _finite_score_summary(group[score_column])
            rows.append(
                {
                    "interaction_id": str(interaction_id),
                    **summary,
                    "status_counts": _table_status_counts(group, "status"),
                    "reason_counts": _table_reason_counts(group),
                }
            )
        finite_mean_values = sorted(
            {cast(float, row["mean"]) for row in rows if row["mean"] is not None},
            reverse=True,
        )
        rank_by_mean = {
            value: index + 1 for index, value in enumerate(finite_mean_values)
        }
        for row in rows:
            mean = row["mean"]
            row["dense_rank"] = (
                None if mean is None else rank_by_mean[cast(float, mean)]
            )
            row["n_interactions_tied_at_rank"] = (
                None if mean is None else sum(other["mean"] == mean for other in rows)
            )
        leaderboard = sorted(
            rows,
            key=lambda row: (
                row["mean"] is None,
                -cast(float, row["mean"] or 0.0),
                str(row["interaction_id"]),
            ),
        )
        active: dict[str, object] = next(
            (
                dict(row)
                for row in leaderboard
                if row["interaction_id"] == active_interaction_id
            ),
            {
                "interaction_id": active_interaction_id,
                "n_rows": 0,
                "n_finite": 0,
                "mean": None,
                "sum": None,
                "minimum": None,
                "maximum": None,
                "all_finite_scores_exactly_zero": None,
                "status_counts": {},
                "reason_counts": {},
                "dense_rank": None,
                "n_interactions_tied_at_rank": None,
            },
        )
        active["present"] = any(
            row["interaction_id"] == active_interaction_id for row in leaderboard
        )
        mode_records.append(
            {
                "mode": mode,
                "n_candidate_interactions": len(leaderboard),
                "n_interactions_with_finite_scores": sum(
                    row["mean"] is not None for row in rows
                ),
                "n_distinct_finite_mean_values": len(finite_mean_values),
                "all_finite_interaction_means_exactly_zero": (
                    None
                    if not finite_mean_values
                    else bool(all(value == 0.0 for value in finite_mean_values))
                ),
                "active_interaction": active,
                "leaderboard": leaderboard,
            }
        )
    return {
        "score_column": score_column,
        "aggregation": "mean_over_finite_heldout_rows",
        "rank_method": "descending_dense_rank_with_exact_score_ties",
        "receiver_filter": receiver,
        "sender_filter": sender,
        "modes": mode_records,
    }


def _paired_input_audit(adata: ad.AnnData) -> dict[str, object]:
    subject_contexts = (
        adata.obs.loc[:, ["subject_id", "condition"]]
        .astype(str)
        .drop_duplicates()
        .groupby("subject_id", observed=True)["condition"]
        .agg(lambda values: tuple(sorted(set(values))))
    )
    expected = {"ctrl", "stim"}
    incomplete = {
        str(subject): sorted(expected.difference(contexts))
        for subject, contexts in subject_contexts.items()
        if set(contexts) != expected
    }
    return {
        "statistical_unit": "subject_id",
        "expected_contexts": sorted(expected),
        "n_subjects": len(subject_contexts),
        "all_subjects_complete_paired": not incomplete,
        "incomplete_subject_missing_contexts": incomplete,
        "cell_level_values_are_not_independent_replicates": True,
    }


def _tuning_candidate_audit(
    tuning: PenaltyTuningArtifact,
) -> list[dict[str, object]]:
    """Expand the paired selection evidence for a small diagnostic run."""

    summary_by_id = {summary.candidate_id: summary for summary in tuning.summaries}
    comparison_by_id = {
        comparison.candidate_id: comparison
        for comparison in tuning.candidate_comparisons
    }
    records: list[dict[str, object]] = []
    for candidate in tuning.spec.candidates:
        summary = summary_by_id.get(candidate.candidate_id)
        comparison = comparison_by_id.get(candidate.candidate_id)
        evaluations = sorted(
            (
                evaluation
                for evaluation in tuning.evaluations
                if evaluation.candidate_id == candidate.candidate_id
            ),
            key=lambda evaluation: evaluation.inner_fold_id,
        )
        records.append(
            {
                "candidate_id": candidate.candidate_id,
                "lambda1_fraction": candidate.lambda1_fraction,
                "lambda2_fraction": candidate.lambda2_fraction,
                "is_empirical_best": (
                    candidate.candidate_id == tuning.best_candidate_id
                ),
                "is_selected": (candidate.candidate_id == tuning.selected_candidate_id),
                "summary_status": None if summary is None else summary.status,
                "summary_reason_code": (
                    None if summary is None else summary.reason_code
                ),
                "subject_ids": ([] if summary is None else list(summary.subject_ids)),
                "subject_losses": (
                    [] if summary is None else summary.subject_losses.tolist()
                ),
                "mean_loss": None if summary is None else summary.mean_loss,
                "raw_loss_standard_error": (
                    None if summary is None else summary.standard_error
                ),
                "mean_loss_difference_to_best": (
                    None if comparison is None else comparison.mean_loss_difference
                ),
                "paired_delta_standard_error": (
                    None if comparison is None else comparison.standard_error
                ),
                "paired_delta_one_se_threshold": (
                    None if comparison is None else comparison.one_se_threshold
                ),
                "within_paired_delta_one_se": (
                    None if comparison is None else comparison.within_one_se
                ),
                "inner_fold_evaluations": [
                    {
                        "evaluation_id": evaluation.evaluation_id,
                        "inner_fold_id": evaluation.inner_fold_id,
                        "inner_training_subject_ids": list(
                            evaluation.inner_training_subject_ids
                        ),
                        "validation_subject_ids": list(
                            evaluation.validation_subject_ids
                        ),
                        "subject_losses": evaluation.subject_losses.tolist(),
                        "status": evaluation.status,
                        "reason_code": evaluation.reason_code,
                        "verification_status": evaluation.verification_status,
                        "scale_resolution_id": evaluation.scale_resolution_id,
                        "resolved_penalty_id": evaluation.resolved_penalty_id,
                        "resolved_lambda1": evaluation.resolved_lambda1,
                        "resolved_lambda2": evaluation.resolved_lambda2,
                    }
                    for evaluation in evaluations
                ],
            }
        )
    return records


def _selected_penalty_summary(artifacts: CrossFitArtifacts) -> dict[str, object]:
    records: list[dict[str, object]] = []
    selected_counts: Counter[tuple[float, float]] = Counter()
    n_without_selection = 0
    for fold in artifacts.folds:
        for model in fold.receiver_incremental_models:
            tuning = model.penalty_tuning_artifact
            candidate = None if tuning is None else tuning.selected_candidate
            if candidate is None:
                n_without_selection += 1
            else:
                selected_counts[
                    (candidate.lambda1_fraction, candidate.lambda2_fraction)
                ] += 1
            diagnostic = model.diagnostic_functional
            nonzero_family_indices = (
                np.empty(0, dtype=int)
                if diagnostic is None
                else np.flatnonzero(diagnostic.family_coefficients > 0.0)
            )
            records.append(
                {
                    "fold_id": fold.fold_id,
                    "receiver": model.receiver,
                    "contrast_name": model.contrast_name,
                    "tuning_id": None if tuning is None else tuning.tuning_id,
                    "tuning_status": None if tuning is None else tuning.status,
                    "tuning_reason_code": (
                        None if tuning is None else tuning.reason_code
                    ),
                    "selection_rule": (
                        None if tuning is None else tuning.spec.selection_rule
                    ),
                    "best_candidate_id": (
                        None if tuning is None else tuning.best_candidate_id
                    ),
                    "best_mean_loss": (
                        None if tuning is None else tuning.best_mean_loss
                    ),
                    "candidate_diagnostics": (
                        [] if tuning is None else _tuning_candidate_audit(tuning)
                    ),
                    "selected_candidate_id": model.selected_penalty_candidate_id,
                    "lambda1_fraction": (
                        None if candidate is None else candidate.lambda1_fraction
                    ),
                    "lambda2_fraction": (
                        None if candidate is None else candidate.lambda2_fraction
                    ),
                    "selected_scale_resolution_id": (
                        model.selected_penalty_scale_resolution_id
                    ),
                    "selected_resolved_penalty_id": (
                        model.selected_resolved_penalty_id
                    ),
                    "resolved_lambda1": (
                        None if diagnostic is None else diagnostic.lambda1
                    ),
                    "resolved_lambda2": (
                        None if diagnostic is None else diagnostic.lambda2
                    ),
                    "final_n_nonzero_families": len(nonzero_family_indices),
                    "final_nonzero_families": [
                        {
                            "family_id": diagnostic.family_ids[index],
                            "coefficient": float(diagnostic.family_coefficients[index]),
                        }
                        for index in nonzero_family_indices
                    ]
                    if diagnostic is not None
                    else [],
                }
            )
    return {
        "selected_fraction_counts": [
            {
                "lambda1_fraction": lambda1,
                "lambda2_fraction": lambda2,
                "count": count,
            }
            for (lambda1, lambda2), count in sorted(selected_counts.items())
        ],
        "n_models_without_selected_candidate": n_without_selection,
        "records": records,
    }


def _algorithm_audit(
    artifacts: CrossFitArtifacts,
    *,
    active_interaction_id: str,
) -> dict[str, object]:
    models = [
        model for fold in artifacts.folds for model in fold.receiver_incremental_models
    ]
    incremental_applications = [
        application
        for fold in artifacts.folds
        for application in fold.receiver_incremental_applications
    ]
    tunings = [
        model.penalty_tuning_artifact
        for model in models
        if model.penalty_tuning_artifact is not None
    ]
    evaluations = [
        evaluation for tuning in tunings for evaluation in tuning.evaluations
    ]
    functionals = [
        functional
        for fold in artifacts.folds
        for functional in fold.family_common_functionals
    ]
    common_applications = [
        application
        for fold in artifacts.folds
        for application in fold.family_common_applications
    ]
    family_attribution = pd.concat(
        [application.family_attribution for application in common_applications],
        ignore_index=True,
    )
    subject_differential = pd.concat(
        [application.subject_differential for application in common_applications],
        ignore_index=True,
    )
    family_scores = pd.concat(
        [application.family_scores for application in common_applications],
        ignore_index=True,
    )
    member_scores = pd.concat(
        [application.member_scores for application in common_applications],
        ignore_index=True,
    )
    sender_scores = pd.concat(
        [application.sender_scores for application in common_applications],
        ignore_index=True,
    )
    manifest = artifacts.to_manifest()
    return {
        "crossfit_id": artifacts.crossfit_id,
        "crossfit_spec_id": artifacts.spec.spec_id,
        "penalty_tuning_spec_id": cast(
            PenaltyTuningSpec, artifacts.spec.penalty_tuning_spec
        ).spec_id,
        "n_folds": len(artifacts.folds),
        "n_incremental_models": len(models),
        "n_family_common_functionals": len(functionals),
        "n_family_common_applications": len(common_applications),
        "tuning_status_counts": _count_strings(tuning.status for tuning in tunings),
        "tuning_reason_counts": _count_reasons(
            tuning.reason_code for tuning in tunings
        ),
        "tuning_certification_status_counts": _count_strings(
            tuning.certification_status for tuning in tunings
        ),
        "tuning_evaluation_status_counts": _count_strings(
            evaluation.status for evaluation in evaluations
        ),
        "tuning_evaluation_verification_status_counts": _count_strings(
            evaluation.verification_status for evaluation in evaluations
        ),
        "incremental_training_diagnostic_status_counts": _count_strings(
            model.diagnostic_status for model in models
        ),
        "incremental_training_official_status_counts": _count_strings(
            model.official_incremental_status for model in models
        ),
        "incremental_application_diagnostic_status_counts": _count_strings(
            application.diagnostic_status for application in incremental_applications
        ),
        "incremental_application_official_status_counts": _count_strings(
            application.official_incremental_status
            for application in incremental_applications
        ),
        "incremental_application_official_reason_counts": _count_reasons(
            application.reason_code for application in incremental_applications
        ),
        "oof_receiver_official_status_counts": _count_strings(
            artifacts.oof_receiver_coverage["official_incremental_status"]
        ),
        "family_common_functional_status_counts": _count_strings(
            "observed"
            if functional.incremental_functional is not None
            else "not_estimable"
            for functional in functionals
        ),
        "family_common_functional_reason_counts": _count_reasons(
            functional.incremental_reason_code for functional in functionals
        ),
        "family_common_application_status_counts": _count_strings(
            "observed" if application.heldout_reason_code is None else "not_estimable"
            for application in common_applications
        ),
        "family_common_application_reason_counts": _count_reasons(
            application.heldout_reason_code for application in common_applications
        ),
        "family_common_application_oof_certified_counts": _count_strings(
            application.is_oof_certified for application in common_applications
        ),
        "selected_penalties": _selected_penalty_summary(artifacts),
        "score_tables": {
            "family_attribution": {
                "n_rows": len(family_attribution),
                "selection_status_counts": _table_status_counts(
                    family_attribution, "selection_status"
                ),
                "reason_counts": _table_reason_counts(family_attribution),
            },
            "subject_differential": {
                "n_rows": len(subject_differential),
                "status_counts": _table_status_counts(subject_differential, "status"),
                "reason_counts": _table_reason_counts(subject_differential),
            },
            "family": {
                "n_rows": len(family_scores),
                "status_counts": _table_status_counts(family_scores, "status"),
                "reason_counts": _table_reason_counts(family_scores),
            },
            "member": {
                "n_rows": len(member_scores),
                "status_counts": _table_status_counts(member_scores, "status"),
                "reason_counts": _table_reason_counts(member_scores),
            },
            "sender": {
                "n_rows": len(sender_scores),
                "status_counts": _table_status_counts(sender_scores, "status"),
                "reason_counts": _table_reason_counts(sender_scores),
            },
        },
        "active_interaction_tracking": {
            "harmonized_interaction_id": active_interaction_id,
            "member_unresolved": _rank_interaction_table(
                member_scores,
                score_column="sender_unresolved_strength",
                active_interaction_id=active_interaction_id,
                receiver=PRIMARY_RECEIVER,
            ),
            "sender_resolved": _rank_interaction_table(
                sender_scores,
                score_column="sender_resolved_strength",
                active_interaction_id=active_interaction_id,
                receiver=PRIMARY_RECEIVER,
                sender=PRIMARY_SENDER,
            ),
        },
        "completed_stage_oof_verified": artifacts.completed_stage_oof_verified,
        "complete_pipeline_oof_certified": artifacts.is_oof_certified,
        "remaining_stages": manifest["remaining_stages"],
    }


def _active_interaction_mapping(
    table_path: Path,
    bundle: ResourceBundle,
) -> dict[str, object]:
    table = pd.read_csv(table_path, sep="\t", dtype=str)
    selected = table.loc[
        (table["ligand"] == ACTIVE_LIGAND) & (table["receptor"] == ACTIVE_RECEPTOR)
    ]
    if len(selected) != 1:
        raise ValueError("synthetic active LR must map to exactly one harmonized row")
    row = selected.iloc[0]
    interaction_id = str(row["harmonized_interaction_id"])
    interaction = next(
        item for item in bundle.interactions if item.interaction_id == interaction_id
    )
    if str(row["cellchat_source_interaction_id"]) != ACTIVE_SOURCE_INTERACTION_ID:
        raise ValueError("synthetic active source interaction mapping changed")
    return {
        "simulation_source_interaction_id": ACTIVE_SOURCE_INTERACTION_ID,
        "simulation_source_namespace": "synthetic_cellchat_style_lr_id",
        "ligand": ACTIVE_LIGAND,
        "receptor": ACTIVE_RECEPTOR,
        "harmonized_interaction_id": interaction_id,
        "cellchat_source_interaction_id": str(row["cellchat_source_interaction_id"]),
        "cellphonedb_source_interaction_id": str(
            row["cellphonedb_source_interaction_id"]
        ),
        "crychic_bundle_source_interaction_id": interaction.source_interaction_id,
        "crychic_bundle_evidence": list(interaction.evidence),
    }


def _synthetic_manifest_records(
    benchmark_root: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, object]]:
    manifest_path = benchmark_root / "synthetic_controls/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "crychic-multimethod-synthetic-v1":
        raise ValueError("unsupported synthetic control manifest")
    records = cast(list[dict[str, Any]], manifest.get("records"))
    registry_path = repository_root / INPUT_REGISTRY_RELATIVE_PATH
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != (
        "crychic-family-common-smoke-input-registry-v1"
    ):
        raise ValueError("unsupported family-common synthetic input registry")
    source_manifest = cast(dict[str, Any], registry.get("source_manifest"))
    if (
        source_manifest.get("schema_version") != manifest.get("schema_version")
        or source_manifest.get("sha256") != _sha256_file(manifest_path)
        or source_manifest.get("size_bytes") != manifest_path.stat().st_size
    ):
        raise ValueError("synthetic control manifest does not match input registry")
    registry_records = cast(list[dict[str, Any]], registry.get("records"))
    observed_records = [
        {field: record.get(field) for field in _INPUT_RECORD_FIELDS}
        for record in records
    ]
    if observed_records != registry_records:
        raise ValueError("synthetic control records do not match input registry")
    by_scenario = {str(record["scenario"]): record for record in records}
    if len(by_scenario) != len(records) or set(by_scenario) != set(ALLOWED_SCENARIOS):
        raise ValueError("synthetic control manifest scenarios are not unique")
    return by_scenario, {
        "registry_relative_path": INPUT_REGISTRY_RELATIVE_PATH,
        "registry_sha256": _sha256_file(registry_path),
        "source_manifest": source_manifest,
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
            "table_path": str(lr_table_path.resolve()),
            "table_sha256": _sha256_file(lr_table_path),
            "manifest_path": str(lr_manifest_path.resolve()),
            "manifest_sha256": _sha256_file(lr_manifest_path),
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


def run_smoke(
    *,
    workspace_root: Path,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
) -> dict[str, object]:
    """Run the trusted-resource tuned path on small synthetic controls."""

    unknown = sorted(set(scenarios).difference(ALLOWED_SCENARIOS))
    if unknown:
        raise ValueError(f"unsupported synthetic scenarios: {unknown}")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be a non-empty unique sequence")

    repository_root = Path(__file__).resolve().parents[2]
    fixture_root = (
        repository_root / "benchmarks/fixtures/synthetic_receiver_autonomous_program"
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
    spec = build_family_common_smoke_spec(autonomous)
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=779404205,
    )
    active_mapping = _active_interaction_mapping(lr_table_path, bundle)
    active_harmonized_id = cast(str, active_mapping["harmonized_interaction_id"])
    truth_by_scenario, input_registry = _synthetic_manifest_records(
        benchmark_root,
        repository_root=repository_root,
    )

    records: list[dict[str, object]] = []
    for scenario in scenarios:
        truth = truth_by_scenario.get(scenario)
        if truth is None:
            raise ValueError(f"scenario is absent from synthetic manifest: {scenario}")
        input_path = benchmark_root / "synthetic_controls" / str(truth["path"])
        observed_sha256 = _sha256_file(input_path)
        if observed_sha256 != str(truth["sha256"]):
            raise ValueError(f"synthetic input checksum mismatch: {scenario}")
        adata = ad.read_h5ad(input_path)
        paired_audit = _paired_input_audit(adata)
        if not cast(bool, paired_audit["all_subjects_complete_paired"]):
            raise ValueError(f"synthetic input is not completely paired: {scenario}")
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
                "dataset": str(truth["dataset_id"]),
                "scenario": scenario,
                "scenario_seed": int(truth["scenario_seed"]),
                "input_path": str(input_path.resolve()),
                "input_sha256": observed_sha256,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_subjects": int(truth["n_subjects"]),
                "n_samples": int(truth["n_samples"]),
                "paired_input_audit": paired_audit,
                "simulation_truth": {
                    "active_interaction": truth["active_interaction"],
                    "expected_state_change": bool(truth["expected_state_change"]),
                    "expected_ecosystem_change": bool(
                        truth["expected_ecosystem_change"]
                    ),
                    "expected_receiver_response": bool(
                        truth["expected_receiver_response"]
                    ),
                    "expected_integrated_edge": bool(truth["expected_integrated_edge"]),
                    "tracked_active_source_interaction_is_positive_truth": (
                        truth["active_interaction"] == ACTIVE_SOURCE_INTERACTION_ID
                    ),
                },
                "elapsed_seconds": elapsed,
                "process_peak_rss_kib_after_scenario": int(
                    process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
                ),
                **_algorithm_audit(
                    artifacts,
                    active_interaction_id=active_harmonized_id,
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "development_only_public_family_common_crossfit_smoke",
        "claims": dict(AUDIT_CLAIMS),
        "certification_boundary": (
            "outer-heldout family-common diagnostics remain explicitly noncertifying; "
            "receiver-program scoring is connected as a diagnostic-only component"
        ),
        "statistical_unit": "subject_id",
        "single_seed_development_diagnostic": True,
        "synthetic_input_registry": input_registry,
        "source_sha256": family_common_smoke_source_sha256(),
        "crossfit_spec": spec.to_dict(),
        "crossfit_spec_id": spec.spec_id,
        "penalty_tuning_spec_id": cast(
            PenaltyTuningSpec, spec.penalty_tuning_spec
        ).spec_id,
        "resource_provenance": _resource_provenance(
            autonomous=autonomous,
            bundle=bundle,
            target_prior=target_prior,
            lr_table_path=lr_table_path,
            lr_manifest_path=lr_manifest_path,
        ),
        "active_interaction_source_mapping": active_mapping,
        "ranking_interpretation": (
            "Descriptive held-out diagnostic ranking only; zero and not-estimable "
            "scores are retained and no biological or superiority claim is made."
        ),
        "process_peak_rss_kib": int(
            process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
        ),
        "records": records,
    }


def _compact_active_modes(
    tracking: dict[str, object], *, score_kind: str
) -> list[dict[str, object]]:
    score_table = cast(dict[str, object], tracking[score_kind])
    modes = cast(list[dict[str, object]], score_table["modes"])
    compact: list[dict[str, object]] = []
    for mode in modes:
        active = cast(dict[str, object], mode["active_interaction"])
        compact.append(
            {
                "mode": mode["mode"],
                "mean": active["mean"],
                "dense_rank": active["dense_rank"],
                "n_interactions_tied_at_rank": active["n_interactions_tied_at_rank"],
                "status_counts": active["status_counts"],
                "reason_counts": active["reason_counts"],
            }
        )
    return compact


def compact_family_common_record(record: dict[str, object]) -> dict[str, object]:
    """Reduce one complete scenario audit to its tracked diagnostic surface."""

    penalties = cast(dict[str, object], record["selected_penalties"])
    penalty_records = cast(list[dict[str, object]], penalties["records"])
    tracking = cast(dict[str, object], record["active_interaction_tracking"])
    score_tables = cast(dict[str, object], record["score_tables"])
    family_scores = cast(dict[str, object], score_tables["family"])
    return {
        "scenario": record["scenario"],
        "dataset": record["dataset"],
        "scenario_seed": record["scenario_seed"],
        "input_sha256": record["input_sha256"],
        "simulation_truth": record["simulation_truth"],
        "n_subjects": record["n_subjects"],
        "n_samples": record["n_samples"],
        "elapsed_seconds": record["elapsed_seconds"],
        "tuning_status_counts": record["tuning_status_counts"],
        "incremental_training_official_status_counts": record[
            "incremental_training_official_status_counts"
        ],
        "family_common_application_status_counts": record[
            "family_common_application_status_counts"
        ],
        "selected_penalty_fractions": penalties["selected_fraction_counts"],
        "receiver_final_nonzero_family_counts": [
            item["final_n_nonzero_families"]
            for item in penalty_records
            if item["receiver"] == PRIMARY_RECEIVER
        ],
        "family_score_status_counts": family_scores["status_counts"],
        "active_interaction_member_modes": _compact_active_modes(
            tracking, score_kind="member_unresolved"
        ),
        "active_interaction_sender_modes": _compact_active_modes(
            tracking, score_kind="sender_resolved"
        ),
    }


def build_artifact_metadata(
    payload: dict[str, object],
    *,
    serialized: bytes,
    relative_workspace_path: str,
) -> dict[str, object]:
    """Build deterministic full-artifact metadata for a compact summary."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_sha256": hashlib.sha256(serialized).hexdigest(),
        "relative_workspace_path": relative_workspace_path,
        "size_bytes": len(serialized),
        "tracked": False,
    }


def _mode_check(record: dict[str, object], *, positive: bool) -> bool:
    for key in ("active_interaction_member_modes", "active_interaction_sender_modes"):
        modes = cast(list[dict[str, object]], record[key])
        for mode in modes:
            mean = float(cast(float, mode["mean"]))
            if positive:
                if not (
                    mean > 0.0
                    and mode["dense_rank"] == 1
                    and mode["n_interactions_tied_at_rank"] == 1
                ):
                    return False
            elif mean != 0.0:
                return False
    return True


def build_compact_summary(
    payload: dict[str, object],
    artifact_metadata: dict[str, object],
    *,
    base_revision: str,
    generated_on: str,
) -> dict[str, object]:
    """Derive the tracked compact summary from one complete smoke payload."""

    records = [
        compact_family_common_record(record)
        for record in cast(list[dict[str, object]], payload["records"])
    ]
    by_scenario = {str(record["scenario"]): record for record in records}
    active = by_scenario["active"]
    controls = tuple(by_scenario[name] for name in DEFAULT_SCENARIOS[1:])
    strongest = [{"count": 2, "lambda1_fraction": 1.0, "lambda2_fraction": 0.0}]
    weaker = [{"count": 2, "lambda1_fraction": 0.1, "lambda2_fraction": 0.0}]
    checks = {
        "active_has_nonzero_families_in_both_folds": all(
            int(value) > 0
            for value in cast(list[int], active["receiver_final_nonzero_family_counts"])
        ),
        "active_interaction_unique_top_member_in_both_modes": _mode_check(
            active, positive=True
        ),
        "active_interaction_unique_top_sender_in_both_modes": _mode_check(
            active, positive=True
        ),
        "active_selects_weaker_penalty_in_both_folds": (
            active["selected_penalty_fractions"] == weaker
        ),
        "all_controls_select_strongest_and_stay_zero": all(
            control["selected_penalty_fractions"] == strongest
            and cast(list[int], control["receiver_final_nonzero_family_counts"])
            == [0, 0]
            for control in controls
        ),
        "all_control_family_scores_structural_zero": all(
            control["family_score_status_counts"] == {"structural_zero": 480}
            for control in controls
        ),
        "tracked_active_interaction_zero_in_all_controls": all(
            _mode_check(control, positive=False) for control in controls
        ),
    }
    crossfit_spec = cast(dict[str, object], payload["crossfit_spec"])
    tuning_spec = cast(dict[str, object], crossfit_spec["penalty_tuning_spec"])
    resource_provenance = cast(dict[str, object], payload["resource_provenance"])
    return {
        "active_interaction_source_mapping": payload[
            "active_interaction_source_mapping"
        ],
        "base_revision": base_revision,
        "candidate_priority": tuning_spec["candidate_priority"],
        "certification_boundary": payload["certification_boundary"],
        "checks": checks,
        "claims": payload["claims"],
        "full_artifact": dict(artifact_metadata),
        "generated_on": generated_on,
        "interpretation": (
            "Single-seed paired synthetic development smoke after the v6 frozen "
            "response-coordinate correction. The known CXCL10-CXCR3 interaction "
            "is uniquely rank 1 in both released modes, while ligand-only, "
            "receiver-autonomous, and global-null controls select the strongest "
            "penalty and retain no nonzero family. This is synthetic algorithm "
            "evidence only and does not certify biological discovery or method "
            "superiority."
        ),
        "limitations": {
            "control_family_score_ok_rows": {
                str(control["scenario"]): int(
                    cast(dict[str, int], control["family_score_status_counts"]).get(
                        "ok", 0
                    )
                )
                for control in controls
            },
            "control_final_nonzero_family_counts": {
                str(control["scenario"]): control[
                    "receiver_final_nonzero_family_counts"
                ]
                for control in controls
            },
            "single_seed_two_candidate_grid": True,
        },
        "records": records,
        "resource_boundary": cast(
            dict[str, object], resource_provenance["receiver_autonomous_program"]
        ),
        "schema_version": "crychic-family-common-crossfit-smoke-summary-v1",
        "scope": payload["scope"],
        "selection_rule": tuning_spec["selection_rule"],
        "source_sha256": payload["source_sha256"],
        "synthetic_input_registry": payload["synthetic_input_registry"],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without running the benchmark."""

    default_workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=ALLOWED_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_work/algorithm_smoke/family_common_crossfit_smoke_v1.json"
        ),
    )
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--base-revision")
    parser.add_argument("--generated-on", default=date.today().isoformat())
    parser.add_argument(
        "--artifact-relative-workspace-path",
        default="benchmark_work/algorithm_smoke/family_common_crossfit_smoke_v1.json",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summary_output is not None:
        if not args.base_revision:
            parser.error("--summary-output requires --base-revision")
        if tuple(args.scenarios) != DEFAULT_SCENARIOS:
            parser.error("--summary-output requires the canonical default scenarios")
    workspace_root = args.workspace_root.expanduser().resolve()
    payload = run_smoke(
        workspace_root=workspace_root,
        scenarios=tuple(args.scenarios),
    )
    output = args.output.expanduser()
    if not output.is_absolute():
        output = workspace_root / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    output.write_text(serialized, encoding="utf-8")
    if args.summary_output is not None:
        summary = build_compact_summary(
            payload,
            build_artifact_metadata(
                payload,
                serialized=serialized.encode("utf-8"),
                relative_workspace_path=args.artifact_relative_workspace_path,
            ),
            base_revision=args.base_revision,
            generated_on=args.generated_on,
        )
        summary_output = args.summary_output.expanduser()
        if not summary_output.is_absolute():
            summary_output = Path(__file__).resolve().parents[2] / summary_output
        summary_output = summary_output.resolve()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(serialized, end="")


if __name__ == "__main__":
    main()
