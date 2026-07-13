"""Run a small real-file smoke benchmark of the public partial cross-fit API."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np

from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from crychic import load_nichenet_target_prior
from crychic.core import CrychicConfig
from crychic.design import balanced_contrast
from crychic.sender import ContrastCommonSenderParameters
from crychic.workflow import (
    CrossFitArtifacts,
    CrossFitSpec,
    FoldTrainingSpec,
    run_subject_crossfit,
)

_SCHEMA_VERSION = "crychic-crossfit-smoke-v4"


def _stage_summary(artifacts: CrossFitArtifacts) -> dict[str, object]:
    design_statuses = Counter(
        application.status
        for fold in artifacts.folds
        for application in fold.design_applications
    )
    receiver_statuses = Counter(
        application.application_status
        for fold in artifacts.folds
        for application in fold.receiver_family_applications
    )
    receiver_training_reasons = Counter(
        model.reason_code or "observed"
        for fold in artifacts.folds
        for model in fold.receiver_family_models
    )
    receiver_application_reasons = Counter(
        application.reason_code or "observed"
        for fold in artifacts.folds
        for application in fold.receiver_family_applications
    )
    response_training_statuses = Counter(
        response.status
        for fold in artifacts.folds
        for response in fold.receiver_responses
    )
    response_application_statuses = Counter(
        application.status
        for fold in artifacts.folds
        for application in fold.receiver_response_applications
    )
    precision_statuses = Counter(
        "estimable" if precision.estimable else "not_estimable"
        for fold in artifacts.folds
        for precision in fold.response_precisions
    )
    diagnostic_training_statuses = Counter(
        model.diagnostic_status
        for fold in artifacts.folds
        for model in fold.receiver_incremental_models
    )
    diagnostic_application_statuses = Counter(
        application.diagnostic_status
        for fold in artifacts.folds
        for application in fold.receiver_incremental_applications
    )
    official_statuses = Counter(
        str(value)
        for value in artifacts.oof_receiver_coverage["official_incremental_status"]
    )
    observed_diagnostics = [
        application.diagnostic_application
        for fold in artifacts.folds
        for application in fold.receiver_incremental_applications
        if application.diagnostic_application is not None
        and application.diagnostic_application.status == "observed"
    ]
    bounded_gains = [
        float(application.model_gain)
        for application in observed_diagnostics
        if application.model_gain is not None
    ]
    raw_gains = [
        float(application.raw_model_gain)
        for application in observed_diagnostics
        if application.raw_model_gain is not None
    ]
    return {
        "n_design_encoders": sum(len(fold.design_encoders) for fold in artifacts.folds),
        "design_application_status_counts": dict(sorted(design_statuses.items())),
        "n_receiver_family_models": sum(
            len(fold.receiver_family_models) for fold in artifacts.folds
        ),
        "receiver_family_application_status_counts": dict(
            sorted(receiver_statuses.items())
        ),
        "receiver_family_training_reason_counts": dict(
            sorted(receiver_training_reasons.items())
        ),
        "receiver_family_application_reason_counts": dict(
            sorted(receiver_application_reasons.items())
        ),
        "receiver_families_by_fold": [
            sum(
                len(model.receiver_family_artifact.family_basis.family_ids)
                for model in fold.receiver_family_models
            )
            for fold in artifacts.folds
        ],
        "eligible_receiver_families_by_fold": [
            sum(len(model.active_family_ids) for model in fold.receiver_family_models)
            for fold in artifacts.folds
        ],
        "n_receiver_response_artifacts": sum(
            len(fold.receiver_responses) for fold in artifacts.folds
        ),
        "n_response_precision_artifacts": sum(
            len(fold.response_precisions) for fold in artifacts.folds
        ),
        "n_receiver_incremental_training_artifacts": sum(
            len(fold.receiver_incremental_models) for fold in artifacts.folds
        ),
        "n_receiver_incremental_applications": sum(
            len(fold.receiver_incremental_applications) for fold in artifacts.folds
        ),
        "n_oof_receiver_coverage_rows": len(artifacts.oof_receiver_coverage),
        "receiver_coverage_audit_id": artifacts.receiver_coverage_audit_id,
        "receiver_response_training_status_counts": dict(
            sorted(response_training_statuses.items())
        ),
        "receiver_response_application_status_counts": dict(
            sorted(response_application_statuses.items())
        ),
        "response_precision_status_counts": dict(sorted(precision_statuses.items())),
        "incremental_diagnostic_training_status_counts": dict(
            sorted(diagnostic_training_statuses.items())
        ),
        "incremental_diagnostic_application_status_counts": dict(
            sorted(diagnostic_application_statuses.items())
        ),
        "official_incremental_status_counts": dict(sorted(official_statuses.items())),
        "observed_incremental_diagnostic_count": len(observed_diagnostics),
        "mean_bounded_incremental_diagnostic_gain": (
            float(np.mean(bounded_gains)) if bounded_gains else None
        ),
        "mean_raw_incremental_diagnostic_gain": (
            float(np.mean(raw_gains)) if raw_gains else None
        ),
        "remaining_stages": artifacts.to_manifest()["remaining_stages"],
    }


def run_smoke(
    *,
    workspace_root: Path,
    scenarios: tuple[str, ...] = ("active", "ligand_only"),
    include_kang_subset: bool = False,
) -> dict[str, object]:
    """Run public partial cross-fit stages without an integrated-score claim."""

    benchmark_root = workspace_root / "benchmark_work/multicondition_v01"
    resource_root = benchmark_root / "resources/synthetic_harmonized_lr"
    bundle = harmonized_resource_bundle(
        resource_root / "harmonized_lr.tsv",
        resource_root / "manifest.json",
    )
    prior = load_nichenet_target_prior(workspace_root / "databases")
    config = CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=779404205,
    )
    spec = CrossFitSpec(
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
    )
    records: list[dict[str, object]] = []
    for scenario in scenarios:
        input_path = benchmark_root / f"synthetic_controls/synthetic_{scenario}.h5ad"
        adata = ad.read_h5ad(input_path)
        started = time.perf_counter()
        artifacts = run_subject_crossfit(
            adata,
            config,
            bundle,
            prior,
            spec=spec,
        )
        elapsed = time.perf_counter() - started
        assignments = artifacts.oof_sender_assignments
        finite_weights = assignments["assignment_weight"].dropna().to_numpy(dtype=float)
        records.append(
            {
                "dataset": f"synthetic_{scenario}",
                "scenario": scenario,
                "input_path": str(input_path.resolve()),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_subjects": len(artifacts.coverage_audit.subject_ids),
                "n_folds": len(artifacts.folds),
                "n_oof_coverage_rows": len(artifacts.oof_coverage),
                "n_oof_sender_rows": len(assignments),
                "n_finite_sender_weights": len(finite_weights),
                "mean_sender_weight": (
                    float(np.mean(finite_weights)) if len(finite_weights) else None
                ),
                "interactions_by_fold": [
                    len(fold.training.frozen_interaction_universe.interaction_ids)
                    for fold in artifacts.folds
                ],
                "elapsed_seconds": elapsed,
                "certification_status": artifacts.certification_status,
                "completed_stage_oof_verified": (
                    artifacts.completed_stage_oof_verified
                ),
                "complete_pipeline_oof_certified": artifacts.is_oof_certified,
                "crossfit_id": artifacts.crossfit_id,
                "crossfit_spec_id": spec.spec_id,
                "resource_manifest_digest": bundle.manifest_digest,
                **_stage_summary(artifacts),
            }
        )
    if include_kang_subset:
        kang = ad.read_h5ad(workspace_root / "benchmark_work/kang2018_batch2.h5ad")
        subjects = ("101", "1015", "1016", "1256")
        cell_types = ("B cells", "CD4 T cells", "CD14+ Monocytes")
        selected = kang.obs["subject_id"].astype(str).isin(subjects) & kang.obs[
            "cell_type"
        ].astype(str).isin(cell_types)
        kang = kang[selected].copy()
        hcommon_root = benchmark_root / "resources/harmonized_simple_lr"
        kang_bundle = harmonized_resource_bundle(
            hcommon_root / "harmonized_lr.tsv",
            hcommon_root / "manifest.json",
        )
        kang_spec = CrossFitSpec(
            contrasts=(
                balanced_contrast(
                    ("stim",),
                    ("ctrl",),
                    name="stim_vs_ctrl",
                ),
            ),
            training_spec=FoldTrainingSpec(
                min_cells=20,
                min_pooled_availability=0.01,
                max_interactions=None,
                sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
            ),
            allowed_n_splits=(2,),
            min_train_subjects_per_context=2,
        )
        started = time.perf_counter()
        kang_artifacts = run_subject_crossfit(
            kang,
            config,
            kang_bundle,
            prior,
            spec=kang_spec,
        )
        elapsed = time.perf_counter() - started
        assignments = kang_artifacts.oof_sender_assignments
        finite_weights = assignments["assignment_weight"].dropna().to_numpy(dtype=float)
        records.append(
            {
                "dataset": "kang2018_4donor_3celltype",
                "scenario": "real_data_subset_smoke",
                "input_path": str(
                    (workspace_root / "benchmark_work/kang2018_batch2.h5ad").resolve()
                ),
                "selected_subjects": list(subjects),
                "selected_cell_types": list(cell_types),
                "n_cells": int(kang.n_obs),
                "n_genes": int(kang.n_vars),
                "n_subjects": len(kang_artifacts.coverage_audit.subject_ids),
                "n_folds": len(kang_artifacts.folds),
                "n_oof_coverage_rows": len(kang_artifacts.oof_coverage),
                "n_oof_sender_rows": len(assignments),
                "n_finite_sender_weights": len(finite_weights),
                "mean_sender_weight": (
                    float(np.mean(finite_weights)) if len(finite_weights) else None
                ),
                "interactions_by_fold": [
                    len(fold.training.frozen_interaction_universe.interaction_ids)
                    for fold in kang_artifacts.folds
                ],
                "elapsed_seconds": elapsed,
                "certification_status": kang_artifacts.certification_status,
                "completed_stage_oof_verified": (
                    kang_artifacts.completed_stage_oof_verified
                ),
                "complete_pipeline_oof_certified": kang_artifacts.is_oof_certified,
                "crossfit_id": kang_artifacts.crossfit_id,
                "crossfit_spec_id": kang_spec.spec_id,
                "resource_manifest_digest": kang_bundle.manifest_digest,
                **_stage_summary(kang_artifacts),
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "scope": "typed_public_crossfit_algorithm_smoke_not_full_benchmark",
        "implemented_stages": [
            "availability_filter",
            "contrast_common_sender_functional",
            "frozen_design_encoder",
            "condition_blind_hard_receptor_gate",
            "strict_medoid_receiver_family_basis",
            "training_reference_receiver_program_transform",
            "fold_receiver_response",
            "response_parented_precision",
            "formula_nuisance_incremental_diagnostic",
        ],
        "exact_oof_coverage_audit_scope": [
            "availability_filter",
            "contrast_common_sender_functional",
            "receiver_incremental_application_diagnostic",
        ],
        "excluded_claims": [
            "complete_pipeline_oof_certification",
            "official_incremental_downstream_gain",
            "receiver_autonomous_nuisance_adjustment",
            "subject_blocked_inner_tuning",
            "integrated_edge_recovery",
            "method_superiority",
        ],
        "target_prior_manifest_digest": prior.manifest_digest,
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_workspace = Path(__file__).resolve().parents[3]
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_work/algorithm_smoke/crossfit_summary.json"),
    )
    parser.add_argument("--scenarios", nargs="+", default=["active", "ligand_only"])
    parser.add_argument("--include-kang-subset", action="store_true")
    args = parser.parse_args()
    payload = run_smoke(
        workspace_root=args.workspace_root.resolve(),
        scenarios=tuple(args.scenarios),
        include_kang_subset=args.include_kang_subset,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
