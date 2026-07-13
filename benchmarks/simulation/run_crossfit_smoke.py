"""Run a small real-file smoke benchmark of the public partial cross-fit API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np

from benchmarks.adapters.crychic.resource import harmonized_resource_bundle
from crychic import load_nichenet_target_prior
from crychic.core import CrychicConfig
from crychic.design import balanced_contrast
from crychic.sender import ContrastCommonSenderParameters
from crychic.workflow import (
    CrossFitSpec,
    FoldTrainingSpec,
    run_subject_crossfit,
)

_SCHEMA_VERSION = "crychic-crossfit-smoke-v1"


def run_smoke(
    *,
    workspace_root: Path,
    scenarios: tuple[str, ...] = ("active", "ligand_only"),
    include_kang_subset: bool = False,
) -> dict[str, object]:
    """Run the implemented availability/common-sender OOF stages."""

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
            }
        )
    return {
        "schema_version": _SCHEMA_VERSION,
        "scope": "algorithm_smoke_not_full_benchmark",
        "implemented_stages": [
            "availability_filter",
            "contrast_common_sender_functional",
        ],
        "excluded_claims": [
            "complete_pipeline_oof_certification",
            "integrated_edge_recovery",
            "method_superiority",
        ],
        "target_prior_manifest_digest": prior.manifest_digest,
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
