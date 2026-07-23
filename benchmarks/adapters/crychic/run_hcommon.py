"""Run a configured CRYCHIC dataset with the frozen H-common LR resource."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import anndata as ad

import crychic
from benchmarks.adapters.common import (
    begin_manifest,
    fail_manifest,
    finalize_manifest,
    git_metadata,
    prepare_output,
    python_environment,
    sha256_file,
    validate_prepared_input,
)
from benchmarks.run_canonical_v01 import load_benchmark_config, resolve_path
from crychic import Crychic, CrychicConfig, CrychicResult
from crychic.resources import load_nichenet_target_prior
from crychic.scoring import DownstreamEvidencePolicy

from .readback import (
    MECHANISTIC_SCORE_NAME,
    METHOD_ID,
    SCORE_DIRECTION,
    SCORE_NAME,
    convert_result_to_long,
)
from .resource import harmonized_resource_bundle

REPO_ROOT = Path(__file__).resolve().parents[3]


def _package_version() -> str:
    try:
        return importlib.metadata.version("CRYCHIC")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _analysis_digest(input_sha256: str, observation_ids: Sequence[object]) -> str:
    payload = json.dumps(
        {
            "input_sha256": input_sha256,
            "observation_ids": list(map(str, observation_ids)),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _target_prior(
    benchmark: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    database_root: Path,
    repo_root: Path,
) -> Any:
    target_name = spec.get("target_prior")
    if not isinstance(target_name, str) or not target_name:
        raise ValueError(
            "H-common comm_strength requires a configured target_prior resource"
        )
    resources = benchmark.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("benchmark config has no resources mapping")
    prior_spec = resources.get(target_name)
    if not isinstance(prior_spec, Mapping):
        raise ValueError(f"unknown target prior resource: {target_name!r}")
    return load_nichenet_target_prior(
        database_root,
        release=str(prior_spec["release"]),
        manifest_path=resolve_path(str(prior_spec["manifest"]), repo_root=repo_root),
    )


def _dataset_spec(benchmark: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    datasets = benchmark.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("benchmark config has no datasets mapping")
    spec = datasets.get(dataset)
    if not isinstance(spec, Mapping):
        raise ValueError(f"unknown benchmark dataset: {dataset!r}")
    return cast(Mapping[str, Any], spec)


def validate_hcommon_workflow(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Reject data-driven interaction caps in the H-common comparison arm."""

    raw = spec.get("workflow")
    if not isinstance(raw, Mapping):
        raise ValueError("H-common dataset spec has no workflow mapping")
    workflow = dict(cast(Mapping[str, Any], raw))
    if workflow.get("max_interactions") is not None:
        raise ValueError(
            "H-common workflow requires max_interactions=None; a data-driven "
            "top-k interaction cap changes the comparison universe"
        )
    return workflow


def run_hcommon_from_benchmark(
    benchmark_config: str | Path,
    dataset: str,
    output_dir: str | Path,
    *,
    harmonized_resource: str | Path,
    harmonized_manifest: str | Path,
    database_root: str | Path | None = None,
    communication_mode: str = "state",
    scoring_functional_ids: Sequence[str] | None = None,
    downstream_evidence_policy: DownstreamEvidencePolicy | str = (
        DownstreamEvidencePolicy.ANNOTATE
    ),
    blas_threads: int = 8,
    overwrite: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run one existing benchmark dataset in a disjoint H-common arm."""

    if communication_mode not in {"state", "ecosystem"}:
        raise ValueError("communication_mode must be 'state' or 'ecosystem'")
    score_policy = DownstreamEvidencePolicy(downstream_evidence_policy)
    if score_policy is DownstreamEvidencePolicy.MODULATE:
        raise ValueError(
            "run_hcommon cannot modulate without signed held-out downstream support"
        )
    root = Path(repo_root).resolve()
    config_path = Path(benchmark_config).resolve()
    benchmark = load_benchmark_config(config_path)
    spec = _dataset_spec(benchmark, dataset)
    workflow = validate_hcommon_workflow(spec)
    input_path = resolve_path(str(spec["input"]), repo_root=root)
    expected_sha256 = str(spec["input_sha256"])
    observed_sha256 = sha256_file(input_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"input checksum mismatch: expected={expected_sha256}, "
            f"observed={observed_sha256}"
        )
    configured_database = resolve_path(str(benchmark["database_root"]), repo_root=root)
    selected_database = (
        configured_database
        if database_root is None
        else Path(database_root).expanduser().resolve()
    )
    bundle = harmonized_resource_bundle(
        harmonized_resource,
        harmonized_manifest,
    )
    prior = _target_prior(
        benchmark,
        spec,
        database_root=selected_database,
        repo_root=root,
    )

    output = prepare_output(output_dir, overwrite=overwrite)
    result_path = output / "result"
    started = time.perf_counter()
    manifest: dict[str, Any] | None = None
    adata: ad.AnnData | None = None
    try:
        adata = ad.read_h5ad(input_path)
        config = CrychicConfig.from_dict(spec["config"])
        included = spec.get("include_cell_types")
        if included is not None:
            selected = tuple(map(str, cast(Sequence[object], included)))
            missing = set(selected).difference(
                adata.obs[config.cell_type_key].astype(str).unique()
            )
            if missing:
                raise ValueError(
                    f"configured cell types are absent from input: {sorted(missing)}"
                )
            mask = adata.obs[config.cell_type_key].astype(str).isin(selected)
            adata = adata[mask].copy()

        sample_metadata = validate_prepared_input(
            adata,
            sample_key=config.sample_key,
            subject_key=config.subject_key,
            cell_type_key=config.cell_type_key,
            context_keys=tuple(config.context_keys),
        )
        method_version = _package_version()
        algorithm_root = Path(crychic.__file__).resolve().parents[2]
        algorithm_code = git_metadata(algorithm_root)
        manifest = begin_manifest(
            repo_root=root,
            dataset_id=str(spec["dataset_id"]),
            method={
                "id": METHOD_ID,
                "version": method_version,
                "benchmark_identity": "generic_multigroup_baseline",
                "entrypoint": "benchmarks.adapters.crychic.run_hcommon",
                "algorithm_code": algorithm_code,
            },
            environment=python_environment(
                environment_name="crychic_project_uv",
                packages=(
                    "CRYCHIC",
                    "anndata",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "threadpoolctl",
                ),
                threads=blas_threads,
            ),
            input_path=input_path,
            input_shape=adata.shape,
            sample_metadata=sample_metadata,
            input_keys={
                "sample_key": config.sample_key,
                "subject_key": config.subject_key,
                "cell_type_key": config.cell_type_key,
                "context_keys": list(config.context_keys),
            },
            resource={
                "mode": "H-common",
                "id": bundle.resource_id,
                "version": bundle.version,
                "manifest_digest": bundle.manifest_digest,
                "interactions": len(bundle.interactions),
                "table_sha256": sha256_file(harmonized_resource),
                "manifest_sha256": sha256_file(harmonized_manifest),
            },
            parameters={
                "benchmark_config": config_path.name,
                "benchmark_config_sha256": sha256_file(config_path),
                "benchmark_dataset": dataset,
                "benchmark_scope": spec.get("benchmark_scope"),
                "input_mode": spec.get("input_mode"),
                "counts_layer": config.counts_layer,
                "analysis_subset": {
                    "include_cell_types": (
                        None if included is None else list(map(str, included))
                    ),
                    "observations": int(adata.n_obs),
                    "variables": int(adata.n_vars),
                },
                "workflow": workflow,
                "communication_mode": communication_mode,
                "scoring_functional_ids": (
                    None
                    if scoring_functional_ids is None
                    else list(scoring_functional_ids)
                ),
                "downstream_evidence_policy": score_policy.value,
            },
            score_semantics={
                "name": (
                    SCORE_NAME
                    if score_policy is DownstreamEvidencePolicy.REQUIRED
                    else MECHANISTIC_SCORE_NAME
                ),
                "direction": SCORE_DIRECTION,
                "interpretation": (
                    "exploratory_downstream_confirmed_strength_not_probability"
                    if score_policy is DownstreamEvidencePolicy.REQUIRED
                    else "exploratory_mechanistic_lr_strength_not_probability"
                ),
                "downstream_evidence_policy": score_policy.value,
                "downstream_evidence_role": (
                    "required_historical_diagnostic"
                    if score_policy is DownstreamEvidencePolicy.REQUIRED
                    else (
                        "independent_annotation_not_required_for_canonical_score"
                        if score_policy is DownstreamEvidencePolicy.ANNOTATE
                        else "disabled"
                    )
                ),
                "probability": None,
                "p_value": None,
                "q_value": None,
                "within_dataset_p_value": "not_emitted",
            },
        )
        model = Crychic(config, resource_bundle=bundle, target_prior=prior)
        dry_keys = {
            "min_cells",
            "min_samples_per_context",
            "min_subjects_per_context",
        }
        dry_kwargs = {key: workflow[key] for key in dry_keys if key in workflow}
        plan = model.dry_run(adata, **dry_kwargs)
        manifest["dry_run"] = {
            "can_fit": plan.can_fit,
            "warnings": list(plan.warnings),
            "stages": [
                {
                    "name": stage.name,
                    "status": stage.status.value,
                    "detail": stage.detail,
                }
                for stage in plan.stages
            ],
        }
        if not plan.can_fit:
            raise RuntimeError("CRYCHIC H-common dry-run blocked fitting")

        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

        code = git_metadata(root)
        with threadpool_limits(limits=blas_threads):
            fitted = model.fit(
                adata,
                output_dir=result_path,
                input_digest=_analysis_digest(observed_sha256, adata.obs_names),
                git_commit=cast(str | None, code["commit"]),
                git_dirty=bool(code["dirty"]),
                package_version=method_version,
                **workflow,
            )
        if not isinstance(fitted, CrychicResult):
            raise TypeError("persisted H-common fit did not return CrychicResult")
        table, views = convert_result_to_long(
            fitted,
            adata,
            bundle,
            dataset_id=str(spec["dataset_id"]),
            resource_mode="H-common",
            communication_mode=communication_mode,
            scoring_functional_ids=scoring_functional_ids,
            downstream_evidence_policy=score_policy,
        )
        score_head_versions = tuple(
            sorted(
                {
                    str(view["score_layer"])
                    for view in views
                    if isinstance(view.get("score_layer"), str)
                }
            )
        )
        if len(score_head_versions) != 1:
            raise ValueError("CRYCHIC long table must have one score-head version")
        cast(dict[str, Any], manifest["method"])["score_head_version"] = (
            score_head_versions[0]
        )
        manifest["source_result"] = {
            "directory": result_path.name,
            "run_id": fitted.manifest["run_id"],
            "result_schema_version": fitted.manifest["result_schema_version"],
            "resource_digests": dict(fitted.manifest["resource_digests"]),
            "score_views": views,
        }
        return cast(
            dict[str, Any],
            finalize_manifest(manifest, table, output, started=started),
        )
    except Exception as exc:
        if manifest is not None:
            fail_manifest(manifest, output, exc, started=started)
        raise
    finally:
        if adata is not None and adata.isbacked:
            adata.file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one configured CRYCHIC dataset with the H-common LR bundle"
    )
    parser.add_argument("benchmark_config", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--harmonized-resource", type=Path, required=True)
    parser.add_argument("--harmonized-manifest", type=Path, required=True)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument(
        "--communication-mode",
        choices=("state", "ecosystem"),
        default="state",
    )
    parser.add_argument("--scoring-functional-id", action="append")
    parser.add_argument(
        "--downstream-evidence-policy",
        choices=(
            DownstreamEvidencePolicy.DISABLED.value,
            DownstreamEvidencePolicy.ANNOTATE.value,
            DownstreamEvidencePolicy.REQUIRED.value,
        ),
        default=DownstreamEvidencePolicy.ANNOTATE.value,
    )
    parser.add_argument("--blas-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = run_hcommon_from_benchmark(
        args.benchmark_config,
        args.dataset,
        args.output_dir,
        harmonized_resource=args.harmonized_resource,
        harmonized_manifest=args.harmonized_manifest,
        database_root=args.database_root,
        communication_mode=args.communication_mode,
        scoring_functional_ids=args.scoring_functional_id,
        downstream_evidence_policy=args.downstream_evidence_policy,
        blas_threads=args.blas_threads,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": manifest["status"], "output": manifest["output"]}))


if __name__ == "__main__":
    main()
