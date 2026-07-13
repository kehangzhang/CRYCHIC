"""Audit Kuppe repeated-measures estimability without reading expression data.

The audit operates on the 29-row sample design, either read directly from a
prepared TSV or derived from ``AnnData.obs`` opened in backed mode.  It freezes
the reviewed repeated-measures backend for every declared contrast but never
fits an effect and never emits inferential statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import anndata as ad
import pandas as pd

from benchmarks.metrics.repeated_measures import (
    FrozenRepeatedMeasuresDesign,
    RepeatedMeasuresSpec,
    freeze_repeated_measures_design,
)

DATASET_ID = "Kuppe_MI_Zenodo6578047"
AUDIT_SCHEMA = "crychic-kuppe-repeated-measures-design-audit-v1"
SUMMARY_SCHEMA = "crychic-kuppe-repeated-measures-design-summary-v1"
DEFAULT_REGIONS = ("CTRL", "RZ", "BZ", "IZ", "FZ")
DEFAULT_CONDITIONS = ("myogenic", "ischemic", "fibrotic")

_ALIASES = {
    "sample_id": ("sample_id", "sample"),
    "subject_id": ("subject_id", "patient"),
    "region": ("region", "major_labl"),
    "condition": ("condition", "patient_group"),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_column(table: pd.DataFrame, field: str) -> pd.Series | None:
    aliases = _ALIASES[field]
    present = [name for name in aliases if name in table.columns]
    if not present:
        return None
    result = table[present[0]].astype(str)
    for name in present[1:]:
        if not table[name].astype(str).equals(result):
            raise ValueError(
                f"Kuppe metadata aliases for {field!r} disagree: {present}"
            )
    if (
        result.isna().any()
        or result.eq("").any()
        or result.str.strip().ne(result).any()
    ):
        raise ValueError(f"Kuppe field {field!r} has missing or non-canonical labels")
    return result


def sample_design_from_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Collapse cell-level Kuppe metadata to one auditable row per sample."""

    canonical: dict[str, pd.Series] = {}
    for field in ("sample_id", "subject_id", "region"):
        values = _canonical_column(obs, field)
        if values is None:
            raise ValueError(f"Kuppe metadata is missing aliases for {field!r}")
        canonical[field] = values
    condition = _canonical_column(obs, "condition")
    if condition is not None:
        canonical["condition"] = condition
    table = pd.DataFrame(canonical, index=obs.index)
    mapping_fields = ["subject_id", "region"]
    if "condition" in table:
        mapping_fields.append("condition")
    grouped = table.groupby("sample_id", observed=True, sort=True)
    for field in mapping_fields:
        ambiguous = grouped[field].nunique().gt(1)
        if ambiguous.any():
            bad = ambiguous.index[ambiguous].astype(str).tolist()
            raise ValueError(f"sample IDs map to multiple {field} values: {bad}")
    result = grouped[mapping_fields].first()
    result["n_cells"] = grouped.size()
    return result.reset_index()


def read_sample_design(path: Path) -> pd.DataFrame:
    """Read and canonicalize a compact Kuppe sample-design TSV."""

    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, sep="\t")
    if table.empty:
        raise ValueError("Kuppe sample-design TSV is empty")
    return sample_design_from_obs(table)


def read_backed_obs_sample_design(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read only ``obs`` from a backed h5ad and close it immediately."""

    if not path.is_file():
        raise FileNotFoundError(path)
    source = ad.read_h5ad(path, backed="r")
    try:
        if not source.isbacked:
            raise ValueError("Kuppe h5ad must be opened in backed mode")
        sample_design = sample_design_from_obs(source.obs)
        shape = [int(source.n_obs), int(source.n_vars)]
    finally:
        source.file.close()
    lineage = {
        "input_kind": "h5ad_obs_backed_r",
        "source_filename": path.name,
        "source_size_bytes": path.stat().st_size,
        "source_shape": shape,
        "expression_matrix_accessed": False,
    }
    return sample_design, lineage


def _ordered_levels(
    observed: Sequence[str], declared: Sequence[str], *, field: str
) -> tuple[str, ...]:
    observed_set = set(observed)
    declared_tuple = tuple(declared)
    if len(set(declared_tuple)) != len(declared_tuple):
        raise ValueError(f"declared {field} levels must be unique")
    if observed_set != set(declared_tuple):
        raise ValueError(
            f"observed {field} levels {sorted(observed_set)} do not match "
            f"declared levels {sorted(declared_tuple)}"
        )
    return declared_tuple


def _allocation_support(
    design: FrozenRepeatedMeasuresDesign,
) -> dict[str, object]:
    spec = design.spec
    cells = design.cell_table
    reference = set(
        cells.loc[
            cells[spec.context_key].eq(spec.reference), spec.subject_key
        ].astype(str)
    )
    target = set(
        cells.loc[
            cells[spec.context_key].eq(spec.target), spec.subject_key
        ].astype(str)
    )
    sample_counts = design.sample_table.groupby("__design_row", observed=True).size()
    subject_counts = cells.groupby(spec.subject_key, observed=True).size()
    return {
        "n_reference_subjects": len(reference),
        "n_target_subjects": len(target),
        "n_paired_subjects": len(reference.intersection(target)),
        "n_reference_only_subjects": len(reference.difference(target)),
        "n_target_only_subjects": len(target.difference(reference)),
        "n_contrast_subject_clusters": len(reference.union(target)),
        "n_all_subject_clusters": int(cells[spec.subject_key].nunique()),
        "n_design_cells": len(cells),
        "n_repeated_subject_clusters": int(subject_counts.gt(1).sum()),
        "n_replicated_design_cells": int(sample_counts.gt(1).sum()),
        "paired_subject_ids": sorted(reference.intersection(target)),
    }


def _complete_coverage_status(
    design: FrozenRepeatedMeasuresDesign,
    support: Mapping[str, object],
) -> tuple[str, str | None]:
    spec = design.spec
    if not design.estimable:
        return "not_estimable", design.reason_code
    if min(
        cast(int, support["n_reference_subjects"]),
        cast(int, support["n_target_subjects"]),
    ) < spec.min_subjects_per_context:
        return "not_estimable", "insufficient_subjects_per_context"
    if cast(int, support["n_contrast_subject_clusters"]) < spec.min_subject_clusters:
        return "not_estimable", "insufficient_subject_clusters"
    if cast(int, support["n_design_cells"]) <= len(design.model_columns):
        return "not_estimable", "insufficient_residual_degrees_of_freedom"
    if cast(int, support["n_all_subject_clusters"]) <= len(design.model_columns):
        return "not_estimable", "insufficient_clusters_for_sandwich"
    return "estimable_under_complete_score_coverage", None


def _contrast_record(
    sample_design: pd.DataFrame,
    *,
    context_key: str,
    reference: str,
    target: str,
    region_key: str | None,
    min_subjects_per_context: int,
    min_subject_clusters: int,
) -> dict[str, object]:
    spec = RepeatedMeasuresSpec(
        context_key=context_key,
        reference=reference,
        target=target,
        region_key=region_key,
        min_subjects_per_context=min_subjects_per_context,
        min_subject_clusters=min_subject_clusters,
    )
    design = freeze_repeated_measures_design(sample_design, spec=spec)
    support = _allocation_support(design)
    status, reason = _complete_coverage_status(design, support)
    return {
        "contrast": f"{target}_vs_{reference}",
        "reference": reference,
        "target": target,
        **support,
        "design_kind": design.design_kind,
        "design_status": design.status,
        "complete_score_coverage_status": status,
        "reason_code": reason,
        "design_rank": design.design_rank,
        "n_model_columns": len(design.model_columns),
        "condition_number": design.condition_number,
        "model_columns": list(design.model_columns),
        "factor_levels": [
            {"field": field, "levels": list(levels)}
            for field, levels in design.factor_levels
        ],
        "spec_id": spec.spec_id,
        "design_id": design.design_id,
        "effect_fit_run": False,
        "formal_inference_allowed": False,
    }


def _manifest_sha256(
    sample_design: pd.DataFrame, *, columns: Sequence[str]
) -> str:
    missing = set(columns).difference(sample_design.columns)
    if missing:
        raise ValueError(f"manifest columns are missing: {sorted(missing)}")
    records = (
        sample_design.loc[:, list(columns)]
        .sort_values("sample_id", kind="stable")
        .to_dict(orient="records")
    )
    return _sha256_bytes(_canonical_json(records).encode("utf-8"))


def _sample_manifest_sha256(sample_design: pd.DataFrame) -> str:
    """Hash the region design shared by TSV and backed-obs input paths."""

    columns = ["sample_id", "subject_id", "region"]
    return _manifest_sha256(sample_design, columns=columns)


def audit_kuppe_repeated_measures(
    sample_design: pd.DataFrame,
    *,
    lineage: Mapping[str, object],
    regions: Sequence[str] = DEFAULT_REGIONS,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    min_subjects_per_context: int = 3,
    min_subject_clusters: int = 6,
) -> dict[str, object]:
    """Audit full-coverage design estimability without fitting score effects."""

    required = {"sample_id", "subject_id", "region"}
    missing = required.difference(sample_design.columns)
    if missing:
        raise ValueError(f"sample design is missing columns: {sorted(missing)}")
    if sample_design["sample_id"].duplicated().any():
        raise ValueError("sample design must contain one row per sample")
    region_levels = _ordered_levels(
        sample_design["region"].astype(str).drop_duplicates().tolist(),
        regions,
        field="region",
    )
    region_contrasts = [
        _contrast_record(
            sample_design,
            context_key="region",
            reference=reference,
            target=target,
            region_key="region",
            min_subjects_per_context=min_subjects_per_context,
            min_subject_clusters=min_subject_clusters,
        )
        for reference, target in itertools.combinations(region_levels, 2)
    ]

    if "condition" in sample_design:
        condition_levels = _ordered_levels(
            sample_design["condition"].astype(str).drop_duplicates().tolist(),
            conditions,
            field="condition",
        )
        condition_contrasts: list[dict[str, object]] = [
            _contrast_record(
                sample_design,
                context_key="condition",
                reference=reference,
                target=target,
                region_key="region",
                min_subjects_per_context=min_subjects_per_context,
                min_subject_clusters=min_subject_clusters,
            )
            for reference, target in itertools.combinations(condition_levels, 2)
        ]
        condition_status = "audited"
        condition_reason = None
        condition_manifest_sha256 = _manifest_sha256(
            sample_design,
            columns=("sample_id", "subject_id", "region", "condition"),
        )
    else:
        condition_contrasts = []
        condition_status = "not_run"
        condition_reason = "condition_field_not_available_in_sample_design"
        condition_manifest_sha256 = None

    region_status_counts = pd.Series(
        [row["complete_score_coverage_status"] for row in region_contrasts]
    ).value_counts()
    condition_status_counts = pd.Series(
        [row["complete_score_coverage_status"] for row in condition_contrasts],
        dtype="object",
    ).value_counts()
    repeated_cells = (
        sample_design.groupby(["subject_id", "region"], observed=True).size()
    )
    replicated_cells: dict[str, int] = {}
    for key, value in repeated_cells[repeated_cells.gt(1)].items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("expected subject/region grouped support")
        subject, region = key
        replicated_cells[f"{subject}|{region}"] = int(value)
    subject_regions = sample_design.drop_duplicates(["subject_id", "region"])
    return {
        "schema_version": AUDIT_SCHEMA,
        "dataset_id": DATASET_ID,
        "analysis_scope": "metadata_only_design_estimability_no_effect_fit",
        "lineage": dict(lineage),
        "sample_manifest_sha256": _sample_manifest_sha256(sample_design),
        "condition_manifest_sha256": condition_manifest_sha256,
        "support": {
            "n_samples": len(sample_design),
            "n_subjects": int(sample_design["subject_id"].nunique()),
            "n_regions": len(region_levels),
            "regions": list(region_levels),
            "subjects_by_region": {
                str(key): int(value)
                for key, value in subject_regions.groupby(
                    "region", observed=True
                )["subject_id"].nunique().items()
            },
            "samples_by_region": {
                str(key): int(value)
                for key, value in sample_design.groupby("region", observed=True)[
                    "sample_id"
                ].nunique().items()
            },
            "within_subject_region_replicates": replicated_cells,
            "batch_key": None,
            "batch_status": "not_declared_no_auditable_batch_field_in_source_obs",
        },
        "backend_policy": {
            "model": "categorical_ols_subject_cluster_cr1_v1",
            "min_subjects_per_context": min_subjects_per_context,
            "min_subject_clusters": min_subject_clusters,
            "replicate_policy": "mean_within_subject_context_adjustment_cell",
            "paired_unpaired_policy": "retain_both_with_subject_cluster",
            "uncertainty": "diagnostic_only_not_run_by_this_design_audit",
        },
        "region_model": {
            "formula_semantics": "categorical_region_fixed_effect_all_five_regions",
            "contrast_count": len(region_contrasts),
            "status_counts": {
                str(key): int(value) for key, value in region_status_counts.items()
            },
            "contrasts": region_contrasts,
        },
        "condition_region_adjusted_model": {
            "condition_field": "patient_group_mapped_to_condition",
            "region_role": "declared_categorical_adjustment_not_batch",
            "status": condition_status,
            "reason_code": condition_reason,
            "contrast_count": len(condition_contrasts),
            "status_counts": {
                str(key): int(value) for key, value in condition_status_counts.items()
            },
            "contrasts": condition_contrasts,
        },
        "guardrails": {
            "expression_matrix_accessed": False,
            "effect_estimated": False,
            "standard_error_estimated": False,
            "p_or_q_reported": False,
            "biological_effect_claim_allowed": False,
            "complete_score_coverage_is_an_assumption_not_an_observation": True,
        },
    }


def write_audit(result: Mapping[str, object], output_path: Path) -> None:
    """Write a stable, strict JSON audit."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def compact_summary(result: Mapping[str, object]) -> dict[str, object]:
    """Reduce a full design audit to a Git-sized benchmark result."""

    region_model = cast(Mapping[str, object], result["region_model"])
    condition_model = cast(
        Mapping[str, object], result["condition_region_adjusted_model"]
    )

    def compact_contrasts(model: Mapping[str, object]) -> list[dict[str, object]]:
        contrasts = cast(Sequence[Mapping[str, object]], model["contrasts"])
        fields = (
            "contrast",
            "complete_score_coverage_status",
            "reason_code",
            "design_kind",
            "n_reference_subjects",
            "n_target_subjects",
            "n_paired_subjects",
            "n_contrast_subject_clusters",
            "design_rank",
            "n_model_columns",
            "design_id",
        )
        return [{field: row[field] for field in fields} for row in contrasts]

    canonical_sha256 = _sha256_bytes(
        _canonical_json(dict(result)).encode("utf-8")
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "dataset_id": result["dataset_id"],
        "analysis_scope": result["analysis_scope"],
        "full_audit_canonical_sha256": canonical_sha256,
        "sample_manifest_sha256": result["sample_manifest_sha256"],
        "condition_manifest_sha256": result["condition_manifest_sha256"],
        "lineage": result["lineage"],
        "support": result["support"],
        "backend_policy": result["backend_policy"],
        "region_model": {
            "formula_semantics": region_model["formula_semantics"],
            "contrast_count": region_model["contrast_count"],
            "status_counts": region_model["status_counts"],
            "contrasts": compact_contrasts(region_model),
        },
        "condition_region_adjusted_model": {
            "condition_field": condition_model["condition_field"],
            "region_role": condition_model["region_role"],
            "contrast_count": condition_model["contrast_count"],
            "status_counts": condition_model["status_counts"],
            "contrasts": compact_contrasts(condition_model),
        },
        "interpretation_limits": {
            "region_estimability_assumption": (
                "complete_score_coverage_for_each_method_edge"
            ),
            "condition_result": (
                "not_estimable_after_declared_region_adjustment_due_to_"
                "rank_deficient_design"
            ),
            "batch_result": "no_auditable_batch_field_declared",
            "effect_or_biological_claim": False,
        },
        "guardrails": result["guardrails"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--h5ad", type=Path)
    source.add_argument("--sample-design", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--min-subjects-per-context", type=int, default=3)
    parser.add_argument("--min-subject-clusters", type=int, default=6)
    args = parser.parse_args()

    if args.h5ad is not None:
        sample_design, lineage = read_backed_obs_sample_design(args.h5ad)
    else:
        sample_design = read_sample_design(args.sample_design)
        lineage = {
            "input_kind": "sample_design_tsv",
            "source_filename": args.sample_design.name,
            "source_size_bytes": args.sample_design.stat().st_size,
            "expression_matrix_accessed": False,
        }
    result = audit_kuppe_repeated_measures(
        sample_design,
        lineage=lineage,
        min_subjects_per_context=args.min_subjects_per_context,
        min_subject_clusters=args.min_subject_clusters,
    )
    write_audit(result, args.output)
    if args.summary_output is not None:
        write_audit(compact_summary(result), args.summary_output)


if __name__ == "__main__":
    main()
