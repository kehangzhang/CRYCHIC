"""Replay the V3 differential head with biological samples as the units.

This adapter deliberately does not reuse the persisted subject-level effect
tables.  It reads the checksum-bound held-out score layers and semantic
availability table, verifies that every sample is represented in exactly one
outer fold, then recomputes the same sender-specific HC2 breadth ranking with
``sample_id`` as the statistical unit.  The source cross-fit is never refit.
"""

from __future__ import annotations

import argparse
import json
import resource as process_resource
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmarks.adapters.common import (
    git_metadata,
    prepare_output,
    sha256_file,
    write_json,
)
from benchmarks.adapters.crychic.des_postprocess import (
    adjusted_subject_directed_lr_effects,
    condition_ranking_diagnostics,
    heldout_sample_coverage_audit,
    score_reason_waterfall,
    sender_specific_direct_scores,
    stable_breadth_unordered_cell_pair_rankings,
)
from benchmarks.adapters.crychic.replay_v3_differential import (
    _bound_output,
    _validated_crossfit_semantic_source,
)

SCHEMA_VERSION = "crychic-multigroup-v3-sample-replay-v1"
DIRECT_EFFECT_FILENAME = "sample_specific_directed_lr_effects.parquet"
RANKING_FILENAME = "condition_cell_pair_rankings.tsv"
PAIR_OPPORTUNITY_FILENAME = "pair_opportunity.tsv"
REASON_WATERFALL_FILENAME = "downstream_reason_waterfall.tsv"

_SOURCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "kuppe": {
        "source_dataset_ids": {"Kuppe_MI_CTRL_vs_IZ"},
        "ranking_dataset_id": "Kuppe_MI_CTRL_vs_IZ",
        "reference": "CTRL",
        "target": "IZ",
        "condition_column": "condition",
        "covariates": (),
        "shape": [76141, 29126],
        "input_sha256": (
            "c47112ce01a192bb157af1ba5feb09c1616601e280102fd11a658f570698c926"
        ),
        "samples_by_condition": {"CTRL": 4, "IZ": 11},
        "subjects_by_condition": {"CTRL": 4, "IZ": 7},
    },
    "ms": {
        "source_dataset_ids": {"UCSC_Lerma_Martin_MS_snRNA_CA_vs_Ctrl"},
        "ranking_dataset_id": "UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca",
        "reference": "Ctrl",
        "target": "CA",
        "condition_column": "lesion_type",
        "covariates": (),
        "shape": [69168, 32115],
        "input_sha256": (
            "433717d9fd98e57e15a444a338a6e1002f7ca3224022321d28a386c0a8498e1c"
        ),
        "samples_by_condition": {"Ctrl": 5, "CA": 6},
        "subjects_by_condition": {"Ctrl": 5, "CA": 5},
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return cast(dict[str, Any], value)


def _output_record(path: Path, table: pd.DataFrame) -> dict[str, object]:
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "columns": list(table.columns),
        "sha256": sha256_file(path),
    }


def _validate_source_contract(
    source_manifest: Mapping[str, object], *, dataset: str
) -> dict[str, Any]:
    contract = _SOURCE_CONTRACTS[dataset]
    if source_manifest.get("status") != "complete":
        raise ValueError("source run must be complete")
    if source_manifest.get("dataset_id") not in contract["source_dataset_ids"]:
        raise ValueError("source run dataset identity is not Figure 3 compatible")
    input_record = source_manifest.get("input")
    if not isinstance(input_record, Mapping):
        raise ValueError("source run lacks an input provenance record")
    if input_record.get("sha256") != contract["input_sha256"]:
        raise ValueError("source run input is not the frozen Figure 3 cohort")
    if input_record.get("shape") != contract["shape"]:
        raise ValueError("source run input shape is not the frozen Figure 3 cohort")
    if input_record.get("samples") != sum(contract["samples_by_condition"].values()):
        raise ValueError("source run sample count disagrees with Figure 3 cohort")
    support = input_record.get("subject_support")
    if support != contract["subjects_by_condition"]:
        raise ValueError("source run subject support disagrees with Figure 3 cohort")
    parameters = source_manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("source run lacks cross-fit parameters")
    if parameters.get("outer_folds") != 2:
        raise ValueError("sample replay requires the two-fold held-out source")
    return contract


def _validate_sample_layer(
    score_layers: pd.DataFrame,
    *,
    condition_column: str,
    samples_by_condition: Mapping[str, int],
    subjects_by_condition: Mapping[str, int],
) -> dict[str, Any]:
    required = {
        "fold_id",
        "sample_id",
        "subject_id",
        condition_column,
        "sender",
        "receiver",
        "interaction_id",
    }
    missing = required.difference(score_layers.columns)
    if missing or score_layers.empty:
        raise ValueError(f"held-out score layer is empty or missing: {sorted(missing)}")
    metadata = score_layers.loc[
        :, ["sample_id", "subject_id", condition_column, "fold_id"]
    ].drop_duplicates(ignore_index=True)
    if metadata["sample_id"].duplicated().any():
        # A sample may contain many LR rows, but its design/fold identity must
        # be unique.  Duplicate rows here indicate inconsistent source layers.
        design = metadata.drop(columns="fold_id").drop_duplicates()
        if design["sample_id"].duplicated().any():
            raise ValueError("sample_id maps to multiple subjects or conditions")
    sample_folds = metadata.loc[:, ["sample_id", "fold_id"]].drop_duplicates()
    if (
        sample_folds.groupby("sample_id", observed=True)["fold_id"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("a sample appears in multiple held-out outer folds")
    design = metadata.drop(columns="fold_id").drop_duplicates(ignore_index=True)
    expected_samples = sum(int(value) for value in samples_by_condition.values())
    if len(design) != expected_samples:
        raise ValueError(
            f"sample design has {len(design)} samples; expected {expected_samples}"
        )
    observed_samples = (
        design.groupby(condition_column, observed=True)["sample_id"].nunique().to_dict()
    )
    if {str(k): int(v) for k, v in observed_samples.items()} != {
        str(k): int(v) for k, v in samples_by_condition.items()
    }:
        raise ValueError("sample condition counts disagree with Figure 3 cohort")
    observed_subjects = (
        design.groupby(condition_column, observed=True)["subject_id"]
        .nunique()
        .to_dict()
    )
    if {str(k): int(v) for k, v in observed_subjects.items()} != {
        str(k): int(v) for k, v in subjects_by_condition.items()
    }:
        raise ValueError("sample subject counts disagree with Figure 3 cohort")
    return {
        "sample_coverage": heldout_sample_coverage_audit(
            score_layers, design.loc[:, ["sample_id", "subject_id"]]
        ),
        "samples_by_condition": {str(k): int(v) for k, v in observed_samples.items()},
        "subjects_by_condition": {str(k): int(v) for k, v in observed_subjects.items()},
        "folds": sorted(sample_folds["fold_id"].astype(str).unique()),
    }


def _sample_effects(
    direct_scores: pd.DataFrame,
    *,
    reference: str,
    target: str,
    condition_column: str,
) -> pd.DataFrame:
    """Apply the reviewed HC2 contrast with sample IDs as independent units.

    ``adjusted_subject_directed_lr_effects`` is the shared implementation of
    the HC2 vectorized contrast.  This wrapper supplies a private unitized copy
    where each sample is one pseudo-subject.  The persisted source subject IDs
    and all subject-level result files remain untouched; no subject result is
    relabelled or consumed here.
    """

    required = {"sample_id", "subject_id", condition_column}
    missing = required.difference(direct_scores.columns)
    if missing:
        raise ValueError(
            f"direct scores are missing sample design columns: {sorted(missing)}"
        )
    sample_subject = direct_scores.loc[:, ["sample_id", "subject_id"]].drop_duplicates(
        ignore_index=True
    )
    if sample_subject["sample_id"].duplicated().any():
        raise ValueError(
            "sample-level direct scores map one sample to multiple subjects"
        )
    working = direct_scores.copy(deep=True)
    working["subject_id"] = working["sample_id"].astype(str)
    effects = adjusted_subject_directed_lr_effects(
        working,
        reference=reference,
        target=target,
        condition_column=condition_column,
        edge_columns=(
            "sender",
            "receiver",
            "interaction_id",
            "ligand",
            "receptor",
            "family_id",
            "driver_id",
        ),
        categorical_covariates=(),
        min_subjects_per_condition=2,
    )
    effects["effect_semantics"] = (
        effects["effect_semantics"]
        .astype(str)
        .str.replace("subject_level", "sample_level", regex=False)
    )
    effects["covariate_adjustment"] = "none"
    return effects


def run(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Replay one completed source run without refitting."""

    if dataset not in _SOURCE_CONTRACTS:
        raise ValueError("dataset must be 'kuppe' or 'ms'")
    started = time.perf_counter()
    source = Path(run_dir).expanduser().resolve()
    output = prepare_output(output_dir, overwrite=overwrite)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    source_manifest = _read_json(manifest_path)
    contract = _validate_source_contract(source_manifest, dataset=dataset)
    method = source_manifest.get("method")
    resource = source_manifest.get("resource")
    if not isinstance(method, Mapping) or not isinstance(resource, Mapping):
        raise ValueError("source run lacks method/resource provenance")
    source_method_version = str(method.get("version"))
    code_metadata = git_metadata(Path(__file__).resolve().parents[3])
    replay_commit = code_metadata.get("commit")
    method_version = (
        f"{source_method_version};sample_replay={replay_commit[:12]}"
        if isinstance(replay_commit, str) and replay_commit
        else f"{source_method_version};sample_replay"
    )
    score_layer_path = _bound_output(
        source, source_manifest, "sender_lr_score_layers.parquet"
    )
    _, semantic_path, crossfit_manifest_sha, semantic_sha = (
        _validated_crossfit_semantic_source(source, source_manifest)
    )
    condition_column = str(contract["condition_column"])
    score_columns = [
        "fold_id",
        "sample_id",
        "subject_id",
        condition_column,
        "sender",
        "receiver",
        "interaction_id",
        "ligand",
        "receptor",
        "family_id",
        "driver_id",
        "mechanistic_status",
        "mechanistic_reason_code",
        "downstream_status",
        "downstream_reason_code",
        "selected_score_status",
        "selected_score_reason_code",
        "status",
        "reason_code",
    ]
    score_layers = pd.read_parquet(score_layer_path, columns=score_columns)
    coverage = _validate_sample_layer(
        score_layers,
        condition_column=condition_column,
        samples_by_condition=contract["samples_by_condition"],
        subjects_by_condition=contract["subjects_by_condition"],
    )
    semantic = pd.read_parquet(
        semantic_path,
        columns=[
            "fold_id",
            "sample_id",
            "subject_id",
            "sender",
            "receiver",
            "interaction_id",
            "mode",
            "availability_score",
            "status",
            "reason_code",
        ],
        filters=[("mode", "==", "state")],
    )
    direct_scores = sender_specific_direct_scores(
        score_layers,
        semantic,
        condition_column=condition_column,
        edge_columns=(
            "sender",
            "receiver",
            "interaction_id",
            "ligand",
            "receptor",
            "family_id",
            "driver_id",
        ),
        response_transform="identity",
    )
    waterfall = score_reason_waterfall(score_layers, direct_scores)
    effects = _sample_effects(
        direct_scores,
        reference=str(contract["reference"]),
        target=str(contract["target"]),
        condition_column=condition_column,
    )
    rankings, pair_opportunity = stable_breadth_unordered_cell_pair_rankings(
        effects,
        dataset=str(contract["ranking_dataset_id"]),
        method="crychic",
        method_version=method_version,
        resource=str(resource.get("resource_id")),
    )
    rankings["ranking_semantics"] = rankings["ranking_semantics"].astype(str) + (
        ";sample_level_replay"
    )
    diagnostics = condition_ranking_diagnostics(rankings)
    if diagnostics["any_degenerate_condition"]:
        raise RuntimeError("sample replay produced a degenerate condition ranking")
    direct_effect_path = output / DIRECT_EFFECT_FILENAME
    ranking_path = output / RANKING_FILENAME
    opportunity_path = output / PAIR_OPPORTUNITY_FILENAME
    waterfall_path = output / REASON_WATERFALL_FILENAME
    effects.to_parquet(direct_effect_path, index=False, compression="zstd")
    rankings.to_csv(ranking_path, sep="\t", index=False, lineterminator="\n")
    pair_opportunity.to_csv(
        opportunity_path, sep="\t", index=False, lineterminator="\n"
    )
    waterfall.to_csv(waterfall_path, sep="\t", index=False, lineterminator="\n")
    observed_effect = effects.loc[
        effects["status"].eq("observed"), "effect_target_minus_reference"
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset": dataset,
        "dataset_id": contract["ranking_dataset_id"],
        "source_dataset_id": source_manifest["dataset_id"],
        "reference": contract["reference"],
        "target": contract["target"],
        "analysis_unit": {
            "replicate_key": "sample_id",
            "subject_key": "subject_id",
            "primary_panel": True,
            "paper_figure3_primary_panel": True,
        },
        "source_run": {
            "path": str(source),
            "manifest_sha256": sha256_file(manifest_path),
            "crossfit_manifest_sha256": crossfit_manifest_sha,
            "score_layers_sha256": sha256_file(score_layer_path),
            "semantic_availability_sha256": semantic_sha,
        },
        "source_input": source_manifest["input"],
        "method": {
            "id": "crychic",
            "version": method_version,
            "backbone_version": source_method_version,
        },
        "resource": dict(resource),
        "implementation": {
            "script_sha256": sha256_file(Path(__file__)),
            "des_postprocess_sha256": sha256_file(
                Path(__file__).with_name("des_postprocess.py")
            ),
            "response": "sender_specific_ligand_x_receptor_availability",
            "statistical_unit": "sample_id",
            "subject_result_reuse": False,
            "technical_sample_policy": "none_each_sample_is_one_unit",
            "covariates": [],
            "uncertainty": (
                "contrast_specific_hc2_heteroskedasticity_robust_standard_error"
            ),
            "ranking": "one_standard_error_stable_lr_breadth",
            "formal_inference_allowed": False,
        },
        "coverage": coverage,
        "diagnostics": {
            "condition_rankings": diagnostics,
            "observed_effect_rows": len(observed_effect),
            "stable_effect_rows": int(
                effects.loc[
                    effects["status"].eq("observed"), "one_standard_error_stable"
                ].sum()
            ),
            "effect_sign_counts": {
                "positive": int(observed_effect.gt(0.0).sum()),
                "negative": int(observed_effect.lt(0.0).sum()),
                "zero": int(observed_effect.eq(0.0).sum()),
            },
        },
        "performance": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(
                process_resource.getrusage(process_resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "code": code_metadata,
        "outputs": {
            DIRECT_EFFECT_FILENAME: _output_record(direct_effect_path, effects),
            RANKING_FILENAME: _output_record(ranking_path, rankings),
            PAIR_OPPORTUNITY_FILENAME: _output_record(
                opportunity_path, pair_opportunity
            ),
            REASON_WATERFALL_FILENAME: _output_record(waterfall_path, waterfall),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", choices=sorted(_SOURCE_CONTRACTS), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        args.run_dir, args.output_dir, dataset=args.dataset, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output_dir.resolve())},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["_sample_effects", "_validate_sample_layer", "run"]
