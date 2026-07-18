"""Run the Cesaro et al. scSeqCommDiff 2.0.0 benchmark protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import anndata as ad
import pandas as pd

from benchmarks.adapters.common import sha256_file, write_json

METHOD_VERSION = "2.0.0"
PAPER_NEIGHBOR_COMMIT = "5a29240a237c5520c83ab82527ba55e0e1ceaff5"
ZENODO_RECORD = "12790607"
ZENODO_ARCHIVE_SHA256 = (
    "0b6c3f6b5c47ef4196763f1bc5bf6a1ee9024b0b9f31c2d54b123c00411c9e5b"
)
RESOURCE_ID = "ConnectomeDB2020_Hou_2020_human"
RESOURCE_SHA256 = "e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a"
RESOURCE_ROWS = 2293
SCHEMA_VERSION = "crychic-scseqcommdiff-paper-benchmark-v1"
VALID_SCENARIOS = frozenset({"multi-condition", "multi-sample"})
MIN_VALID_PSEUDOBULK_UNITS = 2
MIN_VALID_CONDITION_CELLS = 2


def _input_manifest_sha256(payload: dict[str, Any], filename: str) -> str:
    output = payload.get("output")
    if isinstance(output, dict) and output.get("filename") == filename:
        digest = output.get("sha256")
        if isinstance(digest, str):
            return digest
    if payload.get("output") == filename:
        digest = payload.get("output_sha256")
        if isinstance(digest, str):
            return digest
    raise ValueError("input manifest does not bind the requested h5ad checksum")


def _validate_resource(resource_path: Path, manifest_path: Path) -> dict[str, Any]:
    if sha256_file(resource_path) != RESOURCE_SHA256:
        raise ValueError("ConnectomeDB2020 payload checksum does not match the pin")
    manifest_object: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_object, dict):
        raise ValueError("ConnectomeDB2020 manifest must be a JSON object")
    manifest = cast(dict[str, Any], manifest_object)
    if manifest.get("resource_id") != RESOURCE_ID:
        raise ValueError("ConnectomeDB2020 resource_id does not match the pin")
    if manifest.get("payload", {}).get("sha256") != RESOURCE_SHA256:
        raise ValueError("ConnectomeDB2020 manifest payload checksum is inconsistent")
    table = pd.read_csv(resource_path, sep="\t", dtype=str)
    required = {"ligand", "receptor", "scseqcommdiff_covered"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"ConnectomeDB2020 columns are missing: {sorted(missing)}")
    covered = table["scseqcommdiff_covered"].str.lower().eq("true")
    if len(table) != RESOURCE_ROWS or not covered.all():
        raise ValueError("scSeqCommDiff must cover all 2293 ConnectomeDB2020 pairs")
    if table.duplicated(["ligand", "receptor"]).any():
        raise ValueError("ConnectomeDB2020 contains duplicate directed pairs")
    return manifest


def _r_environment(rscript: Path) -> dict[str, Any]:
    expression = (
        "suppressPackageStartupMessages(library(scSeqComm)); "
        "cat(as.character(getRversion()), '\\n', "
        "as.character(packageVersion('scSeqComm')), '\\n', sep='')"
    )
    completed = subprocess.run(
        [str(rscript), "-e", expression],
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.strip().splitlines()
    if len(values) != 2 or values[1] != METHOD_VERSION:
        raise RuntimeError("the R environment does not contain scSeqComm 2.0.0")
    return {"r": values[0], "scSeqComm": values[1]}


def _prepare_metadata(
    input_h5ad: Path,
    *,
    cell_type_key: str,
    condition_key: str,
    sample_unit_key: str,
    target: str,
    reference: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = ad.read_h5ad(input_h5ad, backed="r")
    try:
        required = {cell_type_key, condition_key, sample_unit_key}
        missing = required.difference(source.obs.columns)
        if missing:
            raise ValueError(f"prepared h5ad metadata is missing: {sorted(missing)}")
        if not source.obs_names.is_unique or not source.var_names.is_unique:
            raise ValueError("prepared h5ad cell and gene identifiers must be unique")
        obs = source.obs.loc[:, sorted(required)].copy()
        if obs.isna().any().any():
            raise ValueError("prepared h5ad benchmark metadata contains null values")
        observed = set(obs[condition_key].astype(str))
        if observed != {target, reference}:
            raise ValueError(
                f"observed conditions {sorted(observed)} do not match the contrast"
            )
        metadata = pd.DataFrame(
            {
                "Cell_ID": source.obs_names.astype(str),
                "Cluster_ID": obs[cell_type_key].astype(str).to_numpy(),
                "Condition_ID": obs[condition_key].astype(str).to_numpy(),
                "Sample_ID": obs[sample_unit_key].astype(str).to_numpy(),
            }
        )
        design = metadata.loc[:, ["Sample_ID", "Condition_ID"]].drop_duplicates()
        if design.duplicated("Sample_ID", keep=False).any():
            raise ValueError("one Sample_ID maps to more than one condition")
        samples = design.groupby("Condition_ID", observed=True).size().to_dict()
        audit = {
            "shape": [int(source.n_obs), int(source.n_vars)],
            "x_backend": type(source.X).__name__,
            "cell_types": sorted(metadata["Cluster_ID"].unique()),
            "n_cell_types": int(metadata["Cluster_ID"].nunique()),
            "units_by_condition": {
                str(key): int(value) for key, value in samples.items()
            },
            "sample_unit_key": sample_unit_key,
        }
        return metadata, audit
    finally:
        source.file.close()


def _filter_multi_sample_metadata(
    metadata: pd.DataFrame,
    *,
    target: str,
    reference: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Retain cell types for which native pseudo-Wilcoxon is estimable."""
    design = metadata.loc[:, ["Sample_ID", "Condition_ID"]].drop_duplicates()
    units = {
        condition: sorted(
            design.loc[design["Condition_ID"].eq(condition), "Sample_ID"].astype(str)
        )
        for condition in (target, reference)
    }
    cell_types = sorted(metadata["Cluster_ID"].astype(str).unique())
    counts = (
        metadata.groupby(
            ["Cluster_ID", "Condition_ID", "Sample_ID"],
            observed=True,
            sort=True,
        )
        .size()
        .rename("n_cells")
    )
    records: list[dict[str, Any]] = []
    for cell_type in cell_types:
        for condition in (target, reference):
            for sample_id in units[condition]:
                key = (cell_type, condition, sample_id)
                records.append(
                    {
                        "cell_type": cell_type,
                        "condition": condition,
                        "sample_id": sample_id,
                        "n_cells": int(counts.get(key, 0)),
                    }
                )
    support = pd.DataFrame.from_records(records)
    support["valid_pseudobulk"] = support["n_cells"].gt(1)
    valid_units = (
        support.groupby(["cell_type", "condition"], observed=True, sort=True)[
            "valid_pseudobulk"
        ]
        .sum()
        .astype(int)
    )
    eligible = {
        cell_type
        for cell_type in cell_types
        if all(
            int(valid_units.get((cell_type, condition), 0))
            >= MIN_VALID_PSEUDOBULK_UNITS
            for condition in (target, reference)
        )
    }
    support["valid_units_in_condition"] = [
        int(valid_units.loc[(cell_type, condition)])
        for cell_type, condition in zip(
            support["cell_type"], support["condition"], strict=True
        )
    ]
    support["analysis_eligible"] = support["cell_type"].isin(eligible)
    support["reason_code"] = support["analysis_eligible"].map(
        {True: "", False: "insufficient_common_pseudobulk_support"}
    )
    filtered = metadata.loc[metadata["Cluster_ID"].isin(eligible)].copy()
    excluded = sorted(set(cell_types).difference(eligible))
    if not eligible:
        raise ValueError("no cell type has estimable multi-sample pseudobulk support")
    audit = {
        "policy": ("both_conditions_have_at_least_2_units_with_more_than_1_cell"),
        "minimum_valid_units_per_condition": MIN_VALID_PSEUDOBULK_UNITS,
        "source_cell_types": cell_types,
        "analysis_cell_types": sorted(eligible),
        "excluded_cell_types": excluded,
        "source_cells": len(metadata),
        "analysis_cells": len(filtered),
    }
    return filtered, support, audit


def _filter_multi_condition_metadata(
    metadata: pd.DataFrame,
    *,
    target: str,
    reference: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Retain cell types that produce equal native permutation score frames."""
    cell_types = sorted(metadata["Cluster_ID"].astype(str).unique())
    counts = (
        metadata.groupby(["Cluster_ID", "Condition_ID"], observed=True, sort=True)
        .size()
        .rename("n_cells")
    )
    records: list[dict[str, Any]] = []
    for cell_type in cell_types:
        for condition in (target, reference):
            n_cells = int(counts.get((cell_type, condition), 0))
            records.append(
                {
                    "cell_type": cell_type,
                    "condition": condition,
                    "sample_id": "pooled_condition",
                    "n_cells": n_cells,
                    # The support schema is shared with the multi-sample arm.
                    "valid_pseudobulk": n_cells >= MIN_VALID_CONDITION_CELLS,
                    "valid_units_in_condition": int(
                        n_cells >= MIN_VALID_CONDITION_CELLS
                    ),
                }
            )
    support = pd.DataFrame.from_records(records)
    eligible = {
        cell_type
        for cell_type in cell_types
        if all(
            int(counts.get((cell_type, condition), 0)) >= MIN_VALID_CONDITION_CELLS
            for condition in (target, reference)
        )
    }
    support["analysis_eligible"] = support["cell_type"].isin(eligible)
    support["reason_code"] = support["analysis_eligible"].map(
        {True: "", False: "insufficient_common_condition_support"}
    )
    filtered = metadata.loc[metadata["Cluster_ID"].isin(eligible)].copy()
    excluded = sorted(set(cell_types).difference(eligible))
    if not eligible:
        raise ValueError("no cell type has estimable multi-condition support")
    audit = {
        "policy": "both_conditions_have_at_least_2_cells_for_label_permutation",
        "minimum_cells_per_condition": MIN_VALID_CONDITION_CELLS,
        "source_cell_types": cell_types,
        "analysis_cell_types": sorted(eligible),
        "excluded_cell_types": excluded,
        "source_cells": len(metadata),
        "analysis_cells": len(filtered),
    }
    return filtered, support, audit


def _unfiltered_cell_type_support(metadata: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_type": sorted(metadata["Cluster_ID"].astype(str).unique()),
            "condition": "not_applicable",
            "sample_id": "not_applicable",
            "n_cells": 0,
            "valid_pseudobulk": False,
            "valid_units_in_condition": 0,
            "analysis_eligible": True,
            "reason_code": "",
        }
    )


def _external_process_failure(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    stderr_lines = completed.stderr.strip().splitlines()
    return {
        "type": "ExternalProcessError",
        "message": (
            "scSeqCommDiff R runner exited with code "
            f"{completed.returncode}; inspect stderr.log"
        ),
        "returncode": completed.returncode,
        "stderr_log": "stderr.log",
        "stderr_tail": "\n".join(stderr_lines[-40:]),
    }


def preflight(
    input_h5ad: Path,
    input_manifest: Path,
    resource_path: Path,
    resource_manifest: Path,
    rscript: Path,
    *,
    cell_type_key: str,
    condition_key: str,
    sample_unit_key: str,
    target: str,
    reference: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    for path in (
        input_h5ad,
        input_manifest,
        resource_path,
        resource_manifest,
        rscript,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_object: object = json.loads(input_manifest.read_text(encoding="utf-8"))
    if not isinstance(input_object, dict):
        raise ValueError("prepared h5ad manifest must be a JSON object")
    input_payload = cast(dict[str, Any], input_object)
    expected_input_sha = _input_manifest_sha256(input_payload, input_h5ad.name)
    actual_input_sha = sha256_file(input_h5ad)
    if actual_input_sha != expected_input_sha:
        raise ValueError("prepared h5ad checksum does not match its manifest")
    resource_payload = _validate_resource(resource_path, resource_manifest)
    metadata, input_audit = _prepare_metadata(
        input_h5ad,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
        sample_unit_key=sample_unit_key,
        target=target,
        reference=reference,
    )
    return metadata, {
        "input": {
            "filename": input_h5ad.name,
            "sha256": actual_input_sha,
            "manifest_filename": input_manifest.name,
            "manifest_sha256": sha256_file(input_manifest),
            **input_audit,
        },
        "resource": {
            "resource_id": resource_payload["resource_id"],
            "filename": resource_path.name,
            "sha256": RESOURCE_SHA256,
            "rows": RESOURCE_ROWS,
            "manifest_sha256": sha256_file(resource_manifest),
        },
        "environment": _r_environment(rscript),
    }


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    input_manifest: Path,
    resource_path: Path,
    resource_manifest: Path,
    rscript: Path,
    python_executable: Path,
    dataset_id: str,
    scenario: str,
    cell_type_key: str,
    condition_key: str,
    sample_unit_key: str,
    target: str,
    reference: str,
    cores: int,
    nrep: int,
    min_cells: int,
    preflight_only: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"unsupported scenario: {scenario}")
    if target == reference:
        raise ValueError("target and reference must differ")
    for name, value in (("cores", cores), ("nrep", nrep), ("min_cells", min_cells)):
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metadata, audit = preflight(
        input_h5ad,
        input_manifest,
        resource_path,
        resource_manifest,
        rscript,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
        sample_unit_key=sample_unit_key,
        target=target,
        reference=reference,
    )
    units = audit["input"]["units_by_condition"]
    if scenario == "multi-sample" and min(units.values()) < 4:
        raise ValueError("scSeqCommDiff multi-sample inference needs >=4 units/group")
    source_metadata = metadata
    if scenario == "multi-sample":
        metadata, cell_type_support, eligibility_audit = _filter_multi_sample_metadata(
            source_metadata,
            target=target,
            reference=reference,
        )
    else:
        metadata, cell_type_support, eligibility_audit = (
            _filter_multi_condition_metadata(
                source_metadata,
                target=target,
                reference=reference,
            )
        )
        # Preserve the original audit shape for already compatible cohorts.
        if not eligibility_audit["excluded_cell_types"]:
            cell_type_support = _unfiltered_cell_type_support(source_metadata)
            eligibility_audit = {
                "policy": "not_applicable_multi_condition",
                "source_cell_types": sorted(source_metadata["Cluster_ID"].unique()),
                "analysis_cell_types": sorted(source_metadata["Cluster_ID"].unique()),
                "excluded_cell_types": [],
                "source_cells": len(source_metadata),
                "analysis_cells": len(source_metadata),
            }
    metadata_path = output_dir / "scseqcommdiff_metadata.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=False)
    support_path = output_dir / "scseqcommdiff_pseudobulk_support.tsv"
    cell_type_support.to_csv(support_path, sep="\t", index=False)
    protocol = {
        "scenario": scenario,
        "contrast_order": [target, reference],
        "intercellular_score": "scSeqComm",
        "lr_resource": RESOURCE_ID,
        "permutations": nrep if scenario == "multi-condition" else None,
        "intercellular_test": (
            "condition-label permutation"
            if scenario == "multi-condition"
            else "Wilcoxon"
        ),
        "intracellular_test": (
            "Wilcoxon" if scenario == "multi-condition" else "pseudo-Wilcoxon"
        ),
        "multiple_testing": (
            "BH q<0.05" if scenario == "multi-condition" else "raw p<0.05"
        ),
        "intracellular_gate": "max_S_intra>0.5_or_all_NA",
        "tf_target_prior": "TRRUSTv2+HTRIdb+RegNetwork_high",
        "receptor_tf_prior": "KEGG_human",
        "backend": "doMC",
        "cores": cores,
        "min_cells": min_cells,
        "seed": 20260717,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight_complete" if preflight_only else "running",
        "dataset_id": dataset_id,
        "method": {
            "id": "scseqcommdiff",
            "version": METHOD_VERSION,
            "paper_neighbor_commit": PAPER_NEIGHBOR_COMMIT,
            "zenodo_record": ZENODO_RECORD,
            "zenodo_archive_sha256": ZENODO_ARCHIVE_SHA256,
            "version_note": (
                "Zenodo DESCRIPTION says 1.0.0; Git commit 5a29240 changes the "
                "version to 2.0.0 while preserving the submitted R implementation."
            ),
        },
        "protocol": protocol,
        "preflight": audit,
        "metadata": {
            "filename": metadata_path.name,
            "sha256": sha256_file(metadata_path),
            "rows": len(metadata),
            "source_rows": len(source_metadata),
            "cell_type_eligibility": eligibility_audit,
            "pseudobulk_support_filename": support_path.name,
            "pseudobulk_support_sha256": sha256_file(support_path),
            "pseudobulk_support_rows": len(cell_type_support),
        },
        "formal_p_or_q_emitted": True,
        "failure": None,
    }
    if preflight_only:
        manifest["elapsed_seconds"] = time.time() - started
        write_json(output_dir / "preflight_manifest.json", manifest)
        return manifest
    if not python_executable.is_file():
        raise FileNotFoundError(python_executable)
    r_runner = Path(__file__).with_name("run.R")
    command = [
        str(rscript),
        str(r_runner),
        str(input_h5ad),
        str(metadata_path),
        str(resource_path),
        str(output_dir),
        scenario,
        target,
        reference,
        dataset_id,
        str(cores),
        str(nrep),
        str(min_cells),
        str(support_path),
    ]
    manifest["command"] = command
    write_json(output_dir / "run_manifest.json", manifest)
    environment = os.environ.copy()
    environment.update(
        {
            "RETICULATE_PYTHON": str(python_executable),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        manifest["status"] = "failed"
        manifest["returncode"] = completed.returncode
        manifest["elapsed_seconds"] = time.time() - started
        manifest["failure"] = _external_process_failure(completed)
        write_json(output_dir / "run_manifest.json", manifest)
        raise RuntimeError("scSeqCommDiff R runner failed; inspect stderr.log")
    outputs: dict[str, Any] = {}
    for filename in (
        "differential_comm.rds",
        "selected_differential_interactions.tsv.gz",
        "condition_cell_pair_rankings.tsv",
        "session_info.txt",
    ):
        path = output_dir / filename
        if not path.is_file():
            raise RuntimeError(f"scSeqCommDiff output is missing: {filename}")
        outputs[filename] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest.update(
        {
            "status": "complete",
            "returncode": 0,
            "elapsed_seconds": time.time() - started,
            "outputs": outputs,
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--resource-manifest", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scenario", choices=sorted(VALID_SCENARIOS), required=True)
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--condition-key", required=True)
    parser.add_argument("--sample-unit-key", default="sample_id")
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--nrep", type=int, default=1000)
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.input_h5ad,
        args.output_dir,
        input_manifest=args.input_manifest,
        resource_path=args.resource,
        resource_manifest=args.resource_manifest,
        rscript=args.rscript,
        python_executable=args.python_executable,
        dataset_id=args.dataset_id,
        scenario=args.scenario,
        cell_type_key=args.cell_type_key,
        condition_key=args.condition_key,
        sample_unit_key=args.sample_unit_key,
        target=args.target,
        reference=args.reference,
        cores=args.cores,
        nrep=args.nrep,
        min_cells=args.min_cells,
        preflight_only=args.preflight_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
