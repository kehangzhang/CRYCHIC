"""Freeze MIS-C M-vs-S ligand predictions without opening Olink outcomes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.adapters.common import git_metadata, json_safe, sha256_file
from benchmarks.comprehensive.evaluate_olink import ligand_universe_id
from benchmarks.comprehensive.evaluate_three_group import (
    EDGE_KEYS,
    EXPECTED_METHOD_VERSIONS,
    RunRecord,
    _discover_runs,
    _external_table,
    _require_mapping,
    _target_crychic_view,
)
from benchmarks.metrics.multicondition import (
    external_long_to_score_table,
    unpaired_edge_effects,
)

METHODS = ("crychic", "scseqcommdiff", "cellchat", "liana_rank_aggregate")
TARGET = "MIS-C"
REFERENCE = "healthy_sibling"
CONTRAST = "misc_m_vs_s"
PREDICTION_COLUMNS = (
    "dataset_id",
    "method_id",
    "resource_mode",
    "universe_id",
    "contrast",
    "ligand",
    "score",
    "status",
)


def _json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _external_effects(record: RunRecord) -> pd.DataFrame:
    table = _external_table(record)
    if record.method == "crychic":
        table = _target_crychic_view(table, record.manifest, target=TARGET)
    mapped = external_long_to_score_table(
        table,
        context_key="condition",
        contrast=CONTRAST,
        dataset="misc_olink",
    )
    effects = unpaired_edge_effects(
        mapped,
        reference=REFERENCE,
        target=TARGET,
        min_subjects=4,
        contrast=CONTRAST,
        validated=True,
    )
    return effects.loc[:, [*EDGE_KEYS, "effect", "status", "reason_code"]]


def _scseq_effects(record: RunRecord, resource: pd.DataFrame) -> pd.DataFrame:
    outputs = record.manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("scSeqCommDiff output manifest is absent")
    output = outputs.get("differential_event_scores.tsv.gz")
    if not isinstance(output, Mapping):
        raise ValueError("scSeqCommDiff event score output is absent")
    path = record.directory / "differential_event_scores.tsv.gz"
    if sha256_file(path) != output.get("sha256"):
        raise ValueError("scSeqCommDiff event score checksum mismatch")
    table = pd.read_csv(path, sep="\t").rename(
        columns={"cluster_L": "sender", "cluster_R": "receiver"}
    )
    interaction_map = resource.loc[
        :, ["ligand", "receptor", "harmonized_interaction_id"]
    ].rename(columns={"harmonized_interaction_id": "interaction_id"})
    table = table.merge(
        interaction_map,
        on=["ligand", "receptor"],
        how="inner",
        validate="many_to_one",
    )
    table["effect"] = pd.to_numeric(table["effect"], errors="coerce")
    return table.loc[:, [*EDGE_KEYS, "effect", "status", "reason_code"]]


def _event_universe(resource: pd.DataFrame, cell_types: Sequence[str]) -> pd.DataFrame:
    records = [
        {
            "sender": sender,
            "receiver": receiver,
            "interaction_id": interaction.harmonized_interaction_id,
            "ligand": interaction.ligand,
            "receptor": interaction.receptor,
        }
        for sender in sorted(set(map(str, cell_types)))
        for receiver in sorted(set(map(str, cell_types)))
        for interaction in resource.itertuples(index=False)
    ]
    return pd.DataFrame.from_records(records, columns=EDGE_KEYS)


def _materialize_effects(
    effects: pd.DataFrame, universe: pd.DataFrame, *, method: str
) -> pd.DataFrame:
    if effects.duplicated(list(EDGE_KEYS)).any():
        raise ValueError(f"{method} contains duplicate event effects")
    result = universe.merge(
        effects,
        on=list(EDGE_KEYS),
        how="left",
        validate="one_to_one",
    )
    result["status"] = result["status"].astype("string")
    result["reason_code"] = result["reason_code"].astype("string")
    missing = result["status"].isna()
    result.loc[missing, "status"] = "not_estimable"
    result.loc[missing, "reason_code"] = "event_not_returned"
    result.loc[~missing & result["reason_code"].isna(), "reason_code"] = ""
    result["method_id"] = method
    return result


def _aggregate_ligands(events: pd.DataFrame, ligands: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ligand in sorted(set(map(str, ligands))):
        group = events.loc[events["ligand"].astype(str).eq(ligand)]
        observed = (
            group["status"].isin({"observed", "exploratory"})
            & pd.to_numeric(group["effect"], errors="coerce").notna()
        )
        values = pd.to_numeric(group.loc[observed, "effect"], errors="coerce")
        rows.append(
            {
                "ligand": ligand,
                "score": float(values.max()) if len(values) else np.nan,
                "status": "observed" if len(values) else "not_estimable",
                "n_observed_events": len(values),
                "n_total_events": len(group),
            }
        )
    return pd.DataFrame.from_records(rows)


def _one_record(
    records: Sequence[RunRecord], *, method: str, contrast: str | None
) -> RunRecord:
    matches = [
        record
        for record in records
        if record.method == method
        and record.dataset_id == "misc_olink"
        and record.contrast == contrast
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method} MIS-C run for contrast={contrast!r}: "
            f"{[record.directory.name for record in matches]}"
        )
    return matches[0]


def _validate_misc_run_binding(
    record: RunRecord,
    *,
    expected_input_sha256: str,
    expected_resource_sha256: str,
    expected_code_commit: str,
) -> None:
    method = _require_mapping(
        record.manifest.get("method"), label=f"{record.method} method provenance"
    )
    if method.get("id") != record.method:
        raise ValueError(f"method identity mismatch for {record.directory}")
    expected_version = EXPECTED_METHOD_VERSIONS.get(record.method)
    if expected_version is not None and method.get("version") != expected_version:
        raise ValueError(f"method version mismatch for {record.directory}")
    code = _require_mapping(
        record.manifest.get("code"), label=f"{record.method} code provenance"
    )
    if code.get("dirty") is not False or code.get("commit") != expected_code_commit:
        raise ValueError(f"code provenance mismatch for {record.directory}")
    if record.kind == "scseq":
        preflight = _require_mapping(
            record.manifest.get("preflight"), label="scSeqCommDiff preflight"
        )
        input_record = _require_mapping(
            preflight.get("input"), label="scSeqCommDiff input provenance"
        )
        resource = _require_mapping(
            preflight.get("resource"), label="scSeqCommDiff resource provenance"
        )
    else:
        input_record = _require_mapping(
            record.manifest.get("input"), label=f"{record.method} input provenance"
        )
        resource = _require_mapping(
            record.manifest.get("resource"),
            label=f"{record.method} resource provenance",
        )
    resource_sha = resource.get("sha256")
    if resource_sha is None:
        resource_sha = resource.get("table_sha256", resource.get("payload_sha256"))
    if (
        input_record.get("sha256") != expected_input_sha256
        or resource.get("mode") != "H-common"
        or resource_sha != expected_resource_sha256
    ):
        raise ValueError(f"input/resource binding mismatch for {record.directory}")

    if record.method == "crychic":
        algorithm_code = _require_mapping(
            method.get("algorithm_code"), label="CRYCHIC algorithm provenance"
        )
        parameters = _require_mapping(
            record.manifest.get("parameters"), label="CRYCHIC parameters"
        )
        workflow = _require_mapping(
            parameters.get("workflow"), label="CRYCHIC workflow"
        )
        if (
            method.get("benchmark_identity") != "generic_multigroup_baseline"
            or method.get("entrypoint")
            != "benchmarks.adapters.crychic.run_hcommon"
            or algorithm_code.get("dirty") is not False
            or algorithm_code.get("commit") != expected_code_commit
            or parameters.get("benchmark_scope")
            != "independent_subject_multigroup_hcommon"
            or parameters.get("input_mode") != "counts"
            or parameters.get("counts_layer") != "counts"
            or workflow.get("subject_fixed_effects") is not False
            or workflow.get("min_cells") != 10
        ):
            raise ValueError("CRYCHIC MIS-C protocol mismatch")
    elif record.method == "cellchat":
        parameters = _require_mapping(
            record.manifest.get("parameters"), label="CellChat parameters"
        )
        if (
            parameters.get("layer") is not None
            or parameters.get("nboot") != 100
            or parameters.get("min_cells") != 10
        ):
            raise ValueError("CellChat MIS-C protocol mismatch")
    elif record.method == "liana_rank_aggregate":
        parameters = _require_mapping(
            record.manifest.get("parameters"), label="LIANA parameters"
        )
        if (
            parameters.get("layer") is not None
            or parameters.get("n_perms_per_sample") != 100
            or parameters.get("min_cells") != 10
        ):
            raise ValueError("LIANA MIS-C protocol mismatch")
    else:
        protocol = _require_mapping(
            record.manifest.get("protocol"), label="scSeqCommDiff protocol"
        )
        if (
            protocol.get("scenario") != "multi-sample"
            or protocol.get("resource_mode") != "H-common"
            or protocol.get("min_cells") != 30
        ):
            raise ValueError("scSeqCommDiff MIS-C protocol mismatch")


def freeze_predictions(
    prepared_dir: Path,
    runs_dir: Path,
    input_h5ad: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    prepared_dir = prepared_dir.resolve()
    runs_dir = runs_dir.resolve()
    input_h5ad = input_h5ad.resolve()
    output_dir = output_dir.resolve()
    preparation = _json_object(prepared_dir / "manifest.json")
    preparation_code = _require_mapping(
        preparation.get("code"), label="MIS-C preparation code provenance"
    )
    expected_code_commit = str(preparation_code.get("commit", ""))
    export_code = git_metadata(Path(__file__).resolve().parents[2])
    if (
        preparation_code.get("dirty") is not False
        or not expected_code_commit
        or export_code.get("dirty") is not False
        or export_code.get("commit") != expected_code_commit
    ):
        raise ValueError("MIS-C export requires one clean frozen code commit")
    source_record = _require_mapping(
        preparation.get("source_h5ad"), label="prepared MIS-C source H5AD"
    )
    if sha256_file(input_h5ad) != source_record.get("sha256"):
        raise ValueError("MIS-C source H5AD checksum mismatch")
    resource_path = prepared_dir / str(preparation["resource"]["table"])
    if sha256_file(resource_path) != preparation["resource"]["sha256"]:
        raise ValueError("prepared common resource checksum mismatch")
    resource = pd.read_csv(resource_path, sep="\t", dtype=str)
    pair_records = {
        str(item["contrast"]): item
        for item in preparation.get("contrasts", [])
        if isinstance(item, Mapping)
    }
    source = ad.read_h5ad(input_h5ad, backed="r")
    try:
        cell_types = tuple(sorted(source.obs["cell_type"].astype(str).unique()))
    finally:
        source.file.close()
    universe = _event_universe(resource, cell_types)
    records, excluded = _discover_runs(runs_dir)
    event_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    ligand_id = ligand_universe_id(resource["ligand"].astype(str).tolist())
    run_records: list[dict[str, Any]] = []
    for method in METHODS:
        record = _one_record(
            records,
            method=method,
            contrast=f"{TARGET}_vs_{REFERENCE}" if method == "scseqcommdiff" else None,
        )
        if method == "scseqcommdiff":
            pair_record = _require_mapping(
                pair_records.get("M_vs_S"), label="prepared M_vs_S input"
            )
            expected_input_sha256 = str(pair_record["sha256"])
        else:
            expected_input_sha256 = str(source_record["sha256"])
        _validate_misc_run_binding(
            record,
            expected_input_sha256=expected_input_sha256,
            expected_resource_sha256=str(preparation["resource"]["sha256"]),
            expected_code_commit=expected_code_commit,
        )
        effects = (
            _scseq_effects(record, resource)
            if method == "scseqcommdiff"
            else _external_effects(record)
        )
        materialized = _materialize_effects(effects, universe, method=method)
        event_frames.append(materialized)
        ligands = _aggregate_ligands(materialized, resource["ligand"])
        predictions = pd.DataFrame(
            {
                "dataset_id": "misc_olink",
                "method_id": method,
                "resource_mode": "H-common",
                "universe_id": ligand_id,
                "contrast": CONTRAST,
                "ligand": ligands["ligand"],
                "score": ligands["score"],
                "status": ligands["status"],
            }
        )
        prediction_frames.append(predictions)
        run_records.append(
            {
                "method_id": method,
                "run_directory": record.directory.name,
                "run_manifest_sha256": sha256_file(
                    record.directory
                    / (
                        "run_manifest.json"
                        if record.kind == "scseq"
                        else "manifest.json"
                    )
                ),
                "observed_events": int(
                    materialized["status"].isin({"observed", "exploratory"}).sum()
                ),
                "observed_ligands": int(ligands["status"].eq("observed").sum()),
            }
        )
    prediction_table = pd.concat(prediction_frames, ignore_index=True).loc[
        :, PREDICTION_COLUMNS
    ]
    events = pd.concat(event_frames, ignore_index=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        predictions_path = staged / "olink_blind_predictions.tsv"
        events_path = staged / "misc_m_vs_s_event_effects.tsv.gz"
        prediction_table.to_csv(predictions_path, sep="\t", index=False)
        events.to_csv(events_path, sep="\t", index=False)
        manifest = {
            "schema_version": "crychic-misc-olink-predictions-v1",
            "status": "frozen",
            "dataset_id": "misc_olink",
            "contrast": CONTRAST,
            "target": TARGET,
            "reference": REFERENCE,
            "resource_mode": "H-common",
            "event_universe_size": len(universe),
            "ligand_universe_size": int(resource["ligand"].nunique()),
            "ligand_universe_id": ligand_id,
            "effect_aggregation": (
                "maximum signed target-minus-reference event effect per ligand"
            ),
            "missing_policy": "not_estimable_never_zero_imputed",
            "olink_truth_opened_by_this_entrypoint": False,
            "strict_prospective_blinding_claim": False,
            "strict_blinding_reason": "truth_preexisted_and_was_previously_inspected",
            "prepared_manifest_sha256": sha256_file(prepared_dir / "manifest.json"),
            "code": export_code,
            "export_script_sha256": sha256_file(Path(__file__)),
            "excluded_runs": excluded,
            "runs": run_records,
            "outputs": {
                "predictions": {
                    "filename": predictions_path.name,
                    "rows": len(prediction_table),
                    "sha256": sha256_file(predictions_path),
                },
                "event_effects": {
                    "filename": events_path.name,
                    "rows": len(events),
                    "sha256": sha256_file(events_path),
                },
            },
        }
        (staged / "manifest.json").write_text(
            json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        if output_dir.exists() and not overwrite:
            raise FileExistsError(f"output exists: {output_dir}; pass --overwrite")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staged, output_dir)
        published = True
        return manifest
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--input-h5ad", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = freeze_predictions(
        args.prepared_dir,
        args.runs_dir,
        args.input_h5ad,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
