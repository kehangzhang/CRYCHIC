"""Run CellPhoneDB v5 independently for each biological sample."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import time
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.adapters.common import (
    begin_manifest,
    cell_type_support,
    fail_manifest,
    finalize_manifest,
    load_harmonized_resource,
    materialize_fixed_universe,
    method_frozen_resource,
    prepare_output,
    python_environment,
    sha256_file,
    validate_prepared_input,
)
from benchmarks.adapters.resource_tables import native_lr_resource

METHOD_ID = "cellphonedb"


def _filtered_database(
    source_zip: Path,
    output_zip: Path,
    *,
    source_interaction_ids: set[str],
) -> None:
    """Create a valid CPDB archive with only the frozen interaction rows."""
    with (
        zipfile.ZipFile(source_zip) as source,
        zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename == "interaction_table.csv":
                interactions = pd.read_csv(io.BytesIO(payload))
                interactions = interactions.loc[
                    interactions["id_cp_interaction"]
                    .astype(str)
                    .isin(source_interaction_ids)
                ]
                if interactions.empty:
                    raise ValueError(
                        "selected CellPhoneDB resource has no interactions"
                    )
                payload = interactions.to_csv(index=False).encode("utf-8")
            deterministic = zipfile.ZipInfo(
                filename=member.filename,
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            deterministic.compress_type = zipfile.ZIP_DEFLATED
            deterministic.external_attr = member.external_attr
            target.writestr(deterministic, payload)


def _empty_observed() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id",
            "sender",
            "receiver",
            "interaction_id",
            "target",
            "score",
            "specificity_score",
            "within_dataset_p_value",
            "within_dataset_p_value_semantics",
        ]
    )


def _melt_matrix(
    table: pd.DataFrame,
    *,
    value_name: str,
    separator: str,
) -> pd.DataFrame:
    pair_columns = [column for column in table.columns if separator in str(column)]
    if not pair_columns:
        return pd.DataFrame(
            columns=["id_cp_interaction", "sender", "receiver", value_name]
        )
    if "id_cp_interaction" not in table:
        raise RuntimeError("CellPhoneDB output lacks id_cp_interaction")
    result = table[["id_cp_interaction", *pair_columns]].melt(
        id_vars="id_cp_interaction",
        var_name="cell_pair",
        value_name=value_name,
    )
    pairs = result["cell_pair"].astype(str).str.split(separator, n=1, expand=True)
    if pairs.shape[1] != 2:
        raise RuntimeError("CellPhoneDB cell-pair field cannot be split")
    result["sender"] = pairs[0]
    result["receiver"] = pairs[1]
    return result.drop(columns="cell_pair")


def _normalize_sample(
    result: dict[str, object],
    *,
    sample_id: str,
    source_map: pd.DataFrame,
    statistical: bool,
    separator: str,
) -> pd.DataFrame:
    score_table = result.get("interaction_scores")
    if not isinstance(score_table, pd.DataFrame):
        score_table = result.get("means") if statistical else result.get("means_result")
    if not isinstance(score_table, pd.DataFrame) or score_table.empty:
        return _empty_observed()
    score = _melt_matrix(score_table, value_name="score", separator=separator)
    pvalues = result.get("pvalues")
    if statistical and isinstance(pvalues, pd.DataFrame) and not pvalues.empty:
        pvalue = _melt_matrix(
            pvalues, value_name="within_dataset_p_value", separator=separator
        )
        score = score.merge(
            pvalue,
            on=["id_cp_interaction", "sender", "receiver"],
            how="left",
            validate="one_to_one",
        )
    else:
        score["within_dataset_p_value"] = np.nan
    score = score.merge(
        source_map[["source_interaction_id", "interaction_id"]],
        left_on="id_cp_interaction",
        right_on="source_interaction_id",
        how="inner",
        validate="many_to_one",
    )
    score["score"] = pd.to_numeric(score["score"], errors="coerce")
    score["within_dataset_p_value"] = pd.to_numeric(
        score["within_dataset_p_value"], errors="coerce"
    )
    score = score.loc[score["score"].notna()].copy()
    if score.empty:
        return _empty_observed()
    score = score.groupby(
        ["sender", "receiver", "interaction_id"],
        sort=True,
        observed=True,
        as_index=False,
    ).agg(
        score=("score", "max"),
        within_dataset_p_value=("within_dataset_p_value", "min"),
    )
    score["sample_id"] = sample_id
    score["target"] = pd.NA
    score["specificity_score"] = pd.NA
    score["within_dataset_p_value_semantics"] = (
        "CellPhoneDB cell-label permutation p-value; not a condition test"
        if statistical
        else "not_emitted"
    )
    return score


def _selected_expression(adata: ad.AnnData, *, layer: str | None) -> ad.AnnData:
    selected = adata.copy()
    if layer is not None:
        selected.X = selected.layers[layer].copy()
    selected.X = sparse.csr_matrix(selected.X)
    if selected.X.data.size and (
        not np.isfinite(selected.X.data).all() or (selected.X.data < 0).any()
    ):
        raise ValueError("CellPhoneDB expression must be finite and non-negative")
    return selected


def run(
    input_h5ad: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    database_root: Path,
    sample_key: str,
    subject_key: str,
    cell_type_key: str,
    context_keys: tuple[str, ...],
    resource_mode: str,
    harmonized_resource: Path | None,
    harmonized_manifest: Path | None,
    layer: str | None,
    min_cells: int,
    iterations: int,
    score_interactions: bool,
    threads: int,
    seed: int,
    separator: str,
    overwrite: bool,
) -> dict[str, object]:
    """Execute the CPDB Python API per sample with a frozen database archive."""
    from cellphonedb.src.core.methods import (
        cpdb_analysis_method,
        cpdb_statistical_analysis_method,
    )

    if 0 < iterations < 100:
        raise ValueError(
            "CellPhoneDB 5.0.1 statistical mode requires at least 100 iterations; "
            "smaller values trigger an upstream progress-step division by zero"
        )
    output_dir = prepare_output(output_dir, overwrite=overwrite)
    repo_root = Path(__file__).resolve().parents[3]
    adata = ad.read_h5ad(input_h5ad)
    sample_metadata = validate_prepared_input(
        adata,
        sample_key=sample_key,
        subject_key=subject_key,
        cell_type_key=cell_type_key,
        context_keys=context_keys,
    )
    support = cell_type_support(
        adata, sample_key=sample_key, cell_type_key=cell_type_key
    )
    method_version = importlib.metadata.version("cellphonedb")
    native_resource, native_map, native_metadata = native_lr_resource(
        method="cellphonedb",
        database_root=database_root,
        repo_root=repo_root,
    )
    database_zip = (
        database_root / "cellphonedb/v5.0.0/cellphonedb_07_12_2026_053629.zip"
    )
    if not database_zip.is_file():
        raise FileNotFoundError(
            f"CellPhoneDB database archive is missing: {database_zip}"
        )

    if resource_mode == "native":
        frozen_resource = native_resource
        source_map = native_map
        resource_id = native_metadata["resource_id"]
        resource_version = native_metadata["resource_version"]
        resource_payload: dict[str, object] = {
            "mode": resource_mode,
            **native_metadata,
            "database_filename": database_zip.name,
            "database_sha256": sha256_file(database_zip),
            "interactions": len(frozen_resource),
        }
        selected_zip = database_zip
    else:
        if harmonized_resource is None or harmonized_manifest is None:
            raise ValueError("harmonized arms require resource table and manifest")
        harmonized, frozen_manifest = load_harmonized_resource(
            harmonized_resource, harmonized_manifest
        )
        frozen_resource = method_frozen_resource(
            harmonized, method="cellphonedb", resource_mode=resource_mode
        )
        source_map = frozen_resource.loc[
            frozen_resource["method_covered"],
            ["native_interaction_id", "interaction_id"],
        ].rename(columns={"native_interaction_id": "source_interaction_id"})
        resource_id = str(frozen_manifest["resource_id"])
        resource_version = str(frozen_manifest["version"])
        selected_zip = output_dir / "cellphonedb_frozen_resource.zip"
        _filtered_database(
            database_zip,
            selected_zip,
            source_interaction_ids=set(
                source_map["source_interaction_id"].dropna().astype(str)
            ),
        )
        resource_payload = {
            "mode": resource_mode,
            "resource_id": resource_id,
            "version": resource_version,
            "payload_filename": harmonized_resource.name,
            "payload_sha256": sha256_file(harmonized_resource),
            "manifest_filename": harmonized_manifest.name,
            "manifest_sha256": sha256_file(harmonized_manifest),
            "generated_database_filename": selected_zip.name,
            "generated_database_sha256": sha256_file(selected_zip),
            "interactions": len(frozen_resource),
            "method_covered": int(frozen_resource["method_covered"].sum()),
        }

    statistical = iterations > 0
    score_name = "interaction_score" if score_interactions else "mean_expression"
    parameters = {
        "layer": layer,
        "threshold": 0.1,
        "min_cells_for_benchmark": min_cells,
        "iterations": iterations,
        "score_interactions": score_interactions,
        "threads": threads,
        "seed": seed,
        "separator": separator,
        "sample_level_execution": True,
    }
    manifest = begin_manifest(
        repo_root=repo_root,
        dataset_id=dataset_id,
        method={
            "id": METHOD_ID,
            "name": "CellPhoneDB",
            "version": method_version,
            "license": "MIT",
            "entrypoint": (
                "cpdb_statistical_analysis_method.call"
                if statistical
                else "cpdb_analysis_method.call"
            ),
        },
        environment=python_environment(
            environment_name="ov_2",
            packages=("cellphonedb", "anndata", "numpy", "pandas", "scipy"),
            threads=threads,
        ),
        input_path=input_h5ad,
        input_shape=adata.shape,
        sample_metadata=sample_metadata,
        input_keys={
            "sample": sample_key,
            "subject": subject_key,
            "cell_type": cell_type_key,
            "contexts": context_keys,
        },
        resource=resource_payload,
        parameters=parameters,
        score_semantics={
            "primary_score": score_name,
            "direction": "higher_is_stronger",
            "within_dataset_p_value": (
                "cell-label permutation specificity; not between-condition inference"
                if statistical
                else "not_emitted"
            ),
        },
    )
    started = time.perf_counter()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    observed: list[pd.DataFrame] = []
    sample_status: dict[str, str] = {}
    sample_failures: dict[str, dict[str, object]] = {}
    try:
        for ordinal, sample_id in enumerate(sample_metadata["sample_id"]):
            sample_id = str(sample_id)
            sample_output = raw_dir / f"sample_{ordinal:04d}"
            sample_output.mkdir()
            selected = _selected_expression(
                adata[adata.obs[sample_key].astype(str) == sample_id], layer=layer
            )
            meta_path = sample_output / "metadata.tsv"
            pd.DataFrame(
                {
                    "Cell": selected.obs_names.astype(str),
                    "cell_type": selected.obs[cell_type_key].astype(str).to_numpy(),
                }
            ).to_csv(meta_path, sep="\t", index=False)
            try:
                if statistical:
                    native_result = cpdb_statistical_analysis_method.call(
                        cpdb_file_path=str(selected_zip),
                        meta_file_path=str(meta_path),
                        counts_file_path=selected,
                        counts_data="hgnc_symbol",
                        output_path=str(sample_output),
                        iterations=iterations,
                        threshold=0.1,
                        threads=threads,
                        debug_seed=seed + ordinal,
                        separator=separator,
                        output_suffix=f"sample_{ordinal:04d}",
                        score_interactions=score_interactions,
                    )
                else:
                    native_result = cpdb_analysis_method.call(
                        cpdb_file_path=str(selected_zip),
                        meta_file_path=str(meta_path),
                        counts_file_path=selected,
                        counts_data="hgnc_symbol",
                        output_path=str(sample_output),
                        separator=separator,
                        threshold=0.1,
                        output_suffix=f"sample_{ordinal:04d}",
                        score_interactions=score_interactions,
                        threads=threads,
                    )
                observed.append(
                    _normalize_sample(
                        native_result,
                        sample_id=sample_id,
                        source_map=source_map,
                        statistical=statistical,
                        separator=separator,
                    )
                )
            except BaseException as exc:
                sample_status[sample_id] = "method_failed"
                sample_failures[sample_id] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        sparse_observed = (
            pd.concat(observed, ignore_index=True) if observed else _empty_observed()
        )
        table = materialize_fixed_universe(
            sparse_observed,
            sample_metadata=sample_metadata,
            support=support,
            resource=frozen_resource,
            dataset_id=dataset_id,
            run_id=str(manifest["run_id"]),
            method_id=METHOD_ID,
            method_version=method_version,
            analysis_track="lr_stlr",
            resource_mode=resource_mode,
            resource_id=resource_id,
            resource_version=resource_version,
            score_name=score_name,
            score_direction="higher",
            specificity_score_name=None,
            min_cells=min_cells,
            sample_status=sample_status,
        )
        manifest["sample_failures"] = sample_failures
        return finalize_manifest(manifest, table, output_dir, started=started)
    except BaseException as exc:
        fail_manifest(manifest, output_dir, exc, started=started)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--database-root", required=True, type=Path)
    parser.add_argument("--sample-key", default="sample_id")
    parser.add_argument("--subject-key", default="subject_id")
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--context-key", action="append", required=True)
    parser.add_argument(
        "--resource-mode", choices=("H-common", "H-covered", "native"), default="native"
    )
    parser.add_argument("--harmonized-resource", type=Path)
    parser.add_argument("--harmonized-manifest", type=Path)
    parser.add_argument("--layer")
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--no-interaction-scores", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--separator", default="|")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.input_h5ad,
        args.output_dir,
        dataset_id=args.dataset_id,
        database_root=args.database_root,
        sample_key=args.sample_key,
        subject_key=args.subject_key,
        cell_type_key=args.cell_type_key,
        context_keys=tuple(args.context_key),
        resource_mode=args.resource_mode,
        harmonized_resource=args.harmonized_resource,
        harmonized_manifest=args.harmonized_manifest,
        layer=args.layer,
        min_cells=args.min_cells,
        iterations=args.iterations,
        score_interactions=not args.no_interaction_scores,
        threads=args.threads,
        seed=args.seed,
        separator=args.separator,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
