"""Derive a post-hoc all-method-observed MIS-C ligand sensitivity universe."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.adapters.common import json_safe, sha256_file
from benchmarks.comprehensive.evaluate_olink import (
    PREDICTION_COLUMNS,
    _read_predictions,
    ligand_universe_id,
)


def derive(
    predictions_path: Path,
    output_dir: Path,
    *,
    predictions_sha256: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    predictions_path = predictions_path.resolve()
    output_dir = output_dir.resolve()
    if sha256_file(predictions_path) != predictions_sha256:
        raise ValueError("source prediction checksum mismatch")
    predictions = _read_predictions(predictions_path)
    methods = tuple(sorted(predictions["method_id"].astype(str).unique()))
    observed = predictions["status"].isin({"observed", "ok"})
    support = observed.groupby(predictions["ligand"], observed=True).sum()
    complete_ligands = tuple(
        sorted(support.loc[support.eq(len(methods))].index.astype(str))
    )
    if not complete_ligands:
        raise ValueError("no ligand is observed by every method")
    result = predictions.loc[predictions["ligand"].isin(complete_ligands)].copy()
    expected_rows = len(methods) * len(complete_ligands)
    if (
        len(result) != expected_rows
        or not result["status"].isin({"observed", "ok"}).all()
    ):
        raise ValueError(
            "complete-case ligand universe is not rectangular and observed"
        )
    universe_id = ligand_universe_id(complete_ligands)
    result["universe_id"] = universe_id
    result = result.loc[:, PREDICTION_COLUMNS]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        output_path = staged / "complete_case_predictions.tsv"
        result.to_csv(output_path, sep="\t", index=False)
        manifest = {
            "schema_version": "crychic-misc-complete-case-sensitivity-v1",
            "status": "complete",
            "analysis_role": "post_hoc_method_support_sensitivity_not_primary",
            "selection_rule": "ligand has observed score for every included method",
            "selection_uses_olink_truth": False,
            "strict_prospective_blinding_claim": False,
            "source_predictions": {
                "filename": predictions_path.name,
                "sha256": predictions_sha256,
            },
            "methods": list(methods),
            "ligands": len(complete_ligands),
            "universe_id": universe_id,
            "output": {
                "filename": output_path.name,
                "rows": len(result),
                "sha256": sha256_file(output_path),
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
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = derive(
        args.predictions,
        args.output_dir,
        predictions_sha256=args.predictions_sha256,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
