"""Adapter for derived sparse NicheNet ligand-target Parquet resources."""

from __future__ import annotations

import math
from pathlib import Path, PurePath
from typing import Any

from crychic.core import ContractError, FeatureUnavailableError

from .contracts import GeneNamespace, MappingReport, Species, TargetPrior
from .manifest import ResourceIntegrityError, ResourceManifest

_DEFAULT_RELEASE = "v2_2021"
_DEFAULT_FILE = "ligand_target_top250.parquet"


def _read_parquet(path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FeatureUnavailableError(
            "NicheNet prior loading requires pandas and a Parquet engine",
            code="missing_optional_dependency",
            field="pandas",
            remediation="Install the resource adapter dependencies",
        ) from exc
    try:
        return pd.read_parquet(path)
    except (ImportError, OSError, ValueError) as exc:
        raise FeatureUnavailableError(
            f"Cannot read NicheNet derived prior {path.name}",
            code="unreadable_parquet_resource",
            field="database_root",
            remediation=(
                "Install pyarrow and generate the checksum-pinned derived Parquet"
            ),
        ) from exc


def load_nichenet_target_prior(
    database_root: str | Path,
    *,
    release: str = _DEFAULT_RELEASE,
    derived_filename: str = _DEFAULT_FILE,
    manifest_path: str | Path | None = None,
) -> TargetPrior:
    """Load a positive NicheNet target-by-ligand sparse prior.

    The runtime adapter intentionally consumes the deterministic derived
    Parquet rather than requiring R to deserialize the upstream dense RDS.
    """

    if PurePath(derived_filename).name != derived_filename:
        raise ResourceIntegrityError(
            "NicheNet derived_filename must be a basename",
            code="invalid_resource_path",
            field="derived_filename",
            remediation="Select a file within the frozen NicheNet release directory",
        )
    root = Path(database_root)
    resource_dir = root / "nichenet" / release
    native_manifest_path = (
        resource_dir / "manifest.json" if manifest_path is None else Path(manifest_path)
    )
    manifest = ResourceManifest.from_json(native_manifest_path)
    manifest.require(
        species=Species.HUMAN.value,
        gene_namespace=GeneNamespace.HGNC_SYMBOL.value,
        license="CC-BY-4.0",
    )
    if manifest_path is None:
        manifest.verify(resource_dir, paths=(derived_filename,))
        source_file = f"nichenet/{release}/{derived_filename}"
    else:
        source_file = f"nichenet/{release}/{derived_filename}"
        manifest.verify(root, paths=(source_file,))
    table = _read_parquet(resource_dir / derived_filename)
    required = {"ligand", "target", "weight"}
    missing = required.difference(table.columns)
    if missing:
        raise ContractError(
            f"NicheNet prior lacks columns: {', '.join(sorted(missing))}",
            code="invalid_target_prior_table",
            field="columns",
            remediation="Regenerate the long-form ligand,target,weight Parquet",
        )
    if table[list(required)].isna().any().any():
        raise ContractError(
            "NicheNet prior contains null ligand, target, or weight values",
            code="invalid_target_prior_table",
            field="nulls",
            remediation="Remove invalid links during deterministic derivation",
        )
    if table.duplicated(["ligand", "target"]).any():
        raise ContractError(
            "NicheNet prior contains duplicate ligand-target links",
            code="duplicate_target_prior_link",
            field="ligand,target",
            remediation="Aggregate or deterministically select one weight per link",
        )
    weights = table["weight"].astype(float)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ContractError(
            "NicheNet prior weights must be finite and non-negative",
            code="invalid_prior_weight",
            field="weight",
            remediation="Regenerate the derived positive target prior",
        )
    has_rank = "rank" in table.columns
    sort_columns = ["ligand", *(("rank",) if has_rank else ()), "target"]
    ordered = table.sort_values(sort_columns, kind="stable", ignore_index=True)
    driver_ids = tuple(sorted(map(str, ordered["ligand"].unique())))
    target_ids = tuple(sorted(map(str, ordered["target"].unique())))
    target_index = {target: index for index, target in enumerate(target_ids)}
    target_indices: list[int] = []
    prior_weights: list[float] = []
    ranks: list[int] = []
    indptr = [0]
    for driver in driver_ids:
        group = ordered.loc[ordered["ligand"] == driver]
        target_indices.extend(target_index[str(target)] for target in group["target"])
        prior_weights.extend(float(weight) for weight in group["weight"])
        if has_rank:
            ranks.extend(int(rank) for rank in group["rank"])
        indptr.append(len(target_indices))
    report = MappingReport(
        source_rows=len(table),
        loaded_rows=len(table),
        mapped_entities=len(driver_ids) + len(target_ids),
        notes=(
            f"derived_payload={source_file}",
            "positive_regulatory_potential",
        ),
    )
    return TargetPrior(
        resource_id=manifest.resource_id,
        version=manifest.version,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=target_ids,
        driver_ids=driver_ids,
        indptr=tuple(indptr),
        target_indices=tuple(target_indices),
        weights=tuple(prior_weights),
        ranks=tuple(ranks) if has_rank else None,
        direction=1,
        evidence="NicheNet regulatory potential",
        mapping_report=report,
        manifest_digest=manifest.digest,
    )
