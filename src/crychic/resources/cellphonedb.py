"""Adapter for CellPhoneDB v5 readback Parquet tables."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from crychic.core import ContractError, FeatureUnavailableError

from .contracts import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    build_interaction_id,
)
from .manifest import (
    ResourceIntegrityError,
    ResourceManifest,
    sha256_file,
)

_RESOURCE_ID = "cellphonedb"
_VERSION = "v5.0.0"
_LICENSE = "MIT"
_CITATION = (
    "Garcia-Alonso L et al. Mapping the temporal and spatial dynamics of the "
    "human endometrium in vivo and in vitro. Nature Genetics (2021); "
    "CellPhoneDB v5 resource."
)
_REQUIRED = (
    "complex_composition.parquet",
    "complex_expanded.parquet",
    "genes.parquet",
    "interactions.parquet",
)


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FeatureUnavailableError(
            "CellPhoneDB Parquet loading requires pandas and a Parquet engine",
            code="missing_optional_dependency",
            field="pandas",
            remediation="Install the resource adapter dependencies",
        ) from exc
    return pd


def _read_parquet(path: Path) -> Any:
    pd = _pandas()
    try:
        return pd.read_parquet(path)
    except (ImportError, OSError, ValueError) as exc:
        raise FeatureUnavailableError(
            f"Cannot read CellPhoneDB Parquet table {path.name}",
            code="unreadable_parquet_resource",
            field="database_root",
            remediation="Install pyarrow and provide the v5 readback tables",
        ) from exc


def _text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(math.isnan(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none"} else text


def _verify_native_manifest(resource_dir: Path) -> str:
    path = resource_dir / "manifest.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        file_records = raw["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ResourceIntegrityError(
            "CellPhoneDB native manifest is missing or invalid",
            code="invalid_resource_manifest",
            field="manifest.json",
            remediation="Use the manifest generated with the v5 readback export",
        ) from exc
    checksums = {
        str(record.get("path")): str(record.get("sha256", "")).lower()
        for record in file_records
        if isinstance(record, dict)
    }
    for name in _REQUIRED:
        expected = checksums.get(name)
        if expected is None:
            raise ResourceIntegrityError(
                f"CellPhoneDB manifest does not pin {name}",
                code="unpinned_resource_payload",
                field="manifest.json",
                remediation="Regenerate all readback files and their manifest together",
            )
        payload = resource_dir / name
        if not payload.is_file():
            raise ResourceIntegrityError(
                f"CellPhoneDB payload is missing: {name}",
                code="missing_resource_payload",
                field="database_root",
                remediation="Restore the checksum-pinned v5 resource export",
            )
        if sha256_file(payload) != expected:
            raise ResourceIntegrityError(
                f"Resource checksum mismatch for cellphonedb/{_VERSION}/{name}",
                code="resource_checksum_mismatch",
                field="sha256",
                remediation="Restore the unmodified checksum-pinned payload",
            )
    return sha256_file(path)


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def load_cellphonedb_resource(
    database_root: str | Path,
    *,
    version: str = _VERSION,
    manifest_path: str | Path | None = None,
    strict_mapping: bool = True,
) -> ResourceBundle:
    """Load CellPhoneDB v5 and expand protein/complex multidata IDs to HGNC."""

    if version != _VERSION:
        raise ContractError(
            f"Unsupported CellPhoneDB version {version!r}",
            code="unsupported_resource_version",
            field="version",
            remediation=f"Use the implemented frozen release {_VERSION}",
        )
    root = Path(database_root)
    relative_dir = f"cellphonedb/{version}"
    resource_dir = root / "cellphonedb" / version
    native_digest = _verify_native_manifest(resource_dir)
    source_files = tuple(f"{relative_dir}/{name}" for name in _REQUIRED)
    manifest_digest = native_digest
    if manifest_path is not None:
        manifest = ResourceManifest.from_json(manifest_path)
        manifest.require(
            species=Species.HUMAN.value,
            gene_namespace=GeneNamespace.HGNC_SYMBOL.value,
            license=_LICENSE,
        )
        manifest.verify(root, paths=source_files)
        manifest_digest = manifest.digest

    genes = _read_parquet(resource_dir / "genes.parquet")
    composition = _read_parquet(resource_dir / "complex_composition.parquet")
    expanded = _read_parquet(resource_dir / "complex_expanded.parquet")
    interaction_rows = _read_parquet(resource_dir / "interactions.parquet")

    protein_symbols: dict[int, set[str]] = defaultdict(set)
    for row in genes.itertuples(index=False):
        symbol = _text(row.hgnc_symbol)
        if symbol is not None:
            protein_symbols[int(row.protein_multidata_id)].add(symbol)
    ambiguous = tuple(
        f"protein_multidata_id={identifier}:{'|'.join(sorted(symbols))}"
        for identifier, symbols in protein_symbols.items()
        if len(symbols) != 1
    )
    protein_map = {
        identifier: next(iter(symbols))
        for identifier, symbols in protein_symbols.items()
        if len(symbols) == 1
    }
    complex_names = {
        int(row.complex_multidata_id): str(row.name)
        for row in expanded.itertuples(index=False)
    }
    complex_map: dict[int, tuple[str, ...]] = {}
    unresolved: set[str] = set()
    grouped = composition.groupby("complex_multidata_id", sort=True)
    for complex_id, group in grouped:
        symbols: list[str] = []
        for protein_id in group["protein_multidata_id"]:
            symbol = protein_map.get(int(protein_id))
            if symbol is None:
                unresolved.add(f"protein_multidata_id={int(protein_id)}")
            else:
                symbols.append(symbol)
        if symbols:
            complex_map[int(complex_id)] = tuple(sorted(set(symbols)))

    def resolve(identifier: Any) -> tuple[str, ...] | None:
        key = int(identifier)
        if key in complex_map:
            return complex_map[key]
        symbol = protein_map.get(key)
        return None if symbol is None else (symbol,)

    interactions: list[Interaction] = []
    skipped: list[str] = []
    direction_counts: Counter[str] = Counter()
    swaps = 0
    for row in interaction_rows.itertuples(index=False):
        source_id = str(row.id_cp_interaction)
        first = resolve(row.multidata_1_id)
        second = resolve(row.multidata_2_id)
        if first is None or second is None:
            if first is None:
                unresolved.add(f"multidata_id={int(row.multidata_1_id)}")
            if second is None:
                unresolved.add(f"multidata_id={int(row.multidata_2_id)}")
            skipped.append(source_id)
            continue
        direction = _text(row.directionality) or "unspecified"
        direction_counts[direction] += 1
        first_name = complex_names.get(
            int(row.multidata_1_id), _text(row.name_1) or str(row.multidata_1_id)
        )
        second_name = complex_names.get(
            int(row.multidata_2_id), _text(row.name_2) or str(row.multidata_2_id)
        )
        swap = (
            direction == "Ligand-Receptor"
            and _bool(row.receptor_1)
            and not _bool(row.receptor_2)
        )
        if swap:
            swaps += 1
            ligand, receptor = second, first
            ligand_name = second_name
            receptor_name = first_name
            ligand_complex = _bool(row.is_complex_2)
            receptor_complex = _bool(row.is_complex_1)
        else:
            ligand, receptor = first, second
            ligand_name = first_name
            receptor_name = second_name
            ligand_complex = _bool(row.is_complex_1)
            receptor_complex = _bool(row.is_complex_2)
        interaction_id = build_interaction_id(
            resource_id=_RESOURCE_ID,
            version=version,
            species=Species.HUMAN,
            source_interaction_id=source_id,
            ligand_subunits=ligand,
            receptor_subunits=receptor,
            direction=direction,
        )
        source_text = _text(row.source)
        evidence = (
            ()
            if source_text is None
            else tuple(part.strip() for part in source_text.split(";") if part.strip())
        )
        metadata_values = {
            "annotation_strategy": _text(row.annotation_strategy),
            "curator": _text(row.curator),
            "orientation_swapped": str(swap).lower(),
        }
        metadata = tuple(
            (key, value)
            for key, value in metadata_values.items()
            if value is not None
        )
        classification = _text(row.classification)
        interactions.append(
            Interaction(
                interaction_id=interaction_id,
                source_interaction_id=source_id,
                ligand_name=ligand_name,
                receptor_name=receptor_name,
                ligand_subunits=ligand,
                receptor_subunits=receptor,
                ligand_is_complex=ligand_complex,
                receptor_is_complex=receptor_complex,
                direction=direction,
                source=_RESOURCE_ID,
                version=version,
                species=Species.HUMAN,
                gene_namespace=GeneNamespace.HGNC_SYMBOL,
                pathway=classification,
                annotation=_text(row.annotation_strategy),
                evidence=evidence,
                metadata=metadata,
            )
        )

    if strict_mapping and (unresolved or ambiguous or skipped):
        raise ContractError(
            "CellPhoneDB contains unresolved or ambiguous multidata mappings",
            code="resource_mapping_failure",
            field="multidata_id",
            remediation="Inspect the mapping report with strict_mapping=False",
        )
    notes = [f"orientation_swaps={swaps}"]
    notes.extend(
        f"direction[{direction}]={count}"
        for direction, count in sorted(direction_counts.items())
    )
    report = MappingReport(
        source_rows=len(interaction_rows),
        loaded_rows=len(interactions),
        mapped_entities=len(set(protein_map.values())),
        unmapped_entities=tuple(unresolved),
        ambiguous_entities=ambiguous,
        skipped_source_ids=tuple(skipped),
        notes=tuple(notes),
    )
    return ResourceBundle(
        resource_id=_RESOURCE_ID,
        version=version,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=report,
        manifest_digest=manifest_digest,
        source_files=source_files,
        license=_LICENSE,
        citation=_CITATION,
    )
