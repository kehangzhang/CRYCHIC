"""Adapter for checksum-pinned CellChatDB CSV exports."""

from __future__ import annotations

import csv
from pathlib import Path

from crychic.core import ContractError

from .contracts import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    build_interaction_id,
)
from .manifest import ResourceManifest, sha256_file, verify_gnu_checksum_file

_RESOURCE_ID = "cellchatdb_v2"
_LICENSE = "GPL-3"
_CITATION = (
    "Jin S et al. Inference and analysis of cell-cell communication using "
    "CellChat. Nature Communications (2021)."
)


def _read_records(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ContractError(
            f"Cannot read CellChatDB table {path.name}",
            code="unreadable_resource_payload",
            field="database_root",
            remediation="Provide a complete checksum-pinned CellChatDB export",
        ) from exc


def _row_name(row: dict[str, str]) -> str:
    for key in ("", "Unnamed: 0"):
        value = row.get(key, "").strip()
        if value:
            return value
    raise ContractError(
        "CellChatDB component row is missing its entity name",
        code="invalid_cellchat_table",
        field="row_name",
        remediation="Export CellChatDB tables with row names preserved",
    )


def _component_map(path: Path, column_prefix: str) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for row in _read_records(path):
        name = _row_name(row)
        components = tuple(
            value.strip()
            for key, value in row.items()
            if key.startswith(column_prefix) and value and value.strip()
        )
        if not components:
            raise ContractError(
                f"CellChatDB entity {name!r} has no components",
                code="invalid_cellchat_component",
                field=path.name,
                remediation="Regenerate the split CSV from the frozen R object",
            )
        result[name] = tuple(sorted(set(components)))
    return result


def _optional_text(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _resolve(
    value: str | None,
    *,
    complexes: dict[str, tuple[str, ...]],
    cofactors: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    text = _optional_text(value)
    if text is None:
        return ()
    if text in complexes:
        return complexes[text]
    if cofactors is not None and text in cofactors:
        return cofactors[text]
    return (text,)


def _package_version(path: Path) -> str:
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("CellChat_package_version:"):
            return line.split(":", maxsplit=1)[1].strip()
    raise ContractError(
        "CELLCHATDB_VERSION.txt lacks CellChat_package_version",
        code="invalid_resource_version",
        field="CELLCHATDB_VERSION.txt",
        remediation="Use the version note shipped with the database export",
    )


def load_cellchat_resource(
    database_root: str | Path,
    species: Species | str,
    *,
    manifest_path: str | Path | None = None,
) -> ResourceBundle:
    """Load human or mouse CellChatDB CSVs with complex expansion.

    ``database_root`` is the caller-owned database directory. No machine-local
    path is retained in the returned bundle.
    """

    selected_species = Species(species)
    namespace = (
        GeneNamespace.HGNC_SYMBOL
        if selected_species is Species.HUMAN
        else GeneNamespace.MGI_SYMBOL
    )
    root = Path(database_root)
    resource_dir = root / "cellchat"
    prefix = selected_species.value
    required_names = (
        "CELLCHATDB_VERSION.txt",
        f"{prefix}_cofactor.csv",
        f"{prefix}_complex.csv",
        f"{prefix}_interaction.csv",
    )
    relative_paths = tuple(f"cellchat/{name}" for name in required_names)
    if manifest_path is None:
        verified = verify_gnu_checksum_file(
            resource_dir,
            resource_dir / "CHECKSUMS.sha256",
            required_paths=required_names,
        )
        manifest_digest = sha256_file(resource_dir / "CHECKSUMS.sha256")
        source_files = tuple(f"cellchat/{name}" for name in verified)
    else:
        manifest = ResourceManifest.from_json(manifest_path)
        manifest.require(
            species=selected_species.value,
            gene_namespace=namespace.value,
            license=_LICENSE,
        )
        source_files = manifest.verify(root, paths=relative_paths)
        manifest_digest = manifest.digest

    version = _package_version(resource_dir / "CELLCHATDB_VERSION.txt")
    complexes = _component_map(resource_dir / f"{prefix}_complex.csv", "subunit_")
    cofactors = _component_map(resource_dir / f"{prefix}_cofactor.csv", "cofactor")
    rows = _read_records(resource_dir / f"{prefix}_interaction.csv")
    interactions: list[Interaction] = []
    mapped_genes: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        source_id = (row.get("interaction_name") or "").strip()
        ligand_name = (row.get("ligand") or "").strip()
        receptor_name = (row.get("receptor") or "").strip()
        if not source_id or not ligand_name or not receptor_name:
            raise ContractError(
                f"CellChatDB interaction row {row_number} lacks a required identifier",
                code="invalid_cellchat_interaction",
                field="interaction_name",
                remediation="Regenerate the interaction CSV from the frozen R object",
            )
        ligand = _resolve(ligand_name, complexes=complexes)
        receptor = _resolve(receptor_name, complexes=complexes)
        direction = "Ligand-Receptor"
        interaction_id = build_interaction_id(
            resource_id=_RESOURCE_ID,
            version=version,
            species=selected_species,
            source_interaction_id=source_id,
            ligand_subunits=ligand,
            receptor_subunits=receptor,
            direction=direction,
        )
        evidence_text = _optional_text(row.get("evidence"))
        regulators = {
            "agonist_subunits": _resolve(
                row.get("agonist"), complexes=complexes, cofactors=cofactors
            ),
            "antagonist_subunits": _resolve(
                row.get("antagonist"), complexes=complexes, cofactors=cofactors
            ),
            "activating_coreceptor_subunits": _resolve(
                row.get("co_A_receptor"), complexes=complexes, cofactors=cofactors
            ),
            "inhibiting_coreceptor_subunits": _resolve(
                row.get("co_I_receptor"), complexes=complexes, cofactors=cofactors
            ),
        }
        row_version = _optional_text(row.get("version"))
        metadata = (
            ()
            if row_version is None
            else (("database_entry_version", row_version),)
        )
        interaction = Interaction(
            interaction_id=interaction_id,
            source_interaction_id=source_id,
            ligand_name=ligand_name,
            receptor_name=receptor_name,
            ligand_subunits=ligand,
            receptor_subunits=receptor,
            ligand_is_complex=ligand_name in complexes,
            receptor_is_complex=receptor_name in complexes,
            direction=direction,
            source=_RESOURCE_ID,
            version=version,
            species=selected_species,
            gene_namespace=namespace,
            pathway=_optional_text(row.get("pathway_name")),
            annotation=_optional_text(row.get("annotation")),
            evidence=() if evidence_text is None else (evidence_text,),
            metadata=metadata,
            **regulators,
        )
        interactions.append(interaction)
        mapped_genes.update(ligand)
        mapped_genes.update(receptor)
        for values in regulators.values():
            mapped_genes.update(values)

    report = MappingReport(
        source_rows=len(rows),
        loaded_rows=len(interactions),
        mapped_entities=len(mapped_genes),
        notes=(
            f"expanded_complexes={len(complexes)}",
            f"expanded_cofactor_groups={len(cofactors)}",
        ),
    )
    return ResourceBundle(
        resource_id=_RESOURCE_ID,
        version=version,
        species=selected_species,
        gene_namespace=namespace,
        interactions=tuple(interactions),
        mapping_report=report,
        manifest_digest=manifest_digest,
        source_files=source_files,
        license=_LICENSE,
        citation=_CITATION,
    )
