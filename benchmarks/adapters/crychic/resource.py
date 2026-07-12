"""Resource bridges used by the CRYCHIC benchmark adapter."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.adapters.common import load_harmonized_resource
from crychic.core import canonical_digest
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    load_cellchat_resource,
    load_cellphonedb_resource,
)


def _citation(value: object) -> str:
    if isinstance(value, list):
        entries = [str(item).strip() for item in value if str(item).strip()]
        if entries:
            return " ".join(entries)
    text = str(value).strip()
    if not text:
        raise ValueError("harmonized resource manifest has no citation")
    return text


def harmonized_resource_bundle(
    table_path: str | Path,
    manifest_path: str | Path,
) -> ResourceBundle:
    """Build a checksum-verified CRYCHIC bundle from ``harmonized_lr.tsv``.

    The harmonized interaction identifier is retained verbatim. This is what
    allows every method in H-common to use the same frozen LR identity rather
    than a CRYCHIC-specific remapping.
    """

    table, manifest = load_harmonized_resource(table_path, manifest_path)
    supported_schemas = {
        "crychic-harmonized-lr-v1",
        "crychic-harmonized-lr-v1-synthetic-fixture",
    }
    if manifest.get("schema_version") not in supported_schemas:
        raise ValueError("unsupported harmonized resource schema")
    if manifest.get("species") != Species.HUMAN.value:
        raise ValueError("H-common CRYCHIC currently supports human resources")
    if manifest.get("gene_namespace") != GeneNamespace.HGNC_SYMBOL.value:
        raise ValueError("H-common resource must use HGNC symbols")

    resource_id = str(manifest.get("resource_id", "")).strip()
    version = str(manifest.get("version", "")).strip()
    license_name = str(manifest.get("license", "")).strip()
    if not resource_id or not version or not license_name:
        raise ValueError("harmonized resource manifest lacks governance metadata")

    interactions: list[Interaction] = []
    for row in table.itertuples(index=False):
        interaction_id = str(row.harmonized_interaction_id)
        ligand = str(row.ligand)
        receptor = str(row.receptor)
        interactions.append(
            Interaction(
                interaction_id=interaction_id,
                source_interaction_id=interaction_id,
                ligand_name=ligand,
                receptor_name=receptor,
                ligand_subunits=(ligand,),
                receptor_subunits=(receptor,),
                ligand_is_complex=False,
                receptor_is_complex=False,
                direction="Ligand-Receptor",
                source=resource_id,
                version=version,
                species=Species.HUMAN,
                gene_namespace=GeneNamespace.HGNC_SYMBOL,
                evidence=(
                    f"cellchat:{row.cellchat_source_interaction_id}",
                    f"cellphonedb:{row.cellphonedb_source_interaction_id}",
                ),
            )
        )

    manifest_digest = canonical_digest(manifest)
    count = len(interactions)
    return ResourceBundle(
        resource_id=resource_id,
        version=version,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=MappingReport(
            source_rows=count,
            loaded_rows=count,
            mapped_entities=2 * count,
            notes=("exact H-common monomeric HGNC ligand-receptor pairs",),
        ),
        manifest_digest=manifest_digest,
        source_files=(Path(table_path).name, Path(manifest_path).name),
        license=license_name,
        citation=_citation(manifest.get("citation", "")),
    )


def load_native_resource_bundle(
    adapter: str,
    *,
    database_root: str | Path,
    manifest_path: str | Path,
    species: Species | str = Species.HUMAN,
) -> ResourceBundle:
    """Load one first-party native LR bundle for CRYCHIC readback."""

    if adapter == "cellchat":
        return load_cellchat_resource(
            database_root,
            species,
            manifest_path=manifest_path,
        )
    if adapter == "cellphonedb":
        if Species(species) is not Species.HUMAN:
            raise ValueError("CellPhoneDB v5 native readback supports human only")
        return load_cellphonedb_resource(
            database_root,
            manifest_path=manifest_path,
        )
    raise ValueError(f"unsupported CRYCHIC native resource adapter: {adapter!r}")


def _complex_label(parts: tuple[str, ...]) -> str:
    return "&".join(parts)


def bundle_resource_table(
    bundle: ResourceBundle,
    *,
    input_genes: Collection[str] | None = None,
) -> pd.DataFrame:
    """Map a CRYCHIC bundle to the frozen benchmark resource contract."""

    genes = None if input_genes is None else set(map(str, input_genes))
    records: list[dict[str, Any]] = []
    for interaction in bundle.interactions:
        required = {*interaction.ligand_subunits, *interaction.receptor_subunits}
        records.append(
            {
                "interaction_id": interaction.interaction_id,
                "native_interaction_id": interaction.source_interaction_id,
                "ligand": _complex_label(interaction.ligand_subunits),
                "receptor": _complex_label(interaction.receptor_subunits),
                "target": pd.NA,
                "method_covered": True,
                "input_available": genes is None or required.issubset(genes),
            }
        )
    if not records:
        raise ValueError("CRYCHIC resource bundle contains no interactions")
    return pd.DataFrame.from_records(records)
