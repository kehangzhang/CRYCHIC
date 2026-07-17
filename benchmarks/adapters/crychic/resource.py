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
    MolecularLRMappingStatus,
    ResourceBundle,
    Species,
    freeze_molecular_lr_equivalence_universe,
    load_cellchat_resource,
    load_cellphonedb_resource,
)

MOLECULAR_LR_CROSSWALK_COLUMNS = (
    "resource_id",
    "resource_version",
    "resource_manifest_digest",
    "resource_bundle_content_id",
    "interaction_id",
    "source_interaction_id",
    "mapping_status",
    "reason_code",
    "molecular_lr_equivalence_id",
    "mechanistic_variant_id",
    "molecular_lr_equivalence_universe_id",
    "molecular_lr_axis_id",
    "mechanistic_variant_axis_id",
    "mapping_axis_id",
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
    try:
        species = Species(str(manifest.get("species", "")))
        namespace = GeneNamespace(str(manifest.get("gene_namespace", "")))
    except ValueError as error:
        raise ValueError(
            "H-common resource has unsupported molecular metadata"
        ) from error
    expected_namespace = (
        GeneNamespace.HGNC_SYMBOL
        if species is Species.HUMAN
        else GeneNamespace.MGI_SYMBOL
    )
    if namespace is not expected_namespace:
        raise ValueError("H-common resource species and gene namespace disagree")

    resource_id = str(manifest.get("resource_id", "")).strip()
    version = str(manifest.get("version", "")).strip()
    license_name = str(manifest.get("license", "")).strip()
    if not resource_id or not version or not license_name:
        raise ValueError("harmonized resource manifest lacks governance metadata")

    interactions: list[Interaction] = []
    source_columns = [
        column
        for column in table.columns
        if column.endswith("_source_interaction_id")
    ]
    for row in table.itertuples(index=False):
        interaction_id = str(row.harmonized_interaction_id)
        ligand = str(row.ligand)
        receptor = str(row.receptor)
        evidence = tuple(
            f"{column.removesuffix('_source_interaction_id')}:"
            f"{getattr(row, column)}"
            for column in source_columns
            if str(getattr(row, column)).strip()
        )
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
                species=species,
                gene_namespace=namespace,
                evidence=evidence,
            )
        )

    manifest_digest = canonical_digest(manifest)
    count = len(interactions)
    return ResourceBundle(
        resource_id=resource_id,
        version=version,
        species=species,
        gene_namespace=namespace,
        interactions=tuple(interactions),
        mapping_report=MappingReport(
            source_rows=count,
            loaded_rows=count,
            mapped_entities=2 * count,
            notes=(
                f"exact H-common monomeric {namespace.value} ligand-receptor pairs",
            ),
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


def bundle_molecular_lr_crosswalk(bundle: ResourceBundle) -> pd.DataFrame:
    """Return the exact source-interaction to molecular-LR crosswalk.

    The crosswalk is separate from method scores: repeated source rows that map
    to one molecular class remain distinct here and can be aggregated only by a
    metric with an explicit frozen-member policy.
    """

    if not isinstance(bundle, ResourceBundle):
        raise TypeError("bundle must be a ResourceBundle")
    universe = freeze_molecular_lr_equivalence_universe(bundle)
    if len(universe.source_bindings) != 1:
        raise ValueError("one ResourceBundle must produce one source binding")
    source = universe.source_bindings[0]
    records = [
        {
            "resource_id": bundle.resource_id,
            "resource_version": bundle.version,
            "resource_manifest_digest": bundle.manifest_digest,
            "resource_bundle_content_id": source.resource_bundle_content_id,
            "interaction_id": mapping.interaction_id,
            "source_interaction_id": mapping.source_interaction_id,
            "mapping_status": mapping.status.value,
            "reason_code": mapping.reason_code,
            "molecular_lr_equivalence_id": mapping.molecular_lr_equivalence_id,
            "mechanistic_variant_id": mapping.mechanistic_variant_id,
            "molecular_lr_equivalence_universe_id": universe.universe_id,
            "molecular_lr_axis_id": universe.molecular_lr_axis_id,
            "mechanistic_variant_axis_id": universe.mechanistic_variant_axis_id,
            "mapping_axis_id": universe.mapping_axis_id,
        }
        for mapping in universe.mapping_records
    ]
    table = pd.DataFrame.from_records(
        records, columns=MOLECULAR_LR_CROSSWALK_COLUMNS
    ).sort_values("interaction_id", kind="stable", ignore_index=True)
    if (
        len(table) != len(bundle.interactions)
        or table["interaction_id"].duplicated().any()
        or set(table["interaction_id"])
        != {interaction.interaction_id for interaction in bundle.interactions}
        or set(table["mapping_status"]).difference(
            {status.value for status in MolecularLRMappingStatus}
        )
    ):
        raise ValueError("molecular LR crosswalk does not cover the source bundle")
    mapped = table["mapping_status"].eq(MolecularLRMappingStatus.MAPPED.value)
    if (
        table.loc[
            mapped,
            ["molecular_lr_equivalence_id", "mechanistic_variant_id"],
        ]
        .isna()
        .any(axis=None)
        or table.loc[
            ~mapped,
            ["molecular_lr_equivalence_id", "mechanistic_variant_id"],
        ]
        .notna()
        .any(axis=None)
    ):
        raise ValueError("molecular LR crosswalk status and identifiers disagree")
    return table.loc[:, list(MOLECULAR_LR_CROSSWALK_COLUMNS)].copy(deep=True)


def attach_molecular_lr_equivalence_ids(
    score_table: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a complete molecular-LR axis to one benchmark score table."""

    score_keys = ("resource", "resource_version", "interaction_id")
    missing_scores = set(score_keys).difference(score_table.columns)
    missing_crosswalk = set(MOLECULAR_LR_CROSSWALK_COLUMNS).difference(
        crosswalk.columns
    )
    if missing_scores or missing_crosswalk:
        raise ValueError(
            "molecular LR join inputs are incomplete: "
            f"score={sorted(missing_scores)}, crosswalk={sorted(missing_crosswalk)}"
        )
    if "molecular_lr_equivalence_id" in score_table.columns:
        raise ValueError(
            "score table already contains molecular_lr_equivalence_id; "
            "validate it instead of overwriting it"
        )
    mapping = crosswalk.rename(columns={"resource_id": "resource"}).loc[
        :,
        [
            *score_keys,
            "mapping_status",
            "reason_code",
            "molecular_lr_equivalence_id",
            "mechanistic_variant_id",
            "molecular_lr_equivalence_universe_id",
            "molecular_lr_axis_id",
        ],
    ]
    if mapping.duplicated(list(score_keys)).any():
        raise ValueError("molecular LR crosswalk has duplicate resource-edge keys")
    result = score_table.merge(
        mapping,
        on=list(score_keys),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    complete = (
        result["mapping_status"].eq(MolecularLRMappingStatus.MAPPED.value)
        & result["reason_code"].isna()
        & result["molecular_lr_equivalence_id"].notna()
        & result["mechanistic_variant_id"].notna()
        & result["molecular_lr_equivalence_universe_id"].notna()
        & result["molecular_lr_axis_id"].notna()
    )
    if not complete.all():
        failed = (
            result.loc[~complete, list(score_keys)]
            .drop_duplicates()
            .sort_values(list(score_keys), kind="stable")
            .to_dict(orient="records")
        )
        raise ValueError(
            "molecular LR crosswalk does not completely map the frozen score "
            f"universe: {failed[:5]}"
        )
    return result.drop(columns=["mapping_status", "reason_code"]).copy(deep=True)
