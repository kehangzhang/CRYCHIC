from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from crychic.core import ContractError
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    MolecularLREquivalenceClass,
    MolecularLRMappingStatus,
    ResourceBundle,
    Species,
    build_interaction_id,
    freeze_molecular_lr_equivalence_universe,
)


def _interaction(
    *,
    resource_id: str = "resource-a",
    version: str = "1.0",
    source_id: str = "row-1",
    ligand: tuple[str, ...] = ("L1",),
    receptor: tuple[str, ...] = ("R1",),
    direction: str = "Ligand-Receptor",
    species: Species = Species.HUMAN,
    namespace: GeneNamespace = GeneNamespace.HGNC_SYMBOL,
    agonist: tuple[str, ...] = (),
    antagonist: tuple[str, ...] = (),
    activating_coreceptor: tuple[str, ...] = (),
    inhibiting_coreceptor: tuple[str, ...] = (),
    display_suffix: str = "a",
) -> Interaction:
    return Interaction(
        interaction_id=build_interaction_id(
            resource_id=resource_id,
            version=version,
            species=species,
            source_interaction_id=source_id,
            ligand_subunits=ligand,
            receptor_subunits=receptor,
            direction=direction,
        ),
        source_interaction_id=source_id,
        ligand_name=f"display ligand {display_suffix}",
        receptor_name=f"display receptor {display_suffix}",
        ligand_subunits=ligand,
        receptor_subunits=receptor,
        ligand_is_complex=len(ligand) > 1,
        receptor_is_complex=len(receptor) > 1,
        direction=direction,
        source=resource_id,
        version=version,
        species=species,
        gene_namespace=namespace,
        pathway=f"display pathway {display_suffix}",
        annotation=f"display annotation {display_suffix}",
        evidence=(f"display evidence {display_suffix}",),
        agonist_subunits=agonist,
        antagonist_subunits=antagonist,
        activating_coreceptor_subunits=activating_coreceptor,
        inhibiting_coreceptor_subunits=inhibiting_coreceptor,
    )


def _bundle(
    resource_id: str,
    version: str,
    interactions: tuple[Interaction, ...],
    *,
    species: Species = Species.HUMAN,
    namespace: GeneNamespace = GeneNamespace.HGNC_SYMBOL,
) -> ResourceBundle:
    report = MappingReport(
        source_rows=len(interactions),
        loaded_rows=len(interactions),
        mapped_entities=len(
            {
                gene
                for interaction in interactions
                for genes in (
                    interaction.ligand_subunits,
                    interaction.receptor_subunits,
                )
                for gene in genes
            }
        ),
        notes=(f"source={resource_id}",),
    )
    return ResourceBundle(
        resource_id=resource_id,
        version=version,
        species=species,
        gene_namespace=namespace,
        interactions=interactions,
        mapping_report=report,
        manifest_digest=f"manifest-{resource_id}-{version}",
        source_files=(f"{resource_id}.csv",),
        license="test-only",
        citation="test citation",
    )


def test_equivalence_id_is_resource_independent_and_mapping_is_complete() -> None:
    first = _interaction(
        resource_id="resource-a",
        version="1.0",
        source_id="native-a",
        ligand=("L2", "L1"),
        receptor=("R2", "R1"),
        display_suffix="first",
    )
    second = _interaction(
        resource_id="resource-b",
        version="9.7",
        source_id="native-b",
        ligand=("L1", "L2"),
        receptor=("R1", "R2"),
        display_suffix="second",
    )
    first_bundle = _bundle("resource-a", "1.0", (first,))
    second_bundle = _bundle("resource-b", "9.7", (second,))

    universe = freeze_molecular_lr_equivalence_universe((first_bundle, second_bundle))

    assert len(universe.equivalence_classes) == 1
    assert len(universe.mapping_records) == 2
    assert {
        record.molecular_lr_equivalence_id for record in universe.mapping_records
    } == {universe.molecular_lr_equivalence_ids[0]}
    assert all(
        record.status is MolecularLRMappingStatus.MAPPED
        for record in universe.mapping_records
    )
    assert {binding.mapping_report for binding in universe.source_bindings} == {
        first_bundle.mapping_report,
        second_bundle.mapping_report,
    }
    assert universe.to_dict()["mapped_count"] == 2


def test_component_order_stable_but_orientation_and_partial_complex_differ() -> None:
    ordered = MolecularLREquivalenceClass(
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        ligand_subunits=("L1", "L2"),
        receptor_subunits=("R1", "R2"),
    )
    reordered = MolecularLREquivalenceClass(
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        ligand_subunits=("L2", "L1"),
        receptor_subunits=("R2", "R1"),
    )
    reversed_lr = MolecularLREquivalenceClass(
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        ligand_subunits=("R1", "R2"),
        receptor_subunits=("L1", "L2"),
    )
    partial = MolecularLREquivalenceClass(
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        ligand_subunits=("L1",),
        receptor_subunits=("R1", "R2"),
    )
    mouse = MolecularLREquivalenceClass(
        species=Species.MOUSE,
        gene_namespace=GeneNamespace.MGI_SYMBOL,
        ligand_subunits=("L1", "L2"),
        receptor_subunits=("R1", "R2"),
    )
    other_namespace = MolecularLREquivalenceClass(
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.MGI_SYMBOL,
        ligand_subunits=("L1", "L2"),
        receptor_subunits=("R1", "R2"),
    )

    assert ordered.molecular_lr_equivalence_id == (
        reordered.molecular_lr_equivalence_id
    )
    assert ordered.molecular_lr_equivalence_id != (
        reversed_lr.molecular_lr_equivalence_id
    )
    assert ordered.molecular_lr_equivalence_id != partial.molecular_lr_equivalence_id
    assert ordered.molecular_lr_equivalence_id != mouse.molecular_lr_equivalence_id
    assert ordered.molecular_lr_equivalence_id != (
        other_namespace.molecular_lr_equivalence_id
    )
    assert ordered.direction == "ligand_to_receptor"


def test_modifiers_change_variant_but_not_core_equivalence() -> None:
    plain = _interaction(source_id="plain", ligand=("L1",), receptor=("R1",))
    regulated = _interaction(
        source_id="regulated",
        ligand=("L1",),
        receptor=("R1",),
        agonist=("A2", "A1"),
        antagonist=("I1",),
        activating_coreceptor=("CA1",),
        inhibiting_coreceptor=("CI1",),
    )
    universe = freeze_molecular_lr_equivalence_universe(
        _bundle("resource-a", "1.0", (plain, regulated))
    )

    assert len(universe.equivalence_classes) == 1
    assert len(universe.mechanistic_variants) == 2
    assert (
        len({record.molecular_lr_equivalence_id for record in universe.mapping_records})
        == 1
    )
    assert (
        len({record.mechanistic_variant_id for record in universe.mapping_records}) == 2
    )


def test_unsupported_direction_is_recorded_without_guessing() -> None:
    unsupported = _interaction(direction="unspecified")
    universe = freeze_molecular_lr_equivalence_universe(
        _bundle("resource-a", "1.0", (unsupported,))
    )

    assert universe.equivalence_classes == ()
    assert universe.mechanistic_variants == ()
    assert len(universe.mapping_records) == 1
    record = universe.mapping_records[0]
    assert record.status is MolecularLRMappingStatus.UNSUPPORTED_DIRECTION
    assert record.molecular_lr_equivalence_id is None
    assert record.mechanistic_variant_id is None
    assert record.reason_code == "unsupported_interaction_direction"
    assert universe.to_dict()["unsupported_direction_count"] == 1


@pytest.mark.parametrize(
    ("second_species", "second_namespace"),
    [
        (Species.MOUSE, GeneNamespace.MGI_SYMBOL),
        (Species.HUMAN, GeneNamespace.MGI_SYMBOL),
    ],
)
def test_mixed_species_or_namespace_fails(
    second_species: Species,
    second_namespace: GeneNamespace,
) -> None:
    first = _bundle("resource-a", "1", (_interaction(),))
    second_interaction = _interaction(
        resource_id="resource-b",
        species=second_species,
        namespace=second_namespace,
    )
    second = _bundle(
        "resource-b",
        "1",
        (second_interaction,),
        species=second_species,
        namespace=second_namespace,
    )

    with pytest.raises(ContractError, match="share species and gene namespace"):
        freeze_molecular_lr_equivalence_universe((first, second))


def test_bundle_and_row_order_do_not_change_frozen_ids() -> None:
    first_row = _interaction(resource_id="resource-a", source_id="row-1")
    second_row = _interaction(
        resource_id="resource-a",
        source_id="row-2",
        ligand=("L2",),
        receptor=("R2",),
    )
    bundle_forward = _bundle("resource-a", "1.0", (first_row, second_row))
    bundle_reverse = _bundle("resource-a", "1.0", (second_row, first_row))
    other = _bundle(
        "resource-b",
        "2.0",
        (
            _interaction(
                resource_id="resource-b",
                version="2.0",
                source_id="other-row",
                ligand=("L3",),
                receptor=("R3",),
            ),
        ),
    )

    row_forward = freeze_molecular_lr_equivalence_universe(bundle_forward)
    row_reverse = freeze_molecular_lr_equivalence_universe(bundle_reverse)
    bundle_order_a = freeze_molecular_lr_equivalence_universe((bundle_forward, other))
    bundle_order_b = freeze_molecular_lr_equivalence_universe((other, bundle_reverse))

    assert row_forward.universe_id == row_reverse.universe_id
    assert row_forward.molecular_lr_axis_id == row_reverse.molecular_lr_axis_id
    assert bundle_order_a.universe_id == bundle_order_b.universe_id
    assert bundle_order_a.mapping_axis_id == bundle_order_b.mapping_axis_id


def test_frozen_contracts_reject_mutation_and_detect_tampering() -> None:
    universe = freeze_molecular_lr_equivalence_universe(
        _bundle("resource-a", "1.0", (_interaction(),))
    )

    with pytest.raises(FrozenInstanceError):
        universe.universe_id = "forged"  # type: ignore[misc]

    object.__setattr__(universe, "universe_id", "forged")
    with pytest.raises(ContractError, match="failed integrity validation"):
        universe.to_dict()


def test_interaction_namespace_must_match_its_bundle() -> None:
    interaction = _interaction(namespace=GeneNamespace.MGI_SYMBOL)
    bundle = _bundle(
        "resource-a",
        "1.0",
        (interaction,),
        namespace=GeneNamespace.HGNC_SYMBOL,
    )

    with pytest.raises(ContractError, match="differs from its bundle"):
        freeze_molecular_lr_equivalence_universe(bundle)
