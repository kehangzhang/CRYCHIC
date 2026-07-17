from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from crychic.attribution import (
    FrozenReceiverFamilyLRHypothesisUniverse,
    freeze_receiver_family_lr_hypothesis_universe,
    freeze_receiver_family_opportunity_universe,
)
from crychic.core import CommunicationMode, ContractError, stable_id
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)


def _content_id(kind: str, value: object) -> str:
    return str(stable_id(kind, asdict(value)))


def _prior() -> TargetPrior:
    return TargetPrior(
        resource_id="lr-universe-prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="ligand",
        target_ids=("G1", "G2"),
        driver_ids=("L1", "L2"),
        indptr=(0, 1, 2),
        target_indices=(0, 1),
        weights=(1.0, 1.0),
        ranks=None,
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="p" * 64,
    )


def _interaction(
    interaction_id: str,
    *,
    ligand_name: str,
    ligand_subunit: str,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name=ligand_name,
        receptor_name=f"R-{interaction_id}",
        ligand_subunits=(ligand_subunit,),
        receptor_subunits=(f"R-{interaction_id}",),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="lr-universe-fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _bundle(*, reverse: bool = False) -> ResourceBundle:
    interactions = [
        _interaction("mapped-name", ligand_name="L1", ligand_subunit="L1"),
        _interaction("mapped-subunit", ligand_name="L2-alias", ligand_subunit="L2"),
        _interaction("unmapped", ligand_name="L9", ligand_subunit="L9"),
        _interaction("ambiguous", ligand_name="L1", ligand_subunit="L2"),
    ]
    if reverse:
        interactions.reverse()
    return ResourceBundle(
        resource_id="lr-universe-resource",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=tuple(interactions),
        mapping_report=MappingReport(4, 4, 8),
        manifest_digest="r" * 64,
        source_files=("resource.tsv",),
        license="CC0",
        citation="Synthetic LR hypothesis universe fixture",
    )


def _family_universe(prior: TargetPrior):
    receiver_ids = ("ReceiverB", "ReceiverA")
    canonical_receivers = tuple(sorted(receiver_ids))
    return freeze_receiver_family_opportunity_universe(
        prior,
        feature_ids=("G1", "G2"),
        receiver_ids=receiver_ids,
        receiver_universe_id="receiver-universe",
        receiver_axis_id=str(
            stable_id(
                "receiver_axis",
                {"receiver_ids": list(canonical_receivers)},
                schema_version="1",
            )
        ),
        prior_content_id=_content_id("target_prior_content", prior),
        root_input_identity_id="root-identity",
        root_input_digest="root-digest",
        cosine_threshold=0.99,
    )


def _universe(
    *,
    reverse_resource: bool = False,
    modes: tuple[CommunicationMode | str, ...] = (
        CommunicationMode.STATE,
        CommunicationMode.ECOSYSTEM,
    ),
):
    prior = _prior()
    bundle = _bundle(reverse=reverse_resource)
    return freeze_receiver_family_lr_hypothesis_universe(
        _family_universe(prior),
        bundle,
        prior,
        resource_bundle_content_id=_content_id("resource_bundle_content", bundle),
        target_prior_content_id=_content_id("target_prior_content", prior),
        modes=modes,
    )


def test_freezes_complete_static_mapping_and_receiver_opportunity_grid() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        FrozenReceiverFamilyLRHypothesisUniverse()

    universe = _universe()
    report = universe.mapping_report

    assert report.mapped_interaction_ids == ("mapped-name", "mapped-subunit")
    assert report.unmapped_interaction_ids == ("unmapped",)
    assert report.ambiguous_interactions == (("ambiguous", ("L1", "L2")),)
    assert universe.unmapped_interaction_ids == ("unmapped",)
    assert universe.ambiguous_interaction_ids == ("ambiguous",)
    assert universe.modes == ("state", "ecosystem")

    family_by_driver = {
        driver_id: family.family_id
        for family in universe._receiver_family_universe.family_definitions
        for driver_id in family.driver_ids
    }
    assert {
        (
            membership.interaction_id,
            membership.molecular_lr_equivalence_id,
            membership.driver_id,
            membership.family_id,
        )
        for membership in universe.memberships
    } == {
        (
            "mapped-name",
            next(
                record.molecular_lr_equivalence_id
                for record in universe.molecular_lr_equivalence_universe.mapping_records
                if record.interaction_id == "mapped-name"
            ),
            "L1",
            family_by_driver["L1"],
        ),
        (
            "mapped-subunit",
            next(
                record.molecular_lr_equivalence_id
                for record in universe.molecular_lr_equivalence_universe.mapping_records
                if record.interaction_id == "mapped-subunit"
            ),
            "L2",
            family_by_driver["L2"],
        ),
    }
    assert len(universe.hypothesis_ids) == 2 * 2
    assert len(universe.opportunity_ids) == 2 * 2 * 2
    assert len({row[-1] for row in universe.opportunity_ids}) == 8

    hypothesis_id = universe.hypothesis_id_for("mapped-name", "state")
    assert hypothesis_id
    assert universe.opportunity_id_for(
        "ReceiverA", "mapped-name", CommunicationMode.STATE
    ) != universe.opportunity_id_for("ReceiverB", "mapped-name", "state")
    with pytest.raises(ContractError) as caught:
        universe.hypothesis_id_for("unmapped", "state")
    assert caught.value.details.code == (
        "receiver_family_lr_hypothesis_not_in_universe"
    )

    manifest = universe.to_dict()
    assert manifest["schema_version"] == "2.0.0"
    assert manifest["membership_count"] == 2
    assert manifest["hypothesis_count"] == 4
    assert manifest["opportunity_count"] == 8
    assert manifest["mapping_report"]["unmapped_count"] == 1
    assert manifest["mapping_report"]["ambiguous_count"] == 1
    assert manifest["root_input_identity_id"] == "root-identity"
    assert manifest["root_input_digest"] == "root-digest"
    assert manifest["receiver_family_universe"] == (
        universe._receiver_family_universe.to_dict()
    )
    assert manifest["molecular_lr_equivalence_universe"] == (
        universe.molecular_lr_equivalence_universe.to_dict()
    )
    assert manifest["molecular_lr_equivalence_universe_id"] == (
        universe.molecular_lr_equivalence_universe.universe_id
    )
    assert manifest["molecular_lr_axis_id"] == (
        universe.molecular_lr_equivalence_universe.molecular_lr_axis_id
    )
    assert all(
        row["molecular_lr_equivalence_id"]
        for row in (
            *manifest["memberships"],
            *manifest["hypotheses"],
            *manifest["opportunities"],
        )
    )
    assert "_resource_bundle" not in manifest
    assert "_target_prior" not in manifest


def test_ids_are_independent_of_resource_and_mode_input_order() -> None:
    first = _universe()
    reordered = _universe(
        reverse_resource=True,
        modes=(CommunicationMode.ECOSYSTEM, CommunicationMode.STATE),
    )

    assert first.mapping_report.report_id == reordered.mapping_report.report_id
    assert first.membership_axis_id == reordered.membership_axis_id
    assert first.hypothesis_axis_id == reordered.hypothesis_axis_id
    assert first.opportunity_axis_id == reordered.opportunity_axis_id
    assert first.universe_id == reordered.universe_id
    assert first.to_dict() == reordered.to_dict()


def test_content_and_family_parent_mismatches_fail_closed() -> None:
    prior = _prior()
    bundle = _bundle()
    parent = _family_universe(prior)

    with pytest.raises(ContractError) as resource_error:
        freeze_receiver_family_lr_hypothesis_universe(
            parent,
            bundle,
            prior,
            resource_bundle_content_id="not-the-resource-content-id",
            target_prior_content_id=_content_id("target_prior_content", prior),
        )
    assert resource_error.value.details.code == (
        "frozen_receiver_family_lr_source_mismatch"
    )

    changed_prior = replace(prior, manifest_digest="changed-prior-manifest")
    with pytest.raises(ContractError) as prior_error:
        freeze_receiver_family_lr_hypothesis_universe(
            parent,
            bundle,
            changed_prior,
            resource_bundle_content_id=_content_id("resource_bundle_content", bundle),
            target_prior_content_id=_content_id("target_prior_content", changed_prior),
        )
    assert prior_error.value.details.code == (
        "frozen_receiver_family_lr_source_mismatch"
    )

    with pytest.raises(ValueError, match="non-empty and unique"):
        freeze_receiver_family_lr_hypothesis_universe(
            parent,
            bundle,
            prior,
            resource_bundle_content_id=_content_id("resource_bundle_content", bundle),
            target_prior_content_id=_content_id("target_prior_content", prior),
            modes=("state", "state"),
        )


@pytest.mark.parametrize(
    "tamper",
    (
        lambda universe: object.__setattr__(
            universe, "membership_axis_id", "tampered-membership-axis"
        ),
        lambda universe: object.__setattr__(
            universe, "root_input_digest", "tampered-root-digest"
        ),
        lambda universe: object.__setattr__(
            universe.mapping_report, "report_id", "tampered-mapping-report"
        ),
        lambda universe: object.__setattr__(
            universe.memberships[0],
            "molecular_lr_equivalence_id",
            "tampered-molecular-lr",
        ),
        lambda universe: object.__setattr__(
            universe._resource_bundle,
            "manifest_digest",
            "tampered-resource-manifest",
        ),
    ),
)
def test_universe_rejects_public_and_hidden_parent_tampering(tamper) -> None:
    universe = _universe()
    tamper(universe)

    with pytest.raises(ContractError) as caught:
        universe.to_dict()
    assert caught.value.details.code == (
        "frozen_receiver_family_lr_hypothesis_universe_integrity_violation"
    )
