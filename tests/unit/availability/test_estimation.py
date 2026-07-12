from __future__ import annotations

from crychic.availability import (
    AbundanceObservation,
    AvailabilityStatus,
    GeneObservation,
    estimate_interaction_availability,
)
from crychic.resources import GeneNamespace, Interaction, Species


def _interaction() -> Interaction:
    return Interaction(
        interaction_id="interaction_test",
        source_interaction_id="TEST",
        ligand_name="L",
        receptor_name="R_COMPLEX",
        ligand_subunits=("L",),
        receptor_subunits=("R1", "R2"),
        ligand_is_complex=False,
        receptor_is_complex=True,
        direction="Ligand-Receptor",
        source="test",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _observed(expression: float = 2.0) -> GeneObservation:
    return GeneObservation(
        expression=expression,
        detection_fraction=0.8,
        n_cells=20,
    )


def _abundance(proportion: float) -> AbundanceObservation:
    return AbundanceObservation(captured_proportion=proportion)


def test_abundance_only_change_alters_ecosystem_not_state() -> None:
    interaction = _interaction()
    sender = {"L": _observed()}
    receiver = {"R1": _observed(), "R2": _observed()}
    low = estimate_interaction_availability(
        interaction, sender, receiver, _abundance(0.1), _abundance(0.1)
    )
    high = estimate_interaction_availability(
        interaction, sender, receiver, _abundance(0.4), _abundance(0.4)
    )

    assert low.state == high.state
    assert low.ecosystem is not None and high.ecosystem is not None
    assert high.ecosystem > low.ecosystem
    assert high.ecosystem_label == "capture_weighted_ecosystem_proxy"


def test_sampling_zero_propagates_missingness_instead_of_zero() -> None:
    missing_ligand = GeneObservation(
        expression=None,
        detection_fraction=None,
        n_cells=0,
        status=AvailabilityStatus.SAMPLING_ZERO,
    )
    estimate = estimate_interaction_availability(
        _interaction(),
        {"L": missing_ligand},
        {"R1": _observed(), "R2": _observed()},
        AbundanceObservation(
            captured_proportion=None,
            presence_probability=None,
            capture_weight=None,
            status=AvailabilityStatus.SAMPLING_ZERO,
        ),
        _abundance(0.2),
    )

    assert estimate.state is None
    assert estimate.ecosystem is None
    assert estimate.state_status is AvailabilityStatus.SAMPLING_ZERO
    assert estimate.ecosystem_status is AvailabilityStatus.SAMPLING_ZERO


def test_confirmed_required_subunit_absence_is_biological_zero() -> None:
    absent = GeneObservation(
        expression=None,
        detection_fraction=None,
        n_cells=0,
        status=AvailabilityStatus.CONFIRMED_ABSENCE,
    )
    estimate = estimate_interaction_availability(
        _interaction(),
        {"L": _observed()},
        {"R1": _observed(), "R2": absent},
        _abundance(0.2),
        _abundance(0.2),
    )

    assert estimate.receptor.state == 0.0
    assert estimate.state == 0.0
    assert estimate.ecosystem == 0.0
    assert estimate.state_status is AvailabilityStatus.CONFIRMED_ABSENCE
