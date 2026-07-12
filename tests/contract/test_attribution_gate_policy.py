from __future__ import annotations

import inspect

import crychic.attribution as attribution


def test_attribution_gate_policy_is_versioned_and_opt_in() -> None:
    assert "ReceptorGatePolicy" in attribution.__all__
    assert (
        attribution.ReceptorGatePolicy.LEGACY_CONTINUOUS_V1.value
        == "legacy_continuous_v1"
    )
    assert (
        attribution.ReceptorGatePolicy.HARD_ELIGIBILITY_V2.value
        == "hard_eligibility_v2"
    )
    assert attribution.ReceptorGatePolicy.LEGACY_CONTINUOUS_V1.version == 1
    assert attribution.ReceptorGatePolicy.HARD_ELIGIBILITY_V2.version == 2

    parameters = inspect.signature(attribution.build_gated_target_basis).parameters
    assert (
        parameters["gate_policy"].default
        is attribution.ReceptorGatePolicy.LEGACY_CONTINUOUS_V1
    )
    assert parameters["receptor_gate_threshold"].default is None
