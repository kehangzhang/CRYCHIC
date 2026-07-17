"""Small unregistered calibration declarations for diagnostic unit tests."""

from __future__ import annotations

from crychic.core import canonical_digest
from crychic.inference.calibration_attestation import (
    CalibrationCampaignKind,
    CalibrationGeneratorManifest,
    CalibrationGeneratorProfile,
)


def diagnostic_generator_manifest(
    campaign_kind: CalibrationCampaignKind,
    generator_kind: str,
) -> CalibrationGeneratorManifest:
    """Declare an intentionally unregistered, non-releasing test generator."""

    return CalibrationGeneratorManifest(
        campaign_kind=campaign_kind,
        generator_kind=generator_kind,
        generator_version="diagnostic-v1",
        code_version="test-only-unregistered-v1",
        replay_entrypoint=f"tests.support.calibration:{generator_kind}",
        config_digest=canonical_digest(
            {"generator_kind": generator_kind, "profile": "diagnostic"}
        ),
        implementation_digest=canonical_digest(
            {"unregistered_test_generator": generator_kind}
        ),
        profile=CalibrationGeneratorProfile.DIAGNOSTIC_ONLY,
    )


__all__ = ["diagnostic_generator_manifest"]
