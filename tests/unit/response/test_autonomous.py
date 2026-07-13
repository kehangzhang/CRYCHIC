from __future__ import annotations

import numpy as np
import pytest

from crychic.core import ContractError
from crychic.resources import GeneNamespace, Species
from crychic.response import (
    AutonomousProgramResidualization,
    AutonomousProgramSupportError,
    ReceiverAutonomousProgramResource,
    build_receiver_autonomous_program_resource,
    residualize_against_autonomous_programs,
)

_MANIFEST = "a" * 64


def _resource(
    matrix: np.ndarray | None = None,
    *,
    feature_ids: tuple[str, ...] = ("G1", "G2", "G3"),
    program_ids: tuple[str, ...] = ("stress",),
) -> ReceiverAutonomousProgramResource:
    if matrix is None:
        matrix = np.asarray([[1.0], [-0.5], [0.0]])
    return build_receiver_autonomous_program_resource(
        matrix,
        feature_ids=feature_ids,
        program_ids=program_ids,
        resource_id="registered_receiver_programs",
        version="2026.1",
        manifest_digest=_MANIFEST,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def test_resource_constructor_is_producer_owned() -> None:
    with pytest.raises(TypeError, match="producer-owned"):
        ReceiverAutonomousProgramResource()
    with pytest.raises(TypeError, match="producer-owned"):
        AutonomousProgramResidualization()


def test_resource_canonicalizes_both_axes_and_has_order_stable_identity() -> None:
    reordered = _resource(
        np.asarray(
            [
                [3.0, -3.0],
                [1.0, -1.0],
                [2.0, -2.0],
            ]
        ),
        feature_ids=("G3", "G1", "G2"),
        program_ids=("stress", "cycle"),
    )
    canonical = _resource(
        np.asarray(
            [
                [-1.0, 1.0],
                [-2.0, 2.0],
                [-3.0, 3.0],
            ]
        ),
        feature_ids=("G1", "G2", "G3"),
        program_ids=("cycle", "stress"),
    )

    assert reordered.feature_ids == ("G1", "G2", "G3")
    assert reordered.program_ids == ("cycle", "stress")
    np.testing.assert_array_equal(reordered.matrix, canonical.matrix)
    assert reordered.matrix_digest == canonical.matrix_digest
    assert reordered.artifact_id == canonical.artifact_id


def test_resource_accepts_signed_weights_and_freezes_matrix_storage() -> None:
    resource = _resource()

    np.testing.assert_array_equal(resource.matrix[:, 0], [1.0, -0.5, 0.0])
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        resource.matrix.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        resource.matrix[0, 0] = 2.0


def test_resource_identity_binds_upstream_manifest_and_version() -> None:
    first = _resource()
    changed_version = build_receiver_autonomous_program_resource(
        first.matrix,
        feature_ids=first.feature_ids,
        program_ids=first.program_ids,
        resource_id=first.resource_id,
        version="2026.2",
        manifest_digest=first.manifest_digest,
        species=first.species,
        gene_namespace=first.gene_namespace,
    )
    changed_manifest = build_receiver_autonomous_program_resource(
        first.matrix,
        feature_ids=first.feature_ids,
        program_ids=first.program_ids,
        resource_id=first.resource_id,
        version=first.version,
        manifest_digest="b" * 64,
        species=first.species,
        gene_namespace=first.gene_namespace,
    )

    assert first.artifact_id != changed_version.artifact_id
    assert first.artifact_id != changed_manifest.artifact_id


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_resource_rejects_nonfinite_weights(invalid: float) -> None:
    matrix = np.asarray([[1.0], [invalid], [0.0]])

    with pytest.raises(ValueError, match="finite signed weights"):
        _resource(matrix)


def test_resource_rejects_zero_program_duplicate_id_and_invalid_manifest() -> None:
    with pytest.raises(ValueError, match="non-zero support"):
        _resource(np.zeros((3, 1)))
    with pytest.raises(ValueError, match="unique strings"):
        build_receiver_autonomous_program_resource(
            np.ones((3, 1)),
            feature_ids=("G1", "G1", "G2"),
            program_ids=("stress",),
            resource_id="programs",
            version="1",
            manifest_digest=_MANIFEST,
            species=Species.HUMAN,
            gene_namespace=GeneNamespace.HGNC_SYMBOL,
        )


@pytest.mark.parametrize("magnitude", [1e-200, 1e300])
def test_resource_accepts_finite_nonzero_extreme_program_support(
    magnitude: float,
) -> None:
    resource = _resource(np.asarray([[magnitude], [0.0], [0.0]]))

    assert resource.matrix[0, 0] == magnitude
    resource.to_dict()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_receiver_autonomous_program_resource(
            np.ones((3, 1)),
            feature_ids=("G1", "G2", "G3"),
            program_ids=("stress",),
            resource_id="programs",
            version="1",
            manifest_digest="not-a-digest",
            species=Species.HUMAN,
            gene_namespace=GeneNamespace.HGNC_SYMBOL,
        )


def test_resource_integrity_detects_matrix_and_provenance_tampering() -> None:
    changed_matrix = _resource()
    object.__setattr__(changed_matrix, "matrix", changed_matrix.matrix.copy())
    changed_matrix.matrix[0, 0] = 99.0

    with pytest.raises(ContractError) as matrix_error:
        changed_matrix.to_dict()
    assert (
        matrix_error.value.details.code
        == "receiver_autonomous_program_integrity_violation"
    )

    changed_version = _resource()
    object.__setattr__(changed_version, "version", "other")
    with pytest.raises(ContractError) as version_error:
        changed_version.to_dict()
    assert (
        version_error.value.details.code
        == "receiver_autonomous_program_integrity_violation"
    )


def test_matrix_for_features_preserves_requested_order_and_zero_fills() -> None:
    resource = _resource()

    aligned = resource.matrix_for_features(("G3", "unknown", "G1"))

    np.testing.assert_array_equal(aligned[:, 0], [0.0, 0.0, 1.0])
    assert not aligned.flags.writeable


def test_precision_weighted_residualization_marks_exact_overlap_unidentifiable() -> (
    None
):
    program = np.asarray([[1.0], [2.0], [-1.0]])
    resource = _resource(program)
    result = residualize_against_autonomous_programs(
        program,
        feature_ids=("G1", "G2", "G3"),
        candidate_ids=("family_overlap",),
        precision=np.asarray([1.0, 4.0, 2.0]),
        program_resource=resource,
    )

    np.testing.assert_allclose(result.residualized_basis, 0.0, atol=1e-12)
    assert result.retained_fractions[0] == pytest.approx(0.0, abs=1e-12)
    assert result.identifiable.tolist() == [False]
    assert result.nonidentifiable_candidate_ids == ("family_overlap",)
    assert result.identifiable_candidate_ids == ()


def test_residualization_distinguishes_near_overlap_from_unique_support() -> None:
    resource = _resource(np.asarray([[1.0], [0.0], [0.0]]))
    candidates = np.asarray(
        [
            [1.0, 0.0],
            [1e-9, 1.0],
            [0.0, 0.0],
        ]
    )
    result = residualize_against_autonomous_programs(
        candidates,
        feature_ids=("G1", "G2", "G3"),
        candidate_ids=("near_overlap", "unique"),
        precision=np.ones(3),
        program_resource=resource,
        identifiability_tolerance=1e-6,
    )

    np.testing.assert_allclose(
        result.residualized_basis,
        np.asarray([[0.0, 0.0], [1e-9, 1.0], [0.0, 0.0]]),
        atol=1e-15,
    )
    assert result.retained_fractions[0] == pytest.approx(1e-9)
    assert result.retained_fractions[1] == pytest.approx(1.0)
    assert result.identifiable.tolist() == [False, True]
    assert result.nonidentifiable_candidate_ids == ("near_overlap",)
    assert result.identifiable_candidate_ids == ("unique",)


def test_residualization_is_invariant_to_nonzero_program_column_scaling() -> None:
    programs = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    scaled = programs.copy()
    scaled[:, 1] *= 1e-14
    candidate = np.asarray([[1.0], [1.0], [2.0]])
    first = residualize_against_autonomous_programs(
        candidate,
        feature_ids=("G1", "G2", "G3"),
        candidate_ids=("overlap",),
        program_resource=_resource(programs, program_ids=("p1", "p2")),
        precision=np.ones(3),
    )
    second = residualize_against_autonomous_programs(
        candidate,
        feature_ids=("G1", "G2", "G3"),
        candidate_ids=("overlap",),
        program_resource=_resource(scaled, program_ids=("p1", "p2")),
        precision=np.ones(3),
    )

    np.testing.assert_allclose(
        second.residualized_basis,
        first.residualized_basis,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        second.retained_fractions,
        first.retained_fractions,
        atol=1e-12,
    )
    assert second.identifiable.tolist() == first.identifiable.tolist()


def test_residualization_is_precision_weighted_orthogonal() -> None:
    autonomous = np.asarray([[1.0], [1.0], [0.0]])
    resource = _resource(autonomous)
    candidate = np.asarray([[2.0], [-1.0], [3.0]])
    precision = np.asarray([1.0, 4.0, 2.0])

    result = residualize_against_autonomous_programs(
        candidate,
        feature_ids=("G1", "G2", "G3"),
        candidate_ids=("family",),
        precision=precision,
        program_resource=resource,
    )

    weighted_inner_product = autonomous.T @ (
        precision[:, np.newaxis] * result.residualized_basis
    )
    np.testing.assert_allclose(weighted_inner_product, 0.0, atol=1e-12)


def test_precision_support_rank_collapse_is_typed_not_estimable() -> None:
    resource = _resource(
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        program_ids=("cycle", "stress"),
    )

    with pytest.raises(AutonomousProgramSupportError) as error:
        residualize_against_autonomous_programs(
            np.eye(3, 1),
            feature_ids=("G1", "G2", "G3"),
            candidate_ids=("family",),
            precision=np.asarray([1.0, 0.0, 1.0]),
            program_resource=resource,
        )
    assert error.value.reason_code == "autonomous_program_support_not_estimable"


def test_projection_rank_uses_the_frozen_svd_rcond() -> None:
    resource = _resource(
        np.asarray([[1.0, 1.0], [0.0, 1e-14], [0.0, 0.0]]),
        program_ids=("cycle", "stress"),
    )

    with pytest.raises(AutonomousProgramSupportError):
        residualize_against_autonomous_programs(
            np.eye(3, 1),
            feature_ids=("G1", "G2", "G3"),
            candidate_ids=("family",),
            precision=np.ones(3),
            program_resource=resource,
            projection_rcond=1e-12,
        )


def test_ill_conditioned_exact_span_remains_nonidentifiable() -> None:
    rng = np.random.default_rng(4)
    left, _ = np.linalg.qr(rng.normal(size=(10, 3)))
    right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    smallest = 10.0 ** rng.uniform(-11.9, -9.5)
    programs = left @ np.diag([1.0, 0.1, smallest]) @ right.T
    coefficients = rng.normal(size=3)
    exact_span_candidate = programs @ coefficients
    feature_ids = tuple(f"G{index}" for index in range(10))
    resource = _resource(
        programs,
        feature_ids=feature_ids,
        program_ids=("program-0", "program-1", "program-2"),
    )

    result = residualize_against_autonomous_programs(
        exact_span_candidate[:, np.newaxis],
        feature_ids=feature_ids,
        candidate_ids=("exact-span-family",),
        precision=np.ones(10),
        program_resource=resource,
        projection_rcond=1e-12,
        identifiability_tolerance=1e-6,
    )

    assert result.retained_fractions[0] < 1e-10
    assert result.identifiable.tolist() == [False]
    np.testing.assert_allclose(result.residualized_basis, 0.0, atol=1e-10)


def test_residualization_identity_and_integrity_bind_all_numeric_inputs() -> None:
    resource = _resource()
    first = residualize_against_autonomous_programs(
        np.eye(3, 2),
        feature_ids=("G1", "G2", "G3"),
        candidate_ids=("F1", "F2"),
        precision=np.ones(3),
        program_resource=resource,
    )
    changed_precision = residualize_against_autonomous_programs(
        np.eye(3, 2),
        feature_ids=("G1", "G2", "G3"),
        candidate_ids=("F1", "F2"),
        precision=np.asarray([1.0, 2.0, 1.0]),
        program_resource=resource,
    )
    assert first.residualization_id != changed_precision.residualization_id

    object.__setattr__(first, "retained_fractions", first.retained_fractions.copy())
    first.retained_fractions[0] = 0.25
    with pytest.raises(ContractError) as error:
        first.to_dict()
    assert (
        error.value.details.code
        == "autonomous_program_residualization_integrity_violation"
    )
