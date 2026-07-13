"""Static receiver-autonomous programs and identifiability checks."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

import numpy as np

from crychic.core import ContractError, stable_id
from crychic.resources import GeneNamespace, Species
from crychic.resources.autonomous_registry import (
    AutonomousProgramReviewScope,
    require_receiver_autonomous_program_registration,
)
from crychic.resources.manifest import ResourceIntegrityError, ResourceManifest

_PROGRAM_PRODUCER_MARKER = "crychic.response.autonomous_program_resource.v1"
_RESIDUAL_PRODUCER_MARKER = "crychic.response.autonomous_residualization.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DECLARED_UNVERIFIED: Final = "caller_declared_static_unverified"
_MANIFEST_VERIFIED_TRUSTED: Final = "manifest_verified_static_trusted_v1"
_AUTONOMOUS_MATRIX_ROLE: Final = "receiver_autonomous_feature_by_program_matrix_v1"
_FEATURE_HEADER: Final = "feature_id"
_TRUSTED_LOADER_TOKEN: Final = object()

AutonomousProgramVerificationStatus = Literal[
    "caller_declared_static_unverified",
    "manifest_verified_static_trusted_v1",
]


class AutonomousProgramSupportError(ValueError):
    """A declared program resource has no usable aligned precision support."""

    reason_code = "autonomous_program_support_not_estimable"


def _identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, unpadded string")
    return value


def _canonical_names(
    values: Sequence[str], *, field_name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(_identifier(value, field_name=field_name) for value in values)
    if not result and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique strings")
    return tuple(sorted(result))


def _aligned_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    result = tuple(_identifier(value, field_name=field_name) for value in values)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique strings")
    return result


def _immutable_float64(values: np.ndarray) -> np.ndarray:
    canonical = np.asarray(values, dtype="<f8", order="C").copy(order="C")
    if np.any(~np.isfinite(canonical)):
        raise ValueError("receiver-autonomous program arrays must be finite")
    # Make semantically equal +0.0 and -0.0 resources byte-identical.
    canonical[canonical == 0.0] = 0.0
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
            canonical.shape
        ),
    )
    result.setflags(write=False)
    return result


def _immutable_bool(values: np.ndarray) -> np.ndarray:
    canonical = np.asarray(values, dtype="|b1", order="C").copy(order="C")
    result = cast(
        np.ndarray,
        np.frombuffer(canonical.tobytes(order="C"), dtype="|b1").reshape(
            canonical.shape
        ),
    )
    result.setflags(write=False)
    return result


def _is_immutable_byte_backed(values: np.ndarray) -> bool:
    if values.flags.writeable or not values.flags.c_contiguous:
        return False
    base: object = values
    while isinstance(base, np.ndarray):
        base = base.base
    return isinstance(base, bytes)


def _array_digest(values: np.ndarray, *, dtype: str) -> str:
    canonical = np.asarray(values, dtype=dtype, order="C").copy(order="C")
    if np.issubdtype(canonical.dtype, np.floating):
        if np.any(~np.isfinite(canonical)):
            raise ValueError("array digest requires finite values")
        canonical[canonical == 0.0] = 0.0
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _precision_weighted_span_residual(
    values: np.ndarray,
    *,
    span_basis: np.ndarray,
    precision: np.ndarray,
    projection_rcond: float,
) -> tuple[np.ndarray, int]:
    """Remove a feature-space span through its weighted orthonormal basis.

    Projecting with the left singular vectors avoids the unstable coefficient
    back-substitution produced by ``pinv(span_basis)`` for nearly collinear
    program columns. Rows with zero precision are outside the weighted geometry
    and therefore retain their input values.
    """

    observed = np.asarray(values, dtype=np.float64)
    programs = np.asarray(span_basis, dtype=np.float64)
    weights = np.asarray(precision, dtype=np.float64)
    if observed.ndim != 2 or programs.ndim != 2:
        raise ValueError("values and span_basis must be two-dimensional")
    if observed.shape[0] != programs.shape[0] or weights.shape != (observed.shape[0],):
        raise ValueError("weighted projection inputs must share one feature axis")
    if (
        np.any(~np.isfinite(observed))
        or np.any(~np.isfinite(programs))
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("weighted projection inputs must be finite and non-negative")
    if not math.isfinite(projection_rcond) or not 0.0 < projection_rcond < 1.0:
        raise ValueError("projection_rcond must be strictly between zero and one")

    sqrt_precision = np.sqrt(weights)
    weighted_programs = sqrt_precision[:, np.newaxis] * programs
    column_scales = np.max(np.abs(weighted_programs), axis=0, initial=0.0)
    supported_columns = column_scales > 0.0
    normalized_programs = np.zeros_like(weighted_programs)
    if np.any(supported_columns):
        scaled = (
            weighted_programs[:, supported_columns]
            / column_scales[supported_columns]
        )
        scaled_norms = np.sqrt(np.sum(scaled * scaled, axis=0))
        normalized_programs[:, supported_columns] = scaled / scaled_norms
    left_vectors, singular_values, _ = np.linalg.svd(
        normalized_programs,
        full_matrices=False,
    )
    singular_threshold = (
        0.0
        if not singular_values.size
        else projection_rcond * float(singular_values[0])
    )
    numerical_rank = int(np.count_nonzero(singular_values > singular_threshold))
    weighted_values = sqrt_precision[:, np.newaxis] * observed
    if numerical_rank:
        orthonormal_span = left_vectors[:, :numerical_rank]
        weighted_values = weighted_values - orthonormal_span @ (
            orthonormal_span.T @ weighted_values
        )

    residual = np.array(observed, dtype=np.float64, copy=True)
    supported = sqrt_precision > 0.0
    residual[supported] = (
        weighted_values[supported] / sqrt_precision[supported, np.newaxis]
    )
    return residual, numerical_rank


def _resource_payload(
    *,
    resource_id: str,
    version: str,
    manifest_digest: str,
    species: Species,
    gene_namespace: GeneNamespace,
    feature_ids: tuple[str, ...],
    program_ids: tuple[str, ...],
    matrix_digest: str,
    verification_status: AutonomousProgramVerificationStatus,
    registration_id: str | None,
    review_scope: AutonomousProgramReviewScope | None,
    expected_license: str | None,
    registered_payload_path: str | None,
    registered_payload_role: str | None,
    registered_payload_sha256: str | None,
    registered_payload_bytes: int | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "feature_ids": list(feature_ids),
        "manifest_digest": manifest_digest,
        "species": species.value,
        "gene_namespace": gene_namespace.value,
        "matrix_digest": matrix_digest,
        "verification_status": verification_status,
        "program_ids": list(program_ids),
        "resource_id": resource_id,
        "version": version,
    }
    if registration_id is not None:
        payload.update(
            {
                "expected_license": expected_license,
                "registration_id": registration_id,
                "registered_payload_bytes": registered_payload_bytes,
                "registered_payload_path": registered_payload_path,
                "registered_payload_role": registered_payload_role,
                "registered_payload_sha256": registered_payload_sha256,
                "review_scope": review_scope,
            }
        )
    return payload


@dataclass(frozen=True, slots=True, init=False)
class ReceiverAutonomousProgramResource:
    """Immutable static feature-by-program nuisance basis.

    Signed weights are permitted. Both axes are stored in canonical lexical order,
    so the artifact identity is independent of the input row and column order.
    The resource contains no expression-derived or fold-learned programs.
    """

    resource_id: str
    version: str
    manifest_digest: str
    species: Species
    gene_namespace: GeneNamespace
    feature_ids: tuple[str, ...]
    program_ids: tuple[str, ...]
    matrix: np.ndarray
    matrix_digest: str
    verification_status: AutonomousProgramVerificationStatus
    registration_id: str | None
    review_scope: AutonomousProgramReviewScope | None
    expected_license: str | None
    registered_payload_path: str | None
    registered_payload_role: str | None
    registered_payload_sha256: str | None
    registered_payload_bytes: int | None
    artifact_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "ReceiverAutonomousProgramResource is producer-owned; "
            "use build_receiver_autonomous_program_resource()"
        )

    @classmethod
    def _from_resource(
        cls,
        *,
        resource_id: str,
        version: str,
        manifest_digest: str,
        species: Species,
        gene_namespace: GeneNamespace,
        feature_ids: tuple[str, ...],
        program_ids: tuple[str, ...],
        matrix: np.ndarray,
        verification_status: AutonomousProgramVerificationStatus,
        registration_id: str | None = None,
        review_scope: AutonomousProgramReviewScope | None = None,
        expected_license: str | None = None,
        registered_payload_path: str | None = None,
        registered_payload_role: str | None = None,
        registered_payload_sha256: str | None = None,
        registered_payload_bytes: int | None = None,
        trusted_loader_token: object | None = None,
    ) -> ReceiverAutonomousProgramResource:
        if (
            verification_status == _MANIFEST_VERIFIED_TRUSTED
            and trusted_loader_token is not _TRUSTED_LOADER_TOKEN
        ):
            raise ValueError(
                "manifest-verified autonomous resources require the trusted loader"
            )
        if (
            verification_status != _MANIFEST_VERIFIED_TRUSTED
            and trusted_loader_token is not None
        ):
            raise ValueError(
                "trusted loader token is inconsistent with resource status"
            )
        registration_fields = (
            registration_id,
            review_scope,
            expected_license,
            registered_payload_path,
            registered_payload_role,
            registered_payload_sha256,
            registered_payload_bytes,
        )
        if verification_status == _MANIFEST_VERIFIED_TRUSTED and any(
            value is None for value in registration_fields
        ):
            raise ValueError("trusted resources require complete registration evidence")
        if verification_status != _MANIFEST_VERIFIED_TRUSTED and any(
            value is not None for value in registration_fields
        ):
            raise ValueError("unverified resources cannot claim registration evidence")
        frozen = _immutable_float64(matrix)
        matrix_digest = _array_digest(frozen, dtype="<f8")
        payload = _resource_payload(
            resource_id=resource_id,
            version=version,
            manifest_digest=manifest_digest,
            species=species,
            gene_namespace=gene_namespace,
            feature_ids=feature_ids,
            program_ids=program_ids,
            matrix_digest=matrix_digest,
            verification_status=verification_status,
            registration_id=registration_id,
            review_scope=review_scope,
            expected_license=expected_license,
            registered_payload_path=registered_payload_path,
            registered_payload_role=registered_payload_role,
            registered_payload_sha256=registered_payload_sha256,
            registered_payload_bytes=registered_payload_bytes,
        )
        self = object.__new__(cls)
        attributes: dict[str, Any] = {
            "resource_id": resource_id,
            "version": version,
            "manifest_digest": manifest_digest,
            "species": species,
            "gene_namespace": gene_namespace,
            "feature_ids": feature_ids,
            "program_ids": program_ids,
            "matrix": frozen,
            "matrix_digest": matrix_digest,
            "verification_status": verification_status,
            "registration_id": registration_id,
            "review_scope": review_scope,
            "expected_license": expected_license,
            "registered_payload_path": registered_payload_path,
            "registered_payload_role": registered_payload_role,
            "registered_payload_sha256": registered_payload_sha256,
            "registered_payload_bytes": registered_payload_bytes,
            "artifact_id": stable_id(
                "receiver_autonomous_program_resource", payload, schema_version="1"
            ),
            "_producer_marker": _PROGRAM_PRODUCER_MARKER,
        }
        for name, value in attributes.items():
            object.__setattr__(self, name, value)
        return self

    def _require_producer_owned(self) -> None:
        """Validate producer lineage and the complete resource identity."""

        try:
            resource_id = _identifier(self.resource_id, field_name="resource_id")
            version = _identifier(self.version, field_name="version")
            if _SHA256_PATTERN.fullmatch(self.manifest_digest) is None:
                raise ValueError("manifest_digest is not a lowercase SHA-256 digest")
            species = Species(self.species)
            namespace = GeneNamespace(self.gene_namespace)
            features = _canonical_names(self.feature_ids, field_name="feature_ids")
            programs = _canonical_names(self.program_ids, field_name="program_ids")
            if features != self.feature_ids or programs != self.program_ids:
                raise ValueError("program resource axes are not canonical")
            if not _is_immutable_byte_backed(self.matrix):
                raise ValueError("program resource matrix is not immutable")
            if self.matrix.shape != (len(features), len(programs)):
                raise ValueError("program resource matrix is misaligned")
            if np.any(~np.isfinite(self.matrix)):
                raise ValueError("program resource matrix is not finite")
            if np.any(~np.any(self.matrix != 0.0, axis=0)):
                raise ValueError("program resource contains an all-zero program")
            matrix_digest = _array_digest(self.matrix, dtype="<f8")
            if self.verification_status not in {
                _DECLARED_UNVERIFIED,
                _MANIFEST_VERIFIED_TRUSTED,
            }:
                raise ValueError("unsupported autonomous resource verification status")
            registration_fields = (
                self.registration_id,
                self.review_scope,
                self.expected_license,
                self.registered_payload_path,
                self.registered_payload_role,
                self.registered_payload_sha256,
                self.registered_payload_bytes,
            )
            if self.verification_status == _MANIFEST_VERIFIED_TRUSTED:
                if any(value is None for value in registration_fields):
                    raise ValueError("trusted registration evidence is incomplete")
                registration_id = _identifier(
                    cast(str, self.registration_id), field_name="registration_id"
                )
                expected_license = _identifier(
                    cast(str, self.expected_license), field_name="expected_license"
                )
                registered_payload_path = _identifier(
                    cast(str, self.registered_payload_path),
                    field_name="registered_payload_path",
                )
                registered_payload_role = _identifier(
                    cast(str, self.registered_payload_role),
                    field_name="registered_payload_role",
                )
                registered_payload_sha256 = cast(
                    str, self.registered_payload_sha256
                )
                if (
                    _SHA256_PATTERN.fullmatch(registered_payload_sha256) is None
                    or not isinstance(self.registered_payload_bytes, int)
                    or self.registered_payload_bytes < 0
                ):
                    raise ValueError("registered payload evidence is invalid")
                review_scope = cast(AutonomousProgramReviewScope, self.review_scope)
                if review_scope not in {
                    "synthetic_benchmark_only",
                    "biological_reference",
                }:
                    raise ValueError("unsupported autonomous program review scope")
                registration = require_receiver_autonomous_program_registration(
                    registration_id
                )
                if (
                    registration.manifest_digest != self.manifest_digest
                    or registration.resource_id != resource_id
                    or registration.version != version
                    or registration.species != species
                    or registration.gene_namespace != namespace
                    or registration.expected_license != expected_license
                    or registration.review_scope != review_scope
                    or registration.expected_matrix_digest != matrix_digest
                    or registration.payload_path != registered_payload_path
                    or registration.payload_role != registered_payload_role
                    or registration.payload_sha256 != registered_payload_sha256
                    or registration.payload_bytes != self.registered_payload_bytes
                ):
                    raise ValueError("resource no longer matches its code registration")
            elif any(value is not None for value in registration_fields):
                raise ValueError("unverified resources claim registration evidence")
            expected_id = stable_id(
                "receiver_autonomous_program_resource",
                _resource_payload(
                    resource_id=resource_id,
                    version=version,
                    manifest_digest=self.manifest_digest,
                    species=species,
                    gene_namespace=namespace,
                    feature_ids=features,
                    program_ids=programs,
                    matrix_digest=matrix_digest,
                    verification_status=self.verification_status,
                    registration_id=self.registration_id,
                    review_scope=self.review_scope,
                    expected_license=self.expected_license,
                    registered_payload_path=self.registered_payload_path,
                    registered_payload_role=self.registered_payload_role,
                    registered_payload_sha256=self.registered_payload_sha256,
                    registered_payload_bytes=self.registered_payload_bytes,
                ),
                schema_version="1",
            )
        except (AttributeError, ContractError, TypeError, ValueError) as error:
            raise ContractError(
                "Receiver-autonomous program resource failed integrity validation",
                code="receiver_autonomous_program_integrity_violation",
                field="artifact_id",
                remediation="Rebuild the program resource from its declared manifest",
            ) from error
        if (
            self._producer_marker != _PROGRAM_PRODUCER_MARKER
            or matrix_digest != self.matrix_digest
            or expected_id != self.artifact_id
        ):
            raise ContractError(
                "Receiver-autonomous program resource failed integrity validation",
                code="receiver_autonomous_program_integrity_violation",
                field="artifact_id",
                remediation="Rebuild the program resource from its declared manifest",
            )

    @property
    def is_manifest_verified_trusted(self) -> bool:
        """Report whether the artifact came from the checksum-pinned loader."""

        self._require_producer_owned()
        return self.verification_status == _MANIFEST_VERIFIED_TRUSTED

    @property
    def is_biological_reference_trusted(self) -> bool:
        """Report whether registry review permits biological-reference use."""

        self._require_producer_owned()
        return (
            self.verification_status == _MANIFEST_VERIFIED_TRUSTED
            and self.review_scope == "biological_reference"
        )

    def matrix_for_features(self, feature_ids: Sequence[str]) -> np.ndarray:
        """Return the frozen basis aligned to a requested feature universe.

        Features absent from the caller-declared static resource receive zero
        weights. This permits a broad expression universe while retaining an
        explicit resource support set. Requested order is preserved exactly.
        """

        self._require_producer_owned()
        requested = _aligned_names(feature_ids, field_name="feature_ids")
        source_row = {feature: index for index, feature in enumerate(self.feature_ids)}
        aligned: np.ndarray = np.zeros(
            (len(requested), len(self.program_ids)), dtype=np.float64
        )
        for output_row, feature in enumerate(requested):
            input_row = source_row.get(feature)
            if input_row is not None:
                aligned[output_row] = self.matrix[input_row]
        return _immutable_float64(aligned)

    def to_dict(self) -> dict[str, object]:
        """Return resource provenance without expanding the numeric matrix."""

        self._require_producer_owned()
        return {
            "artifact_id": self.artifact_id,
            "resource_id": self.resource_id,
            "version": self.version,
            "manifest_digest": self.manifest_digest,
            "species": self.species.value,
            "gene_namespace": self.gene_namespace.value,
            "feature_ids": list(self.feature_ids),
            "program_ids": list(self.program_ids),
            "matrix_digest": self.matrix_digest,
            "verification_status": self.verification_status,
            "registration_id": self.registration_id,
            "review_scope": self.review_scope,
            "expected_license": self.expected_license,
            "registered_payload_path": self.registered_payload_path,
            "registered_payload_role": self.registered_payload_role,
            "registered_payload_sha256": self.registered_payload_sha256,
            "registered_payload_bytes": self.registered_payload_bytes,
        }


def build_receiver_autonomous_program_resource(
    matrix: np.ndarray,
    *,
    feature_ids: Sequence[str],
    program_ids: Sequence[str],
    resource_id: str,
    version: str,
    manifest_digest: str,
    species: Species,
    gene_namespace: GeneNamespace,
) -> ReceiverAutonomousProgramResource:
    """Build an explicitly unverified caller-declared static program resource."""

    return _build_receiver_autonomous_program_resource(
        matrix,
        feature_ids=feature_ids,
        program_ids=program_ids,
        resource_id=resource_id,
        version=version,
        manifest_digest=manifest_digest,
        species=species,
        gene_namespace=gene_namespace,
        verification_status=_DECLARED_UNVERIFIED,
    )


def _build_receiver_autonomous_program_resource(
    matrix: np.ndarray,
    *,
    feature_ids: Sequence[str],
    program_ids: Sequence[str],
    resource_id: str,
    version: str,
    manifest_digest: str,
    species: Species,
    gene_namespace: GeneNamespace,
    verification_status: AutonomousProgramVerificationStatus,
    registration_id: str | None = None,
    review_scope: AutonomousProgramReviewScope | None = None,
    expected_license: str | None = None,
    registered_payload_path: str | None = None,
    registered_payload_role: str | None = None,
    registered_payload_sha256: str | None = None,
    registered_payload_bytes: int | None = None,
    trusted_loader_token: object | None = None,
) -> ReceiverAutonomousProgramResource:
    """Canonicalize a resource after its provenance status has been established."""

    source_features = _aligned_names(feature_ids, field_name="feature_ids")
    source_programs = _aligned_names(program_ids, field_name="program_ids")
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape != (
        len(source_features),
        len(source_programs),
    ):
        raise ValueError("matrix must align with feature_ids and program_ids")
    if np.any(~np.isfinite(values)):
        raise ValueError("matrix must contain only finite signed weights")

    canonical_features = _canonical_names(source_features, field_name="feature_ids")
    canonical_programs = _canonical_names(source_programs, field_name="program_ids")
    feature_index = {value: index for index, value in enumerate(source_features)}
    program_index = {value: index for index, value in enumerate(source_programs)}
    canonical = values[
        np.ix_(
            [feature_index[value] for value in canonical_features],
            [program_index[value] for value in canonical_programs],
        )
    ]
    if np.any(~np.any(canonical != 0.0, axis=0)):
        raise ValueError("each receiver-autonomous program must have non-zero support")

    upstream_id = _identifier(resource_id, field_name="resource_id")
    upstream_version = _identifier(version, field_name="version")
    try:
        resolved_species = Species(species)
        resolved_namespace = GeneNamespace(gene_namespace)
    except ValueError as error:
        raise ValueError(
            "species and gene_namespace must be supported enums"
        ) from error
    if not isinstance(manifest_digest, str) or (
        _SHA256_PATTERN.fullmatch(manifest_digest) is None
    ):
        raise ValueError("manifest_digest must be a lowercase SHA-256 digest")
    return ReceiverAutonomousProgramResource._from_resource(
        resource_id=upstream_id,
        version=upstream_version,
        manifest_digest=manifest_digest,
        species=resolved_species,
        gene_namespace=resolved_namespace,
        feature_ids=canonical_features,
        program_ids=canonical_programs,
        matrix=canonical,
        verification_status=verification_status,
        registration_id=registration_id,
        review_scope=review_scope,
        expected_license=expected_license,
        registered_payload_path=registered_payload_path,
        registered_payload_role=registered_payload_role,
        registered_payload_sha256=registered_payload_sha256,
        registered_payload_bytes=registered_payload_bytes,
        trusted_loader_token=trusted_loader_token,
    )


def _raise_resource_metadata_mismatch(
    *, field_name: str, observed: str, expected: str
) -> None:
    raise ResourceIntegrityError(
        f"Resource manifest {field_name} mismatch: {observed!r} != {expected!r}",
        code="resource_metadata_mismatch",
        field=field_name,
        remediation="Select the exact reviewed receiver-autonomous resource release",
    )


def _parse_receiver_autonomous_tsv(
    payload: bytes,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    try:
        text = payload.decode("utf-8")
        rows = list(
            csv.reader(
                io.StringIO(text, newline=""), delimiter="\t", strict=True
            )
        )
    except (UnicodeDecodeError, csv.Error) as error:
        raise ResourceIntegrityError(
            "Receiver-autonomous program payload must be valid UTF-8 TSV",
            code="invalid_receiver_autonomous_program_payload",
            field="payload",
            remediation="Regenerate the payload with the documented TSV schema",
        ) from error
    if not rows or len(rows[0]) < 2 or rows[0][0] != _FEATURE_HEADER:
        raise ResourceIntegrityError(
            "Receiver-autonomous TSV header must start with feature_id and programs",
            code="invalid_receiver_autonomous_program_payload",
            field="header",
            remediation=(
                "Use feature_id as the first column and one program ID per "
                "remaining column"
            ),
        )
    try:
        program_ids = _aligned_names(rows[0][1:], field_name="program_ids")
        feature_ids: list[str] = []
        weights: list[list[float]] = []
        expected_columns = len(rows[0])
        for line_number, row in enumerate(rows[1:], start=2):
            if len(row) != expected_columns:
                raise ValueError(
                    f"line {line_number} has {len(row)} columns; "
                    f"expected {expected_columns}"
                )
            feature_ids.append(_identifier(row[0], field_name="feature_id"))
            weights.append([float(value) for value in row[1:]])
        aligned_features = _aligned_names(feature_ids, field_name="feature_ids")
        matrix = np.asarray(weights, dtype=np.float64)
        if matrix.shape != (len(aligned_features), len(program_ids)):
            raise ValueError("payload must contain at least one feature row")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("program weights must be finite")
    except (TypeError, ValueError) as error:
        raise ResourceIntegrityError(
            "Receiver-autonomous TSV contains invalid axes or numeric weights",
            code="invalid_receiver_autonomous_program_payload",
            field="payload",
            remediation=(
                "Use unique unpadded IDs and a complete finite signed weight matrix"
            ),
        ) from error
    return matrix, aligned_features, program_ids


def _resolve_registered_payload(
    database_root: str | Path,
    *,
    relative_path: str,
) -> tuple[Path, Path]:
    """Resolve one registered payload without following links below its root."""

    try:
        root = Path(database_root).resolve(strict=True)
    except OSError as error:
        raise ResourceIntegrityError(
            "Receiver-autonomous database root does not exist",
            code="missing_resource_payload",
            field="database_root",
            remediation="Provide the directory containing the registered payload",
        ) from error
    if not root.is_dir():
        raise ResourceIntegrityError(
            "Receiver-autonomous database root must be a directory",
            code="invalid_resource_path",
            field="database_root",
            remediation="Provide the directory containing the registered payload",
        )

    candidate = root
    for part in PurePosixPath(relative_path).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ResourceIntegrityError(
                f"Receiver-autonomous payload path contains a symlink: {relative_path}",
                code="resource_payload_symlink",
                field="path",
                remediation="Store the registered payload directly below database_root",
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ResourceIntegrityError(
            f"Resource payload is missing: {relative_path}",
            code="missing_resource_payload",
            field="path",
            remediation="Restore the checksum-pinned registered payload",
        ) from error
    if not resolved.is_relative_to(root):
        raise ResourceIntegrityError(
            "Receiver-autonomous payload resolves outside database_root",
            code="resource_path_escape",
            field="path",
            remediation="Store the registered payload directly below database_root",
        )
    if not resolved.is_file():
        raise ResourceIntegrityError(
            f"Receiver-autonomous payload is not a file: {relative_path}",
            code="missing_resource_payload",
            field="path",
            remediation="Restore the checksum-pinned registered payload file",
        )
    return root, resolved


def load_receiver_autonomous_program_resource(
    database_root: str | Path,
    *,
    manifest_path: str | Path,
    registration_id: str,
) -> ReceiverAutonomousProgramResource:
    """Load a reviewed static feature-by-program matrix without expression data.

    The code registry, rather than caller-provided metadata, pins the manifest
    digest, release, species, namespace, license, adapter and review scope. The
    exact bytes parsed are hashed again after ``ResourceManifest.verify``.
    """

    registration = require_receiver_autonomous_program_registration(registration_id)
    manifest = ResourceManifest.from_json(manifest_path)
    if manifest.digest != registration.manifest_digest:
        raise ResourceIntegrityError(
            "Receiver-autonomous manifest does not match its code registration",
            code="resource_manifest_digest_mismatch",
            field="manifest_digest",
            remediation="Restore the reviewed manifest registered in this release",
        )
    manifest.require(
        species=registration.species.value,
        gene_namespace=registration.gene_namespace.value,
        license=registration.expected_license,
    )
    if manifest.resource_id != registration.resource_id:
        _raise_resource_metadata_mismatch(
            field_name="resource_id",
            observed=manifest.resource_id,
            expected=registration.resource_id,
        )
    if manifest.version != registration.version:
        _raise_resource_metadata_mismatch(
            field_name="version",
            observed=manifest.version,
            expected=registration.version,
        )
    if manifest.adapter_version != registration.adapter_version:
        _raise_resource_metadata_mismatch(
            field_name="adapter_version",
            observed=manifest.adapter_version,
            expected=registration.adapter_version,
        )

    candidates = tuple(
        payload
        for payload in manifest.payloads
        if payload.role == _AUTONOMOUS_MATRIX_ROLE
    )
    if len(candidates) != 1:
        raise ResourceIntegrityError(
            "Manifest must pin exactly one receiver-autonomous matrix payload",
            code="ambiguous_receiver_autonomous_program_payload",
            field="payloads",
            remediation=(
                f"Assign exactly one payload the role {_AUTONOMOUS_MATRIX_ROLE!r}"
            ),
        )
    matrix_payload = candidates[0]
    if (
        matrix_payload.path != registration.payload_path
        or matrix_payload.role != registration.payload_role
        or matrix_payload.sha256 != registration.payload_sha256
        or matrix_payload.bytes != registration.payload_bytes
    ):
        raise ResourceIntegrityError(
            "Receiver-autonomous payload does not match its code registration",
            code="resource_registration_payload_mismatch",
            field="payloads",
            remediation="Restore the exact payload record registered in this release",
        )
    if matrix_payload.bytes is None:
        raise ResourceIntegrityError(
            "Trusted receiver-autonomous matrix payload must pin its byte size",
            code="incomplete_resource_manifest",
            field="bytes",
            remediation="Record the exact payload byte size in the reviewed manifest",
        )
    if PurePosixPath(matrix_payload.path).suffix != ".tsv":
        raise ResourceIntegrityError(
            "Receiver-autonomous matrix payload must use the .tsv schema",
            code="invalid_receiver_autonomous_program_payload",
            field="path",
            remediation="Export the static feature-by-program matrix as UTF-8 TSV",
        )

    root, payload_path = _resolve_registered_payload(
        database_root,
        relative_path=matrix_payload.path,
    )
    manifest.verify(root, paths=(matrix_payload.path,))
    try:
        payload_bytes = payload_path.read_bytes()
    except OSError as error:  # pragma: no cover - verify normally catches this first
        raise ResourceIntegrityError(
            f"Cannot read receiver-autonomous payload {matrix_payload.path}",
            code="missing_resource_payload",
            field="path",
            remediation="Restore the checksum-pinned static payload",
        ) from error
    if (
        hashlib.sha256(payload_bytes).hexdigest() != matrix_payload.sha256
        or len(payload_bytes) != matrix_payload.bytes
    ):
        raise ResourceIntegrityError(
            "Receiver-autonomous payload changed between verification and parsing",
            code="resource_checksum_mismatch",
            field="sha256",
            remediation="Restore the unmodified checksum-pinned static payload",
        )
    matrix, feature_ids, program_ids = _parse_receiver_autonomous_tsv(payload_bytes)
    try:
        resource = _build_receiver_autonomous_program_resource(
            matrix,
            feature_ids=feature_ids,
            program_ids=program_ids,
            resource_id=manifest.resource_id,
            version=manifest.version,
            manifest_digest=manifest.digest,
            species=registration.species,
            gene_namespace=registration.gene_namespace,
            verification_status=_MANIFEST_VERIFIED_TRUSTED,
            registration_id=registration.registration_id,
            review_scope=registration.review_scope,
            expected_license=registration.expected_license,
            registered_payload_path=registration.payload_path,
            registered_payload_role=registration.payload_role,
            registered_payload_sha256=registration.payload_sha256,
            registered_payload_bytes=registration.payload_bytes,
            trusted_loader_token=_TRUSTED_LOADER_TOKEN,
        )
        resource._require_producer_owned()
        return resource
    except (ContractError, TypeError, ValueError) as error:
        raise ResourceIntegrityError(
            "Receiver-autonomous program matrix violates the artifact contract",
            code="invalid_receiver_autonomous_program_payload",
            field="payload",
            remediation="Regenerate and review the static feature-by-program matrix",
        ) from error


def _residual_payload(
    *,
    feature_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    program_resource_artifact_id: str,
    candidate_basis_digest: str,
    precision_digest: str,
    residualized_basis_digest: str,
    original_weighted_norms_digest: str,
    residual_weighted_norms_digest: str,
    retained_fractions_digest: str,
    identifiable_digest: str,
    projection_rcond: float,
    identifiability_tolerance: float,
) -> dict[str, object]:
    return {
        "candidate_basis_digest": candidate_basis_digest,
        "candidate_ids": list(candidate_ids),
        "feature_ids": list(feature_ids),
        "identifiability_tolerance": identifiability_tolerance,
        "identifiable_digest": identifiable_digest,
        "original_weighted_norms_digest": original_weighted_norms_digest,
        "precision_digest": precision_digest,
        "program_resource_artifact_id": program_resource_artifact_id,
        "projection_rcond": projection_rcond,
        "residual_weighted_norms_digest": residual_weighted_norms_digest,
        "residualized_basis_digest": residualized_basis_digest,
        "retained_fractions_digest": retained_fractions_digest,
    }


@dataclass(frozen=True, slots=True, init=False)
class AutonomousProgramResidualization:
    """Precision-weighted candidate basis after removing autonomous programs."""

    feature_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    program_resource_artifact_id: str
    residualized_basis: np.ndarray
    original_weighted_norms: np.ndarray
    residual_weighted_norms: np.ndarray
    retained_fractions: np.ndarray
    identifiable: np.ndarray
    projection_rcond: float
    identifiability_tolerance: float
    candidate_basis_digest: str
    precision_digest: str
    residualized_basis_digest: str
    original_weighted_norms_digest: str
    residual_weighted_norms_digest: str
    retained_fractions_digest: str
    identifiable_digest: str
    residualization_id: str
    _producer_marker: str

    def __init__(self) -> None:
        raise TypeError(
            "AutonomousProgramResidualization is producer-owned; "
            "use residualize_against_autonomous_programs()"
        )

    @property
    def identifiable_candidate_ids(self) -> tuple[str, ...]:
        """Return candidates retaining unique precision-weighted support."""

        self._require_producer_owned()
        return tuple(
            candidate
            for candidate, keep in zip(
                self.candidate_ids, self.identifiable, strict=True
            )
            if bool(keep)
        )

    @property
    def nonidentifiable_candidate_ids(self) -> tuple[str, ...]:
        """Return candidates contained or nearly contained in nuisance span."""

        self._require_producer_owned()
        return tuple(
            candidate
            for candidate, keep in zip(
                self.candidate_ids, self.identifiable, strict=True
            )
            if not bool(keep)
        )

    def _require_producer_owned(self) -> None:
        try:
            features = _aligned_names(self.feature_ids, field_name="feature_ids")
            candidates = _aligned_names(self.candidate_ids, field_name="candidate_ids")
            if not _identifier(
                self.program_resource_artifact_id,
                field_name="program_resource_artifact_id",
            ):
                raise ValueError("program resource artifact ID is invalid")
            arrays = (
                self.residualized_basis,
                self.original_weighted_norms,
                self.residual_weighted_norms,
                self.retained_fractions,
                self.identifiable,
            )
            if any(not _is_immutable_byte_backed(values) for values in arrays):
                raise ValueError("residualization arrays are not immutable")
            if self.residualized_basis.shape != (len(features), len(candidates)):
                raise ValueError("residualized basis is misaligned")
            if any(values.shape != (len(candidates),) for values in arrays[1:]):
                raise ValueError("residualization summaries are misaligned")
            digests = (
                _array_digest(self.residualized_basis, dtype="<f8"),
                _array_digest(self.original_weighted_norms, dtype="<f8"),
                _array_digest(self.residual_weighted_norms, dtype="<f8"),
                _array_digest(self.retained_fractions, dtype="<f8"),
                _array_digest(self.identifiable, dtype="|b1"),
            )
            expected_id = stable_id(
                "autonomous_program_residualization",
                _residual_payload(
                    feature_ids=features,
                    candidate_ids=candidates,
                    program_resource_artifact_id=self.program_resource_artifact_id,
                    candidate_basis_digest=self.candidate_basis_digest,
                    precision_digest=self.precision_digest,
                    residualized_basis_digest=digests[0],
                    original_weighted_norms_digest=digests[1],
                    residual_weighted_norms_digest=digests[2],
                    retained_fractions_digest=digests[3],
                    identifiable_digest=digests[4],
                    projection_rcond=self.projection_rcond,
                    identifiability_tolerance=self.identifiability_tolerance,
                ),
                schema_version="1",
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "Autonomous-program residualization failed integrity validation",
                code="autonomous_program_residualization_integrity_violation",
                field="residualization_id",
                remediation=(
                    "Recompute residualization from the declared static "
                    "program resource"
                ),
            ) from error
        observed_digests = (
            self.residualized_basis_digest,
            self.original_weighted_norms_digest,
            self.residual_weighted_norms_digest,
            self.retained_fractions_digest,
            self.identifiable_digest,
        )
        if (
            self._producer_marker != _RESIDUAL_PRODUCER_MARKER
            or observed_digests != digests
            or expected_id != self.residualization_id
        ):
            raise ContractError(
                "Autonomous-program residualization failed integrity validation",
                code="autonomous_program_residualization_integrity_violation",
                field="residualization_id",
                remediation=(
                    "Recompute residualization from the declared static "
                    "program resource"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        """Return residualization provenance and compact diagnostics."""

        self._require_producer_owned()
        return {
            "residualization_id": self.residualization_id,
            "program_resource_artifact_id": self.program_resource_artifact_id,
            "feature_ids": list(self.feature_ids),
            "candidate_ids": list(self.candidate_ids),
            "candidate_basis_digest": self.candidate_basis_digest,
            "precision_digest": self.precision_digest,
            "residualized_basis_digest": self.residualized_basis_digest,
            "original_weighted_norms": self.original_weighted_norms.tolist(),
            "residual_weighted_norms": self.residual_weighted_norms.tolist(),
            "retained_fractions": self.retained_fractions.tolist(),
            "identifiable": self.identifiable.tolist(),
            "projection_rcond": self.projection_rcond,
            "identifiability_tolerance": self.identifiability_tolerance,
        }


def residualize_against_autonomous_programs(
    candidate_basis: np.ndarray,
    *,
    feature_ids: Sequence[str],
    candidate_ids: Sequence[str],
    precision: np.ndarray,
    program_resource: ReceiverAutonomousProgramResource,
    projection_rcond: float = 1e-12,
    identifiability_tolerance: float = 1e-6,
) -> AutonomousProgramResidualization:
    """Remove the caller-declared autonomous span in precision-weighted space.

    A candidate is identifiable only when its residual weighted norm divided by
    its original weighted norm is strictly greater than
    ``identifiability_tolerance``. Completely unsupported candidates are therefore
    non-identifiable rather than silently retained.
    """

    if not isinstance(program_resource, ReceiverAutonomousProgramResource):
        raise TypeError("program_resource must be a ReceiverAutonomousProgramResource")
    program_resource._require_producer_owned()
    features = _aligned_names(feature_ids, field_name="feature_ids")
    candidates = _aligned_names(candidate_ids, field_name="candidate_ids")
    basis = np.asarray(candidate_basis, dtype=np.float64)
    weights = np.asarray(precision, dtype=np.float64)
    if basis.ndim != 2 or basis.shape != (len(features), len(candidates)):
        raise ValueError(
            "candidate_basis must align with feature_ids and candidate_ids"
        )
    if np.any(~np.isfinite(basis)):
        raise ValueError("candidate_basis must contain only finite values")
    if weights.ndim != 1 or weights.shape != (len(features),):
        raise ValueError("precision must align one-to-one with feature_ids")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("precision must contain finite non-negative values")
    if not np.any(weights > 0.0):
        raise ValueError("precision must contain at least one positive value")
    for value, name in (
        (projection_rcond, "projection_rcond"),
        (identifiability_tolerance, "identifiability_tolerance"),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a finite numeric scalar")
        if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
            raise ValueError(f"{name} must be strictly between zero and one")
    rcond = float(projection_rcond)
    tolerance = float(identifiability_tolerance)

    autonomous_basis = program_resource.matrix_for_features(features)
    residual, numerical_rank = _precision_weighted_span_residual(
        basis,
        span_basis=autonomous_basis,
        precision=weights,
        projection_rcond=rcond,
    )
    if numerical_rank != len(program_resource.program_ids):
        raise AutonomousProgramSupportError(
            "autonomous programs lack full rank on precision-supported features"
        )
    sqrt_precision = np.sqrt(weights)
    weighted_candidates = sqrt_precision[:, np.newaxis] * basis
    weighted_residual = sqrt_precision[:, np.newaxis] * residual
    original_norms = np.linalg.norm(weighted_candidates, axis=0)
    residual_norms = np.linalg.norm(weighted_residual, axis=0)
    retained = np.zeros_like(original_norms)
    supported = original_norms > 0.0
    retained[supported] = residual_norms[supported] / original_norms[supported]
    # Numerical cancellation can exceed one by a few ulps; the ratio is a fraction.
    retained = np.minimum(retained, 1.0)
    identifiable = retained > tolerance

    frozen_residual = _immutable_float64(residual)
    frozen_original = _immutable_float64(original_norms)
    frozen_residual_norms = _immutable_float64(residual_norms)
    frozen_retained = _immutable_float64(retained)
    frozen_identifiable = _immutable_bool(identifiable)
    candidate_digest = _array_digest(basis, dtype="<f8")
    precision_digest = _array_digest(weights, dtype="<f8")
    residual_digest = _array_digest(frozen_residual, dtype="<f8")
    original_digest = _array_digest(frozen_original, dtype="<f8")
    residual_norm_digest = _array_digest(frozen_residual_norms, dtype="<f8")
    retained_digest = _array_digest(frozen_retained, dtype="<f8")
    identifiable_digest = _array_digest(frozen_identifiable, dtype="|b1")
    payload = _residual_payload(
        feature_ids=features,
        candidate_ids=candidates,
        program_resource_artifact_id=program_resource.artifact_id,
        candidate_basis_digest=candidate_digest,
        precision_digest=precision_digest,
        residualized_basis_digest=residual_digest,
        original_weighted_norms_digest=original_digest,
        residual_weighted_norms_digest=residual_norm_digest,
        retained_fractions_digest=retained_digest,
        identifiable_digest=identifiable_digest,
        projection_rcond=rcond,
        identifiability_tolerance=tolerance,
    )
    self = object.__new__(AutonomousProgramResidualization)
    attributes: dict[str, Any] = {
        "feature_ids": features,
        "candidate_ids": candidates,
        "program_resource_artifact_id": program_resource.artifact_id,
        "residualized_basis": frozen_residual,
        "original_weighted_norms": frozen_original,
        "residual_weighted_norms": frozen_residual_norms,
        "retained_fractions": frozen_retained,
        "identifiable": frozen_identifiable,
        "projection_rcond": rcond,
        "identifiability_tolerance": tolerance,
        "candidate_basis_digest": candidate_digest,
        "precision_digest": precision_digest,
        "residualized_basis_digest": residual_digest,
        "original_weighted_norms_digest": original_digest,
        "residual_weighted_norms_digest": residual_norm_digest,
        "retained_fractions_digest": retained_digest,
        "identifiable_digest": identifiable_digest,
        "residualization_id": stable_id(
            "autonomous_program_residualization", payload, schema_version="1"
        ),
        "_producer_marker": _RESIDUAL_PRODUCER_MARKER,
    }
    for name, value in attributes.items():
        object.__setattr__(self, name, value)
    return self
