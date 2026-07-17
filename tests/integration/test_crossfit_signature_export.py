from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from anndata import AnnData
from scipy import sparse

from crychic.attribution import PenaltyTuningSpec
from crychic.core import CrychicConfig
from crychic.design import balanced_contrast
from crychic.resources import (
    GeneNamespace,
    Interaction,
    MappingReport,
    ResourceBundle,
    Species,
    TargetPrior,
)
from crychic.response import build_receiver_autonomous_program_resource
from crychic.sender import ContrastCommonSenderParameters
from crychic.signatures import LayeredSignatureStatus
from crychic.workflow.crossfit import (
    CrossFitArtifacts,
    CrossFitSpec,
    FoldTrainingSpec,
    run_subject_crossfit,
)
from crychic.workflow.signature_export import (
    GENE_MODEL_COORDINATE_VERSION,
    CrossFitLayeredSignatureExport,
    SignatureExportAvailabilityStatus,
    SignatureExportLayer,
    export_crossfit_layered_signatures,
)


def _interaction(interaction_id: str, ligand: str, receptor: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        source_interaction_id=interaction_id,
        ligand_name=ligand,
        receptor_name=receptor,
        ligand_subunits=(ligand,),
        receptor_subunits=(receptor,),
        ligand_is_complex=False,
        receptor_is_complex=False,
        direction="Ligand-Receptor",
        source="signature-export-fixture",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )


def _bundle() -> ResourceBundle:
    return ResourceBundle(
        resource_id="signature_export_lr",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        interactions=(
            _interaction("i1", "L1", "R1"),
            _interaction("i2", "L2", "R2"),
        ),
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="c" * 64,
        source_files=("signature-export.tsv",),
        license="CC0",
        citation="Synthetic signature-export fixture",
    )


def _prior() -> TargetPrior:
    return TargetPrior(
        resource_id="signature_export_prior",
        version="1",
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
        driver_kind="interaction",
        target_ids=("T1", "T2"),
        driver_ids=("i1", "i2"),
        indptr=(0, 1, 2),
        target_indices=(0, 1),
        weights=(1.0, 1.0),
        ranks=None,
        direction=1,
        evidence="synthetic",
        mapping_report=MappingReport(2, 2, 4),
        manifest_digest="d" * 64,
    )


def _adata(
    subjects: tuple[str, ...] = ("p1", "p2", "p3", "p4"),
) -> AnnData:
    genes = ("L1", "R1", "L2", "R2", "T1", "T2")
    rows: list[list[int]] = []
    metadata: list[dict[str, str]] = []
    obs_names: list[str] = []
    for subject_index, subject in enumerate(subjects):
        high = 70 + 5 * subject_index
        low = 2 + subject_index
        for condition in ("control", "stim"):
            sample = f"sample-{subject}-{condition}"
            for cell_type, profile in (
                ("Sender", [high, 0, low, 0, 1, 1]),
                ("Receiver", [0, high, 0, low, 1, 1]),
            ):
                for cell_index in range(3):
                    rows.append(profile)
                    metadata.append(
                        {
                            "sample_id": sample,
                            "subject_id": subject,
                            "cell_type": cell_type,
                            "condition": condition,
                        }
                    )
                    obs_names.append(
                        f"{subject}-{condition}-{cell_type}-{cell_index}"
                    )
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.int64))
    adata = AnnData(
        X=sparse.csr_matrix(counts.shape, dtype=np.float64),
        obs=pd.DataFrame(metadata, index=obs_names),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = counts
    return adata


def _config() -> CrychicConfig:
    return CrychicConfig(
        context_keys=("condition",),
        counts_layer="counts",
        design="~ condition",
        random_seed=19,
    )


def _spec() -> CrossFitSpec:
    return CrossFitSpec(
        contrasts=(
            balanced_contrast(
                ("stim",),
                ("control",),
                name="stim_vs_control",
            ),
        ),
        training_spec=FoldTrainingSpec(
            min_cells=1,
            max_interactions=1,
            sender_parameters=ContrastCommonSenderParameters(min_subjects=2),
        ),
        allowed_n_splits=(2,),
    )


def _persistable_spec() -> CrossFitSpec:
    resource = build_receiver_autonomous_program_resource(
        np.asarray([[1.0], [1.0]]),
        feature_ids=("T1", "T2"),
        program_ids=("generic_program",),
        resource_id="signature-export-autonomous-programs",
        version="1",
        manifest_digest="9" * 64,
        species=Species.HUMAN,
        gene_namespace=GeneNamespace.HGNC_SYMBOL,
    )
    return replace(
        _spec(),
        autonomous_program_resource=resource,
        penalty_tuning_spec=PenaltyTuningSpec(
            lambda1_fractions=(1.0,),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
        ),
    )


def _signal_adata(*, ligand_signal: bool = True) -> AnnData:
    adata = _adata(tuple(f"p{index}" for index in range(1, 9)))
    counts = adata.layers["counts"].toarray()
    for row_index, row in enumerate(adata.obs.itertuples()):
        if ligand_signal and row.cell_type == "Sender":
            counts[row_index, 0] = 120 if row.condition == "stim" else 10
        if row.cell_type == "Receiver":
            counts[row_index, 4] = 60 if row.condition == "stim" else 2
            counts[row_index, 5] = 35 if row.condition == "stim" else 2
    adata.layers["counts"] = sparse.csr_matrix(counts)
    return adata


def _unpenalized_spec() -> CrossFitSpec:
    return replace(
        _persistable_spec(),
        penalty_tuning_spec=PenaltyTuningSpec(
            lambda1_fractions=(0.0,),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
        ),
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def signal_artifacts() -> CrossFitArtifacts:
    return run_subject_crossfit(
        _signal_adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_unpenalized_spec(),
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def signal_export(
    signal_artifacts: CrossFitArtifacts,
) -> CrossFitLayeredSignatureExport:
    return export_crossfit_layered_signatures(
        signal_artifacts,
        entropy_threshold=1.0,
    )


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def receiver_only_export() -> CrossFitLayeredSignatureExport:
    artifacts = run_subject_crossfit(
        _signal_adata(ligand_signal=False),
        _config(),
        _bundle(),
        _prior(),
        spec=_unpenalized_spec(),
    )
    return export_crossfit_layered_signatures(
        artifacts,
        entropy_threshold=1.0,
    )


def test_real_crossfit_reconstructs_nonzero_gene_coordinate_and_conserves_layers(
    signal_export: CrossFitLayeredSignatureExport,
) -> None:
    table = signal_export.table
    assert table is not None
    receiver = table.receiver_context.set_index("signature_id", drop=False)
    lr = table.lr_attributed.set_index("signature_id", drop=False)

    assert (receiver["predicted"].abs() > 1e-12).any()
    np.testing.assert_allclose(
        receiver["observed"].to_numpy(dtype=float),
        receiver["predicted"].to_numpy(dtype=float)
        + receiver["residual"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    for receiver_id, rows in table.lr_attributed.groupby(
        "receiver_signature_id", sort=True
    ):
        assert np.isclose(
            rows["attributed_contribution"].sum(),
            float(receiver.loc[str(receiver_id), "predicted"]),
            rtol=1e-9,
            atol=1e-10,
        )
    for lr_id, rows in table.sender_lr_receiver.groupby(
        "lr_signature_id", sort=True
    ):
        assert np.isclose(
            rows["attributed_contribution"].sum(),
            float(lr.loc[str(lr_id), "attributed_contribution"]),
            rtol=1e-9,
            atol=1e-10,
        )
    for encoded in table.lr_attributed["provenance_json"].unique():
        provenance = json.loads(str(encoded))
        assert provenance["coordinate_version"] == GENE_MODEL_COORDINATE_VERSION
        assert provenance["no_model_refit"] is True
        assert provenance["scalar_score_fields_used_as_gene_contribution"] == []
        assert provenance["weight_sources"] == {
            "lr": "within_family_lr_weight",
            "sender": "assignment_weight",
        }
    assert not signal_export.inference_eligible


def test_availability_counts_match_rows_and_folds_remain_distinct(
    signal_artifacts: CrossFitArtifacts,
    signal_export: CrossFitLayeredSignatureExport,
) -> None:
    table = signal_export.table
    assert table is not None
    audit = signal_export.availability_table()
    available = audit.loc[
        audit["status"].eq(SignatureExportAvailabilityStatus.AVAILABLE.value)
    ]
    expected = {
        SignatureExportLayer.RECEIVER_CONTEXT.value: len(table.receiver_context),
        SignatureExportLayer.LR_ATTRIBUTED.value: len(table.lr_attributed),
        SignatureExportLayer.SENDER_LR_RECEIVER.value: len(
            table.sender_lr_receiver
        ),
    }
    observed = (
        available.groupby("layer", observed=True)["component_count"]
        .sum()
        .astype(int)
        .to_dict()
    )
    assert observed == expected

    receiver = table.receiver_context.loc[
        table.receiver_context["receiver"].eq("Receiver")
    ]
    assert receiver.groupby(
        ["receiver", "context_id", "feature_id"], observed=True
    ).size().eq(len(signal_artifacts.folds)).all()
    fold_sets = receiver["fold_ids_json"].map(json.loads)
    scoring_sets = receiver["scoring_functional_ids_json"].map(json.loads)
    assert fold_sets.map(len).eq(1).all()
    assert scoring_sets.map(len).eq(1).all()
    assert len({tuple(value) for value in fold_sets}) == len(signal_artifacts.folds)
    assert len({tuple(value) for value in scoring_sets}) == len(
        signal_artifacts.folds
    )


def test_missing_lr_weights_do_not_suppress_reconstructable_receiver_layer(
    receiver_only_export: CrossFitLayeredSignatureExport,
) -> None:
    table = receiver_only_export.table
    assert table is not None
    audit = receiver_only_export.availability_table()
    receiver_audit = audit.loc[audit["receiver"].eq("Receiver")]
    assert set(
        receiver_audit.loc[
            receiver_audit["layer"].eq(SignatureExportLayer.RECEIVER_CONTEXT.value),
            "status",
        ]
    ) == {SignatureExportAvailabilityStatus.AVAILABLE.value}
    downstream = receiver_audit.loc[
        receiver_audit["layer"].isin(
            [
                SignatureExportLayer.LR_ATTRIBUTED.value,
                SignatureExportLayer.SENDER_LR_RECEIVER.value,
            ]
        )
    ]
    assert set(downstream["status"]) == {
        SignatureExportAvailabilityStatus.NOT_ESTIMABLE.value
    }
    assert downstream["reason_code"].astype(str).str.startswith(
        "within_family_lr_weight_"
    ).all()
    assert (table.receiver_context["predicted"].abs() > 1e-12).any()
    assert table.lr_attributed.empty
    assert table.sender_lr_receiver.empty
    receiver_provenance = json.loads(table.receiver_context.iloc[0]["provenance_json"])
    assert receiver_provenance[0]["provenance"]["entropy_status"] == (
        "not_available_receiver_only_placeholder_not_released"
    )


def test_high_entropy_gate_keeps_receiver_and_fail_closes_children(
    signal_artifacts: CrossFitArtifacts,
    signal_export: CrossFitLayeredSignatureExport,
) -> None:
    gated = export_crossfit_layered_signatures(
        signal_artifacts,
        entropy_threshold=0.0,
    )
    assert gated.table is not None
    assert signal_export.table is not None
    pdt.assert_series_equal(
        gated.table.receiver_context.set_index("signature_id")["predicted"],
        signal_export.table.receiver_context.set_index("signature_id")["predicted"],
    )
    for frame in (gated.table.lr_attributed, gated.table.sender_lr_receiver):
        assert set(frame["status"]) == {
            LayeredSignatureStatus.FAMILY_HIGH_ENTROPY.value
        }
        assert frame["attributed_contribution"].isna().all()


def test_signature_export_is_invariant_to_input_row_order(
    signal_export: CrossFitLayeredSignatureExport,
) -> None:
    adata = _signal_adata()
    permutation = np.random.default_rng(20260715).permutation(adata.n_obs)
    shuffled = run_subject_crossfit(
        adata[permutation].copy(),
        _config(),
        _bundle(),
        _prior(),
        spec=_unpenalized_spec(),
    )
    reordered = export_crossfit_layered_signatures(
        shuffled,
        entropy_threshold=1.0,
    )
    assert reordered.crossfit_id == signal_export.crossfit_id
    assert reordered.export_id == signal_export.export_id
    assert reordered.table is not None
    assert signal_export.table is not None
    for left, right in (
        (
            signal_export.table.receiver_context,
            reordered.table.receiver_context,
        ),
        (signal_export.table.lr_attributed, reordered.table.lr_attributed),
        (
            signal_export.table.sender_lr_receiver,
            reordered.table.sender_lr_receiver,
        ),
    ):
        pdt.assert_frame_equal(left, right)


def test_missing_family_common_functional_returns_typed_unavailable() -> None:
    artifacts = run_subject_crossfit(
        _adata(),
        _config(),
        _bundle(),
        _prior(),
        spec=_spec(),
    )
    exported = export_crossfit_layered_signatures(artifacts)

    assert exported.table is None
    audit = exported.availability_table()
    assert set(audit["status"]) == {
        SignatureExportAvailabilityStatus.NOT_ESTIMABLE.value
    }
    assert set(audit["reason_code"]) == {
        "family_common_functional_not_produced_without_penalty_tuning"
    }
    assert not {"p_value", "q_value", "probability"}.intersection(audit.columns)
