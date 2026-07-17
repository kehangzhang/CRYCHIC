from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from tests.integration import test_subject_crossfit as crossfit_fixture

from crychic.attribution import GainCalibrationSpec, PenaltyTuningSpec
from crychic.workflow.active_null_rerun import ActiveNullRerunResultStatus
from crychic.workflow.active_probability_pipeline import (
    run_active_probability_pipeline,
)
from crychic.workflow.crossfit import CrossFitSpec


def test_real_point_crossfit_flows_to_typed_active_null_probability_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = crossfit_fixture._spec()
    autonomous = crossfit_fixture._trusted_target_resource(tmp_path, monkeypatch)
    spec = CrossFitSpec(
        contrasts=base.contrasts,
        training_spec=base.training_spec,
        allowed_n_splits=(2,),
        autonomous_program_resource=autonomous,
        penalty_tuning_spec=PenaltyTuningSpec(
            lambda1_fractions=(1.0, 0.1),
            lambda2_fractions=(0.0,),
            inner_allowed_n_splits=(2,),
        ),
        gain_calibration_spec=GainCalibrationSpec(
            min_subjects=4,
            min_supported_families=1,
            min_subjects_per_family=4,
            min_positive_observations=4,
            min_distinct_positive_gains=4,
        ),
    )
    adata = crossfit_fixture._adata(tuple(f"p{index}" for index in range(1, 9)))
    counts = np.asarray(adata.layers["counts"].toarray(), dtype=np.int64)
    sender_mask = adata.obs["cell_type"].astype(str).eq("Sender").to_numpy()
    counts[sender_mask, 1] = counts[sender_mask, 0]
    counts[sender_mask, 3] = counts[sender_mask, 2]
    subject_number = (
        adata.obs["subject_id"].astype(str).str.removeprefix("p").astype(int).to_numpy()
    )
    stim_mask = adata.obs["condition"].astype(str).eq("stim").to_numpy()
    counts[stim_mask, 4] = 13 + 2 * subject_number[stim_mask]
    counts[sender_mask & stim_mask, 0] *= 3
    adata.layers["counts"] = sparse.csr_matrix(counts)

    result = run_active_probability_pipeline(
        adata,
        crossfit_fixture._config(),
        crossfit_fixture._bundle(),
        crossfit_fixture._prior(),
        crossfit_spec=spec,
        contrast_id_or_name="stim_vs_control",
        n_plans=2,
        n_jobs=2,
    )

    assert result.point_artifacts.oof_certification_audit.is_oof_descriptive_certified
    assert result.universe.candidate_edge_ids
    assert (
        tuple(record.candidate_edge_id for record in result.point_records.records)
        == result.universe.candidate_edge_ids
    )
    assert result.null_rerun.status is ActiveNullRerunResultStatus.NOT_ESTIMABLE
    assert result.null_rerun.requested_n_jobs == 2
    assert result.null_rerun.effective_n_jobs == 2
    assert result.null_rerun.to_manifest()["execution_backend"] == (
        "bounded_shared_snapshot_thread_pool_v1"
    )
    assert result.null_rerun.distribution.complete_rectangle
    assert result.probability_collection.complete_candidate_coverage
    assert result.comm_probability_release_allowed is False
    assert all(
        record.active_null_empirical_p_value is None and record.comm_probability is None
        for record in result.probability_collection.records
    )
    assert result.lineage.point_crossfit_id == result.point_artifacts.crossfit_id
    assert result.lineage.candidate_universe_id == result.universe.universe_id
    assert result.lineage.active_null_result_id == result.null_rerun.result_id
    assert result.lineage.probability_collection_id == (
        result.probability_collection.collection_id
    )
    assert result.to_manifest()["n_null_plans"] == 2
