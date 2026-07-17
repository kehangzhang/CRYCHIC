from __future__ import annotations

import pytest
from tests.integration.test_subject_crossfit import (
    _adata,
    _bundle,
    _config,
    _prior,
    _spec,
)

from crychic.core import ContractError
from crychic.workflow import RepeatedCrossFitSpec, run_repeated_subject_crossfit


def test_parallel_repeats_preserve_scientific_identity_and_order() -> None:
    adata = _adata(tuple(f"p{index}" for index in range(1, 9)))
    spec = RepeatedCrossFitSpec(crossfit_spec=_spec(), n_repeats=2)
    serial = run_repeated_subject_crossfit(
        adata,
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
        n_jobs=1,
    )
    parallel = run_repeated_subject_crossfit(
        _adata(tuple(f"p{index}" for index in range(1, 9))),
        _config(),
        _bundle(),
        _prior(),
        spec=spec,
        n_jobs=10,
    )

    assert parallel.repeated_crossfit_id == serial.repeated_crossfit_id
    assert [item.crossfit_id for item in parallel.repeats] == [
        item.crossfit_id for item in serial.repeats
    ]
    assert parallel.repeat_registry_digest == serial.repeat_registry_digest
    assert parallel.family_fold_events_digest == serial.family_fold_events_digest
    assert parallel.subject_family_repeat_values_digest == (
        serial.subject_family_repeat_values_digest
    )
    assert serial.requested_n_jobs == serial.effective_n_jobs == 1
    assert parallel.requested_n_jobs == 10
    assert parallel.effective_n_jobs == 2
    assert serial.execution_metadata_id != parallel.execution_metadata_id
    assert serial.execution_backend == "serial_v1"
    assert parallel.execution_backend == "bounded_shared_snapshot_thread_pool_v1"
    assert parallel.to_manifest()["parallel_ordering_policy"] == (
        "executor_map_repeat_index_order_v1"
    )

    scientific_id = parallel.repeated_crossfit_id
    object.__setattr__(parallel, "requested_n_jobs", 2)
    with pytest.raises(ContractError) as error:
        parallel.to_manifest()
    assert error.value.details.code == (
        "repeated_crossfit_diagnostics_integrity_violation"
    )
    assert parallel.repeated_crossfit_id == scientific_id
