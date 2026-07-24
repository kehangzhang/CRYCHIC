from __future__ import annotations

from benchmarks.comprehensive.run_dcst_under100k_campaign import planned_tasks
from benchmarks.comprehensive.suggest_v5_under100k_contract import load_contract


def test_campaign_plans_two_independent_paper_sweeps() -> None:
    tasks = planned_tasks(
        load_contract(),
        replicates=25,
        first_seed=100,
        requested_sweeps="all",
    )

    subject = [task for task in tasks if task.sweep == "subject_count"]
    receiver = [task for task in tasks if task.sweep == "receiver_cell_count"]
    assert len(subject) == 9 * 25
    assert len(receiver) == 10 * 25
    assert {task.axis_value for task in subject} == set(range(5, 50, 5))
    assert {task.axis_value for task in receiver} == set(range(50, 501, 50))
    assert all(task.receiver_cells_in_condition_2 == 500 for task in subject)
    assert all(task.subjects_per_condition == 20 for task in receiver)
    assert len({task.seed for task in tasks}) == len(tasks)


def test_campaign_can_select_one_paper_sweep() -> None:
    tasks = planned_tasks(
        load_contract(),
        replicates=25,
        first_seed=100,
        requested_sweeps="subject_count",
    )

    assert len(tasks) == 9 * 25
    assert {task.sweep for task in tasks} == {"subject_count"}
