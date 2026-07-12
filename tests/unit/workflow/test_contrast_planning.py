from crychic.design import ContextGraph
from crychic.workflow.baseline import _workflow_contrasts


def test_complete_graph_omits_equivalent_local_aliases() -> None:
    contrasts = _workflow_contrasts(ContextGraph.complete(("a", "b", "c")))

    assert [contrast.name for contrast in contrasts] == [
        "global:'a'",
        "global:'b'",
        "global:'c'",
    ]


def test_chain_retains_local_contrasts_with_distinct_weights() -> None:
    contrasts = _workflow_contrasts(ContextGraph.chain(("a", "b", "c")))

    assert [contrast.name for contrast in contrasts] == [
        "global:'a'",
        "global:'b'",
        "global:'c'",
        "local:'a'",
        "local:'c'",
    ]
