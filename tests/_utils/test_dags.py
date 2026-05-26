# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._utils import Dag, DagNode, topo_sort


@pytest.fixture
def dag() -> Dag[int]:
    nodes = [
        DagNode(1, []),
        DagNode(2, [1]),
        DagNode(3, [1]),
        DagNode(4, [1, 3]),
    ]
    return topo_sort(nodes)


def test_dag_items(dag: Dag[int]):
    items = dag.items
    assert items[0] == 1
    assert items[3] == 4
    assert set(items[1:3]) == {2, 3}


def test_dag_inputs_outputs(dag: Dag[int]):
    ins = dag.num_inputs()
    outs = dag.num_outputs()

    assert ins == (0, 1, 1, 2)
    assert outs[0] == 3
    assert outs[3] == 0
    assert set(outs[1:3]) == {0, 1}
