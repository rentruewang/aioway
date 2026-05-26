# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._utils import topo_sort, DagNode


@pytest.fixture
def dag() -> list[int]:
    nodes = [
        DagNode(1, []),
        DagNode(2, [1]),
        DagNode(3, [1]),
        DagNode(4, [1, 3]),
    ]
    return topo_sort(nodes)


def test_dag(dag: list[int]):
    assert dag[0] == 1
    assert dag[3] == 4
    assert set(dag[1:3]) == {2, 3}
