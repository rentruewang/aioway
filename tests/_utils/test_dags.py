# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest

from aioway._utils import DagNode, topo_sort


@pytest.fixture
def dag():
    nodes = [
        DagNode(1, []),
        DagNode(2, [1]),
        DagNode(3, [1]),
        DagNode(4, [1, 3]),
    ]
    return topo_sort(nodes)


def test_dag_items(dag: list[int]):
    assert dag[0] == 1
    assert dag[3] == 4
    assert set(dag[1:3]) == {2, 3}
