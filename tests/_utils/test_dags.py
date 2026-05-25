# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._utils import Dag


@pytest.fixture
def dag():
    return Dag.from_graph({1: [], 2: [1], 3: [1], 4: [1, 3]})


def test_dag(dag: Dag[int]):
    assert dag[0].data == 1
    assert dag[3].data == 4
    assert {dag[1].data, dag[2].data} == {2, 3}
