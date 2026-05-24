# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._dags import Dag, TupleDagNode


@pytest.fixture
def dag():
    a = TupleDagNode(1, ())
    b = TupleDagNode(2, (a,))
    c = TupleDagNode(3, (a,))
    d = TupleDagNode(4, (a, c))
    return Dag.from_output([a, b, c, d])


def test_dag(dag: Dag[int]):
    assert dag[0].item() == 1
    assert dag[3].item() == 4
    assert {dag[1].item(), dag[2].item()} == {2, 3}
