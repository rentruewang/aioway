# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.cc import topo_sort_id


@pytest.fixture
def dag():
    nodes = [
        (1, []),
        (2, [1]),
        (3, [1]),
        (4, [1, 3]),
    ]
    return topo_sort_id(nodes)


def test_dag_items(dag: list[int]):
    assert dag[0] == 1
    assert dag[3] == 4
    assert set(dag[1:3]) == {2, 3}
