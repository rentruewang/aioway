# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.algos import SupervisedAlgo
from aioway.hop import TensorHop
from aioway.io import LoaderHop
from aioway.nn import NnHop, NnInit


@pytest.fixture
def input_stream(table_stream: LoaderHop):
    return table_stream.column("f1d")


@pytest.fixture
def target_stream(table_stream: LoaderHop):
    return table_stream.column("f2d")


@pytest.fixture
def supervised(input_stream: TensorHop, target_stream: TensorHop):
    return SupervisedAlgo(input_stream, target_stream)


def test_just_linear_supervised(supervised: SupervisedAlgo):
    original = supervised.just_linear()
    dag = supervised()
    assert (
        len(dag.nodes())
        == len(original.nodes()) + len(supervised.target_data.nodes()) + 1
    )

    for node in dag.deps():
        assert isinstance(node, NnHop)
        assert isinstance(node.nn_init, NnInit)

        # Right now only `nn.MSELoss`.
        assert node.nn_init.NN == nn.MSELoss
