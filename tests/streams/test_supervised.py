# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway._streams import TensorStream
from aioway.algos import SupervisedAlgo
from aioway.hop import NnHop, NnInit
from aioway.relalg import FrameStream


@pytest.fixture
def input_stream(table_stream: FrameStream):
    return table_stream.column("f1d")


@pytest.fixture
def target_stream(table_stream: FrameStream):
    return table_stream.column("f2d")


@pytest.fixture
def supervised(input_stream: TensorStream, target_stream: TensorStream):
    return SupervisedAlgo(input_stream, target_stream)


def test_just_linear_supervised(supervised: SupervisedAlgo):
    original = supervised.just_linear()
    dag = supervised()
    assert len(dag) == len(original) + 2

    for node in dag.output_nodes:
        assert isinstance(node, NnHop)
        assert isinstance(node.nn_init, NnInit)

        # Right now only `nn.MSELoss`.
        assert node.nn_init.NN == nn.MSELoss
