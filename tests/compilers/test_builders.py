# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.algos import SupervisedAlgo
from aioway.attrs import Attr
from aioway.compilers import just_linear_builder
from aioway.hop import HopDag, Linear, NnHop, NnInit, NnLayerHop


@pytest.fixture
def input_space():
    pytest.xfail()
    return SchemaSpace(Schema(Attr.build(dtype="float", shape=[3, 4, 5])))


@pytest.fixture
def output_space():
    pytest.xfail()
    return SchemaSpace(Schema(Attr.build(dtype="float", shape=[3, 4, 6])))


def test_just_linear(input_space: SchemaSpace, output_space: SchemaSpace):
    built = just_linear_builder([input_space], [output_space])
    [tensor_node, linear_node] = built
    assert isinstance(linear_node, NnLayerHop)
    assert isinstance(linear_node.module, nn.Linear)
    assert isinstance(linear_node.config, Linear)
    assert linear_node.config.in_features == 5
    assert linear_node.config.out_features == 6

    assert len(built.input_nodes) == 1
    assert len(built.output_nodes) == 1


@pytest.mark.xfail(reason="Supervised learning has bugs.")
def test_just_linear_supervised(input_space: SchemaSpace, output_space: SchemaSpace):
    built = just_linear_builder([input_space], [output_space])
    supervised: HopDag = SupervisedAlgo()(input_space, output_space)

    assert len(built) + len(built.output_nodes) == len(supervised)

    for node in supervised.output_nodes:
        assert isinstance(node, NnHop)
        assert isinstance(node.config, NnInit)

        # Right now only `nn.sMSELoss`.
        assert node.config.NN == nn.MSELoss
