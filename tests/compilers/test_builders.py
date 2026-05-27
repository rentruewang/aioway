# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.algos import SupervisedAlgo
from aioway.compilers import just_linear_builder
from aioway.hop import HopDag, HopInit, Linear, NnHopInit, NnInit
from aioway.schemas import Attr, Schema
from aioway.spaces import SchemaSpace


@pytest.fixture
def input_space():
    return SchemaSpace(Schema(Attr.build(dtype="float", shape=[3, 4, 5])))


@pytest.fixture
def output_space():
    return SchemaSpace(Schema(Attr.build(dtype="float", shape=[3, 4, 6])))


def test_just_linear(input_space: SchemaSpace, output_space: SchemaSpace):
    built = just_linear_builder([input_space], [output_space])
    [tensor_node, linear_node] = built
    assert isinstance(linear_node, NnHopInit)
    assert isinstance(linear_node.nn_init, Linear)
    assert linear_node.nn_init.in_features == 5
    assert linear_node.nn_init.out_features == 6

    assert len(built.input_nodes) == 1
    assert len(built.output_nodes) == 1


def test_just_linear_supervised(input_space: SchemaSpace, output_space: SchemaSpace):
    built = just_linear_builder([input_space], [output_space])
    supervised: HopDag[HopInit] = SupervisedAlgo()(input_space, output_space)

    assert len(built) + len(built.output_nodes) == len(supervised)

    for node in supervised.output_nodes:
        assert isinstance(node, NnHopInit)
        assert isinstance(node.nn_init, NnInit)

        # Right now only `nn.sMSELoss`.
        assert node.nn_init.NN == nn.MSELoss
