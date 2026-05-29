# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.attrs import Attr
from aioway.compilers import JustLinearBuilder
from aioway.hop import Linear, NnLayerHop
from aioway.tags import AttrTag


@pytest.fixture
def input_space():
    return AttrTag.from_attr(Attr.build(dtype="float", shape=[3, 4, 5]))


@pytest.fixture
def output_space():
    return AttrTag.from_attr(Attr.build(dtype="float", shape=[3, 4, 6]))


def test_just_linear(input_space: AttrTag, output_space: AttrTag):
    builder = JustLinearBuilder(input_space, output_space)
    built = builder()
    [tensor_node, linear_node] = built
    assert isinstance(linear_node, NnLayerHop)
    assert isinstance(linear_node.module, nn.Linear)
    assert isinstance(linear_node.nn_init, Linear)
    assert linear_node.nn_init.in_features == 5
    assert linear_node.nn_init.out_features == 6

    assert len(built.input_nodes) == 1
    assert len(built.output_nodes) == 1
