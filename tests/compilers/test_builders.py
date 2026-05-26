# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.compilers import just_linear_builder
from aioway.hop import Linear, NnHopInit
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
