# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import nn

from aioway.algos import SupervisedAlgo
from aioway.compilers import JustLinearEmitter
from aioway.dsets import TensorListHop, TensorStream
from aioway.hop import ListHop, LoaderOpt, NnHop, NnLayerHop, TensorHop
from aioway.nn import Linear, NnInit
from aioway.spaces import Attr, AttrSpace, Shape, ShapeSpace


@pytest.fixture
def input_space():
    return AttrSpace.from_attr(Attr.build(dtype="float", shape=[3, 4, 5]))


@pytest.fixture
def output_space():
    return ShapeSpace(Shape.parse(3, 4, 6))


@pytest.fixture
def input_dataset(input_space: AttrSpace):
    class FakeInputDset(TensorStream):
        def __call__(self, *_):
            return TensorListHop([input_space.to_attr().to_fake_tensor()])

        @typing.override
        def __iter__(self):
            # Not really using this, so we can afford to make a fake one.
            raise NotImplementedError

    return FakeInputDset()


@pytest.fixture
def input_hop(input_dataset: TensorStream) -> TensorHop:
    return input_dataset(LoaderOpt())


@pytest.fixture
def target_hop(input_hop: TensorHop) -> TensorHop:
    return input_hop


@pytest.fixture
def supervised(input_hop: TensorHop, target_hop: TensorHop):
    return SupervisedAlgo(input_hop, target_hop)


def test_just_linear(input_hop: TensorHop, output_space: ShapeSpace):
    builder = JustLinearEmitter(input_hop, output_space)
    built = builder()
    [tensor_node, linear_node, list_node] = built.dag()
    assert isinstance(linear_node, NnLayerHop)
    assert isinstance(linear_node.module, nn.Linear)
    assert isinstance(linear_node.nn_init, Linear)
    assert isinstance(tensor_node, TensorHop)
    assert isinstance(list_node, ListHop)
    assert linear_node.nn_init.in_features == 5
    assert linear_node.nn_init.out_features == 6

    assert len([node.is_source for node in built.hops])


def test_just_linear_supervised(supervised: SupervisedAlgo):
    original = supervised.just_linear()
    dag = supervised()
    assert len(dag.nodes()) > len(original.nodes())

    for node in dag.deps():
        assert isinstance(node, NnHop)
        assert isinstance(node.nn_init, NnInit)

        # Right now only `nn.MSELoss`.
        assert node.nn_init.NN == nn.MSELoss
