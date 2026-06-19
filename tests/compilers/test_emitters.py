# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import nn, optim

from aioway._comps import ListIter, TensorIter
from aioway.compilers import JustLinearEmitter
from aioway.dsets import TensorListIter, TensorStream
from aioway.hop import LoaderOpt
from aioway.spaces import Attr, AttrSpace, Shape, ShapeSpace
from aioway.torch.nn import Linear, MSELoss, NnLayerIter
from aioway.torch.optim import OptimizerIter


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
            return TensorListIter(
                [input_space.to_attr().to_fake_tensor().requires_grad_()]
            )

        @typing.override
        def __iter__(self):
            # Not really using this, so we can afford to make a fake one.
            raise NotImplementedError

    return FakeInputDset()


@pytest.fixture
def input_hop(input_dataset: TensorStream) -> TensorIter:
    return input_dataset(LoaderOpt())


@pytest.fixture
def target_hop(input_hop: TensorIter) -> TensorIter:
    return input_hop


@pytest.fixture
def optimizer(input_hop: TensorListIter, target_hop: TensorIter):
    opt = optim.AdamW(input_hop.sequence)
    loss = MSELoss().apply(input_hop, target_hop)
    assert isinstance(loss, TensorIter)
    return OptimizerIter(loss=loss, optimizer=opt)


def test_just_linear(input_hop: TensorIter, output_space: ShapeSpace):
    builder = JustLinearEmitter(input_hop, output_space)
    built = builder()
    [tensor_node, linear_node, list_node] = built.dag()
    assert isinstance(linear_node, NnLayerIter)
    assert isinstance(linear_node.module, nn.Linear)
    assert isinstance(linear_node.nn_init, Linear)
    assert isinstance(tensor_node, TensorIter)
    assert isinstance(list_node, ListIter)
    assert linear_node.nn_init.in_features == 5
    assert linear_node.nn_init.out_features == 6

    assert len([node.is_source for node in built.hops])


def test_optimize(optimizer: OptimizerIter):
    has_run = False

    for _ in optimizer:
        has_run = True

    assert has_run
