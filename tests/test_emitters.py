# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
from torch import optim

from aioway._iters import TensorIter
from aioway._ufuncs import BuiltUFunc, UFunc
from aioway.emits import MlpEmitter, emit, linear_shape
from aioway.io import TensorListIter, TensorStream
from aioway.relalg import LoaderOpt
from aioway.spaces import Attr, AttrSpace, Shape, ShapeSpace
from aioway.torch.nn import Linear, MSELoss, NnUFunc
from aioway.torch.optim import OptimizerIter


@pytest.fixture
def input_shape_space():
    return ShapeSpace(Shape.parse(3, 4, 6))


@pytest.fixture
def input_attr_space(input_shape_space: ShapeSpace):
    return AttrSpace.from_attr(Attr.build(dtype="float", shape=input_shape_space.shape))


@pytest.fixture
def output_space():
    return ShapeSpace(Shape.parse(3, 4, 6))


@pytest.fixture
def input_dataset(input_attr_space: AttrSpace):
    class FakeInputDset(TensorStream):
        def __call__(self, *_):
            return TensorListIter(
                [input_attr_space.to_attr().to_fake_tensor().requires_grad_()]
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


@pytest.fixture
def consider_linear():
    with linear_shape.consider():
        yield


@pytest.fixture
def consider_mlp():
    with MlpEmitter([100, 100]).consider():
        yield


def test_just_linear(
    input_shape_space: ShapeSpace, output_space: ShapeSpace, consider_linear
):
    linear_found = False

    for ufunc in emit(input_shape_space, output_space):
        assert isinstance(ufunc, UFunc)

        if isinstance(ufunc, NnUFunc):
            linear_found = True
            _check_linear(
                ufunc,
                in_features=input_shape_space.shape[-1],
                out_features=output_space.shape[-1],
            )

    assert linear_found


def test_mlp_emitter(
    input_shape_space: ShapeSpace, output_space: ShapeSpace, consider_mlp
):
    mlp_found = False
    for ufunc in emit(input_shape_space, output_space):
        if isinstance(ufunc, BuiltUFunc):
            mlp_found = True

    assert mlp_found


def _check_linear(linear: NnUFunc, in_features: int, out_features: int):
    assert isinstance(linear, NnUFunc)
    assert isinstance(linear.nn_init, Linear)
    assert linear.nn_init.in_features == in_features
    assert linear.nn_init.out_features == out_features


def test_optimize(optimizer: OptimizerIter):
    has_run = False

    for _ in optimizer:
        has_run = True

    assert has_run
