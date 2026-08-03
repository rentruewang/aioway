# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch import nn

from aioway._torch import Attr, Shape
from aioway.emits import (
    MlpCompoundEmitter,
    MlpEmitter,
    emit,
    emit_one,
    linear_shape,
)
from aioway.io import TensorListExec, TensorStream
from aioway.relalg import LoaderOpt, TensorExec
from aioway.spaces import AttrSpace, ShapeSpace


@pytest.fixture
def input_shape_space():
    return ShapeSpace(Shape.parse(3, 4, 6))


@pytest.fixture
def input_attr_space(input_shape_space: ShapeSpace):
    return AttrSpace.from_attr(Attr.build(dtype="float", shape=input_shape_space.shape))


@pytest.fixture
def output_space():
    return ShapeSpace(Shape.parse(3, 4, 7))


@pytest.fixture
def input_dataset(input_attr_space: AttrSpace):
    class FakeInputDset(TensorStream):
        def __call__(self, *_):
            return TensorListExec(
                [input_attr_space.to_attr().to_fake_tensor().requires_grad_()]
            )

        @typing.override
        def __iter__(self):
            # Not really using this, so we can afford to make a fake one.
            raise NotImplementedError

    return FakeInputDset()


@pytest.fixture
def input_loader(input_dataset: TensorStream) -> TensorExec:
    return input_dataset(LoaderOpt())


@pytest.fixture
def target_loader(input_loader: TensorExec) -> TensorExec:
    return input_loader


@pytest.fixture
def consider_linear():
    with linear_shape.consider():
        yield


def _mlp_emitters():
    yield MlpCompoundEmitter([100, 100])
    yield MlpEmitter([100, 100])


@pytest.fixture(params=_mlp_emitters())
def consider_mlp(request: pytest.FixtureRequest):
    with request.param.consider():
        yield


def test_just_linear(
    input_shape_space: ShapeSpace, output_space: ShapeSpace, consider_linear
):
    module = emit_one(input_shape_space, output_space)
    assert isinstance(module, nn.Linear)
    _check_linear(
        module,
        in_features=input_shape_space.shape[-1],
        out_features=output_space.shape[-1],
    )


def test_mlp_emitter(
    input_shape_space: ShapeSpace, output_space: ShapeSpace, consider_mlp, fake_mode
):
    input = torch.randn(13, 3, 4, 6)

    for module in emit(input_shape_space, output_space):
        output = module(input)

    assert output.shape == (13, 3, 4, 7)


def _check_linear(linear: nn.Linear, in_features: int, out_features: int):
    assert isinstance(linear, nn.Linear)

    # The modern way to check subsets.
    assert linear.in_features == in_features
    assert linear.out_features == out_features
