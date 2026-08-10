# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch import nn
from torch.utils import data
from torchrl import data as rldata

from aioway._specs import unbounded_box_spec
from aioway._torch import Shape
from aioway.emits import (
    MlpCompoundEmitter,
    MlpEmitter,
    emit,
    emit_one,
    linear_shape,
)
from aioway.io import TensorStream


@pytest.fixture
def input_shape_space():
    return unbounded_box_spec(Shape.parse(3, 4, 6))


@pytest.fixture
def output_space():
    return unbounded_box_spec(Shape.parse(3, 4, 7))


@pytest.fixture
def input_dataset(input_attr_space: rldata.Unbounded):
    class FakeInputDset(TensorStream):
        @typing.override
        def __iter__(self):
            # Not really using this, so we can afford to make a fake one.
            raise NotImplementedError

    return FakeInputDset()


@pytest.fixture
def input_loader(input_dataset: TensorStream):
    return data.DataLoader(input_dataset)


@pytest.fixture
def target_loader(input_loader: data.DataLoader):
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
    input_shape_space: rldata.Unbounded, output_space: rldata.Unbounded, consider_linear
):
    module = emit_one(input_shape_space, output_space)
    assert isinstance(module, nn.Linear | nn.Sequential)
    _check_linear(
        module,
        in_features=input_shape_space.shape,
        out_features=output_space.shape,
    )


def test_mlp_emitter(
    input_shape_space: rldata.Unbounded,
    output_space: rldata.Unbounded,
    consider_mlp,
    fake_mode,
):
    input = torch.randn(13, 3, 4, 6)

    for module in emit(input_shape_space, output_space):
        output = module(input)

    assert output.shape == (13, 3, 4, 7)


def _check_linear(linear: nn.Module, in_features: torch.Size, out_features: torch.Size):
    assert isinstance(linear, nn.Linear | nn.Sequential)

    in_tensor = torch.randn(101, *in_features)

    assert linear(in_tensor).shape == (101, *out_features)
