# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch import nn
from torch.utils import data as dutils
from torchrl.data import tensor_specs as tspecs

from aioway.spaces import unbounded_box_spec
from aioway._torch import Shape
from aioway.nets import (
    MlpCompoundEmitter,
    MlpEmitter,
    emit,
    emit_one,
    linear_regression,
)


@pytest.fixture
def input_shape_space():
    return unbounded_box_spec(Shape.parse(3, 4, 6))


@pytest.fixture
def output_space():
    return unbounded_box_spec(Shape.parse(3, 4, 7))


@pytest.fixture
def input_dataset(input_attr_space: tspecs.Unbounded):
    class FakeInputDset(dutils.IterableDataset):
        @typing.override
        def __iter__(self):
            # Not really using this, so we can afford to make a fake one.
            raise NotImplementedError

    return FakeInputDset()


@pytest.fixture
def input_loader(input_dataset: dutils.Dataset):
    return dutils.DataLoader(input_dataset)


@pytest.fixture
def target_loader(input_loader: dutils.DataLoader):
    return input_loader


@pytest.fixture
def consider_linear():
    with linear_regression.consider():
        yield


def _mlp_emitters():
    yield MlpCompoundEmitter([100, 100])
    yield MlpEmitter([100, 100])


@pytest.fixture(params=_mlp_emitters())
def consider_mlp(request: pytest.FixtureRequest):
    with request.param.consider():
        yield


def test_just_linear(
    input_shape_space: tspecs.Unbounded, output_space: tspecs.Unbounded, consider_linear
):
    module = emit_one(input_shape_space, output_space)
    assert isinstance(module, nn.Linear | nn.Sequential)
    _check_linear(
        module,
        in_features=input_shape_space.shape,
        out_features=output_space.shape,
    )


def test_mlp_emitter(
    input_shape_space: tspecs.Unbounded,
    output_space: tspecs.Unbounded,
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
