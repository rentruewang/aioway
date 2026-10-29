# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch import nn
from torch.utils import data as du
from torchrl.data import tensor_specs as tspecs

from aioway.cc.emits import MlpEmitter, emit, emit_one, linear_regression
from aioway.io import TSpec, unbounded_box_tspec
from aioway.t import Shape


@pytest.fixture
def input_shape_tspec():
    return unbounded_box_tspec(Shape.parse(3, 4, 6))


@pytest.fixture
def output_tspec():
    return unbounded_box_tspec(Shape.parse(3, 4, 7))


@pytest.fixture
def input_dataset(input_attr_tspec: tspecs.Unbounded):
    class FakeInputDset(du.IterableDataset):
        @typing.override
        def __iter__(self):
            # Not really using this, so we can afford to make a fake one.
            raise NotImplementedError

    return FakeInputDset()


@pytest.fixture
def input_loader(input_dataset: du.Dataset):
    return du.DataLoader(input_dataset)


@pytest.fixture
def target_loader(input_loader: du.DataLoader):
    return input_loader


@pytest.fixture
def consider_linear():
    with linear_regression.consider():
        yield


@pytest.fixture
def consider_mlp():
    with MlpEmitter([100, 100]).consider():
        yield


def test_just_linear(input_shape_tspec: TSpec, output_tspec: TSpec, consider_linear):
    module = emit_one(input_shape_tspec, output_tspec).module

    _check_linear(
        module,
        in_features=input_shape_tspec.shape,
        out_features=output_tspec.shape,
    )


def test_mlp_emitter(
    input_shape_tspec: TSpec, output_tspec: TSpec, consider_mlp, fake_mode_on
):
    input = torch.randn(13, 3, 4, 6)

    for proc in emit(input_shape_tspec, output_tspec):
        input = proc.module(input)

    assert input.shape == (13, 3, 4, 7)


def _check_linear(linear: nn.Module, in_features: torch.Size, out_features: torch.Size):
    in_tensor = torch.randn(101, *in_features)

    assert linear(in_tensor).shape == (101, *out_features)
