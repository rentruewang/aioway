# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch.utils import data as dutils
from torchrl.data import tensor_specs as tspecs

from aioway.emits import MlpEmitter, emit, emit_one, linear_regression
from aioway.instrs import Instr, Linear, Sequential
from aioway.schemas import Shape
from aioway.tspecs import TSpec, unbounded_box_tspec


@pytest.fixture
def input_shape_tspec():
    return unbounded_box_tspec(Shape.parse(3, 4, 6))


@pytest.fixture
def output_tspec():
    return unbounded_box_tspec(Shape.parse(3, 4, 7))


@pytest.fixture
def input_dataset(input_attr_tspec: tspecs.Unbounded):
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


@pytest.fixture
def consider_mlp():
    with MlpEmitter([100, 100]).consider():
        yield


def test_just_linear(input_shape_tspec: TSpec, output_tspec: TSpec, consider_linear):
    instr = emit_one(input_shape_tspec, output_tspec)
    _check_linear(
        instr,
        in_features=input_shape_tspec.shape,
        out_features=output_tspec.shape,
    )


def test_mlp_emitter(
    input_shape_tspec: TSpec, output_tspec: TSpec, consider_mlp, fake_mode
):
    input = torch.randn(13, 3, 4, 6)

    for instr in emit(input_shape_tspec, output_tspec):
        output = instr.module()(input)

    assert output.shape == (13, 3, 4, 7)


def _check_linear(linear: Instr, in_features: torch.Size, out_features: torch.Size):
    assert isinstance(linear, Linear | Sequential)

    in_tensor = torch.randn(101, *in_features)

    assert linear.module()(in_tensor).shape == (101, *out_features)
