# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest
import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.cc import deduction_for
from aioway.torch import LossTSpec


def _wrong_function():
    def wrong_args(linear: nn.Linear, output):
        pass

    def wrong_module(linear: nn.Bilinear, input):
        pass

    yield wrong_args
    yield wrong_module


@pytest.fixture(params=_wrong_function())
def wrong_func(request):
    return request.param


def test_wrong_func(wrong_func):
    with pytest.raises(TypeError):
        deduction_for(nn.Linear, deductions={}).register(wrong_func)


def test_linear_deduct():
    from aioway.cc.deductions.dense import linear_deduct

    unbounded = tspecs.Unbounded(torch.Size([3, 4, 5, 6]))
    output = linear_deduct(nn.Linear(6, 7), unbounded)
    assert output == tspecs.Unbounded(torch.Size([3, 4, 5, 7]))


def test_sequential_deduct():
    from aioway.cc.deductions.containers import sequential_deduct

    unbounded = tspecs.Unbounded(torch.Size([3, 4, 5, 6]))
    sequential = nn.Sequential(nn.Linear(6, 7), nn.Linear(7, 8), nn.Linear(8, 9))
    output = sequential_deduct(sequential, unbounded)
    assert output == tspecs.Unbounded(torch.Size([3, 4, 5, 9]))


def test_mse_deduct():
    from aioway.cc.deductions.losses import symmetric_loss_deduct

    unbounded = tspecs.Unbounded(torch.Size([3, 4, 5, 6]))
    output = symmetric_loss_deduct(nn.MSELoss(), unbounded, unbounded)
    assert is_loss_output(output)


def is_loss_output(output) -> typing.TypeIs[LossTSpec]:
    return isinstance(output, LossTSpec)
