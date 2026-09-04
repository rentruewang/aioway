# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.deductions import deduction_for, deduction_registry, new_deduction_registry
from aioway.instrs import Bilinear, Linear


@pytest.fixture
def use_new_deductions():
    with new_deduction_registry():
        yield


def test_new_deduction(use_new_deductions):
    assert not deduction_registry()


def _wrong_function():
    def wrong_args(linear: Linear, output):
        pass

    def wrong_instr(linear: Bilinear, input):
        pass

    yield wrong_args
    yield wrong_instr


@pytest.fixture(params=_wrong_function())
def wrong_func(request):
    return request.param


def test_wrong_func(wrong_func, use_new_deductions):
    with pytest.raises(TypeError):
        deduction_for(nn.Linear).register(wrong_func)


def test_linear_deduct():
    from aioway.deductions.dense import linear_deduct

    unbounded = tspecs.Unbounded(torch.Size([3, 4, 5, 6]))
    output = linear_deduct(nn.Linear(6, 7), unbounded)
    assert output == tspecs.Unbounded(torch.Size([3, 4, 5, 7]))


def test_sequential_deduct():
    from aioway.deductions.containers import sequential_deduct

    unbounded = tspecs.Unbounded(torch.Size([3, 4, 5, 6]))
    sequential = nn.Sequential(nn.Linear(6, 7), nn.Linear(7, 8), nn.Linear(8, 9))
    output = sequential_deduct(sequential, unbounded)
    assert output == tspecs.Unbounded(torch.Size([3, 4, 5, 9]))


def test_mse_deduct():
    from aioway.losses.bases import symmetric_loss_deduct

    unbounded = tspecs.Unbounded(torch.Size([3, 4, 5, 6]))
    output = symmetric_loss_deduct(nn.MSELoss(), unbounded, unbounded)
    assert output == tspecs.Unbounded(torch.Size([]))
