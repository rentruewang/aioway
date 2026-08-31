# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.instrs import (
    Bilinear,
    Linear,
    deductor_for,
    deductor_registry,
    new_deductor_registry,
)


@pytest.fixture
def use_new_deductors():
    with new_deductor_registry():
        yield


def test_new_deductor(use_new_deductors):
    assert not deductor_registry()


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


def test_wrong_func(wrong_func, use_new_deductors):
    with pytest.raises(TypeError):
        deductor_for(nn.Linear).register(wrong_func)


def test_linear_deduct():
    from aioway.instrs.layers.dense import linear

    unbounded = tspecs.Unbounded(torch.Size([3, 4, 5, 6]))
    output = linear(Linear(6, 7), unbounded)
    assert output == tspecs.Unbounded(torch.Size([3, 4, 5, 7]))
