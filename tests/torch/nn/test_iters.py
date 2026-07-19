# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import torch
from torch import nn

from aioway._iters import TensorIter, sample_mode
from aioway._ufuncs import UFuncThunk
from aioway._utils import AnyDict, is_fake_tensor, torch_fake_mode
from aioway.io import TensorListIter
from aioway.relalg import StackIter
from aioway.torch.nn import nn_ufunc


@pytest.fixture
def tensor_init():
    return TensorListIter([torch.randn(100, 30)])


@pytest.fixture(params=[False, True])
def maybe_sample_mode(request: pytest.FixtureRequest):
    with sample_mode(request.param):
        yield


def test_iter_mse(tensor_init: TensorIter, maybe_sample_mode):
    thunk = nn_ufunc(nn.MSELoss).thunk(tensor_init, tensor_init)
    out = next(iter(thunk))
    assert isinstance(out, torch.Tensor)


def test_iter_linear(tensor_init: TensorIter, maybe_sample_mode):
    thunk = nn_ufunc(nn.Linear, 30, 31).thunk(tensor_init)

    result = next(iter(thunk))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 31)


def test_iter_replace_with_function(tensor_init: TensorListIter):
    def replace_init(thunk):
        if not isinstance(thunk, TensorListIter):
            return NotImplemented

        return TensorListIter([torch.randn(101, 31)])

    memo = AnyDict()
    stacked = StackIter([tensor_init, tensor_init])
    replaced = stacked.replace(function=replace_init, memo=memo)
    inputs = list(replaced.deps())
    assert len(inputs) == 2
    assert isinstance(inputs[0], TensorListIter)
    assert isinstance(inputs[1], TensorListIter)

    assert inputs[0] is inputs[1]
    assert inputs[0].sequence[0].shape == (101, 31)


def test_iter_no_replace_with_function(tensor_init: TensorIter):
    def replace(hop):
        return NotImplemented

    result = next(iter(tensor_init.replace(replace)))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 30)


def test_iter_linear_rebuild(tensor_init: TensorIter):
    with torch_fake_mode():
        linear: UFuncThunk = nn_ufunc(nn.Linear, 30, 31).thunk(tensor_init)

        result = next(iter(linear))
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 31)
        assert is_fake_tensor(result)

    linear = linear.rebuild()
    result = next(iter(linear))
    assert not is_fake_tensor(result)
