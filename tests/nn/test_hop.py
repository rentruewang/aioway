# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl

import pytest
import torch
from torch import nn

from aioway._utils import AnyDict, is_fake_tensor, torch_fake_mode
from aioway.dsets import TensorListHop
from aioway.hop import Hop, StackHop, TensorHop, hop_cache_on
from aioway.modes import NnInitFn
from aioway.torch.nn import NnLayerHop, NnLossHop, build_nn_hop


@pytest.fixture
def tensor_init():
    return TensorListHop([torch.randn(100, 30)])


@pytest.fixture
def cache_on():
    with hop_cache_on():
        yield


@pytest.fixture
def cache_off():
    return


@pytest.fixture(params=[cache_on.name, cache_off.name])
def maybe_cache_hop(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("cache", [False, True])
def test_hop_cache(cache: bool):
    number: int = 0

    class HopCacheLog(Hop):
        def iterate(self):
            nonlocal number
            while True:
                number += 1
                yield number

    logger = iter(HopCacheLog())
    cacher = hop_cache_on if cache else ctxl.nullcontext

    with cacher():
        assert number == 0, {"number": number, "cache": cache}
        next(logger)
        assert number == 1, {"number": number, "cache": cache}
        next(logger)
        next(logger)
        assert number == (1 if cache else 3), {"number": number, "cache": cache}


def test_layer_hop(layer_thunk, tensor_init: TensorHop, maybe_cache_hop):
    hop = build_nn_hop(layer_thunk, tensor_init)
    assert isinstance(hop, NnLayerHop)
    assert all(isinstance(param, nn.Parameter) for param in hop.parameters())


def test_loss_hop(loss_thunk, tensor_init: TensorHop, maybe_cache_hop):
    result = build_nn_hop(loss_thunk, tensor_init, tensor_init)
    assert isinstance(result, NnLossHop)


def test_hop_mse(tensor_init: TensorHop, maybe_cache_hop):
    result = build_nn_hop(NnInitFn(nn.MSELoss), tensor_init, tensor_init)

    assert result
    out = next(iter(result))
    assert isinstance(out, torch.Tensor)


def test_hop_linear(tensor_init: TensorHop, maybe_cache_hop):
    linear = build_nn_hop(NnInitFn(nn.Linear, 30, 31), tensor_init)
    assert linear

    result = next(iter(linear))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 31)


def test_hop_replace_with_function(tensor_init: TensorListHop):
    def replace_init(hop):
        if not isinstance(hop, TensorListHop):
            return NotImplemented

        return TensorListHop([torch.randn(101, 31)])

    memo = AnyDict()
    stacked = StackHop([tensor_init, tensor_init])
    replaced = stacked.replace(function=replace_init, memo=memo)
    inputs = list(replaced.deps())
    assert len(inputs) == 2
    assert isinstance(inputs[0], TensorListHop)
    assert isinstance(inputs[1], TensorListHop)

    assert inputs[0] is inputs[1]
    assert inputs[0].sequence[0].shape == (101, 31)


def test_hop_no_replace_with_function(tensor_init: TensorHop):
    def replace(hop):
        return NotImplemented

    result = next(iter(tensor_init.replace(replace)))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 30)


def test_hop_linear_rebuild(tensor_init: TensorHop):
    with torch_fake_mode():
        linear = build_nn_hop(NnInitFn(nn.Linear, 30, 31), tensor_init)
        assert linear

        result = next(iter(linear))
        assert isinstance(result, torch.Tensor)
        assert result.shape == (100, 31)
        assert is_fake_tensor(result)

    linear = linear.rebuild()
    result = next(iter(linear))
    assert not is_fake_tensor(result)
