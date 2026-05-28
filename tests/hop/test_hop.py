# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.hop import NnLayerHop, NnLossHop, TensorHop, build_nn_hop, hop_cache_on
from aioway.modes import NnInitFn


@pytest.fixture
def tensor_init():
    return TensorHop(data=torch.randn(100, 30))


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


def test_layer_hop(layer_thunk, tensor_init: TensorHop, maybe_cache_hop):
    hop = build_nn_hop(layer_thunk, tensor_init)
    assert isinstance(hop, NnLayerHop)
    assert all(isinstance(param, nn.Parameter) for param in hop.parameters())


def test_loss_hop(loss_thunk, tensor_init: TensorHop, maybe_cache_hop):
    result = build_nn_hop(loss_thunk, tensor_init, tensor_init)
    assert isinstance(result, NnLossHop)


def test_hop_mse(tensor_init: TensorHop, maybe_cache_hop):
    result = build_nn_hop(
        NnInitFn(nn.MSELoss, args=(), kwargs={}), tensor_init, tensor_init
    )

    assert result
    out = result()
    assert isinstance(out, torch.Tensor)


def test_hop_linear(tensor_init: TensorHop, maybe_cache_hop):
    linear = build_nn_hop(NnInitFn(nn.Linear, args=(30, 31), kwargs={}), tensor_init)
    assert linear

    result = linear()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 31)
