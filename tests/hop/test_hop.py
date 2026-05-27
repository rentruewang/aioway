# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.hop import HopInit, NnHopFwd, TensorHopInit, build_nn_hop
from aioway.hop.hop import cache_hop_init_output, hop_cache_on
from aioway.modes.modules import NnInitFn


@pytest.fixture
def tensor_init():
    return TensorHopInit(tensor=torch.randn(100, 30))


@pytest.fixture
def hop_cache_on():
    with cache_hop_init_output(), hop_cache_on():
        yield


@pytest.fixture
def hop_cache_off():
    return


@pytest.fixture(params=[hop_cache_on.name, hop_cache_off.name])
def maybe_cache_hop(request: pytest.FixtureRequest):
    return request.getfixturevalue(request.param)


def test_hop(module_thunk, tensor_init: TensorHopInit, maybe_cache_hop):
    result = build_nn_hop(module_thunk, tensor_init)
    assert isinstance(result, HopInit)

    fwd = result()
    assert isinstance(fwd, NnHopFwd)

    assert all(isinstance(param, nn.Parameter) for param in fwd.parameters())


def test_hop_linear(tensor_init: TensorHopInit, maybe_cache_hop):
    linear = build_nn_hop(NnInitFn(nn.Linear, args=(30, 31), kwargs={}), tensor_init)
    assert linear

    fwd_node = linear()
    result = fwd_node()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 31)
