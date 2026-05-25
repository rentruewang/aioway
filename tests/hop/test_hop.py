# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.hop import HopInit, NnHopFwd, TensorHopInit, build_nn_hop


@pytest.fixture
def hop():
    return TensorHopInit(tensor=torch.randn(100, 30))


def test_hop(module_thunk, hop: TensorHopInit):
    result = build_nn_hop(module_thunk, hop)
    assert isinstance(result, HopInit)

    fwd = result.do()
    assert isinstance(fwd, NnHopFwd)

    assert all(isinstance(param, nn.Parameter) for param in fwd.parameters())
