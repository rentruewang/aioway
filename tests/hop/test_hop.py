# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.hop import TensorHop, build_nn_hop
from aioway.hop.hop import Hop


@pytest.fixture
def hop():
    return TensorHop(tensor=torch.randn(100, 30))


def test_hop_from_nn_init(module_thunk, hop: TensorHop):
    result = build_nn_hop(module_thunk, hop)
    assert isinstance(result, Hop)
