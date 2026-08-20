# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.schemas import hash_module_state_dict
from torch import nn
import pytest


def _modules():
    yield nn.Linear(3, 5)
    yield nn.BatchNorm1d(4)
    yield nn.Transformer()


@pytest.fixture(params=_modules())
def module(request: pytest.FixtureRequest):
    return request.param


def test_hashing_nn_module(module: nn.Module):
    state_hash = hash_module_state_dict(module)

    for key, val in state_hash.items():
        assert isinstance(key, str)
        assert isinstance(val, int)
