# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.schemas import hash_module_state_dict


@pytest.fixture(autouse=True)
def use_fake_mode(fake_mode):
    yield


def _modules():
    yield nn.Linear(3, 5)
    yield nn.BatchNorm1d(4)
    yield nn.Transformer()


@pytest.fixture(params=_modules())
def module(request: pytest.FixtureRequest):
    return request.param


def _linear():
    return nn.Linear(3, 5)


@pytest.fixture
def linear():
    return _linear()


@pytest.fixture
def sequential():
    linear = _linear()
    return nn.Sequential(linear)


def test_hashing_nn_module(module: nn.Module):
    state_hash = hash_module_state_dict(module)

    for key, val in state_hash.items():
        assert isinstance(key, str)
        assert isinstance(val, int)


def test_same_weights_hash(sequential, linear):
    seq_hash = hash_module_state_dict(sequential)
    linear_hash = hash_module_state_dict(linear)

    assert sorted(seq_hash.values()) == sorted(linear_hash.values())
