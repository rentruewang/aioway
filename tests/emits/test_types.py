# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.nets import PairLossModule


def _loss_modules():
    yield nn.MSELoss()
    yield nn.L1Loss()
    yield nn.SmoothL1Loss()


def _non_loss_modules():
    yield nn.Linear(3, 5)
    yield nn.Conv1d(3, 4, 2)


@pytest.fixture(params=_loss_modules())
def loss_module(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=_non_loss_modules())
def non_loss_module(request: pytest.FixtureRequest):
    return request.param


def test_pair_loss_module(loss_module: nn.Module):
    module = PairLossModule(loss_module)
    assert isinstance(module, nn.Module)


def test_pair_loss_module_fail(non_loss_module: nn.Module):
    with pytest.raises(TypeError):
        module = PairLossModule(non_loss_module)
