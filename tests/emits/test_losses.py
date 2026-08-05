# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.emits import LossTCls, PairLossModule, dispatch_mse_loss, emit_one
from aioway.spaces import AnySpace, space_for_tcls


@pytest.fixture
def emit_mse_loss():
    with dispatch_mse_loss.consider():
        yield


def test_emit_mse_loss(emit_mse_loss):
    loss = emit_one(AnySpace(), space_for_tcls(LossTCls))
    assert isinstance(loss, PairLossModule)
    assert isinstance(loss.loss_func, nn.MSELoss)
