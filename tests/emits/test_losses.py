# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn
from torchrl import data as rldata

from aioway._specs import unbounded_box_spec
from aioway.nets import (
    LossTCls,
    PairLossModule,
    dispatch_mse_loss,
    emit_one,
    sample_from_spec,
    set_batch_size,
)


@pytest.fixture
def emit_mse_loss():
    with dispatch_mse_loss.consider():
        yield


def test_emit_mse_loss(emit_mse_loss):
    input_space = rldata.Composite(
        data_cls=LossTCls, input=unbounded_box_spec(7), target=unbounded_box_spec(7)
    )
    loss_func = emit_one(input_space, unbounded_box_spec(shape=()))

    assert isinstance(loss_func, PairLossModule)
    assert isinstance(loss_func.loss_func, nn.MSELoss)

    with set_batch_size(10):
        mse_computed = sample_from_spec(input_space)
    assert isinstance(mse_computed, LossTCls)

    loss = loss_func(mse_computed)
    assert loss.shape == ()
