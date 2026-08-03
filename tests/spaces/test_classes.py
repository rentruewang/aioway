# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.emits import LossSpace, LossTCls


@pytest.fixture
def loss_space():
    return LossSpace()


def test_loss_space(loss_space: LossSpace):
    inst = LossTCls(torch.randn(3, 5), torch.randn(3, 5))
    assert inst in loss_space
