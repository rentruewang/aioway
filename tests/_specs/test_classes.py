# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

from aioway.emits import LossTCls
from aioway.spaces import space_for_tcls


def test_loss_space():
    inst = LossTCls(torch.randn(3, 5), torch.randn(3, 5))
    assert inst in space_for_tcls(LossTCls)
