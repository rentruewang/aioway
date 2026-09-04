# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn, optim

from aioway.emits import (
    ContrastiveLoss,
)


def test_contrastive_loss():
    loss = ContrastiveLoss(nn.Linear(3, 5), nn.MSELoss(), optim.Adam)

    tensor = torch.randn(7, 3).requires_grad_()
    out = loss(tensor)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0
