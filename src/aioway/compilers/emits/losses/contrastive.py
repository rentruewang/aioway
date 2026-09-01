# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for contrastive loss."

from collections import abc as cabc

import torch
from torch import nn, optim
from torch.nn import functional as F

__all__ = ["ContrastiveLoss"]


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        loss_fn: nn.Module,
        optim_type: cabc.Callable[..., optim.Optimizer],
    ):
        super().__init__()

        self.module = module
        self.loss_fn = loss_fn
        self.optimizer = optim_type(self.module.parameters())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim >= 1
        batch_size = len(tensor)

        out = self.module(tensor)

        # Flatten so we can apply the cross entropy easier.
        out = out.view(batch_size, -1)
        matrix = out @ out.T
        label = torch.arange(batch_size)

        loss = F.cross_entropy(matrix, label)
        loss_t = F.cross_entropy(matrix.T, label)

        return (loss + loss_t) / 2
