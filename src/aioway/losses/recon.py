# Copyright (c) AIoWay Authors - All Rights Reserved

"The module for reconstruction loss."

import typing

import torch
from torch import nn

__all__ = ["LossLike", "ReconstructionLoss"]


@typing.runtime_checkable
class LossLike(typing.Protocol):
    def __call__(self, input: torch.Tensor, target: torch.Tensor): ...


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        super().__init__()

        self.loss_fn = loss_fn

    def forward(self, input: torch.Tensor):
        return self.loss_fn(input, input)
