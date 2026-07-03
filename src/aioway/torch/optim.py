# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import torch
from torch import optim

from aioway._api import public_api
from aioway._ufuncs import UFunc

__all__ = ["OptimizerUFunc"]


@public_api
@dcls.dataclass(frozen=True)
class OptimizerUFunc(UFunc):
    """
    The optimizer `UFunc`.
    """

    optimizer: optim.Optimizer
    "The optimizer to call `.step()` on."

    def forward(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
