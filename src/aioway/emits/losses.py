# Copyright (c) AIoWay Authors - All Rights Reserved

import tensordict as td
import torch
from torch import nn

from aioway.spaces import Space, space_for_tcls

from .emitters import emitter_function

__all__ = ["PairLossModule", "LossTCls", "dispatch_mse_loss"]


class LossTCls(td.TensorClass):
    input: torch.Tensor
    target: torch.Tensor


class PairLossModule(nn.Module):
    """
    The marker for a loss function with signature `(input, target)`.
    """

    def __init__(self, loss_func: nn.Module) -> None:
        super().__init__()

        self.loss_func = loss_func

        # Must be `*Loss` according to torch.
        if not type(loss_func).__name__.endswith("Loss"):
            raise TypeError("Only wraps `nn.*Loss` modules.")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_func(input, target)


@emitter_function
def dispatch_mse_loss(observ: Space, action: Space) -> PairLossModule:
    if action != space_for_tcls(LossTCls):
        return NotImplemented

    return PairLossModule(nn.MSELoss())
