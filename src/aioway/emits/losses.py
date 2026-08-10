# Copyright (c) AIoWay Authors - All Rights Reserved

import tensordict as td
import torch
from torch import nn
from torchrl import data as rldata

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

    def forward(self, pair: LossTCls) -> torch.Tensor:
        return self.loss_func(pair.input, pair.target)


@emitter_function
def dispatch_mse_loss(
    observ: rldata.TensorSpec, action: rldata.TensorSpec
) -> PairLossModule:
    if not isinstance(observ, rldata.Composite) or observ.data_cls != LossTCls:
        return NotImplemented

    if not isinstance(action, rldata.Unbounded) or action.shape != ():
        return NotImplemented

    return PairLossModule(nn.MSELoss())
