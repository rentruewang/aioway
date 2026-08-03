# Copyright (c) AIoWay Authors - All Rights Reserved

from torch import nn
import torch

__all__ = ["PairLossModule"]


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
