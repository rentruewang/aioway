# Copyright (c) AIoWay Authors - All Rights Reserved

import tensordict as td
import torch
from torch import nn

from aioway._torch import Schema
from aioway.spaces import Space, TClsSpace

from .emitters import emitter_function

__all__ = ["PairLossModule", "LossTCls", "LossSpace"]


class LossTCls(td.TensorClass):
    input: torch.Tensor
    target: torch.Tensor


class LossSpace(TClsSpace):
    KLASS = LossTCls

    def _check_attrs(self, attrs: Schema):
        assert attrs.keys() == {"input", "target"}

    def _check_data(self, data: LossTCls):
        assert isinstance(data, LossTCls)

    def _sample_n(self, n: int):
        return LossTCls(torch.randn(n), torch.randn(n))


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
def dispatch_loss(observ: Space, action: Space) -> PairLossModule:
    raise NotImplementedError
