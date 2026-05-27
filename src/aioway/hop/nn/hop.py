# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._fn import thunk_dcls

from ..hop import Hop


@thunk_dcls
class NnLayerHop(Hop):
    """
    The `Hop` subclass for normal layers in `nn.Module`s.
    It is a thunk so it has args, kwargs as attributes.
    """

    module: nn.Module
    "`NnLayerHop` stores the module."

    input: Hop
    "The input of the `NnLayerHop`. Should output a `torch.Tensor`."

    @typing.override
    def forward(self) -> object:
        return self.module(self.input)

    def parameters(self):
        "Pass forward the `.parameters()` of modules."
        yield from self.module.parameters()


@thunk_dcls
class NnLossHop(Hop):
    """
    The `Hop` subclass for loss functions that are `nn.Module`s.
    """

    loss: nn.Module
    "`NnLossHop` stores the module."

    input: Hop
    "The input of the `NnLossHop`. Should output a `torch.Tensor`."

    target: Hop
    "The target of the `NnLossHop`. Should output a `torch.Tensor`."

    @typing.override
    def forward(self) -> object:
        return self.loss(self.input, self.target)
