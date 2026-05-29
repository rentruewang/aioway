# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

from torch import nn
import dataclasses as dcls

from ..hop import Hop, hop_dcls

if typing.TYPE_CHECKING:
    from .modules import NnInit

__all__ = ["NnHop", "NnLayerHop", "NnLossHop"]


@hop_dcls
class NnHop(Hop, abc.ABC):
    nn_init: NnInit
    "The config used to initialize the `Hop`."

    module: nn.Module
    "The `nn.Module` initialized by `nn_init`."

    @typing.override
    def _rebuild(self) -> typing.Self:
        new_module = self.nn_init()
        return dcls.replace(self, module=new_module)


@hop_dcls
class NnLayerHop(NnHop):
    """
    The `Hop` subclass for normal layers in `nn.Module`s.
    It is a thunk so it has args, kwargs as attributes.
    """

    input: Hop
    "The input of the `NnLayerHop`. Should output a `torch.Tensor`."

    @typing.override
    def forward(self) -> object:
        return self.module(self.input())

    def parameters(self):
        "Pass forward the `.parameters()` of modules."
        yield from self.module.parameters()


@hop_dcls
class NnLossHop(NnHop):
    """
    The `Hop` subclass for loss functions that are `nn.Module`s.
    """

    input: Hop
    "The input of the `NnLossHop`. Should output a `torch.Tensor`."

    target: Hop
    "The target of the `NnLossHop`. Should output a `torch.Tensor`."

    @typing.override
    def forward(self) -> object:
        return self.module(self.input(), self.target())
