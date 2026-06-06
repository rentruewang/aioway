# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from torch import nn

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

    input: Hop
    "The input of the `NnHop`. Should output a `torch.Tensor`."

    @typing.override
    def _rebuild(self) -> typing.Self:
        new_module = self.nn_init()
        return dcls.replace(self, module=new_module)

    @property
    @typing.override
    def requires_grad(self) -> bool:
        module_grad = any(param.requires_grad for param in self.module.parameters())
        return self.input.requires_grad or module_grad


@hop_dcls
class NnLayerHop(NnHop):
    """
    The `Hop` subclass for normal layers in `nn.Module`s.
    It is a thunk so it has args, kwargs as attributes.
    """

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

    target: Hop
    "The target of the `NnLossHop`. Should output a `torch.Tensor`."

    @typing.override
    def forward(self) -> object:
        return self.module(self.input(), self.target())
