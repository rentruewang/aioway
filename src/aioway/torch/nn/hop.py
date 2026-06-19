# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

from torch import nn

from aioway._comps import TensorIter, iter_dcls

if typing.TYPE_CHECKING:
    from aioway.torch.nn import NnInit

__all__ = ["NnIter", "NnLayerIter", "NnLossIter"]


@iter_dcls
class NnIter(TensorIter, abc.ABC):
    nn_init: NnInit
    "The config used to initialize the `Iter`."

    module: nn.Module
    "The `nn.Module` initialized by `nn_init`."

    input: TensorIter
    "The input of the `NnIter`. Should output a `torch.Tensor`."

    @typing.override
    def _rebuild(self) -> typing.Self:
        new_module = self.nn_init()
        return dcls.replace(self, module=new_module)

    @property
    @typing.override
    def requires_grad(self) -> bool:
        module_grad = any(param.requires_grad for param in self.module.parameters())
        return self.input.requires_grad or module_grad


@iter_dcls
class NnLayerIter(NnIter):
    """
    The `Iter` subclass for normal layers in `nn.Module`s.
    It is a thunk so it has args, kwargs as attributes.
    """

    @typing.override
    def iterate(self):
        for input in self.input:
            yield self.module(input)

    def parameters(self):
        "Pass forward the `.parameters()` of modules."
        yield from self.module.parameters()


@iter_dcls
class NnLossIter(NnIter):
    """
    The `Iter` subclass for loss functions that are `nn.Module`s.
    """

    target: TensorIter
    "The target of the `NnLossIter`. Should output a `torch.Tensor`."

    @typing.override
    def iterate(self):
        for input, target in zip(self.input, self.target):
            yield self.module(input, target)
