# Copyright (c) AIoWay Authors - All Rights Reserved

"`UFunc` implementation for `torch.nn`."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._iters import TensorIter, node_dcls
from aioway._ufuncs import UFunc

if typing.TYPE_CHECKING:
    from aioway.torch.nn import NnInit

__all__ = ["NnUFunc", "NnLayerIter", "NnLossIter"]


class NnUFunc(UFunc, abc.ABC):
    iter: cabc.Callable[..., NnIter]

    def __init__(self, init: NnInit, module: nn.Module) -> None:
        self._init = init
        self._module = module

        self._validate()

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def nn_init(self) -> NnInit:
        return self._init

    @abc.abstractmethod
    def _validate(self) -> None:
        raise NotImplementedError


@typing.final
class NnLossUFunc(NnUFunc):

    @typing.override
    def forward(self, input: torch.Tensor, target: torch.Tensor, /) -> torch.Tensor:
        return self._module(input, target)

    @typing.override
    def _validate(self):
        from .losses import BaseLoss

        assert isinstance(self.nn_init, BaseLoss)
        assert isinstance(self.module, self.nn_init.NN)

    @typing.override
    def iter(self, input: TensorIter, target: TensorIter) -> NnLossIter:
        return NnLossIter(self.nn_init, self.module, input, target)


@typing.final
class NnLayerUFunc(NnUFunc):
    @typing.override
    def forward(self, input: torch.Tensor, /) -> torch.Tensor:
        return self._module(input)

    @typing.override
    def _validate(self):
        from .losses import BaseLoss

        assert not isinstance(self.nn_init, BaseLoss)
        assert isinstance(self.module, self.nn_init.NN)

    @typing.override
    def iter(self, input: TensorIter) -> NnLayerIter:
        return NnLayerIter(self.nn_init, self.module, input)


@node_dcls
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


@node_dcls
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


@node_dcls
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
