# Copyright (c) AIoWay Authors - All Rights Reserved

"`UFunc` implementation for `torch.nn`."

import abc
import copy
import typing

import torch
from torch import nn

from aioway._iters import TensorIter
from aioway._ufuncs import UFunc, UFuncThunk

if typing.TYPE_CHECKING:
    from aioway.torch.nn import NnInit

__all__ = ["NnUFunc", "NnUFuncThunk"]


class NnUFuncThunk(UFuncThunk, TensorIter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def nn_init(self) -> NnInit:
        return self.ufunc.nn_init

    @property
    def module(self) -> nn.Module:
        return self.ufunc.module

    @property
    def ufunc(self) -> NnUFunc:
        ufunc = super().ufunc
        assert isinstance(ufunc, NnUFunc)
        return ufunc

    @typing.override
    def _rebuild(self) -> typing.Self:
        new_module = self.nn_init()
        self.ufunc.module = new_module
        return copy.copy(self)

    @property
    @typing.override
    def requires_grad(self) -> bool:
        module_grad = any(param.requires_grad for param in self.module.parameters())
        return module_grad or any(dep.requires_grad for dep in self.deps())

    def parameters(self):
        yield from self.module.parameters()


class NnUFunc(UFunc, abc.ABC):
    THUNK: typing.ClassVar[type[UFuncThunk]] = NnUFuncThunk

    def __init__(self, init: NnInit, module: nn.Module) -> None:
        self.nn_init = init
        self.module = module

        self._validate()

    @abc.abstractmethod
    def _validate(self) -> None:
        raise NotImplementedError


@typing.final
class NnLossUFunc(NnUFunc):

    @typing.override
    def forward(self, input: torch.Tensor, target: torch.Tensor, /) -> torch.Tensor:
        return self.module(input, target)

    @typing.override
    def _validate(self):
        from .losses import BaseLoss

        assert isinstance(self.nn_init, BaseLoss)
        assert isinstance(self.module, self.nn_init.NN)


@typing.final
class NnLayerUFunc(NnUFunc):
    @typing.override
    def forward(self, input: torch.Tensor, /) -> torch.Tensor:
        return self.module(input)

    @typing.override
    def _validate(self):
        from .losses import BaseLoss

        assert not isinstance(self.nn_init, BaseLoss)
        assert isinstance(self.module, self.nn_init.NN)


# @node_dcls
# class NnIter(TensorIter, abc.ABC):
#     nn_init: NnInit
#     "The config used to initialize the `Iter`."

#     module: nn.Module
#     "The `nn.Module` initialized by `nn_init`."

#     input: TensorIter
#     "The input of the `NnIter`. Should output a `torch.Tensor`."

#     @typing.override
#     def _rebuild(self) -> typing.Self:
#         new_module = self.nn_init()
#         return dcls.replace(self, module=new_module)

#     @property
#     @typing.override
#     def requires_grad(self) -> bool:
#         module_grad = any(param.requires_grad for param in self.module.parameters())
#         return self.input.requires_grad or module_grad


# @node_dcls
# class NnLayerIter(NnIter):
#     """
#     The `Iter` subclass for normal layers in `nn.Module`s.
#     It is a thunk so it has args, kwargs as attributes.
#     """

#     @typing.override
#     def iterate(self):
#         for input in self.input:
#             yield self.module(input)

#     def parameters(self):
#         "Pass forward the `.parameters()` of modules."
#         yield from self.module.parameters()


# @node_dcls
# class NnLossIter(NnIter):
#     """
#     The `Iter` subclass for loss functions that are `nn.Module`s.
#     """

#     target: TensorIter
#     "The target of the `NnLossIter`. Should output a `torch.Tensor`."

#     @typing.override
#     def iterate(self):
#         for input, target in zip(self.input, self.target):
#             yield self.module(input, target)
