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
    from aioway.torch.nn import NnInit_

__all__ = ["NnUFunc", "NnUFuncThunk"]


class NnUFuncThunk(UFuncThunk, TensorIter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def nn_init(self) -> NnInit_:
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

    def __init__(self, init: NnInit_, module: nn.Module) -> None:
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
