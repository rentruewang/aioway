# Copyright (c) AIoWay Authors - All Rights Reserved

"`UFunc` implementation for `torch.nn`."

import abc
import inspect
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._ufuncs import UFunc
from aioway._utils import Sign
from aioway.modes import NnInitThunk

__all__ = ["module_ufunc", "NnUFunc", "NnLayerUFunc", "NnLossUFunc"]


class _ModuleType[**P, T: nn.Module](typing.Protocol):
    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


def module_ufunc[**P, T: nn.Module](
    module: _ModuleType[P, T], *args: P.args, **kwargs: P.kwargs
):
    type_name = module.__name__

    ufunc_type = NnLossUFunc if type_name.endswith("Loss") else NnLayerUFunc
    return ufunc_type(module, *args, **kwargs)


class NnUFunc[**P = ...](UFunc, abc.ABC):
    """
    The `UFunc` for `nn.Module` type.
    """

    def __init__(
        self, func: cabc.Callable[P, nn.Module], *args: P.args, **kwargs: P.kwargs
    ):
        self._func = func
        self._args = args
        self._kwargs = kwargs

        self._verify_signature()
        self.build()

    @property
    def __signature__(self) -> inspect.Signature:
        return self._siganture.signature

    def _verify_signature(self) -> None:
        # This is supposed to raise `TypeError` upon wrong arguments.
        self._siganture.bind(*self._args, **self._kwargs)

    def build(self) -> None:
        "Rebuild the `self.module`. This is useful for rebuilding under different modes."

        # Using an `NnInitThunk` s.t. the modes would be respected.
        thunk = NnInitThunk(self._func, *self._args, **self._kwargs)
        self._module: nn.Module = thunk()

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def _siganture(self) -> Sign:
        return Sign.from_callable(self._func)

    def arguments(self):
        return self._signature.apply(*self._args, **self._kwargs)


class NnLayerUFunc[**P = ...](NnUFunc[P]):
    "The `NnUFunc` for layers that are not losses."

    @typing.override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert isinstance(input, torch.Tensor)
        output = self.module(input)
        assert isinstance(output, torch.Tensor)
        return output


class NnLossUFunc[**P = ...](NnUFunc[P]):
    "The `NnUFunc` for layers that are losses."

    @typing.override
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        output = self.module(input, target)
        assert isinstance(output, torch.Tensor)
        return output
