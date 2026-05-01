# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import typing
from collections import abc as cabc

import torch
from torch import _ops, overrides
from torch.utils import _python_dispatch as pyd

__all__ = ["TorchFunctionMode", "TorchDispatchMode"]

_FUNCTION_MODES: list[TorchFunctionMode] = []
_DISPATCH_MODES: list[TorchDispatchMode] = []


class TorchFunctionMode(overrides.TorchFunctionMode, abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        func: cabc.Callable[..., typing.Any],
        types: tuple[type, ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        raise NotImplementedError

    @typing.override
    def __torch_function__(self, func, types, args=..., kwargs=None) -> typing.Any:
        kwargs = kwargs or {}
        args = () if args == ... else args

        with _push_mode(self, _FUNCTION_MODES):
            return self(func, types, *args, **kwargs)


class TorchDispatchMode(pyd.TorchDispatchMode, abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        raise NotImplementedError

    @typing.override
    def __torch_dispatch__(self, func, types, args=..., kwargs=None) -> typing.Any:
        kwargs = kwargs or {}
        args = () if args == ... else args

        with _push_mode(self, _DISPATCH_MODES):
            return self(func, types, *args, **kwargs)


@ctxl.contextmanager
def _push_mode[T](item: T, stack: list[T]):
    try:
        stack.append(item)
        yield
    finally:
        stack.pop()
