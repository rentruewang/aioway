# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / logging related utilities."

import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import _ops

from .fn import FnStack, Thunk
from .modes import (
    TorchDispatchFn,
    TorchDispatchMode,
    TorchFunctionFn,
    TorchFunctionMode,
)

__all__ = [
    "print_torch_dispatch",
    "LogTorchDispatch",
    "TorchFunctionStack",
    "TorchDispatchStack",
]

LOGGER = logging.getLogger(__name__)


@TorchDispatchMode.register
def print_torch_dispatch(
    func: _ops.OpOverload,
    types: tuple[type[torch.Tensor], ...],
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
):
    """
    Print the dispatcher.
    """
    invoke = Thunk(func, *args, **kwargs)

    result = invoke.do()
    print(invoke)
    return result


@dcls.dataclass
class LogTorchDispatch(TorchDispatchMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(
        self,
        op: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> torch.Tensor:
        invoke = Thunk(op, *args, **kwargs)
        result = invoke.do()
        self.logger.log(self.level, "%s", invoke)
        return result


@dcls.dataclass
class TorchFunctionStack(TorchFunctionMode):
    stack: FnStack[TorchFunctionFn] = dcls.field(default_factory=FnStack)

    @typing.override
    def __call__(
        self,
        func: cabc.Callable[..., typing.Any],
        types: tuple[type, ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        thunk = TorchFunctionFn(func, types, args, kwargs)
        with self.stack.track(thunk):
            return thunk.do()


@dcls.dataclass
class TorchDispatchStack(TorchDispatchMode):
    stack: FnStack[TorchDispatchFn] = dcls.field(default_factory=FnStack)

    @typing.override
    def __call__(
        self,
        op: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> torch.Tensor:
        thunk = TorchDispatchFn(op, types, args, kwargs)
        with self.stack.track(thunk):
            return thunk.do()
