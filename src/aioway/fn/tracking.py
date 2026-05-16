# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / logging related utilities."

import dataclasses as dcls
import logging
import typing

import rich

from aioway._common import replace_tensors_with_attr

from .modes import TDispatchFn, TDispatchMode, TFunctionFn, TFunctionMode

__all__ = [
    "PrintTorchFunction",
    "PrintTorchDispatch",
    "LogTorchFunction",
    "LogTorchDispatch",
]

LOGGER = logging.getLogger(__name__)


class _HasRichFlagMixin:
    def __init__(self, rich: bool = False) -> None:
        super().__init__()
        self._rich = rich


class PrintTorchFunction(_HasRichFlagMixin, TFunctionMode):
    @typing.override
    def __call__(self, thunk: TFunctionFn, /) -> object:
        return _ThunkPrinter(rich=self._rich)(thunk)


class PrintTorchDispatch(_HasRichFlagMixin, TDispatchMode):
    @typing.override
    def __call__(self, thunk: TDispatchFn, /) -> object:
        return _ThunkPrinter(rich=self._rich)(thunk)


@dcls.dataclass(frozen=True)
class _ThunkPrinter:
    rich: bool
    "Use rich for printing."

    def __call__(self, thunk: TFunctionFn | TDispatchFn) -> object:
        self.print("invoke", thunk)
        result = thunk.do()
        self.print("return", thunk, "->", replace_tensors_with_attr(result))
        return result

    @property
    def print(self):
        return rich.print if self.rich else print


@dcls.dataclass
class LogTorchFunction(TFunctionMode):
    """
    Log every call to function mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(self, thunk: TFunctionFn) -> object:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result


@dcls.dataclass
class LogTorchDispatch(TDispatchMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(self, thunk: TDispatchFn) -> object:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result
