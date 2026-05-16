# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking / routing related `Fn`s."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import rich

from aioway._common import (
    is_aten_op,
    is_prim_op,
    is_torchcodec_op,
    is_torchvision_op,
    replace_tensors,
    replace_tensors_with_attr,
)
from aioway.fake import enabled_fake_mode, torch_fake_mode
from aioway.fn.modes.modules import MInitMode

from .hists import HistoryTensorGraph
from .modes import TDisFn, TDisMode, TFuncFn, TFuncMode
from .op import FateFn

__all__ = [
    "track_fn",
    "fake_fn",
    "PrintTorchFunction",
    "PrintTorchDispatch",
    "LogTorchFunction",
    "LogTorchDispatch",
    "RouteTorchDispatch",
    "RouteTorchFunction",
]

LOGGER = logging.getLogger(__name__)


FateRouter = cabc.Callable[[TDisFn], FateFn]


class _HasRichFlagMixin:
    def __init__(self, rich: bool = False) -> None:
        super().__init__()
        self._rich = rich


class PrintTorchFunction(_HasRichFlagMixin, TFuncMode):
    @typing.override
    def __call__(self, thunk: TFuncFn, /) -> object:
        return _ThunkPrinter(rich=self._rich)(thunk)


class PrintTorchDispatch(_HasRichFlagMixin, TDisMode):
    @typing.override
    def __call__(self, thunk: TDisFn, /) -> object:
        return _ThunkPrinter(rich=self._rich)(thunk)


@dcls.dataclass(frozen=True)
class _ThunkPrinter:
    rich: bool
    "Use rich for printing."

    def __call__(self, thunk: TFuncFn | TDisFn) -> object:
        self.print("invoke", thunk)
        result = thunk.do()
        self.print("return", thunk, "->", replace_tensors_with_attr(result))
        return result

    @property
    def print(self):
        return rich.print if self.rich else print


@dcls.dataclass
class LogTorchFunction(TFuncMode):
    """
    Log every call to function mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(self, thunk: TFuncFn) -> object:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result


@dcls.dataclass
class LogTorchDispatch(TDisMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    @typing.override
    def __call__(self, thunk: TDisFn) -> object:
        result = thunk.do()
        self.logger.log(self.level, "%s", thunk)
        return result


def only_route_in_fake(thunk: TDisFn):
    """
    Route in fake mode.

    `NotImplemented` is returned when this function cannot handle the input,
    e.g. it's not in fake mode, or if there is no `Fate` equivalent.
    """

    # Do not do anything in real mode, or in case the fake mode is temporarily disabled.
    if not enabled_fake_mode():
        return NotImplemented

    # For now, `Fate` supports aten, because `torchvision`, `torchcodec` rely on real data,
    # they do not have a good `Fate` to implement for now.
    # In those operations, real mode is force enabled right now.
    # See aioway#204 issue.
    if is_aten_op(thunk.func):
        return FateFn.find_fate(thunk)

    if not any(
        is_op(thunk.func) for is_op in [is_prim_op, is_torchvision_op, is_torchcodec_op]
    ):
        raise AssertionError(f"Unknown kind of op: {thunk}")

    return NotImplemented


def no_route(thunk: TDisFn):
    return NotImplemented


class CloneDispatchOp(TDisMode):
    @typing.override
    def __call__(self, thunk: TDisFn, /) -> object:
        result = thunk.do()

        # In fake mode, clone the tensor to prevent `FakeTensor` reuse. Should be cheap.
        if enabled_fake_mode():
            result = replace_tensors(result, lambda tensor: tensor.clone())

        return result


@dcls.dataclass
class RouteTorchDispatch(TDisMode):
    "The router at the torch dispatch level."

    router: FateRouter
    "The router that is responsible for finding `Fate` when implemented."

    history: HistoryTensorGraph[TDisFn | FateFn] = dcls.field(
        default_factory=HistoryTensorGraph
    )
    "The history used for tracking."

    def __call__(self, thunk: TDisFn) -> object:
        fn: TDisFn | FateFn

        if (fn := self.router(thunk)) is NotImplemented:
            # Cannot find corresponding operator, set it to the input `thunk`.
            fn = thunk

        assert isinstance(fn, TDisFn | FateFn), type(fn)

        # Here, `FateFn` would do its magic and overwrite functions.
        with capture_do_error(fn):
            result = fn.do()

        self.history.append(fn, result)
        return result


@dcls.dataclass
class RouteModuleInit(MInitMode):
    pass


@dcls.dataclass
class RouteTorchFunction(TFuncMode):
    """
    Saves the intermediate graph into a `FnHistory` object,
    and route the function to using `FateFn` if it's a `torch.ops.*` and in fake mode.
    """

    dispatcher: RouteTorchDispatch
    """
    The router for which to route the `torch.ops.*` operations.
    """

    history: HistoryTensorGraph[TFuncFn] = dcls.field(
        default_factory=HistoryTensorGraph
    )
    """
    The `FnHistory` instance that would be responsible for tracking history,
    and which provides a graph API to interact with saved tensors.
    """

    @typing.override
    def __call__(self, thunk: TFuncFn, /) -> object:
        with self.dispatcher.ctx():
            result = thunk.do()

        self.history.append(thunk, result)
        return result


@ctxl.contextmanager
def capture_do_error(fn: TDisFn | FateFn):
    try:
        yield
    except RuntimeError as err:
        raise ValueError(f"Function call '{fn}' failed.") from err


@ctxl.contextmanager
def track_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    dispatcher = RouteTorchDispatch(no_route)
    tracker = RouteTorchFunction(dispatcher)

    with tracker.ctx():
        yield tracker.history, dispatcher.history


@ctxl.contextmanager
def fake_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    dispatcher = RouteTorchDispatch(only_route_in_fake)
    tracker = RouteTorchFunction(dispatcher)

    with torch_fake_mode(), tracker.ctx(), CloneDispatchOp().ctx():
        yield tracker.history, dispatcher.history
