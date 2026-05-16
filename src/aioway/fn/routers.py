# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

from aioway._common import (
    is_aten_op,
    is_prim_op,
    is_torchcodec_op,
    is_torchvision_op,
    replace_tensors,
)
from aioway.fake import enabled_fake_mode, torch_fake_mode

from .fate import FateFn
from .tensors import TDispatchFn, TDispatchMode, TFunctionFn, TFunctionMode
from .tracking import FnHistory

__all__ = [
    "track_fn",
    "fake_fn",
    "RouteDispatchOp",
    "RouteFunctionOp",
]

LOGGER = logging.getLogger(__name__)


FateRouter = cabc.Callable[[TDispatchFn], FateFn]


def only_route_in_fake(thunk: TDispatchFn):
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


def no_route(thunk: TDispatchFn):
    return NotImplemented


class CloneDispatchOp(TDispatchMode):
    @typing.override
    def __call__(self, thunk: TDispatchFn, /) -> object:
        result = thunk.do()

        # In fake mode, clone the tensor to prevent `FakeTensor` reuse. Should be cheap.
        if enabled_fake_mode():
            result = replace_tensors(result, lambda tensor: tensor.clone())

        return result


@dcls.dataclass
class RouteDispatchOp(TDispatchMode):
    "The router at the torch dispatch level."

    router: FateRouter
    "The router that is responsible for finding `Fate` when implemented."

    history: FnHistory[TDispatchFn | FateFn] = dcls.field(default_factory=FnHistory)
    "The history used for tracking."

    def __call__(self, thunk: TDispatchFn) -> object:
        fn: TDispatchFn | FateFn

        if (fn := self.router(thunk)) is NotImplemented:
            # Cannot find corresponding operator, set it to the input `thunk`.
            fn = thunk

        assert isinstance(fn, TDispatchFn | FateFn), type(fn)

        # Here, `FateFn` would do its magic and overwrite functions.
        with capture_do_error(fn):
            result = fn.do()

        self.history.append(fn, result)
        return result


@dcls.dataclass
class RouteFunctionOp(TFunctionMode):
    """
    Saves the intermediate graph into a `FnHistory` object,
    and route the function to using `FateFn` if it's a `torch.ops.*` and in fake mode.
    """

    dispatcher: RouteDispatchOp
    """
    The router for which to route the `torch.ops.*` operations.
    """

    history: FnHistory[TFunctionFn] = dcls.field(default_factory=FnHistory)
    """
    The `FnHistory` instance that would be responsible for tracking history,
    and which provides a graph API to interact with saved tensors.
    """

    @typing.override
    def __call__(self, thunk: TFunctionFn, /) -> object:
        with self.dispatcher.ctx():
            result = thunk.do()

        self.history.append(thunk, result)
        return result


@ctxl.contextmanager
def capture_do_error(fn: TDispatchFn | FateFn):
    try:
        yield
    except RuntimeError as err:
        raise ValueError(f"Function call '{fn}' failed.") from err


@ctxl.contextmanager
def track_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    dispatcher = RouteDispatchOp(no_route)
    tracker = RouteFunctionOp(dispatcher)

    with tracker.ctx():
        yield tracker.history, dispatcher.history


@ctxl.contextmanager
def fake_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    dispatcher = RouteDispatchOp(only_route_in_fake)
    tracker = RouteFunctionOp(dispatcher)

    with torch_fake_mode(), tracker.ctx(), CloneDispatchOp().ctx():
        yield tracker.history, dispatcher.history
