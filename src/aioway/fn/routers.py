# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch

from .fakaten import FakatenFn, find_fakaten
from .fake import enabled_fake_mode, torch_fake_mode
from .guards import is_aten_op, is_prim_op, is_torchcodec_op, is_torchvision_op
from .modes import TDispatchFn, TDispatchMode, TFunctionFn, TFunctionMode
from .tracking import FnHistory

__all__ = [
    "track_fn",
    "fake_fn",
    "RouteDispatchOp",
    "RouteFunctionOp",
]

LOGGER = logging.getLogger(__name__)


FakatenRouter = cabc.Callable[[TDispatchFn], FakatenFn]


def only_route_in_fake(thunk: TDispatchFn):
    # Do not do anything in real mode, or in case the fake mode is temporarily disabled.
    if not enabled_fake_mode():
        return NotImplemented

    # For now, `Fakaten` supports aten, because `torchvision`, `torchcodec` rely on real data,
    # they do not have a good `Fakaten` to implement for now.
    # In those operations, real mode is force enabled right now.
    # See aioway#204 issue.
    if is_aten_op(thunk.func):
        return find_fakaten(thunk)

    if not any(
        is_op(thunk.func) for is_op in [is_prim_op, is_torchvision_op, is_torchcodec_op]
    ):
        raise AssertionError(f"Unknown kind of op: {thunk}")

    return NotImplemented


def no_route(thunk: TDispatchFn):
    return NotImplemented


@dcls.dataclass
class RouteDispatchOp(TDispatchMode):
    router: FakatenRouter
    history: FnHistory[TDispatchFn | FakatenFn] = dcls.field(default_factory=FnHistory)

    def __call__(self, thunk: TDispatchFn) -> torch.Tensor:
        # Create a `_ThunkType` and route implemented methods.

        fn: TDispatchFn | FakatenFn

        if (fn := self.router(thunk)) is NotImplemented:
            # Fn initialization failed, set it to the input `thunk`.
            fn = thunk

        assert isinstance(fn, TDispatchFn | FakatenFn), type(fn)

        # Here, we overwrite `fn`'s `__call__` inside `FakatenFn` if it's a special function.
        with capture_do_error(fn):
            result = fn.do()

        self.history.append(fn, result)
        return result


@dcls.dataclass
class RouteFunctionOp(TFunctionMode):
    """
    Saves the intermediate graph into a `FnHistory` object,
    and route the function to using `FakatenFn` if it's a `torch.ops.*` and in fake mode.
    """

    dispatch_router: RouteDispatchOp
    """
    The router for which to route the `torch.ops.*` operations.
    """

    history: FnHistory[TFunctionFn] = dcls.field(default_factory=FnHistory)
    """
    The `FnHistory` instance that would be responsible for tracking history,
    and which provides a graph API to interact with saved tensors.
    """

    @typing.override
    def __call__(self, thunk: TFunctionFn, /) -> torch.Tensor:
        with self.dispatch_router:
            result = self._execute_maybe_dispatch(thunk)

        self.history.append(thunk, result)
        return result

    def _execute_maybe_dispatch(self, thunk: TFunctionFn) -> torch.Tensor:
        """
        This function is run in dispatch mode if convertible.

        If `thunk` is convertible to `TDispatchFn`, convert it and run it.
        This happens sometimes with `torch.library` extension ops, which, if not handled,
        would just passed to dispatch mode anyways.
        """

        if (dispatch := thunk.dispatch()) is not NotImplemented:
            return dispatch.do()

        else:
            return thunk.do()


@ctxl.contextmanager
def capture_do_error(fn: TDispatchFn | FakatenFn):
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

    with tracker:
        yield tracker.history, dispatcher.history


@ctxl.contextmanager
def fake_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    dispatcher = RouteDispatchOp(only_route_in_fake)
    tracker = RouteFunctionOp(dispatcher)

    with torch_fake_mode(), tracker:
        yield tracker.history, dispatcher.history
