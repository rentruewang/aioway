# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import _ops

from .fake import enabled_fake_mode, fake_mode
from .guards import is_aten_op, is_prim_op
from .modes import TorchDispatchFn, TorchDispatchMode, TorchFunctionMode
from .previews import PreviewFn, PreviewFnFinder, TensorThunk
from .tracking import DispatchHistory, FnHistory

__all__ = [
    "track_function_fn",
    "track_dispatch_fn",
    "fake_dispatch_fn",
    "RouteDispatchOp",
]

LOGGER = logging.getLogger(__name__)


_CreatePreview = cabc.Callable[..., PreviewFn]
_TorchRouterMode = typing.Literal["dispatch", "function"]
_TorchThunk = TorchDispatchFn | PreviewFn


@typing.runtime_checkable
class TorchRouter(typing.Protocol):
    def __call__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...],
        kwargs: dict[str, typing.Any],
    ) -> torch.Tensor: ...


class TorchRouterFactory(typing.Protocol):
    def __call__(
        self, func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
    ) -> _CreatePreview: ...


def only_route_aten_in_fake(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _CreatePreview:
    if not enabled_fake_mode():
        raise RuntimeError("Only running in fake mode!")

    if is_aten_op(func):
        return aten_ops_preview(func, types)

    assert is_prim_op(func), func
    return NotImplemented


def no_route(*args, **kwargs) -> _CreatePreview:
    return NotImplemented


@dcls.dataclass
class SaveFunctionHistory(TorchFunctionMode):
    """
    Saves the intermediate graph into a `FnHistory` object.
    """

    history: FnHistory[TensorThunk] = dcls.field(default_factory=FnHistory)
    """
    The `FnHistory` instance that would be responsible for tracking history,
    and which provides a graph API to interact with saved tensors.
    """

    @typing.override
    def __call__(
        self,
        func: cabc.Callable[..., typing.Any],
        types: tuple[type, ...],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        thunk = TensorThunk(func, args, kwargs)
        result = thunk.do()
        self.history.append(thunk, result)
        return result


@dcls.dataclass
class RouteDispatchOp(TorchDispatchMode):
    router: TorchRouterFactory
    history: DispatchHistory = dcls.field(default_factory=DispatchHistory)

    def __call__(
        self,
        op: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        *args: tuple[typing.Any, ...],
        **kwargs: dict[str, typing.Any],
    ):
        # Create a `_ThunkType` and route implemented methods.
        fn_init = self.router(op, types)
        fn: _TorchThunk

        if (
            False
            # Not ATen operator.
            or fn_init is NotImplemented
            # Fn is not handled.
            or (fn := fn_init(*args, **kwargs)) is NotImplemented
        ):
            fn = TorchDispatchFn(op, types, args, kwargs)

        assert isinstance(fn, _TorchThunk), type(fn)

        # Here, we overwrite `fn`'s `__call__` inside `PreviewFn` if it's a special function.
        with capture_do_error(fn):
            result = fn.do()

        self.history.append(fn, result)
        return result


def aten_ops_preview(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _CreatePreview:
    assert is_aten_op(func), func
    return PreviewFnFinder(func)


@ctxl.contextmanager
def capture_do_error(fn: _TorchThunk):
    try:
        yield
    except RuntimeError as err:
        raise ValueError(f"Function call '{fn}' failed.") from err


@ctxl.contextmanager
def track_function_fn():
    """
    Track all `torch.*` and `Tensor.*` function calls.
    """

    with SaveFunctionHistory() as sfh:
        yield sfh.history


@ctxl.contextmanager
def track_dispatch_fn(router: TorchRouterFactory = no_route):
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    with RouteDispatchOp(router=router) as sdm:
        yield sdm.history


@ctxl.contextmanager
def fake_dispatch_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    with fake_mode(), track_dispatch_fn(only_route_aten_in_fake) as history:
        yield history
