# Copyright (c) AIoWay Authors - All Rights Reserved

"This module contains utilities that uses the `__torch_dispatch__` mode."

import contextlib as ctxl
import dataclasses as dcls
import logging
import types
import typing
from collections import abc as cabc

import torch
from torch import _ops
from torch import _subclasses as tsc
from torch import overrides
from torch.utils import _python_dispatch as pyd

from aioway.ctx import enabled_fake_mode, fake_mode
from aioway.fn.patches import find_patch

from .fn import Fn, PatchTorchIrFn, TorchIrFn
from .torch import is_aten_op, is_prim_op

__all__ = [
    "print_torch_dispatch",
    "log_torch_dispatch",
    "track_dispatch_fn_mode",
    "fake_dispatch_fn_mode",
    "track_function_fn_mode",
]

LOGGER = logging.getLogger(__name__)

_ThunkType = cabc.Callable[..., TorchIrFn]
_TorchRouterMode = typing.Literal["dispatch", "function"]


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
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
    ) -> _ThunkType: ...


def torch_context_manager(mode: _TorchRouterMode, /):
    @typing.overload
    def decorator(typ: type, /) -> type[typing.ContextManager[torch.Tensor]]: ...

    @typing.overload
    def decorator(func: TorchRouter, /) -> typing.ContextManager[torch.Tensor]: ...

    @typing.no_type_check
    def decorator(obj):
        return _create_mode(mode, isinstance(obj, type))(obj)

    return decorator


def _create_mode(mode: _TorchRouterMode, is_type: bool, /):
    match mode, is_type:
        case "dispatch", False:
            return _dispatch_mode_function
        case "function", False:
            return _function_mode_function
        case "dispatch", True:
            return _dispatch_mode_class
        case "function", True:
            return _function_mode_class


def _dispatch_mode_function(function: TorchRouter, /):
    @typing.final
    class DispatchMode(pyd.TorchDispatchMode):
        @typing.no_type_check
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            return function(func, types, args, kwargs)

    return DispatchMode


def _function_mode_function(function: TorchRouter, /):
    @typing.final
    class FunctionMode(overrides.TorchFunctionMode):
        @typing.no_type_check
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            return function(func, types, args, kwargs)

    return FunctionMode


def _dispatch_mode_class(base: type[TorchRouter], /):
    fname = "__torch_dispatch__"
    assert (dispatch := getattr(base, fname)) is not None
    assert isinstance(dispatch, types.FunctionType)

    @typing.no_type_check
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        return dispatch(self, func, types, args, kwargs)

    new_type = type(
        base.__name__,
        (pyd.TorchDispatchMode, base),
        {"__torch_dispatch__": __torch_dispatch__},
    )

    return new_type


def _function_mode_class(base: type[TorchRouter], /):
    fname = "__torch_function__"
    assert (function := getattr(base, fname)) is not None
    assert isinstance(function, types.FunctionType)

    @typing.no_type_check
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        return function(self, func, types, args, kwargs)

    new_type = type(
        base.__name__,
        (overrides.TorchFunctionMode, base),
        {"__torch_function__": __torch_function__},
    )

    return new_type


@torch_context_manager("dispatch")
def print_torch_dispatch(
    func: _ops.OpOverload,
    types: tuple[type[torch.Tensor], ...],
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
):
    invoke = Fn(func, *args, **kwargs)
    result = invoke()
    print(f"{invoke!s} -> {result!r}")
    return result


@torch_context_manager("dispatch")
@dcls.dataclass
class _LogDispatchMode:
    """
    Log every call to dispatch mode.
    """

    _: dcls.KW_ONLY

    logger: logging.Logger
    "The logger to log to."

    level: int
    "The logging level to use."

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...],
        kwargs: dict[str, typing.Any],
    ):
        invoke = Fn(func, *args, **kwargs)
        result = invoke()
        self.logger.log(self.level, f"%s -> %s", invoke, result)
        return result


log_torch_dispatch = _LogDispatchMode
"""
Context manager to log the `__torch_dispatch__` calls.

Args:
    logger: The logger to use. Default to the one in this module.
    level: The level to log to. Default to `logging.DEBUG`.
"""


def only_route_aten_in_fake(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _ThunkType:
    if not enabled_fake_mode():
        raise RuntimeError("Only running in fake mode!")

    if is_aten_op(func):
        return patch_aten_ops_in_fake(func=func, types=types)

    assert is_prim_op(func), func
    return NotImplemented


def no_route(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _ThunkType:
    return NotImplemented


@dcls.dataclass
class _StoreFunctionMode(overrides.TorchFunctionMode):
    calls: list[Fn] = dcls.field(default_factory=list)

    def __torch_function__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        fn = Fn(func, *args, **kwargs)
        self.calls.append(fn)
        return fn()


@dcls.dataclass
class _StoreDispatchMode(pyd.TorchDispatchMode):
    router: TorchRouterFactory
    calls: list[TorchIrFn] = dcls.field(default_factory=list)

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}

        thunk: TorchIrFn
        # Create a `TorchDispatchThunk` and route implemented methods.
        if (thunk_init := self.router(func=func, types=types)) is NotImplemented:
            thunk = TorchIrFn(func, types, *args, **kwargs)
        else:
            thunk = thunk_init(*args, **kwargs)

        self.calls.append(thunk)

        try:
            return thunk()
        except RuntimeError as re:
            fn = Fn(func, *args, **kwargs)
            raise ValueError(f"Function call '{fn}' failed.") from re


def patch_aten_ops_in_fake(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> cabc.Callable[..., TorchIrFn]:
    assert is_aten_op(func), func

    # If no `tsc.FakeTensor` exists, don't bother patching.
    if not any(issubclass(typ, tsc.FakeTensor) for typ in types):
        return NotImplemented

    if (patch := find_patch(func)) is NotImplemented:
        return NotImplemented

    return lambda *args, **kwargs: PatchTorchIrFn(func, patch, types, *args, **kwargs)


@ctxl.contextmanager
def track_dispatch_fn_mode():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    with _StoreDispatchMode(router=no_route) as sdm:
        yield sdm.calls


@ctxl.contextmanager
def fake_dispatch_fn_mode():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    with fake_mode(), _StoreDispatchMode(router=only_route_aten_in_fake) as sdm:
        yield sdm.calls


@ctxl.contextmanager
def track_function_fn_mode():
    with _StoreFunctionMode() as sfm:
        yield sfm.calls
