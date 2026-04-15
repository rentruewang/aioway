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

from .fn import Fn, PatchFakeTorchFn, TorchFn
from .torch import is_aten_op, is_prim_op

__all__ = [
    "print_torch_dispatch",
    "log_torch_dispatch",
    "track_dispatch_fn_mode",
    "fake_dispatch_fn_mode",
    "track_function_fn_mode",
]

LOGGER = logging.getLogger(__name__)

_ThunkType = cabc.Callable[..., TorchFn]
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


class TorchContextManager(typing.Protocol):
    def __call__(self) -> typing.ContextManager[torch.Tensor]: ...


class TorchRouterFactory(typing.Protocol):
    def __call__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
    ) -> _ThunkType: ...


def torch_context_manager(mode: _TorchRouterMode, /):
    @typing.overload
    def decorator[T: type](typ: T, /) -> T: ...

    @typing.overload
    def decorator(func: TorchRouter, /) -> TorchContextManager: ...

    @typing.no_type_check
    def decorator(obj):
        return _create_mode(mode, obj)

    return decorator


def _create_mode(mode: _TorchRouterMode, obj: typing.Any, /):
    base_cls: type

    match mode:
        case "dispatch":
            func_name = "__torch_dispatch__"
            base_cls = pyd.TorchDispatchMode
        case "function":
            func_name = "__torch_function__"
            base_cls = overrides.TorchFunctionMode

    if isinstance(obj, type):
        return _create_mode_class(
            typ=obj,
            base_cls=base_cls,
            func_name=func_name,
        )

    elif isinstance(obj, types.FunctionType):
        return _create_mode_func(
            function=obj,
            base_cls=base_cls,
            func_name=func_name,
        )

    raise TypeError(f"Unhandled {type(obj)=}.")


def _create_mode_func(function: TorchRouter, base_cls: type, func_name: str):
    if not isinstance(function, types.FunctionType):
        raise ValueError(f"{function=} is not a function.")

    @typing.no_type_check
    def invoke(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        return function(func, types, args, kwargs)

    return type(function.__name__, (base_cls,), {func_name: invoke})


def _create_mode_class(typ: type[TorchRouter], base_cls: type, func_name: str):
    _check_class_has_call(typ)

    @typing.no_type_check
    def invoke(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        return self(func, types, args, kwargs)

    return type(typ.__name__, (typ, base_cls), {func_name: invoke})


def _check_class_has_call(base: type):
    assert callable(function := getattr(base, "__call__"))
    assert isinstance(function, types.FunctionType)


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
class _LogDispatch(TorchRouter):
    """
    Log every call to dispatch mode.
    """

    def __init__(self, level: int, logger: logging.Logger = LOGGER):
        self.level = level
        self.logger = logger

    def __call__(
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


log_torch_dispatch = _LogDispatch
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
    calls: list[TorchFn] = dcls.field(default_factory=list)

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}

        thunk: TorchFn
        # Create a `TorchDispatchThunk` and route implemented methods.
        if (thunk_init := self.router(func=func, types=types)) is NotImplemented:
            thunk = TorchFn(func, types, *args, **kwargs)
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
) -> cabc.Callable[..., TorchFn]:
    assert is_aten_op(func), func

    # If no `tsc.FakeTensor` exists, don't bother patching.
    if not any(issubclass(typ, tsc.FakeTensor) for typ in types):
        return NotImplemented

    if (patch := find_patch(func)) is NotImplemented:
        return NotImplemented

    return lambda *args, **kwargs: PatchFakeTorchFn(func, patch, types, *args, **kwargs)


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
