# Copyright (c) AIoWay Authors - All Rights Reserved

"This module contains utilities that uses the `__torch_dispatch__` mode."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import _ops
from torch import _subclasses as tsc
from torch import overrides
from torch.utils import _python_dispatch as pyd

from aioway.ctx import enabled_fake_mode, fake_mode
from aioway.fn.patches import find_patch
from aioway.schemas.attrs import attr

from .fn import Fn, PatchTorchFn, TorchFn
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


class TorchRouterFactory(typing.Protocol):
    def __call__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
    ) -> _ThunkType: ...


class _PrintDispatch(pyd.TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        invoke = Fn(func, *args, **kwargs)
        result = invoke()
        print(f"{invoke!s} -> {result!r}")
        return result


@ctxl.contextmanager
def print_torch_dispatch():
    with _PrintDispatch():
        yield


@dcls.dataclass
class _LogDispatch(pyd.TorchDispatchMode):
    """
    Log every call to dispatch mode.
    """

    level: int
    "The level to log to."

    logger: logging.Logger = LOGGER
    "The logger to log to. Default to the one in the current module."

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
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


@dcls.dataclass(frozen=True)
class FnList:
    data: list[TorchFn] = dcls.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> TorchFn:
        return self.data[idx]

    def __iter__(self):
        yield from self.data

    def append(self, item: TorchFn, /):
        self.data.append(item)

    def pop(self):
        return self.data.pop()

    def parameters(self):
        def all_params():
            for fn in self.data:
                yield from fn.parameters()

        yield from set(all_params())

    def numel(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def memory(self) -> int:
        return sum(attr(param).memory() for param in self.parameters())


@dcls.dataclass
class _StoreDispatchMode(pyd.TorchDispatchMode):
    router: TorchRouterFactory
    calls: FnList = dcls.field(default_factory=FnList)

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

    return lambda *args, **kwargs: PatchTorchFn(func, patch, types, *args, **kwargs)


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
