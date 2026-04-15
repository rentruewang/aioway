# Copyright (c) AIoWay Authors - All Rights Reserved

"This module contains utilities that uses the `__torch_dispatch__` mode."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import _ops
from torch.utils import _python_dispatch as pyd

from aioway.ctx import enabled_fake_mode, fake_mode
from aioway.schemas.attrs import attr

from ..fn import Fn, PatchTorchFn, TorchFn
from ..torch import is_aten_op, is_prim_op
from .impls import find_preview

__all__ = [
    "print_torch_dispatch",
    "log_torch_dispatch",
    "track_dispatch_fn",
    "fake_dispatch_fn",
]

LOGGER = logging.getLogger(__name__)


_ThunkType = cabc.Callable[..., TorchFn]
_TorchRouterMode = typing.Literal["dispatch", "function"]


_active_dispatch: Fn | None = None
"The current dispatch."


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
        return patch_aten_ops(func=func, types=types)

    assert is_prim_op(func), func
    return NotImplemented


def no_route(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _ThunkType:
    return NotImplemented


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

    def tensors(self):
        def params():
            for fn in self.data:
                yield from fn.tensors()

        yield from set(params())

    def parameters(self):
        for tensor in self.tensors():
            if tensor.requires_grad and tensor.is_leaf:
                yield tensor

    def numel(self) -> int:
        return sum(param.numel() for param in self.tensors())

    def memory(self) -> int:
        return sum(attr(param).memory() for param in self.tensors())


@dcls.dataclass
class TrackDispatchMode(pyd.TorchDispatchMode):
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
            # Ensure that only 1 dispatch is running at any given moment.
            with _ensure_single_dispatch(thunk):
                return thunk()
        except RuntimeError as re:
            fn = Fn(func, *args, **kwargs)
            raise ValueError(f"Function call '{fn}' failed.") from re


def patch_aten_ops(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> cabc.Callable[..., TorchFn]:
    assert is_aten_op(func), func

    if (patch := find_preview(func)) is NotImplemented:
        return NotImplemented

    return lambda *args, **kwargs: PatchTorchFn(func, patch, types, *args, **kwargs)


@ctxl.contextmanager
def track_dispatch_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    with TrackDispatchMode(router=no_route) as sdm:
        yield sdm.calls


@ctxl.contextmanager
def fake_dispatch_fn():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    with fake_mode(), TrackDispatchMode(router=only_route_aten_in_fake) as sdm:
        yield sdm.calls


@ctxl.contextmanager
def _ensure_single_dispatch(fn: Fn):
    "Ensure that only 1 dispatch is running at any given moment."

    global _active_dispatch

    if _active_dispatch is not None:
        raise ValueError("Cannot run 2 dispatches at once.")

    try:
        _active_dispatch = fn
        yield
    finally:
        _active_dispatch = None


def active_dispatch():
    return _active_dispatch
