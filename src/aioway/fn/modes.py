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

from .fn import Fn, TorchIrFn
from .torch import is_aten_op, is_prim_op

__all__ = [
    "print_torch_dispatch",
    "log_torch_dispatch",
    "track_fn_mode",
    "fake_fn_mode",
]

LOGGER = logging.getLogger(__name__)


class _PrintDispatchMode(pyd.TorchDispatchMode):
    """
    Print every call to dispatch mode.
    """

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        invoke = Fn(func, args, kwargs)

        result = invoke()
        print(f"{invoke!s} -> {result!r}")
        return result


@ctxl.contextmanager
def print_torch_dispatch():
    """
    Context manager to print `__torch_dispatch__` calls.
    """

    with _PrintDispatchMode():
        yield


@dcls.dataclass
class _LogDispatchMode(pyd.TorchDispatchMode):
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
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}

        invoke = Fn(func, *args, **kwargs)
        result = invoke()

        self.logger.log(self.level, f"%s -> %s", invoke, result)
        return result


@ctxl.contextmanager
def log_torch_dispatch(*, logger: logging.Logger = LOGGER, level: int = logging.DEBUG):
    """
    Context manager to log the `__torch_dispatch__` calls.

    Args:
        logger: The logger to use. Default to the one in this module.
        level: The level to log to. Default to `logging.DEBUG`.
    """

    with _LogDispatchMode(logger=logger, level=level):
        yield logger


_ThunkType = cabc.Callable[..., TorchIrFn]


class TorchFnRouter(typing.Protocol):
    def __call__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
    ) -> _ThunkType: ...


def only_route_aten(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _ThunkType:
    if is_aten_op(func):
        return aten_router(func=func, types=types)

    assert is_prim_op(func), func
    return NotImplemented


def no_route(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> _ThunkType:
    return NotImplemented


@dcls.dataclass
class _StoreDispatchMode(pyd.TorchDispatchMode):
    router: TorchFnRouter
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
        return thunk()


def aten_router(
    func: _ops.OpOverload, types: tuple[type[torch.Tensor], ...]
) -> cabc.Callable[..., TorchIrFn]:
    assert is_aten_op(func), func
    return NotImplemented


@ctxl.contextmanager
def track_fn_mode():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`.
    """

    with _StoreDispatchMode(router=no_route) as sdm:
        yield sdm.calls


@ctxl.contextmanager
def fake_fn_mode():
    """
    Track all calls into the torch dispatch mode as `TorchIrFn`,
    when fake mode is active.
    """

    if not enabled_fake_mode():
        raise RuntimeError("Fake mode is not enabled.")

    with fake_mode(), _StoreDispatchMode(router=only_route_aten) as sdm:
        yield sdm.calls
