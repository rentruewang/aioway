# Copyright (c) AIoWay Authors - All Rights Reserved

"This module contains utilities that uses the `__torch_dispatch__` mode."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing

import torch
from torch import _ops
from torch.utils import _python_dispatch as pyd

from .thunks import Thunk

__all__ = ["print_torch_dispatch", "log_torch_dispatch"]

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
        invoke = Thunk(func, args, kwargs)

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

        invoke = Thunk(func, *args, **kwargs)
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


@dcls.dataclass
class _StoreDispatchMode(pyd.TorchDispatchMode):
    calls: list[_ops.OpOverload]

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        thunk = Thunk(func, args, kwargs)
        return func(*args, **kwargs)
