# Copyright (c) AIoWay Authors - All Rights Reserved

"This module contains utilities that uses the `__torch_dispatch__` mode."

import contextlib as ctxl
import logging
import typing

import torch
from torch import _ops
from torch.utils import _python_dispatch as pyd

from aioway._common import format_function

__all__ = ["print_torch_dispatch"]

LOGGER = logging.getLogger(__name__)


class _StoreDispatchMode(pyd.TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        print(format_function(func, *args, **kwargs))
        result = func(*args, **kwargs)
        print(result, id(result))
        return result


class _PrintDispatchMode(pyd.TorchDispatchMode):

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        func_call = format_function(func, *args, **kwargs)
        print(f"{func_call} -> {result!r}")
        return result


class _LogDispatchMode(pyd.TorchDispatchMode):

    def __torch_dispatch__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if LOGGER.isEnabledFor(logging.DEBUG):
            func_call = format_function(func, *args, **kwargs)
            LOGGER.debug(f"%s -> %s", func_call, result)
        return result


@ctxl.contextmanager
def print_torch_dispatch():
    with _PrintDispatchMode() as pdm:
        yield pdm
