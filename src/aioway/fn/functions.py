# Copyright (c) AIoWay Authors - All Rights Reserved

"This module contains utilities that uses the `__torch_function__` mode."

import contextlib as ctxl
import dataclasses as dcls
import typing

import torch
from torch import _ops, overrides

from .fn import Thunk

__all__ = ["track_function_fn"]

_active_function: Thunk | None = None
"The current function."


@dcls.dataclass(frozen=True)
class TrackFunctionMode(overrides.TorchFunctionMode):
    functions: list[Thunk] = dcls.field(default_factory=list)

    @typing.override
    def __torch_function__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ) -> typing.Any:
        kwargs = kwargs or {}
        fn = Thunk(func, *args, **kwargs)
        self.functions.append(fn)

        with _ensure_single_function(fn):
            return fn()


@ctxl.contextmanager
def track_function_fn():
    with TrackFunctionMode() as sfm:
        yield sfm.functions


@ctxl.contextmanager
def _ensure_single_function(fn: Thunk):
    "Ensure that only 1 function is running at any given moment."

    global _active_function

    if _active_function is not None:
        raise ValueError("Cannot run 2 functions at once.")

    try:
        _active_function = fn
        yield
    finally:
        _active_function = None


def active_function():
    return _active_function
