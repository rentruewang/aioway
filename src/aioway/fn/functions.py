# Copyright (c) AIoWay Authors - All Rights Reserved

"This module contains utilities that uses the `__torch_function__` mode."

import contextlib as ctxl
import dataclasses as dcls
import typing

import torch
from torch import _ops, overrides

from .fn import Fn

__all__ = ["track_function_fn"]

_function: Fn | None = None
"The current function."


@dcls.dataclass(frozen=True)
class TrackFunctionMode(overrides.TorchFunctionMode):
    functions: list[Fn] = dcls.field(default_factory=list)

    @typing.override
    def __torch_function__(
        self,
        func: _ops.OpOverload,
        types: tuple[type[torch.Tensor], ...],
        args: tuple[typing.Any, ...] = (),
        kwargs: dict[str, typing.Any] | None = None,
    ) -> typing.Any:
        kwargs = kwargs or {}
        fn = Fn(func, *args, **kwargs)
        self.functions.append(fn)

        # Ensure that only 1 function is running at any given moment.
        with _single_function(fn):
            return fn()


@ctxl.contextmanager
def track_function_fn():
    with TrackFunctionMode() as sfm:
        yield sfm.functions


@ctxl.contextmanager
def _single_function(function: Fn):
    global _function

    if _function is not None:
        raise ValueError("Cannot run 2 functions at once.")

    try:
        _function = function
        yield
    finally:
        _function = None
