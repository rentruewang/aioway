# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from aioway._core import TorchThunk
from aioway._utils import AnyDict

from .iters import Iter
from .thunks import Thunk

__all__ = ["UFunc"]


class UFunc[T = typing.Any](typing.Protocol):
    """
    `UFunc`, inspired by `numpy`, stands for universal functions,
    and are the building blocks of other APIs (like thunks and iters).
    """

    __call__: cabc.Callable[..., T]
    "A `UFunc` is callable (imperative)."

    thunk: cabc.Callable[..., Thunk[T]]
    "A `UFunc` takes thunks and tranform it into other thunks."

    iter: cabc.Callable[..., Iter[T]]
    "A `UFunc` takes iterators and tranform it into other iterators."


class UFuncThunk[**P = ..., T = typing.Any](Thunk[T]):
    """
    The `Thunk` type for `UFunc`.
    """

    def __init__(
        self, func: cabc.Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if not callable(self.func):
            raise TypeError(f"{self.func=} is not callable.")

        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        return self.func(*self.args, **self.kwargs)

    @property
    def func(self) -> cabc.Callable[P, T]:
        "The function to call. Must be callable."
        return self._func

    @property
    @typing.no_type_check
    def args(self) -> P.args:
        "The positional args."
        return self._args

    @property
    @typing.no_type_check
    def kwargs(self) -> P.kwargs:
        "The keyword arguments."
        return self._kwargs
