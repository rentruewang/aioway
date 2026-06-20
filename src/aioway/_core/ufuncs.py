# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from .iters import Iter
from .thunks import LazyThunk

__all__ = ["UFunc"]


class UFunc[T = typing.Any](typing.Protocol):
    """
    `UFunc`, inspired by `numpy`, stands for universal functions,
    and are the building blocks of other APIs (like thunks and iters).
    """

    __call__: cabc.Callable[..., T]
    "A `UFunc` is callable (imperative)."

    thunk: cabc.Callable[..., LazyThunk[T]]
    "A `UFunc` takes thunks and tranform it into other thunks."

    iter: cabc.Callable[..., Iter[T]]
    "A `UFunc` takes iterators and tranform it into other iterators."
