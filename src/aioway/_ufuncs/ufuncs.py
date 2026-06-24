# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

from aioway._iters import GraphNode, Iter, node_dcls
from aioway._thunks import Thunk

__all__ = ["UFunc", "LazyThunk"]


@node_dcls
class LazyThunk[T](Thunk[T], GraphNode, abc.ABC):
    """
    The `Thunk` that lazily invokes its dependencies, with back tracing info.
    """


@typing.runtime_checkable
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
