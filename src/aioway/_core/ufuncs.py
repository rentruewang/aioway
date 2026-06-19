# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import typing
from collections import abc as cabc

import torch

from aioway._core import TorchThunk
from aioway._utils import AnyDict

from .iters import Iter
from .thunks import Thunk

__all__ = ["UFunc", "TensorUFunc1"]

_thunk_cache: AnyDict[Thunk] | None = None
"The cache instance for `Thunk`."


@ctxl.contextmanager
def thunk_cache_on() -> cabc.Generator[AnyDict[Thunk]]:
    """
    Turn on caching for `Thunk`. Everytime you call `hop_cache_on`,
    a new scope is created and so a new cache is created.
    (The old cache still stays in memory so it'll still be "active").

    Returns:
        A context manager that when activates, intercept all `AnyThunk.__call__` calls,
        and stores the outputs s.t. second `.__call__()` uses the previous rersult.
    """

    global _thunk_cache
    before, _thunk_cache = _thunk_cache, AnyDict[Thunk](Thunk)

    try:
        yield _thunk_cache
    finally:
        _thunk_cache = before


def thunk_cache() -> AnyDict[Thunk]:
    """
    The active cache for `Thunk`. If there is no active session, raise `RuntimeError`.
    """

    if _thunk_cache is None:
        raise RuntimeError(
            "`thunk_cache` can only be called in `thunk_cache_on` scope."
        )

    return _thunk_cache


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


class AnyUFunc[**P = ..., T = typing.Any](UFunc[T]):
    def __init__(self, func: cabc.Callable[P, T], /) -> None:
        self._func = func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)

    def thunk(self, *args: P.args, **kwargs: P.kwargs) -> Thunk[T]:
        return TorchThunk(self.func, *args, **kwargs)

    def iter(self, *args: P.args, **kwargs: P.kwargs) -> Iter[T]:
        raise NotImplementedError

    @property
    def func(self) -> cabc.Callable[P, T]:
        return self._func


class UFuncThunk[**P, T](Thunk):
    def __init__(
        self, func: cabc.Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self) -> T:
        """
        Perhaps cache the thunks if running under a `thunk_cache_on` block.
        """

        if _thunk_cache is None:
            return self._call()

        if self not in _thunk_cache:
            _thunk_cache[self] = self._call()

        return _thunk_cache[self]

    def _call(self) -> T:
        return self.func(*self.args, **self.kwargs)

    @property
    def func(self) -> cabc.Callable[P, T]:
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs


class TensorUFunc1(UFunc, typing.Protocol):
    @abc.abstractmethod
    def __call__(self, item: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def thunk(self, item: Thunk[torch.Tensor], /) -> Thunk[torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def iter(self, item: Iter[torch.Tensor], /) -> Iter[torch.Tensor]:
        raise NotImplementedError


class TensorUFunc2(UFunc, typing.Protocol):
    @abc.abstractmethod
    def __call__(self, left: torch.Tensor, right: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def thunk(
        self, left: Thunk[torch.Tensor], right: Thunk[torch.Tensor], /
    ) -> Thunk[torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def iter(
        self, left: Iter[torch.Tensor], right: Iter[torch.Tensor], /
    ) -> Iter[torch.Tensor]:
        raise NotImplementedError
