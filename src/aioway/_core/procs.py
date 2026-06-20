# Copyright (c) AIoWay Authors - All Rights Reserved

"The iterator processor."

import abc
import contextlib as ctxl
import typing
from collections import abc as cabc

from aioway._core import current_sample_mode, sample_mode
from aioway._utils import (
    AnyDict,
    torch_fake_mode,
)

from .iters import Iter

__all__ = ["IterProc", "SampleIterProc", "CacheIterProc", "iter_cache", "iter_cache_on"]


_iter_cache: AnyDict[Iter] | None = None
"The cache instance for `Iter`."


@ctxl.contextmanager
def iter_cache_on() -> cabc.Generator[AnyDict[Iter]]:
    """
    Turn on caching for `Iter`. Everytime you call `hop_cache_on`,
    a new scope is created and so a new cache is created.
    (The old cache still stays in memory so it'll still be "active").

    Returns:
        A context manager that when activates, intercept all `Iter.__call__` calls,
        and stores the outputs s.t. second `.__call__()` uses the previous rersult.
    """

    global _iter_cache
    before, _iter_cache = _iter_cache, AnyDict[Iter](Iter)

    try:
        yield _iter_cache
    finally:
        _iter_cache = before


def iter_cache() -> AnyDict[Iter]:
    """
    The active cache for `Iter`. If there is no active session, raise `RuntimeError`.
    """

    if _iter_cache is None:
        raise RuntimeError("`iter_cache` can only be called in `iter_cache_on` scope.")

    return _iter_cache


class IterProc[T = typing.Any](cabc.Iterator[T], abc.ABC):
    def __init__(self, iterable: Iter) -> None:
        self._iter = iterable

    @abc.abstractmethod
    def __next__(self) -> T:
        raise NotImplementedError


class SampleIterProc[T = typing.Any](IterProc[T]):
    """
    Calls the `Iter.sample` method, which returns a fake output.
    This is invoked when `sample_mode` is on.
    """

    def __init__(self, iterable: Iter) -> None:
        super().__init__(iterable)
        assert current_sample_mode()

    def __iter__(self):
        return self

    @typing.final
    def __next__(self) -> T:
        # Set sample mode to false to avoid self recursion.
        with sample_mode(False), torch_fake_mode():
            return self._iter.sample()


class CacheIterProc[T = typing.Any](IterProc[T]):
    """
    Calls the `.iterate` method, and cache it if `iter_cache` is enabled.
    This allows `.iterate` to be generators (more elegant).
    """

    def __init__(self, iterable: Iter) -> None:
        super().__init__(iterable)

        self.__idx: int = 0
        self.__gen = iterable.iterate()

    def __iter__(self):
        return self

    @typing.final
    def __next__(self) -> T:
        # If `StopIteration` is raised here, it's done.
        answer = self.read()
        self.__idx += 1
        return answer

    def read(self) -> T:
        if _iter_cache is None:
            return next(self.__gen)

        elif self.hop not in _iter_cache:
            _iter_cache[self.hop] = next(self.__gen)

        result: typing.Any = _iter_cache[self.hop]
        return result

    @property
    def idx(self) -> int:
        "Get the current iteration count."

        return self.__idx

    @property
    def started(self) -> bool:
        "Shortcut function to check if `self.idx == 0`."

        return self.idx != 0

    @property
    def hop(self) -> Iter:
        return self._iter
