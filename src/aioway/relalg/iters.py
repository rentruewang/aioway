# Copyright (c) AIoWay Authors - All Rights Reserved

"The iterator for the `Exec`s."

import abc
import contextlib as ctxl
import typing
from collections import abc as cabc

from aioway._tensors import torch_fake_mode
from aioway._utils import AnyDict

from .execs import Exec, current_sample_mode, sample_mode

__all__ = ["ExecIter", "SampleExecIter", "CacheExecIter", "iter_cache", "iter_cache_on"]

_iter_cache: AnyDict[Exec] | None = None
"The cache instance for `Exec`."


@ctxl.contextmanager
def iter_cache_on() -> cabc.Generator[AnyDict[Exec]]:
    """
    Turn on caching for `Exec`. The cache is re-used in nested `iter_cache_on` blocks.

    Returns:
        A context manager that when activates, intercept all `Exec.__next__` calls,
        and stores the outputs s.t. second `.__next__()` uses the previous rersult.
    """

    global _iter_cache

    if _iter_cache is not None:
        yield _iter_cache
        return

    _iter_cache = AnyDict[Exec](Exec)

    try:
        yield _iter_cache
    finally:
        _iter_cache = None


def iter_cache() -> AnyDict[Exec]:
    """
    The active cache for `Exec`. If there is no active session, raise `RuntimeError`.
    """

    if _iter_cache is None:
        raise RuntimeError("`iter_cache` can only be called in `iter_cache_on` scope.")

    return _iter_cache


class ExecIter[T = typing.Any](cabc.Iterator[T], abc.ABC):
    "The iterator class for executor"

    def __init__(self, iterable: Exec) -> None:
        self._iter = iterable

    @abc.abstractmethod
    def __next__(self) -> T:
        raise NotImplementedError


class SampleExecIter[T = typing.Any](ExecIter[T]):
    """
    Calls the `Exec.sample` method, which returns a fake output.
    This is invoked when `sample_mode` is on.
    """

    def __init__(self, iterable: Exec) -> None:
        super().__init__(iterable)
        assert current_sample_mode()

    def __iter__(self):
        return self

    @typing.final
    def __next__(self) -> T:
        # Set sample mode to false to avoid self recursion.
        with sample_mode(False), torch_fake_mode():
            return self._iter.sample()


class CacheExecIter[T = typing.Any](ExecIter[T]):
    """
    Calls the `.iterate` method, and cache it if `iter_cache` is enabled.
    This allows `.iterate` to be generators (more elegant).
    """

    def __init__(self, iterable: Exec) -> None:
        super().__init__(iterable)

        self.__idx: int = 0
        self.__gen = iterable.iterate()

    def __iter__(self):
        return self

    @typing.final
    def __next__(self) -> T:
        # If `StopIteration` is raised here, it's done.

        with iter_cache_on():
            answer = self.read()

        self.__idx += 1
        return answer

    def read(self) -> T:
        if _iter_cache is None:
            return next(self.__gen)

        elif self.iterator not in _iter_cache:
            _iter_cache[self.iterator] = next(self.__gen)

        result: typing.Any = _iter_cache[self.iterator]
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
    def iterator(self) -> Exec:
        return self._iter
