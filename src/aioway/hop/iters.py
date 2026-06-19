# Copyright (c) AIoWay Authors - All Rights Reserved

"The iterator for `Iter`."

import contextlib as ctxl
import typing
from collections import abc as cabc

from aioway._utils import AnyDict

from .hop import Iter

__all__ = ["HopIter", "hop_cache", "hop_cache_on"]


_hop_cache: AnyDict[Iter] | None = None
"The cache instance for `Iter`."


@ctxl.contextmanager
def hop_cache_on() -> cabc.Generator[AnyDict[Iter]]:
    """
    Turn on caching for `Iter`. Everytime you call `hop_cache_on`,
    a new scope is created and so a new cache is created.
    (The old cache still stays in memory so it'll still be "active").

    Returns:
        A context manager that when activates, intercept all `Iter.__call__` calls,
        and stores the outputs s.t. second `.__call__()` uses the previous rersult.
    """

    global _hop_cache
    before, _hop_cache = _hop_cache, AnyDict[Iter](Iter)

    try:
        yield _hop_cache
    finally:
        _hop_cache = before


def hop_cache() -> AnyDict[Iter]:
    """
    The active cache for `Iter`. If there is no active session, raise `RuntimeError`.
    """

    if _hop_cache is None:
        raise RuntimeError("`hop_cache` can only be called in `hop_cache_on` scope.")

    return _hop_cache


class HopIter[T = typing.Any](cabc.Iterator[T]):
    def __init__(self, hop: Iter) -> None:
        self.__idx: int = 0
        self.__gen = hop.iterate()
        self.__result = NotImplemented
        self._hop = hop

    def __iter__(self):
        return self

    @typing.final
    def __next__(self) -> T:
        # If `StopIteration` is raised here, it's done.
        answer = self.read()
        self.__idx += 1
        return answer

    def read(self) -> T:
        if _hop_cache is None:
            return next(self.__gen)

        elif self.hop not in _hop_cache:
            _hop_cache[self.hop] = next(self.__gen)

        result: typing.Any = _hop_cache[self.hop]
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
        return self._hop
