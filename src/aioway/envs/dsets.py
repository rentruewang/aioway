# Copyright (c) AIoWay Authors - All Rights Reserved

import copy
import typing
from collections import abc as cabc

from aioway.spaces import DataSpace

from .envs import Env

__all__ = ["TdictEnv"]


@typing.final
class IteratorEnv[T](Env[T]):
    "An env around a normal iterator."

    def __init__(self, iterator: cabc.Iterator[T], space: DataSpace) -> None:
        self._iter = iterator
        self._space = space

    def __iter__(self) -> typing.Self:
        return self

    @property
    @typing.override
    def observ_space(self) -> DataSpace:
        return self._space

    @typing.override
    def _get_next(self) -> T:
        return next(self._iter)

    @typing.override
    def clone(self) -> typing.Self:
        return copy.deepcopy(self)


@typing.final
class CachedIteratorEnv[T](Env[T]):
    "An env that caches its iterator."

    def __init__(
        self,
        iterator: cabc.Iterator[T],
        space: DataSpace,
        idx: int = 0,
        cache: list[T] | None = None,
    ) -> None:
        self._iter = iterator
        self._space = space
        self._idx = idx
        self._cache: list[T] = cache or []

    def __iter__(self) -> typing.Self:
        return self

    @property
    @typing.override
    def observ_space(self) -> DataSpace:
        return self._space

    @typing.override
    def _get_next(self) -> T:
        if self._idx > len(self._cache):
            raise RuntimeError("Impossible.")

        if self._idx == len(self._cache):
            # If this line raises `StopIteration`,
            # next `next` call will also hit this line,
            # which is intended because the iterator is exhausted.
            item = next(self._iter)
            self._cache.append(item)

        assert self._idx < len(self._cache)
        result = self._cache[self._idx]
        self._idx += 1
        return result

    @typing.override
    def clone(self) -> typing.Self:
        # No copy for cache.
        return type(self)(self._iter, self._space, self._idx, self._cache)


@typing.final
class ListEnv[T](Env[T]):
    """
    This kind of `Env` stores a list of inputs and yield them one by one.
    """

    def __init__(self, sequence: cabc.Sequence[T], space: DataSpace) -> None:
        self._idx = 0
        self._sequence = sequence
        self._space = space

    @typing.override
    def _get_next(self) -> T:
        try:
            result = self._sequence[self._idx]
        except IndexError:
            raise StopIteration

        self._idx += 1
        return result

    @property
    @typing.override
    def observ_space(self) -> DataSpace:
        return self._space

    @property
    def idx(self) -> int:
        return self._idx

    def clone(self) -> typing.Self:
        # Does not need deepcopy!
        return copy.copy(self)
