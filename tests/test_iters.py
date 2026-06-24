# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from collections import abc as cabc

from aioway._iters import Iter, StructIter, iter_cache_on


def test_iter_struct():
    @dcls.dataclass
    class ItemIter[T](Iter[T]):
        item: T

        def iterate(self) -> cabc.Generator[T]:
            yield self.item

    iterable = {"hello": ItemIter("world"), "number": ItemIter(3)}
    result = list(StructIter(iterable))
    assert result == [{"hello": "world", "number": 3}]


def test_iter_cache():
    number: int = 0

    class CacheLogIter(Iter):
        def iterate(self):
            nonlocal number
            while True:
                number += 1
                yield number

    logger = iter(CacheLogIter())

    with iter_cache_on():
        assert number == 0, number
        next(logger)
        next(logger)
        next(logger)
        assert number == 1, number
