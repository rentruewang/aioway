# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
from collections import abc as cabc

import pytest

from aioway._core import Iter, StructIter, iter_cache_on


def test_iter_struct():
    @dcls.dataclass
    class ItemIter[T](Iter[T]):
        item: T

        def iterate(self) -> cabc.Generator[T]:
            yield self.item

    iterable = {"hello": ItemIter("world"), "number": ItemIter(3)}
    result = list(StructIter(iterable))
    assert result == [{"hello": "world", "number": 3}]


@pytest.mark.parametrize("cache", [False, True])
def test_iter_cache(cache: bool):
    number: int = 0

    class CacheLogIter(Iter):
        def iterate(self):
            nonlocal number
            while True:
                number += 1
                yield number

    logger = iter(CacheLogIter())
    cacher = iter_cache_on if cache else ctxl.nullcontext

    with cacher():
        assert number == 0, {"number": number, "cache": cache}
        next(logger)
        assert number == 1, {"number": number, "cache": cache}
        next(logger)
        next(logger)
        assert number == (1 if cache else 3), {"number": number, "cache": cache}
