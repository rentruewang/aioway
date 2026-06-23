# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl

import pytest

from aioway._core import Iter, iter_cache_on


def test_iter_struct():
    pass


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
