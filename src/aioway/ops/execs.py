# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from aioway._utils import AnyDict

from .ops import Op

__all__ = ["OpExec"]


class OpExec:
    """
    This is the executor responsible for executing an `Op`.
    It does so by recursively executing its dependencies,
    caching the values on the way, until the `Op` is not consumed by anyone else.

    Note that each `Op` would be executed at max once.
    """

    def __init__(self, cache: AnyDict[Op, typing.Any]) -> None:
        self._cache = cache

    def __call__(self, op: Op) -> typing.Any:
        if op in self._cache:
            return self._cache[op]

        raise NotImplementedError
