# Copyright (c) AIoWay Authors - All Rights Reserved

import copy
import typing
from collections import abc as cabc

from aioway.spaces import DataSpace

from .envs import Env

__all__ = ["TdictEnv"]


@typing.final
class IteratorEnv[T](Env[T]):
    "An env around a normal"

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
