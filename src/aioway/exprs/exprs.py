# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
from collections import abc as cabc

__all__ = ["Expr"]


class Expr[T](abc.ABC):
    """
    The base class for expressions.

    Expressions refer to those operations that are

    1. Pure.
    2. Operate on batches.

    It can be converted into an `nn.Module` easily.
    """

    @abc.abstractmethod
    def __call__(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def inputs(self) -> cabc.Iterable[Expr]:
        raise NotImplementedError
