# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = ["Expr", "expr_dcls"]


@typing.dataclass_transform(frozen_default=True)
def expr_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


class Expr[T = object](abc.ABC):
    """
    The base class for expressions.

    Expressions refer to those operations that are

    1. Pure.
    2. Operate on batches.

    It can be converted into an `nn.Module` easily.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]

    @abc.abstractmethod
    def __call__(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def inputs(self) -> cabc.Iterable[Expr]:
        raise NotImplementedError
