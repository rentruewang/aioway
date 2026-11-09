# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = ["Op", "op_dcls", "exec_op"]


@typing.dataclass_transform(frozen_default=True)
def op_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


class Op[T = object](abc.ABC):
    """
    The base class for expressions.

    Opessions refer to those operations that are

    1. Pure.
    2. Operate on batches.

    It can be converted into an `nn.Module` easily.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]

    def __call__(self) -> T:
        from .execs import executor

        return executor()(self)

    @abc.abstractmethod
    def execute(self) -> T:
        """
        Evaluate and compute the result of this expression.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def deps(self) -> cabc.Iterable[Op]:
        """
        The expressions that this current expression depends on.
        """

        raise NotImplementedError


def exec_op(obj):
    """
    Execute `obj` if it is `Op`, else return identity.
    """

    if isinstance(obj, Op):
        return obj()

    else:
        return obj
