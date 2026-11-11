# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = ["Op", "exec_op"]


@typing.dataclass_transform()
class Op[T = object](abc.ABC):
    """
    The base class for expressions.

    Opessions refer to those operations that are

    1. Pure.
    2. Operate on batches.

    It can be converted into an `nn.Module` easily.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]

    def __init_subclass__(cls, **kwargs):
        _ = dcls.dataclass()(cls)

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

    def post_order(self) -> cabc.Iterator[Op]:
        """
        Get the post order traversal of `Op`.
        """

        for dep in self.deps():
            yield from dep.post_order()

        yield self


def exec_op(obj):
    """
    Execute `obj` if it is `Op`, else return identity.
    """

    if isinstance(obj, Op):
        return obj()

    else:
        return obj
