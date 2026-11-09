# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import typing
from collections import abc as cabc

from aioway._utils import AnyDict, any_dict

from .ops import Op

__all__ = ["OpExec", "executor", "set_executor", "default_op_exec", "CachedOpExec"]


def executor() -> OpExec:
    """
    The global executor that `Op` uses to execute itself.

    It's a function s.t. it stays unchanged until rebind during use.
    """

    return _op_exec


@ctxl.contextmanager
def set_executor[O: OpExec](executor: O) -> cabc.Generator[O]:
    global _op_exec

    before, _op_exec = _op_exec, executor
    try:
        yield executor
    finally:
        _op_exec = before


class OpExec[T = typing.Any](typing.Protocol):
    """
    This is the executor responsible for executing an `Op`.
    It does so by recursively executing its dependencies,
    caching the values on the way, until the `Op` is not consumed by anyone else.

    Note that each `Op` would be executed at max once.
    """

    def __call__(self, op: Op[T]) -> T: ...


def default_op_exec(op: Op) -> typing.Any:
    "The default execution strategy. Only calls `.execute()`."
    return op.execute()


class CachedOpExec(OpExec):
    def __init__(self, cache: AnyDict[Op, typing.Any] | None = None) -> None:
        self._cache: AnyDict[Op, typing.Any] = any_dict() if cache is None else cache

    def __contains__(self, obj) -> bool:
        return obj in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key: Op) -> typing.Any:
        return self._cache[key]

    def __setitem__(self, key: Op, val) -> None:
        self._cache[key] = val

    def __repr__(self) -> str:
        return repr(self._cache)

    def __call__(self, op: Op) -> typing.Any:
        if op not in self:
            try:
                result = op.execute()
            except Exception as e:
                self[op] = e
            else:
                self[op] = result

        if isinstance(result := self[op], Exception):
            raise result
        else:
            return self[op]

    @classmethod
    @ctxl.contextmanager
    def enable(cls) -> cabc.Generator[typing.Self]:
        """
        A context manager that caches the current operator.
        """

        cache = cls()

        with set_executor(cache):
            yield cache


_op_exec: OpExec = default_op_exec
