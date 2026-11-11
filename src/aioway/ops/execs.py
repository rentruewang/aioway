# Copyright (c) AIoWay Authors - All Rights Reserved

"The package reponsible for executing the `Op`s."

import contextlib as ctxl
import typing
from collections import abc as cabc

from aioway._utils import AnyDict, any_dict

from .ops import Op

__all__ = [
    "OpExec",
    "executor",
    "set_executor",
    "default_op_exec",
    "LazyCacheOpExec",
]


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
    It does so by recursively executing its dependencies.
    """

    def __call__(self, op: Op[T]) -> T: ...


def default_op_exec(op: Op) -> typing.Any:
    "The default execution strategy. Only calls `.execute()`."
    return op.execute()


class LazyCacheOpExec(OpExec):
    """
    The `OpExec` that caches each call during evaluation, and frees up once done.

    An evaluation is one outermost call to the executor, together with
    every nested call it makes while executing dependencies.
    Within it, each `Op` is executed at most once, and an error it raises
    is cached and re-raised as well.
    """

    def __init__(self) -> None:
        self._nested_count: int = 0
        """
        Number of calls that this `OpExec` is currently nested.
        Since `OpExec` is used by `Op` which is recursive,
        a count of 0 means that execution is inactive.
        """

        self.reset()

    def __contains__(self, obj) -> bool:
        return obj in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key: Op) -> typing.Any:
        return self._cache[key]

    def __setitem__(self, key: Op, val) -> None:
        self._cache[key] = val

    def __call__(self, op: Op) -> typing.Any:
        with self._cleanup_on_finish():
            return self._execute(op)

    @property
    def is_active(self) -> bool:
        return self._nested_count != 0

    def _execute(self, op: Op) -> typing.Any:
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
            return result

    def reset(self) -> None:
        "Drop all cached results."
        self._cache: AnyDict[Op, typing.Any] = any_dict()

    @ctxl.contextmanager
    def _cleanup_on_finish(self) -> cabc.Generator[None]:
        self._nested_count += 1
        try:
            yield
        finally:
            self._nested_count -= 1

            # Clean up once done.
            if not self.is_active:
                self.reset()

    @classmethod
    @ctxl.contextmanager
    def enable(cls) -> cabc.Generator[typing.Self]:
        """
        Use a new `LazyCacheOpExec` as the executor within the block.
        """

        cache = cls()

        with set_executor(cache):
            yield cache


_op_exec: OpExec = default_op_exec
