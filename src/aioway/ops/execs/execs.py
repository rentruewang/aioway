# Copyright (c) AIoWay Authors - All Rights Reserved

"The package reponsible for executing the `Op`s."

import contextlib as ctxl
import typing
from collections import abc as cabc

from aioway.ops.ops import Op

__all__ = [
    "OpExec",
    "executor",
    "set_executor",
    "default_op_exec",
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


_op_exec: OpExec = default_op_exec
