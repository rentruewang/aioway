# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._utils import AnyDict, any_dict

from .ops import Op

__all__ = [
    "OpExec",
    "OpExecRc",
    "executor",
    "set_executor",
    "DefaultOpExec",
    "LazyCacheOpExec",
]


def executor() -> OpExecRc:
    """
    The global executor that `Op` uses to execute itself.

    It's a function s.t. it stays unchanged until rebind during use.
    """

    return _op_exec


@ctxl.contextmanager
def set_executor[O: OpExec](executor: O) -> cabc.Generator[O]:
    global _op_exec

    before, _op_exec = _op_exec, OpExecRc(executor)
    try:
        yield executor
    finally:
        _op_exec = before


class OpExec[T = typing.Any](abc.ABC):
    """
    This is the executor responsible for executing an `Op`.
    It does so by recursively executing its dependencies.
    """

    @abc.abstractmethod
    def __call__(self, op: Op[T]) -> T:
        """
        Call `op.execute` somewhere in this function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the `OpExec` to its original state, dropping anything it holds.
        """

        raise NotImplementedError


@dcls.dataclass
class OpExecRc(OpExec):
    """
    Reference counts the active calls to the wrapped `OpExec`,
    and resets it once no call is active.

    Since `Op` executes its dependencies recursively through `executor()`,
    the count only returns to 0 when the outermost call is done.
    """

    exec: OpExec
    "The wrapped `OpExec`."

    count: int = 0
    "The number of currently active (nested) calls."

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.exec!r}, count={self.count})"

    def __call__(self, op: Op) -> typing.Any:
        self.count += 1
        try:
            return self.exec(op)
        finally:
            self.count -= 1

            # Clean up once done.
            if self.count == 0:
                self.reset()

    def reset(self) -> None:
        self.exec.reset()


class DefaultOpExec(OpExec):
    "The default execution strategy. Only calls `.execute()`."

    def __call__(self, op: Op) -> typing.Any:
        return op.execute()

    def reset(self) -> None:
        pass


class LazyCacheOpExec(OpExec):
    """
    The `OpExec` that caches each `Op`'s result (or raised error) until reset.
    """

    def __init__(self) -> None:
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
        self._cache: AnyDict[Op, typing.Any] = any_dict()

    @classmethod
    @ctxl.contextmanager
    def enable(cls) -> cabc.Generator[typing.Self]:
        """
        Use a new `LazyCacheOpExec` as the executor within the block.
        """

        cache = cls()

        with set_executor(cache):
            yield cache


_op_exec: OpExecRc = OpExecRc(DefaultOpExec())
