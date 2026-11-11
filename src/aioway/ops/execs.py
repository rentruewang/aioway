# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctxl
import typing
from collections import abc as cabc

from aioway._utils import AnyDict, any_dict

from .ops import Op

__all__ = ["OpExec", "executor", "set_executor", "DefaultOpExec", "LazyCacheOpExec"]


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


class OpExec[T = typing.Any](abc.ABC):
    """
    This is the executor responsible for executing an `Op`.
    It does so by recursively executing its dependencies,
    caching the values on the way, until the `Op` is not consumed by anyone else.

    `OpExec` should not persist state, and should handle cleanup after done.
    """

    def __init__(self) -> None:
        self._nested_count: int = 0
        """
        Number of calls that this `OpExec` is currently nested.
        Since `OpExec` is used by `Op` which is recursive,
        a count of 0 means that execution is inactive.
        """

    @typing.final
    def __call__(self, op: Op[T]) -> T:
        with self._cleanup_on_finish():
            return self.execute(op)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(active={self.is_active})"

    @property
    def is_active(self) -> bool:
        return self._nested_count != 0

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the entire `OpExec` to the original state.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, op: Op[T]) -> T:
        """
        The implementation for `__call__`.
        """
        raise NotImplementedError

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


class DefaultOpExec(OpExec):
    "The default execution strategy. Only calls `.execute()`."

    @typing.override
    def reset(self) -> None:
        pass

    @typing.override
    def execute[T](self, op: Op[T]) -> T:
        return op.execute()


class LazyCacheOpExec(OpExec):
    """
    The `OpExec` that caches each call during evaluation, and frees up once done.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def __contains__(self, obj) -> bool:
        return obj in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key: Op) -> typing.Any:
        return self._cache[key]

    def __setitem__(self, key: Op, val) -> None:
        self._cache[key] = val

    def reset(self) -> None:
        self._cache: AnyDict[Op, typing.Any] = any_dict()

    def execute(self, op: Op) -> typing.Any:
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

    @classmethod
    @ctxl.contextmanager
    def enable(cls) -> cabc.Generator[typing.Self]:
        """
        A context manager that caches the current operator.
        """

        cache = cls()

        with set_executor(cache):
            yield cache


_op_exec: OpExec = DefaultOpExec()
