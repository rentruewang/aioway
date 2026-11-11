# Copyright (c) AIoWay Authors - All Rights Reserved

"The package reponsible for executing the `Op`s."

import contextlib as ctxl
import dataclasses as dcls
import graphlib
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
    "DagSchedule",
    "DagOpExec",
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


class _ExceptionWrapper(typing.NamedTuple):
    exception: Exception


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
                self[op] = _ExceptionWrapper(e)
            else:
                self[op] = result

        # The exception is also cached.
        if isinstance(result := self[op], _ExceptionWrapper):
            raise result.exception
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


class DagSchedule(typing.Protocol):
    """
    Orders the `Op`s under an `Op` (inclusive) for `DagOpExec`.

    Each `Op` appears once, after all of its dependencies.
    """

    def __call__(self, op: Op, /) -> list[Op]: ...


def _topo_sort(op: Op) -> list[Op]:
    "The `Op`s under `op` (inclusive), each once, dependencies first."

    # Collect `Op` by `id`.
    ops: dict[int, Op] = {}

    def collect(node: Op) -> None:
        if id(node) in ops:
            return

        ops[id(node)] = node
        for dep in node.deps():
            collect(dep)

    collect(op)

    # Build the dependency graph on the ids.
    graph = {i: [id(dep) for dep in node.deps()] for i, node in ops.items()}

    # Sort.
    return [ops[i] for i in graphlib.TopologicalSorter(graph).static_order()]


@dcls.dataclass
class DagOpExec(OpExec):
    """
    Executes the DAG under an `Op` in the order given by `schedule`.

    Each `Op.execute` is called exactly once, in order.

    A result is freed once its last consumer is done.
    """

    schedule: DagSchedule = _topo_sort
    "Orders the `Op`s to execute."

    _results: AnyDict[Op, typing.Any] = dcls.field(
        default_factory=any_dict, init=False, repr=False
    )
    "The computed results, kept until last consumer has used it."

    _consumers: AnyDict[Op, int] = dcls.field(
        default_factory=any_dict, init=False, repr=False
    )
    "Remaining times each result will be looked up."

    def __call__(self, op: Op) -> typing.Any:
        order = self.schedule(op)

        # Each `Op` is looked up once per consumer.
        for node in order:
            for dep in node.deps():
                self._consumers[dep] = self._consumers.get(dep, 0) + 1

        try:
            return self._execute_in_order(order, op)
        finally:
            self._results = any_dict()
            self._consumers = any_dict()

    def _execute_in_order(self, order: list[Op], op: Op) -> typing.Any:
        # Only need to look up because it's in order already with dag.
        with set_executor(self.lookup):
            for node in order:
                self._results[node] = node.execute()

        return self._results[op]

    def lookup(self, op: Op) -> typing.Any:
        """
        The temporary executor during `__call__`, looking up computed results.

        A result is freed once its last consumer is done.
        """

        result = self._results[op]
        self._consumers[op] -= 1

        if self._consumers[op] == 0:
            del self._results[op]
            del self._consumers[op]

        return result

    @classmethod
    @ctxl.contextmanager
    def enable(cls) -> cabc.Generator[typing.Self]:
        """
        Use a new `DagOpExec` as the executor within the block.
        """

        dag = cls()

        with set_executor(dag):
            yield dag


_op_exec: OpExec = default_op_exec
