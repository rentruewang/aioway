# Copyright (c) AIoWay Authors - All Rights Reserved

"The package reponsible for executing the `Op`s."

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.ops.ops import Op

from .reqs import DomainReqs

__all__ = [
    "OpExec",
    "executor",
    "set_executor",
    "direct_op_exec",
    "disable_domain_checks",
    "DomainCheckOpExec",
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


def direct_op_exec(op: Op) -> typing.Any:
    "Executes `op` directly, only calls `.execute()`."
    return op.execute()


@ctxl.contextmanager
def disable_domain_checks() -> cabc.Generator[None]:
    "Temporarily execute with `direct_op_exec`, without the run time domain checks."

    with set_executor(direct_op_exec):
        yield


@dcls.dataclass
class DomainCheckOpExec(OpExec):
    """
    Checks each result against its domain, and what its consumers require of it.

    The requirements are collected on the outermost call, and reused by the nested ones.
    """

    fallback: OpExec = dcls.field(default_factory=executor)
    "Executes the `Op`s."

    reqs: DomainReqs = dcls.field(default_factory=DomainReqs, init=False, repr=False)
    "The requirements during an evaluation."

    def __call__(self, op: Op) -> typing.Any:
        # `.reqs` is set when it's nested.
        ctx = ctxl.nullcontext() if self.reqs else self.collect_reqs(op)

        with ctx:
            return self._checked(op)

    @ctxl.contextmanager
    def collect_reqs(self, op: Op):
        self.reqs = DomainReqs.collect(op)
        try:
            yield
        finally:
            self.reqs = DomainReqs()

    def _checked(self, op: Op) -> typing.Any:
        result = self.fallback(op)
        self.reqs.check(op, result)
        return result


_op_exec: OpExec = DomainCheckOpExec(fallback=direct_op_exec)
