# Copyright (c) AIoWay Authors - All Rights Reserved

"The package reponsible for executing the `Op`s."

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._utils import AnyDict, any_dict
from aioway.ops.ops import Op
from aioway.t import Domain

from .reqs import domain_reqs

__all__ = [
    "OpExec",
    "executor",
    "set_executor",
    "direct_op_exec",
    "disable_domain_checks",
    "check_result",
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


def check_result(op: Op, result, reqs: AnyDict[Op, list[Domain]]) -> None:
    """
    Check `result` of `op` against its domain,
    and what its consumers require of it (from `domain_reqs`).
    """

    op.__domain__.check(result)

    if op not in reqs:
        return

    for req in reqs[op]:
        req.check(result)


@dcls.dataclass
class DomainCheckOpExec(OpExec):
    """
    Checks each result against its domain, and what its consumers require of it.

    The requirements are collected on the outermost call, and reused by the nested ones.
    """

    fallback: OpExec = dcls.field(default_factory=executor)
    "Executes the `Op`s."

    reqs: AnyDict[Op, list[Domain]] = dcls.field(
        default_factory=any_dict, init=False, repr=False
    )
    "The requirements, from `domain_reqs`, during an evaluation."

    def __call__(self, op: Op) -> typing.Any:
        # `.reqs` is set when it's nested.
        ctx = ctxl.nullcontext() if self.reqs else self.set_domain_reqs(op)

        with ctx:
            return self._checked(op)

    @ctxl.contextmanager
    def set_domain_reqs(self, op: Op):
        self.reqs = domain_reqs(op)
        try:
            yield
        finally:
            self.reqs = any_dict()

    def _checked(self, op: Op) -> typing.Any:
        result = self.fallback(op)
        check_result(op, result, self.reqs)
        return result


_op_exec: OpExec = DomainCheckOpExec(fallback=direct_op_exec)
