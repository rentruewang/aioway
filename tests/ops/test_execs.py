# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect
import typing

import pytest

from aioway._signs import Param
from aioway.ops import (
    CachedOpExec,
    FuncInputOp,
    Op,
    OpFunc,
    default_op_exec,
    exec_op,
    executor,
    set_executor,
)


class SumCountOp(Op):
    "Sums its children (or returns `value` if none) and counts executions."

    def __init__(self, value=0, *children: Op) -> None:
        self.value = value
        self.children = children
        self.calls = 0

    @typing.override
    def execute(self):
        self.calls += 1
        return self.value + sum(exec_op(c) for c in self.children)

    @typing.override
    def deps(self):
        return self.children


def test_default_executor():
    assert executor() == default_op_exec


def test_default_no_cache():
    op = SumCountOp(1)
    op()
    op()
    assert op.calls == 2


def test_set_executor():
    cache = CachedOpExec()
    with set_executor(cache) as got:
        assert got is cache
        assert executor() is cache
    assert executor() == default_op_exec


def test_set_executor_error():
    with pytest.raises(RuntimeError):
        with set_executor(CachedOpExec()):
            raise RuntimeError
    assert executor() is default_op_exec


def test_cached_executes_once():
    cache = CachedOpExec()
    op = SumCountOp(1)
    assert cache(op) == 1
    assert cache(op) == 1
    assert op.calls == 1
    assert op in cache and cache[op] == 1 and len(cache) == 1


def test_cached_shared_dep():
    shared = SumCountOp(1)
    root = SumCountOp(0, SumCountOp(2, shared), SumCountOp(3, shared))
    with set_executor(CachedOpExec()):
        assert root() == 7
    assert shared.calls == 1


def test_cache_op():
    op = SumCountOp(1)
    with CachedOpExec.enable() as cache:
        assert executor() is cache
        op()
        op()
        assert len(cache) == 1
    assert executor() is default_op_exec
    assert op.calls == 1


def inp(name: str) -> FuncInputOp:
    return FuncInputOp(param=Param.build(name), typ=int)


def test_func_signature():
    f = OpFunc(SumCountOp(0, inp("a"), inp("b")))
    assert list(inspect.signature(f).parameters) == ["a", "b"]


def test_func_call():
    f = OpFunc(SumCountOp(0, inp("a"), SumCountOp(0, inp("b"), inp("c"))))
    assert f(1, 2, 3) == 6
    assert f(a=1, b=2, c=3) == 6


def test_func_shared_input():
    x = inp("x")
    op = SumCountOp(0, x, x)

    # Disallow (x, x) as signature.
    with pytest.raises(ValueError, match="duplicate"):
        OpFunc(op)


def test_func_bad_args():
    with pytest.raises(TypeError):
        OpFunc(inp("a"))()


def test_func_no_stale_results_under_cache():
    f = OpFunc(SumCountOp(0, inp("a"), inp("b")))
    with CachedOpExec.enable():
        assert f(1, 2) == 3
        assert f(3, 4) == 7


def test_func_input_outside_func_raises():
    with pytest.raises(RuntimeError):
        inp("x")()
