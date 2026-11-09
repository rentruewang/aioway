# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest

from aioway.ops import (
    CachedOpExec,
    Op,
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
    with CachedOpExec.cache_op() as cache:
        assert executor() is cache
        op()
        op()
        assert len(cache) == 1
    assert executor() is default_op_exec
    assert op.calls == 1
