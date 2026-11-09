# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect
import typing

import pytest
import torch
from torch import nn
from torch import testing as tt

from aioway._signs import Param, Sign
from aioway.ops import (
    CachedOpExec,
    FuncInputOp,
    ModuleThunkOp,
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
    cache = CachedOpExec()
    with pytest.raises(RuntimeError):
        with set_executor(cache):
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


def test_func_dedup_input():
    x = inp("x")
    op = SumCountOp(0, x, x)

    # (x, x) would have signature (x)
    assert OpFunc(op).signature == Sign.from_input_names(["x"])


def test_func_input_mismatch():
    x = inp("x")

    # Same name, different param.
    diff_param = SumCountOp(
        0, x, FuncInputOp(param=Param.build("x", default=0), typ=int)
    )
    with pytest.raises(TypeError, match="same name"):
        OpFunc(diff_param)

    # Same name, different typ.
    diff_typ = SumCountOp(0, x, FuncInputOp(param=Param.build("x"), typ=float))
    with pytest.raises(TypeError, match="same name"):
        OpFunc(diff_typ)


def test_func_bad_args():
    func = OpFunc(inp("a"))
    with pytest.raises(TypeError):
        func()


def test_func_no_stale_result():
    f = OpFunc(SumCountOp(0, inp("a"), inp("b")))
    # This enable should have no effect.
    with CachedOpExec.enable():
        assert f(1, 2) == 3
        assert f(3, 4) == 7


def test_func_input_without_exec():
    x = inp("x")
    with pytest.raises(RuntimeError):
        x()


def test_func_module():
    layer = nn.Linear(2, 3)
    x = torch.randn(1, 2)
    f = OpFunc(ModuleThunkOp(func=layer, args=(inp("x"),), kwargs={}))

    f_params = set(f.parameters())
    layer_params = set(layer.parameters())
    assert f_params == layer_params
    tt.assert_close(f(x), layer(x))
